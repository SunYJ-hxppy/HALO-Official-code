import os
import argparse
from pathlib import Path
import numpy as np
import math
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.amp import GradScaler
from tqdm import tqdm
from transformers import logging
import torch.nn.functional as F
import imageio
from torchvision.io import read_video
from torchvision.transforms import ToPILImage
from PIL import Image
from torchvision.io import write_video
import gc
from typing import Union, List, Optional
import torchvision.transforms as T
from dift_models.dift_sd import SDFeaturizer
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.utils import export_to_video
from guidance_utils.custom_transformer_inject import ControlledTransformer
from guidance_utils.custom_embeddings import prepare_rotary_positional_embeddings
from guidance_utils.custom_modules_inject_final import ModuleWithGuidance, InjectionProcessor
from guidance_utils.motion_flow_utils_final import compute_motion_flow_withdift_re, compute_motion_flow_with_bonus
from guidance_utils.utils import mask_head_byentropy
import decord

# suppress partial model loading warning
logging.set_verbosity_error()



def extract_dift_and_cal_sim(args, guidance, dift_up_ft_index=2, feature_level=16, thre=0.7, occlusion_thre=0.9):
    
        
    dift = SDFeaturizer('stabilityai/stable-diffusion-2-1', null_prompt='')
    
    dift_feature = []
    
    if dift_up_ft_index == 0:
        size = feature_level * 32
    if dift_up_ft_index == 1:
        size = feature_level * 16
    if dift_up_ft_index == 2:
        size = feature_level * 8
    size = (360, 240)

    video_path = args.video_path
    frame_count = args.video_length
    latent_num_frames = (config.video_length - 1) // guidance.pipe.vae_scale_factor_temporal + 1
    sample_indices = np.linspace(0, frame_count - 1, latent_num_frames, dtype=int)

    if video_path.endswith(".mp4"):
        video = read_video(video_path, pts_unit="sec")[0].permute(0, 3, 1, 2).cuda() / 255
        video = [ToPILImage()(video[i]).resize((size[0],size[1])) for i in range(video.shape[0])]
    video = video[: frame_count]
    video = guidance.pipe.video_processor.preprocess_video(video)
    
    for img_tensor in video.squeeze().permute(1,0,2,3):
        
        ft = dift.forward(img_tensor,
                        prompt= 'a photo of ' + " ",
                        t=261,
                        up_ft_index=dift_up_ft_index,
                        ensemble_size=8)
        dift_feature.append(ft)
    
    dift = None
    dift_feature = torch.concat(dift_feature, dim=0)
    dift_feature = dift_feature.permute(0, 2, 3, 1).cuda() #[49, 30, 45, 640]
        
    F, H, W, d = dift_feature.shape

    dift_feature = dift_feature.reshape(-1, d).half()
    dift_feature = torch.nn.functional.normalize(dift_feature, p=2, dim=1)
    
    
    displacement_matrices_2d = []; target_2d = []
    for f1 in range(F):
        for f2 in range(F):

            features_f = dift_feature[f1*H*W:(f1+1)*H*W, :]
            features_f_plus_1 = dift_feature[f2*H*W:(f2+1)*H*W, :]
            
            sim_f_to_f_plus_1 = features_f @ features_f_plus_1.t() # Shape: [H*W, H*W]
            
            _, best_match_indices_flat = sim_f_to_f_plus_1.topk(1, dim=-1) # Shape: [H*W, 1]
            best_match_indices_flat = best_match_indices_flat.squeeze(-1) # Shape: [H*W]
            
            source_indices_frame_flat = torch.arange(H * W, device=features_f.device)
            
            def flat_index_to_coords_2d(flat_idx, height, width):
                row_idx = flat_idx // width
                col_idx = flat_idx % width
                return torch.stack([row_idx, col_idx], dim=-1)

            coords_source_2d = flat_index_to_coords_2d(source_indices_frame_flat, H, W)
            coords_target_2d = flat_index_to_coords_2d(best_match_indices_flat, H, W)
            
            target_2d.append(coords_target_2d)
            displacement_2d = coords_target_2d - coords_source_2d # Shape: [H*W, 2] (dy, dx)
            
            displacement_matrices_2d.append(displacement_2d.reshape(H, W, 2))

    final_displacement_matrix_2d = torch.stack(displacement_matrices_2d, dim=0)
    
    final_target_2d = torch.stack(target_2d, dim=0)
    print("Shape of 2D Displacement Matrix:", final_displacement_matrix_2d.shape)
    flows_to_save = final_displacement_matrix_2d.detach().cpu().numpy()
    np.savez(args.output_path +'/correspondence.npz', flows=flows_to_save)

    torch.cuda.empty_cache()
    clean_memory()
   
    return final_displacement_matrix_2d, final_target_2d

def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list

def isinstance_str(x: object, cls_name: Union[str, List[str]]):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.

    Useful for patching!
    """
    if type(cls_name) == str:
        for _cls in x.__class__.__mro__:
            if _cls.__name__ == cls_name:
                return True
    else:
        for _cls in x.__class__.__mro__:
            if _cls.__name__ in cls_name:
                return True
    return False

def clean_memory():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

def save_video(video, path):
    video_codec = "libx264"
    video_options = {
        "crf": "17",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
        "preset": "slow",  # Encoding preset (e.g., ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
    }   
    write_video(
        path,
        video,
        fps=10,
        video_codec=video_codec,
        options=video_options,
    )

def get_timesteps(timesteps, guidance_timestep_range, skip_timesteps=1):
    max_guidance_timestep, min_guidance_timestep = guidance_timestep_range
    num_inference_steps = len(timesteps)
    init_timestep = min(max_guidance_timestep, num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    t_end = min_guidance_timestep
    if t_end > 0:
        guidance_schedule = timesteps[t_start : -t_end : skip_timesteps]
    else:
        guidance_schedule = timesteps[t_start::skip_timesteps]
    return guidance_schedule



class Guidance(nn.Module):
    def __init__(self, config, prior=None):
        super().__init__()
        self.config = config
        self.device = torch.device(config["device"])
        self.eta=0
        self.batch_size = 1
        self.num_inference_steps = config["num_inference_steps"]
        self._guidance_scale = self.config.guidance_scale

        print("Loading video model")
        print(f"Seed: {config.seed}")
        print(f"Using device: {self.device}")

        if config.model_key=="THUDM/CogVideoX-2b":
            self.dtype = torch.float16
            self.pipe = CogVideoXPipeline.from_pretrained(config.model_key, torch_dtype=self.dtype).to("cuda")
            self.pipe.scheduler = CogVideoXDDIMScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")
            self.use_dynamic_cfg = False
        else:
            self.dtype = torch.bfloat16
            self.pipe = CogVideoXPipeline.from_pretrained(config.model_key, torch_dtype=self.dtype).to("cuda")
            self.pipe.scheduler = CogVideoXDPMScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")
            self.use_dynamic_cfg = True

        # Controlled transformer
        controlled_transformer = ControlledTransformer(**self.pipe.transformer.config)
        controlled_transformer.load_state_dict(self.pipe.transformer.state_dict())
        self.pipe.transformer = controlled_transformer.to(device=self.device, dtype=self.dtype)
        self.pipe.transformer.init_pos_embedding = self.pipe.transformer.init_pos_embedding.to(self.device)
        
        if self.config.enable_gradient_checkpointing:
            self.pipe.transformer.enable_gradient_checkpointing()
        
        self.pipe.scheduler.set_timesteps(self.num_inference_steps, device="cuda")
        self.timesteps = self.pipe.scheduler.timesteps
        self.guidance_schedule = get_timesteps(self.timesteps, self.config.guidance_timestep_range)

        ## Optimizations
        self.pipe.enable_model_cpu_offload()
        # self.pipe.enable_sequential_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

        self.vae = self.pipe.vae
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.transformer = self.pipe.transformer
        self.scheduler = self.pipe.scheduler
        print("video model loaded")

        self.generator = torch.Generator(device='cuda').manual_seed(config.seed)

        #### Pipeline setup - simplified from CogVideoX pipeline code ####
        height = config.height or self.transformer.config.sample_size * self.vae_scale_factor_spatial
        width = config.width or self.transformer.config.sample_size * self.vae_scale_factor_spatial
        num_videos_per_prompt = 1
        assert (height % 16 == 0) and (width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
        self.resolution = (width, height)
        self.config.text_seq_length = 226 # TODO: extract from pipeline
        self.video_length = config.video_length
        self.latent_num_frames = (config.video_length - 1) // self.pipe.vae_scale_factor_temporal + 1
        # print("self.pipe.vae_scale_factor_temporal:", self.pipe.vae_scale_factor_temporal)
        self.latent_height = height // self.pipe.vae_scale_factor_spatial
        self.latent_width = width // self.pipe.vae_scale_factor_spatial
        self.patch_size = self.pipe.transformer.config.patch_size
        self.patches_height = self.latent_height // self.patch_size
        self.patches_width = self.latent_width // self.patch_size

        self.pipe.check_inputs(
            config["target_prompt"],
            height,
            width,
            config["negative_prompt"],
            callback_on_step_end_tensor_inputs=None,
        )

        with torch.no_grad():
            self.source_embeds, _ = self.pipe.encode_prompt(
                config["source_prompt"],
                device=self.device,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=True,
                negative_prompt=config["negative_prompt"],
            )

            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                config["target_prompt"],
                device=self.device,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=True,
                negative_prompt=config["negative_prompt"],
            )
            self.guidance_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        latent_channels = self.transformer.config.in_channels
        self.init_latents = self.pipe.prepare_latents(
            self.batch_size * num_videos_per_prompt,
            latent_channels,
            self.video_length,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            self.generator,
        )

        self.extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(self.generator, self.eta)

        init_rope = (
            prepare_rotary_positional_embeddings(
                height, 
                width, 
                self.init_latents.size(1), 
                self.pipe.vae_scale_factor_spatial, 
                self.transformer.config.patch_size, 
                self.transformer.config.attention_head_dim,
                self.device
            )
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )
        # init_rope = get_motion_warped_rope(inds, init_rope)
        self.transformer.init_rope = init_rope
        self.transformer.guidance_blocks = self.config.guidance_blocks

        # Path verification
        self.output_path = self.config['output_path']
        os.makedirs(self.output_path, exist_ok=True)

        # embeds_path = os.path.join(self.output_path, "embeds")
        # if self.config.inject_embeds:
        #     if not os.path.exists(embeds_path):
        #         raise FileNotFoundError(f"Embeds folder not found at {embeds_path}. Make sure to first run motion guidance without inject_embeds=True so that the trained embeddings can be stored. These can then be injected with a new prompt by running again with inject_embeds=True.")
        # else:
        #     if os.path.exists(embeds_path):
        #         shutil.rmtree(embeds_path)
        #     os.makedirs(embeds_path, exist_ok=True)

        ## GUIDANCE SETUP
        self.motion_timestep = torch.tensor([0], device='cuda')

        self.register_guidance(block_idxs=self.config.guidance_blocks)
        self.register_attention_processor(block_idxs=list(range(len(self.transformer.transformer_blocks))))
        
        num_guidance_steps = self.config.guidance_timestep_range[0] - self.config.guidance_timestep_range[1] + 1
        self.lr_range = np.linspace(self.config.lr[0], self.config.lr[1], num_guidance_steps)
        
        print("Loading features from motion video")
        self.motion_latent = self.load_latent()

    
    def register_guidance(self, block_idxs):
        for out_i in block_idxs:
            block_name = f"block_{out_i}"
            self.transformer.transformer_blocks[out_i] = ModuleWithGuidance(
                self.transformer.transformer_blocks[out_i],
                self.latent_height,
                self.latent_width,
                self.pipe.transformer.config.patch_size,
                block_name=block_name,
                num_frames=self.latent_num_frames
            )
    
    def register_attention_processor(self, block_idxs):
        for out_i in block_idxs:
            block_name = f"block_{out_i}_attn1_processor"
            processor = InjectionProcessor(
                            block_name=block_name, 
                            output_path=self.output_path)
            self.transformer.transformer_blocks[out_i].attn1.set_processor(processor)
    
    @property
    def guidance_scale(self):
        return self._guidance_scale
    
    def _probe_total_frames(self, path):
        if path.endswith(".mp4"):
            frames = read_video(path, pts_unit="sec")[0].shape[0]
        else:
            frames = len(sorted(Path(path).glob("*.png"))) + len(sorted(Path(path).glob("*.jpg")))
        return frames
    
    @torch.no_grad()
    def load_latent(self):

        data_path = self.config.video_path

        if data_path.endswith(".mp4"):
            video = read_video(data_path, pts_unit="sec")[0].permute(0, 3, 1, 2).cuda() / 255
            video = [ToPILImage()(video[i]).resize(self.resolution) for i in range(video.shape[0])]
        else:
            images = list(Path(data_path).glob("*.png")) + list(Path(data_path).glob("*.jpg"))
            images = sorted(images, key=lambda x: int(x.stem.split('f')[-1]))
            video = [Image.open(img).resize(self.resolution).convert('RGB') for img in images]

        video = video[: self.config.video_length]
        
        save_video([np.array(img) for img in video], str(Path(self.config.output_path) / f"original.mp4"))

        video = self.pipe.video_processor.preprocess_video(video)
        video = video.to(self.dtype).to("cuda")
        latents = self.vae.config.scaling_factor * self.vae.encode(video)[0].sample()
        latents = latents.permute(0,2,1,3,4)
        
        return latents
    
    def get_video_frames(self,
        video_path: str,
        width: int,
        height: int,
        skip_frames_start: int,
        skip_frames_end: int,
        max_num_frames: int,
        frame_sample_step: Optional[int],
    ) -> torch.FloatTensor:

        with decord.bridge.use_torch():
            video_reader = decord.VideoReader(uri=video_path, width=width, height=height)
            video_num_frames = len(video_reader)
            start_frame = min(skip_frames_start, video_num_frames)
            end_frame = max(0, video_num_frames - skip_frames_end)

            if end_frame <= start_frame:
                indices = [start_frame]
            elif end_frame - start_frame <= max_num_frames:
                indices = list(range(start_frame, end_frame))
            else:
                step = frame_sample_step or (end_frame - start_frame) // max_num_frames
                indices = list(range(start_frame, end_frame, step))

            frames = video_reader.get_batch(indices=indices)
            frames = frames[:max_num_frames].float()  # ensure that we don't go over the limit

            # Normalize the frames
            transform = T.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)
            frames = torch.stack(tuple(map(transform, frames)), dim=0)

            return frames.permute(0, 3, 1, 2).contiguous()  # [F, C, H, W]
        
    def get_video_frames_latent(self,
        video_path: str,
        width: int,
        height: int,
        num_target_frames: int,
    ) -> torch.FloatTensor:
        with decord.bridge.use_torch():
            video_reader = decord.VideoReader(uri=video_path, width=width, height=height)
            video_num_frames = len(video_reader)

            video_reader = decord.VideoReader(uri=video_path)
            total_frames = len(video_reader)
            
            positions = np.arange(num_target_frames, dtype=float) * (total_frames - 1) / (num_target_frames - 1)
            sample_index = np.rint(positions).astype(int)
            
            sampled_frames = video_reader.get_batch(sample_index).to(device="cuda").float()
            sampled_frames = sampled_frames.permute(0, 3, 1, 2) # [F, C, H, W]

            return sampled_frames  # [F, C, H, W]

    @torch.no_grad()
    def load_attn_features(self, inds=None):
        """ 🔍 AMF Extraction """
        for block_id in self.config.guidance_blocks:
            self.transformer.transformer_blocks[block_id].attn1.processor.inject_kv = False
            self.transformer.transformer_blocks[block_id].attn1.processor.copy_kv = True
        
        attn_features = {}
        # Store keys and queries for all attention blocks
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            self.transformer(
                hidden_states=self.motion_latent,
                encoder_hidden_states=self.source_embeds,
                timestep=self.motion_timestep,
                return_dict=False,
            )

        for block_id in self.config.guidance_blocks:
            frame_size = self.patches_height * self.patches_width
            file_name_stem = os.path.splitext(self.config.file_name)[0]
            save_prefix = f"ref_{file_name_stem}_block{block_id}_timestep{self.motion_timestep.item()}"
            module = self.transformer.transformer_blocks[block_id].attn1.processor
            attn_features[module.block_name] = compute_motion_flow_withdift_re(module.query, module.key, module.value,
                                                    context_length=self.config.text_seq_length,
                                                    num_frame=self.latent_num_frames,
                                                    frame_size=frame_size,
                                                    h=self.patches_height, 
                                                    w=self.patches_width, 
                                                    temp=self.config.motion_temp, 
                                                    argmax=self.config.argmax_motion_flow,
                                                    output_dir=self.output_path,
                                                    save_prefix=save_prefix,
                                                    dift_displacement_map=inds, 
                                                    top_k=10)
        
            self.transformer.transformer_blocks[block_id].attn1.processor.copy_kv = False
            self.transformer.transformer_blocks[block_id].attn1.processor.key = None
            self.transformer.transformer_blocks[block_id].attn1.processor.query = None
            self.transformer.transformer_blocks[block_id].attn1.processor.value = None
        return attn_features
    
    def change_mode(self, train=True):
        """During guidance training, pass through later output blocks to reduce unnecessary computation"""
        @staticmethod
        def dummy_pass(*args, **kwargs):
            if len(args) == 0:
                return kwargs["hidden_states"], kwargs['encoder_hidden_states']
            elif len(args)<2:
                return args[0]
            else:
                return args[0], args[1]
        
        def set_forward_mode(module, pass_through=True):
            if pass_through:
                module.original_forward = module.forward
                module.forward = dummy_pass
            else:
                try:
                    module.forward = module.original_forward
                except AttributeError:
                    pass
        
        # Switch mode
        if len(self.config.guidance_blocks) != 0:
            index = max(self.config.guidance_blocks)
            for i, block in enumerate(self.transformer.transformer_blocks):
                if i > index:
                    set_forward_mode(block, pass_through=train)
        for block in [self.transformer.norm_out, self.transformer.norm_final]:
            set_forward_mode(block, pass_through=train)
    
    def compute_motion_flow_loss(self, x, ts, step_i, rope=None, pos_emb=None):
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            self.transformer(
                hidden_states=x,
                encoder_hidden_states=self.guidance_embeds[1:2],
                timestep=ts.expand(x.shape[0]).to('cuda'),
                rope=rope,
                pos_embedding=pos_emb,
                return_dict=False,
                cross_attention_kwargs={"timestep_inference": ts}
            )

        # Attention guidance
        total_loss = 0
        for block_id in self.config.guidance_blocks:
            frame_size = self.patches_height * self.patches_width
            module = self.transformer.transformer_blocks[block_id].attn1.processor
            file_name_stem = os.path.splitext(self.config.file_name)[0]
            save_prefix = f"{file_name_stem}_block{block_id}_timestep{ts.item()}_optstep{step_i}"
            motion_flow = compute_motion_flow_with_bonus(module.query, module.key, module.value,
                                                    context_length=self.config.text_seq_length,
                                                    num_frame=self.latent_num_frames,
                                                    frame_size=frame_size,
                                                    h=self.patches_height, 
                                                    w=self.patches_width, 
                                                    temp=self.config.motion_temp, 
                                                    output_dir=self.output_path,
                                                    save_prefix=save_prefix,
                                                    dift_displacement_map=self.prior[0])
            
            ref_motion_flow = self.motion_attn_features[module.block_name]

            # Threshold loss on motion flow (d x 1350 x 2) for d displacement maps
            if self.config.threshloss:
                flow_norms = torch.norm(ref_motion_flow.to(torch.float32), dim=-1) # torch.float32
                idxs = flow_norms > 0
                temporal_loss = F.mse_loss(ref_motion_flow[idxs].to(torch.float32), motion_flow[idxs])
            else:
                temporal_loss = F.mse_loss(ref_motion_flow, motion_flow)

            total_loss = temporal_loss 

        if len(self.config.guidance_blocks) > 0:
            total_loss /= len(self.config.guidance_blocks)
        
        for block_id in self.config.guidance_blocks:
            self.transformer.transformer_blocks[block_id].attn1.processor.query = None
            self.transformer.transformer_blocks[block_id].attn1.processor.key = None
            self.transformer.transformer_blocks[block_id].attn1.processor.value = None
        return total_loss



    def guidance_step(self, x, i, t, mode, loss_type):
       
        for block_id in self.config.guidance_blocks:
            self.transformer.transformer_blocks[block_id].attn1.processor.inject_kv = False
            self.transformer.transformer_blocks[block_id].attn1.processor.copy_kv = True
            self.transformer.transformer_blocks[block_id].attn1.processor.inject_feature = False
            self.transformer.transformer_blocks[block_id].attn1.processor.copy_feature = False
        
        lr = self.lr_range[i]
        optimized_emb = None
        optimized_rope = None
        self.change_mode(train=True)
        
        scaler = GradScaler()

        if loss_type == "flow":
            loss_method = self.compute_motion_flow_loss
        elif loss_type == "moft":
            loss_method = self.compute_moft_loss
        elif loss_type == "smm":
            loss_method = self.compute_smm_loss
        else:
            print("Invalid loss type")
        
        if mode=="rope":
            if self.transformer.trainable_rope is None:
                optimized_rope = torch.stack([self.transformer.init_rope, self.transformer.init_rope], dim=0)
            else:
                optimized_rope = self.transformer.trainable_rope
            
            optimized_rope = optimized_rope.clone().detach().to(dtype=torch.float32, device=self.device).requires_grad_(True)
            optimizer = torch.optim.Adam([optimized_rope], lr=lr)

            for step_i in tqdm(range(self.config.optimization_steps)):
                optimizer.zero_grad()

                total_loss = loss_method(x, t, step_i, rope=optimized_rope)
                
                if self.config.verbose:
                    print(f"Loss t={t}: {total_loss.item()}")
                scaler.scale(total_loss).backward()

                scaler.step(optimizer)
                scaler.update()
                clean_memory()
            
            self.transformer.trainable_rope = optimized_rope.detach()
            if self.config.save_embeds:
                os.makedirs(os.path.join(self.output_path, 'embeds'), exist_ok=True)
                torch.save(optimized_rope.detach(), os.path.join(self.output_path, 'embeds', f"rope_{t}.pt"))
            optimized_x = x
        elif mode == "posemb":
            if self.transformer.trainable_pos_embedding is None:
                text_seq_length = self.config.text_seq_length
                seq_length = self.patches_height * self.patches_width * self.latent_num_frames
                optimized_emb = self.transformer.init_pos_embedding[:, text_seq_length:(text_seq_length+seq_length)].clone().detach().to(dtype=torch.float32, device=self.device).requires_grad_(True)
            else:
                optimized_emb = self.transformer.trainable_pos_embedding.clone().detach().to(dtype=torch.float32, device=self.device).requires_grad_(True)

            optimizer = torch.optim.Adam([optimized_emb], lr=lr)

            for step_i in tqdm(range(self.config.optimization_steps)):
                optimizer.zero_grad()

                total_loss = loss_method(x, t, step_i, pos_emb=optimized_emb)

                if self.config.verbose:
                    print(f"Loss t={t}: {total_loss.item()}")
                scaler.scale(total_loss).backward()

                scaler.step(optimizer)
                scaler.update()
                clean_memory()
            self.transformer.trainable_pos_embedding = optimized_emb.detach()
            if self.config.save_embeds:
                os.makedirs(os.path.join(self.output_path, 'embeds'), exist_ok=True)
                torch.save(optimized_emb.detach(), os.path.join(self.output_path, 'embeds', f"posemb_{t}.pt"))
            optimized_x = x
        elif mode=="latent":
            optimized_x = x.clone().detach().to(dtype=torch.float32).requires_grad_(True)
            optimizer = torch.optim.Adam([optimized_x], lr=lr)

            for step_i in tqdm(range(self.config.optimization_steps)):
                optimizer.zero_grad()

                total_loss = loss_method(optimized_x, t, step_i)
                
                if self.config.verbose:
                    print(f"Loss t={t}: {total_loss.item()}")
                scaler.scale(total_loss.float()).backward()

                scaler.step(optimizer)
                scaler.update()
            
            if self.config.save_embeds:
                os.makedirs(os.path.join(self.output_path, 'embeds'), exist_ok=True)
                torch.save(optimized_x, os.path.join(self.output_path, 'embeds', f"latent_{t}.pt"))
                
        self.change_mode(train=False)
        return optimized_x.detach(), optimized_emb, optimized_rope

    @torch.no_grad()
    def denoise_step(self, latents, i, prompt_embeds, old_pred_original_sample, pos_emb=None, rope=None):

        t = self.timesteps[i]

        latent_model_input = latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        ts = t.expand(latent_model_input.shape[0]).to('cuda')

        cross_attention_kwargs = {"timestep": t}

        noise_pred_text = self.transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds[1:2],
            timestep=ts,
            return_dict=False,
            pos_embedding=pos_emb,
            rope=rope,
            cross_attention_kwargs=cross_attention_kwargs, 
        )[0]
        noise_pred_text = noise_pred_text.float()

        noise_pred_uncond = self.transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds[:1],
            timestep=ts,
            return_dict=False,
            pos_embedding=pos_emb,
            rope=rope,
            cross_attention_kwargs=cross_attention_kwargs, 
        )[0]
        noise_pred_uncond = noise_pred_uncond.float()


        if self.use_dynamic_cfg:
            self._guidance_scale = 1 + self.config.guidance_scale * (
                (1 - math.cos(math.pi * ((self.num_inference_steps - t.item()) / self.num_inference_steps) ** 5.0)) / 2
            )

        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        if not isinstance(self.scheduler, CogVideoXDPMScheduler):
            # CogVideo-2B
            latents = self.scheduler.step(noise_pred, t, latents, **self.extra_step_kwargs, return_dict=False)[0]
        else:
            latents, old_pred_original_sample = self.scheduler.step(
                noise_pred,
                old_pred_original_sample,
                t,
                self.timesteps[i - 1] if i > 0 else None,
                latents,
                **self.extra_step_kwargs,
                return_dict=False,
            )
        latents = latents.to(prompt_embeds.dtype)

        return latents, old_pred_original_sample

    @torch.no_grad()
    # @torch.autocast(device_type="cuda")
    def run(self, pos_emb=None, rope=None):
        clean_memory()
        latents = self.init_latents
        x0_prev = None # for DPM-solver++
        
        for i, t in enumerate(tqdm(self.timesteps, desc="Sampling")):
            is_guidance_step = t in self.guidance_schedule
            # Clear embeddings after guidance phase
            if not is_guidance_step:
                pos_emb = rope = None
            
            # KV Injection
            for block_id in self.config.injection_blocks:
                processor = self.transformer.transformer_blocks[block_id].attn1.processor
                processor.inject_kv = False
                processor.copy_kv = True
                processor.inject_feature = False
                processor.copy_feature = False

            if is_guidance_step:
                # Store KV from motion video in injection_blocks
                noise = self.init_latents
                noisy_latent = self.scheduler.add_noise(self.motion_latent, noise, t)
                noisy_latent = self.scheduler.scale_model_input(noisy_latent, t)
                
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    self.transformer(
                        hidden_states=noisy_latent,
                        encoder_hidden_states=self.guidance_embeds[1:2],
                        timestep=t.expand(noisy_latent.shape[0]).to('cuda'),
                        return_dict=False,
                    )
                with torch.no_grad():
                    # with torch.autocast(device_type="cuda", enabled=False):
                    for block_id in self.config.injection_blocks:
                        processor = self.transformer.transformer_blocks[block_id].attn1.processor
                        processor.inject_kv = True
                        processor.copy_kv = False
                        processor.inject_feature = False
                        processor.copy_feature = False
                        
                        mask_head_byentropy(processor=processor, num_sampled_rows=128, entropy_threshold=7)
                    
            
            # Apply guidance if needed
            with torch.enable_grad():
                if is_guidance_step and self.config.guidance_blocks:
                    if not self.config.inject_embeds:
                        latents, pos_emb, rope = self.guidance_step(latents, i, t, 
                                                                    mode=self.config.guidance_mode, loss_type=self.config.loss_type)
                    else:
                        # 🔄 Zero-shot Motion Injection - Load pre-computed embeddings
                        embeds_path = os.path.join(self.output_path, "embeds")
                        if self.config.guidance_mode == "rope":
                            rope = torch.load(os.path.join(embeds_path, f"rope_{t}.pt")).to(dtype=self.dtype, device=self.device)
                        elif self.config.guidance_mode == "posemb":
                            pos_emb = torch.load(os.path.join(embeds_path, f"posemb_{t}.pt")).to(dtype=self.dtype, device=self.device)
                        elif self.config.guidance_mode == "latent":
                            latents = torch.load(os.path.join(embeds_path, f"latent_{t}.pt")).to(dtype=self.dtype, device=self.device)
            
            # Perform denoising step
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                latents, x0_prev = self.denoise_step(
                    latents, 
                    i, 
                    self.guidance_embeds,
                    x0_prev,
                    pos_emb=pos_emb,
                    rope=rope,
                )
        
        # Decode and save results
        with torch.no_grad():
            decoded_frames = self.pipe.decode_latents(latents)
        video = self.pipe.video_processor.postprocess_video(video=decoded_frames, output_type='pil')[0]
        result_name = getattr(self.config, "file_name", "result")  
        save_path = Path(self.config["output_path"]) / result_name

        del decoded_frames
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print("Saving to:", save_path)

        result_name = f"results"
        # result_name = f"{prompt_idx+1:04d}"
        if self.config.inject_embeds:
            result_name += '_inject_embeds'
        if self.config.save_format=="frames":
            Path(self.config["output_path"], result_name).mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(video):
                frame.save(Path(self.config["output_path"], result_name, f"{i:04d}.png"))
        elif self.config.save_format=="gif":
            imageio.mimsave(str(Path(self.config["output_path"]) / f"{result_name}.gif"), video, loop=0)
        elif self.config.save_format=="mp4":
            export_to_video(video, str(Path(self.config["output_path"]) / f"{result_name}.mp4"), fps=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video_path", type=str, required=True, help="Motion video path to transfer motion from (.mp4 or directory of .png)")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Prompt for new generation")

    parser.add_argument("--model", type=str, default="5b", choices=['5b','2b'])
    parser.add_argument("-n", "--video_length", type=int, default=49, help="Load the first n frames of the video in video_path")
    parser.add_argument("--negative_prompt", type=str, default="bad quality, distortions, unrealistic, distorted image, watermark, signature", help="Negative prompt for new generation")
    parser.add_argument("--loss_type", type=str, default="flow", choices=["flow", "moft", "smm"], help="Use MOFT or SMM for guidance")
    parser.add_argument("--opt_mode", type=str, default="latent", choices=["latent", "emb"])
    parser.add_argument("--no_guidance", action="store_true", help="Disable guidance")
    parser.add_argument("--no_injection", action="store_true", help="Disable KV injection")
    parser.add_argument("--inject_embeds", action="store_true", help="Inject previously trained embeddings in embeds/ into the new generation specified by the prompt argument")
    parser.add_argument("--output_path", type=str, default="output_path", help="Output path for the generated video")
    parser.add_argument("--seed", type=int, default=1, help="Seed")
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--save_format", type=str, default="mp4", choices=["mp4", "gif", "frames"])
    parser.add_argument("--verbose", action="store_true", help="Print loss values") 
    parser.add_argument("--file_name", type=str, default="test", help="File name for the output video") 
    parser.add_argument("--range", type=int, nargs=2, help="Start and end indices")
    
    
    opt = parser.parse_args()

    config = OmegaConf.load(opt.config_path)

    if opt.no_injection:
        config.injection_blocks = []

    cli_config = {
        'model_key': f"THUDM/CogVideoX-{opt.model}",
        'video_path': opt.video_path,
        'target_prompt': opt.prompt,
        'negative_prompt': opt.negative_prompt,
        'video_length': opt.video_length,
        'output_path': opt.output_path,
        'seed': opt.seed,
        'opt_mode': opt.opt_mode,
        'loss_type': opt.loss_type,
        'save_format': opt.save_format,
        'save_embeds': False, #change!!
        'inject_embeds': opt.inject_embeds,
        'verbose': opt.verbose,
        'file_name': opt.file_name,
        'range': opt.range,
    }
    config = OmegaConf.merge(config, cli_config)

    # Model-specific arguments
    config['guidance_blocks'] = config[f'guidance_blocks_{opt.model}']
    if opt.no_guidance:
        config['guidance_blocks'] = []
    
    if config.opt_mode == "latent":
        config.guidance_mode = "latent"
    elif config.opt_mode == "emb":
        if opt.model == "5b":
            config.guidance_mode = "rope"
        elif opt.model == "2b":
            config.guidance_mode = "posemb"

    config.output_path = os.path.join(config['output_path']) 
    Path(config["output_path"]).mkdir(parents=True, exist_ok=True)
    
    config.video_path = opt.video_path
    guidance = Guidance(config)
    prior = extract_dift_and_cal_sim(config, guidance)
    guidance.prior = prior
    guidance.motion_attn_features = guidance.load_attn_features(inds=guidance.prior[1])
    OmegaConf.save(config, Path(config["output_path"]) / "config.yaml")
    guidance.run()
        
    del guidance
    clean_memory()
   
    
import os
import imageio
import numpy as np
from typing import Union
import decord
decord.bridge.set_bridge('torch')
import torch
import torchvision
import PIL
from typing import List
from tqdm import tqdm
from einops import rearrange
import torchvision.transforms.functional as F
import random
from dift_models.dift_sd import SDFeaturizer
from torchvision.transforms import PILToTensor
from PIL import Image
import os
import cv2
import gc
from .merge import bipartite_soft_matching, random_bipartite_soft_matching

def extract_dift_and_cal_sim(args, video_path, dift_up_ft_index=2, feature_level=64, thre=0.7, occlusion_thre=0.9):
    #dift_save_path
    if os.path.exists(os.path.join(args.dift_save_path, f"position.pth")) and args.load_saved_position:
        positions = torch.load(os.path.join(args.dift_save_path, f"position.pth"))
        frame_indices = torch.load(os.path.join(args.dift_save_path, f"frame_indices.pth"))
        mask = torch.load(os.path.join(args.dift_save_path, f"mask.pth"))
        return [positions[..., 0].reshape(F, H* W, F, 1), positions[..., 1].reshape(F, H* W, F, 1), frame_indices, mask]
        
    dift = SDFeaturizer(args.sd_path)
    
    frame_count = save_frame(args, video_path)
    
    dift_feature = []
    
    if dift_up_ft_index == 0:
        size = feature_level * 32
    if dift_up_ft_index == 1:
        size = feature_level * 16
    if dift_up_ft_index == 2:
        size = feature_level * 8
    for i in range(frame_count): 
        file_path = os.path.join(args.dift_save_path, f"frame_{i}.jpg")
        img = Image.open(file_path).convert('RGB')
        
        # size = feature_level * 8
        img = img.resize([size, size])
        img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
        ft = dift.forward(img_tensor,
                        prompt= 'a photo of ' + args.input_class,
                        t=args.dift_t,
                        up_ft_index=dift_up_ft_index,
                        ensemble_size=args.dift_ensemble_size)
        dift_feature.append(ft)
    
    dift = None
    dift_feature = torch.concat(dift_feature, dim=0)
    dift_feature = dift_feature.permute(0, 2, 3, 1).cuda() #[15, 64, 64, 640]
    
    torch.save(dift_feature, os.path.join(args.dift_save_path, f"dift_feature.pth"))
    
    F, H, W, d = dift_feature.shape
    
    before_merge = dift_feature.reshape(F, H * W, d) #15,4096,640
    merge, unmerge = bipartite_soft_matching(before_merge, int(before_merge.shape[1] * (1-args.merge_ratio)))
    # dift_feature = merge(before_merge) #15,2048,640
    
    dift_feature = dift_feature.reshape(-1, d).half() #F*H*W, d   15*2048, 640
    dift_feature = torch.nn.functional.normalize(dift_feature, p=2, dim=1)
    
    torch.cuda.empty_cache()
    sim = dift_feature @ dift_feature.t() #[F*H*W, F*H*W]  
    # sim = sim.reshape(F, H, W, F, H, W)
    sim = sim.view(int(F* H* W* F * args.merge_ratio) , -1)
    
    values, indices = sim.topk(1, dim=-1)   #[indices: F*H*W*F, 3]

    del sim
    torch.cuda.empty_cache()
    gc.collect()
    
    mask = values > thre
    
    positions = indices.reshape(F, int(H* W * args.merge_ratio), F, 1)
    mask = mask.reshape(F, int(H* W* args.merge_ratio), F, 1)
    
    print('positions',positions.max())
    print('positions',positions.min())
    
    frame_indices = torch.zeros((F, int(H* W* args.merge_ratio), F), dtype=torch.long)
    for i in range(F):
        base_sequence = torch.arange(F)
        sequence = torch.cat((base_sequence[i:i+1], base_sequence[:i], base_sequence[i+1:]))
        frame_indices[i] = sequence.repeat(int(H* W* args.merge_ratio), 1)
        
    frame_indices = frame_indices.unsqueeze(-1).repeat(1,1,1,1)
    
    torch.save(positions, os.path.join(args.dift_save_path, f"position.pth"))
    torch.save(frame_indices, os.path.join(args.dift_save_path, f"frame_indices.pth"))
    torch.save(mask, os.path.join(args.dift_save_path, f"mask.pth"))
    return [positions, frame_indices, mask, merge, unmerge]#, occlusion]
  

def save_frame(args, video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success, frame = cap.read()
    if not os.path.exists(args.dift_save_path):
        os.makedirs(args.dift_save_path)
    while success:
        file_path = os.path.join(args.dift_save_path, f"frame_{frame_count}.jpg")
        cv2.imwrite(file_path, frame)
        success, frame = cap.read()
        frame_count += 1
    cap.release()
    return frame_count
    

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def save_videos_grid_pil(videos: List[PIL.Image.Image], path: str, rescale=False, n_rows=4, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def read_video(video_path, video_length, width=512, height=512, frame_rate=None):
    vr = decord.VideoReader(video_path, width=width, height=height)
    if frame_rate is None:
        frame_rate = max(1, len(vr) // video_length)
    sample_index = list(range(0, len(vr), frame_rate))[:video_length]
    video = vr.get_batch(sample_index)
    video = rearrange(video, "f h w c -> f c h w")
    video = (video / 127.5 - 1.0)
    return video


# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents


"""optical flow and trajectories sampling"""
def preprocess(img1_batch, img2_batch, transforms):
    img1_batch = F.resize(img1_batch, size=[512, 512], antialias=False)
    img2_batch = F.resize(img2_batch, size=[512, 512], antialias=False)
    return transforms(img1_batch, img2_batch)

def keys_with_same_value(dictionary):
    result = {}
    for key, value in dictionary.items():
        if value not in result:
            result[value] = [key]
        else:
            result[value].append(key)

    conflict_points = {}
    for k in result.keys():
        if len(result[k]) > 1:
            conflict_points[k] = result[k]
    return conflict_points

def find_duplicates(input_list):
    seen = set()
    duplicates = set()

    for item in input_list:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)

    return list(duplicates)

def neighbors_index(point, window_size, H, W):
    """return the spatial neighbor indices"""
    t, x, y = point
    neighbors = []
    for i in range(-window_size, window_size + 1):
        for j in range(-window_size, window_size + 1):
            if i == 0 and j == 0:
                continue
            if x + i < 0 or x + i >= H or y + j < 0 or y + j >= W:
                continue
            neighbors.append((t, x + i, y + j))
    return neighbors


@torch.no_grad()
def sample_trajectories(video_path, device):
    from torchvision.models.optical_flow import Raft_Large_Weights
    from torchvision.models.optical_flow import raft_large

    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    frames, _, _ = torchvision.io.read_video(str(video_path), output_format="TCHW")

    clips = list(range(len(frames)))

    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()

    finished_trajectories = []

    current_frames, next_frames = preprocess(frames[clips[:-1]], frames[clips[1:]], transforms)
    list_of_flows = model(current_frames.to(device), next_frames.to(device)) # 14,2,512,512
    predicted_flows = list_of_flows[-1]

    predicted_flows = predicted_flows/512   #14,2,512,512

    resolutions = [64, 32, 16, 8]
    res = {}
    window_sizes = {64: 2,
                    32: 1,
                    16: 1,
                    8: 1}

    for resolution in resolutions:
        print("="*30)
        trajectories = {}
        predicted_flow_resolu = torch.round(resolution*torch.nn.functional.interpolate(predicted_flows, scale_factor=(resolution/512, resolution/512))) #14,2,64,64

        T = predicted_flow_resolu.shape[0]+1
        H = predicted_flow_resolu.shape[2]
        W = predicted_flow_resolu.shape[3]

        is_activated = torch.zeros([T, H, W], dtype=torch.bool)

        for t in range(T-1):
            flow = predicted_flow_resolu[t] #2.64.64
            for h in range(H):
                for w in range(W):

                    if not is_activated[t, h, w]:
                        is_activated[t, h, w] = True
                        # this point has not been traversed, start new trajectory
                        x = h + int(flow[1, h, w])
                        y = w + int(flow[0, h, w])
                        if x >= 0 and x < H and y >= 0 and y < W:
                            # trajectories.append([(t, h, w), (t+1, x, y)])
                            trajectories[(t, h, w)]= (t+1, x, y)

        conflict_points = keys_with_same_value(trajectories) #这一帧的多个点移动到了下一帧的同一点，只留一个点
        for k in conflict_points:
            index_to_pop = random.randint(0, len(conflict_points[k]) - 1)
            conflict_points[k].pop(index_to_pop)
            for point in conflict_points[k]:
                if point[0] != T-1:
                    trajectories[point]= (-1, -1, -1) # stupid padding with (-1, -1, -1)

        active_traj = []
        all_traj = []
        for t in range(T):
            pixel_set = {(t, x//H, x%H):0 for x in range(H*W)}
            new_active_traj = []
            for traj in active_traj:
                if traj[-1] in trajectories:
                    v = trajectories[traj[-1]]
                    new_active_traj.append(traj + [v])
                    pixel_set[v] = 1
                else:
                    all_traj.append(traj)
            active_traj = new_active_traj
            active_traj+=[[pixel] for pixel in pixel_set if pixel_set[pixel] == 0] 
        #轨迹，list中每个元素包含15个(t,x,y)，表示一条轨迹
        all_traj += active_traj

        useful_traj = [i for i in all_traj if len(i)>1]
        for idx in range(len(useful_traj)):
            if useful_traj[idx][-1] == (-1, -1, -1):
                useful_traj[idx] = useful_traj[idx][:-1]
        print("how many points in all trajectories for resolution{}?".format(resolution), sum([len(i) for i in useful_traj]))
        print("how many points in the video for resolution{}?".format(resolution), T*H*W)

        # validate if there are no duplicates in the trajectories
        trajs = []
        for traj in useful_traj:
            trajs = trajs + traj
        assert len(find_duplicates(trajs)) == 0, "There should not be duplicates in the useful trajectories."

        # check if non-appearing points + appearing points = all the points in the video
        all_points = set([(t, x, y) for t in range(T) for x in range(H) for y in range(W)])
        left_points = all_points- set(trajs)
        print("How many points not in the trajectories for resolution{}?".format(resolution), len(left_points))
        for p in list(left_points):
            useful_traj.append([p])
        print("how many points in all trajectories for resolution{} after pending?".format(resolution), sum([len(i) for i in useful_traj]))


        longest_length = max([len(i) for i in useful_traj])
        sequence_length = (window_sizes[resolution]*2+1)**2 + longest_length - 1

        seqs = []
        masks = []

        # create a dictionary to facilitate checking the trajectories to which each point belongs.
        point_to_traj = {}
        for traj in useful_traj:
            for p in traj:
                point_to_traj[p] = traj

        for t in range(T):
            for x in range(H):
                for y in range(W):
                    neighbours = neighbors_index((t,x,y), window_sizes[resolution], H, W)
                    sequence = [(t,x,y)]+neighbours + [(0,0,0) for i in range((window_sizes[resolution]*2+1)**2-1-len(neighbours))]
                    sequence_mask = torch.zeros(sequence_length, dtype=torch.bool)
                    sequence_mask[:len(neighbours)+1] = True

                    traj = point_to_traj[(t,x,y)].copy()
                    traj.remove((t,x,y))
                    sequence = sequence + traj + [(0,0,0) for k in range(longest_length-1-len(traj))]
                    sequence_mask[(window_sizes[resolution]*2+1)**2: (window_sizes[resolution]*2+1)**2 + len(traj)] = True
                    #前半段是neighbor，后半段是traj？
                    seqs.append(sequence)
                    masks.append(sequence_mask)

        seqs = torch.tensor(seqs)
        masks = torch.stack(masks)
        res["traj{}".format(resolution)] = seqs
        res["mask{}".format(resolution)] = masks
    return res


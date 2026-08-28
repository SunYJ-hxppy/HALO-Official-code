# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py

from dataclasses import dataclass
from typing import Optional, Callable
import math
import torch
import torch.nn.functional as F
from torch import nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import FeedForward, AdaLayerNorm
from diffusers.models.cross_attention import CrossAttention
from einops import rearrange, repeat
from .merge import bipartite_soft_matching, random_bipartite_soft_matching
import os
@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class Transformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, t_real, inverse, h_state, attention_t, inds, encoder_hidden_states=None, timestep=None, return_dict: bool = True, \
                inter_frame=False, **kwargs):
        # Input##########################################################################

        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        encoder_hidden_states = repeat(encoder_hidden_states, 'b n c -> (b f) n c', f=video_length)

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        # check resolution
        resolu = hidden_states.shape[-1]

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                t_real, inverse, ###############################
                h_state=h_state, attention_t=attention_t, inds=inds,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length,
                inter_frame=inter_frame,
            )

        # Output
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None

        # Fully
        self.attn1 = FullyFrameAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        # Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = CrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn2 = None

        if cross_attention_dim is not None:
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        else:
            self.norm2 = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            if self.attn2 is not None:
                self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers


    def forward(self, hidden_states, t_real, inverse,h_state, attention_t, inds, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None, \
                inter_frame=False, **kwargs):
        # SparseCausal-Attention
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )

        if self.only_cross_attention:
            hidden_states = (
                self.attn1(norm_hidden_states, encoder_hidden_states, attention_mask=attention_mask, inter_frame=inter_frame, **kwargs) + hidden_states
            )
        else:
            hidden_states = self.attn1(norm_hidden_states, t_real, inverse, h_state=h_state, attention_t=attention_t, inds=inds, attention_mask=attention_mask, video_length=video_length, inter_frame=inter_frame, **kwargs) + hidden_states

        if self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            hidden_states = (
                self.attn2(
                    norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
                )
                + hidden_states
            )

        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        return hidden_states

class FullyFrameAttention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.scale = dim_head**-0.5

        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads
        self._slice_size = None
        self._use_memory_efficient_attention_xformers = False
        self.added_kv_proj_dim = added_kv_proj_dim

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=inner_dim, num_groups=norm_num_groups, eps=1e-5, affine=True)
        else:
            self.group_norm = None

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

        self.q = None
        self.inject_q = None
        self.k = None
        self.inject_k = None


    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_heads_to_batch_dim2(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    def reshape_heads_to_batch_dim3(self, tensor):
        batch_size1, batch_size2, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size1, batch_size2, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 3, 1, 2, 4)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def set_attention_slice(self, slice_size):
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

        self._slice_size = slice_size

    def _attention(self, query, key, value, attention_mask=None):
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # compute attention output
        hidden_states = torch.bmm(attention_probs, value)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim, attention_mask):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]

            if self.upcast_attention:
                query_slice = query_slice.float()
                key_slice = key_slice.float()

            attn_slice = torch.baddbmm(
                torch.empty(slice_size, query.shape[1], key.shape[1], dtype=query_slice.dtype, device=query.device),
                query_slice,
                key_slice.transpose(-1, -2),
                beta=0,
                alpha=self.scale,
            )

            if attention_mask is not None:
                attn_slice = attn_slice + attention_mask[start_idx:end_idx]

            if self.upcast_softmax:
                attn_slice = attn_slice.float()

            attn_slice = attn_slice.softmax(dim=-1)

            # cast back to the original dtype
            attn_slice = attn_slice.to(value.dtype)
            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _memory_efficient_attention_xformers(self, query, key, value, attention_mask):
        # TODO attention_mask
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def forward(self, hidden_states, t_real, inverse, h_state, attention_t, inds, encoder_hidden_states=None, attention_mask=None, video_length=None, inter_frame=False, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states #[30, 4096, 320]

        h = w = int(math.sqrt(sequence_length))
        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2) # [30, 4096, 320]

        query = self.to_q(hidden_states)  # (bf) x d(hw) x c
        self.q = query
        if self.inject_q is not None:
            query = self.inject_q
        dim = query.shape[-1]
        query_old = query.clone()

        # All frames
        query = rearrange(query, "(b f) d c -> b (f d) c", f=video_length) # [2, 61440, 320]
        
        # merge, unmerge = bipartite_soft_matching(query, query.shape[1]//2)
        # query = merge(query)
        
        query = self.reshape_heads_to_batch_dim(query) # [10, 61440, 64]
            
        if self.added_kv_proj_dim is not None:
            raise NotImplementedError
        
        #####################self attn#######################
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states) #[30, 4096, 320]
        self.k = key
        if self.inject_k is not None:
            key = self.inject_k
        key_old = key.clone()
        value = self.to_v(encoder_hidden_states)

        if inter_frame:
            key = rearrange(key, "(b f) d c -> b f d c", f=video_length)[:, [0, -1]]
            value = rearrange(value, "(b f) d c -> b f d c", f=video_length)[:, [0, -1]]
            key = rearrange(key, "b f d c -> b (f d) c",)
            value = rearrange(value, "b f d c -> b (f d) c")
        else:
            # All frames
            key = rearrange(key, "(b f) d c -> b (f d) c", f=video_length) #[2, 61440, 320]
            value = rearrange(value, "(b f) d c -> b (f d) c", f=video_length) #[2, 61440, 320]
            
        # merge, _ = random_bipartite_soft_matching(key, (key.shape[1] * 0.75))
        # key = merge(key)
        # value = merge(value)

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)
                
        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask) #[2*5, 15*64*64, 64]
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        ##############################################################################################
        # hidden_states = unmerge(hidden_states)
        #######################occulution-guided attn############################
        # if h in [64]:
        #     _, _, _, _, occlusion = inds #[15,64,64]
            
        #     Frame_indx, Height_indx, Width_indx = occlusion.shape
        #     occlusion = occlusion.reshape(Frame_indx * Height_indx * Width_indx)
        #     hidden_states_copy = rearrange(hidden_states, "b (f d) c -> (b f) d c", f=video_length) #[30, 4096, 320]
        #     if self.group_norm is not None:
        #         hidden_states_copy = self.group_norm(hidden_states_copy.transpose(1, 2)).transpose(1, 2)
            
        #     query = hidden_states_copy
        #     query = rearrange(query, "(b f) d c -> b (f d) c", f=video_length) # [10, 61440, 64]
        #     query = self.reshape_heads_to_batch_dim(query)
            
        #     key = hidden_states_copy
        #     key = rearrange(key, "(b f) d c -> b (f d) c", f=video_length) #[10, 61440, 64]  [bs,seq_length,1,embedding_size]
        #     key = self.reshape_heads_to_batch_dim(key)
        #     key = key[:, occlusion, :]
        
        #     value = hidden_states_copy
        #     value = rearrange(value, "(b f) d c -> b (f d) c", f=video_length) #[10, 61440, 64]
        #     value = self.reshape_heads_to_batch_dim(value)
        #     value = value[:, occlusion, :]
            
        #     hidden_states_copy = self._memory_efficient_attention_xformers(query, key, value, attention_mask=None) #[2, 15*64*64, 64]
        #     hidden_states_copy = hidden_states_copy.to(query.dtype)
        #     occlusion = occlusion.unsqueeze(0).unsqueeze(2)
            
        #####################optical attn#######################
        if h in [64]:
            
            hidden_states = rearrange(hidden_states, "b (f d) c -> (b f) d c", f=video_length) #[30, 4096, 320]
            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            xy_inds, t_inds, mask, merge, unmerge = inds #occ:[15,64,64]
            
            query = hidden_states   # [30, 4096, 320]
            key = hidden_states    # [30, 4096, 320]
            value = hidden_states
            
            query_tempo = query.unsqueeze(-2)
            
            # if t_real > attention_t:
            if t_real < 100000:
                _key = rearrange(key, '(b f) l d -> b f l d', b=int(batch_size/video_length), f=video_length) #[2, 15, 64, 64, 320]
                _value = rearrange(value, '(b f) l d -> b f l d', b=int(batch_size/video_length), f=video_length) #[2, 15, 64, 64, 320]
                key_tempo = _key[:, t_inds, xy_inds]                             #[2, 15, 4096, 15, 3, 320]
                value_tempo = _value[:, t_inds, xy_inds] 

                
                key_tempo = rearrange(key_tempo, 'b f n l k d -> (b f) n (l k) d')            #[30, 4096, 45, 320]
                value_tempo = rearrange(value_tempo, 'b f n l k d -> (b f) n (l k) d')        #[30, 4096, 45, 320]

            else:
                key_tempo = self.self_tracking_mm_save(key, h, t=t_real, k_single=1)
                value_tempo = self.self_tracking_mm_save(value, h, t=t_real, k_single=1)
            
            
            query_tempo = self.reshape_heads_to_batch_dim3(query_tempo)       
            key_tempo = self.reshape_heads_to_batch_dim3(key_tempo)[:,:,:,:15,:]            
            value_tempo = self.reshape_heads_to_batch_dim3(value_tempo)[:,:,:,:15,:]       
            
            # if t_real > attention_t:
            if t_real < 100000:
                attn_matrix2 = query_tempo @ key_tempo.transpose(-2, -1) / math.sqrt(query_tempo.size(-1)) 
            else:
                attn_matrix2 = query_tempo @ key_tempo.transpose(-2, -1) / math.sqrt(query_tempo.size(-1)) 
            attn_matrix2 = F.softmax(attn_matrix2, dim=-1)
            out = (attn_matrix2@value_tempo).squeeze(-2) #[30, 5, 2048, 64]
            out = rearrange(out,'(b f) k l d -> (b f) l (k d)', b=int(batch_size/video_length), f=video_length)  
            # print(out[0]==out[15])
            # out = unmerge(out[0:out.shape[0]//2]).repeat(2,1,1) #[30, 4096, 320]
            # out = torch.cat((unmerge(out[:out.shape[0]//2]), unmerge(out[out.shape[0]//2:])), dim=0)
            

            hidden_states = rearrange(out,'(b f) (h w) d -> b (f h w) d', b=int(batch_size/video_length), f=video_length, h=h, w=w) #2,61440,320

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)

        # All frames
        hidden_states = rearrange(hidden_states, "b (f d) c -> (b f) d c", f=video_length)

        return hidden_states
    

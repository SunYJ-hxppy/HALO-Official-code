import torch
from typing import Tuple, List
import torch.nn.functional as TF
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.models.embeddings import get_3d_rotary_pos_embed

def prepare_rotary_positional_embeddings(
        height: int,
        width: int,
        num_frames: int,
        vae_scale_factor_spatial: float,
        patch_size: int,
        attention_head_dim: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * patch_size)
        grid_width = width // (vae_scale_factor_spatial * patch_size)
        base_size_width = 720 // (vae_scale_factor_spatial * patch_size)
        base_size_height = 480 // (vae_scale_factor_spatial * patch_size)

        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
            use_real=True,
        )

        freqs_cos = freqs_cos.to(device=device)
        freqs_sin = freqs_sin.to(device=device)
        return torch.stack([freqs_cos, freqs_sin], dim=0)
    
def get_motion_warped_rope(
    displacement_matrices_2d: torch.Tensor,
    rope_embed: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    F, H_dift, W_dift = 13, 30, 45
    D_head_half = rope_embed.shape[-1]
    device = rope_embed.device

    rope_tables = _prepare_static_rope_tables(
        rope_embed, F, H_dift, W_dift, D_head_half
    ).to(device)
   
    normalized_warp_grid = _create_motion_warp_grid(
        displacement_matrices_2d, F, H_dift, W_dift, device
    )
   
    displaced_freqs = TF.grid_sample(
        rope_tables,
        normalized_warp_grid.repeat(2, 1, 1, 1, 1), 
        mode='bilinear', 
        padding_mode='border',
        align_corners=True
    )

    displaced_freqs = displaced_freqs.permute(0, 2, 3, 4, 1)
    
    # (2, F*H*W, D/2)
    displaced_freqs_flat = displaced_freqs.reshape(2, -1, D_head_half)
    
    displaced_cos = displaced_freqs_flat[0]
    displaced_sin = displaced_freqs_flat[1]
    
    return torch.stack([displaced_cos, displaced_sin], dim=0)

def _prepare_static_rope_tables(
    stacked_freqs: torch.Tensor,
    T: int,
    H: int,
    W: int,
    D_head: int
) -> torch.Tensor:
    
    # (2, T*H*W, D) -> (2, T, H, W, D)
    tables = stacked_freqs.view(2, T, H, W, D_head)
    # (2, T, H, W, D) -> (2, D, T, H, W)
    tables = tables.permute(0, 4, 1, 2, 3)
    return tables.contiguous() 


def _create_motion_warp_grid(
    displacement_matrices_2d: List[torch.Tensor],
    F: int,
    H: int,
    W: int,
    device: torch.device
) -> torch.Tensor:
   
    t_coords = torch.arange(F, device=device, dtype=torch.float32).view(F, 1, 1).expand(F, H, W)
    h_coords = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1).expand(F, H, W)
    w_coords = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W).expand(F, H, W)
    
    warped_grid = torch.stack([t_coords, h_coords, w_coords], dim=-1)

    for t in range(1, F):
        # D(f1, f2) -> f1=t, f2=0
        # D(t, 0) = Coords(0) - Coords(t) = B(t, 0) (Backward flow)
        try:
            flow_index = t * F + 0 
            B_t_0 = displacement_matrices_2d[flow_index].to(device) # (H, W, 2) (dy, dx)
        except (IndexError, TypeError):
            print(f"Warning")
            continue
            
        # (h_0, w_0) = (h + dy, w + dx)
        warped_grid[t, ..., 1] = warped_grid[t, ..., 1] + B_t_0[..., 0] # h_0 = h + dy
        warped_grid[t, ..., 2] = warped_grid[t, ..., 2] + B_t_0[..., 1] # w_0 = w + dx

    norm_w = (warped_grid[..., 2] / (W - 1)) * 2 - 1  # x (W)
    norm_h = (warped_grid[..., 1] / (H - 1)) * 2 - 1  # y (H)
    norm_t = (warped_grid[..., 0] / (F - 1)) * 2 - 1  # z (T)
    
    normalized_grid = torch.stack([norm_w, norm_h, norm_t], dim=-1)
    
    return normalized_grid.unsqueeze(0)
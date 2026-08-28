import torch
import torch.nn.functional as F
from einops import rearrange
from guidance_utils.utils import head_classification
import numpy as np
import os 

def compute_motion_flow_withdift_re(
    q, k, v, context_length, num_frame, frame_size, 
    h=30, w=45, temp=5, nframes=4, argmax=False, output_dir=None, save_prefix=None, dift_displacement_map=None, top_k=4
):
    
    def compute_displacement(A, dift_map):
        device = A.device
        
        if argmax:
            N = h * w
            
            def to_coordinates_tensor(indices, width):
                x = indices % width
                y = indices // width
                return torch.stack((x, y), dim=-1) # Shape: (..., 2)
            topk_values, topk_indices = torch.topk(A, k=top_k, dim=-1)
            source_indices = torch.arange(N, device=device)
            coords_source = to_coordinates_tensor(source_indices, width=w) # Shape: (N, 2)

            coords_dift_target = dift_map.reshape(N, 2)

            coords_candidates = to_coordinates_tensor(topk_indices, width=w)
            
            distance_sq = torch.sum((coords_candidates - coords_dift_target.unsqueeze(1))**2, dim=-1)
            
            closest_candidate_idx_in_k = torch.argmin(distance_sq, dim=-1)
            
            best_match_indices = torch.gather(
                topk_indices, 
                1, 
                closest_candidate_idx_in_k.unsqueeze(1)
            ).squeeze(1)


            coords_best_match = to_coordinates_tensor(best_match_indices, width=w) # Shape: (N, 2)
            displacements = coords_best_match - coords_source 

        else:
            # Create grid of relative coordinates
            y_coords, x_coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
            relative_x = x_coords.flatten().unsqueeze(0) - x_coords.flatten().unsqueeze(1)
            relative_y = y_coords.flatten().unsqueeze(0) - y_coords.flatten().unsqueeze(1)
            
            # Compute weighted average
            displacement_x = (relative_x * A).sum(dim=1)
            displacement_y = (relative_y * A).sum(dim=1)
            
            displacements = torch.stack([displacement_x, displacement_y], dim=-1)
        return displacements

    v = v.to(dtype=q.dtype)
    scale = torch.sqrt(torch.tensor(q.shape[-1], dtype=q.dtype, device=q.device))


    q_slice = q[-1, :, 226:, :]
    k_slice = k[-1, :, 226:, :]
    num_heads = q_slice.shape[0]
    seq_len_slice = q_slice.shape[1]
    sum_of_attn_maps = torch.zeros(
        (seq_len_slice, seq_len_slice), 
        device=q.device, 
        dtype=q.dtype
    )
    
    best_mask_idx = head_classification(q, k, v, context_length, num_frame, frame_size)
    head_types = best_mask_idx[-1]

    spatial_indices = (head_types == 1).nonzero(as_tuple=True)[0]
    for i in spatial_indices:
       
        q_head = q_slice[i] # shape: (seq_len_slice, head_dim)
        k_head = k_slice[i] # shape: (seq_len_slice, head_dim)
        
      
        attn_map_head = torch.matmul(q_head, k_head.transpose(-1, -2)) / scale
  
        sum_of_attn_maps += attn_map_head
 
    A_all_mean = sum_of_attn_maps / len(spatial_indices)
    
    total_predicted_flows = 0

    predicted_flows = []
    A_head = A_all_mean
    # Softmax per frame
    nframes_real = A_head.shape[0] // (h*w)  # 8
    A_head = rearrange(A_head, 's (f hw) -> s f hw', f=nframes_real)
    A_head = F.softmax(A_head*temp, dim=-1)
    A_head = rearrange(A_head, '(f1 s1) f2 s2 -> f1 f2 s1 s2', f1=nframes_real, f2=nframes_real, s1=h*w, s2=h*w)
    
    idx = 0
    for frame_i in range(nframes_real): # nframe
        for frame_j in range(nframes_real): # nframe
            current_dift_target_map = dift_displacement_map[idx]
            displacement = compute_displacement(A_head[frame_i, frame_j], dift_map = current_dift_target_map)
            predicted_flows.append(displacement)
            idx += 1

    predicted_flows = torch.stack(predicted_flows, dim=0)

    if output_dir is not None:
        filename = f"{save_prefix}_flow.npz"
        filepath = os.path.join(output_dir, "flow")
        filepath_final = os.path.join(filepath, filename)
        os.makedirs(filepath, exist_ok=True)
        
        flows_to_save = predicted_flows.detach().cpu().numpy()
        np.savez(filepath_final, flows=flows_to_save)
        
    total_predicted_flows += predicted_flows
    del predicted_flows  

    return total_predicted_flows / 1


def compute_motion_flow_with_bonus(
    q, k, v, context_length, num_frame, frame_size, 
    h=30, w=45, temp=5, nframes=4, argmax=False, output_dir=None, save_prefix=None, dift_displacement_map=None, beta=0.1  
):
    def to_coordinates_tensor(indices, width, device):
        x = indices % width
        y = indices // width
        return torch.stack((x, y), dim=-1).to(device)

    def coords_to_flat_index(coords, width):
        x = coords[..., 0]
        y = coords[..., 1]
        return y * width + x
    
    
    def compute_displacement(A):
        device = A.device
        
        if argmax:
            matches = A.argmax(dim=-1)
        
            def to_coordinates(indices, width=w):
                x = indices % width
                y = indices // width
                return x, y

            x1, y1 = to_coordinates(torch.arange(A.shape[0], device=device))
            x2, y2 = to_coordinates(matches)
            dx = x2 - x1
            dy = y2 - y1
            displacements = torch.stack((dx, dy), dim=-1)

        else:
            # Create grid of relative coordinates
            y_coords, x_coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
            relative_x = x_coords.flatten().unsqueeze(0) - x_coords.flatten().unsqueeze(1)
            relative_y = y_coords.flatten().unsqueeze(0) - y_coords.flatten().unsqueeze(1)
            
            # Compute weighted average
            displacement_x = (relative_x * A).sum(dim=1)
            displacement_y = (relative_y * A).sum(dim=1)
            
            displacements = torch.stack([displacement_x, displacement_y], dim=-1)
        return displacements

    v = v.to(dtype=q.dtype)
    scale = torch.sqrt(torch.tensor(q.shape[-1], dtype=q.dtype, device=q.device))

    
    q_slice = q[-1, :, 226:, :]
    k_slice = k[-1, :, 226:, :]
    num_heads = q_slice.shape[0]
    seq_len_slice = q_slice.shape[1]
    sum_of_attn_maps = torch.zeros(
        (seq_len_slice, seq_len_slice), 
        device=q.device, 
        dtype=q.dtype
    )
    
    best_mask_idx = head_classification(q, k, v, context_length, num_frame, frame_size)
    head_types = best_mask_idx[-1]

    spatial_indices = (head_types == 1).nonzero(as_tuple=True)[0]
    for i in spatial_indices:
        q_head = q_slice[i] # shape: (seq_len_slice, head_dim)
        k_head = k_slice[i] # shape: (seq_len_slice, head_dim)
        
        
        attn_map_head = torch.matmul(q_head, k_head.transpose(-1, -2)) / scale
    
  
        sum_of_attn_maps += attn_map_head

    A_all_mean = sum_of_attn_maps / len(spatial_indices)
    
    # ----- Bias Matrix -----
    bias_matrix = torch.zeros_like(A_all_mean) # (seq_len, seq_len)
    
    total_predicted_flows = 0

    predicted_flows = []
    A_head = A_all_mean
    # Softmax per frame
    N = (h*w)
    nframes_real = A_head.shape[0] // N  # 8
    
    idx = 0 
    for frame_i in range(nframes_real):
        for frame_j in range(nframes_real):
            
            dift_disp_ij = dift_displacement_map[idx].reshape(N, 2) # shape (N, 2)

            source_indices = torch.arange(N, device=A_head.device)
            coords_source = to_coordinates_tensor(source_indices, width=w, device=A_head.device)
            coords_dift_target = coords_source + dift_disp_ij
            coords_dift_target[..., 0].clamp_(0, w - 1)
            coords_dift_target[..., 1].clamp_(0, h - 1)
            coords_dift_target_int = torch.round(coords_dift_target).long()
            target_indices_flat = coords_to_flat_index(coords_dift_target_int, width=w) # Shape: (N,)

            start_i, end_i = frame_i * N, (frame_i + 1) * N
            start_j, end_j = frame_j * N, (frame_j + 1) * N
            

            current_bias = torch.zeros((N, N), device=A_head.device, dtype=A_head.dtype)
            current_bias[source_indices, target_indices_flat] = beta
            bias_matrix[start_i:end_i, start_j:end_j] = current_bias
            idx += 1


    A_head = A_head + bias_matrix
    
    A_head = rearrange(A_head, 's (f hw) -> s f hw', f=nframes_real)
    
    
    
    A_head = F.softmax(A_head*temp, dim=-1)
    A_head = rearrange(A_head, '(f1 s1) f2 s2 -> f1 f2 s1 s2', f1=nframes_real, f2=nframes_real, s1=h*w, s2=h*w)
    

    for frame_i in range(nframes_real): # nframe
        for frame_j in range(nframes_real): # nframe
            displacement = compute_displacement(A_head[frame_i, frame_j])
            predicted_flows.append(displacement)

    predicted_flows = torch.stack(predicted_flows, dim=0)

    # if output_dir is not None:
    #     filename = f"{save_prefix}_flow.npz"
    #     filepath = os.path.join(output_dir, "flow")
    #     filepath_final = os.path.join(filepath, filename)
    #     os.makedirs(filepath, exist_ok=True)
        
    #     flows_to_save = predicted_flows.detach().cpu().numpy()
    #     np.savez(filepath_final, flows=flows_to_save)
        
    total_predicted_flows += predicted_flows
    del predicted_flows  

    return total_predicted_flows / 1



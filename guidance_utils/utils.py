import math
from math import floor
import os
import random
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_attention_mask(mask_name, context_length, num_frame, frame_size):
    # TODO: Replace with real implementation

    attention_mask = torch.zeros((context_length + num_frame * frame_size, context_length + num_frame * frame_size)).cuda()
    
    if mask_name == "spatial":
        attention_mask[:context_length, :] = 1
        attention_mask[:, :context_length] = 1
        block_size, block_thres = 128, frame_size * 1.5
        num_block = math.ceil(num_frame * frame_size / block_size)
        for i in range(num_block):
            for j in range(num_block):
                if abs(i - j) < block_thres // block_size:
                    attention_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1

    elif mask_name == "temporal":
        pixel_attn_mask = torch.zeros_like(attention_mask[context_length:, context_length:])

        block_size, block_thres = 128, frame_size * 1.5
        num_block = math.ceil(num_frame * frame_size / block_size)
        for i in range(num_block):
            for j in range(num_block):
                if abs(i - j) < block_thres // block_size:
                    pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1

        pixel_attn_mask = pixel_attn_mask.reshape(frame_size, num_frame, frame_size, num_frame)\
            .permute(1, 0, 3, 2).reshape(frame_size * num_frame, frame_size * num_frame)
        attention_mask[context_length:, context_length:] = pixel_attn_mask

    return attention_mask


def sample_mse(query, key, value, attention_masks=None):

    # Get Attention Masks 
    assert len(attention_masks) == 2
    generator = torch.Generator(device='cpu').manual_seed(42)

    # Query Sampling for MSE
    cfg, num_heads, seq_len, dim = query.size()
    num_sampled_rows = 32
    num_sampled_rows = min(num_sampled_rows, seq_len)
    
    sampled_rows = torch.randint(low=0, high=seq_len, size=(num_sampled_rows,), generator=generator).sort().values
    sampled_q = query[:, :, sampled_rows, :]

    sampled_qk_scores = torch.matmul(sampled_q, key.transpose(-2, -1)) / (dim**0.5)
    
    # Attention 
    sampled_attn_weights = F.softmax(sampled_qk_scores, dim=-1)
    sampled_attn_weights = sampled_attn_weights.to(dtype=value.dtype)
    sampled_golden_hidden_states = torch.matmul(sampled_attn_weights, value)  # (1, seq_len, dim)

    
    mse_list = [] 

    # Only have Tri-diagonal and Striped

    for mask_idx, attn_mask in enumerate(attention_masks):
        sampled_attention_mask = attn_mask[sampled_rows, :].to(device=query.device)
        sampled_attention_scores = sampled_qk_scores.masked_fill(sampled_attention_mask == 0, float('-inf'))
        sampled_attn_weights = F.softmax(sampled_attention_scores, dim=-1)
        sampled_attn_weights = torch.nan_to_num(sampled_attn_weights, nan=0.0) 
        sampled_hidden_states = torch.matmul(sampled_attn_weights, value) 
        mse = torch.mean((sampled_hidden_states - sampled_golden_hidden_states) ** 2, dim=(2, 3))
        
        mse_list.append(mse) 

    sampled_mses = torch.stack(mse_list, dim=0) 
    
    return sampled_mses, attention_masks

def head_classification(q, k, v, context_length, num_frame, frame_size):
    """Classify attention heads into spatial and temporal using sample MSE"""
    
    masks = ["spatial", "temporal"]
    class_attention_masks = [get_attention_mask(mask_name, context_length, num_frame, frame_size) for mask_name in masks]
    sampled_mses, attention_masks = sample_mse(q, k, v, class_attention_masks)
    safe_mses = torch.nan_to_num(sampled_mses.detach(), nan=float('inf'))
    best_mask_idx = torch.argmin(safe_mses.detach(), dim=0)  # (cfg, num_heads)

    return best_mask_idx


def mask_head_byentropy(processor, num_sampled_rows=32, entropy_threshold=7):
    query = processor.query[-1:, :, 226:].detach().to(torch.float32)
    key = processor.key[-1:, :, 226:].detach().to(torch.float32)
    
    scale = torch.sqrt(torch.tensor(query.shape[-1], dtype=torch.float32))
    
    generator = torch.Generator(device='cpu').manual_seed(42)
    _, num_heads, seq_len, dim = query.size()

    num_sampled_rows = min(num_sampled_rows, seq_len)

    sampled_rows = torch.randint(low=0, high=seq_len, size=(num_sampled_rows,), generator=generator).sort().values
    sampled_q = query[:, :, sampled_rows, :]

    attn_scores_chunk = torch.matmul(sampled_q, key.transpose(-1, -2)) / scale
        
    attn_probs_chunk = F.softmax(attn_scores_chunk.to(torch.float32), dim=-1).to(key.dtype)
    epsilon = 1e-9
    entropy_chunk = -torch.sum(attn_probs_chunk * torch.log(attn_probs_chunk + epsilon), dim=-1)
        
    avg_entropy_per_head = entropy_chunk.mean(dim=-1)
    
    del attn_probs_chunk, entropy_chunk
    processor.heads_to_inject_mask = (avg_entropy_per_head[-1] < entropy_threshold).to(query.device)
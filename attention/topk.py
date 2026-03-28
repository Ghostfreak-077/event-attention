import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKAttention(nn.Module):
    """Custom Top-K Attention mechanism that only attends to k most relevant tokens"""
    def __init__(self, dim, num_heads=8, k=5, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.k = k

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)

        # Top-K selection: only keep top-k attention scores per query
        k_val = min(self.k, N)
        topk_vals, topk_indices = torch.topk(attn, k_val, dim=-1)

        # Create mask: set non-topk values to -inf
        mask = torch.full_like(attn, float('-inf'))
        mask.scatter_(-1, topk_indices, topk_vals)

        # Apply softmax only on top-k values
        attn = F.softmax(mask, dim=-1)
        attn = self.attn_drop(attn)

        # Compute output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn  # Return attention weights for visualization
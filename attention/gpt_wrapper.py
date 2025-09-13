import torch.nn as nn
import torch.nn.functional as F

class FlashAttentionWrapper(nn.Module):
    def __init__(self, orig_attn):
        super().__init__()
        # Reuse weights from the original attention layer
        self.c_attn = orig_attn.c_attn
        self.c_proj = orig_attn.c_proj
        self.num_heads = orig_attn.num_heads
        self.split_size = orig_attn.split_size  # = hidden_dim
        self.head_dim = self.split_size // self.num_heads

    def forward(self, hidden_states, **kwargs):
        # Compute QKV projections
        qkv = self.c_attn(hidden_states)  # [batch, seq, 3*hidden_dim]
        query, key, value = qkv.split(self.split_size, dim=2)

        bsz, seq_len, _ = query.shape

        # Reshape to [batch, seq, n_heads, head_dim]
        query = query.view(bsz, seq_len, self.num_heads, self.head_dim)
        key   = key.view(bsz, seq_len, self.num_heads, self.head_dim)
        value = value.view(bsz, seq_len, self.num_heads, self.head_dim)

        # FlashAttention expects [batch, seq, n_heads, head_dim]
        # attn_output = flash_attn_func(
        #     query, key, value,
        #     dropout_p=0.0,
        #     softmax_scale=None,
        #     causal=True
        # )

        attn_output = F.scaled_dot_product_attention(
            query.transpose(1, 2),  # to [batch, heads, seq, head_dim]
            key.transpose(1, 2),
            value.transpose(1, 2),
            is_causal=True
        ).transpose(1, 2)

        # Merge heads back: [batch, seq, hidden_dim]
        attn_output = attn_output.reshape(bsz, seq_len, self.split_size)
        return self.c_proj(attn_output), None
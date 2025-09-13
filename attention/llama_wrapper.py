import torch.nn as nn
import torch.nn.functional as F

class LlamaSDPAWrapper(nn.Module):
    def __init__(self, orig_attn, config):
        super().__init__()
        # Copy original projections
        self.q_proj = orig_attn.q_proj
        self.k_proj = orig_attn.k_proj
        self.v_proj = orig_attn.v_proj
        self.o_proj = orig_attn.o_proj

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        cache_position=None,
        use_cache=False,
        position_embeddings=None,
    ):

        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)  # [bsz, seq_len, hidden_size]
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape Q/K/V to heads
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Expand K/V if using grouped query attention
        if self.num_key_value_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)

        # SDPA
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)

        out = out.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        out = self.o_proj(out)

        # Match LLaMA API: return (attn_output, present_key_value, attn_weights)
        return (out, None)
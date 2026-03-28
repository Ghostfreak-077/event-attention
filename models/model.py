import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from attention.topk import TopKAttention


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, k, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TopKAttention(dim, num_heads=num_heads, k=k, qkv_bias=qkv_bias,
                                   attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), dropout=drop)

    def forward(self, x):
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class StreamingTransformer(nn.Module):
    """
    Streaming Transformer for video classification on Kinetics-400
    Processes frames one by one, maintaining temporal context with Top-K attention
    """
    def __init__(
        self,
        num_classes=400,
        frame_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=512,
        depth=6,
        num_heads=8,
        k=5,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        max_frames=64
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_patches = (frame_size // patch_size) ** 2
        self.max_frames = max_frames

        # Patch embedding for each frame
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.temporal_embed = nn.Parameter(torch.zeros(1, max_frames, embed_dim))

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, k, mlp_ratio, qkv_bias,
                           drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Dropout
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def process_frame(self, frame, frame_idx):
        """Process a single frame and return its embedding"""
        B = frame.shape[0]

        # Patch embedding: (B, C, H, W) -> (B, embed_dim, h, w) -> (B, embed_dim, num_patches)
        x = self.patch_embed(frame)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed[:, 1:, :]

        # Add temporal embedding
        if frame_idx < self.max_frames:
            x = x + self.temporal_embed[:, frame_idx:frame_idx+1, :].expand(-1, x.shape[1], -1)

        return x

    def forward(self, frames):
        """
        Args:
            frames: (B, T, C, H, W) - batch of T frames
        Returns:
            logits: (B, num_classes)
            attention_weights: list of attention weights for visualization
        """
        B, T, C, H, W = frames.shape

        # Initialize with CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = cls_tokens + self.pos_embed[:, :1, :]

        attention_weights_all = []

        # Process frames one by one (streaming)
        for t in range(T):
            # Process current frame
            frame_embed = self.process_frame(frames[:, t], t)

            # Concatenate with existing sequence
            x = torch.cat([x, frame_embed], dim=1)

            # Apply transformer blocks with Top-K attention
            attn_weights_frame = []
            for block in self.blocks:
                x, attn = block(x)
                attn_weights_frame.append(attn)

            attention_weights_all.append(attn_weights_frame)

        # Final normalization
        x = self.norm(x)

        # Use CLS token for classification
        cls_output = x[:, 0]
        logits = self.head(cls_output)

        return logits, attention_weights_all

    def forward_streaming(self, frame, state=None):
        """
        Streaming inference: process one frame at a time
        Args:
            frame: (B, C, H, W) - single frame
            state: dict containing previous state (embeddings, frame_count)
        Returns:
            logits: (B, num_classes)
            new_state: updated state dict
        """
        B = frame.shape[0]

        if state is None:
            # Initialize state
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = cls_tokens + self.pos_embed[:, :1, :]
            frame_idx = 0
        else:
            x = state['embeddings']
            frame_idx = state['frame_count']

        # Process current frame
        frame_embed = self.process_frame(frame, frame_idx)

        # Concatenate with existing sequence
        x = torch.cat([x, frame_embed], dim=1)

        # Apply transformer blocks
        for block in self.blocks:
            x, _ = block(x)

        # Normalize and classify
        x_norm = self.norm(x)
        cls_output = x_norm[:, 0]
        logits = self.head(cls_output)

        # Update state
        new_state = {
            'embeddings': x,
            'frame_count': frame_idx + 1
        }

        return logits, new_state

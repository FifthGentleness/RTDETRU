# Frequency-Aware Affinity version of RT-DETR AIFI.
#
# Reference:
#   Frequency-Aware Affinity for Weakly Supervised Semantic Segmentation
#   CVPR 2026. https://github.com/yay97/DFA
#
# This AIFI keeps the original residual + FFN structure and replaces the
# single global attention with a low-frequency affinity branch plus a
# high-frequency affinity branch. The low branch preserves object-level
# semantics; the high branch enhances edge/texture evidence useful for small
# objects.

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AIFI_DFA']


class AffinityAttention(nn.Module):
    """Pairwise affinity attention used by each frequency branch."""

    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim ({dim}) must be divisible by num_heads ({num_heads})')
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        b, n, c = x.shape
        h, d = self.num_heads, self.head_dim
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, n, h, d).permute(0, 2, 1, 3) for t in qkv]

        affinity = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        affinity = self.attn_drop(affinity.softmax(dim=-1))
        out = torch.matmul(affinity, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class DualFrequencyAffinity(nn.Module):
    """Low- and high-frequency affinity branches with a learned gate."""

    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.low_attn = AffinityAttention(dim, num_heads, dropout)
        self.high_attn = AffinityAttention(dim, num_heads, dropout)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, low, high):
        low_out = self.low_attn(low)
        high_out = self.high_attn(high)
        gate = torch.sigmoid(self.gate(torch.cat([low_out, high_out], dim=-1)))
        return low_out + gate * high_out


class AIFI_DFA(nn.Module):
    """AIFI with dual low-/high-frequency affinity branches.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0,
                 act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = DualFrequencyAffinity(c1, num_heads, dropout)
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256,
                                           temperature=10000.0):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sincos embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.cat([torch.sin(out_w), torch.cos(out_w),
                          torch.sin(out_h), torch.cos(out_h)], dim=1)[None]

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.c1:
            raise ValueError(f'AIFI_DFA expected {self.c1} channels, got {c}')

        pos = self.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)

        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = x - low
        low_tokens = low.flatten(2).permute(0, 2, 1) + pos
        high_tokens = high.flatten(2).permute(0, 2, 1) + pos

        if self.normalize_before:
            src = x.flatten(2).permute(0, 2, 1)
            src = src + self.dropout1(self.attn(
                self.norm1(low_tokens), self.norm1(high_tokens)))
            src_norm = self.norm2(src)
            src = src + self.dropout2(self.fc2(self.dropout(self.act(self.fc1(src_norm)))))
        else:
            src = x.flatten(2).permute(0, 2, 1)
            src = src + self.dropout1(self.attn(low_tokens, high_tokens))
            src = self.norm1(src)
            src = src + self.dropout2(self.fc2(self.dropout(self.act(self.fc1(src)))))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()

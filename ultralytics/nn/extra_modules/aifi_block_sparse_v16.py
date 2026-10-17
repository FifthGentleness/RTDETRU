# AIFI_BlockSparseV16: BlockSparseV1 (global block-sparse attention) fused with
# DCI-DETR "AIFI-C" (TransformerEncoderLayer_LocalWindowAttention, i.e. EfficientViT
# CascadedGroupAttention inside local windows).
#
# Fusion design: parallel dual-branch with a zero-initialized per-channel gate.
#   - Global branch: AIFI_BlockSparse (token-level top-k block routing, sincos pos).
#   - Local branch:  CascadedGroupAttention over 7x7 windows applied directly to the
#                    BCHW feature map (no positional embedding needed).
#   - out_tokens = tokens + dropout(attn_global(tokens)) + gamma * local(x)
#     with gamma init to zero, so step 0 of training is exactly BlockSparseV1 and
#     the network learns how much local cascaded attention to inject.
#
# Reference:
#   Block-Sparse Global Attention for Efficient Multi-View Geometry Transformers,
#   CVPR 2026. https://github.com/brianwang00001/sparse-vggt
#   EfficientViT: Memory Efficient Attention Design with Cascaded Group Attention,
#   CVPR 2023 (CascadedGroupAttention / LocalWindowAttention, as used by DCI-DETR).

import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aifi_block_sparse import BlockSparseAttention

__all__ = ['AIFI_BlockSparseV16']


class Conv2d_BN(torch.nn.Sequential):
    """Conv2d + BatchNorm2d with controllable BN init (from EfficientViT)."""

    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1.0):
        super().__init__()
        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)


class CascadedGroupAttention(nn.Module):
    """EfficientViT Cascaded Group Attention (2D, windowed)."""

    def __init__(self, dim, key_dim, num_heads=4, attn_ratio=4,
                 resolution=7, kernels=(5, 5, 5, 5)):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = dim // num_heads
        self.attn_ratio = attn_ratio

        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(Conv2d_BN(dim // num_heads,
                                  self.key_dim * 2 + self.d))
            dws.append(Conv2d_BN(self.key_dim, self.key_dim, kernels[i], 1,
                                 kernels[i] // 2, groups=self.key_dim))
        self.qkvs = nn.ModuleList(qkvs)
        self.dws = nn.ModuleList(dws)
        self.proj = nn.Sequential(
            nn.ReLU(), Conv2d_BN(self.d * num_heads, dim, bn_weight_init=0.0))

        points = list(itertools.product(range(resolution), range(resolution)))
        n = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(n, n))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x: (B, C, H, W), H=W=window_resolution
        b, _, h, w = x.shape
        training_ab = self.attention_biases[:, self.attention_bias_idxs]
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0:  # cascade: add previous head output to current head input
                feat = feat + feats_in[i]
            feat = qkv(feat)
            q, k, v = feat.view(b, -1, h, w).split(
                [self.key_dim, self.key_dim, self.d], dim=1)
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
            attn = (
                (q.transpose(-2, -1) @ k) * self.scale
                + (training_ab[i] if self.training else self.ab[i])
            )
            attn = attn.softmax(dim=-1)
            feat = (v @ attn.transpose(-2, -1)).view(b, self.d, h, w)
            feats_out.append(feat)
        return self.proj(torch.cat(feats_out, 1))


class LocalWindowAttention(nn.Module):
    """EfficientViT Local Window Attention over arbitrary (H, W) maps."""

    def __init__(self, dim, key_dim=16, num_heads=4, attn_ratio=4,
                 window_resolution=7, kernels=(5, 5, 5, 5)):
        super().__init__()
        self.window_resolution = int(window_resolution)
        assert self.window_resolution > 0, 'window_resolution must be > 0'
        self.attn = CascadedGroupAttention(
            dim, key_dim, num_heads=num_heads, attn_ratio=attn_ratio,
            resolution=self.window_resolution, kernels=tuple(kernels))

    def forward(self, x):
        b, c, h, w = x.shape
        ws = self.window_resolution
        if h <= ws and w <= ws:
            return self.attn(x)

        x = x.permute(0, 2, 3, 1)
        pad_b = (ws - h % ws) % ws
        pad_r = (ws - w % ws) % ws
        padding = pad_b > 0 or pad_r > 0
        if padding:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

        ph, pw = h + pad_b, w + pad_r
        nh, nw = ph // ws, pw // ws
        x = x.view(b, nh, ws, nw, ws, c).transpose(2, 3).reshape(
            b * nh * nw, ws, ws, c).permute(0, 3, 1, 2)
        x = self.attn(x)
        x = x.permute(0, 2, 3, 1).view(b, nh, nw, ws, ws, c).transpose(2, 3).reshape(b, ph, pw, c)
        if padding:
            x = x[:, :h, :w].contiguous()
        return x.permute(0, 3, 1, 2)


class AIFI_BlockSparseV16(nn.Module):
    """AIFI_BlockSparse (V1) + DCI-DETR AIFI-C (local cascaded group attention).

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Heads of the global block-sparse attention.
        block_size: Tokens per sparse routing block.
        topk: Key blocks attended by each query block.
        window_resolution: Local window size (7, same as DCI-DETR AIFI-C).
        num_heads_local: Heads of the local cascaded group attention.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=2048, num_heads=8, block_size=4, topk=4,
                 window_resolution=7, num_heads_local=4, dropout=0.0,
                 act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before

        # Global branch: identical to AIFI_BlockSparse (V1).
        self.attn = BlockSparseAttention(
            dim=c1, num_heads=num_heads, block_size=block_size,
            topk=topk, dropout=dropout)

        # Local branch: DCI-DETR AIFI-C (CascadedGroupAttention in windows).
        self.local_attn = LocalWindowAttention(
            dim=c1, num_heads=num_heads_local,
            window_resolution=window_resolution)

        # Zero-initialized gate: training starts as pure BlockSparseV1.
        self.gamma = nn.Parameter(torch.zeros(c1))

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
            raise ValueError(
                f'AIFI_BlockSparseV16 expected {self.c1} channels, got {c}')

        pos = self.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1) + pos

        # Local cascaded-group-attention branch on the BCHW map (AIFI-C).
        local = self.local_attn(x).flatten(2).permute(0, 2, 1)

        if self.normalize_before:
            src = src + self.dropout1(self.attn(self.norm1(src))) \
                     + self.gamma * local
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(self.norm2(src))))))
        else:
            src = src + self.dropout1(self.attn(src)) + self.gamma * local
            src = self.norm1(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src)))))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()

    def extra_repr(self):
        return (f'c1={self.c1}, block_size={self.attn.block_size}, '
                f'topk={self.attn.topk}, '
                f'window_resolution={self.local_attn.window_resolution}')

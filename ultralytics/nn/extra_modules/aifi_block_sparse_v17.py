# AIFI_BlockSparseV17: serial local->global fusion of BlockSparseV1 with
# DCI-DETR "AIFI-C" (EfficientViT CascadedGroupAttention inside local windows).
#
# Fixes relative to V16:
#   P0 (deadlock fix): V16 double-gated the local branch with a zero-initialized
#       gamma AND CGA's zero-init proj BN. With local_out == 0 at init, dL/dgamma
#       == 0, and gamma == 0 blocks all gradients into CGA, so the branch could
#       never start training. V17 removes the gate entirely: CGA's native
#       zero-init BN alone makes the local refinement output 0 at step 0
#       (training starts exactly as BlockSparseV1) while the BN weight still
#       receives gradient and the branch can learn.
#   P1 (serial local->global): the local cascaded-group attention first refines
#       the feature map (residual), then BlockSparseV1 global attention runs on
#       the refined tokens. Refined inputs also improve V1's mean-pooled block
#       routing descriptors, which depend directly on input feature quality.
#
# Reference:
#   Block-Sparse Global Attention for Efficient Multi-View Geometry Transformers,
#   CVPR 2026. https://github.com/brianwang00001/sparse-vggt
#   EfficientViT: Memory Efficient Attention Design with Cascaded Group Attention,
#   CVPR 2023 (as used by DCI-DETR's "AIFI-C").

import math

import torch
import torch.nn as nn

from .aifi_block_sparse import BlockSparseAttention
from .aifi_c import LocalWindowAttention

__all__ = ['AIFI_BlockSparseV17']


class AIFI_BlockSparseV17(nn.Module):
    """BlockSparseV1 with a serial AIFI-C local refinement stage.

    Forward (post-norm):
        refined = x + LocalWindowAttention(x)          # AIFI-C, zero-init BN
        tokens  = flatten(refined) + sincos_pos
        tokens  = tokens + BlockSparseAttention(tokens) # global top-k routing
        tokens  = LayerNorm -> FFN -> LayerNorm

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

        # Stage 1 (local): DCI-DETR AIFI-C cascaded group attention.
        self.local_attn = LocalWindowAttention(
            dim=c1, num_heads=num_heads_local,
            window_resolution=window_resolution)
        self.norm_local = nn.LayerNorm(c1)  # used in pre-norm mode only

        # Stage 2 (global): identical to AIFI_BlockSparse (V1).
        self.attn = BlockSparseAttention(
            dim=c1, num_heads=num_heads, block_size=block_size,
            topk=topk, dropout=dropout)

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
                f'AIFI_BlockSparseV17 expected {self.c1} channels, got {c}')

        # Stage 1: local cascaded-group-attention refinement (AIFI-C).
        if self.normalize_before:
            refined = x + self.local_attn(self.norm_local(x))
        else:
            refined = x + self.local_attn(x)

        # Stage 2: global block-sparse attention (V1) on refined tokens.
        pos = self.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        src = refined.flatten(2).permute(0, 2, 1) + pos

        if self.normalize_before:
            src = src + self.dropout1(self.attn(self.norm1(src)))
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(self.norm2(src))))))
        else:
            src = src + self.dropout1(self.attn(src))
            src = self.norm1(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src)))))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()

    def extra_repr(self):
        return (f'c1={self.c1}, block_size={self.attn.block_size}, '
                f'topk={self.attn.topk}, '
                f'window_resolution={self.local_attn.window_resolution}')

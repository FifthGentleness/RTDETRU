# AIFI_ASConvFFN: Anisotropic Strip-Complementary Convolutional FFN for Block-Sparse AIFI.
#
# 方案一（主位置5 FFN + 配套位置4 零初始化门控）
# 灵感来源: Poly Kernel Inception Network for Remote Sensing Detection, CVPR 2024.
#   - 位置5: FFN 隐层 GELU 之后、fc2 之前插入各向异性可分离深度卷积
#     (3x1 垂直互补 + 1x3 水平条带内), 感受野严格 <= 3
#   - 位置4: 两条卷积支路的逐通道零初始化门控 g_v / g_h
#   - 位置1: BlockSparse 主干原样叠加, 注意力结构不改动
#
# 设计依据（本仓库 60+ 实验复盘）:
#   BlockSparse V1 的块划分为栅格序 1x4 水平条带, 块内注意力只覆盖同行 4 连
#   token, 同列垂直相邻 token 结构上不在同一块内, 垂直耦合只能依赖 top-k 内容
#   路由命中。本方案以 3x1 深度卷积补垂直、1x3 补条带内, 注意力管"内容相关的
#   稀疏路由", 卷积管"确定性的空间相邻", 两条信息流零重叠。
#   不引入任何全局信息回注（RABSA/TGHBSA/DGMB 的失败根源）; 参数增量约 8.2K;
#   g_v/g_h 零初始化 => step-0 前向与 AIFI_BlockSparse 逐位相等。

import torch
import torch.nn as nn

from .aifi_block_sparse import AIFI_BlockSparse, BlockSparseAttention

__all__ = ['AIFI_ASConvFFN']


class ASConvFFN(nn.Module):
    """各向异性条带互补卷积FFN: fc1 -> GELU -> (+g_v*DW3x1, +g_h*DW1x3) -> fc2。

    Args:
        dim: Token channel dim.
        cm: Hidden dim of the FFN.
        dropout: Dropout applied to the hidden representation before fc2.
        act: Activation applied after fc1.
    """

    def __init__(self, dim, cm=1024, dropout=0.0, act=nn.GELU()):
        super().__init__()
        self.fc1 = nn.Linear(dim, cm)
        self.fc2 = nn.Linear(cm, dim)
        # 3x1: 跨行垂直耦合 (BlockSparse 1x4 条带块结构上缺失的方向)
        self.dw_v = nn.Conv2d(cm, cm, (3, 1), padding=(1, 0), groups=cm)
        # 1x3: 条带内水平耦合 (与 1x4 块对齐的确定性直连)
        self.dw_h = nn.Conv2d(cm, cm, (1, 3), padding=(0, 1), groups=cm)
        # 位置4: 逐通道零初始化门控, step-0 等价于原 FFN
        self.g_v = nn.Parameter(torch.zeros(cm))
        self.g_h = nn.Parameter(torch.zeros(cm))
        self.dropout = nn.Dropout(dropout)
        self.act = act

    def forward(self, x, h, w):
        b = x.shape[0]
        cm = self.fc1.out_features
        hid = self.act(self.fc1(x)).transpose(1, 2).reshape(b, cm, h, w)
        hid = hid + self.g_v.view(1, -1, 1, 1) * self.dw_v(hid) \
                 + self.g_h.view(1, -1, 1, 1) * self.dw_h(hid)
        hid = hid.reshape(b, cm, h * w).transpose(1, 2)
        return self.fc2(self.dropout(hid))


class AIFI_ASConvFFN(nn.Module):
    """AIFI with block-sparse attention + anisotropic strip-complementary ConvFFN.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads of the block-sparse trunk.
        block_size: Number of tokens per sparse block.
        topk: Number of key blocks attended by each query block.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=1024, num_heads=8, block_size=4, topk=4,
                 dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before

        self.attn = BlockSparseAttention(
            dim=c1, num_heads=num_heads, block_size=block_size, topk=topk,
            dropout=dropout)

        self.ffn = ASConvFFN(dim=c1, cm=cm, dropout=dropout, act=act)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.c1:
            raise ValueError(f'AIFI_ASConvFFN expected {self.c1} channels, got {c}')

        pos = AIFI_BlockSparse.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1) + pos

        # --- BlockSparse 主干（位置1, 不改动）---
        if self.normalize_before:
            src = src + self.dropout1(self.attn(self.norm1(src)))
        else:
            src = src + self.dropout1(self.attn(src))

        # --- 各向异性卷积FFN（位置5 卷积注入, 位置4 零初始化门控）---
        if self.normalize_before:
            src = src + self.dropout2(self.ffn(self.norm2(src), h, w))
        else:
            src = self.norm1(src)
            src = src + self.dropout2(self.ffn(src, h, w))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()

    def extra_repr(self):
        g_v = self.ffn.g_v.detach().abs().mean().item()
        g_h = self.ffn.g_h.detach().abs().mean().item()
        return (f'c1={self.c1}, cm={self.ffn.fc1.out_features}, '
                f'|g_v|_mean={g_v:.4f}, |g_h|_mean={g_h:.4f}')

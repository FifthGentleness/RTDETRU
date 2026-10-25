# AIFI_CPAFFN: Channel-wise Piecewise-Adaptive Activation FFN for Block-Sparse AIFI.
#
# 方案三（主位置5 FFN + 配套位置4 零初始化门控）
# 灵感来源: KAN or MLP: A Fairer Comparison, ICLR 2025.
#           （理论根基: KAN: Kolmogorov-Arnold Networks, Liu et al., MIT, 2024）
#   - 位置5: FFN 隐层激活替换为 act(h) + gamma ⊙ PWL(h)。PWL 为逐通道可学习
#     分段线性函数: 固定网格上的软三角基 phi(u)=relu(1-|u|) 与每通道系数
#     theta∈R^{cm×K} 的加权求和 —— 全静态稠密算子（relu/sub/mul/add）,
#     TensorRT 零插件部署, 无 gather/floor 等动态索引
#   - 位置4: gamma 零初始化门控
#   - 位置1: BlockSparse 主干原样叠加, 注意力结构不改动
#
# 设计依据（本仓库 60+ 实验复盘）:
#   已有 AIFI_CBKFFN（同灵感）采用低秩投影+边界桥接实现, 参数增量约 29.5 万,
#   落在 RABSA(+20万)/TGHBSA(+38万) 的过拟合危险区。本方案为容量校准的极简版:
#   不投影、不桥接, 直接在完整隐维度上学习逐通道激活曲线, 参数增量约 1 万;
#   gamma 零初始化 => step-0 前向与 AIFI_BlockSparse 逐位相等。

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aifi_block_sparse import AIFI_BlockSparse, BlockSparseAttention

__all__ = ['AIFI_CPAFFN']


class CPAAct(nn.Module):
    """逐通道分段自适应激活: base_act(h) + gamma ⊙ PWL(h)。

    PWL(h)_{:,c} = sum_k theta_{c,k} * relu(1 - |h_{:,c} - grid_k|)
    固定等距网格 grid ∈ [-span, span] (K 个节点), 仅节点系数可学习。

    Args:
        cm: Hidden dim (per-channel activation).
        n_knots: Number of grid knots K.
        span: Grid half-range (h∈[-span, span] 之外基函数为 0, 回落到 base_act)。
        base_act: 底座激活 (保持原 FFN 的 GELU)。
    """

    def __init__(self, cm, n_knots=9, span=4.0, base_act=nn.GELU()):
        super().__init__()
        self.cm = cm
        self.register_buffer('grid', torch.linspace(-span, span, n_knots))
        self.theta = nn.Parameter(torch.randn(cm, n_knots) * 0.01)
        self.gamma = nn.Parameter(torch.zeros(cm))
        self.base_act = base_act

    def forward(self, h):                          # h: [B, N, cm]
        basis = F.relu(1.0 - (h.unsqueeze(-1) - self.grid).abs())  # [B,N,cm,K]
        pw = torch.einsum('bnck,ck->bnc', basis, self.theta)
        return self.base_act(h) + self.gamma * pw


class CPAFFN(nn.Module):
    """分段自适应激活FFN: fc1 -> CPAAct -> fc2。

    Args:
        dim: Token channel dim.
        cm: Hidden dim of the FFN.
        n_knots: Number of PWL grid knots.
        span: PWL grid half-range.
        dropout: Dropout applied to the hidden representation before fc2.
        act: Base activation (GELU 底座)。
    """

    def __init__(self, dim, cm=1024, n_knots=9, span=4.0,
                 dropout=0.0, act=nn.GELU()):
        super().__init__()
        self.fc1 = nn.Linear(dim, cm)
        self.fc2 = nn.Linear(cm, dim)
        self.cpa = CPAAct(cm, n_knots=n_knots, span=span, base_act=act)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.cpa(self.fc1(x))))


class AIFI_CPAFFN(nn.Module):
    """AIFI with block-sparse attention + channel-wise piecewise-adaptive FFN.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads of the block-sparse trunk.
        block_size: Number of tokens per sparse block.
        topk: Number of key blocks attended by each query block.
        n_knots: Number of PWL grid knots of the adaptive activation.
        span: PWL grid half-range.
        dropout: Dropout probability.
        act: Base activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=1024, num_heads=8, block_size=4, topk=4,
                 n_knots=9, span=4.0, dropout=0.0, act=nn.GELU(),
                 normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before

        self.attn = BlockSparseAttention(
            dim=c1, num_heads=num_heads, block_size=block_size, topk=topk,
            dropout=dropout)

        self.ffn = CPAFFN(dim=c1, cm=cm, n_knots=n_knots, span=span,
                          dropout=dropout, act=act)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.c1:
            raise ValueError(f'AIFI_CPAFFN expected {self.c1} channels, got {c}')

        pos = AIFI_BlockSparse.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1) + pos

        # --- BlockSparse 主干（位置1, 不改动）---
        if self.normalize_before:
            src = src + self.dropout1(self.attn(self.norm1(src)))
        else:
            src = src + self.dropout1(self.attn(src))

        # --- 分段自适应激活FFN（位置5 激活函数替换, 位置4 零初始化门控）---
        if self.normalize_before:
            src = src + self.dropout2(self.ffn(self.norm2(src)))
        else:
            src = self.norm1(src)
            src = src + self.dropout2(self.ffn(src))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()

    def extra_repr(self):
        g = self.ffn.cpa.gamma.detach().abs().mean().item()
        return (f'c1={self.c1}, cm={self.ffn.fc1.out_features}, '
                f'n_knots={self.ffn.cpa.grid.numel()}, '
                f'span={self.ffn.cpa.grid.max().item():.1f}, '
                f'|gamma|_mean={g:.4f}')

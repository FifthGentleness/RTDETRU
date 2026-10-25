# AIFI_TWCBS: Twiced-Correction Output Fusion for Block-Sparse AIFI.
#
# 方案二（主位置4 输出融合 + 配套位置6 残差拓扑）
# 灵感来源: Transformer Meets Twicing, ICLR 2025（over-smoothing 表示塌缩的
# 预测-校正治理）。
#   - 位置4: 注意力级展开两程——第一程把 top-4 块上下文混入(预测), 第二程对
#     已混合的更新特征再跑一遍同构 BlockSparse(校正), 校正项乘零初始化门
#     gamma => step-0 与 AIFI_BlockSparse 逐位相等
#   - 位置6(配套): 第二程残差支路的宏观连接
#   - 位置1: BlockSparseAttention 模块原样(默认共享版两程共用同一权重实例,
#     增量仅 +256; shared=False 可启用独立权重第二程, +0.26M 作容量上界消融)
#
# 设计依据(仓库教训): 过平滑修复此前全在"加东西"轴上(FFN路径/卷积/可变形);
# 本方案换到输出融合轴——校正程作用于已平滑特征, 恰好在残差流里恢复被平均的
# 弱目标响应。GFLOPs +0.16G(+0.2%)。

import torch
import torch.nn as nn

from .aifi_block_sparse import AIFI_BlockSparse, BlockSparseAttention

__all__ = ['AIFI_TWCBS']


class AIFI_TWCBS(nn.Module):
    """AIFI with block-sparse attention + twiced-correction output fusion.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads of the block-sparse trunk.
        block_size: Number of tokens per sparse block.
        topk: Number of key blocks attended by each query block.
        shared: True=两程共用同一 BlockSparseAttention 实例(权重共享, 推荐);
                False=第二程独立权重(+0.26M, 容量上界消融用, 非 step-0 等价).
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=1024, num_heads=8, block_size=4, topk=4,
                 shared=True, dropout=0.0, act=nn.GELU(),
                 normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = BlockSparseAttention(
            dim=c1, num_heads=num_heads, block_size=block_size, topk=topk,
            dropout=dropout)
        if shared:
            self.attn2 = self.attn                              # 权重共享: +0参数
        else:
            self.attn2 = BlockSparseAttention(                   # 独立权重: +0.26M
                dim=c1, num_heads=num_heads, block_size=block_size,
                topk=topk, dropout=dropout)
        self.gamma = nn.Parameter(torch.zeros(c1))               # 位置4 零门控
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.c1:
            raise ValueError(f'AIFI_TWCBS expected {self.c1} channels, got {c}')
        pos = AIFI_BlockSparse.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1) + pos

        # --- 第一程: 预测(BlockSparse 原样混入 top-4 块上下文) ---
        t1 = src + self.dropout1(self.attn(src))
        # --- 第二程: 校正(对已混合特征, 零初始化门起步) ---
        t2 = t1 + self.gamma * self.dropout1(self.attn2(t1))

        # --- FFN（与 AIFI_BlockSparse 一致）---
        src = t2
        if self.normalize_before:
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(self.norm2(src))))))
        else:
            src = self.norm1(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src)))))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()

    def extra_repr(self):
        g = self.gamma.detach().abs().mean().item()
        shared = self.attn2 is self.attn
        return (f'c1={self.c1}, cm={self.fc1.out_features}, '
                f'shared={shared}, |gamma|_mean={g:.4f}')

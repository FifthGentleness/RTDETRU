# AIFI_SAGFFN: Saliency-Adaptive Gated FFN for Block-Sparse AIFI.
#
# 方案二（纯主位置5 FFN, 无配套位置）
# 灵感来源: Mixture-of-Depths: Upscaling Transformers with Conditional
#           Computation, ICLR 2025 (Google DeepMind).
#   - 位置5: FFN 隐层上的逐 token 显著性增益 g_i = 1 + tanh(w_s^T h_i) ∈ (0,2),
#     背景 token 的 FFN 语义更新被压缩、疑似目标 token 被放大 —— 计算预算按
#     显著性在 token 维重分配 (MoD 的软门控迁移, 无路由、无丢token、零分支)
#   - 位置1: BlockSparse 主干原样叠加, 注意力结构不改动
#
# 设计依据（本仓库 60+ 实验复盘）:
#   VisDrone S5 上 70%~90% token 为空旷背景, 却与弱小目标共享同一份 FFN 变换,
#   均匀语义化是背景虚警的来源之一。本方案以 1024 个参数实现推理零开销的软性
#   前景掩码; 无全局池化(区别于SE)、无隐层分裂门控(区别于SwiGLU/StarFFN);
#   w_s 零初始化 => g≡1, step-0 前向与 AIFI_BlockSparse 逐位相等。

import torch
import torch.nn as nn

from .aifi_block_sparse import AIFI_BlockSparse, BlockSparseAttention

__all__ = ['AIFI_SAGFFN']


class SAGFFN(nn.Module):
    """显著性自适应FFN: fc1 -> GELU -> 逐token增益 g∈(0,2) -> fc2。

    Args:
        dim: Token channel dim.
        cm: Hidden dim of the FFN.
        dropout: Dropout applied to the gated hidden representation.
        act: Activation applied after fc1.
    """

    def __init__(self, dim, cm=1024, dropout=0.0, act=nn.GELU()):
        super().__init__()
        self.fc1 = nn.Linear(dim, cm)
        self.fc2 = nn.Linear(cm, dim)
        # 逐 token 显著性打分 (零初始化 => g ≡ 1 起步)
        self.w_s = nn.Linear(cm, 1, bias=False)
        nn.init.zeros_(self.w_s.weight)
        self.dropout = nn.Dropout(dropout)
        self.act = act
        # 推理期诊断统计 (不进 state_dict, 训练路径零开销)
        self._gate_mean = 1.0
        self._gate_std = 0.0

    def forward(self, x):
        h = self.act(self.fc1(x))
        g = 1.0 + torch.tanh(self.w_s(h))          # [B, N, 1] ∈ (0, 2)
        if not self.training:
            self._gate_mean = g.detach().mean().item()
            self._gate_std = g.detach().std().item()
        return self.fc2(self.dropout(g * h))

    @property
    def gate_stats(self):
        """返回最近一次 eval 前向的增益分布 (mean, std), 用于监控门控是否分化。"""
        return self._gate_mean, self._gate_std


class AIFI_SAGFFN(nn.Module):
    """AIFI with block-sparse attention + saliency-adaptive gated FFN.

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

        self.ffn = SAGFFN(dim=c1, cm=cm, dropout=dropout, act=act)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.c1:
            raise ValueError(f'AIFI_SAGFFN expected {self.c1} channels, got {c}')

        pos = AIFI_BlockSparse.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1) + pos

        # --- BlockSparse 主干（位置1, 不改动）---
        if self.normalize_before:
            src = src + self.dropout1(self.attn(self.norm1(src)))
        else:
            src = src + self.dropout1(self.attn(src))

        # --- 显著性自适应FFN（位置5 逐token条件计算）---
        if self.normalize_before:
            src = src + self.dropout2(self.ffn(self.norm2(src)))
        else:
            src = self.norm1(src)
            src = src + self.dropout2(self.ffn(src))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()

    def extra_repr(self):
        m, s = self.ffn.gate_stats
        return (f'c1={self.c1}, cm={self.ffn.fc1.out_features}, '
                f'gate_mean={m:.4f}, gate_std={s:.4f}')

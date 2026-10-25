# AIFI_SigAttn: Sigmoid Self-Attention for AIFI (softmax 替换).
#
# 灵感来源: Theory, Analysis, and Best Practices for Sigmoid Self-Attention,
#           ICLR 2025 (Ramapuram et al., Apple; arXiv 2409.04431)。
#   - 论文机制: A = sigmoid(QK^T/√d) —— 无行归一化的逐元素注意力权重。
#     无凸组合 => 不存在"必须把概率质量分给全图400 token"的过平滑根源;
#     全并行(无 rowmax 依赖)。论文理论+实践(ViT/ASR/LLM)给出两条关键
#     最佳实践: (1) 对 Q 做序列均值中心化; (2) Q 投影用受限谱范数初始化。
#   - 本模块: 按论文最佳实践实现 —— forward 中对 Q 逐头均值中心化(数据
#     依赖, 部署安全); Q 投影权重以 0.5 增益初始化; 逐头可学习 logit 温度
#     tau (init 1)。逐头 tau 允许"部分头尖锐/部分头平缓"的分工。
#   - 替换式模块(BlockSparse 先例: 替换无 step-0 等价, 靠机制动机与稳定
#     初始化兜底); 参数增量 = +num_heads; exp/sigmoid 均为 TRT 原生算子。

import torch
import torch.nn as nn

from .aifi_block_sparse import AIFI_BlockSparse

__all__ = ['AIFI_SigAttn']


class SigmoidAttention(nn.Module):
    """Sigmoid 自注意力(ICLR 2025 最佳实践版): 均值中心化Q + 受限Q初始化。"""

    def __init__(self, dim, num_heads=8, dropout=0.0, q_init_gain=0.5):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim ({dim}) must be divisible by num_heads')
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        # 论文最佳实践: Q 投影受限谱范数初始化(近似: 全体权重乘 0.5 增益)
        with torch.no_grad():
            self.q_proj.weight.mul_(q_init_gain)
        # 逐头可学习温度(分工: 尖锐头/平缓头)
        self.tau = nn.Parameter(torch.ones(num_heads))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                # x: [B, N, C]
        b, n, c = x.shape
        h, d = self.num_heads, self.head_dim
        q = self.q_proj(x).view(b, n, h, d).transpose(1, 2)   # [B,h,N,d]
        k = self.k_proj(x).view(b, n, h, d).transpose(1, 2)
        v = self.v_proj(x).view(b, n, h, d).transpose(1, 2)
        # 最佳实践1: Q 序列均值中心化
        q = q - q.mean(dim=2, keepdim=True)
        logits = (q @ k.transpose(-1, -2)) * (d ** -0.5)      # [B,h,N,N]
        logits = logits * self.tau.view(1, -1, 1, 1)
        attn = torch.sigmoid(logits)                          # 无行归一化
        attn = self.drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.out_proj(out)


class AIFI_SigAttn(nn.Module):
    """AIFI with sigmoid self-attention (softmax 替换, ICLR 2025)。

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=1024, num_heads=8, dropout=0.0,
                 act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = SigmoidAttention(c1, num_heads, dropout)
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
            raise ValueError(f'AIFI_SigAttn expected {self.c1} channels, got {c}')
        pos = AIFI_BlockSparse.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        tokens = x.flatten(2).permute(0, 2, 1)
        qk = tokens + pos                                 # pos只进q/k(基线约定)

        if self.normalize_before:
            q = self.norm1(tokens) + pos
            src = tokens + self.dropout1(self.attn(q))
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(self.norm2(src))))))
        else:
            src = tokens + self.dropout1(self.attn(qk))
            src = self.norm1(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src)))))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()

    def extra_repr(self):
        tau = self.attn.tau.detach()
        return (f'c1={self.c1}, cm={self.fc1.out_features}, '
                f'tau=[{tau.min():.2f},{tau.max():.2f}]')

# AIFI_LASER: Log-space Value Aggregation for AIFI (standard dense attention).
#
# 灵感来源: LASER: Attention with Exponential Transformation, ICML 2025
#           (Duvvuri & Dhillon, UT Austin/Google; ViT ImageNet 增益已验证)。
#   - 论文机制(Eq.6/9): O = log(softmax(QK^T) exp(V)), 数值稳定式为
#     对 V 逐通道 max 平移后聚合再取 log —— 注意力权重不变, 值聚合从
#     算术平均(凸组合, 过平滑根源)搬到对数域(几何平均型), 且经 softmax 的
#     梯度信号更大。全静态算子(exp/log 为 TRT 原生)。
#   - 本模块适配: 标准 MHA 原样计算 O_std(复用 nn.MultiheadAttention,
#     权重名与基线 AIFI 完全一致); 对数空间路径经同一组投影权重手动计算,
#     输出 = O_std + gamma ⊙ (O_laser - O_std_manual), gamma 逐通道零初始化
#     => step-0 与基线 AIFI 逐位相等(gamma=0 时校正项精确为0)。
#   - 教训对齐: 不加通路/不改路由/无分支; 参数增量仅 +256;
#     与 SwiGLU/SE/频域/大核零形态重叠。

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aifi_block_sparse import AIFI_BlockSparse

__all__ = ['AIFI_LASER']


class LaserBlendedAttention(nn.Module):
    """标准MHA + 零初始化对数空间值聚合校正(共享同一组投影权重)。"""

    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.gamma = nn.Parameter(torch.zeros(dim))   # 单守门零初始化

    def _manual_attention(self, qk_in, v_in):
        """用 self.mha 的投影权重手动计算 std/laser 两条值聚合路径。

        qk_in: [B, N, C] 含pos的q/k输入; v_in: [B, N, C] 不含pos的value
        (与基线 forward_post 的 ma(q,k,value=src) 语义一致)。
        """
        b, n, c = qk_in.shape
        h, d = self.num_heads, self.head_dim
        w = self.mha.in_proj_weight
        bias = self.mha.in_proj_bias
        q = qk_in @ w[:c].t() + (bias[:c] if bias is not None else 0)
        k = qk_in @ w[c:2 * c].t() + (bias[c:2 * c] if bias is not None else 0)
        v = v_in @ w[2 * c:].t() + (bias[2 * c:] if bias is not None else 0)
        q = q.view(b, n, h, d).transpose(1, 2)          # [B,h,N,d]
        k = k.view(b, n, h, d).transpose(1, 2)
        v = v.view(b, n, h, d).transpose(1, 2)
        attn = (q @ k.transpose(-1, -2)) * (d ** -0.5)
        attn = attn.softmax(dim=-1)                     # [B,h,N,N]

        o_std = attn @ v                                # 算术平均路径
        # LASER 路径(Eq.9 数值稳定式): 逐通道 max 平移 -> exp -> 聚合 -> log
        m = v.max(dim=2, keepdim=True).values.detach()  # [B,h,1,d]
        laser = torch.log((attn @ torch.exp(v - m)).clamp(min=1e-12)) + m
        return o_std, laser

    def forward(self, q_in, v_in):
        # 标准路径走 MHA 本体(与基线逐位一致); k=q_in
        o_std_mha, _ = self.mha(q_in.transpose(0, 1), q_in.transpose(0, 1),
                                v_in.transpose(0, 1), need_weights=False)
        o_std_mha = o_std_mha.transpose(0, 1)           # [B,N,C]
        # 校正项: gamma ⊙ (O_laser - O_std_manual)
        with torch.cuda.amp.autocast(enabled=False):
            o_std_m, o_laser = self._manual_attention(q_in.float(), v_in.float())
            delta = (o_laser - o_std_m).transpose(1, 2).reshape(
                q_in.shape[0], q_in.shape[1], -1)
            out = o_std_mha + self.gamma * delta.to(o_std_mha.dtype)
        return out


class AIFI_LASER(nn.Module):
    """AIFI (standard dense attention) + LASER 对数空间值聚合校正。

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
        self.laser_attn = LaserBlendedAttention(c1, num_heads, dropout)
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
            raise ValueError(f'AIFI_LASER expected {self.c1} channels, got {c}')
        pos = AIFI_BlockSparse.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        tokens = x.flatten(2).permute(0, 2, 1)          # 无pos: 基线value语义

        qk = tokens + pos                                # pos只进q/k(基线约定)
        if self.normalize_before:
            q = self.norm1(tokens) + pos
            src = tokens + self.dropout1(self.laser_attn(q, tokens))
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(self.norm2(src))))))
        else:
            src = tokens + self.dropout1(self.laser_attn(qk, tokens))
            src = self.norm1(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src)))))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()

    def extra_repr(self):
        g = self.laser_attn.gamma.detach().abs().mean().item()
        return (f'c1={self.c1}, cm={self.fc1.out_features}, '
                f'|gamma|_mean={g:.4f}')

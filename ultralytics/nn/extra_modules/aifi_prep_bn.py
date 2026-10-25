# AIFI_PRepBN: Progressive Re-parameterized BatchNorm for AIFI.
#
# 灵感来源: SLAB: Efficient Transformers with Simplified Linear Attention and
#           Progressive Re-parameterized BatchNorm, ICML 2024 (Guo et al.,
#           Huawei Noah's Ark Lab; PMLR v235)。
#   - 论文机制(PRepBN): 以 BatchNorm 渐进替换 LayerNorm —— BN 训练更
#     并行友好且推理期可重参数化折叠; "渐进"过渡解决 BN 直接替换的
#     可训练性缺口。论文在检测任务上验证过该组件。
#   - 本模块适配: y = LN(x) + gamma ⊙ (BN(x) - LN(x)), gamma 逐通道零初始化
#     => step-0 与基线 AIFI 逐位相等, BN 支路随训练渐进接管(渐进重参数化
#     精神的零初始化实现)。BN 推理用 running stats(部署友好, 可折叠)。
#   - 作用于 AIFI 的两个 norm(注意力后/FFN后); 注意力/FFN 本体不动。
#   - 教训对齐: 参数增量 ~+2.6K; batch=4 的 BN 统计噪声是主要风险(见消融)。

import torch
import torch.nn as nn

from .aifi_block_sparse import AIFI_BlockSparse

__all__ = ['AIFI_PRepBN']


class ProgressiveBN(nn.Module):
    """y = LN(x) + gamma ⊙ (BN(x) - LN(x)); gamma 零初始化 => 恒等起步。

    BN 在 (B, N) 维上按通道统计(token 流形状 [B, N, C] -> 转 [B*N, C])。
    """

    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.bn = nn.BatchNorm1d(dim)                  # affine 默认(1, 0)
        self.gamma = nn.Parameter(torch.zeros(dim))    # 单守门零初始化

    def forward(self, x):                              # x: [B, N, C]
        b, n, c = x.shape
        y_ln = self.ln(x)
        y_bn = self.bn(x.reshape(b * n, c)).reshape(b, n, c)
        return y_ln + self.gamma * (y_bn - y_ln)


class AIFI_PRepBN(nn.Module):
    """AIFI with progressive re-parameterized BatchNorm (SLAB, ICML 2024)。

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
        self.attn = nn.MultiheadAttention(c1, num_heads, dropout=dropout)
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.norm1 = ProgressiveBN(c1)
        self.norm2 = ProgressiveBN(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.c1:
            raise ValueError(f'AIFI_PRepBN expected {self.c1} channels, got {c}')
        pos = AIFI_BlockSparse.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        tokens = x.flatten(2).permute(0, 2, 1)          # 无pos: 基线value语义

        qk = tokens + pos                                # pos只进q/k(基线约定)
        if self.normalize_before:
            q = self.norm1(tokens) + pos
            attn_out, _ = self.attn(q.transpose(0, 1), q.transpose(0, 1),
                                    tokens.transpose(0, 1), need_weights=False)
            src = tokens + self.dropout1(attn_out.transpose(0, 1))
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(self.norm2(src))))))
        else:
            attn_out, _ = self.attn(qk.transpose(0, 1), qk.transpose(0, 1),
                                    tokens.transpose(0, 1), need_weights=False)
            src = tokens + self.dropout1(attn_out.transpose(0, 1))
            src = self.norm1(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src)))))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()

    def extra_repr(self):
        g1 = self.norm1.gamma.detach().abs().mean().item()
        g2 = self.norm2.gamma.detach().abs().mean().item()
        return (f'c1={self.c1}, cm={self.fc1.out_features}, '
                f'|gamma1|={g1:.4f}, |gamma2|={g2:.4f}')

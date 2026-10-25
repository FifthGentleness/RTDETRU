# AIFI_DTSBS: Dynamic Token-Selection Value Gating for Block-Sparse AIFI.
#
# 方案三（主位置2 Token集合-KV参与度 + 配套位置4 零初始化门控）
# 灵感来源: DHVT: Dynamic Hybrid Vision Transformer, TPAMI 2025
#           （动态 token 选择治理 over-smoothing / attention 分散）。
#   - 位置2: 被选中的 top-4 块内部, 逐 key token 由自身内容预测参与度标量
#     s = 1 + tanh(net(x))*gamma ∈ (0,2), softmax 之后对 V 加权(值侧);
#     掩码/路由/打分零触碰(BlockSparse 路由结构与 V1 逐行一致)
#   - 位置4: gamma 零初始化 => s恒1 => step-0 与 AIFI_BlockSparse 逐位相等
#   - 位置1: 本方案是"更少混合"(制胜规律从块间推进到块内), 与已证伪的
#     全局桥家族(加通路)方向相反; 与SE(全局池化通道注意力,创1已用)不同:
#     逐token、纯局部、无池化
#
# 参数增量 ~4.1K; 0 FLOPs 量级; 全静态稠密算子。

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aifi_block_sparse import AIFI_BlockSparse, BlockSparseAttention

__all__ = ['AIFI_DTSBS']


class ValueGate(nn.Module):
    """逐 token 参与度标量: s = 1 + tanh(lowrank(x)) * gamma ∈ (0,2)。"""

    def __init__(self, dim, d=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, d), nn.GELU(), nn.Linear(d, 1))
        self.gamma = nn.Parameter(torch.zeros(1))          # 位置4: s恒1起步

    def forward(self, x):                                  # x: [B, N, dim]
        return 1.0 + torch.tanh(self.net(x)) * self.gamma  # [B, N, 1]


class GatedValueBlockSparseAttention(BlockSparseAttention):
    """BlockSparse V1 的逐行复刻, 仅在 softmax 之后对 V 乘逐 token 参与度。

    路由(mean池化top-k)/打分(qk·scale)/掩码(-1e4)/softmax 与父类完全一致;
    唯一改动: out = attn @ (v * s)。s=1 时与父类逐位相等。
    """

    def __init__(self, dim, num_heads=8, block_size=4, topk=4,
                 gate_dim=16, dropout=0.0):
        super().__init__(dim=dim, num_heads=num_heads, block_size=block_size,
                         topk=topk, dropout=dropout)
        self.vgate = ValueGate(dim, d=gate_dim)

    def forward(self, x):
        b, n, c = x.shape
        h, d = self.num_heads, self.head_dim
        bs = self.block_size
        num_blocks = math.ceil(n / bs)
        pad_len = num_blocks * bs - n

        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))
        npad = num_blocks * bs

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, npad, h, d).permute(0, 2, 1, 3) for t in qkv]

        # ---- 逐 token 参与度(值侧, 位置2): softmax 之前算好, V 加权用 ----
        s = self.vgate(x).squeeze(-1)                       # [B, npad]

        # ---- 以下与 BlockSparseAttention.forward 逐行一致 ----
        q_blk = x.view(b, num_blocks, bs, c).mean(dim=2)
        k_blk = x.view(b, num_blocks, bs, c).mean(dim=2)
        block_logits = torch.matmul(q_blk, k_blk.transpose(-2, -1)) * self.scale
        topk = min(self.topk, num_blocks)
        topk_idx = block_logits.topk(k=topk, dim=-1).indices

        block_mask = torch.zeros(
            b, num_blocks, num_blocks, dtype=torch.bool, device=x.device)
        block_mask.scatter_(-1, topk_idx, True)
        eye = torch.eye(num_blocks, dtype=torch.bool, device=x.device)
        block_mask = block_mask | eye.unsqueeze(0)

        token_mask = block_mask.repeat_interleave(bs, dim=1)
        token_mask = token_mask.repeat_interleave(bs, dim=2)
        token_mask = token_mask[:, :npad, :npad]
        if pad_len:
            valid = torch.zeros(b, npad, dtype=torch.bool, device=x.device)
            valid[:, :n] = True
            token_mask = token_mask & valid[:, None, :]

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        logits = logits.masked_fill(~token_mask.unsqueeze(1), -1e4)
        attn = self.attn_drop(logits.softmax(dim=-1))

        # ---- 唯一改动: V 乘参与度(广播到各头; 打分/掩码/softmax 未动) ----
        v = v * s[:, None, :, None]

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, npad, c)
        if pad_len:
            out = out[:, :n, :]
        return self.proj_drop(self.proj(out))


class AIFI_DTSBS(nn.Module):
    """AIFI with block-sparse attention + dynamic token-selection value gating.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads of the block-sparse trunk.
        block_size: Number of tokens per sparse block.
        topk: Number of key blocks attended by each query block.
        gate_dim: Low-rank width of the value participation gate.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=1024, num_heads=8, block_size=4, topk=4,
                 gate_dim=16, dropout=0.0, act=nn.GELU(),
                 normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = GatedValueBlockSparseAttention(
            dim=c1, num_heads=num_heads, block_size=block_size, topk=topk,
            gate_dim=gate_dim, dropout=dropout)
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
            raise ValueError(f'AIFI_DTSBS expected {self.c1} channels, got {c}')
        pos = AIFI_BlockSparse.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1) + pos

        # --- BlockSparse 主干 + 值侧参与度门（位置1原样 + 位置2/4叠加）---
        if self.normalize_before:
            src = src + self.dropout1(self.attn(self.norm1(src)))
        else:
            src = src + self.dropout1(self.attn(src))

        # --- FFN（与 AIFI_BlockSparse 一致）---
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
        g = self.attn.vgate.gamma.item()
        return (f'c1={self.c1}, cm={self.fc1.out_features}, '
                f'gate|gamma|={g:.4f}')

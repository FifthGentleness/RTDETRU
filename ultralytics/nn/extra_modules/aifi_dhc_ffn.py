# AIFI_DHCFFN: Block-aligned Deformable Hyper-Connected FFN for Block-Sparse AIFI.
#
# 方案三·完整组合最终形态（主位置5 + 配套位置4 零初始化门控 + 位置6 流混合）
# 双锚点: Hyper-Connections (ICLR 2025, 框架) x DCNv4 (CVPR 2024, 机制)。
#   - 位置5: 路径A=原MLP; 路径B=低秩可变形路径(256->64->DefDW3x3->256),
#     其中可变形偏移的预测粒度绑定 BlockSparse 的 1x4 条带块: 每块由池化
#     描述子预测一组共享偏移(9 taps), 块内 4 token 共用 —— FFN 的采样几何
#     与注意力的分块几何显式对齐; 块级预测把偏移自由度 400->100, 天然正则化
#   - 位置4: 超连接逐通道门 a2 零初始化 => step-0 与 AIFI_BlockSparse 逐位相等
#   - 位置1: BlockSparse 主干原样叠加, 注意力结构不改动
#
# 设计依据(仓库教训): 参数增量 ~41K; 偏移有界(rho*tanh)+头零初始化+门零初始化
# 三重保险; 偏移严格局部(±1.5 token), 非路由非全局桥; grid_sample 为
# TensorRT 8.5+ 原生算子。

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aifi_block_sparse import AIFI_BlockSparse, BlockSparseAttention

__all__ = ['AIFI_DHCFFN']

# 3x3 采样 tap 的 (di, dj) 偏移(单位: token)
_TAPS = ((-1, -1), (-1, 0), (-1, 1),
         (0, -1), (0, 0), (0, 1),
         (1, -1), (1, 0), (1, 1))


class BlockAlignedDeform(nn.Module):
    """块对齐可变形 3x3 深度聚合(作用于低秩隐层 d=64)。

    偏移/调制由 1x4 条带块的池化描述子预测, 广播至块内全部 token。
    """

    def __init__(self, d, c1, block_size=4, rho=1.5):
        super().__init__()
        self.block_size = int(block_size)
        self.rho = float(rho)
        self.off_head = nn.Linear(c1, 18)             # 9 taps x (di, dj)
        self.mod_head = nn.Linear(c1, 9)
        self.tap_w = nn.Parameter(torch.empty(d, 9))
        nn.init.zeros_(self.off_head.weight)
        nn.init.zeros_(self.off_head.bias)
        # mod 头非零初始化(bias=1), 避免与 a2=0 互锁梯度; 见 aifi_dh_ffn.py 注释
        nn.init.zeros_(self.mod_head.weight)
        nn.init.ones_(self.mod_head.bias)
        nn.init.kaiming_uniform_(self.tap_w, a=5 ** 0.5)
        self._meta_cache = {}

    def _block_meta(self, n, device):
        key = (n, str(device))
        if key not in self._meta_cache:
            bs = self.block_size
            block_id = torch.arange(n, device=device) // bs
            counts = torch.bincount(block_id, minlength=(n + bs - 1) // bs)
            counts = counts.clamp(min=1).float()
            self._meta_cache[key] = (block_id, counts)
        return self._meta_cache[key]

    def _block_pool(self, x):
        # x: [B, N, C] -> [B, nb, C] (1x4条带块均值)
        b, n, c = x.shape
        block_id, counts = self._block_meta(n, x.device)
        s = x.new_zeros(b, counts.numel(), c)
        s.index_add_(1, block_id, x)
        return s / counts.view(1, -1, 1)

    def forward(self, z2d, t, h, w):
        # z2d: [B, d, H, W] 低秩隐层;  t: [B, N, c1] BlockSparse输出token流
        b, n = t.shape[:2]
        mu = self._block_pool(t)                              # [B, nb, c1]
        off = self.rho * torch.tanh(self.off_head(mu))        # [B, nb, 18]
        mod = self.mod_head(mu)                               # [B, nb, 9]
        # 广播至块内 token (栅格序: 块j覆盖token[4j,4j+3])
        off = off.repeat_interleave(self.block_size, dim=1)[:, :n]   # [B,N,18]
        mod = mod.repeat_interleave(self.block_size, dim=1)[:, :n]   # [B,N,9]
        off = off.view(b, h, w, 18)
        mod = mod.view(b, h, w, 9).permute(0, 3, 1, 2)        # [B,9,H,W]

        ys = (2 * torch.arange(h, device=z2d.device, dtype=z2d.dtype) + 1) / h - 1
        xs = (2 * torch.arange(w, device=z2d.device, dtype=z2d.dtype) + 1) / w - 1
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        out = 0
        for k, (di, dj) in enumerate(_TAPS):
            oy = gy + (di + off[..., 2 * k]) * (2.0 / h)
            ox = gx + (dj + off[..., 2 * k + 1]) * (2.0 / w)
            grid = torch.stack([ox, oy], dim=-1)              # [B,H,W,2]
            s = F.grid_sample(z2d, grid, mode='bilinear',
                              padding_mode='zeros', align_corners=False)
            out = out + (mod[:, k:k + 1] * s) * self.tap_w[:, k].view(1, -1, 1, 1)
        return out


class DHCFFN(nn.Module):
    """块对齐可变形超连接FFN: FFN = MLP(x) + a2 ⊙ SiLU(W2B·DefAgg(W1B·x))。"""

    def __init__(self, dim, cm=1024, d=64, block_size=4, rho=1.5,
                 dropout=0.0, act=nn.GELU()):
        super().__init__()
        self.fc1 = nn.Linear(dim, cm)
        self.fc2 = nn.Linear(cm, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = act
        self.b1 = nn.Linear(dim, d)
        self.b2 = nn.Linear(d, dim)
        self.defagg = BlockAlignedDeform(d, dim, block_size=block_size, rho=rho)
        self.a2 = nn.Parameter(torch.zeros(dim))              # 位置4 零初始化门

    def forward(self, x, h, w):
        b = x.shape[0]
        a = self.fc2(self.dropout(self.act(self.fc1(x))))
        z = self.b1(x).transpose(1, 2).view(b, -1, h, w)      # [B,d,H,W]
        z = self.defagg(z, x, h, w)
        z = F.silu(z.reshape(b, -1, h * w).transpose(1, 2))
        bp = self.b2(z)
        return a + self.a2 * bp


class AIFI_DHCFFN(nn.Module):
    """AIFI with block-sparse attention + block-aligned deformable hyper-connected FFN.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of path A.
        num_heads: Number of attention heads of the block-sparse trunk.
        block_size: Number of tokens per sparse block (also deform offset granularity).
        topk: Number of key blocks attended by each query block.
        d: Low-rank width of the deformable path B.
        rho: Offset bound in token units.
        dropout: Dropout probability.
        act: Activation of path A.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=1024, num_heads=8, block_size=4, topk=4,
                 d=64, rho=1.5, dropout=0.0, act=nn.GELU(),
                 normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = BlockSparseAttention(
            dim=c1, num_heads=num_heads, block_size=block_size, topk=topk,
            dropout=dropout)
        self.ffn = DHCFFN(dim=c1, cm=cm, d=d, block_size=block_size,
                          rho=rho, dropout=dropout, act=act)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.c1:
            raise ValueError(f'AIFI_DHCFFN expected {self.c1} channels, got {c}')
        pos = AIFI_BlockSparse.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1) + pos

        # --- BlockSparse 主干（位置1, 不改动）---
        if self.normalize_before:
            src = src + self.dropout1(self.attn(self.norm1(src)))
        else:
            src = src + self.dropout1(self.attn(src))

        # --- 块对齐可变形超连接FFN（位置5, 位置4零门控, 位置6流混合）---
        if self.normalize_before:
            src = src + self.dropout2(self.ffn(self.norm2(src), h, w))
        else:
            src = self.norm1(src)
            src = src + self.dropout2(self.ffn(src, h, w))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()

    def extra_repr(self):
        a2 = self.ffn.a2.detach().abs().mean().item()
        return (f'c1={self.c1}, cm={self.ffn.fc1.out_features}, '
                f'd={self.ffn.b1.out_features}, rho={self.ffn.defagg.rho}, '
                f'|a2|_mean={a2:.4f}')

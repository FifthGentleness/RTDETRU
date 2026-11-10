"""Test polarity correction within the original BlockSparse support."""

from contextlib import nullcontext
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aifi_block_sparse import AIFI_BlockSparse, BlockSparseAttention

__all__ = ('AIFI_BlockSparsePola',)


class BlockSparsePolarityAttention(BlockSparseAttention):
    """Keep signed logits and add a bounded per-head cancellation correction."""

    def __init__(self, dim, num_heads=8, block_size=4, topk=4, dropout=0.0):
        super().__init__(dim, num_heads, block_size, topk, dropout)
        self.polarity_gain = nn.Parameter(torch.zeros(num_heads))

    def _token_logits(self, q, k):
        signed = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        dtype = torch.float32 if signed.dtype in (torch.float16, torch.bfloat16) else signed.dtype
        # Avoid reduced precision for the subtraction of nearly equal magnitudes.
        autocast_off = (torch.autocast(q.device.type, enabled=False)
                        if q.device.type in ('cpu', 'cuda') else nullcontext())
        with autocast_off:
            # For same/opposite-sign masses S and O: signed = S - O,
            # total = S + O, so total - |signed| = 2 min(S, O).
            total = torch.matmul(q.to(dtype).abs(), k.to(dtype).abs().transpose(-2, -1)) * self.scale
            cancelled = (total - signed.to(dtype).abs()).clamp_min(0)
            # The fixed floor of one scaled-logit unit damps weak matches.
            correction = cancelled / (total + 1.0)
            gain = self.polarity_gain.to(dtype).tanh().view(1, -1, 1, 1)

        # The hard mask is applied after scoring, before the single softmax.
        return signed + (gain * correction).to(signed.dtype)

    def forward(self, x):
        b, n, c = x.shape
        h, d = self.num_heads, self.head_dim
        bs = self.block_size
        num_blocks = math.ceil(n / bs)
        npad = num_blocks * bs
        pad_len = npad - n
        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, npad, h, d).permute(0, 2, 1, 3) for t in qkv]

        descriptors = x.reshape(b, num_blocks, bs, c).mean(dim=2)
        block_logits = torch.matmul(descriptors, descriptors.transpose(-2, -1)) * self.scale
        topk_idx = block_logits.topk(k=min(self.topk, num_blocks), dim=-1).indices
        block_mask = torch.zeros(b, num_blocks, num_blocks, dtype=torch.bool, device=x.device)
        block_mask.scatter_(-1, topk_idx, True)
        block_mask = block_mask | torch.eye(num_blocks, dtype=torch.bool, device=x.device).unsqueeze(0)
        token_mask = block_mask.repeat_interleave(bs, dim=1).repeat_interleave(bs, dim=2)
        if pad_len:
            token_mask = token_mask & (torch.arange(npad, device=x.device) < n)

        logits = self._token_logits(q, k).masked_fill(~token_mask.unsqueeze(1), -1e4)
        attn = self.attn_drop(logits.softmax(dim=-1))
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(b, npad, c)
        return self.proj_drop(self.proj(out[:, :n]))


class AIFI_BlockSparsePola(AIFI_BlockSparse):
    """Original BlockSparse AIFI with polarity correction on visible token scores."""

    def __init__(self, c1, cm=2048, num_heads=8, block_size=4, topk=4,
                 dropout=0.0, act=nn.GELU(), normalize_before=False):
        # Retain the original encoder forward, substituting only the attention instance.
        nn.Module.__init__(self)
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = BlockSparsePolarityAttention(c1, num_heads, block_size, topk, dropout)
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act

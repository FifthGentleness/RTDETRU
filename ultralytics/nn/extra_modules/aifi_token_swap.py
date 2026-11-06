"""Equal-budget token selection for BlockSparse V1, inspired by VecAttention (CVPR 2026)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aifi_block_sparse import AIFI_BlockSparse, BlockSparseAttention

__all__ = ['AIFI_BlockSparseTokenSwap']


class _TokenSwapAttention(BlockSparseAttention):
    def __init__(self, original):
        # Keep V1's exact initialization and state_dict paths; only this
        # instance is replaced. No extra draw from the initialization RNG.
        nn.Module.__init__(self)
        self.dim, self.num_heads, self.head_dim = original.dim, original.num_heads, original.head_dim
        self.scale, self.block_size, self.topk = original.scale, original.block_size, original.topk
        self.qkv, self.proj = original.qkv, original.proj
        self.attn_drop, self.proj_drop = original.attn_drop, original.proj_drop

    def forward(self, x):
        b, n, c = x.shape
        bs = self.block_size
        blocks = (n + bs - 1) // bs
        if int(blocks * self.swap_fraction) == 0:
            return super().forward(x)

        padded = blocks * bs
        if padded != n:
            x = F.pad(x, (0, 0, 0, padded - n))
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, padded, self.num_heads, self.head_dim).permute(0, 2, 1, 3) for t in qkv]

        # V1's block selection, followed by the plugin's equal-budget mask.
        descriptors = x.view(b, blocks, bs, c).mean(dim=2)
        routing = (descriptors @ descriptors.transpose(-2, -1)) * self.scale
        indices = routing.topk(k=min(self.topk, blocks), dim=-1).indices
        block_mask = torch.zeros(b, blocks, blocks, dtype=torch.bool, device=x.device)
        block_mask.scatter_(-1, indices, True)
        block_mask = block_mask | torch.eye(blocks, dtype=torch.bool, device=x.device)[None]
        logits = (q @ k.transpose(-2, -1)) * self.scale
        mask = self._key_mask(block_mask, logits, n)
        attention = self.attn_drop(logits.masked_fill(~mask.unsqueeze(1), -1e4).softmax(dim=-1))
        out = (attention @ v).permute(0, 2, 1, 3).reshape(b, padded, c)[:, :n]
        return self.proj_drop(self.proj(out))

    @torch.no_grad()
    def _key_mask(self, block_mask, logits, n):
        b, blocks, _ = block_mask.shape
        bs = self.block_size
        limit = int(blocks * self.swap_fraction)
        padded = blocks * bs
        token_idx = torch.arange(padded, device=logits.device)
        block_tokens = token_idx.view(blocks, bs)
        valid = token_idx < n
        visible = block_mask.repeat_interleave(bs, dim=-1) & valid
        if limit == 0:
            return visible.repeat_interleave(bs, dim=1)
        valid_rows = valid.view(blocks, bs)
        # The reference statistics exclude padded keys and padded query rows.
        weights = logits.float().masked_fill(~valid, -torch.inf).softmax(dim=-1).mean(dim=1)
        weights = weights.reshape(b, blocks, bs, padded)
        mass = (weights * valid_rows[None, :, :, None]).sum(dim=2)
        mass = mass / valid_rows.sum(dim=1)[None, :, None]

        # The self block and any incomplete key block are protected.
        eye = torch.eye(blocks, dtype=torch.bool, device=logits.device)
        removable = block_mask & ~eye & valid_rows.all(dim=-1)
        block_mass = mass.view(b, blocks, blocks, bs).sum(dim=-1)
        removed_mass, removed = block_mass.masked_fill(~removable, torch.inf).min(dim=-1)
        added_mass, added = mass.masked_fill(visible | ~valid, -torch.inf).topk(bs, dim=-1)
        added_mass = added_mass.sum(dim=-1)
        gain = added_mass - removed_mass
        eligible = torch.isfinite(gain) & (gain >= self.min_gain) & (added_mass >= self.min_ratio * removed_mass)
        winners = gain.masked_fill(~eligible, -torch.inf).topk(limit, dim=-1).indices
        chosen = torch.zeros_like(eligible).scatter_(-1, winners, True) & eligible
        swapped = visible.scatter(-1, block_tokens[removed], False).scatter_(-1, added, True)
        visible = torch.where(chosen[..., None], swapped, visible)
        return visible.repeat_interleave(bs, dim=1)


class AIFI_BlockSparseTokenSwap(AIFI_BlockSparse):
    """Keep V1 parameters and attention, changing only the selected key mask."""

    innovation_schedule = True

    def __init__(self, c1, cm=1024, num_heads=8, block_size=4, topk=4, swap_params=None, **kwargs):
        super().__init__(c1, cm, num_heads, block_size, topk, **kwargs)
        self.attn = _TokenSwapAttention(self.attn)
        self._configure_swap(**(swap_params or {}))

    def _configure_swap(self, max_fraction=0.10, min_gain=0.02, min_ratio=1.25, start_epoch=0, full_epoch=1):
        if not 0 <= max_fraction <= 1 or min_gain < 0 or min_ratio < 1 or not 0 <= start_epoch < full_epoch:
            raise ValueError('Invalid BlockSparse token swap parameters')
        self.max_fraction, self.start_epoch, self.full_epoch = max_fraction, start_epoch, full_epoch
        self.attn.min_gain, self.attn.min_ratio = min_gain, min_ratio
        self.attn.swap_fraction = max_fraction  # standalone evaluation uses the configured cap

    def set_epoch(self, epoch, total_epochs=0):
        progress = (epoch + 1 - self.start_epoch) / (self.full_epoch - self.start_epoch)
        self.attn.swap_fraction = self.max_fraction * min(1.0, max(0.0, progress))

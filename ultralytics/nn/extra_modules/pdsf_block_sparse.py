"""PDSF with the original BlockSparse routing applied to both polarity branches."""

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from .aifi_block_sparse import AIFI_BlockSparse
from .pdsf_attention import DynamicTanh, GlobalPolaAttention, Mona, PDSF_Attention, subchannel

__all__ = ('PDSF_BlockSparse',)


class BlockRoutedPolaAttention(GlobalPolaAttention):
    def __init__(self, dim, hw, block_size, topk):
        if block_size < 1 or (hw[0] * hw[1]) % block_size:
            raise ValueError('block_size must divide the number of PDSF tokens')
        if topk < 1:
            raise ValueError('topk must be >= 1')
        super().__init__(dim, hw)
        self.block_size = block_size
        self.topk = topk

    def _route(self, tokens):
        b, n, c = tokens.shape
        bs = self.block_size
        blocks = n // bs
        # The fixed sine/cosine encoding belongs to the original router only.
        # PDSF's learned K encoding and Q/K/V projections stay unchanged.
        pos = AIFI_BlockSparse.build_2d_sincos_position_embedding(self.w, self.h, c).to(tokens)
        descriptors = (tokens + pos).reshape(b, blocks, bs, c).mean(dim=2)
        logits = (descriptors @ descriptors.transpose(-2, -1)) * self.head_dim ** -0.5
        chosen = logits.topk(k=min(self.topk, blocks), dim=-1).indices
        own = torch.arange(blocks, device=tokens.device).view(1, blocks, 1).expand(b, -1, -1)
        # Append the own block; disable its appended slot if it was already selected.
        valid = torch.cat((torch.ones_like(chosen, dtype=torch.bool), ~(chosen == own).any(-1, keepdim=True)), -1)
        block_indices = torch.cat((chosen, own), -1)
        offsets = torch.arange(bs, device=tokens.device)
        key_indices = (block_indices.unsqueeze(-1) * bs + offsets).flatten(-2)
        return key_indices, valid.repeat_interleave(bs, dim=-1)

    def _aggregate_polarity(self, tokens, q_sim, q_opp, k, v):
        b, n, _ = tokens.shape
        bs = self.block_size
        key_indices, valid = self._route(tokens)
        batch_indices = torch.arange(b, device=tokens.device).view(b, 1, 1)

        def gather(x):
            return x.transpose(1, 2)[batch_indices, key_indices].permute(0, 3, 1, 2, 4)

        keys = gather(k)
        v1, v2 = gather(v).chunk(2, dim=-1)
        allowed = valid[:, None, :, None, :]
        epsilon = 1e-6 * allowed.sum(dim=-1, keepdim=True)
        # Keep the selected-key reductions in FP32 under AMP.
        autocast_off = (torch.autocast(tokens.device.type, enabled=False)
                        if tokens.device.type in ('cpu', 'cuda') else nullcontext())
        dtype = torch.float64 if q_sim.dtype == torch.float64 else torch.float32
        with autocast_off:
            key_t = keys.to(dtype).transpose(-2, -1)

            def read(query, values):
                query = query.reshape(b, self.num_heads, n // bs, bs, -1).to(dtype)
                scores = (query @ key_t).masked_fill(~allowed, 0)
                # Original PDSF divides by mean(score) + eps. After cancelling
                # the mean factors, the denominator is sum(score) + count * eps.
                result = (scores @ values.to(dtype)) / (scores.sum(-1, keepdim=True) + epsilon)
                return result.flatten(2, 3).to(v.dtype)

            return read(q_sim, v1), read(q_opp, v2)

    def forward(self, tokens):
        b, n, c = tokens.shape
        if self.topk >= n // self.block_size:
            return super().forward(tokens)

        # This variant uses the original stride-1 PDSF projections and polarity split.
        q, gate = self.qg(tokens).reshape(b, n, 2, c).unbind(2)
        k, v = self.kv(tokens).reshape(b, n, 2, c).unbind(2)
        k = k + self.positional_encoding
        scale = F.softplus(self.scale)
        power = 1 + self.alpha * self.power.sigmoid()

        def heads(x):
            return x.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        q, k, v = heads(q / scale), heads(k / scale), heads(v)
        q_pos, q_neg = F.relu(q) ** power, F.relu(-q) ** power
        k_pos, k_neg = F.relu(k) ** power, F.relu(-k) ** power
        q_sim = torch.cat((q_pos, q_neg), dim=-1)
        q_opp = torch.cat((q_neg, q_pos), dim=-1)
        k = torch.cat((k_pos, k_neg), dim=-1)
        x_sim, x_opp = self._aggregate_polarity(tokens, q_sim, q_opp, k, v)
        x = torch.cat((x_sim, x_opp), dim=-1).transpose(1, 2).reshape(b, n, c)

        local = v.reshape(b * self.num_heads, self.h, self.w, -1).permute(0, 3, 1, 2)
        local = self.dwc(local).reshape(b, c, n).permute(0, 2, 1)
        return self.proj_drop(self.proj((x + local) * gate))


class PDSF_BlockSparse(PDSF_Attention):
    """Original PDSF block with the same sparse support for both polarity streams."""

    def __init__(self, c1, block_size=4, topk=4, dropout=0.0):
        # Keep the original PDSF forward and state-dict layout, using our attention core.
        nn.Module.__init__(self)
        self.pola_attention = BlockRoutedPolaAttention(c1, (20, 20), block_size, topk)
        self.ffn = subchannel(c1, 2.0, False)
        self.norm1 = DynamicTanh(c1, channels_last=False)
        self.norm2 = DynamicTanh(c1, channels_last=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mona1 = Mona(c1)
        self.mona2 = Mona(c1)

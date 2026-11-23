"""Multi-head adaptation of the ESSAttn self-attention used in URMIF."""

import torch
from torch import nn
from torch.nn import functional as F

from ..modules.transformer import AIFI

__all__ = ("AIFI_ESSAttn",)


class _ESSAttention(nn.Module):
    """Apply the URMIF attention kernel with AIFI's existing projection weights."""

    def __init__(self, attention):
        super().__init__()
        self.num_heads = attention.num_heads
        self.head_dim = attention.head_dim
        self.in_proj_weight = attention.in_proj_weight
        self.in_proj_bias = attention.in_proj_bias
        self.out_proj = attention.out_proj

    def _split_heads(self, tokens):
        batch, length, _ = tokens.shape
        return tokens.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _normalized_energy(tokens, dim):
        # FP32 avoids underflow when squaring and normalizing AMP projections.
        if tokens.dtype in (torch.float16, torch.bfloat16):
            tokens = tokens.float()
        energy = (tokens - tokens.mean(dim=-1, keepdim=True)).square()
        energy = energy / (energy.sum(dim=-1, keepdim=True) + 1e-7)
        return F.normalize(energy, dim=dim)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        if attn_mask is not None or key_padding_mask is not None:
            raise ValueError("AIFI_ESSAttn expects unmasked P5 tokens.")

        q_weight, k_weight, v_weight = self.in_proj_weight.chunk(3)
        q_bias, k_bias, v_bias = self.in_proj_bias.chunk(3)
        q = self._split_heads(F.linear(query, q_weight, q_bias))
        k = self._split_heads(F.linear(key, k_weight, k_bias))
        v = self._split_heads(F.linear(value, v_weight, v_bias))

        # URMIF ATT/attention_layer.py normalizes Q over channels and K over tokens.
        q_energy = self._normalized_energy(q, dim=-1).to(v.dtype)
        k_energy = self._normalized_energy(k, dim=-2).to(v.dtype)
        context = k_energy.transpose(-2, -1) @ v
        mixed = v + (q_energy @ context) * (v.shape[-2] ** -0.5)
        return self.out_proj(mixed.transpose(1, 2).reshape_as(value)), None


class AIFI_ESSAttn(AIFI):
    """Replace P5 token mixing with ESSAttn while reusing AIFI's encoder layer."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        self.ma = _ESSAttention(self.ma)

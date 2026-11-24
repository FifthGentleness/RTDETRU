"""AIFI with the global magnitude-aware kernel of Fan et al., ICCV 2025.

Reference: https://openaccess.thecvf.com/content/ICCV2025/html/Fan_Rectifying_Magnitude_Neglect_in_Linear_Attention_ICCV_2025_paper.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.transformer import AIFI

__all__ = ('AIFI_MagnitudeAware',)


class _MagnitudeAwareAttention(nn.Module):
    """Factorized global attention with a query-dependent gain and offset."""

    def __init__(self, attention):
        super().__init__()
        self.num_heads = attention.num_heads
        self.head_dim = attention.head_dim
        self.scale = self.head_dim ** -0.5
        self.in_proj_weight = attention.in_proj_weight
        self.in_proj_bias = attention.in_proj_bias
        self.out_proj = attention.out_proj
        self.value_dropout = nn.Dropout(attention.dropout)

    def _split_heads(self, x):
        return x.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        if attn_mask is not None or key_padding_mask is not None:
            raise ValueError('Magnitude-aware AIFI requires an unmasked feature map')

        q_weight, k_weight, v_weight = self.in_proj_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = self.in_proj_bias.chunk(3, dim=0)
        q = self._split_heads(F.linear(query, q_weight, q_bias))
        k = self._split_heads(F.linear(key, k_weight, k_bias))
        v = self._split_heads(self.value_dropout(F.linear(value, v_weight, v_bias)))
        projection_dtype = v.dtype
        compute_dtype = torch.float32 if v.dtype in (torch.float16, torch.bfloat16) else v.dtype

        # The gain and offset can nearly cancel. Accumulate both in full precision under AMP.
        with torch.autocast(device_type=v.device.type, enabled=False):
            q = F.elu(q.to(compute_dtype)) + 1.0
            k = F.elu(k.to(compute_dtype)) + 1.0
            v = v.to(compute_dtype)

            # Normalize by the key count as in the authors' vision implementation.
            mean_score = (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1)) * self.scale
            key_value = k.transpose(-2, -1) @ (v * (self.scale / v.shape[-2]))
            mixed = (q @ key_value) * (1.0 + (mean_score + 1e-6).reciprocal())
            mixed = mixed - mean_score * v.mean(dim=-2, keepdim=True)

        mixed = mixed.transpose(1, 2).flatten(2).to(projection_dtype)
        return self.out_proj(mixed), None


class AIFI_MagnitudeAware(AIFI):
    """Use MALA for P5 token mixing while retaining AIFI's position embedding and FFN."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        self.ma = _MagnitudeAwareAttention(self.ma)

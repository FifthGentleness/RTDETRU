# Adapted from NLformer, https://github.com/Ryann2001/NLformer_Lightweight_Denoising
# Copyright (c) 2026 Runyang He
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("AIFI_DWSA",)


class DWSA(nn.Module):
    """NLformer spatial, second-order channel and rank-2 correction features."""

    def __init__(self, dim, num_heads=8, window_size=5, overlap_ratio=0.5):
        super().__init__()
        if dim <= 0 or num_heads <= 0 or dim % num_heads:
            raise ValueError("DWSA channels must be positive and divisible by num_heads")
        if window_size <= 0 or not 0 <= overlap_ratio <= 1:
            raise ValueError("DWSA requires a positive window size and overlap ratio in [0, 1]")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        extra = int(window_size * overlap_ratio)
        self.kv_window = window_size + extra
        # Asymmetric when extra is odd, so the unfold always matches the query grid.
        left, right = extra // 2, extra - extra // 2
        self.halo = (left, right, left, right)
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, bias=False)
        self.qkv_c = nn.Conv2d(dim, 2 * dim, 1, bias=False)
        self.qkv_dwconv = nn.Conv2d(2 * dim, 2 * dim, 3, padding=1, groups=2 * dim, bias=False)
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.rel_pos_emb = nn.Parameter(torch.empty(window_size ** 2, self.kv_window ** 2))
        nn.init.trunc_normal_(self.rel_pos_emb, std=0.02)
        self.phi = nn.Conv2d(dim, 2 * dim, 1, bias=False)
        self.project_out = nn.Conv2d(3 * dim, dim, 1, bias=False)

    def _spatial(self, x):
        b, c, h, w = x.shape
        s, k = self.window_size, self.kv_window
        heads, d = self.num_heads, self.head_dim
        pad_h, pad_w = (-h) % s, (-w) % s
        nh, nw = (h + pad_h) // s, (w + pad_w) // s

        qkv = self.qkv(x)
        if pad_h or pad_w:
            qkv = F.pad(qkv, (0, pad_w, 0, pad_h))
        q, kv = qkv.split((c, 2 * c), dim=1)
        q = q.reshape(b, heads, d, nh, s, nw, s)
        q = q.permute(0, 3, 5, 1, 4, 6, 2).reshape(b, nh * nw, heads, s * s, d)
        kv = F.unfold(F.pad(kv, self.halo, mode="circular"), k, stride=s)
        kv = kv.reshape(b, 2, heads, d, k * k, nh * nw).permute(1, 0, 5, 2, 4, 3)
        key, value = kv.unbind(0)
        logits = (q * self.scale) @ key.transpose(-2, -1)
        logits = logits + self.rel_pos_emb.to(dtype=logits.dtype)
        if pad_h or pad_w:
            valid = x.new_ones((1, 1, h, w))
            valid = F.pad(valid, (0, pad_w, 0, pad_h))
            valid = F.unfold(F.pad(valid, self.halo, mode="circular"), k, stride=s)
            logits = logits.masked_fill(~valid.transpose(1, 2).unsqueeze(2).unsqueeze(2).bool(), float("-inf"))
        out = logits.softmax(dim=-1) @ value
        out = out.reshape(b, nh, nw, heads, s, s, d).permute(0, 3, 6, 1, 4, 2, 5)
        return out.reshape(b, c, nh * s, nw * s)[:, :, :h, :w]

    @staticmethod
    def _rank2(u, v, value):
        """Apply U^T (I + U U^T)^-1 U to value, with U stacking u and v."""
        with torch.autocast(device_type=value.device.type, enabled=False):
            dtype = torch.float32 if value.dtype in (torch.float16, torch.bfloat16) else value.dtype
            u, v, z = u.to(dtype), v.to(dtype), value.to(dtype)
            uu = u.square().sum(dim=-2, keepdim=True)
            vv = v.square().sum(dim=-2, keepdim=True)
            uv = (u * v).sum(dim=-2, keepdim=True)
            # Expand the determinant: the Gram determinant is nonnegative.
            det = 1 + uu + vv + (uu * vv - uv.square()).clamp_min(0) + 1e-4
            uz = (u * z).sum(dim=-2, keepdim=True)
            vz = (v * z).sum(dim=-2, keepdim=True)
            a = ((1 + vv) * uz - uv * vz) / det
            b = ((1 + uu) * vz - uv * uz) / det
            return (u * a + v * b).to(value.dtype)

    def forward(self, x):
        b, c, h, w = x.shape
        head_shape = (b, self.num_heads, self.head_dim, h * w)
        spatial = self._spatial(x)

        q, value = self.qkv_dwconv(self.qkv_c(x)).chunk(2, dim=1)
        q = F.normalize(q.reshape(head_shape), dim=-1, eps=1e-6)
        value = value.reshape(head_shape)
        gram = (q @ q.transpose(-2, -1)) * self.temperature.to(q.dtype)
        channel = (gram.softmax(dim=-1) @ value).reshape(b, c, h, w)

        u, v = self.phi(x).chunk(2, dim=1)
        correction = self._rank2(u.reshape(head_shape), v.reshape(head_shape), value)
        return self.project_out(torch.cat((spatial, channel, correction.reshape(b, c, h, w)), dim=1))


class AIFI_DWSA(nn.Module):
    """Replace AIFI's attention with DWSA and retain its post-norm FFN."""

    def __init__(self, c1, cm=1024, num_heads=8, window_size=5, overlap_ratio=0.5, dropout=0.0):
        super().__init__()
        self.attn = DWSA(c1, num_heads, window_size, overlap_ratio)
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.dropout1(self.attn(x))
        tokens = self.norm1(x.flatten(2).transpose(1, 2))
        ffn = self.fc2(self.dropout(self.act(self.fc1(tokens))))
        tokens = self.norm2(tokens + self.dropout2(ffn))
        return tokens.transpose(1, 2).reshape(b, c, h, w).contiguous()

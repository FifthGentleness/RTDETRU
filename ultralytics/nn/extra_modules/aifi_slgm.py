"""Query-weighted diagonal geometry for AIFI (SLGM).

Inspired by Local Geometry Attention, ICLR 2026:
https://github.com/dongbeank/LGA
This is an analytic, bounded adaptation, not a reproduction of LGA's learned metric.
The main class preserves standard AIFI's global support and Q/K-only position encoding.
The optional BlockSparse class preserves V1's routing and position/residual conventions.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.transformer import AIFI
from .aifi_block_sparse import AIFI_BlockSparse

__all__ = ['AIFI_SLGM', 'AIFI_BlockSparseSLGM']


def geometry_logits(q, k, logits, strength, shrinkage=0.1, eps=1e-6):
    """Correct projected Q/K logits without a B*H*N*N*D intermediate.

    Neighborhood statistics and normalization are detached; the distance still
    differentiates through Q/K. Existing masked logits remain masked.
    """
    with torch.cuda.amp.autocast(enabled=False):
        qf = q.float() if q.dtype in (torch.float16, torch.bfloat16) else q
        kf = k.float() if k.dtype in (torch.float16, torch.bfloat16) else k
        scores = logits.to(qf.dtype)
        with torch.no_grad():
            a0 = scores.softmax(-1)
            variance = (a0 @ kf.square() - 2 * qf * (a0 @ kf) + qf.square()).clamp_min(0)
            shrunk = variance + shrinkage * variance.mean(-1, keepdim=True) + eps
            metric = (shrunk.mean(-1, keepdim=True) / shrunk).clamp(0.25, 4.)
            metric = metric / metric.mean(-1, keepdim=True)
        distance = (metric @ kf.square().transpose(-2, -1)
                    - 2 * (qf * metric) @ kf.transpose(-2, -1)
                    + (metric * qf.square()).sum(-1, keepdim=True)) / qf.shape[-1]
        distance = distance.clamp_min(0)
        scale = (a0 * distance.detach()).sum(-1, keepdim=True) + eps
        return scores - strength * (distance / scale).clamp(0, 4.)


class _SLGMSchedule:
    innovation_schedule = True

    def _init_slgm(self, eta_max, warmup_epochs, shrinkage, eps):
        if eta_max < 0 or warmup_epochs < 0 or shrinkage < 0 or eps <= 0:
            raise ValueError('Invalid SLGM strength, warmup, shrinkage or epsilon')
        self.eta_max, self.warmup_epochs = float(eta_max), float(warmup_epochs)
        self.shrinkage, self.eps = float(shrinkage), float(eps)
        self.register_buffer('slgm_strength', torch.tensor(0. if warmup_epochs else float(eta_max)))

    def set_epoch(self, epoch, total_epochs=0):
        """Called by RTDETRTrainer using the actual zero-based resumed epoch."""
        ramp = min(max(float(epoch), 0.) / self.warmup_epochs, 1.) if self.warmup_epochs else 1.
        self.slgm_strength.fill_(self.eta_max * ramp)


class AIFI_SLGM(_SLGMSchedule, AIFI):
    """Standard global AIFI + bounded diagonal geometry; no sparse routing.

    YAML args: [cm, heads, eta_max, warmup_epochs, shrinkage, eps].
    Standard AIFI parameter names are retained for direct weight transfer.
    """
    def __init__(self, c1, cm=2048, num_heads=8, eta_max=0.25, warmup_epochs=30,
                 shrinkage=0.1, eps=1e-6, dropout=0., act=None, normalize_before=False):
        super().__init__(c1, cm, num_heads, dropout, nn.GELU() if act is None else act, normalize_before)
        self._init_slgm(eta_max, warmup_epochs, shrinkage, eps)

    def _attention(self, src, pos, src_mask=None, src_key_padding_mask=None):
        b, n, c = src.shape
        h, d = self.ma.num_heads, c // self.ma.num_heads
        # Match nn.MultiheadAttention: Q/K include pos BEFORE their projection, V does not.
        w, bias = self.ma.in_proj_weight, self.ma.in_proj_bias
        q, k = F.linear(self.with_pos_embed(src, pos), w[:2*c], bias[:2*c]).chunk(2, -1)
        v = F.linear(src, w[2*c:], bias[2*c:])
        q, k, v = [t.reshape(b, n, h, d).transpose(1, 2) for t in (q, k, v)]
        logits = (q @ k.transpose(-2, -1)) * d ** -0.5
        if src_mask is not None:
            mask = src_mask if src_mask.ndim == 2 else src_mask.reshape(b, h, n, n)
            logits = logits.masked_fill(mask, float('-inf')) if mask.dtype == torch.bool else logits + mask
        if src_key_padding_mask is not None:
            mask = src_key_padding_mask[:, None, None, :]
            logits = logits.masked_fill(mask, float('-inf')) if mask.dtype == torch.bool else logits + mask
        logits = geometry_logits(q, k, logits, self.slgm_strength, self.shrinkage, self.eps)
        attention = F.dropout(logits.softmax(-1).to(v.dtype), p=self.ma.dropout, training=self.training)
        out = (attention @ v).transpose(1, 2).reshape(b, n, c)
        return self.ma.out_proj(out)

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src = self.norm1(src + self.dropout1(self._attention(src, pos, src_mask, src_key_padding_mask)))
        return self.norm2(src + self.dropout2(self.fc2(self.dropout(self.act(self.fc1(src))))))

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src = src + self.dropout1(self._attention(self.norm1(src), pos, src_mask, src_key_padding_mask))
        return src + self.dropout2(self.fc2(self.dropout(self.act(self.fc1(self.norm2(src))))))


class AIFI_BlockSparseSLGM(_SLGMSchedule, AIFI_BlockSparse):
    """Optional V1 + SLGM. Decoder/loss experiments do not use this class.

    YAML args: [cm, heads, block_size, topk, eta_max, warmup_epochs, shrinkage, eps].
    This remains dense-masked attention, not a hardware sparse implementation.
    """
    def __init__(self, c1, cm=2048, num_heads=8, block_size=4, topk=4,
                 eta_max=0.25, warmup_epochs=30, shrinkage=0.1, eps=1e-6,
                 dropout=0., act=None, normalize_before=False):
        super().__init__(c1, cm, num_heads, block_size, topk, dropout,
                         nn.GELU() if act is None else act, normalize_before)
        self._init_slgm(eta_max, warmup_epochs, shrinkage, eps)

    def _attention(self, x):
        attn = self.attn
        b, n, c = x.shape
        h, d, bs = attn.num_heads, attn.head_dim, attn.block_size
        blocks = math.ceil(n / bs)
        padded = blocks * bs
        if padded != n:
            x = F.pad(x, (0, 0, 0, padded-n))
        q, k, v = [t.reshape(b, padded, h, d).transpose(1, 2) for t in attn.qkv(x).chunk(3, -1)]
        raw = x.reshape(b, blocks, bs, c).mean(2)
        routing = (raw @ raw.transpose(-2, -1)) * attn.scale
        indices = routing.topk(min(attn.topk, blocks), -1).indices
        mask = torch.zeros(b, blocks, blocks, device=x.device, dtype=torch.bool)
        mask.scatter_(-1, indices, True)
        mask = mask | torch.eye(blocks, device=x.device, dtype=torch.bool)[None]
        mask = mask.repeat_interleave(bs, 1).repeat_interleave(bs, 2)
        if padded != n:
            mask = mask & (torch.arange(padded, device=x.device) < n)[None, None]
        logits = ((q @ k.transpose(-2, -1)) * attn.scale).masked_fill(~mask[:, None], -1e4)
        logits = geometry_logits(q, k, logits, self.slgm_strength, self.shrinkage, self.eps)
        weights = attn.attn_drop(logits.softmax(-1).to(v.dtype))
        out = (weights @ v).transpose(1, 2).reshape(b, padded, c)[:, :n]
        return attn.proj_drop(attn.proj(out))

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.c1:
            raise ValueError(f'AIFI_BlockSparseSLGM expected {self.c1} channels, got {c}')
        pos = self.build_2d_sincos_position_embedding(w, h, c).to(x.device, x.dtype)
        src = x.flatten(2).transpose(1, 2) + pos  # preserve V1, including its residual convention
        if self.normalize_before:
            src = src + self.dropout1(self._attention(self.norm1(src)))
            src = src + self.dropout2(self.fc2(self.dropout(self.act(self.fc1(self.norm2(src))))))
        else:
            src = self.norm1(src + self.dropout1(self._attention(src)))
            src = self.norm2(src + self.dropout2(self.fc2(self.dropout(self.act(self.fc1(src))))))
        return src.transpose(1, 2).reshape(b, c, h, w).contiguous()

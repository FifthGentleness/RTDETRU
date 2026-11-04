"""Content-aware interpolation at existing P3 sampling locations (CIR).

Inspired by IART, CVPR 2024 (video super-resolution):
https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Enhancing_Video_Super-Resolution_via_Implicit_Resampling-based_Alignment_CVPR_2024_paper.html
This bounded bilinear-prior adaptation is not the original IART architecture.
No BlockSparse dependency; self-attention, DN masks and reference refinement are unchanged.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.head import RTDETRDecoder
from ..modules.transformer import MSDeformAttn

__all__ = ['RTDETRDecoderCIR']


class ContentAwareBilinear(nn.Module):
    """Reweight the same four lattice neighbors, including virtual zero-padding.

    Inputs: feature [B*heads,D,H,W], query [B*heads,Q,D], location [B*heads,Q,K,2].
    Output: [B*heads,D,Q,K]. Default D=32 gives exactly 945 trainable parameters.
    """
    def __init__(self, head_dim, hidden=8, mlp_hidden=16, frequencies=2):
        super().__init__()
        if min(head_dim, hidden, mlp_hidden, frequencies) < 1:
            raise ValueError('CIR dimensions/frequencies must be positive')
        self.query_proj = nn.Linear(head_dim, hidden)
        self.value_proj = nn.Linear(head_dim, hidden)
        self.mlp = nn.Sequential(nn.Linear(2*hidden + 4*frequencies, mlp_hidden),
                                 nn.ReLU(), nn.Linear(mlp_hidden, 1))
        self.register_buffer('frequencies', math.pi * 2. ** torch.arange(frequencies), persistent=False)

    def forward(self, feature, query, location, strength):
        bh, d, height, width = feature.shape
        _, nq, points, _ = location.shape
        # align_corners=False: integer lattice coordinate is s*size - 0.5.
        xy = location * location.new_tensor([width, height]) - 0.5
        base = xy.floor()
        corners = base[..., None, :] + xy.new_tensor([[0,0], [1,0], [0,1], [1,1]])
        ix, iy = corners[..., 0], corners[..., 1]
        valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        index = (iy.clamp(0, height-1).long()*width + ix.clamp(0, width-1).long()).reshape(bh, -1)
        values = feature.flatten(2).gather(2, index[:, None].expand(-1, d, -1))
        values = values.transpose(1, 2).reshape(bh, nq, points, 4, d) * valid[..., None]
        dx, dy = (xy-base).unbind(-1)
        prior = torch.stack(((1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy), -1)
        phase = (xy[..., None, :] - corners)[..., None] * self.frequencies.to(location.dtype)
        position = torch.cat((phase.sin(), phase.cos()), -1).flatten(-2)
        q = self.query_proj(query)[:, :, None, None].expand(-1, -1, points, 4, -1)
        delta = self.mlp(torch.cat((q, self.value_proj(values), position), -1)).squeeze(-1).tanh()
        # Keep the virtual out-of-bounds neighbors in the denominator. Dropping
        # them would break zero-padding equivalence even when strength is zero.
        weights = prior * (strength * delta).exp()
        weights = weights / weights.sum(-1, keepdim=True)
        return (values * weights[..., None]).sum(-2).permute(0, 3, 1, 2).contiguous()


class MSDeformAttnCIR(MSDeformAttn):
    innovation_schedule = True

    def __init__(self, d_model=256, n_levels=3, n_heads=8, n_points=4,
                 matching_queries=300, kappa_max=0.5, warmup_epochs=30, level=0):
        super().__init__(d_model, n_levels, n_heads, n_points)
        if matching_queries < 1 or kappa_max < 0 or warmup_epochs < 0 or not 0 <= level < n_levels:
            raise ValueError('Invalid CIR query count, strength, warmup or level')
        self.matching_queries, self.level = int(matching_queries), int(level)
        self.kappa_max, self.warmup_epochs = float(kappa_max), float(warmup_epochs)
        self.register_buffer('cir_strength', torch.tensor(0. if warmup_epochs else float(kappa_max)))
        self.resampler = ContentAwareBilinear(d_model // n_heads)

    def set_epoch(self, epoch, total_epochs=0):
        ramp = min(max(float(epoch), 0.) / self.warmup_epochs, 1.) if self.warmup_epochs else 1.
        self.cir_strength.fill_(self.kappa_max * ramp)

    def forward(self, query, refer_bbox, value, value_shapes, value_mask=None):
        b, nq, _ = query.shape
        if nq < self.matching_queries:
            raise ValueError('CIR expects matching queries at the end, optionally preceded by DN queries')
        lengths = [h*w for h,w in value_shapes]
        if sum(lengths) != value.shape[1]:
            raise ValueError('CIR value shapes do not match flattened features')
        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], 0.)
        d = self.d_model // self.n_heads
        value = value.reshape(b, -1, self.n_heads, d)
        offsets = self.sampling_offsets(query).reshape(b, nq, self.n_heads, self.n_levels, self.n_points, 2)
        weights = self.attention_weights(query).reshape(b, nq, self.n_heads, -1).softmax(-1)
        weights = weights.reshape(b, nq, self.n_heads, self.n_levels, self.n_points)
        if refer_bbox.shape[-1] == 2:
            normalizer = query.new_tensor(value_shapes).flip(-1)
            locations = refer_bbox[:, :, None, :, None] + offsets / normalizer[None, None, None, :, None]
        elif refer_bbox.shape[-1] == 4:
            locations = (refer_bbox[:, :, None, :, None, :2]
                         + offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError('Reference boxes must have 2 or 4 coordinates')
        dn_num = nq - self.matching_queries
        query_heads = query.reshape(b, nq, self.n_heads, d).transpose(1, 2).reshape(b*self.n_heads, nq, d)
        sampled = []
        for level, (part, (height, width)) in enumerate(zip(value.split(lengths, 1), value_shapes)):
            feature = part.flatten(2).transpose(1, 2).reshape(b*self.n_heads, d, height, width)
            loc = locations[:, :, :, level].transpose(1, 2).reshape(b*self.n_heads, nq, self.n_points, 2)
            if level == self.level:
                matching = self.resampler(feature, query_heads[:, dn_num:], loc[:, dn_num:], self.cir_strength)
                if dn_num:
                    denoising = F.grid_sample(feature, 2*loc[:, :dn_num]-1, mode='bilinear',
                                              padding_mode='zeros', align_corners=False)
                    matching = torch.cat((denoising, matching), 2)
                sampled.append(matching)
            else:
                sampled.append(F.grid_sample(feature, 2*loc-1, mode='bilinear',
                                             padding_mode='zeros', align_corners=False))
        weights = weights.transpose(1, 2).reshape(b*self.n_heads, 1, nq, self.n_levels*self.n_points)
        output = (torch.stack(sampled, -2).flatten(-2) * weights).sum(-1)
        output = output.reshape(b, self.d_model, nq).transpose(1, 2).contiguous()
        return self.output_proj(output)


class RTDETRDecoderCIR(RTDETRDecoder):
    """Only the final layer's P3 matching-query read uses CIR; original head forward is inherited.

    YAML args: [nc,hd,nq,ndp,nh,ndl,kappa_max,warmup_epochs,level].
    """
    def __init__(self, nc=80, ch=(512,1024,2048), hd=256, nq=300, ndp=4, nh=8, ndl=6,
                 kappa_max=0.5, warmup_epochs=30, level=0, **kwargs):
        super().__init__(nc, ch, hd, nq, ndp, nh, ndl, **kwargs)
        if self.decoder.eval_idx != ndl-1:
            raise ValueError('CIR modifies the final decoder layer; eval_idx must select that layer')
        original = self.decoder.layers[-1].cross_attn
        adapted = MSDeformAttnCIR(hd, len(ch), nh, ndp, nq, kappa_max, warmup_epochs, level)
        adapted.load_state_dict(original.state_dict(), strict=False)
        self.decoder.layers[-1].cross_attn = adapted

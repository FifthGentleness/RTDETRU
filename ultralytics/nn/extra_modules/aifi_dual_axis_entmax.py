"""Dual-axis competitive entmax attention on the projected P5 grid.

The key-axis evidence is inspired by the dual-softmax assignment in CoMatch
(ICCV 2025). Adaptive exact-zero query support is inspired by the learnable
temperature in Long-Context Generalization with Sparse Attention (ICLR 2026).
This layer adapts the two ideas to within-image token mixing.
"""

import torch
from torch import nn

__all__ = ("AIFI_DualAxisEntmax",)


@torch.no_grad()
def _entmax15_threshold(scores):
    # The maximum score is zero, so the threshold is at least -1.
    ordered = scores.clamp_min(-1).sort(dim=-1, descending=True).values
    ranks = torch.arange(1, ordered.shape[-1] + 1, device=scores.device, dtype=scores.dtype)
    mean = ordered.cumsum(dim=-1) / ranks
    second_moment = ordered.square().cumsum(dim=-1) / ranks
    thresholds = mean - (ranks.reciprocal() - second_moment + mean.square()).clamp_min(0).sqrt()
    support = ((ordered > thresholds) & (ordered > -1)).sum(dim=-1, keepdim=True).clamp_min(1)
    return thresholds.gather(-1, support - 1)

def _entmax15(logits):
    """Exact alpha=1.5 entmax over the last axis, with its analytic backward."""
    scores = 0.5 * (logits - logits.amax(dim=-1, keepdim=True).detach())
    threshold = _entmax15_threshold(scores)
    # Implicit differentiation of sum((scores - threshold)_+**2) = 1
    # gives the threshold's gradient without differentiating the sort.
    roots = (scores.detach() - threshold).clamp_min(0)
    weighted = (scores * roots).sum(dim=-1, keepdim=True)
    threshold = threshold + (weighted - weighted.detach()) / roots.sum(dim=-1, keepdim=True)
    weights = (scores - threshold).clamp_min(0).square()
    # Correct round-off from the cumulative moments on sharp distributions.
    return weights / weights.sum(dim=-1, keepdim=True)


class AIFI_DualAxisEntmax(nn.Module):
    """Replace P5 token mixing while preserving the [B, C, H, W] interface."""

    def __init__(self, c1, cm=1024, num_heads=8):
        super().__init__()
        if num_heads < 1 or c1 < 1 or c1 % num_heads:
            raise ValueError("P5 channels must be positive and divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = c1 // num_heads
        self.qkv = nn.Linear(c1, 3 * c1)
        self.proj = nn.Linear(c1, c1)
        self.radius = nn.Linear(c1, num_heads)
        self.competition = nn.Parameter(torch.full((num_heads, 1, 1), 0.4))
        self.logit_scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.ffn = nn.Sequential(nn.Linear(c1, cm), nn.GELU(), nn.Linear(cm, c1))

        # Start with a range of spatial extents across heads and allow each
        # query to adjust its extent from the same P5 feature used for Q/K/V.
        with torch.no_grad():
            self.radius.weight.mul_(0.1)
            self.radius.bias.copy_(torch.linspace(-2.2, 1.2, num_heads))

    def _attend(self, tokens, height, width):
        batch, length, channels = tokens.shape
        q, k, v = self.qkv(tokens).reshape(batch, length, 3, self.num_heads, self.head_dim).permute(
            2, 0, 3, 1, 4
        ).unbind(0)
        radius = self.radius(tokens).transpose(1, 2).unsqueeze(-1)

        # Keep scores and the support solve in FP32 under AMP, including QK.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            score_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            q, k = q.to(score_dtype), k.to(score_dtype)
            similarity = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5
            # A popular key must explain this query better than its other
            # queries. Blend in column competition without enforcing a quota.
            evidence = similarity - self.competition.to(score_dtype).sigmoid() * similarity.logsumexp(
                dim=-2, keepdim=True
            )

            y, x = torch.meshgrid(
                torch.arange(height, device=tokens.device, dtype=score_dtype),
                torch.arange(width, device=tokens.device, dtype=score_dtype),
                indexing="ij",
            )
            grid = torch.stack((y.flatten(), x.flatten()), dim=1)
            distance2 = (grid[:, None] - grid[None, :]).square().sum(dim=-1)
            radius = 0.75 + 5.25 * radius.to(score_dtype).sigmoid()
            spatial_bias = 0.8 * torch.exp(-0.5 * distance2 / radius.square())
            scale = 0.5 + 3.0 * self.logit_scale.to(score_dtype).sigmoid()
            weights = _entmax15(scale * evidence + spatial_bias)

        mixed = weights.to(v.dtype) @ v
        return self.proj(mixed.transpose(1, 2).reshape(batch, length, channels))

    def forward(self, x):
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm1(tokens + self._attend(tokens, height, width))
        tokens = self.norm2(tokens + self.ffn(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width).contiguous()

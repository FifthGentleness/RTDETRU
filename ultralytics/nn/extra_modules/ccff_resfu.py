"""Fine-guided upsampling for RT-DETR CCFF, adapted from ReSFU.

Zhou et al., IEEE TNNLS 2026, https://doi.org/10.1109/TNNLS.2025.3638370.
The operator keeps guided-filter alignment, Gaussian detail self-alignment,
paired central differences, and target-grid neighborhood reassembly. It uses
narrower embeddings and score heads, a smaller guided-filter window, and native
PyTorch operations in place of the fused CUDA implementation.
"""

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ("CCFFReSFU",)


class _PairedCentralDifference(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.norm = nn.GroupNorm(4, channels)
        self.difference = nn.Conv2d(channels, channels, 3, dilation=dilation, groups=4)
        self.score = nn.Sequential(nn.SiLU(), nn.Conv2d(channels, 9, 1))
        nn.init.normal_(self.score[-1].weight, std=0.01)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, query, key):
        query, key = self.norm(query), self.norm(key)
        dilation = self.difference.dilation[0]
        key = F.pad(key, (dilation,) * 4, mode="replicate")
        # Eq. 12: convolution on the neighbors minus the kernel sum on the center.
        center_weight = self.difference.weight.sum(dim=(2, 3), keepdim=True)
        center = F.conv2d(query, center_weight, groups=self.difference.groups)
        return self.score(self.difference(key) - center)


class CCFFReSFU(nn.Module):
    """Align coarse features to the projected finer map at an adjacent CCFF level."""

    def __init__(self, channels, embedding_dim=16, radius=2):
        super().__init__()
        coarse_channels, fine_channels = channels
        if embedding_dim < 8 or embedding_dim % 4:
            raise ValueError("embedding_dim must be a multiple of four and at least eight")
        if radius < 1:
            raise ValueError("radius must be positive")

        self.radius = radius
        self.dilation = 2
        self.coarse_key = nn.Conv2d(coarse_channels, embedding_dim, 1)
        self.fine_query = nn.Conv2d(fine_channels, embedding_dim, 1)
        self.semantic_score = _PairedCentralDifference(embedding_dim, self.dilation)
        self.detail_score = _PairedCentralDifference(embedding_dim, self.dilation)
        # A center preference starts close to bilinear interpolation on the target grid.
        with torch.no_grad():
            self.semantic_score.score[-1].bias[4] = 3.0

        offsets = torch.arange(-1, 2, dtype=torch.float32)
        gaussian = torch.exp(-0.5 * offsets.square())
        gaussian = gaussian / gaussian.sum()
        kernel = torch.outer(gaussian, gaussian).view(1, 1, 3, 3)
        self.register_buffer("gaussian", kernel.repeat(embedding_dim, 1, 1, 1), persistent=False)

    def _semantic_alignment(self, query, key):
        # Evaluate the local variance and covariance in FP32 under mixed precision.
        dtype = query.dtype
        if dtype in (torch.float16, torch.bfloat16):
            query, key = query.float(), key.float()

        def local_mean(x):
            return F.avg_pool2d(x, 2 * self.radius + 1, stride=1, padding=self.radius, count_include_pad=False)

        query_mean, key_mean = local_mean(query), local_mean(key)
        covariance = local_mean(query * key) - query_mean * key_mean
        variance = (local_mean(query.square()) - query_mean.square()).clamp_min(0.0)
        slope = covariance / (variance + 1e-3)
        intercept = key_mean - slope * query_mean
        return (local_mean(slope) * query + local_mean(intercept)).to(dtype=dtype)

    def _reassemble(self, value, weights):
        dilation = self.dilation
        height, width = value.shape[-2:]
        padded = F.pad(value, (dilation,) * 4, mode="replicate")
        # Sliced neighbors share storage. This avoids a B x C x 9 x H x W unfold.
        fused = value * weights[:, 4:5]
        for index, weight in enumerate(weights.split(1, dim=1)):
            if index == 4:
                continue
            row, column = divmod(index, 3)
            top, left = row * dilation, column * dilation
            neighbor = padded[:, :, top : top + height, left : left + width]
            fused = fused + neighbor * weight
        return fused

    def forward(self, features):
        coarse, fine = features
        size = fine.shape[-2:]
        query = self.fine_query(fine)
        key = F.interpolate(self.coarse_key(coarse), size=size, mode="bilinear", align_corners=False)
        aligned_query = self._semantic_alignment(query, key)
        smooth_query = F.conv2d(
            F.pad(query, (1,) * 4, mode="replicate"), self.gaussian.to(query), groups=query.shape[1]
        )
        logits = self.semantic_score(aligned_query, key) + self.detail_score(query, smooth_query)
        weight_dtype = torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
        weights = logits.softmax(dim=1, dtype=weight_dtype)
        value = F.interpolate(coarse, size=size, mode="bilinear", align_corners=False)
        return self._reassemble(value, weights.to(dtype=value.dtype)).to(dtype=coarse.dtype)

"""Coupled graph refinement for the P3 and P5 outputs of the FSV7 encoder.

The graph-prior unfolding is inspired by LapNet (Zhong et al., CVPR 2026).
Here the variables are residual corrections on two feature grids, coupled
through an appearance-weighted, adjoint restriction and prolongation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.transformer import AIFI

__all__ = ("AIFI_CoupledGraph",)


class AIFI_CoupledGraph(AIFI):
    """Run native P5 AIFI, then jointly refine the P3 and P5 streams."""

    stride_ratio = 4
    coupling = 1.5
    smoothness = 0.25

    def __init__(self, channels, cm=1024, num_heads=8, graph_width=64, steps=4):
        c3, c5 = channels
        super().__init__(c5, cm, num_heads)
        if graph_width < 1 or steps < 1:
            raise ValueError("graph_width and steps must be positive")
        self.steps = steps
        self.fine_in = nn.Sequential(nn.Conv2d(c3, graph_width, 1, bias=False), nn.GroupNorm(1, graph_width))
        self.coarse_in = nn.Sequential(nn.Conv2d(c5, graph_width, 1, bias=False), nn.GroupNorm(1, graph_width))
        self.fine_out = nn.Conv2d(graph_width, c3, 1, bias=False)
        self.coarse_out = nn.Conv2d(graph_width, c5, 1, bias=False)
        self.fine_gain = nn.Parameter(torch.full((1, c3, 1, 1), 0.30))
        self.coarse_gain = nn.Parameter(torch.full((1, c5, 1, 1), 0.20))

        # L <= 8 for the four-neighbor graphs. A centered sigmoid gate
        # satisfies mean(gate**2) <= 1.25 in every restriction cell.
        self.step_size = 1 / (1 + 8 * self.smoothness + 2.25 * self.coupling)

    def _restrict(self, field):
        return F.avg_pool2d(field, kernel_size=self.stride_ratio, stride=self.stride_ratio)

    def _prolong(self, field):
        return F.interpolate(field, scale_factor=self.stride_ratio, mode="nearest")

    @staticmethod
    def _edges(unit_features):
        """Build symmetric horizontal and vertical affinities once per image."""
        horizontal = (unit_features[..., 1:] * unit_features[..., :-1]).sum(dim=1, keepdim=True)
        vertical = (unit_features[..., 1:, :] * unit_features[..., :-1, :]).sum(dim=1, keepdim=True)
        return (5 * (horizontal - 0.45)).sigmoid(), (5 * (vertical - 0.45)).sigmoid()

    @staticmethod
    def _laplacian(field, edges):
        horizontal, vertical = edges
        dx = horizontal * (field[..., 1:] - field[..., :-1])
        dy = vertical * (field[..., 1:, :] - field[..., :-1, :])
        return (F.pad(dx, (1, 0)) - F.pad(dx, (0, 1))
                + F.pad(dy, (0, 0, 1, 0)) - F.pad(dy, (0, 0, 0, 1)))

    def _solve(self, mismatch, gate, fine_edges, coarse_edges):
        """Unroll gradient descent on two anchored residual fields.

        For residuals u (fine) and v (coarse), the agreement term is
        0.5 * coupling * ||mismatch + v - restrict(gate * u)||**2.
        Fine-grid norms use 1/16 per site, so gate * prolong is the
        adjoint of the weighted restriction. Graph terms act on u and v.
        """
        fine_delta = torch.zeros_like(self._prolong(mismatch))
        coarse_delta = torch.zeros_like(mismatch)
        for _ in range(self.steps):
            error = mismatch + coarse_delta - self._restrict(gate * fine_delta)
            fine_grad = (fine_delta + self.smoothness * self._laplacian(fine_delta, fine_edges)
                         - self.coupling * gate * self._prolong(error))
            coarse_grad = (coarse_delta + self.smoothness * self._laplacian(coarse_delta, coarse_edges)
                           + self.coupling * error)
            fine_delta = fine_delta - self.step_size * fine_grad
            coarse_delta = coarse_delta - self.step_size * coarse_grad
        return fine_delta, coarse_delta

    def forward(self, features):
        p3, p5 = features
        if p3.shape[-2:] != tuple(self.stride_ratio * size for size in p5.shape[-2:]):
            raise ValueError("AIFI_CoupledGraph expects P3 at four times the P5 resolution")
        encoded = super().forward(p5)
        fine = self.fine_in(p3)
        coarse = self.coarse_in(encoded)
        if fine.dtype in (torch.float16, torch.bfloat16):
            fine, coarse = fine.float(), coarse.float()

        fine_unit = F.normalize(fine, dim=1, eps=1e-4)
        coarse_unit = F.normalize(coarse, dim=1, eps=1e-4)
        agreement = (fine_unit * self._prolong(coarse_unit)).sum(dim=1, keepdim=True)
        confidence = (4 * agreement).sigmoid()
        gate = 1 + confidence - self._prolong(self._restrict(confidence))

        mismatch = coarse - self._restrict(gate * fine)
        fine_delta, coarse_delta = self._solve(
            mismatch, gate, self._edges(fine_unit), self._edges(coarse_unit)
        )
        fine_update = self.fine_out(fine_delta.to(p3.dtype))
        coarse_update = self.coarse_out(coarse_delta.to(encoded.dtype))
        p3 = p3 + 0.5 * self.fine_gain.tanh().to(p3.dtype) * fine_update
        encoded = encoded + 0.5 * self.coarse_gain.tanh().to(encoded.dtype) * coarse_update
        return encoded, p3

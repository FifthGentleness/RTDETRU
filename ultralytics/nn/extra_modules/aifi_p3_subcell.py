"""Query-conditioned P3 subcell evidence for the FSV7 P5 encoder.

The center-surround contrast is inspired by the Weber Local Adapter of
ORSATR-X (CVPR 2026, Sec. 3.2). The local gradient structure tensor is
inspired by SARD (CVPR 2026, Sec. 3.2). The query-dependent joint attention
over P5 locations and their P3 subcells is specific to this adaptation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["AIFI_P3Subcell"]


class AIFI_P3Subcell(nn.Module):
    """Encode P5 using four signed P3 detail descriptors per P5 key location."""

    def __init__(self, in_channels, cm=1024, num_heads=8, detail_channels=64):
        super().__init__()
        c3, c5 = in_channels
        if num_heads < 1 or c5 % 4 or c5 % (2 * num_heads):
            raise ValueError("P5 channels must be divisible by 4 and twice the head count")
        self.num_heads = num_heads
        self.head_dim = c5 // num_heads
        self.detail_dim = self.head_dim // 2

        self.p3_proj = nn.Conv2d(c3, detail_channels, 1, bias=False)
        self.query = nn.Linear(c5, c5)
        self.key = nn.Linear(c5, c5)
        self.value = nn.Linear(c5, c5)
        self.detail_query = nn.Linear(c5, num_heads * self.detail_dim, bias=False)
        # Contrast, horizontal difference, vertical difference, with x/y phase interactions.
        self.detail_kv = nn.Linear(
            9 * detail_channels, num_heads * (self.detail_dim + self.head_dim), bias=False
        )
        self.proj = nn.Linear(c5, c5)
        self.norm1 = nn.LayerNorm(c5)
        self.norm2 = nn.LayerNorm(c5)
        self.ffn = nn.Sequential(nn.Linear(c5, cm), nn.GELU(), nn.Linear(cm, c5))

        # Small nonzero gains keep both detail paths trainable on the first update.
        self.key_gain = nn.Parameter(torch.full((num_heads, 1, 1), 0.35))
        self.value_gain = nn.Parameter(torch.full((num_heads, 1, 1), 0.20))
        self.register_buffer(
            "phase_xy", torch.tensor([[-1., -1.], [1., -1.], [-1., 1.], [1., 1.]]),
            persistent=False,
        )

    @staticmethod
    def _position(h, w, channels, device, dtype):
        # Preserve the baseline AIFI sinusoidal convention on the semantic path.
        grid_w, grid_h = torch.meshgrid(
            torch.arange(w, device=device, dtype=torch.float32),
            torch.arange(h, device=device, dtype=torch.float32),
            indexing="ij",
        )
        dim = channels // 4
        omega = 10000.0 ** (-torch.arange(dim, device=device, dtype=torch.float32) / dim)
        pos_w = grid_w.reshape(-1, 1) * omega
        pos_h = grid_h.reshape(-1, 1) * omega
        return torch.cat((pos_w.sin(), pos_w.cos(), pos_h.sin(), pos_h.cos()), dim=-1).to(dtype)

    @staticmethod
    def _pack_subcells(x, h, w):
        """Pack stride-8 samples into [B, H*W, 4 quadrants, 4 samples, C]."""
        b, c = x.shape[:2]
        return (
            x.reshape(b, c, h, 2, 2, w, 2, 2)
            .permute(0, 2, 5, 3, 6, 4, 7, 1)
            .reshape(b, h * w, 4, 4, c)
        )

    def _p3_evidence(self, p3, h, w):
        z = self.p3_proj(p3)
        dtype = z.dtype
        if dtype in (torch.float16, torch.bfloat16):
            z = z.float()

        padded = F.pad(z, (1, 1, 1, 1), mode="replicate")
        surround = (9 * F.avg_pool2d(padded, 3, stride=1) - z) / 8
        left, right = padded[..., 1:-1, :-2], padded[..., 1:-1, 2:]
        above, below = padded[..., :-2, 1:-1], padded[..., 2:, 1:-1]
        contrast = (z - surround) / (z.abs() + surround.abs() + 1e-3)
        dx = (right - left) / (right.abs() + left.abs() + 1e-3)
        dy = (below - above) / (below.abs() + above.abs() + 1e-3)

        # A local second moment favors compact structure without vetoing thin edges.
        moments = torch.stack((dx.square(), dy.square(), dx * dy), dim=1).mean(dim=2)
        jxx, jyy, jxy = F.avg_pool2d(moments, 3, stride=1, padding=1, count_include_pad=False).unbind(1)
        determinant = (jxx * jyy - jxy.square()).clamp_min(0)
        isotropy = (4 * determinant / ((jxx + jyy).square() + 1e-4)).clamp_max(1)
        saliency = contrast.abs().mean(dim=1) * (1 + isotropy)

        # Soft pooling within each quadrant preserves all four spatial alternatives.
        weights = self._pack_subcells(saliency.unsqueeze(1), h, w).mul(4).softmax(dim=3)
        samples = self._pack_subcells(torch.cat((contrast, dx, dy), dim=1), h, w)
        pooled = (weights * samples).sum(dim=3).to(dtype)
        phase = self.phase_xy.to(dtype).reshape(1, 1, 4, 2)
        return torch.cat((pooled, pooled * phase[..., :1], pooled * phase[..., 1:]), dim=-1)

    def forward(self, inputs):
        p3, p5 = inputs
        b, c, h, w = p5.shape
        if p3.shape[-2:] != (4 * h, 4 * w):
            raise ValueError("AIFI_P3Subcell expects P3 height and width to be four times P5")

        n = h * w
        heads, d, r = self.num_heads, self.head_dim, self.detail_dim
        x = p5.flatten(2).transpose(1, 2)
        positioned = x + self._position(h, w, c, x.device, x.dtype)

        q = self.query(positioned).reshape(b, n, heads, d).transpose(1, 2)
        k = self.key(positioned).reshape(b, n, heads, d).transpose(1, 2)
        v = self.value(x).reshape(b, n, heads, d).transpose(1, 2)
        q_detail = self.detail_query(x).reshape(b, n, heads, r).transpose(1, 2)
        details = self.detail_kv(self._p3_evidence(p3, h, w))
        details = details.reshape(b, n, 4, heads, r + d).permute(0, 3, 1, 2, 4)
        k_detail, v_detail = details.split((r, d), dim=-1)
        key_gain = self.key_gain.tanh().to(k.dtype)
        value_gain = self.value_gain.tanh().to(v.dtype)

        # Each key (j, a) shares P5 semantics at j and carries P3 evidence from quadrant a.
        # SDPA evaluates (Q_i K_j / sqrt(d)) + gain * (Q'_i K'_{j,a} / sqrt(r)).
        q = torch.cat((q * ((d + r) / d) ** 0.5, q_detail * ((d + r) / r) ** 0.5), dim=-1)
        k = torch.cat((k.unsqueeze(3).expand(-1, -1, -1, 4, -1),
                       key_gain.unsqueeze(3) * k_detail), dim=-1)
        v = v.unsqueeze(3) + value_gain.unsqueeze(3) * v_detail
        v = F.pad(v, (0, r))  # Match query/key head width for fused SDPA kernels.
        mixed = F.scaled_dot_product_attention(
            q, k.reshape(b, heads, 4 * n, d + r), v.reshape(b, heads, 4 * n, d + r)
        )[..., :d]
        x = self.norm1(x + self.proj(mixed.transpose(1, 2).reshape(b, n, c)))
        x = self.norm2(x + self.ffn(x))
        return x.transpose(1, 2).reshape(b, c, h, w).contiguous()

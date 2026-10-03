# InvDet-style target-aware information preservation for RT-DETR AIFI.
#
# Reference:
#   Target-Aware Invertible Encoder with Reconstruction Guidance for Infrared
#   Small Target Detection. CVPR 2026.
#
# This is a stable AIFI adaptation: the original AIFI residual path is kept
# unchanged, and a TARM/GCTM-guided detail branch is added with zero-init gate.

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['TargetAwareReconstructionModulation', 'GeometryContentToleranceMetric', 'AIFI_InvDet']


class TargetAwareReconstructionModulation(nn.Module):
    """TARM-style high-pass gating and mild low-pass gain.

    The module explicitly separates low-frequency structure from high-frequency
    residuals, which are often where tiny targets and edges survive.
    """

    def __init__(self, c1):
        super().__init__()
        self.high_gate = nn.Conv2d(c1, c1, kernel_size=3, padding=1)
        self.low_gain = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.high_gate.weight)
        nn.init.zeros_(self.high_gate.bias)

    def forward(self, x):
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = x - low
        high = high * torch.sigmoid(self.high_gate(low))
        low = low * (1.0 + self.low_gain)
        return high + low


class GeometryContentToleranceMetric(nn.Module):
    """GCTM-style geometry/content tolerance weighting.

    It uses local residual energy and content norm to softly weight the
    reconstruction/detail branch, avoiding domination by flat background.
    """

    def __init__(self, c1):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        geometry = (x - low).abs().mean(dim=1, keepdim=True)
        content = x.norm(dim=1, keepdim=True)
        geometry = geometry / geometry.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        content = content / content.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return torch.sigmoid(self.conv(torch.cat([geometry, content], dim=1)))


class AIFI_InvDet(nn.Module):
    """RT-DETR AIFI with InvDet-style target-aware detail preservation.

    The forward AIFI MHSA/FFN remains intact. A TARM + GCTM residual branch is
    added after the FFN with zero-init scale, so identity behavior is preserved
    at initialization.
    """

    def __init__(self, c1, cm=1024, num_heads=8, dropout=0.0,
                 act=nn.GELU(), normalize_before=False):
        super().__init__()
        if c1 % num_heads != 0:
            raise ValueError('c1 must be divisible by num_heads')
        self.c1 = c1
        self.normalize_before = normalize_before

        self.global_attn = nn.MultiheadAttention(c1, num_heads, dropout=dropout,
                                                 batch_first=True)
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act

        self.tarm = TargetAwareReconstructionModulation(c1)
        self.gctm = GeometryContentToleranceMetric(c1)
        self.detail_gamma = nn.Parameter(torch.zeros(1))

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, 'embed_dim must be divisible by 4'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature ** omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.cat([torch.sin(out_w), torch.cos(out_w),
                          torch.sin(out_h), torch.cos(out_h)], dim=1)[None]

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.c1:
            raise ValueError(f'AIFI_InvDet expected {self.c1} channels, got {c}')

        pos = self.build_2d_sincos_position_embedding(w, h, c).to(device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1)

        if self.normalize_before:
            src_norm = self.norm1(src)
            attn_out, _ = self.global_attn(src_norm + pos, src_norm + pos, src_norm,
                                           need_weights=False)
            src = src + self.dropout1(attn_out)
            src_norm = self.norm2(src)
            src = src + self.dropout2(self.fc2(self.dropout(self.act(self.fc1(src_norm)))))
        else:
            src_norm = src + pos
            attn_out, _ = self.global_attn(src_norm, src_norm, src, need_weights=False)
            src = src + self.dropout1(attn_out)
            src = self.norm1(src)
            src = src + self.dropout2(self.fc2(self.dropout(self.act(self.fc1(src)))))
            src = self.norm2(src)

        out = src.permute(0, 2, 1).reshape(b, c, h, w).contiguous()
        detail = self.tarm(out)
        weight = self.gctm(out)
        out = out + self.detail_gamma * detail * weight
        return out

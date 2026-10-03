# UPLiFT-style high-resolution guided detail rescue for RT-DETR AIFI.
#
# Reference:
#   UPLiFT: Efficient Pixel-Dense Feature Upsampling with Local Attenders
#   CVPR 2026.
#
# This adaptation keeps the original AIFI MHSA/FFN and adds a zero-init
# high-resolution guide branch:
#   guide(P3) -> LocalAttender -> high-res detail -> pool -> P5 residual

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LocalAttender', 'AIFI_UPLiFT']


def residual_auto(x1, x2):
    """Channel-wise truncation/padding residual, following UPLiFT."""
    s1 = x1.shape[1]
    s2 = x2.shape[1]
    if s1 == s2:
        return x1 + x2
    elif s1 > s2:
        s = x2.shape[1]
        return x1[:, :s, :, :] + x2
    else:
        s = x1.shape[1]
        pad = torch.zeros_like(x2)
        pad[:, :s, :, :] = x1
        return pad + x2


class LocalAttender(nn.Module):
    """Local attentional pooling from UPLiFT.

    The guide map produces a compact attention map; the value map is pooled
    inside a fixed local neighborhood. No global QK compression is used.
    """

    def __init__(self, in_channels, num_connected=5, conv_res=True):
        super().__init__()
        if num_connected not in (5, 9, 13, 17, 25):
            raise ValueError(f'LocalAttender invalid num_connected={num_connected}')
        self.in_channels = in_channels
        self.num_connected = num_connected
        if num_connected == 5:
            self.offsets = [(-1,0),(0,-1),(0,0),(0,1),(1,0)]
            self.pn = 1
        elif num_connected == 9:
            self.offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
            self.pn = 1
        elif num_connected == 13:
            self.offsets = [(-2,0),(-1,-1),(-1,0),(-1,1),(0,-2),(0,-1),(0,0),(0,1),(0,2),(1,-1),(1,0),(1,1),(2,0)]
            self.pn = 2
        elif num_connected == 17:
            self.offsets = [(-2,-2),(-2,0),(-2,2),(-1,-1),(-1,0),(-1,1),(0,-2),(0,-1),(0,0),(0,1),(0,2),(1,-1),(1,0),(1,1),(2,-2),(2,0),(2,2)]
            self.pn = 2
        elif num_connected == 25:
            self.offsets = [(-2,-2),(-2,-1),(-2,0),(-2,1),(-2,2),(-1,-2),(-1,-1),(-1,0),(-1,1),(-1,2),
                            (0,-2),(0,-1),(0,0),(0,1),(0,2),(1,-2),(1,-1),(1,0),(1,1),(1,2),
                            (2,-2),(2,-1),(2,0),(2,1),(2,2)]
            self.pn = 2
        self.conv1 = nn.Conv2d(in_channels, num_connected, kernel_size=1)
        self.conv_res = conv_res
        self.pad = nn.ReplicationPad2d(self.pn)

    def make_offsets(self, x):
        h = x.shape[2]
        w = x.shape[3]
        x = self.pad(x)
        out = []
        for oy, ox in self.offsets:
            y0 = self.pn + oy
            x0 = self.pn + ox
            out.append(x[:, :, y0:y0 + h, x0:x0 + w])
        return torch.stack(out, dim=2)  # B,C,D,H,W

    def forward(self, guide, value):
        """guide: B,Cg,Hg,Wg; value: B,Cv,Hv,Wv. Hg/Wg must be Hv/Wv * I."""
        att = self.conv1(guide)
        if self.conv_res:
            att = residual_auto(guide, att)
        d, h_out, w_out = att.shape[1:]
        i = h_out // value.shape[2]
        if i * value.shape[2] != h_out or i * value.shape[3] != w_out:
            raise ValueError('LocalAttender requires integer guide/value spatial ratio')

        b = value.shape[0]
        c = value.shape[1]
        h, w = value.shape[2], value.shape[3]
        value = self.make_offsets(value)
        value = value.unsqueeze(4).unsqueeze(6)

        att = F.softmax(att, dim=1)
        att = att.reshape(b, d, h, i, w, i).unsqueeze(1)
        out = value * att
        out = torch.sum(out, dim=2).reshape(b, c, h_out, w_out)
        return out


class AIFI_UPLiFT(nn.Module):
    """RT-DETR AIFI with a UPLiFT-style high-resolution detail branch.

    Inputs: ``[guide, value]``. The guide is usually P3 or P2 and the value is
    the P5 feature after the input projection. The original AIFI MHSA/FFN is
    preserved and the guide branch is zero-init gated.
    """

    def __init__(self, in_channels, embed_dim=256, cm=1024, num_heads=8,
                 num_connected=5, dropout=0.0, act=nn.GELU(),
                 normalize_before=False, conv_res=True, temperature=10000.0):
        super().__init__()
        if isinstance(in_channels, int):
            in_channels = [in_channels, embed_dim]
        if len(in_channels) != 2:
            raise ValueError('AIFI_UPLiFT expects [guide_channels, value_channels]')
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads')

        self.guide_channels, self.value_channels = map(int, in_channels)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.normalize_before = normalize_before
        self.temperature = temperature

        self.attender = LocalAttender(self.guide_channels, num_connected=num_connected,
                                      conv_res=conv_res)
        self.detail_proj = nn.Identity() if self.value_channels == embed_dim else \
            nn.Conv2d(self.value_channels, embed_dim, kernel_size=1)
        self.value_proj = nn.Identity() if self.value_channels == embed_dim else \
            nn.Conv2d(self.value_channels, embed_dim, kernel_size=1)
        self.detail_gamma = nn.Parameter(torch.zeros(1))

        self.global_attn = nn.MultiheadAttention(embed_dim, num_heads,
                                                 dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(embed_dim, cm)
        self.fc2 = nn.Linear(cm, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act

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

    def forward(self, inputs):
        if len(inputs) != 2:
            raise ValueError(f'AIFI_UPLiFT expects 2 inputs, got {len(inputs)}')
        guide, value = inputs
        if guide.shape[1] != self.guide_channels or value.shape[1] != self.value_channels:
            raise ValueError('AIFI_UPLiFT input channel mismatch')

        detail = self.attender(guide, value)
        detail = self.detail_proj(detail)
        detail = F.adaptive_avg_pool2d(detail, output_size=value.shape[-2:])
        value = self.value_proj(value)
        value = value + self.detail_gamma * detail

        b, c, h, w = value.shape
        pos = self.build_2d_sincos_position_embedding(w, h, c, self.temperature).to(
            device=value.device, dtype=value.dtype)
        src = value.flatten(2).permute(0, 2, 1)

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

        return src.permute(0, 2, 1).reshape(b, c, h, w).contiguous()

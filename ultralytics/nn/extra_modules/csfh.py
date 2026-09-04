# CSFH (Cross-Scale Feature Hybridization) Backbone from CSFPR-RTDETR.
#
# Self-contained module. No dependency on extra_modules/block.py.
#
# CSFH_Block = C2f_SFHF: C2f with SFHF_Block replacing Bottleneck
#   cv1(c1→2c, 1×1) → split(2 chunks) → [SFHF_Block(c)×n] → cat → cv2(→c2, 1×1)
#
# SFHF_Block internals:
#   - norm1 → SFHF_Mixer (local TokenMixer + global FourierUnit + channel attention) → β residual
#   - norm2 → SFHF_FFN (multi-scale DWConv) → γ residual
#
# Original paper backbone (CSFPR-RTDETR):
#   Conv(3→64, s=2) → Conv(64→128, s=2) → CSFH_Block(128)
#   → Conv(128→256, s=2) → CSFH_Block(256)
#   → Conv(256→384, s=2) → CSFH_Block(384)
#   → Conv(384→384, s=2) → CSFH_Block(384)×3

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..modules.block import C2f

__all__ = ['CSFH_Block']


class SFHF_FFN(nn.Module):
    def __init__(self, dim):
        super(SFHF_FFN, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=1,
                      groups=self.dim_sp),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=5, padding=2,
                      groups=self.dim_sp),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=7, padding=3,
                      groups=self.dim_sp),
        )
        self.gelu = nn.GELU()
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
        )

    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim_sp, dim=1))
        x[1] = self.conv1_1(x[1])
        x[2] = self.conv1_2(x[2])
        x[3] = self.conv1_3(x[3])
        x = torch.cat(x, dim=1)
        x = self.gelu(x)
        x = self.conv_fina(x)
        return x


class TokenMixer_For_Local(nn.Module):
    def __init__(self, dim):
        super(TokenMixer_For_Local, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2
        self.CDilated_1 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=1, dilation=1, groups=self.dim_sp)
        self.CDilated_2 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=2, dilation=2, groups=self.dim_sp)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        cd1 = self.CDilated_1(x1)
        cd2 = self.CDilated_2(x2)
        x = torch.cat([cd1, cd2], dim=1)
        return x


class SFHF_FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=4):
        super(SFHF_FourierUnit, self).__init__()
        self.groups = groups
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.fdc = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2 * self.groups,
                              kernel_size=1, stride=1, padding=0, groups=self.groups, bias=True)
        self.weight = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=self.groups, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1)
        )
        self.fpe = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3,
                              padding=1, stride=1, groups=in_channels * 2, bias=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        ffted = rearrange(ffted, 'b c h w d -> b (c d) h w').contiguous()
        ffted = self.bn(ffted)
        ffted = self.fpe(ffted) + ffted
        dy_weight = self.weight(ffted)
        ffted = self.fdc(ffted).view(batch, self.groups, 2 * c, h, -1)
        ffted = torch.einsum('ijkml,ijml->ikml', ffted, dy_weight)
        ffted = F.gelu(ffted)
        ffted = rearrange(ffted, 'b (c d) h w -> b c h w d', d=2).contiguous()
        ffted = torch.view_as_complex(ffted)
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')
        return output


class TokenMixer_For_Gloal(nn.Module):
    def __init__(self, dim):
        super(TokenMixer_For_Gloal, self).__init__()
        self.dim = dim
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU()
        )
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU()
        )
        self.FFC = SFHF_FourierUnit(self.dim * 2, self.dim * 2)

    def forward(self, x):
        x = self.conv_init(x)
        x0 = x
        x = self.FFC(x)
        x = self.conv_fina(x + x0)
        return x


class SFHF_Mixer(nn.Module):
    def __init__(self, dim, token_mixer_for_local=TokenMixer_For_Local,
                 token_mixer_for_gloal=TokenMixer_For_Gloal):
        super(SFHF_Mixer, self).__init__()
        self.dim = dim
        self.mixer_local = token_mixer_for_local(dim=self.dim)
        self.mixer_gloal = token_mixer_for_gloal(dim=self.dim)
        self.ca_conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * dim, 2 * dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * dim // 2, 2 * dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.gelu = nn.GELU()
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, 2 * dim, 1),
        )

    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim, dim=1))
        x_local = self.mixer_local(x[0])
        x_gloal = self.mixer_gloal(x[1])
        x = torch.cat([x_local, x_gloal], dim=1)
        x = self.gelu(x)
        x = self.ca(x) * x
        x = self.ca_conv(x)
        return x


class SFHF_Block(nn.Module):
    def __init__(self, dim, norm_layer=nn.BatchNorm2d, token_mixer=SFHF_Mixer):
        super(SFHF_Block, self).__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mixer = token_mixer(dim=self.dim)
        self.ffn = SFHF_FFN(dim=self.dim)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        copy = x
        x = self.norm1(x)
        x = self.mixer(x)
        x = x * self.beta + copy

        copy = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x * self.gamma + copy

        return x


class CSFH_Block(C2f):
    """CSFH Block: C2f with SFHF_Block replacing Bottleneck.

    Identical to the original CSFPR-RTDETR C2f_SFHF implementation:
      cv1(c1→2c, 1×1) → split(2 chunks) → [SFHF_Block(c)×n] → cat → cv2(→c2, 1×1)

    Args:
        c1: Input channels.
        c2: Output channels.
        n: Number of SFHF_Block repeats.
        shortcut: Whether to use shortcut connection in Bottleneck (unused here).
        g: Groups (unused, kept for C2f interface).
        e: Expansion ratio for hidden channels (default 0.5).
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(SFHF_Block(self.c) for _ in range(n))
# Ultralytics YOLO, AGPL-3.0 license
"""Original RSRT-DETR PDSF block adapted for the FSV7 P5 encoder slot.

Source: https://github.com/zhy1109/RSRT-DETR (ultralytics.zip,
ultralytics/nn/modules/transformer.py); DOI: 10.1109/TGRS.2026.3677535.
"""

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ('PDSF_Attention',)


class GlobalPolaAttention(nn.Module):
    def __init__(self, dim, hw, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 kernel_size=5, alpha=4):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.h = hw[0]
        self.w = hw[1]
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.qg = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size, groups=head_dim, padding=kernel_size // 2)
        self.power = nn.Parameter(torch.zeros(size=(1, self.num_heads, 1, self.head_dim)))
        self.alpha = alpha
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, (self.w * self.h) // (sr_ratio * sr_ratio), dim)))

    def forward(self, x):
        B, N, C = x.shape
        q, g = self.qg(x).reshape(B, N, 2, C).unbind(2)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, self.h, self.w)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        n = k.shape[1]
        k = k + self.positional_encoding
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        power = 1 + self.alpha * nn.functional.sigmoid(self.power)
        q = q / scale
        k = k / scale
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        q_pos = kernel_function(q) ** power
        q_neg = kernel_function(-q) ** power
        k_pos = kernel_function(k) ** power
        k_neg = kernel_function(-k) ** power
        q_sim = torch.cat([q_pos, q_neg],dim=-1)
        q_opp = torch.cat([q_neg, q_pos],dim=-1)
        k = torch.cat([k_pos, k_neg],dim=-1)
        v1,v2 = torch.chunk(v,2,dim=-1)
        z = 1 / (q_sim @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v1 * (n ** -0.5))
        x_sim = q_sim @ kv * z
        z = 1 / (q_opp @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v2 * (n ** -0.5))
        x_opp = q_opp @ kv * z
        x = torch.cat([x_sim, x_opp],dim=-1)
        x = x.transpose(1, 2).reshape(B, N, C)
        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v.transpose(-2, -1).reshape(B * self.num_heads, -1, n), size=N, mode='linear').reshape(B, self.num_heads, -1, N).transpose(-2, -1)
        v = v.reshape(B * self.num_heads, self.h, self.w, -1).permute(0, 3, 1, 2)
        v = self.dwc(v).reshape(B, C, N).permute(0, 2, 1)
        x = x + v
        x = x * g
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class sim_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(sim_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class opp_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(opp_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = sim_LayerNorm(dim)
        else:
            self.body = opp_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class subchannel(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(subchannel, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.fusion = nn.Conv2d(hidden_features + dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv_afterfusion = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3,stride=1,padding=1,bias=True), LayerNorm(dim, 'WithBias'), nn.ReLU(inplace=True), nn.Conv2d(dim, dim, kernel_size=3,stride=1,padding=1,bias=True), LayerNorm(dim, 'WithBias'), nn.ReLU(inplace=True))
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x, spatial):
        x = self.project_in(x)
        y = self.avg_pool(spatial)
        y = self.conv(y)
        y = self.upsample(y)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.fusion(torch.cat((x1, y),dim=1))
        x1 = self.dwconv_afterfusion(x1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)
        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)
        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity
        identity = x
        x = self.projector(x)
        return identity + x

class Mona(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.project1 = nn.Conv2d(in_dim, 64, 1)
        self.nonlinear = F.gelu
        self.project2 = nn.Conv2d(64, in_dim, 1)
        self.dropout = nn.Dropout(p=0.1)
        self.adapter_conv = MonaOp(64)
        self.norm = LayerNorm2d(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim, 1, 1) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim, 1, 1))


    def forward(self, x, hw_shapes=None):
        identity = x
        x = self.norm(x) * self.gamma + x * self.gammax
        project1 = self.project1(x)
        project1 = self.adapter_conv(project1)
        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)
        return identity + project2


class PDSF_Attention(nn.Module):
    """PDSF with the reference 20x20 positional grid (640px at stride 32)."""

    def __init__(self, c1, dropout=0.0):
        super().__init__()
        self.pola_attention = GlobalPolaAttention(c1, (20, 20))
        self.ffn = subchannel(c1, 2.0, False)
        self.norm1 = DynamicTanh(c1, channels_last=False)
        self.norm2 = DynamicTanh(c1, channels_last=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mona1 = Mona(c1)
        self.mona2 = Mona(c1)

    def forward(self, src):
        _, c, h, w = src.shape
        if (h, w) != (20, 20):
            raise ValueError(f'PDSF_Attention expects 20x20 P5 features (imgsz=640); got {h}x{w}.')
        tokens = src.flatten(2).permute(0, 2, 1)
        attended = self.pola_attention(tokens).permute(0, 2, 1).reshape(-1, c, h, w).contiguous()
        x = self.mona1(self.norm1(src + self.dropout1(attended)))
        x = x + self.dropout2(self.ffn(x, src))
        return self.mona2(self.norm2(x))

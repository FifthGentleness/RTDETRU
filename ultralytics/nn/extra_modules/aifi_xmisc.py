# X021/X024/X030 AIFI配置型变体 (批次3, [C]).
# X021 A-ADrop01: 注意力dropout 0->0.1(正则探针)
# X024 A-TEMP1e3: sincos位置编码温度 1e4->1e3(高频PE探针)
# X030 A-PreNorm: normalize_before=True(pre-norm拓扑, 原生开关从未启用)

import torch
import torch.nn as nn

from .aifi_xbase import AIFIXBase
from .aifi_xbase import AIFIXBase
from .aifi_xipc import AIFI_XIPC, _MHA2Arg

__all__ = ('AIFI_XADrop', 'AIFI_XTemp', 'AIFI_XPre')


def _make_aifi(c1, cm, num_heads, dropout, act, normalize_before):
    m = AIFI_XIPC.__new__(AIFI_XIPC)
    nn.Module.__init__(m)
    m.c1, m.normalize_before = c1, normalize_before
    m.attn = _MHA2Arg(c1, num_heads, dropout)
    m.fc1 = nn.Linear(c1, cm)
    m.fc2 = nn.Linear(cm, c1)
    m.norm1 = nn.LayerNorm(c1)
    m.norm2 = nn.LayerNorm(c1)
    m.dropout = nn.Dropout(dropout)
    m.dropout1 = nn.Dropout(dropout)
    m.dropout2 = nn.Dropout(dropout)
    m.act = act
    return m


class AIFI_XADrop(nn.Module):
    """X021: 注意力/FFN dropout=0.1。"""
    def __init__(self, c1, cm=1024, num_heads=8):
        super().__init__()
        self.inner = _make_aifi(c1, cm, num_heads, 0.1, nn.GELU(), False)

    def forward(self, x):
        return self.inner._forward_core(x)


class AIFI_XPre(nn.Module):
    """X030: pre-norm拓扑。"""
    def __init__(self, c1, cm=1024, num_heads=8):
        super().__init__()
        self.inner = _make_aifi(c1, cm, num_heads, 0.0, nn.GELU(), True)

    def forward(self, x):
        return self.inner._forward_core(x)


class AIFI_XTemp(nn.Module):
    """X024: sincos温度1e3(拷贝原生build并改温度)。"""
    def __init__(self, c1, cm=1024, num_heads=8, temperature=1000.0):
        super().__init__()
        self.inner = _make_aifi(c1, cm, num_heads, 0.0, nn.GELU(), False)
        self.temperature = temperature
        import types
        orig = self.inner._forward_core

        def core_with_temp(mod, x):
            b, c, h, w = x.shape
            from .aifi_block_sparse import AIFI_BlockSparse
            gw = torch.arange(int(w), dtype=torch.float32)
            gh = torch.arange(int(h), dtype=torch.float32)
            gw, gh = torch.meshgrid(gw, gh, indexing='ij')
            pd = c // 4
            omega = torch.arange(pd, dtype=torch.float32) / pd
            omega = 1. / (self.temperature ** omega)
            ow = gw.flatten()[..., None] @ omega[None]
            oh = gh.flatten()[..., None] @ omega[None]
            pos = torch.cat([torch.sin(ow), torch.cos(ow),
                             torch.sin(oh), torch.cos(oh)], 1)[None].to(x.device, x.dtype)
            tokens = x.flatten(2).permute(0, 2, 1)
            qk = tokens + pos
            attn = mod.attn
            src = tokens + mod.dropout1(attn(qk, tokens))
            src = mod.norm1(src)
            src = src + mod.dropout2(
                mod.fc2(mod.dropout(mod.act(mod.fc1(src)))))
            src = mod.norm2(src)
            return src.permute(0, 2, 1).view(b, c, h, w).contiguous()

        self._core = core_with_temp.__get__(self.inner, type(self.inner))

    def forward(self, x):
        return self._core(x)

# 批次8 AIFI机制 (X076/X079, [C]).
# X076 A-HeadTemp: 逐头可学习温度(logits/tau_h, init=1恒等)
# X079 A-FFNSiLU: FFN激活GELU->SiLU

import torch
import torch.nn as nn

from .aifi_xbase import AIFIXBase
from .aifi_xipc import _MHA2Arg

__all__ = ('AIFI_XHeadTemp', 'AIFI_XFFNSiLU')


def _bi(m, c1, cm, nh, act):
    m.c1, m.normalize_before = c1, False
    m.attn = _MHA2Arg(c1, nh, 0.0)
    m.fc1 = nn.Linear(c1, cm)
    m.fc2 = nn.Linear(cm, c1)
    m.norm1 = nn.LayerNorm(c1)
    m.norm2 = nn.LayerNorm(c1)
    m.dropout = nn.Dropout(0.0)
    m.dropout1 = nn.Dropout(0.0)
    m.dropout2 = nn.Dropout(0.0)
    m.act = act


class _HeadTempMHA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.inner = _MHA2Arg(dim, num_heads)
        self.tau = nn.Parameter(torch.ones(num_heads))

    def forward(self, qk, tokens):
        b, n, c = qk.shape
        h = self.tau.shape[0]
        d = c // h
        w = self.inner.inner
        q = qk @ w.in_proj_weight[:c].t() + w.in_proj_bias[:c]
        k = qk @ w.in_proj_weight[c:2 * c].t() + w.in_proj_bias[c:2 * c]
        v = tokens @ w.in_proj_weight[2 * c:].t() + w.in_proj_bias[2 * c:]
        q = (q / self.tau.view(1, 1, h, 1).expand(b, n, h, d).reshape(b, n, c))
        k = k  # 只缩q等价于logits/tau
        q = q.view(b, n, h, d).transpose(1, 2)
        k = k.view(b, n, h, d).transpose(1, 2)
        v = v.view(b, n, h, d).transpose(1, 2)
        att = ((q @ k.transpose(-1, -2)) * (d ** -0.5)).softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(b, n, c)
        return o @ w.out_proj.weight.t() + w.out_proj.bias


class AIFI_XHeadTemp(nn.Module):
    """X076: 逐头温度tau_h(init=1恒等)。+8参数。"""
    def __init__(self, c1, cm=1024, num_heads=8):
        super().__init__()
        import copy
        self.inner = AIFIXBase.__new__(AIFIXBase)
        nn.Module.__init__(self.inner)
        _bi(self.inner, c1, cm, num_heads, nn.GELU())
        self.inner.attn = _HeadTempMHA(c1, num_heads)

    def forward(self, x):
        return self.inner._forward_core(x)


class AIFI_XFFNSiLU(nn.Module):
    """X079: FFN激活SiLU(与X054解码器互补)。0新参。"""
    def __init__(self, c1, cm=1024, num_heads=8):
        super().__init__()
        self.inner = AIFIXBase.__new__(AIFIXBase)
        nn.Module.__init__(self.inner)
        _bi(self.inner, c1, cm, num_heads, nn.SiLU())

    def forward(self, x):
        return self.inner._forward_core(x)

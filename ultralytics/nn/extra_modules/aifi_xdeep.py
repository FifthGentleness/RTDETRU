# 批次7深水区AIFI机制 (X061/X064/X069, [C]为主).
# X061 A-DoRA1: DoRA加在fc1(第一层)——幅向解耦第二落点
# X064 A-AttnLN: 注意力输出后接额外LayerNorm(双norm探针, norm后仍走原残差)
# X069 A-GLUFFN: FFN隐层SiLU门控(SwiGLU形态; 风险: 创1 StarFFN同族——
#                仅供消融对照, 不建议作为论文主创新)

import torch
import torch.nn as nn

from .aifi_xbase import AIFIXBase
from .aifi_xipc import _MHA2Arg
from .aifi_xdora import DoRALinear

__all__ = ('AIFI_XDoRA1', 'AIFI_XAttnLN', 'AIFI_XGLUFFN')


def _base_init(m, c1, cm, num_heads, dropout, act, nb):
    m.c1, m.normalize_before = c1, nb
    m.attn = _MHA2Arg(c1, num_heads, dropout)
    m.fc1 = nn.Linear(c1, cm)
    m.fc2 = nn.Linear(cm, c1)
    m.norm1 = nn.LayerNorm(c1)
    m.norm2 = nn.LayerNorm(c1)
    m.dropout = nn.Dropout(dropout)
    m.dropout1 = nn.Dropout(dropout)
    m.dropout2 = nn.Dropout(dropout)
    m.act = act


class AIFI_XDoRA1(AIFIXBase):
    """X061: fc1(升维层)DoRA幅向解耦。"""
    def __init__(self, c1, cm=1024, num_heads=8, dropout=0.0, act=nn.GELU(),
                 normalize_before=False):
        super().__init__()
        _base_init(self, c1, cm, num_heads, dropout, act, normalize_before)
        self.fc1 = DoRALinear(nn.Linear(c1, cm))

    def forward(self, x):
        return self._forward_core(x)


class AIFI_XAttnLN(AIFIXBase):
    """X064: 注意力输出先过额外LN再进残差(norm探针)。"""
    def __init__(self, c1, cm=1024, num_heads=8, dropout=0.0, act=nn.GELU(),
                 normalize_before=False):
        super().__init__()
        _base_init(self, c1, cm, num_heads, dropout, act, normalize_before)
        self.attn_ln = nn.LayerNorm(c1)

    def forward(self, x):
        import torch.nn.functional as F
        b, c, h, w = x.shape
        from .aifi_block_sparse import AIFI_BlockSparse
        pos = AIFI_BlockSparse.build_2d_sincos_position_embedding(w, h, c).to(x.device, x.dtype)
        tokens = x.flatten(2).permute(0, 2, 1)
        qk = tokens + pos
        src = tokens + self.dropout1(self.attn_ln(self.attn(qk, tokens)))
        src = self.norm1(src)
        src = src + self.dropout2(self.fc2(self.dropout(self.act(self.fc1(src)))))
        return self.norm2(src).permute(0, 2, 1).view(b, c, h, w).contiguous()


class AIFI_XGLUFFN(AIFIXBase):
    """X069: FFN隐层SiLU门控(hidden分裂两半, SwiGLU形态; 消融对照用)。"""
    def __init__(self, c1, cm=1024, num_heads=8, dropout=0.0, act=nn.GELU(),
                 normalize_before=False):
        super().__init__()
        _base_init(self, c1, cm, num_heads, dropout, act, normalize_before)
        self.gate1 = nn.Linear(c1, cm // 2)

    def forward(self, x):
        import torch.nn.functional as F
        b, c, h, w = x.shape
        from .aifi_block_sparse import AIFI_BlockSparse
        pos = AIFI_BlockSparse.build_2d_sincos_position_embedding(w, h, c).to(x.device, x.dtype)
        tokens = x.flatten(2).permute(0, 2, 1)
        qk = tokens + pos
        src = tokens + self.dropout1(self.attn(qk, tokens))
        src = self.norm1(src)
        hid = self.act(self.fc1(src))
        gate = torch.nn.functional.silu(self.gate1(src))
        hh = hid[..., :hid.shape[-1] // 2] * gate
        src = src + self.dropout2(self.fc2(torch.cat([hh, hid[..., hid.shape[-1] // 2:]], -1)))
        return self.norm2(src).permute(0, 2, 1).view(b, c, h, w).contiguous()

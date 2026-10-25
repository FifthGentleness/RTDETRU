# X033/X034/X040 AIFI正则与消融探针 (批次4, [C]).
# X033 A-DP01: AIFI残差DropPath 0.1(训练期, eval恒等)
# X034 A-FDrop: FFN隐层dropout 0.1
# X040 A-PEOff: 无位置编码探针(诊断: S5上PE到底值多少; pos=None)

import torch
import torch.nn as nn

from .aifi_xbase import AIFIXBase
from .aifi_xipc import _MHA2Arg
from .decoder_innov_common import DropPath

__all__ = ('AIFI_XDP01', 'AIFI_XFDrop', 'AIFI_XPEOff')


class AIFI_XDP01(nn.Module):
    """X033: 注意力与FFN两处残差的DropPath(0.1)。"""
    def __init__(self, c1, cm=1024, num_heads=8, dp=0.1):
        super().__init__()
        self.inner = AIFIXBase.__new__(AIFIXBase)
        nn.Module.__init__(self.inner)
        from .aifi_xipc import AIFI_XIPC
        self.inner.c1, self.inner.normalize_before = c1, False
        self.inner.attn = _MHA2Arg(c1, num_heads, 0.0)
        self.inner.fc1 = nn.Linear(c1, cm)
        self.inner.fc2 = nn.Linear(cm, c1)
        self.inner.norm1 = nn.LayerNorm(c1)
        self.inner.norm2 = nn.LayerNorm(c1)
        self.inner.dropout = nn.Dropout(0.0)
        self.inner.dropout1 = nn.Dropout(0.0)
        self.inner.dropout2 = nn.Dropout(0.0)
        self.inner.act = nn.GELU()
        self.dp = DropPath(dp)

    def forward(self, x):
        import torch.nn.functional as F
        b, c, h, w = x.shape
        from .aifi_block_sparse import AIFI_BlockSparse
        pos = AIFI_BlockSparse.build_2d_sincos_position_embedding(w, h, c).to(x.device, x.dtype)
        tokens = x.flatten(2).permute(0, 2, 1)
        qk = tokens + pos
        m = self.inner
        a = m.attn(qk, tokens)
        src = tokens + self.dp(a)
        src = m.norm1(src)
        f = m.fc2(m.dropout(m.act(m.fc1(src))))
        src = m.norm1(src + self.dp(f)) if False else m.norm2(src + self.dp(f))
        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()


class AIFI_XFDrop(nn.Module):
    """X034: FFN隐层dropout 0.1(注意力dropout保持0)。"""
    def __init__(self, c1, cm=1024, num_heads=8):
        super().__init__()
        self.inner = AIFIXBase.__new__(AIFIXBase)
        nn.Module.__init__(self.inner)
        self.inner.c1, self.inner.normalize_before = c1, False
        self.inner.attn = _MHA2Arg(c1, num_heads, 0.0)
        self.inner.fc1 = nn.Linear(c1, cm)
        self.inner.fc2 = nn.Linear(cm, c1)
        self.inner.norm1 = nn.LayerNorm(c1)
        self.inner.norm2 = nn.LayerNorm(c1)
        self.inner.dropout = nn.Dropout(0.1)        # 仅FFN隐层
        self.inner.dropout1 = nn.Dropout(0.0)
        self.inner.dropout2 = nn.Dropout(0.0)
        self.inner.act = nn.GELU()

    def forward(self, x):
        return self.inner._forward_core(x)


class AIFI_XPEOff(nn.Module):
    """X040: 无位置编码(pos=None)探针。"""
    def __init__(self, c1, cm=1024, num_heads=8):
        super().__init__()
        self.inner = AIFIXBase.__new__(AIFIXBase)
        nn.Module.__init__(self.inner)
        self.inner.c1, self.inner.normalize_before = c1, False
        self.inner.attn = _MHA2Arg(c1, num_heads, 0.0)
        self.inner.fc1 = nn.Linear(c1, cm)
        self.inner.fc2 = nn.Linear(cm, c1)
        self.inner.norm1 = nn.LayerNorm(c1)
        self.inner.norm2 = nn.LayerNorm(c1)
        self.inner.dropout = nn.Dropout(0.0)
        self.inner.dropout1 = nn.Dropout(0.0)
        self.inner.dropout2 = nn.Dropout(0.0)
        self.inner.act = nn.GELU()

    def forward(self, x):
        import torch.nn.functional as F
        b, c, h, w = x.shape
        tokens = x.flatten(2).permute(0, 2, 1)
        m = self.inner
        src = tokens + m.dropout1(m.attn(tokens, tokens))   # pos=None
        src = m.norm1(src)
        src = src + m.dropout2(m.fc2(m.dropout(m.act(m.fc1(src)))))
        return m.norm2(src).permute(0, 2, 1).view(b, c, h, w).contiguous()

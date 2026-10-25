# AIFI 变体库共享基座(第三创新点方案库).
# 约定与仓库标准AIFI(TransformerEncoderLayer)严格一致:
#   pos只进q/k; value与残差基址均无pos(2026-10会话确认的基线约定)。
# 子类提供 self.attn(qk, tokens)->[B,N,C] 与标准属性即可。

import torch
import torch.nn as nn

from .aifi_block_sparse import AIFI_BlockSparse

__all__ = ['AIFIXBase']


class AIFIXBase(nn.Module):
    """标准AIFI骨架: 子类只需实现 __init__(建attn/fc/norm) 并复用forward。"""

    def _forward_core(self, x):
        b, c, h, w = x.shape
        if c != self.c1:
            raise ValueError(f'{type(self).__name__} expected {self.c1} channels, got {c}')
        pos = AIFI_BlockSparse.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        tokens = x.flatten(2).permute(0, 2, 1)
        qk = tokens + pos
        if self.normalize_before:
            q = self.norm1(tokens) + pos
            src = tokens + self.dropout1(self.attn(q, tokens))
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(self.norm2(src))))))
        else:
            src = tokens + self.dropout1(self.attn(qk, tokens))
            src = self.norm1(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src)))))
            src = self.norm2(src)
        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()

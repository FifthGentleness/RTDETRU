# CSFH (Cross-Scale Feature Hybridization) Backbone from CSFPR-RTDETR.
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

import torch.nn as nn

from ..modules.block import C2f
from .block import SFHF_Block

__all__ = ['CSFH_Block']


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
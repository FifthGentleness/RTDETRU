"""Query-conditioned head gating; inspired by Gated Attention (NeurIPS 2025).
The 2*sigmoid gate is an identity-initialized adaptation, not the paper's gate.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.transformer import AIFI
from .aifi_block_sparse import AIFI_BlockSparse

__all__ = ['AIFI_QHGate', 'AIFI_BlockSparseQHGate']


class AIFI_QHGate(AIFI):
    def __init__(self, c1, cm=1024, num_heads=8, dropout=0., act=nn.GELU(), normalize_before=False):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        self.gate_norm = nn.LayerNorm(c1, elementwise_affine=False)
        self.head_gate = nn.Linear(c1, num_heads)
        nn.init.zeros_(self.head_gate.weight)
        nn.init.zeros_(self.head_gate.bias)

    def gated_attention(self, src, pos):
        # Keep stock AIFI parameter names for checkpoint transfer. Position only enters Q/K.
        c, heads = src.shape[-1], self.ma.num_heads
        qk = src + pos
        weight, bias = self.ma.in_proj_weight, self.ma.in_proj_bias
        q = F.linear(qk, weight[:c], bias[:c])
        k = F.linear(qk, weight[c:2*c], bias[c:2*c])
        v = F.linear(src, weight[2*c:], bias[2*c:])
        def split(t):
            return t.reshape(t.shape[0], t.shape[1], heads, c // heads).transpose(1, 2)
        q, k, v = map(split, (q, k, v))
        attn = ((q * (c // heads) ** -0.5) @ k.transpose(-2, -1)).softmax(-1)
        attn = F.dropout(attn, self.ma.dropout, self.training)
        out = attn @ v
        gate = 2 * self.head_gate(self.gate_norm(src)).sigmoid()
        out = out * gate.transpose(1, 2).unsqueeze(-1)
        out = out.transpose(1, 2).reshape_as(src)
        return self.ma.out_proj(out)

    def forward(self, x):
        b, c, h, w = x.shape
        src = x.flatten(2).transpose(1, 2)
        pos = self.build_2d_sincos_position_embedding(w, h, c).to(x)
        if self.normalize_before:
            src = src + self.dropout1(self.gated_attention(self.norm1(src), pos))
            src = src + self.dropout2(self.fc2(self.dropout(self.act(self.fc1(self.norm2(src))))))
        else:
            src = self.norm1(src + self.dropout1(self.gated_attention(src, pos)))
            src = self.norm2(src + self.dropout2(self.fc2(self.dropout(self.act(self.fc1(src))))))
        return src.transpose(1, 2).reshape(b, c, h, w).contiguous()


class AIFI_BlockSparseQHGate(AIFI_BlockSparse):
    """Keep the existing sparse routing/position convention; gate before output projection."""
    def __init__(self, c1, cm=1024, num_heads=8, block_size=4, topk=4, **kwargs):
        super().__init__(c1, cm, num_heads, block_size, topk, **kwargs)
        # Wrap only the projection, whose input is concatenated per-head SDPA output.
        self.attn.proj = _GatedProjection(self.attn.proj, c1, num_heads)

    def forward(self, x):
        self.attn.proj.content = x.flatten(2).transpose(1, 2)
        try:
            return super().forward(x)
        finally:
            self.attn.proj.content = None


class _GatedProjection(nn.Module):
    def __init__(self, proj, dim, heads):
        super().__init__()
        self.proj, self.heads = proj, heads
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.gate = nn.Linear(dim, heads)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.content = None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Accept weights from the unwrapped BlockSparse projection as well.
        for name in ('weight', 'bias'):
            old, new = prefix + name, prefix + 'proj.' + name
            if old in state_dict and new not in state_dict:
                state_dict[new] = state_dict.pop(old)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, out):
        gate = 2 * self.gate(self.norm(self.content)).sigmoid()
        b, n, c = out.shape
        return self.proj((out.reshape(b, n, self.heads, c // self.heads) * gate.unsqueeze(-1)).reshape_as(out))

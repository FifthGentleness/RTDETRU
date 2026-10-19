# RA-AIFI: Retrieval-Augmented AIFI with cross-image token memory
# (AIFI_RetrievalAugmented).
#
# Problem addressed: every module in this network (including DSAWACGAv5,
# P2SPDOKMFSV7 and all previous AIFI variants) only sees the CURRENT image.
# Dataset-level knowledge of what rare/small objects look like can never be
# accumulated.
#
# Mechanism (RAT: Retrieval-Augmented Transformer, NeurIPS 2024):
#   1. A non-parametric FIFO memory bank stores detached P5 token features
#      (position-FREE content features, so matching is content-based) across
#      batches (MoCo-style queue, sampled tokens per image).
#   2. Each token retrieves its top-k most similar memory entries (token-level
#      retrieval with detached scores) and attends to them with the shared
#      qkv/proj weights (zero new attention parameters).
#   3. The retrieved context is added to the in-image attention output through
#      a single zero-initialized gate (repo warm-up philosophy: the branch
#      starts at 0 but receives gradient immediately).
#
# Zero-overlap: pure attention + queue. No wavelet/frequency, no conv branches,
# no linear attention, no block routing.
#
# Reference: RAT (NeurIPS 2024).

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AIFI_RetrievalAugmented']


class RetrievalAugmentedAttention(nn.Module):
    """Full MHSA over image tokens + top-k retrieval from a cross-image
    token memory bank."""

    def __init__(self, dim, num_heads=8, dropout=0.0, mem_size=8192,
                 num_sampled=64, retrieve_k=64, min_fill=256):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f'dim ({dim}) must be divisible by num_heads ({num_heads})')
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.mem_size = int(mem_size)
        self.num_sampled = int(num_sampled)
        self.retrieve_k = int(retrieve_k)
        self.min_fill = int(min_fill)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Cross-image memory queue (content features, detached by design).
        self.register_buffer('memory', torch.zeros(self.mem_size, dim),
                             persistent=True)
        self.register_buffer('mem_ptr', torch.zeros(1, dtype=torch.long),
                             persistent=True)
        self.register_buffer('mem_count', torch.zeros(1, dtype=torch.long),
                             persistent=True)

        # Single zero-init gate on the retrieved-context branch.
        self.ret_gate = nn.Parameter(torch.zeros(1))

    def forward(self, src, content, hw=None):
        """src: pos-embedded tokens (b, n, c); content: pos-free tokens."""
        b, n, c = src.shape
        qkv = self.qkv(src).chunk(3, dim=-1)
        q, k, v = [t.view(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                   for t in qkv]
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(logits.softmax(dim=-1))
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, c)
        out = self.proj_drop(self.proj(out))

        # ---- retrieval branch (training fills the queue; inference reuses) ----
        count = int(self.mem_count.item())
        ret_out = 0.0
        if count >= min(self.min_fill, self.retrieve_k):
            k_ret = min(self.retrieve_k, count)
            mem = self.memory[:count]                       # (m, c)
            # Token-level retrieval scores (detached: top-k is discrete).
            with torch.no_grad():
                scores = content.detach() @ mem.t()         # (b, n, m)
                idx = scores.topk(k_ret, dim=-1).indices    # (b, n, k)
            gathered = mem[idx]                             # (b, n, k, c)
            gn = gathered.reshape(-1, c)
            gqkv = self.qkv(gn).chunk(3, dim=-1)
            gk, gv = gqkv[1], gqkv[2]
            gk = gk.view(b, n, k_ret, self.num_heads, self.head_dim) \
                   .permute(0, 3, 1, 2, 4)                  # (b, h, n, k, d)
            gv = gv.view(b, n, k_ret, self.num_heads, self.head_dim) \
                   .permute(0, 3, 1, 2, 4)
            lm = torch.einsum('bhnd,bhnkd->bhnk', q, gk) * self.scale
            am = self.attn_drop(lm.softmax(dim=-1))
            om = torch.einsum('bhnk,bhnkd->bhnd', am, gv)
            om = om.permute(0, 2, 1, 3).reshape(b, n, c)
            ret_out = self.proj_drop(self.proj(om))

        if self.training:
            self._enqueue(content)
        return out + self.ret_gate * ret_out

    @torch.no_grad()
    def _enqueue(self, content):
        """FIFO-enqueue a random sample of detached content tokens."""
        b, n, c = content.shape
        s = min(self.num_sampled, n)
        idx = torch.rand(b, n, device=content.device).topk(s, dim=1).indices
        feats = content.detach()[torch.arange(b, device=content.device).unsqueeze(1), idx]
        feats = feats.reshape(-1, c)
        m = feats.shape[0]
        m = min(m, self.mem_size)
        ptr = int(self.mem_ptr.item())
        end = ptr + m
        if end <= self.mem_size:
            self.memory[ptr:end] = feats
        else:
            first = self.mem_size - ptr
            self.memory[ptr:] = feats[:first]
            self.memory[:end - self.mem_size] = feats[first:]
        self.mem_ptr[0] = end % self.mem_size
        self.mem_count[0] = torch.clamp(self.mem_count + m, max=self.mem_size)[0]


class AIFI_RetrievalAugmented(nn.Module):
    """AIFI augmented with cross-image retrieval memory.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads.
        mem_size: Capacity of the token memory bank.
        num_sampled: Tokens sampled per image per iteration for the queue.
        retrieve_k: Memory entries retrieved per token.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=2048, num_heads=8, mem_size=8192,
                 num_sampled=64, retrieve_k=64, dropout=0.0, act=nn.GELU(),
                 normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = RetrievalAugmentedAttention(
            dim=c1, num_heads=num_heads, dropout=dropout, mem_size=mem_size,
            num_sampled=num_sampled, retrieve_k=retrieve_k)
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256,
                                           temperature=10000.0):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sincos embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.cat([torch.sin(out_w), torch.cos(out_w),
                          torch.sin(out_h), torch.cos(out_h)], dim=1)[None]

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.c1:
            raise ValueError(
                f'AIFI_RetrievalAugmented expected {self.c1} channels, got {c}')

        pos = self.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        content = x.flatten(2).permute(0, 2, 1)
        src = content + pos

        if self.normalize_before:
            src = src + self.dropout1(self.attn(self.norm1(src), content))
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(self.norm2(src))))))
        else:
            src = src + self.dropout1(self.attn(src, content))
            src = self.norm1(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src)))))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()

    def extra_repr(self):
        return (f'c1={self.c1}, num_heads={self.attn.num_heads}, '
                f'mem_size={self.attn.mem_size}, '
                f'retrieve_k={self.attn.retrieve_k}')

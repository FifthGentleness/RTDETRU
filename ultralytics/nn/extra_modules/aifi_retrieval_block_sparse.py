# RBSv2-AIFI: Retrieval-augmented + Block-sparse fused AIFI
# (AIFI_RetrievalBlockSparse).
#
# Fuses two orthogonal AIFI improvements:
#   1. Block-sparse token routing (BlockSparse, sparse-vggt CVPR 2026):
#      each image-token query block attends to its own block plus its top-k
#      related key blocks - global receptive field at sparse-block cost.
#   2. Cross-image retrieval memory (RA-AIFI; RAT, NeurIPS 2024): a
#      non-parametric FIFO memory bank stores detached position-FREE content
#      tokens across batches; each token retrieves its top-k most similar
#      memory entries and cross-attends to them with the SHARED qkv/proj
#      weights (zero new attention parameters); the retrieved context is
#      added through a single zero-initialized gate.
#
# Fusion design:
#   - In-image attention core is block-sparse (BlockSparse unchanged).
#   - The retrieval branch is PER-TOKEN top-k over the memory bank and is
#     inherently orthogonal to block routing: it does not replace any
#     in-image key, it augments every token (sparsely attended or not) with
#     dataset-level context. Small/duplicate-prone tokens therefore recover
#     both (a) routed in-image context and (b) cross-image similar evidence.
#   - The retrieval branch keeps the zero-init gate: training starts exactly
#     at the BlockSparse baseline behavior.
#
# Zero-overlap: pure attention + queue, no wavelet/frequency, no conv
# branches, no register tokens, no linear attention.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AIFI_RetrievalBlockSparse']


class RetrievalBlockSparseAttention(nn.Module):
    """Block-sparse in-image attention + per-token cross-image retrieval."""

    def __init__(self, dim, num_heads=8, block_size=4, topk=4,
                 mem_size=8192, num_sampled=64, retrieve_k=64,
                 min_fill=256, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f'dim ({dim}) must be divisible by num_heads ({num_heads})')
        if block_size < 1 or topk < 1:
            raise ValueError('block_size and topk must be >= 1')

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.block_size = int(block_size)
        self.topk = int(topk)
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

    def _sparse_attention(self, x):
        """Block-sparse top-k routed attention (BlockSparse core)."""
        b, n, c = x.shape
        h, d = self.num_heads, self.head_dim
        bs = self.block_size
        num_blocks = math.ceil(n / bs)
        pad_len = num_blocks * bs - n
        npad = num_blocks * bs

        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, npad, h, d).permute(0, 2, 1, 3) for t in qkv]

        # Block routing via mean-pooled block descriptors.
        q_blk = x.view(b, num_blocks, bs, c).mean(dim=2)
        k_blk = q_blk
        block_logits = torch.matmul(q_blk, k_blk.transpose(-2, -1)) \
            * self.scale
        topk = min(self.topk, num_blocks)
        topk_idx = block_logits.topk(k=topk, dim=-1).indices

        block_mask = torch.zeros(
            b, num_blocks, num_blocks, dtype=torch.bool, device=x.device)
        block_mask.scatter_(-1, topk_idx, True)
        eye = torch.eye(num_blocks, dtype=torch.bool, device=x.device)
        block_mask = block_mask | eye.unsqueeze(0)

        token_mask = block_mask.repeat_interleave(bs, dim=1)
        token_mask = token_mask.repeat_interleave(bs, dim=2)
        token_mask = token_mask[:, :npad, :npad]
        if pad_len:
            valid = torch.zeros(b, npad, dtype=torch.bool, device=x.device)
            valid[:, :n] = True
            token_mask = token_mask & valid[:, None, :]

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        logits = logits.masked_fill(~token_mask.unsqueeze(1), -1e4)
        attn = self.attn_drop(logits.softmax(dim=-1))
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, npad, c)
        if pad_len:
            out = out[:, :n, :]
        return out

    def _retrieval(self, content):
        """Per-token top-k cross-attention to the memory bank."""
        count = int(self.mem_count.item())
        if count < min(self.min_fill, self.retrieve_k):
            return 0.0
        b, n, c = content.shape
        h, d = self.num_heads, self.head_dim
        k_ret = min(self.retrieve_k, count)
        mem = self.memory[:count]                          # (m, c)
        with torch.no_grad():
            scores = content.detach() @ mem.t()            # (b, n, m)
            idx = scores.topk(k_ret, dim=-1).indices       # (b, n, k)
        gathered = mem[idx]                                # (b, n, k, c)

        qkv = self.qkv(content).chunk(3, dim=-1)
        q = qkv[0].view(b, n, h, d).permute(0, 2, 1, 3)    # (b, h, n, d)
        gn = gathered.reshape(-1, c)
        gk, gv = self.qkv(gn).chunk(3, dim=-1)[1:]
        gk = gk.view(b, n, k_ret, h, d).permute(0, 3, 1, 2, 4)  # (b,h,n,k,d)
        gv = gv.view(b, n, k_ret, h, d).permute(0, 3, 1, 2, 4)
        lm = torch.einsum('bhnd,bhnkd->bhnk', q, gk) * self.scale
        am = self.attn_drop(lm.softmax(dim=-1))
        om = torch.einsum('bhnk,bhnkd->bhnd', am, gv)
        om = om.permute(0, 2, 1, 3).reshape(b, n, c)
        return self.proj_drop(self.proj(om))

    def forward(self, src, content):
        """src: pos-embedded tokens (b, n, c); content: pos-free tokens."""
        b, n, c = src.shape
        out = self.proj_drop(self.proj(
            self._sparse_attention(src).reshape(b, n, c)))
        ret_out = self._retrieval(content)
        if self.training:
            self._enqueue(content)
        return out + self.ret_gate * ret_out

    @torch.no_grad()
    def _enqueue(self, content):
        """FIFO-enqueue a random sample of detached content tokens."""
        b, n, c = content.shape
        s = min(self.num_sampled, n)
        idx = torch.rand(b, n, device=content.device).topk(s, dim=1).indices
        feats = content.detach()[
            torch.arange(b, device=content.device).unsqueeze(1), idx]
        feats = feats.reshape(-1, c)
        m = min(feats.shape[0], self.mem_size)
        ptr = int(self.mem_ptr.item())
        end = ptr + m
        if end <= self.mem_size:
            self.memory[ptr:end] = feats[:m]
        else:
            first = self.mem_size - ptr
            self.memory[ptr:] = feats[:first]
            self.memory[:end - self.mem_size] = feats[first:]
        self.mem_ptr[0] = end % self.mem_size
        self.mem_count[0] = torch.clamp(self.mem_count + m,
                                        max=self.mem_size)[0]


class AIFI_RetrievalBlockSparse(nn.Module):
    """AIFI with block-sparse in-image attention + cross-image retrieval.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads.
        block_size: Number of tokens per sparse block.
        topk: Number of key blocks attended by each query block.
        mem_size: Capacity of the token memory bank.
        num_sampled: Tokens sampled per image per iteration for the queue.
        retrieve_k: Memory entries retrieved per token.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=2048, num_heads=8, block_size=4, topk=4,
                 mem_size=8192, num_sampled=64, retrieve_k=64,
                 min_fill=256, dropout=0.0, act=nn.GELU(),
                 normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = RetrievalBlockSparseAttention(
            dim=c1, num_heads=num_heads, block_size=block_size, topk=topk,
            mem_size=mem_size, num_sampled=num_sampled,
            retrieve_k=retrieve_k, min_fill=min_fill, dropout=dropout)
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
                f'AIFI_RetrievalBlockSparse expected {self.c1} channels, got {c}')

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
                f'block_size={self.attn.block_size}, topk={self.attn.topk}, '
                f'mem_size={self.attn.mem_size}, '
                f'retrieve_k={self.attn.retrieve_k}')

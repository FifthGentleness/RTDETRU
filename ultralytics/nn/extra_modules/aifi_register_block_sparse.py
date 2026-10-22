# RBS-AIFI: Register-memory + Block-sparse fused AIFI
# (AIFI_RegisterBlockSparse).
#
# Fuses two orthogonal AIFI improvements:
#   1. Block-sparse token routing (BlockSparse, sparse-vggt CVPR 2026):
#      each image-token query block attends to its own block plus its top-k
#      related key blocks - global receptive field at sparse-block cost.
#   2. Amodal register tokens with task-semantic grounding (RM-AIFI;
#      Registers, ICLR 2024): M position-free learnable tokens absorb global
#      context and are supervised (training-only) by an image-level class-
#      presence BCE built from batch GT labels.
#
# Key fusion design - registers are GLOBAL HUBS, exempt from block routing:
#   - Register query rows attend densely to ALL image tokens (otherwise the
#     'global memory' would only see a sparse subset, defeating its purpose).
#   - All image-token query blocks attend densely to ALL register columns
#     (registers act as a always-visible global relay, restoring the exact
#     global path that block sparsity removes for far-apart blocks).
#   - Image<->image attention keeps the BlockSparse top-k block mask.
# So the attention is 'block-sparse core + dense register hub': global
# information travels image-block -> registers -> any other image-block in
# two hops, while the quadratic core stays sparse.
#
# Training-only grounding loss (zero inference cost): pooled register output
# is classified against the image-level class-presence vector (multi-label
# BCE), collected by DetectionModel.loss via compute_ground_loss.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AIFI_RegisterBlockSparse']


class RegisterBlockSparseAttention(nn.Module):
    """Block-sparse attention with dense amodal register hub tokens."""

    def __init__(self, dim, num_heads=8, block_size=4, topk=4,
                 num_registers=16, nc=10, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f'dim ({dim}) must be divisible by num_heads ({num_heads})')
        if block_size < 1:
            raise ValueError('block_size must be >= 1')
        if topk < 1:
            raise ValueError('topk must be >= 1')

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.block_size = int(block_size)
        self.topk = int(topk)
        self.num_registers = int(num_registers)

        # Amodal memory tokens (position-free by design).
        self.registers = nn.Parameter(
            torch.randn(num_registers, dim) * 0.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Task-semantic grounding head (training-only supervision target).
        self.ground_head = nn.Linear(dim, int(nc))

        # Transient training state consumed by compute_ground_loss.
        self._reg_feats = None

    def __deepcopy__(self, memo):
        """Drop stale graph-carrying cache tensors (ModelEMA deepcopy)."""
        import copy as _copy

        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k == '_reg_feats':
                new._reg_feats = None
                continue
            new.__dict__[k] = _copy.deepcopy(v, memo)
        return new

    def forward(self, x):
        b, n, c = x.shape
        h, d = self.num_heads, self.head_dim
        bs = self.block_size
        m = self.num_registers

        num_blocks = math.ceil(n / bs)
        pad_len = num_blocks * bs - n
        npad = num_blocks * bs

        reg = self.registers.unsqueeze(0).expand(b, -1, -1)     # (b, m, c)
        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))
        seq = torch.cat([reg, x], dim=1)                        # (b, m+npad, c)
        length = m + npad

        qkv = self.qkv(seq).chunk(3, dim=-1)
        q, k, v = [t.view(b, length, h, d).permute(0, 2, 1, 3)
                   for t in qkv]

        # ---- block routing among image tokens (same as BlockSparse) ----
        img = seq[:, m:]                                        # (b, npad, c)
        q_blk = img.view(b, num_blocks, bs, c).mean(dim=2)
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
        token_mask = token_mask.repeat_interleave(bs, dim=2)    # (b,npad,npad)

        # ---- fuse the dense register hub ----
        full_mask = torch.zeros(b, length, length, dtype=torch.bool,
                                device=x.device)
        full_mask[:, m:, m:] = token_mask                       # image core
        full_mask[:, :m, :] = True                              # reg queries: dense
        full_mask[:, :, :m] = True                              # reg keys: always visible

        # Padded image positions are disabled as keys.
        if pad_len:
            valid = torch.zeros(b, length, dtype=torch.bool, device=x.device)
            valid[:, :m + n] = True
            full_mask = full_mask & valid[:, None, :]

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        logits = logits.masked_fill(~full_mask.unsqueeze(1), -1e4)
        attn = self.attn_drop(logits.softmax(dim=-1))
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, length, c)

        reg_out, img_out = out[:, :m], out[:, m:m + n]
        if self.training and torch.is_grad_enabled():
            self._reg_feats = reg_out                           # (b, m, c)
        return self.proj_drop(self.proj(img_out))

    def compute_ground_loss(self, targets, batch=None):
        """Multi-label BCE: registers predict image-level class presence."""
        if not self.training or self._reg_feats is None:
            return None
        pooled = self._reg_feats.mean(dim=1)                    # (b, c)
        logits = self.ground_head(pooled)                       # (b, nc)
        b = logits.shape[0]
        nc = logits.shape[1]
        labels = torch.zeros(b, nc, device=logits.device)
        cls = targets['cls'].long()
        bidx = targets['batch_idx'].long()
        valid = (cls >= 0) & (cls < nc)
        labels[bidx[valid], cls[valid]] = 1.0
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        self._reg_feats = None                                  # consume
        return loss


class AIFI_RegisterBlockSparse(nn.Module):
    """AIFI with amodal register hub + block-sparse image attention.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads.
        block_size: Number of tokens per sparse block.
        topk: Number of key blocks attended by each image query block.
        num_registers: Number of amodal register tokens (dense hub).
        nc: Number of classes for the grounding head.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=2048, num_heads=8, block_size=4, topk=4,
                 num_registers=16, nc=10, dropout=0.0, act=nn.GELU(),
                 normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = RegisterBlockSparseAttention(
            dim=c1, num_heads=num_heads, block_size=block_size, topk=topk,
            num_registers=num_registers, nc=nc, dropout=dropout)
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
                f'AIFI_RegisterBlockSparse expected {self.c1} channels, got {c}')

        pos = self.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        # Registers are amodal: only image tokens receive position embedding.
        src = x.flatten(2).permute(0, 2, 1) + pos

        if self.normalize_before:
            src = src + self.dropout1(self.attn(self.norm1(src)))
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(self.norm2(src))))))
        else:
            src = src + self.dropout1(self.attn(src))
            src = self.norm1(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src)))))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()

    def compute_ground_loss(self, targets, batch=None):
        return self.attn.compute_ground_loss(targets)

    def extra_repr(self):
        return (f'c1={self.c1}, num_heads={self.attn.num_heads}, '
                f'block_size={self.attn.block_size}, topk={self.attn.topk}, '
                f'num_registers={self.attn.num_registers}')

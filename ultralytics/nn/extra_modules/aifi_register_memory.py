# RM-AIFI: Register-memory AIFI with task-semantic grounding
# (AIFI_RegisterMemory).
#
# Problem addressed: the entire network is one-directional (encoder ->
# decoder) and the AIFI is task-blind. DSAWACGAv5 operates per-block locally
# and P2SPDOKMFSV7 only performs spatial fusion - neither provides any
# cross-image "memory" carrier or task-semantics anchor.
#
# Mechanism (Registers: "Vision Transformers Need Registers", ICLR 2024):
#   1. M amodal register tokens (no spatial position) are concatenated to the
#      P5 token sequence and participate EQUALLY in the full self-attention -
#      they attend to image tokens and are attended by them, absorbing global
#      context and relieving attention artifacts. The main attention path is
#      untouched (no compression, no gating).
#   2. Task-semantic grounding (training only): the pooled register output is
#      classified against the image-level class-presence vector built from
#      the batch GT labels (BCE, collected by tasks.py through
#      compute_ground_loss). This forces the registers to carry detection-
#      relevant semantics instead of acting as free capacity.
#
# NOTE on the original proposal: decoder-query EMA distillation was simplified
# to GT class-presence grounding - same goal (task semantics enter the
# registers) without touching the decoder head and without any
# train/inference mismatch.

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AIFI_RegisterMemory']


class RegisterMemoryAttention(nn.Module):
    """Full MHSA over [register tokens, image tokens]."""

    def __init__(self, dim, num_heads=8, num_registers=16, nc=10, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f'dim ({dim}) must be divisible by num_heads ({num_heads})')
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
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

    def forward(self, x, hw=None):
        b, n, c = x.shape
        reg = self.registers.unsqueeze(0).expand(b, -1, -1)   # (b, M, c)
        seq = torch.cat([reg, x], dim=1)                      # (b, M+n, c)

        qkv = self.qkv(seq).chunk(3, dim=-1)
        q, k, v = [t.view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                   for t in qkv]
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(logits.softmax(dim=-1))
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, -1, c)

        reg_out, img_out = out[:, :self.num_registers], out[:, self.num_registers:]
        if self.training and torch.is_grad_enabled():
            self._reg_feats = reg_out                         # (b, M, c)
        return self.proj_drop(self.proj(img_out))

    def compute_ground_loss(self, targets):
        """Multi-label BCE: registers predict image-level class presence."""
        if not self.training or self._reg_feats is None:
            return None
        pooled = self._reg_feats.mean(dim=1)                  # (b, c)
        logits = self.ground_head(pooled)                     # (b, nc)
        b = logits.shape[0]
        nc = logits.shape[1]
        labels = torch.zeros(b, nc, device=logits.device)
        cls = targets['cls'].long()
        bidx = targets['batch_idx'].long()
        valid = (cls >= 0) & (cls < nc)
        labels[bidx[valid], cls[valid]] = 1.0
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        self._reg_feats = None                                # consume
        return loss


class AIFI_RegisterMemory(nn.Module):
    """AIFI augmented with amodal register-memory tokens.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads.
        num_registers: Number of amodal register tokens.
        nc: Number of classes for the grounding head.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=2048, num_heads=8, num_registers=16, nc=10,
                 dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = RegisterMemoryAttention(
            dim=c1, num_heads=num_heads, num_registers=num_registers,
            nc=nc, dropout=dropout)
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
                f'AIFI_RegisterMemory expected {self.c1} channels, got {c}')

        pos = self.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
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

    def compute_ground_loss(self, targets):
        return self.attn.compute_ground_loss(targets)

    def extra_repr(self):
        return (f'c1={self.c1}, num_heads={self.attn.num_heads}, '
                f'num_registers={self.attn.num_registers}')

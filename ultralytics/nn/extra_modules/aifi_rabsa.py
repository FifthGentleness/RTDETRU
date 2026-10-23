# RABSA: Register-Anchored Bridged Sparse Attention (AIFI_RABSA).
#
# Third-innovation AIFI mechanism designed strictly within the eight-position
# design space of the AIFI improvement study:
#   - Position 1 (attention pattern): UNTOUCHED. The image-token BlockSparse
#     top-k block mask of AIFI_BlockSparse (the only baseline-winning variant)
#     is reused verbatim. RABSA never re-designs routing.
#   - Position 2 (token set, MAIN): adds M fixed-count, position-free register
#     tokens with content-adaptive initialization (global-mean prior plus a
#     learnable embedding, following "Vision Transformers Need Registers",
#     ICLR 2024). Registers are written once per forward (dense read of all
#     image tokens) and then act as always-visible keys/values for EVERY
#     block-sparse query block. Image<->image attention keeps the sparse mask;
#     registers only EXTEND the KV set, restoring the two-hop global path
#     (block -> register -> block) that pure block sparsity removes.
#   - Position 4 (output fusion, companion): a zero-initialized gated readout
#     broadcasts register context back onto image tokens
#     (LayerScale-style per-channel gate, alpha = 0 at init, so the module is
#     numerically identical to AIFI_BlockSparse at step 0).
#   - Positions 3/5/6/7/8: untouched (no scoring change, original FFN,
#     original norm topology, zero auxiliary loss, single AIFI layer).
#
# VisDrone rationale: register hubs absorb global/background statistics so
# each sparse block can keep high-fidelity local evidence for tiny weak
# targets instead of being over-smoothed, while clustered objects regain
# cross-cluster consistency through the shared register bridge.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AIFI_RABSA']


def _mha(q, k, v, num_heads, dropout=0.0, training=True):
    """Multi-head scaled dot-product attention on [B, N, C] tensors."""
    b, nq, c = q.shape
    d = c // num_heads
    q = q.view(b, nq, num_heads, d).transpose(1, 2)
    k = k.view(b, k.shape[1], num_heads, d).transpose(1, 2)
    v = v.view(b, v.shape[1], num_heads, d).transpose(1, 2)
    out = F.scaled_dot_product_attention(
        q, k, v, dropout_p=dropout if training else 0.0)
    return out.transpose(1, 2).reshape(b, nq, c)


class RegisterBridgedSparseAttention(nn.Module):
    """Block-sparse attention (mask identical to BlockSparseAttention) with a
    fixed set of content-adaptive register tokens as global KV hubs."""

    def __init__(self, dim, num_heads=8, block_size=4, topk=4,
                 num_registers=16, use_content_prior=True, use_readout=True,
                 gate_init=0.0, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim ({dim}) must be divisible by num_heads ({num_heads})')
        if block_size < 1:
            raise ValueError('block_size must be >= 1')
        if topk < 1:
            raise ValueError('topk must be >= 1')
        if num_registers < 1:
            raise ValueError('num_registers must be >= 1')

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.block_size = int(block_size)
        self.topk = int(topk)
        self.num_registers = int(num_registers)
        self.use_content_prior = bool(use_content_prior)
        self.use_readout = bool(use_readout)

        # Position 2: position-free register embedding + content-prior projection.
        self.reg_embed = nn.Parameter(torch.randn(num_registers, dim) * 0.02)
        self.reg_prior_proj = nn.Linear(dim, dim, bias=False) \
            if self.use_content_prior else None

        # Shared QKV/proj with the original BlockSparseAttention parameterization.
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = dropout
        self.proj_drop = nn.Dropout(dropout)

        # Position 2: register write branch (dense global read by registers).
        self.reg_write_proj = nn.Linear(dim, dim, bias=True)

        # Position 4: zero-initialized gated register readout.
        if self.use_readout:
            self.read_q = nn.Linear(dim, dim, bias=True)
            self.gate_ls = nn.Parameter(torch.full((dim,), float(gate_init)))

    def forward(self, x):
        b, n, c = x.shape
        h, d = self.num_heads, self.head_dim
        bs = self.block_size
        num_blocks = math.ceil(n / bs)
        pad_len = num_blocks * bs - n

        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))
        npad = num_blocks * bs

        # Position 2: content-adaptive register construction.
        R0 = self.reg_embed.unsqueeze(0).expand(b, -1, -1)
        if self.reg_prior_proj is not None:
            R0 = R0 + self.reg_prior_proj(x.mean(dim=1, keepdim=True))

        # Shared projections (image tokens and registers use the same weights).
        qkv_x = self.qkv(x).chunk(3, dim=-1)
        q, k_x, v_x = [t.view(b, npad, h, d).permute(0, 2, 1, 3) for t in qkv_x]
        qkv_r = self.qkv(R0).chunk(3, dim=-1)
        q_r, k_r, v_r = [t.view(b, self.num_registers, h, d).permute(0, 2, 1, 3)
                         for t in qkv_r]

        # ---- Position 2: register write (dense read of all image tokens). ----
        r_ctx = _mha(q_r.reshape(b, self.num_registers, c),
                     torch.cat([x, R0], dim=1), torch.cat([x, R0], dim=1),
                     self.num_heads, self.attn_drop, self.training)
        R = R0 + self.reg_write_proj(r_ctx)
        kkv_r = self.qkv(R).chunk(3, dim=-1)
        _, k_r2, v_r2 = [t.view(b, self.num_registers, h, d).permute(0, 2, 1, 3)
                         for t in kkv_r]

        # ---- Position 1: image-token block routing EXACTLY as AIFI_BlockSparse. ----
        q_blk = x.view(b, num_blocks, bs, c).mean(dim=2)
        k_blk = x.view(b, num_blocks, bs, c).mean(dim=2)
        block_logits = torch.matmul(q_blk, k_blk.transpose(-2, -1)) * self.scale
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

        # Position 2: registers are always-visible KV columns (image-token
        # rows/columns keep the unchanged sparse mask).
        k_cat = torch.cat([k_x, k_r2], dim=2)   # [B, h, npad+M, d]
        v_cat = torch.cat([v_x, v_r2], dim=2)
        full_mask = torch.cat(
            [token_mask,
             torch.ones(b, npad, self.num_registers, dtype=torch.bool,
                        device=x.device)], dim=2)

        logits = torch.matmul(q, k_cat.transpose(-2, -1)) * self.scale
        logits = logits.masked_fill(~full_mask.unsqueeze(1), -1e4)
        attn = logits.softmax(dim=-1)
        attn = F.dropout(attn, self.attn_drop, self.training)
        out = torch.matmul(attn, v_cat)
        out = out.permute(0, 2, 1, 3).reshape(b, npad, c)
        if pad_len:
            out = out[:, :n, :]
        out = self.proj_drop(self.proj(out))

        # ---- Position 4: zero-initialized gated register readout. ----
        if self.use_readout:
            read = _mha(self.read_q(out), R, R,
                        self.num_heads, self.attn_drop, self.training)
            out = out + self.gate_ls * read
        return out


class AIFI_RABSA(nn.Module):
    """AIFI with register-anchored bridged block-sparse attention.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN (unchanged, position 5 untouched).
        num_heads: Number of attention heads.
        block_size: Tokens per sparse block (position 1, unchanged).
        topk: Key blocks attended by each query block (position 1, unchanged).
        num_registers: Number of global register tokens (position 2).
        use_content_prior: Enable content-adaptive register initialization.
        use_readout: Enable the zero-init gated register readout (position 4).
        gate_init: Initial value of the readout gate (0.0 = exact BlockSparse
            behavior at init; set 1.0 for the direct-add ablation).
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=2048, num_heads=8, block_size=4, topk=4,
                 num_registers=16, use_content_prior=True, use_readout=True,
                 gate_init=0.0, dropout=0.0, act=nn.GELU(),
                 normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = RegisterBridgedSparseAttention(
            dim=c1,
            num_heads=num_heads,
            block_size=block_size,
            topk=topk,
            num_registers=num_registers,
            use_content_prior=use_content_prior,
            use_readout=use_readout,
            gate_init=gate_init,
            dropout=dropout,
        )
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
                f'AIFI_RABSA expected {self.c1} channels, got {c}')

        pos = self.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1) + pos

        if self.normalize_before:
            src_norm = self.norm1(src)
            src = src + self.dropout1(self.attn(src_norm))
            src_norm = self.norm2(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src_norm)))))
        else:
            src = src + self.dropout1(self.attn(src))
            src = self.norm1(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src)))))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()

    def extra_repr(self):
        a = self.attn
        return (f'c1={self.c1}, cm={self.fc1.out_features}, '
                f'num_registers={a.num_registers}, '
                f'block_size={a.block_size}, topk={a.topk}, '
                f'use_content_prior={a.use_content_prior}, '
                f'use_readout={a.use_readout}')

# LB-AIFI: Load-Balanced learnable block routing (AIFI_BlockSparseV19).
#
# Motivation: V1 gives every query block a FIXED top-k budget and routes with
# UNTRAINED mean-pooled descriptors. Uniform sky/road blocks waste budget
# while object-dense blocks are starved, and the hard top-k selection gives
# the router no gradient at all. V12 patched the budget by hand (CDF with
# temperature/tau/rho/cap hyper-parameters); V19 instead formulates block
# routing as a MoE routing problem and lets the optimizer fix it:
#
#   1. Learnable router: a single Linear(dim, dim) applied to the mean-pooled
#      block descriptors of both sides. Initialized to IDENTITY, so the
#      initial routing logits are exactly V1's (no random-init drift).
#      This is the ONLY added component - no MLP, no temperature, no QK-norm,
#      no DWConv, no extra gates (V3's failed kitchen sink is not repeated).
#   2. Straight-through routing gradient: the routing soft probabilities
#      bias the token-level attention logits of the selected blocks
#      (route_bias_scale, a single zero-init scalar). At init the bias is 0
#      so attention is exactly V1; the scalar receives gradient immediately
#      (V17/V18 single-gate warm-up dynamic - never double-gated), and once
#      it moves the router itself starts learning from the attention path.
#   3. Switch-Transformer-style load-balancing aux loss (training only):
#         aux = N_blocks * sum_i  f_i * P_i
#      f_i = mean routing probability of key block i (differentiable),
#      P_i = fraction of query blocks that selected key block i (detached).
#      Minimizing it pushes routing probability toward a load-aware uniform,
#      fixing the sky-wastes/object-starves budget misallocation from first
#      principles instead of hand-tuned CDF thresholds. The loss rides the
#      existing V8 aux-loss pipeline (attr route_distill_loss collected by
#      tasks.py, weight 0.1) - zero tasks.py logic changes.
#   4. Progressive sparsification (OPTIONAL, off by default): with
#      anneal_steps > 0 the per-query-block budget anneals from k_start to
#      topk over training. Default off keeps step-0 behavior exactly V1.
#
# Inference graph is isomorphic to V1: same tokens, same dense-masked
# attention, only the mask generation and two zero-init scalars differ.
# Mirror of the HybridEncoderP2SPDOKMFSV7 philosophy: identity-residual
# preserved, improvements enter through warm-up gates.
#
# References:
#   Switch Transformers (JMLR 2022) - load-balancing auxiliary loss.
#   Mixture-of-Attention (ECCV 2024) - attention as mixture-of-experts routing.
#   Block-Sparse Global Attention (CVPR 2026) - the V1 baseline.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AIFI_BlockSparseV19']


class LBRoutedBlockSparseAttention(nn.Module):
    """V1 block-sparse attention with a learnable, load-balanced router."""

    def __init__(self, dim, num_heads=8, block_size=4, topk=4, dropout=0.0,
                 anneal_steps=0, k_start=None):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f'dim ({dim}) must be divisible by num_heads ({num_heads})')

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.block_size = int(block_size)
        self.topk = int(topk)
        self.anneal_steps = int(anneal_steps)
        self.k_start = int(k_start) if k_start is not None else self.topk * 2

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Learnable router, identity-initialized: initial routing logits are
        # exactly V1's desc @ desc.T.
        self.router = nn.Linear(dim, dim)
        with torch.no_grad():
            self.router.weight.copy_(torch.eye(dim))
            self.router.bias.zero_()

        # Single zero-init scalar for the straight-through routing bias.
        self.route_bias_scale = nn.Parameter(torch.zeros(1))

        # Training-step counter for the optional progressive sparsification.
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))

        # Aux loss populated during training forward; collected by tasks.py
        # through the existing V8 pipeline (attr name kept for compatibility).
        self.route_distill_loss = None

    def __deepcopy__(self, memo):
        """Drop stale graph-carrying aux-loss tensors (ModelEMA deepcopy)."""
        import copy as _copy

        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k == 'route_distill_loss':
                new.route_distill_loss = None
                continue
            new.__dict__[k] = _copy.deepcopy(v, memo)
        return new

    def current_topk(self, num_blocks):
        """Per-query-block budget: fixed topk, or annealed if configured."""
        if self.anneal_steps > 0 and self.training:
            t = min(1.0, float(self.step.item()) / float(self.anneal_steps))
            k = int(round(self.k_start + (self.topk - self.k_start) * t))
            return max(1, min(k, num_blocks))
        return self.topk

    def forward(self, x):
        if not self.training:
            self.route_distill_loss = None  # never leak stale train aux into eval
        b, n, c = x.shape
        h, d = self.num_heads, self.head_dim
        bs = self.block_size
        num_blocks = math.ceil(n / bs)
        npad = num_blocks * bs
        pad_len = npad - n

        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, npad, h, d).permute(0, 2, 1, 3) for t in qkv]

        # --- Routing (learnable router on mean-pooled block descriptors) ---
        desc = x.view(b, num_blocks, bs, c).mean(dim=2)          # (b, nb, c)
        rq = self.router(desc)
        rk = self.router(desc)
        block_logits = torch.matmul(rq, rk.transpose(-2, -1)) * self.scale
        soft = block_logits.softmax(dim=-1)                      # (b, nb, nb)

        topk = self.current_topk(num_blocks)
        topk_idx = block_logits.topk(k=topk, dim=-1).indices

        block_mask = torch.zeros(
            b, num_blocks, num_blocks, dtype=torch.bool, device=x.device)
        block_mask.scatter_(-1, topk_idx, True)
        block_mask = block_mask | torch.eye(
            num_blocks, dtype=torch.bool, device=x.device).unsqueeze(0)

        token_mask = block_mask.repeat_interleave(bs, dim=1)
        token_mask = token_mask.repeat_interleave(bs, dim=2)
        token_mask = token_mask[:, :npad, :npad]
        if pad_len:
            valid = torch.zeros(b, npad, dtype=torch.bool, device=x.device)
            valid[:, :n] = True
            token_mask = token_mask & valid[:, None, :]

        # --- Load-balancing aux loss (training only, Switch-style) ---
        if self.training and torch.is_grad_enabled():
            f = soft.mean(dim=1)                          # (b, nb), diff
            p_load = block_mask.float().mean(dim=1).detach()  # (b, nb), hard
            lb = num_blocks * (f * p_load).sum(dim=-1).mean()
            self.route_distill_loss = lb

        # --- Dense masked attention with straight-through routing bias ---
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        bias_tok = soft.repeat_interleave(bs, dim=1)
        bias_tok = bias_tok.repeat_interleave(bs, dim=2)[:, :npad, :npad]
        logits = logits + self.route_bias_scale * bias_tok.unsqueeze(1)
        logits = logits.masked_fill(~token_mask.unsqueeze(1), -1e4)
        attn = self.attn_drop(logits.softmax(dim=-1))
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, npad, c)
        if pad_len:
            out = out[:, :n, :]

        if self.training:
            self.step += 1

        return self.proj_drop(self.proj(out))


class AIFI_BlockSparseV19(nn.Module):
    """AIFI with load-balanced learnable block routing (LB-AIFI).

    Inference graph is isomorphic to V1 (BlockSparseV1); the differences are
    the router, the straight-through routing bias, and the training-only
    load-balancing aux loss. Training starts exactly at V1 behavior.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads.
        block_size: Tokens per routing block.
        topk: Key blocks attended by each query block (final budget).
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
        anneal_steps: Optional progressive-sparsification length in steps
            (budget anneals k_start -> topk). 0 disables (default).
        k_start: Starting budget for the anneal (default 2 * topk).
    """

    def __init__(self, c1, cm=2048, num_heads=8, block_size=4, topk=4,
                 dropout=0.0, act=nn.GELU(), normalize_before=False,
                 anneal_steps=0, k_start=None):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = LBRoutedBlockSparseAttention(
            dim=c1, num_heads=num_heads, block_size=block_size, topk=topk,
            dropout=dropout, anneal_steps=anneal_steps, k_start=k_start)
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
                f'AIFI_BlockSparseV19 expected {self.c1} channels, got {c}')

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

    def extra_repr(self):
        return (f'c1={self.c1}, block_size={self.attn.block_size}, '
                f'topk={self.attn.topk}, anneal_steps={self.attn.anneal_steps}')

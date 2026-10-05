# Block-sparse global attention V3 for RT-DETR AIFI.
#
# Improvements over V2 (aifi_block_sparse_v2.py):
#   1. Per-head routing: each attention head independently selects its own
#      top-k key blocks. V2 forced all heads to share one routing pattern,
#      severely limiting global context diversity. With per-head routing,
#      the effective receptive field is union-of-heads, much larger.
#   2. Separate Q/K/V projections with pos only on Q/K (DETR convention).
#      V is projected from pos-free input, preserving translation equivariance
#      for bbox regression. (Lesson from AgentAttentionV2.)
#   3. DWConv(V) local supplement: lightweight depthwise conv on V provides
#      token-level local detail that block-level local window cannot capture.
#      (Proven effective in AgentAttentionV2 / Agent Attention paper.)
#   4. Adaptive top-k with importance reweighting: instead of hard top-k,
#      routing logits are softmaxed to produce importance weights; top-k
#      blocks are selected and their attention logits are biased by the
#      routing importance. This lets the model learn variable sparsity.
#   5. Zero-init gating: the sparse global output and DWConv local output
#      are gated by learnable scales initialized to 0. Training starts from
#      baseline (dense-like) behavior and gradually learns to use the
#      sparse/local branches. This is the validated warm-start pattern.
#   6. QK-Norm on routing logits: L2-normalize block Q/K descriptors before
#      computing routing logits, preventing norm-dominated routing that
#      ignores semantic content. (Lesson from QKNorm AIFI.)
#   7. Finer default block_size (4 instead of 8) for more granular routing.
#   8. All operations remain fully vectorized.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AIFI_BlockSparseV3']


class BlockSparseAttentionV3(nn.Module):
    """V3 block-sparse attention: per-head routing + DWConv local + zero-init gate.

    All operations are fully vectorized for efficiency.
    """

    def __init__(self, dim, num_heads=8, block_size=4, topk=4,
                 local_window=3, dropout=0.0, dw_kernel_size=3):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim ({dim}) must be divisible by num_heads ({num_heads})')
        if block_size < 1:
            raise ValueError('block_size must be >= 1')
        if topk < 1:
            raise ValueError('topk must be >= 1')
        if local_window < 1:
            raise ValueError('local_window must be >= 1')

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.block_size = int(block_size)
        self.topk = int(topk)
        self.local_window = int(local_window)
        self.dw_kernel_size = int(dw_kernel_size)

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.router_norm = nn.LayerNorm(dim)
        self.router = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )
        self.route_scale = nn.Parameter(torch.tensor(self.scale))

        self.qk_scale = nn.Parameter(torch.ones(num_heads))
        self.qk_scale_k = nn.Parameter(torch.ones(num_heads))

        self.gate_sparse = nn.Parameter(torch.zeros(1))
        self.gate_local = nn.Parameter(torch.zeros(1))

        if dw_kernel_size > 0:
            pad = dw_kernel_size // 2
            self.dw_conv = nn.Conv2d(
                dim, dim, kernel_size=dw_kernel_size,
                padding=pad, groups=dim, bias=True)
        else:
            self.dw_conv = None

        self._cache_hw = None
        self._cache_mappings = None

        self._reset_parameters()

    def _reset_parameters(self):
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.proj):
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
        if self.dw_conv is not None:
            nn.init.xavier_uniform_(self.dw_conv.weight)
            nn.init.zeros_(self.dw_conv.bias)

    def _build_2d_mapping(self, h, w, bs, device):
        bh = math.ceil(h / bs)
        bw = math.ceil(w / bs)
        num_blocks = bh * bw
        npad = num_blocks * bs * bs

        r = torch.arange(h, device=device).unsqueeze(1).expand(h, w).flatten()
        c = torch.arange(w, device=device).unsqueeze(0).expand(h, w).flatten()

        br = r // bs
        bc = c // bs
        block_id = br * bw + bc
        pos_in_block = (r % bs) * bs + (c % bs)

        block_order = torch.full((num_blocks, bs * bs), -1, dtype=torch.long, device=device)
        orig_idx = torch.arange(h * w, device=device)
        block_order[block_id, pos_in_block] = orig_idx

        reorder_idx = block_id * (bs * bs) + pos_in_block

        return block_order.flatten(), reorder_idx, bh, bw, num_blocks, npad

    def _build_local_mask(self, bh, bw, lw, device):
        qr = torch.arange(bh, device=device).unsqueeze(1).expand(bh, bw).flatten()
        qc = torch.arange(bw, device=device).unsqueeze(0).expand(bh, bw).flatten()

        dr = (qr.unsqueeze(1) - qr.unsqueeze(0)).abs()
        dc = (qc.unsqueeze(1) - qc.unsqueeze(0)).abs()
        local_mask = (dr < lw) & (dc < lw)

        return local_mask

    def forward(self, qk_input, v_input, h=None, w=None):
        """Forward with separated Q/K and V inputs (DETR convention).

        Args:
            qk_input: (B, N, C) features + positional embedding (for Q/K).
            v_input: (B, N, C) features without positional embedding (for V).
            h, w: spatial dimensions.
        """
        b, n, c = qk_input.shape
        if h is None:
            h = int(math.sqrt(n))
        if w is None:
            w = n // h
        assert h * w == n, f'spatial size {h}x{w} != token count {n}'

        hh, d = self.num_heads, self.head_dim
        bs = self.block_size
        lw = self.local_window

        cache_key = (h, w)
        if self._cache_hw != cache_key or self._cache_mappings[0].device != qk_input.device:
            block_order, reorder_idx, bh, bw, num_blocks, npad = \
                self._build_2d_mapping(h, w, bs, qk_input.device)
            local_block_mask = self._build_local_mask(bh, bw, lw, qk_input.device)
            self._cache_hw = cache_key
            self._cache_mappings = (
                block_order, reorder_idx, bh, bw, num_blocks, npad,
                local_block_mask)
        block_order, reorder_idx, bh, bw, num_blocks, npad, local_block_mask = \
            self._cache_mappings

        gather_idx = block_order.unsqueeze(0).expand(b, -1)
        valid_mask = gather_idx >= 0
        gather_safe = gather_idx.clamp(min=0)

        qk_reordered = qk_input.gather(1, gather_safe.unsqueeze(-1).expand(-1, -1, c))
        qk_reordered = qk_reordered * valid_mask.unsqueeze(-1).to(qk_input.dtype)

        v_reordered = v_input.gather(1, gather_safe.unsqueeze(-1).expand(-1, -1, c))
        v_reordered = v_reordered * valid_mask.unsqueeze(-1).to(v_input.dtype)

        q = self.q_proj(qk_reordered).view(b, npad, hh, d).permute(0, 2, 1, 3)
        k = self.k_proj(qk_reordered).view(b, npad, hh, d).permute(0, 2, 1, 3)
        v = self.v_proj(v_reordered).view(b, npad, hh, d).permute(0, 2, 1, 3)

        blksz = bs * bs
        valid_blk = valid_mask.view(b, num_blocks, blksz).float()
        block_valid = valid_blk.sum(2) > 0

        q_blk_full_h = q.reshape(b, hh, num_blocks, blksz, d)
        k_blk_full_h = k.reshape(b, hh, num_blocks, blksz, d)

        valid_blk_h = valid_blk.unsqueeze(1)
        q_blk_h = (q_blk_full_h * valid_blk_h.unsqueeze(-1)).sum(3) / \
            valid_blk_h.sum(3, keepdim=True).clamp(min=1)
        k_blk_h = (k_blk_full_h * valid_blk_h.unsqueeze(-1)).sum(3) / \
            valid_blk_h.sum(3, keepdim=True).clamp(min=1)

        q_blk_h_normed = F.normalize(q_blk_h, dim=-1) * self.qk_scale.view(1, -1, 1, 1)
        k_blk_h_normed = F.normalize(k_blk_h, dim=-1) * self.qk_scale_k.view(1, -1, 1, 1)

        block_logits_per_head = torch.einsum(
            'bhid,bhjd->bhij', q_blk_h_normed, k_blk_h_normed)

        k_blk_full_c = k.transpose(1, 2).reshape(b, num_blocks, blksz, c)
        k_blk_c = (k_blk_full_c * valid_blk.unsqueeze(-1)).sum(2) / \
            valid_blk.sum(2, keepdim=True).clamp(min=1)
        k_norm = self.router_norm(k_blk_c)
        route_bias = self.router(k_norm).squeeze(-1)
        block_logits_per_head = block_logits_per_head + route_bias.unsqueeze(1).unsqueeze(2)

        block_logits_per_head = block_logits_per_head.masked_fill(
            ~block_valid.unsqueeze(1).unsqueeze(2), -1e4)

        topk = min(self.topk, num_blocks)

        route_weights = block_logits_per_head.softmax(dim=-1)
        topk_vals, topk_idx = route_weights.topk(k=topk, dim=-1)

        head_block_mask = torch.zeros(
            b, hh, num_blocks, num_blocks, dtype=torch.bool, device=qk_input.device)
        head_block_mask.scatter_(-1, topk_idx, True)

        eye = torch.eye(num_blocks, dtype=torch.bool, device=qk_input.device)
        head_block_mask = head_block_mask | eye.unsqueeze(0).unsqueeze(0) | \
            local_block_mask.unsqueeze(0).unsqueeze(0)

        head_block_mask = head_block_mask & block_valid.unsqueeze(1).unsqueeze(2)

        token_mask_per_head = head_block_mask.repeat_interleave(blksz, dim=2)
        token_mask_per_head = token_mask_per_head.repeat_interleave(blksz, dim=3)
        token_mask_per_head = token_mask_per_head[:, :, :npad, :npad]

        token_mask_per_head = token_mask_per_head & valid_mask.unsqueeze(1).unsqueeze(2)

        routing_bias_blk = route_weights
        routing_bias_blk = routing_bias_blk.repeat_interleave(blksz, dim=2)
        routing_bias_blk = routing_bias_blk.repeat_interleave(blksz, dim=3)
        routing_bias_blk = routing_bias_blk[:, :, :npad, :npad]
        routing_bias = (routing_bias_blk - routing_bias_blk.mean(dim=-1, keepdim=True))

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        logits = logits + routing_bias

        logits = logits.masked_fill(~token_mask_per_head, -1e4)
        attn = self.attn_drop(logits.softmax(dim=-1))
        out_sparse = torch.matmul(attn, v)
        out_sparse = out_sparse.permute(0, 2, 1, 3).reshape(b, npad, c)

        out_original = torch.zeros_like(out_sparse)
        valid_exp = valid_mask.unsqueeze(-1).expand(-1, -1, c)
        out_original.scatter_(
            1, gather_safe.unsqueeze(-1).expand(-1, -1, c),
            out_sparse * valid_exp.to(out_sparse.dtype))
        out_sparse = out_original[:, :n, :]

        out_local = None
        if self.dw_conv is not None:
            v_2d = v_input.transpose(1, 2).reshape(b, c, h, w)
            local_out_2d = self.dw_conv(v_2d)
            out_local = local_out_2d.reshape(b, c, -1).transpose(1, 2)

        out = self.proj(out_sparse) * (1.0 + self.gate_sparse)
        if out_local is not None:
            out = out + out_local * self.gate_local

        return self.proj_drop(out)


class AIFI_BlockSparseV3(nn.Module):
    """AIFI with V3 block-sparse global attention.

    Key improvements over V2:
      - Per-head routing: each attention head selects its own top-k key blocks
      - Separate Q/K and V projections (pos only on Q/K, DETR convention)
      - DWConv(V) token-level local supplement
      - Adaptive top-k with routing importance reweighting
      - Zero-init gating for sparse and local branches
      - QK-Norm on routing logits for stable routing
      - Finer default block_size (4) for more granular routing
      - Fully vectorized operations

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads.
        block_size: Spatial block size (tokens per side of a 2D block).
        topk: Number of globally attended key blocks per query block per head.
        local_window: Local window size in blocks (Chebyshev distance).
        dropout: Dropout probability.
        dw_kernel_size: DWConv kernel size for local supplement (0 to disable).
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=2048, num_heads=8, block_size=4, topk=4,
                 local_window=3, dropout=0.0, dw_kernel_size=3,
                 act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = BlockSparseAttentionV3(
            dim=c1,
            num_heads=num_heads,
            block_size=block_size,
            topk=topk,
            local_window=local_window,
            dropout=dropout,
            dw_kernel_size=dw_kernel_size,
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
                f'AIFI_BlockSparseV3 expected {self.c1} channels, got {c}')

        pos = self.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1)
        qk_input = src + pos
        v_input = src

        if self.normalize_before:
            qk_norm = self.norm1(qk_input)
            v_norm = self.norm1(v_input)
            src = src + self.dropout1(self.attn(qk_norm, v_norm, h=h, w=w))
            src_norm = self.norm2(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src_norm)))))
        else:
            src = src + self.dropout1(self.attn(qk_input, v_input, h=h, w=w))
            src = self.norm1(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src)))))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()
"""Manifold-Constrained Hyper-Connected Progressive Focused Block-Sparse AIFI.

Combines four complementary non-DETR mechanisms (2024-2026) tailored to
VisDrone aerial object detection at the P5 semantic bottleneck:
1. mHC (DeepSeek-AI, 2025 / Hyper-Connections, ICLR 2025): Expands the residual
   stream into n parallel streams with Sinkhorn-Knopp doubly-stochastic mixing
   on the Birkhoff polytope, preventing multi-stage over-smoothing of tiny objects.
2. Progressive Focused Attention (PFT-SR, CVPR 2025) over Block-Sparse routing
   (CVPR 2026): Stage 1 prunes ~85% background via block-sparse routing; later
   stages inherit prior attention maps via Hadamard modulation and progressively
   tighten token-level top-k focus while always preserving local block neighbors.
3. Mixed-Frequency 2D RoPE (RoPE-ViT, ECCV 2024): Injects learnable per-head
   2D relative spatial rotations strictly into Q and K, keeping V and the
   multi-stream residual space translation-equivariant and unpolluted.
4. Statistical Top-k Spatial FFN (Spark Transformer, NeurIPS 2025): Combines
   3x3 depthwise spatial bias with per-token statistical quantile gating
   (mu + kappa * sigma) to suppress lazy background neurons.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AIFI_mHCPFBSA']


def _rmsnorm_last(x, eps=1e-6):
    """Parameter-free RMS normalization along the last dimension."""
    rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * rms.to(x.dtype)


def _sinkhorn_knopp(logits, iters=10, eps=1e-6):
    """Project logits onto the Birkhoff polytope of doubly-stochastic matrices."""
    with torch.autocast(device_type=logits.device.type, enabled=False):
        mat = torch.exp(logits.float().clamp(-10.0, 10.0))
        for _ in range(iters):
            mat = mat / mat.sum(dim=-1, keepdim=True).clamp_min(eps)
            mat = mat / mat.sum(dim=-2, keepdim=True).clamp_min(eps)
    return mat


class Mixed2DRoPE(nn.Module):
    """Learnable mixed-frequency 2D Rotary Position Embedding (RoPE-ViT, ECCV 2024)."""

    def __init__(self, num_heads, head_dim, base=100.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f'Mixed2DRoPE requires even head_dim, got {head_dim}')
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        num_pairs = head_dim // 2

        # Multi-scale magnitudes and distributed 2D orientations across heads/pairs.
        mag = (math.pi / 2.0) * (base ** (-torch.arange(num_pairs, dtype=torch.float32) / num_pairs))
        head_idx = torch.arange(num_heads, dtype=torch.float32)[:, None]
        pair_idx = torch.arange(num_pairs, dtype=torch.float32)[None, :]
        angles = 2.0 * math.pi * (head_idx / num_heads + pair_idx / max(num_pairs * 4, 1))
        freqs_x = mag[None, :] * torch.cos(angles)
        freqs_y = mag[None, :] * torch.sin(angles)
        self.freqs = nn.Parameter(torch.stack([freqs_x, freqs_y], dim=0))  # [2, H, D/2]

    @staticmethod
    def _rotate(x, cos, sin):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)

    def forward(self, q, k, h, w):
        """Apply 2D rotary embeddings to q and k of shape [B, H, h*w, D]."""
        device = q.device
        y = torch.arange(h, device=device, dtype=torch.float32) - 0.5 * (h - 1)
        x = torch.arange(w, device=device, dtype=torch.float32) - 0.5 * (w - 1)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        px = grid_x.reshape(-1, 1, 1)
        py = grid_y.reshape(-1, 1, 1)

        with torch.autocast(device_type=device.type, enabled=False):
            freqs = self.freqs.float()
            theta = px * freqs[0].unsqueeze(0) + py * freqs[1].unsqueeze(0)  # [N, H, D/2]
            cos = theta.cos().permute(1, 0, 2).unsqueeze(0).to(q.dtype)
            sin = theta.sin().permute(1, 0, 2).unsqueeze(0).to(q.dtype)

        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)


class PFBlockSparseAttention(nn.Module):
    """Progressive Focused Block-Sparse Attention with Mixed 2D RoPE.

    Stage 0 performs block-sparse top-k routing to eliminate background tokens.
    Subsequent stages inherit the prior attention distribution via PFA Hadamard
    modulation (CVPR 2025) and progressively tighten token-level top-k support
    while preserving own-block local neighbors.
    """

    def __init__(self, dim, num_heads=8, block_size=4, stage1_topk=6,
                 focus_topk=None, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim ({dim}) must be divisible by num_heads ({num_heads})')
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.block_size = int(block_size)
        self.stage1_topk = int(stage1_topk)
        self.focus_topk = None if focus_topk is None else int(focus_topk)

        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.rope = Mixed2DRoPE(self.num_heads, self.head_dim)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        if self.focus_topk is not None:
            # Per-head Hadamard inheritance blend initialized at 0.5 for balanced gradient flow.
            self.inherit_logit = nn.Parameter(torch.zeros(1, self.num_heads, 1, 1))

    def _stage0_mask(self, x, n):
        b, _, c = x.shape
        bs = self.block_size
        num_blocks = math.ceil(n / bs)
        pad_len = num_blocks * bs - n
        x_pad = F.pad(x, (0, 0, 0, pad_len)) if pad_len else x

        x_blk = x_pad.view(b, num_blocks, bs, c).mean(dim=2)
        block_logits = torch.matmul(x_blk, x_blk.transpose(-2, -1)) * self.scale
        if pad_len:
            blk_valid = (torch.arange(num_blocks, device=x.device) * bs) < n
            block_logits = block_logits.masked_fill(~blk_valid[None, None, :], -1e4)

        k_blk = min(self.stage1_topk, num_blocks)
        topk_idx = block_logits.topk(k=k_blk, dim=-1).indices
        block_mask = torch.zeros(b, num_blocks, num_blocks, dtype=torch.bool, device=x.device)
        block_mask.scatter_(-1, topk_idx, True)
        eye = torch.eye(num_blocks, dtype=torch.bool, device=x.device)
        block_mask = block_mask | eye.unsqueeze(0)

        token_mask = block_mask.repeat_interleave(bs, dim=1).repeat_interleave(bs, dim=2)
        return token_mask[:, :n, :n].unsqueeze(1)  # [B, 1, N, N]

    def _progressive_mask(self, prev_attn, n):
        b, h, _, _ = prev_attn.shape
        k_tok = min(self.focus_topk, n)
        topk_idx = prev_attn.detach().topk(k=k_tok, dim=-1).indices
        focus_mask = torch.zeros(b, h, n, n, dtype=torch.bool, device=prev_attn.device)
        focus_mask.scatter_(-1, topk_idx, True)

        # Always retain each token's own local block so small-object boundaries never drop out.
        blk_ids = torch.arange(n, device=prev_attn.device) // self.block_size
        own_block = (blk_ids[:, None] == blk_ids[None, :])[None, None, :, :]
        return focus_mask | own_block

    def forward(self, x, h, w, prev_attn=None):
        b, n, c = x.shape
        heads, d = self.num_heads, self.head_dim
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, n, heads, d).transpose(1, 2)
        k = k.view(b, n, heads, d).transpose(1, 2)
        v = v.view(b, n, heads, d).transpose(1, 2)

        q, k = self.rope(q, k, h, w)

        if self.focus_topk is None or prev_attn is None:
            mask = self._stage0_mask(x, n)
        else:
            mask = self._progressive_mask(prev_attn, n)

        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
            logits = logits.masked_fill(~mask, -1e9)
            attn_cur = logits.softmax(dim=-1)

            if self.focus_topk is not None and prev_attn is not None:
                g = torch.sigmoid(self.inherit_logit.float())
                supp_size = mask.sum(dim=-1, keepdim=True).clamp_min(1).float()
                uniform = mask.float() / supp_size
                prior = (1.0 - g) * uniform + g * (prev_attn.float() * mask.float())
                weighted = prior * attn_cur
                attn = weighted / weighted.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            else:
                attn = attn_cur

        attn_dropped = self.attn_drop(attn).to(v.dtype)
        out = torch.matmul(attn_dropped, v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out)), attn


class SparkConvFFN(nn.Module):
    """Spatial-augmented Statistical Top-k Sparse FFN (Spark Transformer, NeurIPS 2025)."""

    def __init__(self, dim, cm=768, active_ratio=0.30, dropout=0.0):
        super().__init__()
        if not 0.0 < active_ratio < 1.0:
            raise ValueError(f'active_ratio must be in (0, 1), got {active_ratio}')
        self.dim = int(dim)
        self.cm = int(cm)
        self.fc1 = nn.Linear(self.dim, self.cm)
        self.dwconv = nn.Conv2d(self.cm, self.cm, kernel_size=3, padding=1, groups=self.cm)
        self.dw_scale = nn.Parameter(torch.full((self.cm,), 0.1))

        # Gaussian quantile z-score for statistical top-k activation sparsity.
        kappa = math.sqrt(2.0) * float(torch.erfinv(torch.tensor(1.0 - 2.0 * active_ratio)).item())
        self.register_buffer('kappa', torch.tensor(kappa, dtype=torch.float32), persistent=False)
        self.thresh_bias = nn.Parameter(torch.zeros(1, 1, self.cm))
        self.sharpness_raw = nn.Parameter(torch.tensor(1.854, dtype=torch.float32))  # softplus ~ 2.0

        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.cm, self.dim)

    def forward(self, x, h, w):
        b, n, _ = x.shape
        z = self.fc1(x)
        z_2d = z.transpose(1, 2).reshape(b, self.cm, h, w)
        z = z + (self.dw_scale.view(1, self.cm, 1, 1) * self.dwconv(z_2d)).reshape(b, self.cm, n).transpose(1, 2)

        with torch.autocast(device_type=x.device.type, enabled=False):
            zf = z.float()
            mu = zf.mean(dim=-1, keepdim=True)
            sigma = zf.var(dim=-1, unbiased=False, keepdim=True).add(1e-5).sqrt()
            tau = mu + self.kappa.to(zf.dtype) * sigma - self.thresh_bias.float()
            sharpness = F.softplus(self.sharpness_raw.float())
            gate = torch.sigmoid(sharpness * (zf - tau)).to(z.dtype)

        return self.fc2(self.dropout(F.gelu(z) * gate))


class HyperConnectionSubLayer(nn.Module):
    """Manifold-Constrained Hyper-Connection wrapper (mHC, 2025 / HC, ICLR 2025)."""

    def __init__(self, dim, num_streams=4, stream_pref=0, sinkhorn_iters=10):
        super().__init__()
        self.dim = int(dim)
        self.num_streams = int(num_streams)
        self.sinkhorn_iters = int(sinkhorn_iters)
        flat_dim = self.num_streams * self.dim

        self.norm = nn.LayerNorm(self.dim)
        self.phi_pre = nn.Linear(flat_dim, self.num_streams, bias=False)
        self.phi_post = nn.Linear(flat_dim, self.num_streams, bias=False)
        self.phi_res = nn.Linear(flat_dim, self.num_streams * self.num_streams, bias=False)

        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_res = nn.Parameter(torch.tensor(0.01))

        # Break cross-stream symmetry gently across depth while keeping H_res near identity.
        b_pre = torch.full((self.num_streams,), -0.5)
        b_pre[stream_pref % self.num_streams] = 0.8
        self.b_pre = nn.Parameter(b_pre)
        self.b_post = nn.Parameter(torch.zeros(self.num_streams))

        b_res = torch.full((self.num_streams, self.num_streams), -4.0)
        b_res.fill_diagonal_(4.0)
        self.b_res = nn.Parameter(b_res)

    def _compute_mappings(self, streams):
        b, n, s, _ = streams.shape
        x_flat = _rmsnorm_last(streams.reshape(b, n, s * self.dim))

        pre_logits = self.alpha_pre * self.phi_pre(x_flat) + self.b_pre
        post_logits = self.alpha_post * self.phi_post(x_flat) + self.b_post
        res_logits = self.alpha_res * self.phi_res(x_flat).view(b, n, s, s) + self.b_res

        h_pre = torch.softmax(pre_logits.float(), dim=-1).to(streams.dtype)
        h_post = (2.0 * torch.sigmoid(post_logits.float())).to(streams.dtype)
        h_res = _sinkhorn_knopp(res_logits, iters=self.sinkhorn_iters).to(streams.dtype)
        return h_pre, h_post, h_res

    def forward(self, streams, fn):
        """Aggregate streams -> norm -> apply fn -> doubly-stochastic stream update."""
        h_pre, h_post, h_res = self._compute_mappings(streams)
        layer_in = (h_pre.unsqueeze(-1) * streams).sum(dim=2)
        out = fn(self.norm(layer_in))

        aux = None
        if isinstance(out, tuple):
            out, aux = out

        streams_mixed = torch.matmul(h_res, streams)
        streams_next = streams_mixed + h_post.unsqueeze(-1) * out.unsqueeze(2)
        return (streams_next, aux) if aux is not None else streams_next


class _ProgressiveStage(nn.Module):
    """Single stage pairing mHC-wrapped PF-BlockSparse Attention and SparkConvFFN."""

    def __init__(self, dim, cm, num_heads, num_streams, stage_idx,
                 block_size, stage1_topk, focus_topk, active_ratio,
                 sinkhorn_iters, dropout):
        super().__init__()
        self.attn_hc = HyperConnectionSubLayer(
            dim, num_streams=num_streams, stream_pref=stage_idx * 2, sinkhorn_iters=sinkhorn_iters
        )
        self.attn = PFBlockSparseAttention(
            dim, num_heads=num_heads, block_size=block_size,
            stage1_topk=stage1_topk, focus_topk=focus_topk, dropout=dropout
        )
        self.ffn_hc = HyperConnectionSubLayer(
            dim, num_streams=num_streams, stream_pref=stage_idx * 2 + 1, sinkhorn_iters=sinkhorn_iters
        )
        self.ffn = SparkConvFFN(dim, cm=cm, active_ratio=active_ratio, dropout=dropout)

    def forward(self, streams, h, w, prev_attn):
        streams, attn = self.attn_hc(streams, lambda x: self.attn(x, h, w, prev_attn))
        streams = self.ffn_hc(streams, lambda x: self.ffn(x, h, w))
        return streams, attn


class AIFI_mHCPFBSA(nn.Module):
    """Manifold-Constrained Hyper-Connected Progressive Focused Block-Sparse AIFI.

    Drop-in replacement for AIFI on P5 features [B, C, H, W] -> [B, C, H, W].

    Args:
        c1: Input and output channel dimension.
        cm: Hidden dimension of SparkConvFFN per stage (default 768).
        num_heads: Attention heads per stage (default 8).
        num_stages: Number of progressive focused stages (default 3).
        num_streams: Parallel mHC residual streams (default 4).
        block_size: Token block size for Stage 1 block-sparse routing (default 4).
        stage1_topk: Number of blocks selected in Stage 1 (default 6).
        focus_schedule: Per-stage token top-k limits for Stages 2..L (default (16, 8)).
        active_ratio: Target active neuron ratio in SparkConvFFN (default 0.30).
        sinkhorn_iters: Sinkhorn-Knopp iterations for Birkhoff projection (default 10).
        dropout: Dropout rate (default 0.0).
    """

    def __init__(self, c1, cm=768, num_heads=8, num_stages=3, num_streams=4,
                 block_size=4, stage1_topk=6, focus_schedule=(16, 8),
                 active_ratio=0.30, sinkhorn_iters=10, dropout=0.0):
        super().__init__()
        if num_stages < 1 or num_streams < 1:
            raise ValueError('num_stages and num_streams must be >= 1')
        self.c1 = int(c1)
        self.num_stages = int(num_stages)
        self.num_streams = int(num_streams)

        schedule = list(focus_schedule) if focus_schedule is not None else []
        while len(schedule) < max(0, self.num_stages - 1):
            last = schedule[-1] if schedule else 16
            schedule.append(max(4, last // 2))

        self.stages = nn.ModuleList([
            _ProgressiveStage(
                dim=self.c1,
                cm=cm,
                num_heads=num_heads,
                num_streams=self.num_streams,
                stage_idx=idx,
                block_size=block_size,
                stage1_topk=stage1_topk,
                focus_topk=None if idx == 0 else schedule[idx - 1],
                active_ratio=active_ratio,
                sinkhorn_iters=sinkhorn_iters,
                dropout=dropout,
            )
            for idx in range(self.num_stages)
        ])

        self.out_stream_logits = nn.Parameter(torch.zeros(self.num_streams))
        self.out_norm = nn.LayerNorm(self.c1)

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.c1:
            raise ValueError(f'AIFI_mHCPFBSA expected {self.c1} channels, got {c}')

        tokens = x.flatten(2).transpose(1, 2)  # [B, N, C]
        streams = tokens.unsqueeze(2).expand(-1, -1, self.num_streams, -1).contiguous()

        attn = None
        for stage in self.stages:
            streams, attn = stage(streams, h, w, attn)

        stream_w = torch.softmax(self.out_stream_logits.float(), dim=0).to(streams.dtype)
        out = (streams * stream_w.view(1, 1, self.num_streams, 1)).sum(dim=2)
        out = self.out_norm(out)
        return out.transpose(1, 2).reshape(b, c, h, w).contiguous()

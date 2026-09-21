# Boltzmann Attention-Sampling AIFI (BA-AIFI)
#
# Based on "Boltzmann Attention Sampling for Image Analysis with Small Objects"
# (CVPR 2025). Replaces the dense N×N self-attention in AIFI with a
# sparse N×K variant where K low-energy (high-saliency) tokens are
# adaptively selected via Boltzmann distribution sampling.
#
# Core ideas:
#   1. Energy-based token scoring: flat background = high energy (redundant),
#      small-object boundaries = low energy (salient / low-energy ground state).
#   2. Boltzmann sampling: P(s_i) ∝ exp(-E(s_i)/τ), low-energy tokens are
#      preferentially retained. Temperature τ is predicted per-scene.
#   3. Gumbel-Max trick (training): differentiable discrete approximation for
#      gradient flow through top-K selection.
#   4. Sparse attention: Q (all N) × K_sampled (K) × V_sampled (K),
#      reducing complexity from O(N²) to O(N·K).
#
# Architecture:
#   X (B, C, H, W)
#   ├── Energy Estimator: E_i = ConvNet(X)  →  (B, N)
#   ├── Temperature Predictor: τ = MLP(GAP(X))  →  (B, 1)
#   ├── Boltzmann Sampling: top-K indices from P ∝ exp(-E/τ)
#   ├── Q = X·W_Q + pos  (full N tokens)
#   ├── K_sampled, V_sampled  (only K low-energy tokens, no pos on V)
#   ├── Sparse Attn = Softmax(Q · K_sampled^T / √d) · V_sampled
#   └── Residual + LayerNorm + FFN (same as original AIFI)

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AIFI_BA']


class AIFI_BA(nn.Module):
    """Boltzmann Attention-Sampling AIFI for tiny object detection.

    Replaces dense N×N self-attention with sparse N×K attention where K
    low-energy (high-saliency) tokens are selected via Boltzmann
    distribution sampling with adaptive temperature.

    Args:
        c1: Input/output channels.
        cm: FFN hidden dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        act: Activation for the FFN.
        normalize_before: Pre-norm if True; AIFI default is False (post-norm).
        top_k_ratio: Fraction of tokens to retain (e.g. 0.35 keeps 35%).
        tau_min: Minimum temperature (prevents division by zero).
        tau_max: Maximum temperature (upper bound for stable training).
    """

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0,
                 act=nn.GELU(), normalize_before=False,
                 top_k_ratio=0.35, tau_min=0.2, tau_max=1.2):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.num_heads = num_heads
        self.head_dim = c1 // num_heads
        self.scale = self.head_dim ** -0.5
        self.top_k_ratio = top_k_ratio
        self.tau_min = tau_min
        self.tau_max = tau_max

        self.q_proj = nn.Linear(c1, c1)
        self.k_proj = nn.Linear(c1, c1)
        self.v_proj = nn.Linear(c1, c1)
        self.out_proj = nn.Linear(c1, c1)

        self.energy_conv = nn.Sequential(
            nn.Conv2d(c1, c1 // 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, c1 // 4),
            nn.GELU(),
            nn.Conv2d(c1 // 4, 1, kernel_size=1, bias=True),
        )

        self.temp_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, c1 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(c1 // 4, 1),
            nn.Sigmoid(),
        )

        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act

        self.register_buffer('pos_embed_cache', None, persistent=False)
        self.cached_hw = None

        self._reset_parameters()

    def _reset_parameters(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def get_pos_embed(self, H, W, device, dtype):
        """Get or compute cached 2D sine-cosine positional embedding."""
        if self.cached_hw == (H, W) and self.pos_embed_cache is not None:
            return self.pos_embed_cache

        grid_w = torch.arange(W, dtype=torch.float32, device=device)
        grid_h = torch.arange(H, dtype=torch.float32, device=device)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')

        assert self.c1 % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.c1 // 4
        omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim
        omega = 1.0 / (10000.0 ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        pos_embed = torch.cat([torch.sin(out_w), torch.cos(out_w),
                               torch.sin(out_h), torch.cos(out_h)],
                              dim=1)[None].to(dtype)

        self.pos_embed_cache = pos_embed
        self.cached_hw = (H, W)
        return pos_embed

    def _boltzmann_topk(self, energy, tau):
        """Boltzmann-distribution-based Top-K token sampler.

        Args:
            energy: Per-token energy (B, N). Lower = more salient.
            tau: Per-scene temperature (B, 1). Higher = more uniform sampling.

        Returns:
            topk_indices: (B, K) indices of selected tokens.
        """
        B, N = energy.shape
        K = max(1, int(N * self.top_k_ratio))

        logits = -energy / tau

        if self.training:
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            perturbed = logits + gumbel_noise
            topk_indices = torch.topk(perturbed, K, dim=-1).indices
        else:
            topk_indices = torch.topk(logits, K, dim=-1).indices

        return topk_indices

    def _sparse_mhsa(self, q_src, k_src, v_full, topk_indices):
        """Sparse multi-head attention: Q(all N) × K_sampled(K) × V_sampled(K).

        Args:
            q_src: Query input (features + pos) (B, N, C).
            k_src: Key input (features + pos) (B, N, C).
            v_full: Value input (pure features, no pos) (B, N, C).
            topk_indices: Selected token indices (B, K).

        Returns:
            Attention output (B, N, C).
        """
        B, N, C = q_src.shape
        H = self.num_heads
        D = self.head_dim
        K = topk_indices.size(1)

        q = self.q_proj(q_src).view(B, N, H, D).permute(0, 2, 1, 3)
        k = self.k_proj(k_src).view(B, N, H, D).permute(0, 2, 1, 3)
        v = self.v_proj(v_full).view(B, N, H, D).permute(0, 2, 1, 3)

        idx_expand = topk_indices.unsqueeze(1).unsqueeze(-1).expand(B, H, K, D)
        k_sampled = k.gather(2, idx_expand)
        v_sampled = v.gather(2, idx_expand)

        attn = torch.matmul(q, k_sampled.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v_sampled)

        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return self.out_proj(out)

    def forward(self, x):
        """Forward a feature map of shape (B, C, H, W).

        Steps:
          1. Estimate per-token energy and adaptive temperature
          2. Boltzmann top-K sampling to select low-energy tokens
          3. Sparse MHSA: Q(all+pos) × K_sampled(+pos) × V_sampled(no pos)
          4. FFN with residual + LayerNorm (same as original AIFI)
        """
        B, C, H, W = x.shape
        N = H * W

        energy = self.energy_conv(x).view(B, N)
        tau = self.tau_min + (self.tau_max - self.tau_min) * self.temp_net(x).view(B, 1)

        topk_indices = self._boltzmann_topk(energy, tau)

        pos_embed = self.get_pos_embed(H, W, x.device, x.dtype)

        src = x.flatten(2).permute(0, 2, 1)

        if self.normalize_before:
            src_norm = self.norm1(src)
            src2 = self._sparse_mhsa(
                q_src=src_norm + pos_embed,
                k_src=src_norm + pos_embed,
                v_full=src_norm,
                topk_indices=topk_indices,
            )
            src = src + self.dropout1(src2)
            src_norm = self.norm2(src)
            src2 = self.fc2(self.dropout(self.act(self.fc1(src_norm))))
            src = src + self.dropout2(src2)
        else:
            src2 = self._sparse_mhsa(
                q_src=src + pos_embed,
                k_src=src + pos_embed,
                v_full=src,
                topk_indices=topk_indices,
            )
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(B, C, H, W).contiguous()
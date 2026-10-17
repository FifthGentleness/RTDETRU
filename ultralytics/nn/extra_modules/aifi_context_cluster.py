# CC-AIFI: Content-adaptive Context-Cluster AIFI (AIFI_ContextCluster).
#
# Problem addressed: the global modeling in this network is content-diffuse.
# FLatten (DSAWACGAv5) uses a static focusing mapping; the DSA branches are
# fixed conv receptive fields; P2SPDOKMFSV7 fuses with a fixed topology.
# No module organizes tokens ADAPTIVELY according to their content.
#
# Mechanism (Context Clusters, ICCV 2023 oral; Slot Attention, NeurIPS 2020):
#   1. K learnable anchor centers compete for the P5 tokens: for a few rounds,
#      each token is softly assigned over the centers (cosine similarity) and
#      each center is re-estimated as the weighted mean of its members.
#      The resulting grouping is CONTENT-ADAPTIVE - clusters correspond to
#      semantic regions / object groups of the current image.
#   2. The K centers (K=64) exchange information through one lightweight
#      self-attention round (K x K, negligible cost) - the global exchange
#      happens in the compact cluster space.
#   3. Cluster features are broadcast back to tokens through the assignment
#      matrix and a zero-initialized output projection (single-gate philosophy:
#      stable start, gradient flows from step 0).
#
# Zero-overlap statement: no wavelet/frequency, no multi-scale DWConv, no
# large kernel, no linear attention, no channel attention - pure adaptive
# token grouping.

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AIFI_ContextCluster']


class ContextClusterMixer(nn.Module):
    """Content-adaptive cluster-exchange token mixer (replaces MHSA)."""

    def __init__(self, dim, num_clusters=64, num_heads=4, num_iters=2):
        super().__init__()
        self.dim = dim
        self.num_clusters = int(num_clusters)
        self.num_iters = int(num_iters)

        # Learnable cluster anchors (content-adaptive refinement per image).
        self.centers = nn.Parameter(torch.randn(num_clusters, dim) * 0.02)
        self.center_norm = nn.LayerNorm(dim)

        # One round of inter-center exchange (K x K attention).
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True)

        # Zero-initialized output projection (single-gate philosophy).
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        b, n, c = x.shape
        x_norm = F.normalize(x, dim=-1)

        centers = self.centers.unsqueeze(0).expand(b, -1, -1)  # (b, K, c)
        assign = None
        for _ in range(self.num_iters):
            # Soft assignment of every token over the K centers (cosine).
            c_norm = F.normalize(centers, dim=-1)
            sim = torch.einsum('bnc,bkc->bnk', x_norm, c_norm) * (c ** 0.5)
            assign = sim.softmax(dim=-1)                        # (b, n, K)

            # Re-estimate each center as the weighted mean of its members.
            denom = assign.sum(dim=1).unsqueeze(-1) + 1e-6      # (b, K, 1)
            centers = torch.einsum('bnk,bnc->bkc', assign, x) / denom
            centers = self.center_norm(centers)

        # Inter-center global exchange (K x K - negligible cost).
        updated, _ = self.cross_attn(centers, centers, centers)
        centers = centers + updated

        # Broadcast cluster features back to tokens.
        out = torch.einsum('bnk,bkc->bnc', assign, centers)     # (b, n, c)
        return self.out_proj(out)


class AIFI_ContextCluster(nn.Module):
    """AIFI with a content-adaptive context-cluster mixer.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_clusters: Number of adaptive cluster centers (K).
        num_heads: Heads of the inter-center exchange attention.
        num_iters: Assignment/update rounds.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=2048, num_clusters=64, num_heads=4, num_iters=2,
                 dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = ContextClusterMixer(
            dim=c1, num_clusters=num_clusters, num_heads=num_heads,
            num_iters=num_iters)
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
                f'AIFI_ContextCluster expected {self.c1} channels, got {c}')

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
        return (f'c1={self.c1}, num_clusters={self.attn.num_clusters}, '
                f'num_iters={self.attn.num_iters}')

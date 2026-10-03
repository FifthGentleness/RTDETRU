# HOPS-style attention filtering / affinity enhancement for RT-DETR AIFI.
#
# Reference:
#   HOPS: Hierarchical Open-vocabulary Part Segmentation with Attention-Aware
#   Filtering and Affinity-Guided Enhancement. CVPR 2026.
#
# Adaptation: use AIFI attention heads and a small objectness head as the
# foreground prior, then apply AFM-style filtering and AEM-style affinity
# propagation. The original MHSA/FFN residual flow is retained.

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AttentionAwareFiltering', 'AffinityGuidedEnhancement', 'AIFI_HOPS']


class AttentionAwareFiltering(nn.Module):
    """AFM-style attention-aware filtering.

    attention_matrices_list: list of [B,N,N] maps.
    P_coarse: [B,N,C] objectness/class logits or scores.
    """

    def __init__(self, num_layers=12, K=6, tau=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.K = K
        self.tau = tau

    def attention_based_patch_filtering(self, attention_matrices_list):
        attention_matrices = torch.stack(attention_matrices_list, dim=1)
        a_bar = torch.mean(attention_matrices, dim=(2, 3), keepdim=True)
        layer_masks = (attention_matrices > a_bar).float()
        vote_count = torch.sum(layer_masks, dim=1)
        return (vote_count > self.K).float()

    def class_confidence_based_filtering(self, p_coarse):
        max_prob, _ = torch.max(p_coarse, dim=2, keepdim=True)
        relative_confidence = p_coarse - max_prob
        return (relative_confidence > -self.tau).float()

    def attention_aware_cost_refinement(self, attention_matrices_list, m_attn, m_cls, p_coarse):
        attention_matrices = torch.stack(attention_matrices_list, dim=1)
        b, l, n, _ = attention_matrices.shape
        m_attn_expanded = m_attn.unsqueeze(1).expand(-1, l, -1, -1)
        masked_attention = attention_matrices * m_attn_expanded
        avg_attention = torch.mean(masked_attention, dim=1)
        masked_cost = p_coarse * m_cls
        return torch.bmm(avg_attention, masked_cost)

    def forward(self, attention_matrices_list, p_coarse):
        m_attn = self.attention_based_patch_filtering(attention_matrices_list)
        m_cls = self.class_confidence_based_filtering(p_coarse)
        return self.attention_aware_cost_refinement(attention_matrices_list, m_attn, m_cls, p_coarse)


class AffinityGuidedEnhancement(nn.Module):
    """AEM-style local affinity propagation with Sinkhorn normalization."""

    def __init__(self, num_sinkhorn=3, eps=1e-6):
        super().__init__()
        self.num_sinkhorn = num_sinkhorn
        self.eps = eps

    def sinkhorn(self, x):
        for _ in range(self.num_sinkhorn):
            x = x / x.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            x = x / x.sum(dim=-2, keepdim=True).clamp_min(self.eps)
        return x

    def forward(self, x):
        b, n, c = x.shape
        norm = F.normalize(x, p=2, dim=-1)
        affinity = torch.bmm(norm, norm.transpose(1, 2)) * (c ** 0.5)
        affinity = self.sinkhorn(affinity)
        affinity = 0.5 * (affinity + affinity.transpose(1, 2))
        return torch.bmm(affinity, x)


class AIFI_HOPS(nn.Module):
    """RT-DETR AIFI with HOPS-style background filtering and weak response enhancement."""

    def __init__(self, c1, cm=1024, num_heads=8, K=6, tau=0.2, dropout=0.0,
                 act=nn.GELU(), normalize_before=False):
        super().__init__()
        if c1 % num_heads != 0:
            raise ValueError('c1 must be divisible by num_heads')
        self.c1 = c1
        self.num_heads = num_heads
        self.normalize_before = normalize_before

        self.global_attn = nn.MultiheadAttention(c1, num_heads, dropout=dropout,
                                                 batch_first=True)
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act

        self.afm = AttentionAwareFiltering(num_layers=num_heads, K=K, tau=tau)
        self.aem = AffinityGuidedEnhancement()
        self.obj_head = nn.Linear(c1, 2)
        self.afm_gamma = nn.Parameter(torch.zeros(1))
        self.aem_gamma = nn.Parameter(torch.zeros(1))

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, 'embed_dim must be divisible by 4'
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
            raise ValueError(f'AIFI_HOPS expected {self.c1} channels, got {c}')

        pos = self.build_2d_sincos_position_embedding(w, h, c).to(device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1)

        if self.normalize_before:
            src_norm = self.norm1(src)
            attn_out, attn_weights = self.global_attn(src_norm + pos, src_norm + pos,
                                                      src_norm, need_weights=True,
                                                      average_attn_weights=False)
            src2 = attn_out
            src2 = src2 + self.hops_refine(src2, attn_weights)
            src = src + self.dropout1(src2)
            src_norm = self.norm2(src)
            src = src + self.dropout2(self.fc2(self.dropout(self.act(self.fc1(src_norm)))))
        else:
            src_norm = src + pos
            attn_out, attn_weights = self.global_attn(src_norm, src_norm, src,
                                                      need_weights=True,
                                                      average_attn_weights=False)
            src2 = attn_out
            src2 = src2 + self.hops_refine(src2, attn_weights)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src = src + self.dropout2(self.fc2(self.dropout(self.act(self.fc1(src)))))
            src = self.norm2(src)

        return src.permute(0, 2, 1).reshape(b, c, h, w).contiguous()

    def hops_refine(self, src2, attn_weights):
        """src2: B,N,C; attn_weights: B,H,N,N."""
        p_coarse = self.obj_head(src2)
        p_refined = self.afm(list(attn_weights.unbind(dim=1)), p_coarse)
        foreground = torch.softmax(p_refined, dim=-1)[..., 0:1]
        afm_out = src2 * foreground
        aem_out = self.aem(src2)
        return self.afm_gamma * afm_out + self.aem_gamma * aem_out

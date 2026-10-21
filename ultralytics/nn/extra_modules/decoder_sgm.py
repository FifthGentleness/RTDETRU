"""
SGM: Surprise-Gated Memory cross-attention (RT-DETR decoder innovation).

Inspired by:
  - Titans: Learning to Memorize at Test Time (Behrouz et al., NeurIPS 2025):
    a neural long-term memory whose write strength is driven by "surprise" -
    the mismatch between what the memory predicts and what it observes.

Mechanism:
  - A block-recurrent memory M (seeded by learned persistent memory slots) is
    built from the multi-scale encoder features in Morton (Z-order) scan
    order. Each token first READS the memory (predicted context); the
    normalized mismatch ||token - read|| defines its surprise. High-surprise
    tokens - isolated small-object evidence that the surrounding context
    cannot predict - are written into the memory strongly, while redundant
    background (predictable from context) decays away.
  - The memory read uses PROJECTION-FREE cosine attention (normalized tokens
    x normalized slots with a learnable temperature): no per-token QKV/out
    projections, which keeps the added compute of the whole decoder within
    ~3 GFLOPs (the total model stays under 80 GFLOPs).
  - Decoder queries then read the memory globally and, through a zero-init
    gated path, the spatially nearest tokens around their reference points
    (rule-based window, no learned offset sampling).

The innovation replaces the cross-attention of matching queries only;
denoising queries keep the original MSDeformAttn path and self-attention is
the baseline masked attention (DN training semantics unchanged).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.transformer import DeformableTransformerDecoderLayer

from .decoder_innov_common import (_InnovCrossAttnLayer, _RTDETRDecoderInnov,
                                   _morton_codes, _token_centers,
                                   DeformableTransformerDecoderInnov)

__all__ = ('SurpriseMemoryBlock', 'DecoderLayerSGM', 'RTDETRDecoderSGM')


class SurpriseMemoryBlock(nn.Module):
    """Block-recurrent surprise-gated neural memory over feature tokens."""

    def __init__(self, d_model=256, n_slots=48, n_blocks=2, n_local=64, dropout=0.):
        super().__init__()
        self.n_slots = n_slots
        self.n_blocks = n_blocks
        self.n_local = n_local
        self.grid = 64  # morton quantization for the token scan order

        # persistent memory slots (Titans: persistent memory independent of input)
        self.mem = nn.Parameter(torch.randn(1, n_slots, d_model) * 0.02)
        self.norm_m = nn.LayerNorm(d_model)
        self.norm_x = nn.LayerNorm(d_model)
        self.tau = nn.Parameter(torch.tensor(0.25))    # cosine-attention temperature
        self.surprise = nn.Linear(d_model, 1)          # mismatch -> surprise logit
        self.write_gate = nn.Linear(2 * d_model, 1)    # content-aware write gate
        self.slot_logits = nn.Linear(d_model, n_slots)  # soft slot assignment

        # query readout: global memory read (MHA on N_q=300 queries, cheap) +
        # zero-init gated local fine read (projection-free cosine attention)
        self.norm_q = nn.LayerNorm(d_model)
        self.q_read = nn.MultiheadAttention(d_model, 4, dropout=dropout, batch_first=True)
        self.local_out = nn.Linear(d_model, d_model)
        self.local_gate = nn.Parameter(torch.zeros(1))
        self.out_gate = nn.Parameter(torch.zeros(1))   # zero-init: SGM refines the deformable path

    def build_memory(self, feats, centers):
        """Build the surprise-gated memory from multi-scale tokens.

        Args:
            feats (Tensor):  [B, T, C] flattened multi-scale features.
            centers (Tensor): [T, 2] token centers in [0, 1].

        Returns:
            Tensor: [B, n_slots, C] memory.
        """
        B, T, C = feats.shape
        idx = _morton_codes(centers.unsqueeze(0).expand(B, -1, -1), self.grid).argsort(1)
        xs = torch.gather(feats, 1, idx.unsqueeze(-1).expand(-1, -1, C))
        chunk = (T + self.n_blocks - 1) // self.n_blocks
        pad = self.n_blocks * chunk - T
        if pad:
            xs = torch.cat([xs, xs[:, -1:].expand(-1, pad, -1)], 1)

        M = self.mem.to(dtype=feats.dtype).expand(B, -1, -1)
        Mn = self.norm_m(M)
        for b in range(self.n_blocks):
            x = xs[:, b * chunk:(b + 1) * chunk]
            xn = self.norm_x(x)
            # projection-free cosine read: token content directly queries memory
            xu = F.normalize(xn, dim=-1)
            mu = F.normalize(Mn, dim=-1)
            attn = torch.softmax(xu @ mu.transpose(-1, -2) * self.tau, dim=-1)  # [B, L, S]
            read = attn @ Mn                                                    # predicted context
            surprise = torch.sigmoid(self.surprise(read - xn))                  # [B, L, 1]
            gate = torch.sigmoid(self.write_gate(torch.cat([xn, read], -1)))
            w = surprise * gate                                                 # write weight
            slot = self.slot_logits(xn).softmax(-1)                             # [B, L, S]
            mass = slot.transpose(1, 2) @ w                                     # [B, S, 1]
            write = (slot.transpose(1, 2) @ (w * x)) / mass.clamp(min=1e-6)
            mass = mass.clamp(max=0.9)
            M = (1.0 - mass) * M + mass * write                                 # gated replace
            Mn = self.norm_m(M)
        return M

    def forward(self, q, ref_boxes, feats, shapes, padding_mask=None):
        """Cross-read for matching queries.

        Args:
            q (Tensor):     [B, N, C] query content + positional embedding.
            ref_boxes:      [B, N, 4] cxcywh reference boxes in [0, 1].
            feats (Tensor): [B, T, C] flattened multi-scale features.
            shapes (List):  [(h, w), ...] per-level shapes.

        Returns:
            Tensor: [B, N, C] cross-attention output (pre-residual).
        """
        B, N, C = q.shape
        centers = _token_centers(shapes, feats.device, feats.dtype)   # [T, 2]
        M = self.build_memory(feats, centers)
        Mn = self.norm_m(M)
        o_mem, _ = self.q_read(self.norm_q(q), Mn, Mn, need_weights=False)

        # local fine evidence: rule-based nearest-token window (no learned offsets)
        k = min(self.n_local, feats.shape[1])
        dist = torch.cdist(ref_boxes[..., :2], centers.unsqueeze(0).expand(B, -1, -1))
        top = dist.topk(k, dim=2, largest=False).indices                              # [B, N, k]
        local = torch.gather(feats, 1, top.reshape(B, N * k, 1).expand(-1, -1, C)).view(B, N, k, C)
        # projection-free cosine fine read
        qn = F.normalize(self.norm_q(q), dim=-1)                         # [B, N, C]
        ln = F.normalize(self.norm_x(local), dim=-1)                     # [B, N, k, C]
        attn = torch.softmax(torch.einsum('bnc,bnkc->bnk', qn, ln) * self.tau, dim=-1)
        o_loc = self.local_out(torch.einsum('bnk,bnkc->bnc', attn, ln))

        return torch.tanh(self.out_gate) * (o_mem + torch.tanh(self.local_gate) * o_loc)


class DecoderLayerSGM(_InnovCrossAttnLayer):
    """Baseline decoder layer whose matching-query cross-attention is replaced
    by the SGM surprise-gated memory read."""

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0., act=nn.ReLU(),
                 n_levels=4, n_points=4, n_slots=48, n_blocks=2, n_local=64):
        super().__init__(d_model, n_heads, d_ffn, dropout, act, n_levels, n_points)
        self.sgm = SurpriseMemoryBlock(d_model, n_slots, n_blocks, n_local=n_local)

    def forward(self, embed, refer_bbox, feats, shapes, padding_mask=None,
                attn_mask=None, query_pos=None, dn_num=0, dec_boxes=None,
                shared_states=None):
        # self-attention (baseline; dn queries masked)
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1),
                             attn_mask=attn_mask)[0].transpose(0, 1)
        embed = self.norm1(embed + self.dropout1(tgt))
        # cross-attention: baseline deformable EVERYWHERE + zero-init gated SGM
        # refinement on the matching part (identical to baseline at init)
        tgt = self.cross_attn(self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2),
                              feats, shapes, padding_mask)
        if dn_num > 0:
            tgt_mt = self.sgm(self.with_pos_embed(embed[:, dn_num:], query_pos[:, dn_num:]),
                              refer_bbox[:, dn_num:], feats, shapes, padding_mask)
            tgt = torch.cat([tgt[:, :dn_num], tgt[:, dn_num:] + tgt_mt], dim=1)
        else:
            tgt = tgt + self.sgm(self.with_pos_embed(embed, query_pos), refer_bbox,
                                 feats, shapes, padding_mask)
        embed = self.norm2(embed + self.dropout2(tgt))
        return self.forward_ffn(embed)


class RTDETRDecoderSGM(_RTDETRDecoderInnov):
    """RT-DETR decoder + SGM (surprise-gated neural memory cross-attention).

    Extra yaml args (after the standard RTDETRDecoder args):
        n_slots (int, default 48): memory slot count.
        n_blocks (int, default 2): recurrent blocks for memory building.
        n_local (int, default 64): local fine-read window size.
    """

    def _rebuild_decoder(self, hd, nh, ndl, d_ffn, dropout, act, ndp, eval_idx,
                         n_slots=48, n_blocks=2, n_local=64, **_):
        layer = DecoderLayerSGM(hd, nh, d_ffn, dropout, act, self.nl, ndp,
                                n_slots, n_blocks, n_local)
        return DeformableTransformerDecoderInnov(hd, layer, ndl, eval_idx)

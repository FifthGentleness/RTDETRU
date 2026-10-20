"""
SGM: Surprise-Gated Memory cross-attention (RT-DETR decoder innovation).

Inspired by:
  - Titans: Learning to Memorize at Test Time (NeurIPS 2025): a neural
    long-term memory whose write strength is driven by "surprise" - the
    mismatch between what the memory predicts and what it observes.

Mechanism:
  - A block-recurrent memory M (seeded by learned persistent memory slots) is
    built from the multi-scale encoder features in Morton (Z-order) scan
    order. Each token first READS the memory (predicted context); the
    normalized mismatch ||token - read|| defines its surprise. High-surprise
    tokens - isolated small-object evidence that the surrounding context
    cannot predict - are written into the memory strongly, while redundant
    background (predictable from context) decays away.
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

    def __init__(self, d_model=256, n_slots=128, n_blocks=8, n_heads=4, n_local=128,
                 dropout=0.):
        super().__init__()
        self.n_slots = n_slots
        self.n_blocks = n_blocks
        self.n_local = n_local
        self.grid = 64  # morton quantization for the token scan order

        # persistent memory slots (Titans: persistent memory independent of input)
        self.mem = nn.Parameter(torch.randn(1, n_slots, d_model) * 0.02)
        self.norm_m = nn.LayerNorm(d_model)
        self.norm_x = nn.LayerNorm(d_model)
        self.read = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.surprise = nn.Linear(d_model, 1)          # mismatch -> surprise logit
        self.write_gate = nn.Linear(2 * d_model, 1)    # content-aware write gate
        self.slot_logits = nn.Linear(d_model, n_slots)  # soft slot assignment

        # query readout: global memory read + zero-init gated local fine read
        self.norm_q = nn.LayerNorm(d_model)
        self.q_read = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.local_read = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.local_gate = nn.Parameter(torch.zeros(1))

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
            read, _ = self.read(xn, Mn, Mn, need_weights=False)      # predicted context
            surprise = torch.sigmoid(self.surprise(read - xn))        # [B, L, 1]
            gate = torch.sigmoid(self.write_gate(torch.cat([xn, read], -1)))
            w = surprise * gate                                       # write weight
            slot = self.slot_logits(xn).softmax(-1)                   # [B, L, S]
            mass = slot.transpose(1, 2) @ w                           # [B, S, 1]
            write = (slot.transpose(1, 2) @ (w * x)) / mass.clamp(min=1e-6)
            mass = mass.clamp(max=0.9)
            M = (1.0 - mass) * M + mass * write                       # gated replace
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
        qf = self.norm_q(q).reshape(B * N, 1, C)
        lf = self.norm_x(local).reshape(B * N, k, C)
        o_loc, _ = self.local_read(qf, lf, lf, need_weights=False)
        o_loc = o_loc.reshape(B, N, C)

        return o_mem + torch.tanh(self.local_gate) * o_loc


class DecoderLayerSGM(_InnovCrossAttnLayer):
    """Baseline decoder layer whose matching-query cross-attention is replaced
    by the SGM surprise-gated memory read."""

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0., act=nn.ReLU(),
                 n_levels=4, n_points=4, n_slots=128, n_blocks=8, n_local=128):
        super().__init__(d_model, n_heads, d_ffn, dropout, act, n_levels, n_points)
        self.sgm = SurpriseMemoryBlock(d_model, n_slots, n_blocks, n_local=n_local)

    def _innov_cross(self, q, embed, ref_boxes, feats, shapes, padding_mask=None):
        return self.sgm(q, ref_boxes, feats, shapes, padding_mask)


class RTDETRDecoderSGM(_RTDETRDecoderInnov):
    """RT-DETR decoder + SGM (surprise-gated neural memory cross-attention).

    Extra yaml args (after the standard RTDETRDecoder args):
        n_slots (int, default 128): memory slot count.
        n_blocks (int, default 8): recurrent blocks for memory building.
        n_local (int, default 128): local fine-read window size.
    """

    def _rebuild_decoder(self, hd, nh, ndl, d_ffn, dropout, act, ndp, eval_idx,
                         n_slots=128, n_blocks=8, n_local=128, **_):
        layer = DecoderLayerSGM(hd, nh, d_ffn, dropout, act, self.nl, ndp,
                                n_slots, n_blocks, n_local)
        return DeformableTransformerDecoderInnov(hd, layer, ndl, eval_idx)

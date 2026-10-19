"""
CLDM: Cross-Layer Detection Memory (RT-DETR decoder innovation).

Inspired by:
  - SAM 2 / SAM 3 memory attention (2024/2025): storing high-confidence target
    states in a memory bank and retrieving them via memory attention.

A fixed-capacity bank stores (detached) query snapshots whose confidence
exceeds `tau`. Later decoder layers retrieve this evidence through a small
gated cross-attention, so a partially-occluded target whose query was
discriminative in a shallow layer cannot be 'dragged away' by overlapping
neighbors in deeper layers.
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.transformer import DeformableTransformerDecoderLayer

from .decoder_innov_common import _RTDETRDecoderInnov, DeformableTransformerDecoderInnov

__all__ = ('CLDMMemory', 'RTDETRDecoderCLDM')


class CLDMMemory(nn.Module):
    """Cross-layer detection memory bank (SAM-style memory attention on the
    decoder depth axis).

    Args:
        d_model (int): query feature dim.
        n_heads (int): heads of the memory cross-attention.
        max_mem (int): memory capacity (per image).
        tau (float): confidence threshold for memory writes.
    """

    def __init__(self, d_model=256, n_heads=2, max_mem=64, tau=0.3):
        super().__init__()
        self.max_mem, self.tau = max_mem, tau
        self.box_mlp = nn.Sequential(nn.Linear(4, d_model), nn.ReLU(inplace=True),
                                     nn.Linear(d_model, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.alpha = nn.Parameter(torch.zeros(1))

    def reset(self, B, N, device, dtype):
        """Reset the bank at the beginning of every decoder forward pass."""
        self.mem_feat = torch.zeros(B, self.max_mem, self.attn.embed_dim,
                                    device=device, dtype=dtype)
        self.mem_box = torch.zeros(B, self.max_mem, 4, device=device, dtype=dtype)
        self.mem_valid = torch.zeros(B, self.max_mem, dtype=torch.bool, device=device)
        self.ptr = torch.zeros(B, dtype=torch.long, device=device)
        self.recorded = torch.zeros(B, N, dtype=torch.bool, device=device)

    def read(self, x, boxes):
        """Retrieve memory evidence into current queries.

        Args:
            x (Tensor): [B, N, C] matching-part queries.
            boxes (Tensor): [B, N, 4] current reference boxes (sigmoid, cxcywh).
        """
        if not bool(self.mem_valid.any()):
            return x
        q = x + self.box_mlp(boxes)
        k = self.mem_feat + self.box_mlp(self.mem_box)
        kpm = ~self.mem_valid
        empty = kpm.all(1)
        if bool(empty.any()):            # let empty samples attend a zero slot
            kpm = kpm.clone()
            kpm[empty, 0] = False
        out = self.attn(q, k, self.mem_feat, key_padding_mask=kpm, need_weights=False)[0]
        return self.norm(x + torch.tanh(self.alpha) * out)

    @torch.no_grad()
    def update(self, x, boxes, scores):
        """Write new confident detections (detached) into the bank.

        Args:
            x (Tensor): [B, N, C] matching-part queries.
            boxes (Tensor): [B, N, 4] refined boxes (sigmoid, cxcywh).
            scores (Tensor): [B, N] max-class sigmoid confidence.
        """
        cand = (scores > self.tau) & (~self.recorded)
        for b in range(x.shape[0]):
            idx = cand[b].nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            room = self.max_mem - int(self.ptr[b])
            if room <= 0:
                continue
            idx = idx[torch.argsort(scores[b, idx], descending=True)][:room]
            t = idx.numel()
            p = int(self.ptr[b])
            self.mem_feat[b, p:p + t] = x[b, idx].detach()
            self.mem_box[b, p:p + t] = boxes[b, idx].detach()
            self.mem_valid[b, p:p + t] = True
            self.recorded[b, idx] = True
            self.ptr[b] += t


class RTDETRDecoderCLDM(_RTDETRDecoderInnov):
    """RT-DETR decoder + cross-layer detection memory bank (SAM-style memory
    attention along the decoder depth axis). Standard decoder layers.

    Extra yaml args (after the standard RTDETRDecoder args):
        mem_heads (int, default 2): memory attention heads.
        max_mem (int, default 64): memory capacity per image.
        mem_tau (float, default 0.3): write confidence threshold.
    """

    def _rebuild_decoder(self, hd, nh, ndl, d_ffn, dropout, act, ndp, eval_idx,
                         mem_heads=2, max_mem=64, mem_tau=0.3, **_):
        layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act,
                                                  self.nl, ndp)
        cldm = CLDMMemory(hd, mem_heads, max_mem, mem_tau)
        return DeformableTransformerDecoderInnov(hd, layer, ndl, eval_idx, cldm=cldm)

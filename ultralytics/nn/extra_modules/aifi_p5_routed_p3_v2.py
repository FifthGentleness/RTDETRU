"""P5-routed bidirectional cross-scale attention between P5 and P3 (v2).

Builds on AIFI_P5RoutedP3 while aligning P5 capacity with BlockSparse (cm=2048),
adding a spatially adaptive foreground writeback gate for P3 detail refinement,
and feeding a bounded, zero-initialized P3 saliency summary back to P5.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .aifi_block_sparse import AIFI_BlockSparse

__all__ = ("AIFI_P5RoutedP3V2",)


class AIFI_P5RoutedP3V2(nn.Module):
    """Map [P3, projected P5] to [encoded P5, refined P3] with adaptive gates."""

    block_size = 4
    topk = 4
    stride_ratio = 4

    def __init__(self, channels, cm=2048, num_heads=8, fine_dim=96, fine_heads=4):
        super().__init__()
        c3, c5 = channels
        if num_heads < 1 or c5 % num_heads or c5 % 4:
            raise ValueError("P5 channels must be divisible by num_heads and by four")
        if fine_heads < 1 or fine_dim < 1 or fine_dim % fine_heads:
            raise ValueError("fine_dim must be positive and divisible by fine_heads")

        self.num_heads = num_heads
        self.head_dim = c5 // num_heads
        self.fine_heads = fine_heads
        self.fine_head_dim = fine_dim // fine_heads

        # Keep P5 encoder identical in structure and width (cm=2048) to BlockSparse.
        self.qkv = nn.Linear(c5, 3 * c5)
        self.proj = nn.Linear(c5, c5)
        self.norm1 = nn.LayerNorm(c5)
        self.norm2 = nn.LayerNorm(c5)
        self.fc1 = nn.Linear(c5, cm)
        self.fc2 = nn.Linear(cm, c5)

        # Shared-route P3 detail attention path.
        self.fine_in = nn.Sequential(nn.Linear(c3, fine_dim), nn.LayerNorm(fine_dim))
        self.fine_qk = nn.Linear(fine_dim, 2 * fine_dim)
        self.fine_value = nn.Linear(fine_dim, fine_dim)
        self.p5_context = nn.Linear(c5, fine_dim, bias=False)
        self.fine_phase = nn.Parameter(torch.empty(self.stride_ratio ** 2, fine_dim))
        nn.init.normal_(self.fine_phase, std=0.02)
        self.fine_out = nn.Linear(fine_dim, c3, bias=False)
        self.fine_gain = nn.Parameter(torch.full((c3,), 0.5))

        # Foreground-conditioned spatial gate for P3 writeback (sigmoid(0)=0.5 at init).
        self.p3_gate = nn.Linear(2 * fine_dim, 1)
        nn.init.zeros_(self.p3_gate.weight)
        nn.init.zeros_(self.p3_gate.bias)

        # Zero-initialized bounded P3->P5 detail feedback.
        self.p3_to_p5 = nn.Sequential(
            nn.LayerNorm(fine_dim),
            nn.Linear(fine_dim, c5, bias=False),
        )
        self.p5_fb_gain = nn.Parameter(torch.zeros(c5))

    def _route(self, tokens, sites):
        """Select cells shared by a block of P5 queries and their P3 children."""
        descriptors = tokens.mean(dim=2)
        similarity = (descriptors @ descriptors.transpose(-2, -1)) * self.head_dim ** -0.5
        selected = similarity.topk(min(self.topk, tokens.shape[1]), dim=-1).indices
        own = torch.arange(tokens.shape[1], device=tokens.device).view(1, -1, 1).expand(tokens.shape[0], -1, 1)
        own_missing = ~(selected == own).any(dim=-1, keepdim=True)
        blocks = torch.cat((selected, own), dim=-1)
        block_valid = torch.cat((torch.ones_like(selected, dtype=torch.bool), own_missing), dim=-1)
        offsets = torch.arange(self.block_size, device=tokens.device)
        cells = (blocks.unsqueeze(-1) * self.block_size + offsets).flatten(-2)
        valid = block_valid.repeat_interleave(self.block_size, dim=-1) & (cells < sites)
        return cells, valid

    def _encode_p5(self, tokens, cells, valid, batch_ids):
        """Use the shared block route with Post-Norm and full cm=2048 FFN."""
        batch, groups, block_size, _ = tokens.shape
        q, k, v = self.qkv(tokens).reshape(batch, -1, 3, self.num_heads, self.head_dim).unbind(2)
        q = q.reshape(batch, groups, block_size, self.num_heads, self.head_dim).transpose(2, 3)
        k = k[batch_ids, cells].transpose(2, 3)
        v = v[batch_ids, cells].transpose(2, 3)
        scores = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5
        weights = scores.masked_fill(~valid[:, :, None, None, :], float("-inf")).softmax(dim=-1)
        mixed = (weights @ v).transpose(2, 3).reshape_as(tokens)
        encoded = self.norm1(tokens + self.proj(mixed))
        return self.norm2(encoded + self.fc2(F.gelu(self.fc1(encoded))))

    def _refine_p3(self, p3, encoded_sites, cells, valid, batch_ids, height, width, padding):
        batch, channels = p3.shape[:2]
        groups, block_size = valid.shape[1], self.block_size
        stride = self.stride_ratio
        sites, children = height * width, stride ** 2
        heads, head_dim = self.fine_heads, self.fine_head_dim
        fine_dim = heads * head_dim

        patches = p3.reshape(batch, channels, height, stride, width, stride)
        patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(batch, sites, children, channels)
        fine = self.fine_in(patches)
        query, key = self.fine_qk(fine + self.fine_phase.to(fine.dtype)).chunk(2, dim=-1)
        value = self.fine_value(fine)
        context = self.p5_context(encoded_sites)
        query = query + context.unsqueeze(2)

        kv = torch.cat((key, value), dim=-1)
        anchors = kv.reshape(batch, sites, 2, 2, 2, 2, 2 * fine_dim).mean(dim=(3, 5))
        anchors = anchors.reshape(batch, sites, 4, 2 * fine_dim)
        anchors = F.pad(anchors, (0, 0, 0, 0, 0, padding))[batch_ids, cells]
        anchors = anchors.reshape(batch, groups, -1, 2, heads, head_dim).permute(3, 0, 1, 4, 2, 5)
        remote_key, remote_value = anchors.unbind(0)

        qkv = F.pad(torch.cat((query, kv), dim=-1), (0, 0, 0, 0, 0, padding))
        qkv = qkv.reshape(batch, groups, block_size, children, 3, heads, head_dim)
        q, k, v = qkv.permute(4, 0, 1, 5, 2, 3, 6).unbind(0)

        local_scores = q @ k.transpose(-2, -1)
        remote_scores = q.flatten(3, 4) @ remote_key.transpose(-2, -1)
        remote_scores = remote_scores.reshape(batch, groups, heads, block_size, children, -1)
        query_cells = torch.arange(groups * block_size, device=p3.device).reshape(1, groups, block_size, 1)
        remote_valid = valid.unsqueeze(2) & (cells.unsqueeze(2) != query_cells)
        remote_valid = remote_valid.repeat_interleave(4, dim=-1)
        remote_scores = remote_scores.masked_fill(~remote_valid[:, :, None, :, None, :], float("-inf"))

        scores = torch.cat((local_scores, remote_scores), dim=-1) * head_dim ** -0.5
        weights = scores.softmax(dim=-1)
        local_weights, remote_weights = weights.split((children, remote_value.shape[-2]), dim=-1)
        remote = remote_weights.flatten(3, 4) @ remote_value
        mixed = (local_weights @ v) + remote.reshape_as(q)
        mixed = mixed.permute(0, 1, 3, 4, 2, 5).reshape(batch, -1, children, fine_dim)[:, :sites]

        # Spatially adaptive foreground gate conditioned on local P3 detail and P5 semantics.
        gate_in = torch.cat((fine, context.unsqueeze(2).expand(-1, -1, children, -1)), dim=-1)
        token_gate = torch.sigmoid(self.p3_gate(gate_in)).to(mixed.dtype)

        # Preserve zero mean inside each P5 cell while allowing foreground-selective amplification.
        delta = self.fine_out(mixed - mixed.mean(dim=2, keepdim=True))
        amp = (0.4 * self.fine_gain.tanh()).to(delta.dtype) * token_gate.to(delta.dtype)
        refined = patches + amp * delta.to(patches.dtype)
        refined = refined.reshape(batch, height, width, stride, stride, channels)
        refined_p3 = refined.permute(0, 5, 1, 3, 2, 4).reshape_as(p3).contiguous()

        # Saliency-weighted P3 summary for zero-initialized P5 feedback.
        fg_summary = (mixed * token_gate).sum(dim=2) / token_gate.sum(dim=2).clamp_min(1e-4)
        p5_feedback = (0.2 * self.p5_fb_gain.tanh()).to(encoded_sites.dtype) * self.p3_to_p5(fg_summary).to(encoded_sites.dtype)
        return refined_p3, p5_feedback

    def forward(self, features):
        p3, p5 = features
        batch, channels, height, width = p5.shape
        stride = self.stride_ratio
        if p3.shape[-2:] != (stride * height, stride * width):
            raise ValueError("AIFI_P5RoutedP3V2 expects P3 at four times the P5 resolution")

        sites = height * width
        padding = -sites % self.block_size
        pos = AIFI_BlockSparse.build_2d_sincos_position_embedding(width, height, channels).to(p5)
        tokens = p5.flatten(2).transpose(1, 2) + pos
        tokens = F.pad(tokens, (0, 0, 0, padding)).reshape(batch, -1, self.block_size, channels)
        cells, valid = self._route(tokens, sites)
        batch_ids = torch.arange(batch, device=p5.device).reshape(batch, 1, 1)
        encoded = self._encode_p5(tokens, cells, valid, batch_ids)
        encoded_sites = encoded.reshape(batch, -1, channels)[:, :sites]
        refined, p5_feedback = self._refine_p3(p3, encoded_sites, cells, valid, batch_ids, height, width, padding)
        p5_out = (encoded_sites + p5_feedback).transpose(1, 2).reshape(batch, channels, height, width).contiguous()
        return p5_out, refined

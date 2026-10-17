# CSB-AIFI: Cross-Scale Block-routed Sparse AIFI (AIFI_BlockSparseV18).
#
# Motivation (rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7): AIFI is the only global
# self-attention in the network but it only ever sees P5 (20x20). Small-object
# tokens live at P3/P4, so P5 semantics are broadcast to them only AFTER the
# fact, through CCFFP2V7's channel concatenation. V18 lets P4 tokens
# participate directly in the AIFI semantic exchange.
#
# Design:
#   Branch 1 (P5 self): exactly BlockSparseV1 dense-masked block-sparse
#      attention (400 tokens at P5). Untouched - the proven baseline.
#   Branch 2 (P4 -> P5 cross): P5 queries attend to P4 keys/values through
#      V1's dense-masked attention path (NO gather): the full (n5 x npad4)
#      logit matrix is computed and non-selected entries are masked with
#      -1e4 before softmax. Block routing selects, for each P5 query block,
#      its top-k related P4 blocks (mean-pooled descriptors, zero routing
#      parameters, identical philosophy to V1).
#   Weight sharing: the cross branch reuses V1's qkv/proj weights (V7's
#      compressed-branch trick) -> zero new attention parameters. Only a
#      Linear(c_p4 -> c) input projection for P4 is added.
#   Gating: ONE zero-initialized scalar gate on the cross branch (V16
#      lesson: never double-gate). out_cross is nonzero at init, so the
#      gate - and through it the whole cross branch - receives gradient
#      immediately. Training starts exactly at BlockSparseV1 behavior.
#   Positional embedding: 2D sincos per grid (P5 20x20, P4 40x40), same
#      embed dim. K uses pos-embedded P4 tokens; V uses pos-free P4 tokens
#      (AgentAttentionV2 lesson: V must stay translation-equivariant).
#
# References:
#   MuViT: Multi-Resolution Vision Transformers (CVPR 2026) - cross-scale
#      query/KV attention with a shared token space.
#   SeerAttention (ICLR 2025) / NSA (2025) - learned block routing.
#   Block-Sparse Global Attention (CVPR 2026) - the V1 baseline.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aifi_block_sparse import BlockSparseAttention

__all__ = ['AIFI_BlockSparseV18']


class AIFI_BlockSparseV18(nn.Module):
    """Cross-scale block-routed sparse AIFI: [P4, P5] -> P5-shaped output.

    Args:
        in_channels: Channel list [c_p4, c_p5] from the yaml ``from`` field.
            A single int (or 1-element list) degrades gracefully to pure V1.
        embed_dim: Attention dimension (= c_p5 after input projection).
        cm: Hidden dimension of the FFN.
        num_heads: Heads of the block-sparse attention.
        block_size: Tokens per routing block (both scales).
        topk: P4 key blocks attended by each P5 query block.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, in_channels, embed_dim=256, cm=1024, num_heads=8,
                 block_size=4, topk=4, dropout=0.0, act=nn.GELU(),
                 normalize_before=False):
        super().__init__()
        if isinstance(in_channels, int):
            in_channels = [in_channels]
        if len(in_channels) == 0:
            raise ValueError('in_channels must not be empty')
        self.c_p5 = int(in_channels[-1])
        self.embed_dim = int(embed_dim)
        self.c_p4 = int(in_channels[0]) if len(in_channels) > 1 else None
        self.cross_topk = int(topk)
        self.normalize_before = normalize_before

        # P4 input projection (only when a cross-scale guide is provided).
        if self.c_p4 is not None and self.c_p4 != self.embed_dim:
            self.p4_proj = nn.Linear(self.c_p4, self.embed_dim)
        else:
            self.p4_proj = nn.Identity()

        # Branch 1: exactly V1's block-sparse attention (self branch).
        self.attn = BlockSparseAttention(
            dim=self.embed_dim, num_heads=num_heads, block_size=block_size,
            topk=topk, dropout=dropout)

        # Single zero-init scalar gate on the cross branch (V16 lesson:
        # one gate only - out_cross is nonzero at init so the gate trains).
        self.cross_gate = nn.Parameter(torch.zeros(1))

        self.fc1 = nn.Linear(self.embed_dim, cm)
        self.fc2 = nn.Linear(cm, self.embed_dim)
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
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

    def _pad_tokens(self, tokens, block_size):
        """Zero-pad (b, n, c) tokens so n is a multiple of block_size.

        Returns (padded, npad, valid) where valid is a (npad,) bool mask.
        """
        b, n, c = tokens.shape
        num_blocks = math.ceil(n / block_size)
        npad = num_blocks * block_size
        if npad == n:
            return tokens, npad, torch.ones(
                npad, dtype=torch.bool, device=tokens.device)
        out = F.pad(tokens, (0, 0, 0, npad - n))
        valid = torch.zeros(npad, dtype=torch.bool, device=tokens.device)
        valid[:n] = True
        return out, npad, valid

    def _cross_scale_attention(self, q_src, p4):
        """Dense-masked P5-query x P4-key/value attention with block routing.

        q_src: (b, n5, c) P5 token stream (position embedded, already
            normalized in pre-norm mode).
        p4: (b, c4, h4, w4) raw P4 feature map.
        Returns (b, n5, c) cross-scale attention output.
        """
        b, n5, c = q_src.shape
        bs = self.attn.block_size
        h, d = self.attn.num_heads, self.attn.head_dim
        scale = self.attn.scale

        # --- P4 token stream ---
        p4_tok = self.p4_proj(p4.flatten(2).permute(0, 2, 1))  # (b, n4, c)
        p4_tok, npad4, valid4 = self._pad_tokens(p4_tok, bs)
        pos4 = self.build_2d_sincos_position_embedding(
            p4.shape[3], p4.shape[2], c).to(device=p4.device, dtype=p4.dtype)
        if pos4.shape[1] != npad4:
            pos4 = F.pad(pos4, (0, 0, 0, npad4 - pos4.shape[1]))

        # K from pos-embedded P4, V from pos-free P4 (AgentV2 lesson).
        k4 = self.attn.qkv(p4_tok + pos4).chunk(3, dim=-1)[1].view(
            b, npad4, h, d).permute(0, 2, 1, 3)          # (b, h, npad4, d)
        v4 = self.attn.qkv(p4_tok).chunk(3, dim=-1)[2].view(
            b, npad4, h, d).permute(0, 2, 1, 3)          # (b, h, npad4, d)

        # --- P5 queries (reuses the SAME qkv weights, zero new params) ---
        q5 = self.attn.qkv(q_src).chunk(3, dim=-1)[0].view(
            b, n5, h, d).permute(0, 2, 1, 3)             # (b, h, n5, d)

        # --- Block routing: mean-pooled descriptors, zero parameters ---
        nblk5 = math.ceil(n5 / bs)
        nblk4 = npad4 // bs
        q_src_pad, _, _ = self._pad_tokens(q_src, bs)
        desc_q = q_src_pad.view(b, nblk5, bs, c).mean(dim=2)      # (b, nb5, c)
        desc_k = (p4_tok + pos4).view(b, nblk4, bs, c).mean(dim=2)  # (b, nb4, c)
        block_logits = torch.matmul(desc_q, desc_k.transpose(-2, -1)) * scale
        topk = min(self.cross_topk, nblk4)
        topk_idx = block_logits.topk(k=topk, dim=-1).indices       # (b, nb5, k)

        block_mask = torch.zeros(
            b, nblk5, nblk4, dtype=torch.bool, device=q_src.device)
        block_mask.scatter_(-1, topk_idx, True)

        # Expand to token mask (dense-masked path, same as V1).
        token_mask = block_mask.repeat_interleave(bs, dim=1)
        token_mask = token_mask.repeat_interleave(bs, dim=2)
        token_mask = token_mask[:, :n5, :npad4]
        token_mask = token_mask & valid4[None, None, :]

        # --- Dense masked attention (V1 path, no gather) ---
        logits = torch.matmul(q5, k4.transpose(-2, -1)) * scale  # (b,h,n5,npad4)
        logits = logits.masked_fill(~token_mask.unsqueeze(1), -1e4)
        attn = self.attn.attn_drop(logits.softmax(dim=-1))
        out = torch.matmul(attn, v4)                               # (b,h,n5,d)
        out = out.permute(0, 2, 1, 3).reshape(b, n5, c)
        return self.attn.proj_drop(self.attn.proj(out))

    def forward(self, inputs):
        """Multi-scale forward: inputs = [P4, P5]; P5 (last) is the query."""
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]
        p4 = inputs[0] if len(inputs) > 1 and self.c_p4 is not None else None
        p5 = inputs[-1]

        b, c, h, w = p5.shape
        if c != self.c_p5:
            raise ValueError(
                f'AIFI_BlockSparseV18 expected {self.c_p5} channels at P5, '
                f'got {c}')
        # (cross branch validated against c_p4 below)
        if p4 is not None and p4.shape[1] != self.c_p4:
            raise ValueError(
                f'AIFI_BlockSparseV18 expected {self.c_p4} channels at P4, '
                f'got {p4.shape[1]}')

        pos5 = self.build_2d_sincos_position_embedding(w, h, c).to(
            device=p5.device, dtype=p5.dtype)
        src = p5.flatten(2).permute(0, 2, 1) + pos5

        if self.normalize_before:
            src_n = self.norm1(src)
            out_self = self.attn(src_n)
            out_cross = (self._cross_scale_attention(src_n, p4)
                         if p4 is not None else 0.0)
            src = src + self.dropout1(
                out_self + self.cross_gate * out_cross)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(self.norm2(src))))))
        else:
            out_self = self.attn(src)
            out_cross = (self._cross_scale_attention(src, p4)
                         if p4 is not None else 0.0)
            src = src + self.dropout1(
                out_self + self.cross_gate * out_cross)
            src = self.norm1(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src)))))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()


    def extra_repr(self):
        return (f'c_p4={self.c_p4}, c_p5={self.c_p5}, '
                f'block_size={self.attn.block_size}, topk={self.attn.topk}, '
                f'cross_topk={self.cross_topk}')

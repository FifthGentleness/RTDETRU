# Block-sparse global attention V8 for RT-DETR AIFI.
#
# V8 = V1 + routing self-distillation (SeerAttention-inspired, training-only).
#
# Problem with V1: routing logits come from mean-pooled raw x - the router
#   "blindly guesses" which blocks matter. Meanwhile, the TRUE importance of
#   each block is available for free inside the forward pass: the dense
#   attention map. At P5 the token count is only 400, so computing the dense
#   attention for supervision costs almost nothing.
#
# V8 design (structure and inference behavior identical to V1):
#   1. Selected branch: V1 top-k block sparse attention, unchanged.
#   2. During TRAINING only, compute the dense attention softmax (detached)
#      and average it within/across blocks -> ground-truth block scores.
#   3. Route logits are pulled toward the ground-truth scores with a KL
#      loss (target = softmax of dense block scores). The aux loss is
#      stored on the module and summed into the total loss by
#      RTDETRDetectionModel.loss (see tasks.py patch).
#   4. Inference: dense attention is never computed; V8 == V1 exactly.
#
# Zero new parameters, zero inference cost, single variable vs V1.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AIFI_BlockSparseV8']


class BlockSparseAttentionV8(nn.Module):
    """V1 block-sparse attention + training-time routing self-distillation."""

    def __init__(self, dim, num_heads=8, block_size=4, topk=4, dropout=0.0,
                 distill_tau=1.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim ({dim}) must be divisible by num_heads ({num_heads})')
        if block_size < 1:
            raise ValueError('block_size must be >= 1')
        if topk < 1:
            raise ValueError('topk must be >= 1')

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.block_size = int(block_size)
        self.topk = int(topk)
        self.distill_tau = float(distill_tau)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Aux loss populated during training forward; collected by tasks.py.
        self.route_distill_loss = None

    def __deepcopy__(self, memo):
        """Deepcopy that drops the stale routing aux-loss tensor.

        Model construction runs a dummy forward (stride computation) which
        stores a graph-carrying tensor in self.route_distill_loss. Non-leaf
        tensors cannot be deep-copied (torch raises "Only Tensors created
        explicitly by the user (graph leaves) support the deepcopy protocol"),
        which breaks ModelEMA's deepcopy at training start. The aux loss is
        transient per-iteration state, so the copy simply starts with None.
        """
        import copy as _copy

        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k == 'route_distill_loss':
                new.route_distill_loss = None
                continue
            new.__dict__[k] = _copy.deepcopy(v, memo)
        return new

    def forward(self, x):
        b, n, c = x.shape
        h, d = self.num_heads, self.head_dim
        bs = self.block_size
        num_blocks = math.ceil(n / bs)
        pad_len = num_blocks * bs - n

        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))
        npad = num_blocks * bs

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, npad, h, d).permute(0, 2, 1, 3) for t in qkv]

        # --- V1 routing (unchanged) ---
        q_blk = x.view(b, num_blocks, bs, c).mean(dim=2)
        k_blk = x.view(b, num_blocks, bs, c).mean(dim=2)
        block_logits = torch.matmul(q_blk, k_blk.transpose(-2, -1)) * self.scale

        topk = min(self.topk, num_blocks)
        topk_idx = block_logits.topk(k=topk, dim=-1).indices

        block_mask = torch.zeros(
            b, num_blocks, num_blocks, dtype=torch.bool, device=x.device)
        block_mask.scatter_(-1, topk_idx, True)
        eye = torch.eye(num_blocks, dtype=torch.bool, device=x.device)
        block_mask = block_mask | eye.unsqueeze(0)

        token_mask = block_mask.repeat_interleave(bs, dim=1)
        token_mask = token_mask.repeat_interleave(bs, dim=2)
        token_mask = token_mask[:, :npad, :npad]
        if pad_len:
            valid = torch.zeros(b, npad, dtype=torch.bool, device=x.device)
            valid[:, :n] = True
            token_mask = token_mask & valid[:, None, :]

        # Single dense-logits computation shared by both branches.
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (b,h,npad,npad)

        # --- Routing self-distillation (training only) ---
        if self.training:
            with torch.no_grad():
                # Dense attention as ground truth. Padded keys get -inf so
                # they carry no mass.
                if pad_len:
                    key_valid = torch.zeros(b, npad, dtype=torch.bool,
                                            device=x.device)
                    key_valid[:, :n] = True
                    logits_gt = logits.masked_fill(
                        ~key_valid[:, None, None, :], float('-inf'))
                else:
                    logits_gt = logits
                attn_full = logits_gt.softmax(dim=-1)     # (b,h,npad,npad)
                attn_full = attn_full.view(
                    b, h, num_blocks, bs, num_blocks, bs).mean(dim=(3, 5))
                gt_scores = attn_full.mean(dim=1)         # (b, nb, nb)
                target_p = (gt_scores / self.distill_tau).softmax(dim=-1)

            log_pred = F.log_softmax(block_logits, dim=-1)
            kl = (target_p * (torch.log(target_p.clamp_min(1e-8)) - log_pred))
            self.route_distill_loss = kl.sum(dim=-1).mean()

        # --- Selected branch: identical to V1 ---
        logits_m = logits.masked_fill(~token_mask.unsqueeze(1), -1e4)
        attn = self.attn_drop(logits_m.softmax(dim=-1))
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, npad, c)
        if pad_len:
            out = out[:, :n, :]
        return self.proj_drop(self.proj(out))


class AIFI_BlockSparseV8(nn.Module):
    """AIFI with V8 block-sparse attention: V1 + routing self-distillation.

    Structure and inference behavior are identical to V1. The only
    difference is an auxiliary KL loss computed during training that pulls
    routing logits toward the true dense-attention block scores.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads.
        block_size: Number of tokens per sparse block.
        topk: Number of key blocks attended by each query block.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
        distill_tau: Temperature for the dense-attention soft targets.
    """

    def __init__(self, c1, cm=2048, num_heads=8, block_size=4, topk=4,
                 dropout=0.0, act=nn.GELU(), normalize_before=False,
                 distill_tau=1.0):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = BlockSparseAttentionV8(
            dim=c1,
            num_heads=num_heads,
            block_size=block_size,
            topk=topk,
            dropout=dropout,
            distill_tau=distill_tau,
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
                f'AIFI_BlockSparseV8 expected {self.c1} channels, got {c}')

        pos = self.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1) + pos

        if self.normalize_before:
            src_norm = self.norm1(src)
            src = src + self.dropout1(self.attn(src_norm))
            src_norm = self.norm2(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src_norm)))))
        else:
            src = src + self.dropout1(self.attn(src))
            src = self.norm1(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src)))))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()


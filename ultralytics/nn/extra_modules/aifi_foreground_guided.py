# FG-AIFI: GT-guided foreground-aware AIFI (AIFI_ForegroundGuided).
#
# Problem addressed: P5 self-attention is background-dominated and small-object
# tokens are starved. Every existing module in this repository learns its prior
# implicitly (DSA conv branches, FLatten focusing) or self-distills it from the
# dense attention (V8). None uses the EXTERNAL ground-truth signal.
#
# Mechanism (Focus-DETR, ICLR 2023 / Sparse DETR, ICLR 2022):
#   1. A tiny MLP foreground scorer scores every P5 token.
#   2. During training the scores are supervised with GT boxes projected onto
#      the P5 grid (a token is positive when its center falls inside any GT
#      box) through a BCE loss with pos_weight (collected by tasks.py via
#      compute_ground_loss, reusing the aux-loss pipeline).
#   3. The sigmoid scores act as an ADDITIVE attention-logit prior through a
#      single zero-initialized gate: logits += fg_gate * sigmoid(score).
#      Order-preserving and differentiable - no token is dropped.
#   4. At inference the scorer alone provides the learned task prior
#      (zero extra supervision, negligible cost).
#
# Zero-overlap statement: DSAWACGAv5 / P2SPDOKMFSV7 contain no task-supervised
# component; GT boxes otherwise only appear in the decoder matching loss.

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AIFI_ForegroundGuided']


class ForegroundGuidedAttention(nn.Module):
    """Full MHSA whose logits are biased by a GT-supervised foreground prior."""

    def __init__(self, dim, num_heads=8, dropout=0.0, hidden_dim=64,
                 fg_pos_weight=5.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f'dim ({dim}) must be divisible by num_heads ({num_heads})')
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Foreground scorer: 2-layer MLP, one logit per token.
        self.fg_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        # Single zero-init gate on the additive prior (no double gating).
        self.fg_gate = nn.Parameter(torch.zeros(1))
        self.fg_pos_weight = float(fg_pos_weight)

        # Transient training state consumed by compute_ground_loss (tasks.py).
        self._fg_logits = None
        self._fg_hw = None

    def __deepcopy__(self, memo):
        """Drop stale graph-carrying cache tensors (ModelEMA deepcopy)."""
        import copy as _copy

        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k in ('_fg_logits', '_fg_hw'):
                new._fg_logits = None
                new._fg_hw = None
                continue
            new.__dict__[k] = _copy.deepcopy(v, memo)
        return new

    def forward(self, x, hw):
        b, n, c = x.shape
        h, w = hw
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                   for t in qkv]
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Foreground prior (task-supervised at train time, learned prior at
        # inference): additive, order-preserving, differentiable.
        fg = self.fg_mlp(x)                       # (b, n, 1)
        if self.training and torch.is_grad_enabled():
            self._fg_logits = fg
            self._fg_hw = (h, w)
        prior = fg.sigmoid().transpose(1, 2).unsqueeze(1)  # (b, 1, n, 1)
        logits = logits + self.fg_gate * prior

        logits = logits.softmax(dim=-1)
        attn = self.attn_drop(logits)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, c)
        return self.proj_drop(self.proj(out))

    def compute_ground_loss(self, targets):
        """BCE supervision of the scorer with GT boxes projected to the grid.

        Called by RTDETRDetectionModel.loss after the training forward.
        """
        if not self.training or self._fg_logits is None:
            return None
        logits = self._fg_logits                  # (b, n, 1), graph tensor
        b, n = logits.shape[:2]
        h, w = self._fg_hw
        device = logits.device

        # Token centers in normalized coordinates ((i + 0.5) / size).
        xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
        ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        centers = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (n, 2)

        labels = torch.zeros(b, n, device=device)
        boxes = targets['bboxes']
        batch_idx = targets['batch_idx']
        for i in range(b):
            m = batch_idx == i
            if m.any():
                bx = boxes[m]                     # (m_i, 4) xywh normalized
                inside_x = (centers[:, 0].unsqueeze(0) - bx[:, 0:1]).abs() \
                    <= bx[:, 2:3] / 2
                inside_y = (centers[:, 1].unsqueeze(0) - bx[:, 1:2]).abs() \
                    <= bx[:, 3:4] / 2
                labels[i] = (inside_x & inside_y).any(dim=0).float()

        pos_weight = torch.tensor(self.fg_pos_weight, device=device)
        loss = F.binary_cross_entropy_with_logits(
            logits.squeeze(-1), labels, pos_weight=pos_weight)
        self._fg_logits = None                    # consume
        self._fg_hw = None
        return loss


class AIFI_ForegroundGuided(nn.Module):
    """AIFI with a GT-guided foreground-aware attention prior.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
        fg_hidden: Hidden width of the foreground scorer MLP.
        fg_pos_weight: BCE pos_weight (small objects are rare positives).
    """

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(),
                 normalize_before=False, fg_hidden=64, fg_pos_weight=5.0):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = ForegroundGuidedAttention(
            dim=c1, num_heads=num_heads, dropout=dropout,
            hidden_dim=fg_hidden, fg_pos_weight=fg_pos_weight)
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
                f'AIFI_ForegroundGuided expected {self.c1} channels, got {c}')

        pos = self.build_2d_sincos_position_embedding(w, h, c).to(
            device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1) + pos

        if self.normalize_before:
            src = src + self.dropout1(self.attn(self.norm1(src), (h, w)))
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(self.norm2(src))))))
        else:
            src = src + self.dropout1(self.attn(src, (h, w)))
            src = self.norm1(src)
            src = src + self.dropout2(
                self.fc2(self.dropout(self.act(self.fc1(src)))))
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()

    def compute_ground_loss(self, targets):
        return self.attn.compute_ground_loss(targets)

    def extra_repr(self):
        return (f'c1={self.c1}, num_heads={self.attn.num_heads}, '
                f'fg_pos_weight={self.attn.fg_pos_weight}')

# SD-AIFI: Foundation-model feature distillation for AIFI tokens
# (AIFI_FoundationDistill).
#
# Problem addressed: the whole network (including DSAWACGAv5, P2SPDOKMFSV7 and
# every AIFI variant) is trained from scratch on the (small) aerial dataset.
# Pretraining-level semantics and precise object boundaries never enter the
# encoder.
#
# Mechanism (foundation-model dense feature distillation, 2024-2025 trend;
# anchors: SAM ICCV 2023 / DINOv2 2023 + distillation-into-detector works
# 2024-2025):
#   1. OFFLINE (one-time): extract dense features of the training images with
#      a frozen foundation model (SAM ViT-B image encoder or DINOv2) and store
#      them as .npy files (shape (ref_dim, h_f, w_f), float16) named
#      "<image_stem>.npy" in one directory.
#   2. TRAINING-TIME: AIFI tokens are aligned to the (bilinearly resized)
#      foundation features through a learnable Linear(ref_dim -> c) and a
#      cosine loss. The loss is collected by tasks.py through
#      compute_ground_loss(targets, batch) using batch["im_file"].
#   3. INFERENCE: zero overhead - the alignment head can even be dropped.
#
# The forward graph is IDENTICAL to the native AIFI; the innovation lives in
# the training signal. Zero-overlap: no architectural mechanism from
# DSAWACGAv5 / P2SPDOKMFSV7 is involved.
#
# Example offline extraction (DINOv2, run once):
#   import torch, einops
#   from pathlib import Path
#   model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').cuda().eval()
#   for img_path in image_list:
#       x = preprocess_224_or_multiple_of_14(img_path).cuda()
#       feat = model.get_intermediate_layers(x)[0]            # (1, n, 768)
#       s = int(feat.shape[1] ** 0.5)
#       feat = einops.rearrange(feat, '1 (h w) c -> c h w', h=s, w=s)
#       np.save(out_dir / f'{Path(img_path).stem}.npy', feat.half().cpu().numpy())

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AIFI_FoundationDistill']


class FeatureStore:
    """Lazy LRU loader for offline foundation features (.npy per image stem)."""

    def __init__(self, root, cache_size=256):
        self.root = Path(root) if root else None
        self.cache = {}
        self.cache_size = int(cache_size)

    def get(self, im_file, ref_dim):
        if self.root is None:
            return None
        stem = Path(str(im_file)).stem
        if stem in self.cache:
            return self.cache[stem]
        f = self.root / f'{stem}.npy'
        if not f.exists():
            return None
        arr = np.load(f).astype(np.float32)
        if arr.ndim == 2:                      # (n, d) flat -> reshape square
            s = int(arr.shape[0] ** 0.5)
            arr = arr.reshape(s, s, -1).transpose(2, 0, 1)
        item = torch.from_numpy(arr)           # (ref_dim, hf, wf)
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[stem] = item
        return item


class FoundationDistillAttention(nn.Module):
    """Native full MHSA (identical graph to the original AIFI attention)."""

    def __init__(self, dim, num_heads=8, dropout=0.0):
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

        # Distillation alignment head (usable only when ref features exist).
        self.ref_dim = None
        self.ref_proj = None

        # Transient training state consumed by compute_ground_loss.
        self._tokens = None
        self._hw = None

    def __deepcopy__(self, memo):
        import copy as _copy

        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k in ('_tokens', '_hw'):
                new._tokens = None
                new._hw = None
                continue
            new.__dict__[k] = _copy.deepcopy(v, memo)
        return new

    def _ensure_head(self, ref_dim, device, dtype):
        if self.ref_proj is None or self.ref_dim != ref_dim:
            self.ref_dim = ref_dim
            self.ref_proj = nn.Linear(ref_dim, self.dim).to(
                device=device, dtype=dtype)

    def forward(self, x, hw=None):
        b, n, c = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                   for t in qkv]
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(logits.softmax(dim=-1))
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, c)
        if self.training and torch.is_grad_enabled():
            self._tokens = x
            self._hw = hw
        return self.proj_drop(self.proj(out))

    def compute_ground_loss(self, targets, batch=None, store=None):
        """Cosine alignment between AIFI tokens and foundation features."""
        if (not self.training or self._tokens is None or store is None
                or batch is None or 'im_file' not in batch):
            return None
        tokens = self._tokens                  # (b, n, c), graph tensor
        b, n, c = tokens.shape
        h, w = self._hw
        im_files = batch['im_file']
        losses, cnt = 0.0, 0
        for i, imf in enumerate(im_files):
            ref = store.get(imf, self.ref_dim or 0)
            if ref is None:
                continue
            ref = ref.to(device=tokens.device, dtype=tokens.dtype)
            ref = F.interpolate(ref.unsqueeze(0), size=(h, w), mode='bilinear',
                                align_corners=False)[0]        # (d, h, w)
            self._ensure_head(ref.shape[0], tokens.device, tokens.dtype)
            ref_tok = self.ref_proj(ref.flatten(1).t())        # (n, c)
            t_i = F.normalize(tokens[i], dim=-1)
            r_i = F.normalize(ref_tok, dim=-1)
            losses = losses + (1.0 - (t_i * r_i).sum(-1)).mean()
            cnt += 1
        self._tokens = None
        self._hw = None
        if cnt == 0:
            return None
        return losses / cnt


class AIFI_FoundationDistill(nn.Module):
    """Native AIFI + training-only foundation-model feature distillation.

    Args:
        c1: Input and output channels.
        cm: Hidden dimension of the FFN.
        num_heads: Number of attention heads.
        ref_root: Directory containing '<stem>.npy' foundation features.
            Falls back to the SD_FEATURE_ROOT environment variable.
        dropout: Dropout probability.
        act: Activation used by the FFN.
        normalize_before: Use pre-normalization if True.
    """

    def __init__(self, c1, cm=2048, num_heads=8, ref_root=None, dropout=0.0,
                 act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = FoundationDistillAttention(
            dim=c1, num_heads=num_heads, dropout=dropout)
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act
        root = ref_root or os.environ.get('SD_FEATURE_ROOT')
        self.store = FeatureStore(root) if root else None

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
                f'AIFI_FoundationDistill expected {self.c1} channels, got {c}')

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

    def compute_ground_loss(self, targets, batch=None):
        if self.store is None:
            return None
        return self.attn.compute_ground_loss(
            targets, batch=batch, store=self.store)

    def extra_repr(self):
        return (f'c1={self.c1}, num_heads={self.attn.num_heads}, '
                f'store={"on" if self.store is not None else "off"}')

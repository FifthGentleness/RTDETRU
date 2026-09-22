# MuViT-style multi-scale coordinate-aligned AIFI for RT-DETR.
#
# Reference:
#   MuViT: Multi-Resolution Vision Transformers for Learning Across Scales in Microscopy
#   Albert Dominguez Mantes, Gioele La Manno, Martin Weigert. CVPR 2026.
#
# This module adapts the core idea of MuViT to RT-DETR's AIFI:
#   1. Use multiple feature scales from the same image.
#   2. Embed all tokens in a shared, normalized 2D coordinate system.
#   3. Add a learnable scale embedding to distinguish P3/P4/P5.
#   4. Use P5 as the query scale and P3/P4/P5 as key/value tokens.
#
# The result is a P5-level AIFI that can still access high-resolution P3/P4
# information without moving the entire encoder to those resolutions.

import torch
import torch.nn as nn

__all__ = ['AIFI_MuViT']


class AIFI_MuViT(nn.Module):
    """Multi-scale coordinate-aligned AIFI.

    Expected input order is ``[P3, P4, P5]``. The last input is used as the
    query scale; all inputs are used as key/value scales. This keeps the output
    spatial layout identical to P5 while allowing P5 queries to attend to
    high-resolution P3/P4 tokens.

    Args:
        in_channels: Channel count for each input scale, e.g. ``(128, 256, 256)``.
        embed_dim: Common token dimension, also the output channel count.
        cm: Hidden dimension of the AIFI feed-forward network.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        act: Activation module for the FFN.
        normalize_before: Use pre-normalization if True. AIFI default is False.
        temperature: Temperature for the 2D sine-cosine coordinate embedding.
    """

    def __init__(self, in_channels, embed_dim=256, cm=1024, num_heads=8,
                 dropout=0.0, act=nn.GELU(), normalize_before=False,
                 temperature=10000.0):
        super().__init__()
        if isinstance(in_channels, int):
            in_channels = (in_channels,)
        if not in_channels:
            raise ValueError('AIFI_MuViT requires at least one input scale')
        if embed_dim % num_heads != 0:
            raise ValueError(
                f'embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})'
            )

        self.in_channels = tuple(in_channels)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.normalize_before = normalize_before
        self.temperature = temperature
        self.num_scales = len(self.in_channels)

        # Project each input scale to the common token dimension. If a scale
        # already has the target dimension, use identity to avoid a redundant
        # projection; this is useful when P5 is already projected by RT-DETR.
        self.input_projections = nn.ModuleList([
            nn.Identity() if c == embed_dim else nn.Conv2d(c, embed_dim, kernel_size=1)
            for c in self.in_channels
        ])

        # One learnable embedding per scale. This distinguishes tokens from
        # P3/P4/P5 after they are placed in the shared coordinate system.
        self.scale_embeddings = nn.Parameter(
            torch.zeros(self.num_scales, 1, 1, embed_dim)
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Keep the original AIFI FFN and normalization structure.
        self.fc1 = nn.Linear(embed_dim, cm)
        self.fc2 = nn.Linear(cm, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act

    def _coordinate_embedding(self, h, w, device, dtype):
        """Build a shared-coordinate 2D sine-cosine embedding.

        Coordinates are normalized to ``[0, 1]`` so that tokens from different
        feature-map resolutions represent the same image coordinate system.
        """
        if self.embed_dim % 4 != 0:
            raise ValueError(
                'embed_dim must be divisible by 4 for 2D sine-cosine embeddings'
            )

        ys = torch.arange(h, device=device, dtype=torch.float32)
        xs = torch.arange(w, device=device, dtype=torch.float32)
        ys = ys / max(h - 1, 1)
        xs = xs / max(w - 1, 1)

        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        grid_y = grid_y.reshape(-1)
        grid_x = grid_x.reshape(-1)

        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, device=device, dtype=torch.float32) / pos_dim
        omega = 1.0 / (self.temperature ** omega)

        out_y = grid_y[:, None] * omega[None, :]
        out_x = grid_x[:, None] * omega[None, :]

        embedding = torch.cat(
            [torch.sin(out_y), torch.cos(out_y),
             torch.sin(out_x), torch.cos(out_x)],
            dim=1,
        )
        return embedding.to(dtype=dtype)[None]  # (1, h*w, embed_dim)

    def _prepare_tokens(self, x, scale_index):
        """Project one feature map and add coordinate and scale embeddings."""
        if not torch.is_tensor(x):
            raise TypeError(f'AIFI_MuViT input {scale_index} must be a Tensor')
        if x.ndim != 4:
            raise ValueError(
                f'AIFI_MuViT input {scale_index} must be 4D, got {tuple(x.shape)}'
            )

        b, c, h, w = x.shape
        expected_c = self.in_channels[scale_index]
        if c != expected_c:
            raise ValueError(
                f'AIFI_MuViT input {scale_index} expected {expected_c} channels, got {c}'
            )

        projected = self.input_projections[scale_index](x)
        tokens = projected.flatten(2).permute(0, 2, 1)  # (B, h*w, C)
        pos = self._coordinate_embedding(h, w, tokens.device, tokens.dtype)
        scale = self.scale_embeddings[scale_index].to(dtype=tokens.dtype)
        return tokens + pos + scale

    def forward(self, inputs):
        """Forward multi-scale inputs.

        Args:
            inputs: Sequence of feature maps ordered as ``[P3, P4, P5]``. The
                last feature map is used as the query scale.

        Returns:
            Tensor with the same spatial shape and channel count as the last
            input after projection to ``embed_dim``.
        """
        if len(inputs) != self.num_scales:
            raise ValueError(
                f'AIFI_MuViT expected {self.num_scales} inputs, got {len(inputs)}'
            )

        prepared = [self._prepare_tokens(x, i) for i, x in enumerate(inputs)]
        query = prepared[-1]
        key_value = torch.cat(prepared, dim=1)

        if self.normalize_before:
            query_norm = self.norm1(query)
            attn_out = self.attn(
                query_norm, key_value, key_value,
                need_weights=False,
            )[0]
            query = query + self.dropout1(attn_out)

            query_norm = self.norm2(query)
            ff_out = self.fc2(self.dropout(self.act(self.fc1(query_norm))))
            query = query + self.dropout2(ff_out)
        else:
            attn_out = self.attn(
                query, key_value, key_value,
                need_weights=False,
            )[0]
            query = query + self.dropout1(attn_out)
            query = self.norm1(query)

            ff_out = self.fc2(self.dropout(self.act(self.fc1(query))))
            query = query + self.dropout2(ff_out)
            query = self.norm2(query)

        # Restore the P5 feature-map layout.
        p5 = inputs[-1]
        b, _, h, w = p5.shape
        return query.permute(0, 2, 1).reshape(b, self.embed_dim, h, w).contiguous()

"""Dense P5 attention conditioned on directional structure in P4.

StructSA (CVPR 2024) motivates scoring structural correlations. QUEST (ICLR
2026) motivates bounding key norms while retaining query-dependent strength.
Here both ends of each P5 attention edge use aligned P4 directional contrasts.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.transformer import AIFI

__all__ = ('AIFI_DirectionalRelation',)


class AIFI_DirectionalRelation(AIFI):
    """Add a bounded, content-dependent relation score to every dense P5 logit."""

    def __init__(self, channels, cm=1024, num_heads=8, relation_width=4,
                 max_logit_bias=2.0, initial_logit_bias=0.25):
        context_channels, semantic_channels = channels
        super().__init__(semantic_channels, cm, num_heads)
        if relation_width < 1:
            raise ValueError('relation_width must be positive')
        if not (math.isfinite(max_logit_bias) and math.isfinite(initial_logit_bias)
                and max_logit_bias >= 0
                and (max_logit_bias == 0 or 0 < initial_logit_bias < max_logit_bias)):
            raise ValueError('Use max_logit_bias=0 or 0 < initial_logit_bias < max_logit_bias')
        self.relation_width = relation_width
        self.max_logit_bias = float(max_logit_bias)

        relation_channels = num_heads * relation_width
        self.context_proj = nn.Conv2d(context_channels, relation_channels, 1, bias=False)
        self.context_norm = nn.GroupNorm(num_heads, relation_channels, affine=False)
        self.relation_query = nn.Linear(relation_width, relation_width, bias=False)
        self.relation_key = nn.Linear(relation_width, relation_width, bias=False)
        self.query_gate = nn.Linear(semantic_channels, num_heads)

        nn.init.eye_(self.relation_query.weight)
        nn.init.eye_(self.relation_key.weight)
        nn.init.zeros_(self.query_gate.weight)
        if self.max_logit_bias:
            ratio = initial_logit_bias / self.max_logit_bias
            nn.init.constant_(self.query_gate.bias, math.log(ratio / (1.0 - ratio)))

    def _directional_relations(self, context):
        """Compute signed one-step P4 contrasts before pooling to the P5 grid."""
        context = self.context_norm(self.context_proj(context))
        east = context[..., 1:] - context[..., :-1]
        south = context[..., 1:, :] - context[..., :-1, :]
        # The boundary has zero contrast in a direction with no neighbor.
        contrasts = torch.stack((F.pad(east, (0, 1)), F.pad(-east, (1, 0)),
                                 F.pad(south, (0, 0, 0, 1)), F.pad(-south, (0, 0, 1, 0))), dim=1)
        pooled = F.avg_pool2d(contrasts.flatten(1, 2).tanh(), kernel_size=2, stride=2)
        # [batch, heads, sites, east/west/south/north, relation channels]
        return pooled.reshape(context.shape[0], 4, self.ma.num_heads, self.relation_width, -1).permute(0, 2, 4, 1, 3)

    def _attend(self, tokens, pos, context):
        batch, sites, channels = tokens.shape
        heads, head_dim = self.ma.num_heads, self.ma.head_dim
        qk_weights, v_weights = self.ma.in_proj_weight.split((2 * channels, channels))
        qk_bias, v_bias = self.ma.in_proj_bias.split((2 * channels, channels))
        qk = F.linear(tokens + pos, qk_weights, qk_bias)
        query, key = qk.reshape(batch, sites, 2, heads, head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        value = F.linear(tokens, v_weights, v_bias).reshape(batch, sites, heads, head_dim).transpose(1, 2)

        relations = self._directional_relations(context)
        rel_query = self.relation_query(relations).flatten(-2).float()
        rel_key = self.relation_key(relations).flatten(-2).float()
        # Small keys retain weak evidence. Query norm keeps local contrast strength.
        rel_query = rel_query * torch.rsqrt(1.0 + rel_query.square().sum(-1, keepdim=True))
        rel_key = F.normalize(rel_key, dim=-1, eps=0.25)
        strength = self.max_logit_bias * self.query_gate(tokens).sigmoid().transpose(1, 2).unsqueeze(-1)

        # One SDPA softmax combines the native semantic score and signed relation score.
        # Compensate for SDPA scaling from the enlarged head and pad V for equal head widths.
        relation_dim = rel_key.shape[-1]
        combined_dim = head_dim + relation_dim
        query = torch.cat((query * math.sqrt(combined_dim / head_dim),
                           (rel_query * strength * math.sqrt(combined_dim)).to(query.dtype)), dim=-1)
        key = torch.cat((key, rel_key.to(key.dtype)), dim=-1)
        value = F.pad(value, (0, relation_dim))
        dropout = self.ma.dropout if self.training else 0.0
        encoded = F.scaled_dot_product_attention(query, key, value, dropout_p=dropout)[..., :head_dim]
        return self.ma.out_proj(encoded.transpose(1, 2).reshape(batch, sites, channels))

    def forward(self, features):
        context, semantic = features
        if self.max_logit_bias == 0:
            return super().forward(semantic)
        batch, channels, height, width = semantic.shape
        if context.shape[-2:] != (height * 2, width * 2):
            raise ValueError('Directional relation attention expects P4 at twice the P5 spatial resolution')

        pos = self.build_2d_sincos_position_embedding(width, height, channels).to(semantic)
        tokens = semantic.flatten(2).transpose(1, 2)
        tokens = self.norm1(tokens + self.dropout1(self._attend(tokens, pos, context)))
        ffn = self.fc2(self.dropout(self.act(self.fc1(tokens))))
        tokens = self.norm2(tokens + self.dropout2(ffn))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width).contiguous()

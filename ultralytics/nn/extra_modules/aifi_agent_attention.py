# Agent Attention version of RT-DETR AIFI.
#
# Reference:
#   Agent Attention: On the Integration of Softmax and Linear Attention
#   Dongchen Han et al., ECCV 2024. arXiv:2312.08874
#
# This implementation keeps the original RT-DETR AIFI interface and its
# 2D sine-cosine positional embedding, and replaces only the full softmax
# self-attention with a two-stage agent attention:
#   Q -> A -> K/V
# where A is a small learnable set of agent tokens.

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AgentAttention', 'AIFI_AgentAttention']


class AgentAttention(nn.Module):
    """Two-stage softmax attention with a small set of learnable agent tokens.

    For N query/key tokens and M agent tokens, the attention cost is
    O(N*M + M*N) instead of O(N*N) full attention. With M << N, this gives
    an efficient global-context operation while retaining softmax attention.
    """

    def __init__(self, embed_dim, num_heads=8, num_agents=49, dropout=0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f'embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})')
        if num_agents < 1:
            raise ValueError('num_agents must be greater than 0')

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_agents = num_agents
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.agent = nn.Parameter(torch.empty(1, num_agents, embed_dim))
        # Per-head learnable bias for the Q -> agent attention.
        self.agent_bias = nn.Parameter(torch.zeros(1, num_heads, 1, num_agents))
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        nn.init.normal_(self.agent, mean=0.0, std=0.02)
        nn.init.zeros_(self.agent_bias)
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        """Apply agent attention to tensor x of shape (B, N, C)."""
        b, n, c = x.shape
        h, d = self.num_heads, self.head_dim
        m = self.num_agents

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, H, N, D)
        q = q.view(b, n, h, d).permute(0, 2, 1, 3)
        k = k.view(b, n, h, d).permute(0, 2, 1, 3)
        v = v.view(b, n, h, d).permute(0, 2, 1, 3)

        # Learnable agent tokens: (B, H, M, D)
        agent = self.agent.expand(b, -1, -1).view(b, m, h, d).permute(0, 2, 1, 3)

        # Stage 1: agents aggregate information from all K/V tokens.
        agent_to_kv = torch.matmul(agent, k.transpose(-2, -1)) * self.scale
        agent_to_kv = agent_to_kv.softmax(dim=-1)
        agent_to_kv = self.attn_dropout(agent_to_kv)
        agent_value = torch.matmul(agent_to_kv, v)  # (B, H, M, D)

        # Stage 2: queries retrieve global information from agent tokens.
        q_to_agent = torch.matmul(q, agent.transpose(-2, -1)) * self.scale
        q_to_agent = q_to_agent + self.agent_bias
        q_to_agent = q_to_agent.softmax(dim=-1)
        q_to_agent = self.attn_dropout(q_to_agent)
        out = torch.matmul(q_to_agent, agent_value)  # (B, H, N, D)

        out = out.permute(0, 2, 1, 3).reshape(b, n, c)
        return self.out_proj(out)


class AIFI_AgentAttention(nn.Module):
    """RT-DETR AIFI with Agent Attention replacing full self-attention.

    The original AIFI feed-forward network, normalization order, residual
    connections, and 2D sine-cosine positional embedding are retained. Only
    nn.MultiheadAttention is replaced by AgentAttention.

    Args:
        c1: Input/output channels.
        cm: FFN hidden dimension.
        num_heads: Number of attention heads.
        num_agents: Number of learnable agent tokens (default: 49).
        dropout: Dropout probability.
        act: Activation module used in the FFN.
        normalize_before: Use pre-normalization if True; AIFI default is False.
    """

    def __init__(self, c1, cm=2048, num_heads=8, num_agents=49,
                 dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before

        self.attn = AgentAttention(
            embed_dim=c1,
            num_heads=num_heads,
            num_agents=num_agents,
            dropout=dropout,
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
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """Build the same 2D sine-cosine embedding used by RT-DETR AIFI."""
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.cat([torch.sin(out_w), torch.cos(out_w),
                          torch.sin(out_h), torch.cos(out_h)], dim=1)[None]

    def forward(self, x):
        """Forward a feature map of shape (B, C, H, W)."""
        b, c, h, w = x.shape
        if c != self.c1:
            raise ValueError(f'AIFI_AgentAttention expected {self.c1} channels, got {c}')

        pos_embed = self.build_2d_sincos_position_embedding(
            w, h, c).to(device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

        if self.normalize_before:
            src_norm = self.norm1(src)
            src2 = self.attn(src_norm + pos_embed)
            src = src + self.dropout1(src2)
            src_norm = self.norm2(src)
            src2 = self.fc2(self.dropout(self.act(self.fc1(src_norm))))
            src = src + self.dropout2(src2)
        else:
            src2 = self.attn(src + pos_embed)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()

# Agent Attention V2 for RT-DETR AIFI — 修复版
#
# 基于 ECCV 2024 "Agent Attention: On the Integration of Softmax and Linear Attention"
# 相比 V1 的三处关键修复:
#
# 修复 1 (必须): 分离 Q/K 和 V 的位置编码
#   V1: self.attn(src + pos)  → Q, K, V 全部含 pos
#   V2: self.attn(qk_input=src+pos, v_input=src)  → Q, K 含 pos, V 不含 pos
#   原因: DETR 系列设计约定 — pos 控制"从哪里看"(注意力权重),
#         V 提供"看什么"(聚合内容). V 含 pos 会破坏平移等变性,
#         对检测任务的 bbox 回归引入位置相关系统性偏置.
#
# 修复 2 (推荐): 增加 agent 数量
#   V1: num_agents=49, P5=20x20=400 tokens, 压缩比 8.2:1
#   V2: num_agents=144(12x12), 压缩比 2.8:1
#   原因: 信息瓶颈从秩49提升到秩144, 小目标信号在agent中的保留率显著提高.
#   原论文在ViT(196 tokens)上用49 agents(4:1), P5有400 tokens需要更多agents.
#
# 修复 3 (可选): 添加 DWConv 局部注意力补充
#   V1: 只有全局 agent 注意力 (低秩近似, 秩 <= num_agents)
#   V2: 全局 agent 注意力 + DWConv(V) 局部补充
#   原因: 这是原论文的核心组件(V1遗漏). Agent注意力是全局低秩近似,
#         丢失局部细节; DWConv从V中恢复每个token的3x3邻域信息.
#         对小目标: 即使agent瓶颈丢失信号, DWConv仍能从V恢复局部细节.
#
# 架构:
#   X (B, C, H, W)
#   ├── Q/K = W_QK(X + pos),  V = W_V(X)  ← 修复1: V不含pos
#   ├── Agent Attention (全局低秩近似)
#   │     Stage 1: agents → K/V  (M agents aggregate from N tokens)
#   │     Stage 2: Q → agents    (N queries retrieve from M agents)
#   ├── + DWConv(V_2d)           ← 修复3: 局部补充
#   └── out_proj → 残差 + LayerNorm + FFN

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AgentAttentionV2', 'AIFI_AgentAttentionV2']


class AgentAttentionV2(nn.Module):
    """Two-stage agent attention with separated Q/K and V inputs + DWConv local supplement.

    Key differences from V1 (AgentAttention):
      1. forward() takes qk_input and v_input separately (pos only on Q/K)
      2. Q/K projected from qk_input, V projected from v_input
      3. DWConv on V provides local detail supplement (from original paper)

    Args:
        embed_dim: Token dimension.
        num_heads: Number of attention heads.
        num_agents: Number of learnable agent tokens.
        dropout: Dropout probability.
        local_kernel_size: DWConv kernel size for local supplement (0 to disable).
    """

    def __init__(self, embed_dim, num_heads=8, num_agents=144, dropout=0.0,
                 local_kernel_size=3):
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
        self.local_kernel_size = local_kernel_size

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.agent = nn.Parameter(torch.empty(1, num_agents, embed_dim))
        self.agent_bias = nn.Parameter(torch.zeros(1, num_heads, 1, num_agents))

        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        if local_kernel_size > 0:
            padding = local_kernel_size // 2
            self.local_dwconv = nn.Conv2d(
                embed_dim, embed_dim,
                kernel_size=local_kernel_size,
                padding=padding,
                groups=embed_dim,
                bias=True,
            )
        else:
            self.local_dwconv = None

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.agent, mean=0.0, std=0.02)
        nn.init.zeros_(self.agent_bias)
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        if self.local_dwconv is not None:
            nn.init.kaiming_uniform_(self.local_dwconv.weight, a=math.sqrt(5))
            if self.local_dwconv.bias is not None:
                fan_in = self.local_dwconv.weight.shape[0]
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.local_dwconv.bias, -bound, bound)

    def forward(self, qk_input, v_input, v_2d=None):
        """Apply agent attention with separated Q/K and V.

        Args:
            qk_input: Input for Q/K projection (features + pos), shape (B, N, C).
            v_input: Input for V projection (features only, no pos), shape (B, N, C).
            v_2d: V features in 2D layout (B, C, H, W) for DWConv local supplement.
                  If None and local_dwconv is enabled, v_2d is derived from v_input
                  using the stored spatial shape.

        Returns:
            Output tensor of shape (B, N, C).
        """
        b, n, c = qk_input.shape
        h, d = self.num_heads, self.head_dim
        m = self.num_agents

        q = self.q_proj(qk_input).view(b, n, h, d).permute(0, 2, 1, 3)
        k = self.k_proj(qk_input).view(b, n, h, d).permute(0, 2, 1, 3)
        v = self.v_proj(v_input).view(b, n, h, d).permute(0, 2, 1, 3)

        agent = self.agent.expand(b, -1, -1).view(b, m, h, d).permute(0, 2, 1, 3)

        agent_to_kv = torch.matmul(agent, k.transpose(-2, -1)) * self.scale
        agent_to_kv = agent_to_kv.softmax(dim=-1)
        agent_to_kv = self.attn_dropout(agent_to_kv)
        agent_value = torch.matmul(agent_to_kv, v)

        q_to_agent = torch.matmul(q, agent.transpose(-2, -1)) * self.scale
        q_to_agent = q_to_agent + self.agent_bias
        q_to_agent = q_to_agent.softmax(dim=-1)
        q_to_agent = self.attn_dropout(q_to_agent)
        out = torch.matmul(q_to_agent, agent_value)

        out = out.permute(0, 2, 1, 3).reshape(b, n, c)

        if self.local_dwconv is not None:
            if v_2d is None:
                v_2d = v_input.permute(0, 2, 1).view(b, c, self._h, self._w)
            local_out = self.local_dwconv(v_2d)
            local_out = local_out.flatten(2).permute(0, 2, 1)
            out = out + local_out

        return self.out_proj(out)


import math


class AIFI_AgentAttentionV2(nn.Module):
    """RT-DETR AIFI with Agent Attention V2 (three critical fixes applied).

    Fixes vs V1:
      1. V does NOT contain position encoding (DETR design convention)
      2. Default num_agents=144 for P5(20x20=400 tokens, 2.8:1 ratio)
      3. DWConv local supplement from original paper (compensates low-rank)

    Args:
        c1: Input/output channels.
        cm: FFN hidden dimension.
        num_heads: Number of attention heads.
        num_agents: Number of learnable agent tokens (default: 144).
        dropout: Dropout probability.
        act: Activation module used in the FFN.
        normalize_before: Use pre-normalization if True; AIFI default is False.
        local_kernel_size: DWConv kernel for local supplement (0 to disable).
    """

    def __init__(self, c1, cm=2048, num_heads=8, num_agents=144,
                 dropout=0.0, act=nn.GELU(), normalize_before=False,
                 local_kernel_size=3):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before

        self.attn = AgentAttentionV2(
            embed_dim=c1,
            num_heads=num_heads,
            num_agents=num_agents,
            dropout=dropout,
            local_kernel_size=local_kernel_size,
        )
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act

        self.register_buffer('pos_embed_cache', None, persistent=False)
        self.cached_hw = None

    def get_pos_embed(self, H, W, device, dtype):
        """Get or compute cached 2D sine-cosine positional embedding."""
        if self.cached_hw == (H, W) and self.pos_embed_cache is not None:
            return self.pos_embed_cache

        grid_w = torch.arange(W, dtype=torch.float32, device=device)
        grid_h = torch.arange(H, dtype=torch.float32, device=device)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')

        assert self.c1 % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.c1 // 4
        omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim
        omega = 1.0 / (10000.0 ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        pos_embed = torch.cat([torch.sin(out_w), torch.cos(out_w),
                               torch.sin(out_h), torch.cos(out_h)],
                              dim=1)[None].to(dtype)

        self.pos_embed_cache = pos_embed
        self.cached_hw = (H, W)
        return pos_embed

    def forward(self, x):
        """Forward a feature map of shape (B, C, H, W).

        Key fix: pos_embed is added to Q/K input only, NOT to V input.
        This follows the DETR design convention:
          - pos controls "where to look" (attention weights via Q/K)
          - V provides "what to look at" (aggregated content, no pos)
        """
        b, c, h, w = x.shape
        if c != self.c1:
            raise ValueError(f'AIFI_AgentAttentionV2 expected {self.c1} channels, got {c}')

        self.attn._h = h
        self.attn._w = w

        pos_embed = self.get_pos_embed(h, w, x.device, x.dtype)

        src = x.flatten(2).permute(0, 2, 1)
        qk_src = src + pos_embed
        v_src = src
        v_2d = x

        if self.normalize_before:
            src_norm = self.norm1(src)
            qk_norm = src_norm + pos_embed
            v_norm = src_norm
            src2 = self.attn(qk_norm, v_norm, v_2d=x)
            src = src + self.dropout1(src2)
            src_norm = self.norm2(src)
            src2 = self.fc2(self.dropout(self.act(self.fc1(src_norm))))
            src = src + self.dropout2(src2)
        else:
            src2 = self.attn(qk_src, v_src, v_2d=v_2d)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()
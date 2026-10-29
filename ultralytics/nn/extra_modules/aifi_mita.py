# MiTA-AIFI: Mixture-of-Top-k Attention (Wen et al., 2025) adapted for RT-DETR AIFI.
#
# 论文: "Mixture-of-Top-k Attention: Efficient Attention via Scalable Fast Weights"
#       (BUPT, NeurIPS-style 2025). https://github.com/QishuaiWen/MiTA
#
# 设计动机(本地实验对照):
#   - BlockSparse V1(纯路由, 刚性块专家)是 60+ AIFI 实验唯一赢家 (0.3219 vs base 0.3194);
#   - AgentAttention(纯压缩退化案例)两次失败 (0.3165 / 0.2920);
#   - MiTA 按论文分类学正是两者的原则性组合:
#       路由: landmark query 的 top-k KV 对 -> query-aware 可变形专家
#             (修复 BlockSparse 刚性块专家的"rigidity"缺陷)
#       压缩: landmark value -> 共享专家(全局粗摘要, BlockSparse 缺失的部分)
#
# 机制:
#   m 个 landmark query(均匀窗口平均池化得到), 一物两用:
#     (a) 对全序列 K 做 top-k 注意 -> 第 i 个可变形专家的 KV 集合 I_i  (Eq.7)
#     (b) 对全序列做 cross-attn 得 landmark value V~_i                (Eq.8)
#   每 query 路由到 top-s 个专家, attend 到:
#     K* = [Q~(共享专家keys), K^(e1)..K^(es)]   (Eq.11)
#     V* = [V~,            V^(e1)..V^(es)]      (Eq.12)
#   复杂度 O(N(m + k·s)), N=400 时 m=25,k=64,s=1 -> 89/400 keys.
#
# N=400 下直接稠密拼接注意力(无需 FlashAttention 工程化); gather 用索引收集实现.

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MiTAAttention', 'AIFI_MiTA']


class MiTAAttention(nn.Module):
    """Mixture-of-Top-k attention with landmark-query routing + compression."""

    def __init__(self, dim, num_heads=8, num_landmarks=25, topk=64, s_routed=1,
                 dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim ({dim}) must be divisible by num_heads ({num_heads})')
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_landmarks = int(num_landmarks)
        self.topk = int(topk)
        self.s_routed = int(s_routed)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def _landmark_pool(self, t, h, w):
        """t: [B, N, C] -> pooled landmark features [B, m, C] (uniform windows)."""
        b, n, c = t.shape
        mh = max(1, int(round((self.num_landmarks * h / w) ** 0.5)))
        mw = max(1, self.num_landmarks // mh)
        pooled = F.adaptive_avg_pool2d(t.transpose(1, 2).reshape(b, c, h, w), (mh, mw))
        return pooled.flatten(2).transpose(1, 2)   # [B, mh*mw, C]

    def forward(self, x):
        b, n, c = x.shape
        h, d = self.num_heads, self.head_dim
        hh = int(n ** 0.5)
        # AIFI 输入恒为方阵网格(20x20); 兜底: 非平方数按矩形近似
        w = n // hh if hh * hh == n else n

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, n, h, d).permute(0, 2, 1, 3) for t in qkv]  # [B,h,N,d]

        # ---- landmark queries: 均匀窗口平均池化 + qkv 投影 ----
        lm = self._landmark_pool(x, hh, w)                          # [B, m, C]
        m = lm.shape[1]
        q_lm = self.qkv(lm)[..., :c].view(b, m, h, d).permute(0, 2, 1, 3)   # [B,h,m,d]

        k_eff = min(self.topk, n)
        s = min(self.s_routed, m)

        # ---- landmark 对全序列的注意力(路由与压缩共用) ----
        lm_logits = torch.matmul(q_lm, k.transpose(-2, -1)) * self.scale   # [B,h,m,N]
        lm_attn = lm_logits.softmax(dim=-1)
        v_lm = torch.matmul(lm_attn, v)                              # [B,h,m,d] 共享专家 values

        # ---- 每个landmark的top-k KV集合 -> 可变形专家 ----
        topk_idx = lm_logits.topk(k=k_eff, dim=-1).indices           # [B,h,m,k]

        # ---- N个query -> m个expert 的路由(按query与landmark的内容相似度) ----
        route_logits = torch.matmul(q, q_lm.transpose(-2, -1))       # [B,h,N,m]
        e_top = route_logits.topk(k=s, dim=-1).indices               # [B,h,N,s]

        # ---- gather 路由专家的 KV (Eq.6/11): 沿expert维gather后展开 ----
        e_flat = e_top.reshape(b, h, -1)                             # [B,h,N*s]
        idx = topk_idx.gather(2, e_flat.unsqueeze(-1).expand(-1, -1, -1, k_eff))  # [B,h,N*s,k]
        idx = idx.reshape(b, h, n, s * k_eff)                        # [B,h,N,s,k]
        k_sel = k.gather(2, idx.reshape(b, h, -1, 1).expand(-1, -1, -1, d).reshape(b, h, -1, d))  # [B,h,N*s*k,d]
        v_sel = v.gather(2, idx.reshape(b, h, -1, 1).expand(-1, -1, -1, d).reshape(b, h, -1, d))

        # ---- 联合注意力: 共享专家(m个landmark KV) + 路由专家(s*k个原始KV) (Eq.10-12) ----
        # softmax 在拼接后的全部 keys 上计算, 再分块加权求和(数学上等价于对拼接 KV 的标准注意力)
        k_sel = k_sel.view(b, h, n, s * k_eff, d)
        v_sel = v_sel.view(b, h, n, s * k_eff, d)

        logits_sh = torch.matmul(q, q_lm.transpose(-2, -1))          # [B,h,N,m]  共享专家
        logits_rt = torch.matmul(q.unsqueeze(3), k_sel.transpose(-2, -1)).squeeze(3)  # [B,h,N,s*k]
        logits = torch.cat([logits_sh, logits_rt], dim=-1) * self.scale
        attn = self.attn_drop(logits.softmax(dim=-1))
        attn_sh, attn_rt = attn.split([m, s * k_eff], dim=-1)

        out_sh = torch.matmul(attn_sh, v_lm)                         # [B,h,N,d]
        out_rt = torch.matmul(attn_rt.unsqueeze(3), v_sel).squeeze(3)  # [B,h,N,d]
        out = (out_sh + out_rt).permute(0, 2, 1, 3).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class AIFI_MiTA(nn.Module):
    """AIFI with Mixture-of-Top-k attention (landmark routing + compression).

    Args:
        c1: Input/output channels (256 for P5).
        cm: FFN hidden dim (1024).
        num_heads: Attention heads.
        num_landmarks: Landmark query count (uniform window avg-pool grid).
        topk: KV pairs per routed expert (per landmark).
        s_routed: Experts each query is routed to (1 = Llama-4 style).
        dropout: Dropout probability.
        act: FFN activation.
        normalize_before: Pre-norm if True.
    """

    def __init__(self, c1, cm=2048, num_heads=8, num_landmarks=25, topk=64,
                 s_routed=1, dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.c1 = c1
        self.normalize_before = normalize_before
        self.attn = MiTAAttention(
            dim=c1, num_heads=num_heads, num_landmarks=num_landmarks,
            topk=topk, s_routed=s_routed, dropout=dropout)
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
            raise ValueError(f'AIFI_MiTA expected {self.c1} channels, got {c}')

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





# AIFI_HSB: Hetero-Scale Block-Sparse intra-scale interaction relocated to P4.
#
# 位置8"下放P4"组合拳的三个方案共用模块:
#   方案一 HS-BS  : S5 保留 AIFI_BlockSparse + P4 新增本模块(anchor_grid=0)
#   方案二 HS-R   : S5 换 AIFI_Passthrough + P4 独占本模块(anchor_grid=0)
#   方案三 HS-A   : S5 保留 + P4 本模块(anchor_grid=5, 追加 25 个 P5 块摘要锚)
#
# 设计公理(本仓库 60+ 实验复盘导出):
#   1. P4 交互保持块稀疏形态(信息密度 ~1%), 绝不做全量混合——BlockSparse V1
#      原样复用(400 个 1x4 条带块, top-4 路由), 只是把作用尺度从 20x20 换到
#      40x40(1600 token, 32px 目标在 P4 尚有 2x2=4 token 的空间结构)。
#   2. step-0 恒等: 内部采用 pre-norm + 两条残差各乘零初始化逐通道 LayerScale
#      (g1/g2), 位置编码只进注意力分支不进残差流 => 初始前向 P4_out == P4_proj,
#      与当前冠军拓扑(S5-BS, 0.3219)逐位相等, 训练差异全部归因于新机制。
#   3. 锚点分支(方案三): 25 个 P5 块摘要锚(对 Y5_up 固定 5x5 池化, 零新增
#      投影参数), P4 token 经独立小 MHA(1600x25) 只读查询, 零初始化标量门 g_a
#      => 初始等价于方案一。与已失败的 RABSA 寄存器不同: 锚点是真实 P5 特征
#      (跨尺度、单向、不写回), 非同尺度可学习全局量。
#
# 显存/算力(640 输入, batch=4): BlockSparse V1 为稠密打分+掩码, P4 注意力矩阵
#   [B,8,1600,1600] fp32 约 330MB(反传缓存约再翻倍), GFLOPs 约 +3.2G(+4.2%)。
#   若显存吃紧可将 batch 降至 2 或改用 additive-mask + SDPA memory-efficient
#   内核(不影响逻辑)。

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aifi_block_sparse import AIFI_BlockSparse, BlockSparseAttention

__all__ = ['AIFI_HSB', 'AIFI_Passthrough']


class AIFI_Passthrough(nn.Module):
    """恒等直通层(方案二): 替换 S5 的 AIFI, 保持 yaml 拓扑与解析兼容。"""

    def __init__(self, c1):
        super().__init__()
        self.c1 = c1

    def forward(self, x):
        return x


class P5AnchorRead(nn.Module):
    """P4 token 对 25 个 P5 块摘要锚的只读小 MHA(查询 1600 x 键值 25)。

    与主稀疏注意力完全解耦的独立分支: 自有 q/kv/o 投影, 不占用也不稀释
    块稀疏掩码内的注意力质量。
    """

    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim ({dim}) must be divisible by num_heads')
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.o = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, anchors):
        # x: [B, N, C] P4 tokens; anchors: [B, M, C] P5 块摘要(内容特征, 无位置编码)
        b, n, c = x.shape
        m = anchors.shape[1]
        h, d = self.num_heads, c // self.num_heads
        q = self.q(x).view(b, n, h, d).transpose(1, 2)
        k, v = self.kv(anchors).chunk(2, dim=-1)
        k = k.view(b, m, h, d).transpose(1, 2)
        v = v.view(b, m, h, d).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.drop(self.o(out.transpose(1, 2).reshape(b, n, c)))


class AIFI_HSB(nn.Module):
    """[Y5_up, P4_proj] -> P4-shaped block-sparse intra-scale interaction.

    Args:
        c1: 输入通道列表 [c_y5, c_p4] (多输入解析) 或 int (仅 P4).
        cm: FFN hidden 维度.
        num_heads: 块稀疏注意力头数.
        anchor_grid: P5 摘要锚网格边长(0=关闭锚点分支即方案一/二; 5=25 锚即方案三).
        block_size: 每个 1D 条带块的 token 数.
        topk: 每个 query 块路由的 key 块数.
        dropout: Dropout probability.
        act: FFN 激活函数.
    """

    def __init__(self, c1, cm=1024, num_heads=8, anchor_grid=0,
                 block_size=4, topk=4, dropout=0.0, act=nn.GELU()):
        super().__init__()
        if isinstance(c1, (list, tuple)):
            if len(set(c1)) != 1:
                raise ValueError(f'AIFI_HSB expects equal input channels, got {c1}')
            c1 = c1[-1]
        self.c1 = c1
        self.anchor_grid = int(anchor_grid)

        # P4 尺度内块稀疏交互: BlockSparse V1 原样复用
        self.attn = BlockSparseAttention(
            dim=c1, num_heads=num_heads, block_size=block_size, topk=topk,
            dropout=dropout)
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act

        # step-0 恒等的零初始化逐通道门 (方案公理2)
        self.g1 = nn.Parameter(torch.zeros(c1))
        self.g2 = nn.Parameter(torch.zeros(c1))

        # 方案三: P5 块摘要锚只读分支 (标量零初始化门)
        if self.anchor_grid > 0:
            self.anchor_read = P5AnchorRead(c1, num_heads=num_heads,
                                            dropout=dropout)
            self.g_a = nn.Parameter(torch.zeros(1))

    def forward(self, feats):
        y5_up, p4 = feats
        b, c, h, w = p4.shape
        if c != self.c1:
            raise ValueError(f'AIFI_HSB expected {self.c1} channels, got {c}')

        pos = AIFI_BlockSparse.build_2d_sincos_position_embedding(w, h, c).to(
            device=p4.device, dtype=p4.dtype)
        src = p4.flatten(2).permute(0, 2, 1)          # 干净残差流(不加位置编码)
        inp = src + pos                                # 位置编码只进注意力分支

        # --- P4 块稀疏交互 (pre-norm + 零初始化门) ---
        src = src + self.g1 * self.dropout1(self.attn(self.norm1(inp)))

        # --- P5 语义锚只读分支 (方案三) ---
        if self.anchor_grid > 0:
            a = F.adaptive_avg_pool2d(y5_up, self.anchor_grid)
            a = a.flatten(2).permute(0, 2, 1)          # [B, 25, C] 内容摘要
            src = src + self.g_a * self.anchor_read(src, a)

        # --- FFN (pre-norm + 零初始化门) ---
        src = src + self.g2 * self.dropout2(
            self.fc2(self.dropout(self.act(self.fc1(self.norm2(src))))))

        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()

    def extra_repr(self):
        s = (f'c1={self.c1}, cm={self.fc1.out_features}, '
             f'anchor_grid={self.anchor_grid}, '
             f'|g1|={self.g1.detach().abs().mean():.4f}, '
             f'|g2|={self.g2.detach().abs().mean():.4f}')
        if self.anchor_grid > 0:
            s += f', |g_a|={self.g_a.item():.4f}'
        return s


def tie_hsb_to_s5_aifi(hsb, s5_aifi):
    """可选: 方案一共享权重版——把 P4 的 HSB 层与 S5 的 AIFI_BlockSparse 权重绑定。

    绑定 attn(qkv/proj) 与 FFN(fc1/fc2); norm 与门控保持独立(尺度统计不同)。
    用法(构建模型后、训练前调用一次):
        tie_hsb_to_s5_aifi(model.model.model[14], model.model.model[10])
    """
    hsb.attn.qkv = s5_aifi.attn.qkv
    hsb.attn.proj = s5_aifi.attn.proj
    hsb.fc1 = s5_aifi.fc1
    hsb.fc2 = s5_aifi.fc2
    return hsb

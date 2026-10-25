# PCS-SA: Progressive Competition-Schedule Self-Attention (RT-DETR decoder innovation).
#
# 灵感来源(原则级迁移): Edge-RecViT: Efficient Vision Transformer via
#           Semantic-Refined Dynamic Recursion, CVPR 2026 —— "表征塌缩应沿深度
#           方向用调度手段治理"(中层递归缓解over-smoothing的实证结论)。
#   - 机制: 每层一个可学习温度标量 lambda_l = 1 + delta_l (delta零初始化),
#     以缩放query输入的方式等价缩放自注意力 logits (W_q线性 => logits精确
#     乘lambda; k/v不动, softmax语义保持, -inf掩码在lambda>0下不受影响):
#         logits' = lambda_l * logits
#     预期学习形态: 浅层lambda<1(软化注意力, query保持多样性充分探索),
#     深层lambda>1(锐化注意力, 果断去重) —— 把"先探索后去重"变成可微分课程。
#     同时是仓库文档《Decoder改进位置详解》4.4条(逐层稀疏度递增, 从未试过)
#     的最小干预实现。
#   - 参数增量 = ndl 个标量; 温度可融合进QK权重导出(TRT零新增算子)。
#   - DN queries 走原始路径(_InnovSelfAttnLayer), 调度仅作用于 matching 部分。

import torch
import torch.nn as nn

from .decoder_innov_common import (_InnovSelfAttnLayer, _RTDETRDecoderInnov,
                                    DeformableTransformerDecoderInnov)

__all__ = ('DecoderLayerPCSSA', 'RTDETRDecoderPCSSA')


class DecoderLayerPCSSA(_InnovSelfAttnLayer):
    """Matching-query 自注意力: 逐层可学习温度(竞争强度课程调度)。"""

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0.,
                 act=nn.ReLU(), n_levels=4, n_points=4):
        super().__init__(d_model, n_heads, d_ffn, dropout, act, n_levels, n_points)
        self.delta = nn.Parameter(torch.zeros(1))     # lambda = 1 + delta, 恒等起步

    def _innov_attn(self, mt, ref_xy, pos, boxes):
        q = k = mt if pos is None else mt + pos
        # W_q 线性: (lambda*q)W_q = lambda*(qW_q) => logits 精确缩放 lambda
        q = q * (1.0 + self.delta)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1),
                             mt.transpose(0, 1))[0].transpose(0, 1)
        return self.norm1(mt + self.dropout1(tgt))


class RTDETRDecoderPCSSA(_RTDETRDecoderInnov):
    """RTDETRDecoder + PCS-SA(层递进竞争调度自注意力)。"""

    def _rebuild_decoder(self, hd, nh, ndl, d_ffn, dropout, act, ndp,
                         eval_idx, **kw):
        layer = DecoderLayerPCSSA(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        return DeformableTransformerDecoderInnov(hd, layer, ndl, eval_idx)

    def extra_repr(self):
        lams = [f'{1.0 + l.delta.item():.3f}' for l in self.decoder.layers]
        return f'PCS-SA lambda per layer={lams}'

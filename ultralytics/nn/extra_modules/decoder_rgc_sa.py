# RGC-SA: Reference-Geometric Correspondence Self-Attention (RT-DETR decoder innovation).
#
# 灵感来源(原则级迁移): RefPose: Leveraging Reference Geometric Correspondences
#           for Accurate 6D Pose Estimation, CVPR 2025 —— "实体间交互应由几何
#           对应关系显式引导, 而非仅靠内容相似度"。
#   - 基线解码器自注意力打分完全不知道 query 的几何状态: 空间相距半张图的
#     query 与真正空间重叠的竞争者被打分一视同仁 -> 去重压力泼向无关query
#     (无效串扰->同质化), 真正需要互相竞争的重叠者未必充分交互。
#   - 机制: 每对query的参考框相对几何(尺度归一化中心距 + 对数尺寸比) 经小MLP
#     生成逐头加性 logit 偏置, 以 float attn_mask 方式注入标准 MHA:
#         logits' = logits + gamma * MLP(rel(b_i, b_j))
#     QKV/softmax/残差全部原样; DN queries 走原始布尔掩码路径(偏置仅作用于
#     matching-matching 块, 经 _InnovSelfAttnLayer 的 dn/matching 分流保证)。
#   - gamma 零初始化 => step-0 逐位等价; MLP 末层常规初始化(单守门)。
#   - 参数增量 ~0.8K; 偏置计算 <0.02 GFLOPs; 全静态稠密算子。

import torch
import torch.nn as nn

from .decoder_innov_common import (_InnovSelfAttnLayer, _RTDETRDecoderInnov,
                                    DeformableTransformerDecoderInnov)

__all__ = ('DecoderLayerRGCSA', 'RTDETRDecoderRGCSA')


class DecoderLayerRGCSA(_InnovSelfAttnLayer):
    """Matching-query 自注意力: 参考框几何关系 -> 逐头加性 logit 偏置。"""

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0.,
                 act=nn.ReLU(), n_levels=4, n_points=4, hidden=64):
        super().__init__(d_model, n_heads, d_ffn, dropout, act, n_levels, n_points)
        self.n_heads = n_heads
        self.rel_mlp = nn.Sequential(nn.Linear(4, hidden), nn.GELU(),
                                     nn.Linear(hidden, n_heads))
        # 末层常规初始化(活性), gamma 单守门
        self.gamma = nn.Parameter(torch.zeros(n_heads))

    def _geometric_bias(self, boxes):
        """boxes: [B, N, 4] cxcywh(已sigmoid, [0,1]) -> [B*nh, N, N] 加性偏置。"""
        b, n = boxes.shape[:2]
        xy, wh = boxes[..., :2], boxes[..., 2:].clamp(min=1e-4)
        scale = (wh[:, :, None, :] * wh[:, None, :, :]).clamp(min=1e-8).sqrt()
        dxy = (xy[:, :, None, :] - xy[:, None, :, :]) / scale            # [B,N,N,2]
        log_ratio = torch.log((wh[:, :, None, :] / wh[:, None, :, :]).clamp(min=1e-6))
        rel = torch.cat([dxy, log_ratio], dim=-1)                        # [B,N,N,4]
        bias = self.rel_mlp(rel) * self.gamma                            # [B,N,N,nh]
        return bias.permute(0, 3, 1, 2).reshape(b * self.n_heads, n, n)

    def _innov_attn(self, mt, ref_xy, pos, boxes):
        q = k = mt if pos is None else mt + pos
        bias = self._geometric_bias(boxes)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1),
                             mt.transpose(0, 1), attn_mask=bias)[0].transpose(0, 1)
        return self.norm1(mt + self.dropout1(tgt))


class RTDETRDecoderRGCSA(_RTDETRDecoderInnov):
    """RTDETRDecoder + RGC-SA(参考几何对应自注意力偏置)。

    Args: 与 RTDETRDecoder 一致; hidden: 关系MLP隐宽(默认64)。
    """

    def _rebuild_decoder(self, hd, nh, ndl, d_ffn, dropout, act, ndp,
                         eval_idx, **kw):
        layer = DecoderLayerRGCSA(hd, nh, d_ffn, dropout, act, self.nl, ndp,
                                  hidden=kw.get('hidden', 64))
        return DeformableTransformerDecoderInnov(hd, layer, ndl, eval_idx)

    def extra_repr(self):
        g = self.decoder.layers[0].gamma.detach().abs().mean().item()
        return f'RGC-SA |gamma|_mean={g:.4f}'

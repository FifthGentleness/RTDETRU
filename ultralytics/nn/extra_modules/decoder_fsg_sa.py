# FSG-SA: Focus-Small Gated Self-Attention (RT-DETR decoder innovation).
#
# 灵感来源: LSNet: See Large, Focus Small, CVPR 2025 (核心门控机制迁移,
# 此前仅引用过其"FFN平凡"的结论, 机制从未被使用)。
#   - 自注意力输出不作为内容加性注入(标准残差会把其他query的平均内容灌入,
#     密集场景导致query同质化/过度去重/漏检), 而是作为门控信号调制query自身:
#         q' = q + q ⊙ tanh(MLP(A)) · gamma
#   - gamma 零初始化 => step-0 与基线解码器逐位相等; MLP 常规初始化(单守门,
#     避免双零死锁)
#   - 标准 nn.MultiheadAttention 主干原样复用(权重名与基线一致); DN queries
#     走原始加性路径(_InnovSelfAttnLayer), 仅 matching 部分启用门控调制
#   - 参数增量 ~33K; 与 SwiGLU/StarFFN(FFN隐层门控)、SE(全局池化)形态无关:
#     逐query、由注意力输出条件化的乘性调制

import torch
import torch.nn as nn

from .decoder_innov_common import (_InnovSelfAttnLayer, _RTDETRDecoderInnov,
                                    DeformableTransformerDecoderInnov)

__all__ = ('DecoderLayerFSGSA', 'RTDETRDecoderFSGSA')


class DecoderLayerFSGSA(_InnovSelfAttnLayer):
    """Matching-query 自注意力: 输出门控调制自身(See Large -> Focus Small)。"""

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0.,
                 act=nn.ReLU(), n_levels=4, n_points=4, rank=64):
        super().__init__(d_model, n_heads, d_ffn, dropout, act, n_levels, n_points)
        self.mod = nn.Sequential(
            nn.Linear(d_model, rank), nn.GELU(), nn.Linear(rank, d_model))
        self.gamma = nn.Parameter(torch.zeros(d_model))     # 单守门零初始化

    def _innov_attn(self, mt, ref_xy, pos, boxes):
        q = k = mt if pos is None else mt + pos
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1),
                             mt.transpose(0, 1))[0].transpose(0, 1)
        # 基线加性项保留(step-0保证), 门控调制作为叠加通道: gamma=0 => gate恒1
        gate = 1.0 + torch.tanh(self.mod(self.dropout1(tgt))) * self.gamma
        return self.norm1(mt + self.dropout1(tgt) * gate)


class RTDETRDecoderFSGSA(_RTDETRDecoderInnov):
    """RTDETRDecoder + FSG-SA(自注意力输出门控调制)。

    Args: 与 RTDETRDecoder 一致; rank: 调制MLP低秩宽度(默认64)。
    """

    def _rebuild_decoder(self, hd, nh, ndl, d_ffn, dropout, act, ndp,
                         eval_idx, **kw):
        layer = DecoderLayerFSGSA(hd, nh, d_ffn, dropout, act, self.nl, ndp,
                                  rank=kw.get('rank', 64))
        return DeformableTransformerDecoderInnov(hd, layer, ndl, eval_idx)

    def extra_repr(self):
        g = self.decoder.layers[0].gamma.detach().abs().mean().item()
        return f'FSG-SA rank={self.decoder.layers[0].mod[0].out_features}, |gamma|_mean={g:.4f}'

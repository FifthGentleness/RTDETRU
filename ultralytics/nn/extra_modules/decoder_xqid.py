# X001 D-QID: 查询初始化多样化 (解码器, [C]诊断驱动: query初始坍缩).
# 诊断(2026-10会话): tgt_head随机初始化使300个query内容向量99.9%相同
# (跨query std≈0.001), 自注意力在"相同值"上加权=恒等, query侧改进早期全失效。
# 机制: tgt_head末层权重正交初始化(初始query张成正交基, 多样性最大化),
# 可选每query抖动嵌入(零初始化, 默认关)。参数增量0(默认)。

import torch
import torch.nn as nn

from ultralytics.nn.modules.head import RTDETRDecoder
from .decoder_innov_common import DeformableTransformerDecoderInnov, \
    _RTDETRDecoderInnov

__all__ = ('RTDETRDecoderXQID', 'RTDETRDecoderXND200')


class RTDETRDecoderXQID(RTDETRDecoder):
    """RTDETRDecoder + tgt_head正交初始化(可选零初始化query抖动嵌入)。"""

    def __init__(self, nc=80, ch=(512, 1024, 2048), hd=256, nq=300, ndp=4,
                 nh=8, ndl=6, d_ffn=1024, eval_idx=-1, dropout=0., act=nn.ReLU(),
                 nd=100, label_noise_ratio=0.5, box_noise_scale=1.0,
                 learnt_init_query=False, ortho_gain=1.0, use_jitter=False):
        super().__init__(nc, ch, hd, nq, ndp, nh, ndl, d_ffn, eval_idx,
                         dropout, act, nd, label_noise_ratio, box_noise_scale,
                         learnt_init_query)
        # 查询内容来自 enc_output(Linear+LN)投影的top-k; 正交初始化最大化
        # 初始query方向多样性(治跨query std~0.001的初始坍缩)
        lin = [m for m in self.enc_output if isinstance(m, nn.Linear)][0]
        nn.init.orthogonal_(lin.weight, gain=ortho_gain)
        nn.init.zeros_(lin.bias)
        self.use_jitter = bool(use_jitter)
        if self.use_jitter:
            self.q_jitter = nn.Parameter(torch.zeros(nq, hd))

    def _get_decoder_input(self, feats, shapes, dn_embed, dn_bbox):
        embed, refer, enc_b, enc_s = super()._get_decoder_input(
            feats, shapes, dn_embed, dn_bbox)
        if self.use_jitter:
            n = embed.shape[1]
            embed = embed + self.q_jitter[None, :n]
        return embed, refer, enc_b, enc_s

    def extra_repr(self):
        return f'QID jitter={self.use_jitter}'


class RTDETRDecoderXND200(RTDETRDecoder):
    """X003: 去噪组数 nd 100->200 (训练期监督增密, 推理零开销, [C]文档3.1)。"""

    def __init__(self, nc=80, ch=(512, 1024, 2048), hd=256, nq=300, ndp=4,
                 nh=8, ndl=6, d_ffn=1024, eval_idx=-1, dropout=0., act=nn.ReLU(),
                 nd=200, **kw):
        super().__init__(nc, ch, hd, nq, ndp, nh, ndl, d_ffn, eval_idx,
                         dropout, act, nd, kw.get('label_noise_ratio', 0.5),
                         kw.get('box_noise_scale', 1.0),
                         kw.get('learnt_init_query', False))

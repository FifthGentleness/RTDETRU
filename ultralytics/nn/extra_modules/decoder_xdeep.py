# 批次7深水区解码器机制 (X062/X065, [C]).
# X062 D-EncDoRA: encoder score head DoRA(第三落点, [A:DoRA ICML24])
# X065 D-QScale: 查询内容逐层可学习缩放(init=1恒等)

import torch
import torch.nn as nn
from ultralytics.nn.modules.head import RTDETRDecoder
from .decoder_xmisc import _RowDoRA

__all__ = ('RTDETRDecoderXEncDoRA', 'RTDETRDecoderXQScale')


class RTDETRDecoderXEncDoRA(RTDETRDecoder):
    def __init__(self, nc=80, ch=(512, 1024, 2048), hd=256, nq=300, ndp=4,
                 nh=8, ndl=6, d_ffn=1024, eval_idx=-1, dropout=0., act=nn.ReLU(),
                 nd=100, **kw):
        super().__init__(nc, ch, hd, nq, ndp, nh, ndl, d_ffn, eval_idx,
                         dropout, act, nd, kw.get('label_noise_ratio', 0.5),
                         kw.get('box_noise_scale', 1.0),
                         kw.get('learnt_init_query', False))
        self.enc_score_head = _RowDoRA(self.enc_score_head)


class RTDETRDecoderXQScale(RTDETRDecoder):
    def __init__(self, nc=80, ch=(512, 1024, 2048), hd=256, nq=300, ndp=4,
                 nh=8, ndl=6, d_ffn=1024, eval_idx=-1, dropout=0., act=nn.ReLU(),
                 nd=100, **kw):
        super().__init__(nc, ch, hd, nq, ndp, nh, ndl, d_ffn, eval_idx,
                         dropout, act, nd, kw.get('label_noise_ratio', 0.5),
                         kw.get('box_noise_scale', 1.0),
                         kw.get('learnt_init_query', False))
        self.q_scale = nn.Parameter(torch.ones(hd))   # init=1恒等

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        embed, refer, enc_b, enc_s = super()._get_decoder_input(
            feats, shapes, dn_embed, dn_bbox)
        return embed * self.q_scale, refer, enc_b, enc_s

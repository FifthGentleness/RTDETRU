# 批次8 解码器机制 (X077, [C]).
# X077 D-CrossScale: 交叉注意力输出逐层可学习缩放(init=1恒等)

import torch
import torch.nn as nn
from ultralytics.nn.modules.transformer import (DeformableTransformerDecoder,
                                                DeformableTransformerDecoderLayer)
from ultralytics.nn.modules.head import RTDETRDecoder

__all__ = ('RTDETRDecoderXCrossScale',)


class _CrossScaleLayer(DeformableTransformerDecoderLayer):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.cs = nn.Parameter(torch.ones(1))   # init=1恒等

    def forward(self, embed, refer_bbox, feats, shapes, padding_mask=None,
                attn_mask=None, query_pos=None):
        tgt = self.cross_attn(self.with_pos_embed(embed, query_pos),
                              refer_bbox.unsqueeze(2), feats, shapes, padding_mask)
        embed = self.norm2(embed + self.dropout2(tgt * self.cs))
        # 再跑FFN与norm3(与父类forward_ffn一致)
        from ultralytics.nn.modules.utils import inverse_sigmoid
        return self.forward_ffn(embed)


class RTDETRDecoderXCrossScale(RTDETRDecoder):
    def __init__(self, nc=80, ch=(512, 1024, 2048), hd=256, nq=300, ndp=4,
                 nh=8, ndl=6, d_ffn=1024, eval_idx=-1, dropout=0., act=nn.ReLU(),
                 nd=100, **kw):
        super().__init__(nc, ch, hd, nq, ndp, nh, ndl, d_ffn, eval_idx,
                         dropout, act, nd, kw.get('label_noise_ratio', 0.5),
                         kw.get('box_noise_scale', 1.0),
                         kw.get('learnt_init_query', False))
        layers = nn.ModuleList()
        for i in range(ndl):
            layers.append(_CrossScaleLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp))
        self.decoder = DeformableTransformerDecoder(hd, layers[0], ndl, eval_idx)
        self.decoder.layers = layers

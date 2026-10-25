# X016 D-LIQ / X017 D-BN15: 解码器配置型变体 (解码器, [C]文档1.3/3.2/3.3).
# X016: learnt_init_query=True(可学习初始query与encoder选择混合), 原生开关从未启用。
# X017: box_noise_scale 1.0->1.5, label_noise_ratio 0.5->0.7(小目标定位鲁棒+治cls过拟合)。

import torch.nn as nn
from ultralytics.nn.modules.head import RTDETRDecoder

__all__ = ('RTDETRDecoderXLIQ', 'RTDETRDecoderXBN15')


class RTDETRDecoderXLIQ(RTDETRDecoder):
    """X016: learnt_init_query=True。"""

    def __init__(self, nc=80, ch=(512, 1024, 2048), hd=256, nq=300, ndp=4,
                 nh=8, ndl=6, d_ffn=1024, eval_idx=-1, dropout=0., act=nn.ReLU(),
                 nd=100, **kw):
        super().__init__(nc, ch, hd, nq, ndp, nh, ndl, d_ffn, eval_idx,
                         dropout, act, nd, kw.get('label_noise_ratio', 0.5),
                         kw.get('box_noise_scale', 1.0), True)


class RTDETRDecoderXBN15(RTDETRDecoder):
    """X017: box_noise=1.5, label_noise=0.7(训练期, 推理零开销)。"""

    def __init__(self, nc=80, ch=(512, 1024, 2048), hd=256, nq=300, ndp=4,
                 nh=8, ndl=6, d_ffn=1024, eval_idx=-1, dropout=0., act=nn.ReLU(),
                 nd=100, **kw):
        super().__init__(nc, ch, hd, nq, ndp, nh, ndl, d_ffn, eval_idx,
                         dropout, act, nd, 0.7, 1.5,
                         kw.get('learnt_init_query', False))

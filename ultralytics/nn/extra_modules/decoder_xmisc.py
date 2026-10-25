# X026/X027 解码器变体 (批次3, [C]).
# X026 D-SDrop01: 解码器dropout 0->0.1
# X027 D-HShare: 各层bbox/score头共享(参数大降, 容量换正则)

import torch.nn as nn
from ultralytics.nn.modules.head import RTDETRDecoder

__all__ = ('RTDETRDecoderXSD', 'RTDETRDecoderXHS')


class RTDETRDecoderXSD(RTDETRDecoder):
    def __init__(self, nc=80, ch=(512, 1024, 2048), hd=256, nq=300, ndp=4,
                 nh=8, ndl=6, d_ffn=1024, eval_idx=-1, dropout=0.1, act=nn.ReLU(),
                 nd=100, **kw):
        super().__init__(nc, ch, hd, nq, ndp, nh, ndl, d_ffn, eval_idx,
                         dropout, act, nd, kw.get('label_noise_ratio', 0.5),
                         kw.get('box_noise_scale', 1.0),
                         kw.get('learnt_init_query', False))


class RTDETRDecoderXHS(RTDETRDecoder):
    def __init__(self, nc=80, ch=(512, 1024, 2048), hd=256, nq=300, ndp=4,
                 nh=8, ndl=6, d_ffn=1024, eval_idx=-1, dropout=0., act=nn.ReLU(),
                 nd=100, **kw):
        super().__init__(nc, ch, hd, nq, ndp, nh, ndl, d_ffn, eval_idx,
                         dropout, act, nd, kw.get('label_noise_ratio', 0.5),
                         kw.get('box_noise_scale', 1.0),
                         kw.get('learnt_init_query', False))
        for i in range(1, len(self.dec_bbox_head)):
            self.dec_bbox_head[i] = self.dec_bbox_head[0]
        for i in range(1, len(self.dec_score_head)):
            self.dec_score_head[i] = self.dec_score_head[0]

# X026/X027 解码器变体 (批次3, [C]).
# X026 D-SDrop01: 解码器dropout 0->0.1
# X027 D-HShare: 各层bbox/score头共享(参数大降, 容量换正则)

import torch
import torch.nn as nn
from ultralytics.nn.modules.head import RTDETRDecoder

__all__ = ('RTDETRDecoderXSD', 'RTDETRDecoderXHS', 'RTDETRDecoderXDoRAH', 'RTDETRDecoderXNoDetach')


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


class _RowDoRA(nn.Module):
    """输出权重按行幅向解耦(DoRA原则, 恒等init)。"""

    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        with torch.no_grad():
            self.mag = nn.Parameter(self.linear.weight.norm(dim=1).clone())

    def forward(self, x):
        import torch.nn.functional as F
        d = self.linear.weight / self.linear.weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return F.linear(x, d * self.mag.view(-1, 1), self.linear.bias)


class RTDETRDecoderXDoRAH(RTDETRDecoder):
    """X041: score/bbox头输出权重DoRA化解耦[A:DoRA ICML24]。"""

    def __init__(self, nc=80, ch=(512, 1024, 2048), hd=256, nq=300, ndp=4,
                 nh=8, ndl=6, d_ffn=1024, eval_idx=-1, dropout=0., act=nn.ReLU(),
                 nd=100, **kw):
        super().__init__(nc, ch, hd, nq, ndp, nh, ndl, d_ffn, eval_idx,
                         dropout, act, nd, kw.get('label_noise_ratio', 0.5),
                         kw.get('box_noise_scale', 1.0),
                         kw.get('learnt_init_query', False))
        for i in range(len(self.dec_score_head)):
            self.dec_score_head[i] = _RowDoRA(self.dec_score_head[i])


class RTDETRDecoderXNoDetach(RTDETRDecoder):
    """X048: 层间refer_bbox不detach(梯度贯穿精调链, 训练动力学探针)。"""

    def __init__(self, nc=80, ch=(512, 1024, 2048), hd=256, nq=300, ndp=4,
                 nh=8, ndl=6, d_ffn=1024, eval_idx=-1, dropout=0., act=nn.ReLU(),
                 nd=100, **kw):
        super().__init__(nc, ch, hd, nq, ndp, nh, ndl, d_ffn, eval_idx,
                         dropout, act, nd, kw.get('label_noise_ratio', 0.5),
                         kw.get('box_noise_scale', 1.0),
                         kw.get('learnt_init_query', False))
        from ultralytics.nn.modules.utils import inverse_sigmoid

        def nd_forward(*args, **kwargs):
            (embed, refer_bbox, feats, shapes, bbox_head, score_head,
             pos_mlp) = args[:7]
            attn_mask = kwargs.get('attn_mask', args[7] if len(args) > 7 else None)
            output = embed
            dec_bboxes, dec_cls = [], []
            last = None
            refer_bbox = refer_bbox.sigmoid()
            for i, layer in enumerate(self.decoder.layers):
                output = layer(output, refer_bbox, feats, shapes, None,
                               attn_mask, pos_mlp(refer_bbox))
                bbox = bbox_head[i](output)
                refined = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))
                if self.decoder.training:
                    dec_cls.append(score_head[i](output))
                    dec_bboxes.append(refined if i == 0 else
                                       torch.sigmoid(bbox + inverse_sigmoid(last)))
                elif i == self.decoder.eval_idx:
                    dec_cls.append(score_head[i](output))
                    dec_bboxes.append(refined)
                    break
                last = refined
                refer_bbox = refined            # 不detach(唯一改动)
            return torch.stack(dec_bboxes), torch.stack(dec_cls)

        self.decoder.forward = nd_forward

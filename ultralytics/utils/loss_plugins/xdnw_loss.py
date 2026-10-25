# X047 L-DNw: 去噪支路损失权重 (损失, [C]).
# 机制: *_dn 键损失乘 dn_weight(默认0.5)——DN监督过强可能挤占主匹配监督;
# 1.0等价基线。推理零开销。

from ultralytics.models.utils.loss import RTDETRDetectionLoss

__all__ = ('RTDETRDetectionLossXDNW',)


class RTDETRDetectionLossXDNW(RTDETRDetectionLoss):
    def __init__(self, nc=80, loss_gain=None, aux_loss=True, use_fl=True,
                 use_vfl=False, dn_weight=0.5, **kw):
        super().__init__(nc=nc, loss_gain=loss_gain, aux_loss=aux_loss,
                         use_fl=use_fl, use_vfl=use_vfl, **kw)
        self.dn_weight = float(dn_weight)

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        out = super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
        for k in list(out.keys()):
            if k.endswith('_dn'):
                out[k] = out[k] * self.dn_weight
        return out

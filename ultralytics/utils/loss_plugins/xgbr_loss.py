# X018 L-GBR: GIoU损失增益重调 (损失, [C]).
# 机制: loss_gain['giou'] 2.0->3.0(可配), 定位分支整体加权; 与几何变体不同
# (度量不动, 只调权重); 默认即改, g=2等价基线。

from ultralytics.models.utils.loss import RTDETRDetectionLoss

__all__ = ('RTDETRDetectionLossXGBR',)


class RTDETRDetectionLossXGBR(RTDETRDetectionLoss):
    def __init__(self, nc=80, loss_gain=None, aux_loss=True, use_fl=True,
                 use_vfl=False, giou_gain=3.0, **kw):
        if loss_gain is None:
            loss_gain = {'class': 1, 'bbox': 5, 'giou': float(giou_gain),
                         'no_object': 0.1, 'mask': 1, 'dice': 1}
        super().__init__(nc=nc, loss_gain=loss_gain, aux_loss=aux_loss,
                         use_fl=use_fl, use_vfl=use_vfl, **kw)

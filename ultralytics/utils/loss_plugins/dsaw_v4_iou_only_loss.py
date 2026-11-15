"""Retain DSAWv4 box blending with native classification and plain L1."""

from .dsaw_v4_loss import RTDETRDetectionLossDSAWV4


class RTDETRDetectionLossDSAWV4IoUOnly(RTDETRDetectionLossDSAWV4):
    """Use exactly the native VFL path, including its no-GT fallback."""

    def __init__(self, *args, beta=0.0, alpha=0.0, alpha_l1=0.0, alpha_iou=0.0, **kwargs):
        if any(v != 0 for v in (beta, alpha, alpha_l1, alpha_iou)):
            raise ValueError("IoUOnly requires all extra sample weighting coefficients to be zero")
        kwargs.setdefault('use_vfl', True)
        super().__init__(*args, beta=beta, alpha=alpha, alpha_l1=alpha_l1,
                         alpha_iou=alpha_iou, **kwargs)

    def _get_loss_class_w(self, pred_scores, targets, gt_scores, num_gts, w_cls_map, postfix=''):
        return self._get_loss_class(pred_scores, targets, gt_scores, num_gts, postfix)

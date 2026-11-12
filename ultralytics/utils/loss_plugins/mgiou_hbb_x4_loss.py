from .mgiou_hbb_loss import RTDETRDetectionLossMGIoUHBB


class RTDETRDetectionLossMGIoUHBBX4(RTDETRDetectionLossMGIoUHBB):
    """Quadruple the original MGIoU-HBB geometry loss."""

    def _geometry_loss(self, pred, target):
        return 4.0 * super()._geometry_loss(pred, target)

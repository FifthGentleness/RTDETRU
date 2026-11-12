from .mgiou_hbb_loss import RTDETRDetectionLossMGIoUHBB


class RTDETRDetectionLossMGIoUHBBX2(RTDETRDetectionLossMGIoUHBB):
    """Double the original MGIoU-HBB geometry loss."""

    def _geometry_loss(self, pred, target):
        return 2.0 * super()._geometry_loss(pred, target)

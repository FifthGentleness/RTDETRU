from .gcd_loss import RTDETRDetectionLossGCD


class RTDETRDetectionLossGCDX2(RTDETRDetectionLossGCD):
    """Double the original GCD geometry loss."""

    def _geometry_loss(self, pred, target):
        return 2.0 * super()._geometry_loss(pred, target)

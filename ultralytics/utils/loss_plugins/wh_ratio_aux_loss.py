"""Bounded width/height ratio supervision added to the native RT-DETR loss.

Only the final ordinary decoder matches receive the addition. Native VFL,
L1, GIoU, matching, encoder/intermediate and DN supervision are retained.
This is a stabilized auxiliary adaptation, not the original D3Q box loss.
"""

import math

import torch
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss

__all__ = ('RTDETRDetectionLossWHRatioAux',)


class RTDETRDetectionLossWHRatioAux(RTDETRDetectionLoss):
    """Add scale-gated, gradient-bounded log side errors to final ``loss_bbox``.

    For d in (width, height), eps_d = smooth_px / input_axis_d and
    r_d = log1p(pred_d / eps_d) - log1p(gt_d / eps_d).
    The addition is bbox_gain * lambda * mean(g * sum(eps_d * Huber(r_d))).
    Huber means PyTorch Smooth L1 with beta=huber_beta. The size gate uses
    sqrt(GT pixel area), is one through size_full_px and zero at size_max_px.
    The mean divides by ALL final matches, including inactive large objects.

    Before match normalization, each added side-coordinate derivative is
    bounded by bbox_gain * lambda * g; no center/logit gradient is added.
    Pixel parameters use the actual input dimensions, including rectangular
    inputs. set_epoch() receives a zero-based trainer epoch; the first
    start_epoch epochs have exactly the native objective.
    """

    innovation_schedule = True

    def __init__(self, *args, lambda_max=1., smooth_px=8., size_full_px=16.,
                 size_max_px=32., huber_beta=.1, start_epoch=10, full_epoch=40,
                 use_vfl=True, **kwargs):
        super().__init__(*args, use_vfl=use_vfl, **kwargs)
        values = (lambda_max, smooth_px, size_full_px, size_max_px, huber_beta,
                  start_epoch, full_epoch)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('WH ratio auxiliary parameters must be finite')
        if not (lambda_max >= 0 and smooth_px > 0 and huber_beta > 0
                and 0 < size_full_px < size_max_px):
            raise ValueError('WH ratio auxiliary needs nonnegative lambda, positive smoothing/beta '
                             'and 0 < size_full_px < size_max_px')
        if not (int(start_epoch) == start_epoch and int(full_epoch) == full_epoch
                and 0 <= start_epoch < full_epoch):
            raise ValueError('WH ratio auxiliary needs integer epochs 0 <= start_epoch < full_epoch')
        self.lambda_max = float(lambda_max)
        self.smooth_px, self.huber_beta = float(smooth_px), float(huber_beta)
        self.size_full_px, self.size_max_px = float(size_full_px), float(size_max_px)
        self.start_epoch, self.full_epoch = int(start_epoch), int(full_epoch)
        self._ramp, self._wh_context = 0., None
        self.last_wh_stats = {}

    def set_epoch(self, epoch, total_epochs=0):
        self._ramp = min(1., max(0., (epoch + 1 - self.start_epoch)
                                 / (self.full_epoch - self.start_epoch)))

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        self.last_wh_stats = {}
        if not (self.training and torch.is_grad_enabled() and self.lambda_max and self._ramp):
            return super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
        hw = batch.get('image_hw')
        if (hw is None or len(hw) != 2
                or not all(math.isfinite(v) and v > 0 for v in hw)):
            raise ValueError('WH ratio auxiliary requires positive image_hw=(height, width)')
        boxes, _ = preds
        # Native iteration creates tensor views: recognize the final view by
        # its data pointer, and explicitly exclude the DN postfix as well.
        context = dict(final_ptr=boxes[-1].data_ptr(), hw=hw, seen=False)
        self._wh_context = context
        try:
            losses = super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
        finally:
            self._wh_context = None
        if not context['seen']:
            raise RuntimeError('WH ratio auxiliary did not encounter final ordinary matching')
        return losses

    def _get_loss(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                  masks=None, gt_mask=None, postfix='', match_indices=None):
        context = self._wh_context
        final = (context is not None and not postfix
                 and pred_bboxes.data_ptr() == context['final_ptr'])
        if final:
            if context['seen']:
                raise RuntimeError('WH ratio auxiliary encountered the final stage twice')
            context['seen'] = True
            if match_indices is None:
                # Reuse this exact native assignment for BOTH objectives.
                match_indices = self.matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                                             masks=masks, gt_mask=gt_mask)
        losses = super()._get_loss(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                                   masks=masks, gt_mask=gt_mask, postfix=postfix,
                                   match_indices=match_indices)
        if final:
            idx, gt_idx = self._get_index(match_indices)
            idx = tuple(i.to(pred_bboxes.device) for i in idx)
            gt_idx = gt_idx.to(gt_bboxes.device)
            pred = pred_bboxes[idx].float()
            gt = gt_bboxes[gt_idx].detach().float()
            extra, count = self._width_height_aux(pred, gt, context['hw'])
            strength = self.lambda_max * self._ramp
            extra = self.loss_gain['bbox'] * strength * extra
            losses['loss_bbox'] = losses['loss_bbox'] + extra
            self.last_wh_stats = dict(matches=len(gt), active=count, strength=strength,
                                      aux_loss=extra.detach())
        return losses

    def _width_height_aux(self, pred, gt, hw):
        """Unweighted per-match auxiliary mean, evaluated in FP32 under AMP."""
        pred, gt = pred.float(), gt.detach().float()
        if not len(gt):
            return pred[:, 2:].sum() * 0., gt.new_zeros((), dtype=torch.long)
        axes = pred.new_tensor((hw[1], hw[0]))
        eps = self.smooth_px / axes
        with torch.no_grad():
            gt_sides = gt[:, 2:] * axes
            size = gt_sides.prod(-1).clamp_min(0).sqrt()
            gate = ((self.size_max_px - size)
                    / (self.size_max_px - self.size_full_px)).clamp(0, 1)
            gate *= (gt_sides > 0).all(-1)
        # Decoder widths/heights are sigmoid outputs. log1p avoids dividing
        # two tiny sides and remains differentiable at a predicted zero side.
        error = torch.log1p(pred[:, 2:] / eps) - torch.log1p(gt[:, 2:] / eps)
        huber = F.smooth_l1_loss(error, torch.zeros_like(error), beta=self.huber_beta, reduction='none')
        loss = (gate * (eps * huber).sum(-1)).sum() / len(gt)
        return loss, gate.gt(0).sum()

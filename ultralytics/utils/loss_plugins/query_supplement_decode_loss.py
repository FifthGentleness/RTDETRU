"""Coverage ranking plus GT supervision for the inserted decoder queries."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ultralytics.utils.metrics import bbox_iou

from .query_supplement_loss import RTDETRDetectionLossQuerySupplement, query_supplement_losses


@dataclass(frozen=True)
class QueryDecodeBatch:
    features: torch.Tensor
    shapes: list
    embeddings: torch.Tensor
    references: torch.Tensor
    slots: torch.Tensor
    gt_boxes: torch.Tensor
    gt_classes: torch.Tensor
    weights: torch.Tensor


def build_query_decode_batch(context, image, slots, outside_ids, outside_refs, gt_boxes, gt_classes, weights):
    """Replace the selected slots only in the extra training view."""
    ids = context['top'][image].clone()
    ids[slots] = outside_ids
    refs = context['references'][image:image + 1].clone()
    refs[0, slots] = outside_refs.to(refs.dtype)
    return QueryDecodeBatch(context['feats'][image:image + 1], context['shapes'],
                            context['features'][image:image + 1, ids], refs, slots,
                            gt_boxes, gt_classes, weights)


def _decode_loss(head, job, gains, max_swaps):
    # The decoder is in training mode, so it returns every layer. Native matching
    # queries cannot attend to DN in the main forward either.
    boxes, logits = head.decoder(job.embeddings, job.references, job.features, job.shapes, head.dec_bbox_head,
                                 head.dec_score_head, head.query_pos_head)
    with torch.cuda.amp.autocast(enabled=False):
        boxes, logits = boxes[:, 0, job.slots].float(), logits[:, 0, job.slots].float()
        gt_boxes = job.gt_boxes.unsqueeze(0).expand_as(boxes)
        iou = bbox_iou(boxes.detach(), gt_boxes, xywh=True)
        label = F.one_hot(job.gt_classes, num_classes=logits.shape[-1]).float().unsqueeze(0)
        gt_score = iou * label
        # Same elementwise VFL as the native criterion, before reduction so each
        # inserted query can be weighted by its coverage gain.
        vfl_weight = 0.75 * logits.sigmoid().square() * (1 - label) + gt_score
        vfl = (F.binary_cross_entropy_with_logits(logits, gt_score, reduction='none') * vfl_weight).sum(-1)
        l1 = (boxes - gt_boxes).abs().sum(-1)
        giou = 1 - bbox_iou(boxes, gt_boxes, xywh=True, GIoU=True).squeeze(-1)
        loss = gains['class'] * vfl + gains['bbox'] * l1 + gains['giou'] * giou
        loss = (loss * job.weights).sum() / (len(boxes) * max_swaps)
        with torch.no_grad():
            correct = logits[-1].argmax(-1) == job.gt_classes
            iou = iou[-1, :, 0]
            return loss, ((iou >= 0.50) & correct).sum(), ((iou >= 0.75) & correct).sum()


def query_supplement_decode_losses(head, work, settings, gains):
    """Reuse the ranking loss and add the loss on the replacement queries."""
    losses, stats = query_supplement_losses(head, work, settings, gains)
    if work.decode is not None:
        loss, tp50, tp75 = _decode_loss(head, work.decode, gains, settings.max_swaps)
        losses['loss_query_decode'] = settings.decode_weight * loss
        stats[1] = losses['loss_query_decode'].detach()
        stats[4] = work.decode.slots.numel()
        stats[5] = tp50
        stats[6] = tp75
    return losses, stats


class RTDETRDetectionLossQuerySupplementDecode(RTDETRDetectionLossQuerySupplement):
    decode_builder = staticmethod(build_query_decode_batch)

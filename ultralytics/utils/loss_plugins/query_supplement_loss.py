"""Coverage ranking loss for FSV7 and the shared RT-DETR training hook."""

import sys
from dataclasses import dataclass
from math import isfinite
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.nn.modules.head import RTDETRDecoder
from ultralytics.utils.metrics import bbox_iou, box_iou
from ultralytics.utils.ops import xywh2xyxy

if TYPE_CHECKING:
    from .query_supplement_decode_loss import QueryDecodeBatch


QUERY_SUPPLEMENT_STATS = (
    'query_rank_loss', 'query_decode_loss', 'query_eligible_images', 'query_selected',
    'query_decode_count', 'query_decode_tp50', 'query_decode_tp75',
)


@dataclass(frozen=True)
class QuerySupplementSettings:
    start_epoch: int = 31  # One-based, as shown by the trainer.
    pool_size: int = 300
    max_swaps: int = 3
    rank_weight: float = 0.04
    decode_weight: float = 0.0

    def __post_init__(self):
        for key in ('start_epoch', 'pool_size', 'max_swaps'):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f'query_supplement.{key} must be a positive integer')
        for key in ('rank_weight', 'decode_weight'):
            value = getattr(self, key)
            if not isfinite(value) or value < 0:
                raise ValueError(f'query_supplement.{key} must be finite and nonnegative')

    def enabled(self, epoch):
        return epoch + 1 >= self.start_epoch and (self.rank_weight > 0 or self.decode_weight > 0)


@dataclass(frozen=True)
class QuerySupplementBatch:
    features: torch.Tensor
    classes: torch.Tensor
    cutoffs: torch.Tensor
    weights: torch.Tensor
    batch_size: int
    eligible_images: int
    decode: Optional['QueryDecodeBatch']


def _coverage_value(iou):
    return ((iou - 0.50) / 0.45).clamp(0, 1)


@torch.no_grad()
def select_supplements(iou, k, valid, deficit, max_swaps):
    """Return replaced slots, outside-pool indices, GT owners and detached weights for one image.

    The first k rows of iou are the native top-k, in score order. A swap must add
    owner-specific coverage without losing any GT's geometric coverage at .50 or .75.
    """
    device = iou.device
    empty = torch.empty(0, dtype=torch.long, device=device)
    if not iou.shape[1] or not valid.numel():
        return empty, empty, empty, iou.new_empty(0)

    best_iou, owners = iou.max(1)
    next_iou = iou.topk(2, dim=1).values[:, 1] if iou.shape[1] > 1 else torch.zeros_like(best_iou)
    reliability = (best_iou - next_iou) / best_iou.clamp_min(1e-8)
    quality = _coverage_value(best_iou)
    gt = torch.arange(iou.shape[1], device=device)
    owned = quality[:, None] * (owners[:, None] == gt)
    claimed = torch.zeros_like(gt, dtype=torch.bool)
    slots = torch.arange(k, device=device)
    selected = slots.clone()
    thresholds = iou.new_tensor([0.50, 0.75])
    candidates, weights = [], []
    active = torch.ones((), device=device, dtype=torch.bool)
    owners_out = owners[k:]
    n_swaps = min(max_swaps, k, len(owners_out))
    for step in range(n_swaps):
        slot = k - 1 - step
        remain = torch.cat((selected[:slot], selected[slot + 1:]))
        before_geo = iou[selected].amax(0)
        before = owned[selected].amax(0)
        remain_geo = iou[remain].amax(0) if remain.numel() else torch.zeros_like(before_geo)
        remain_coverage = owned[remain].amax(0) if remain.numel() else torch.zeros_like(before)
        lost = (before_geo[:, None] >= thresholds) & (remain_geo[:, None] < thresholds)
        safe = ~((iou[k:, :, None] < thresholds) & lost).flatten(1).any(1)

        gained = (quality[k:] - remain_coverage[owners_out]).clamp_min(0)
        delta = gained - (before - remain_coverage).sum()
        weight = reliability[k:] * deficit[owners_out] * delta
        weight = torch.where(safe & valid & ~claimed[owners_out] & active, weight, 0.0)
        value, candidate = weight.max(0, keepdim=True)
        active = value[0] > 1e-7  # If the current cutoff is protected, don't skip it.
        owner = owners_out[candidate]
        selected[slot:slot + 1] = torch.where(active, k + candidate, selected[slot:slot + 1])
        claimed |= active & (gt == owner)
        candidates.append(candidate)
        weights.append(value)

    candidates = torch.cat(candidates)
    weights = torch.cat(weights)
    keep = weights > 1e-7
    return slots.flip(0)[:n_swaps][keep], candidates[keep], owners_out[candidates[keep]], weights[keep]


@torch.no_grad()
def build_query_supplement(head, context, targets, settings, step, decode_builder=None):
    """Build a graph-free job using the main branch's already computed matching."""
    last_boxes, last_scores, matches = context['last_boxes'], context['last_scores'], context['matches']
    features, top = context['features'], context['top']
    bs, k = top.shape
    pool_size = min(settings.pool_size, features.shape[1] - k)
    if pool_size <= 0:
        return None
    with torch.cuda.amp.autocast(enabled=False):
        scores = context['scores'].float()
        search = scores.masked_fill(~context['valid'] | ~scores.isfinite(), -torch.inf)
        search.scatter_(1, top, -torch.inf)
        outside_scores, outside = search.topk(pool_size, dim=1)
        outside_features = features.gather(1, outside.unsqueeze(-1).expand(-1, -1, features.shape[-1]))
        box_dtype = next(head.enc_bbox_head.parameters()).dtype
        outside_ref = head.enc_bbox_head(outside_features.to(box_dtype)).float() + context['anchors'][outside].float()
        outside_valid = outside_scores.isfinite() & outside_ref.isfinite().all(-1)
        outside_box = torch.where(outside_valid.unsqueeze(-1), outside_ref.sigmoid(), 0.0)

        gt_boxes, gt_cls = targets['bboxes'].float(), targets['cls']
        height, width = targets['image_hw']
        scale = gt_boxes.new_tensor([width, height, width, height])
        records = []
        rank_features, rank_classes, rank_cutoffs, rank_weights = [], [], [], []
        begin = 0
        for i, (n_gt, (iq, ig)) in enumerate(zip(targets['gt_groups'], matches)):
            end = begin + n_gt
            if n_gt:
                gt_i = gt_boxes[begin:end]
                recovered = gt_i.new_zeros(n_gt)
                if iq.numel():
                    iq, ig = iq.to(features.device), ig.to(features.device)
                    same = last_scores[i, iq].argmax(-1) == gt_cls[ig]
                    recovered[ig - begin] = bbox_iou(last_boxes[i, iq].float() * scale, gt_boxes[ig] * scale,
                                                      xywh=True).squeeze(-1) * same
                initial = torch.cat((context['references'][i].float().sigmoid(), outside_box[i]))
                iou = box_iou(xywh2xyxy(initial) * scale, xywh2xyxy(gt_i) * scale)
                slots, candidates, owners, weights = select_supplements(
                    iou, k, outside_valid[i], 1.0 - _coverage_value(recovered), settings.max_swaps)
                if slots.numel():
                    indices = owners + begin
                    records.append((i, slots, candidates, indices, weights))
                    rank_features.append(outside_features[i, candidates])
                    rank_classes.append(gt_cls[indices])
                    rank_cutoffs.append(scores[i, top[i, slots]].max().expand_as(weights))
                    rank_weights.append(weights)
            begin = end

    if not records:
        return None
    decode = None
    if settings.decode_weight and decode_builder is not None:
        i, slots, candidates, indices, weights = records[step % len(records)]
        decode = decode_builder(context, i, slots, outside[i, candidates], outside_ref[i, candidates],
                                gt_boxes[indices], gt_cls[indices], weights)
    return QuerySupplementBatch(torch.cat(rank_features), torch.cat(rank_classes), torch.cat(rank_cutoffs),
                                torch.cat(rank_weights), bs, len(records), decode)


def query_supplement_losses(head, work, settings, gains):
    """Compute the supplemental ranking loss and its detached telemetry."""
    losses = {}
    stats = work.weights.new_zeros(len(QUERY_SUPPLEMENT_STATS))
    stats[2:4] = stats.new_tensor((work.eligible_images, len(work.weights)))
    if settings.rank_weight:
        with torch.cuda.amp.autocast(enabled=False):
            weights = head.enc_score_head.weight[work.classes].float()
            bias = head.enc_score_head.bias.detach()[work.classes].float()
            z = (work.features.float() * weights).sum(-1) + bias
            loss = (work.weights * F.softplus(work.cutoffs - z)).sum() / (work.batch_size * settings.max_swaps)
        losses['loss_query_rank'] = settings.rank_weight * loss
        stats[0] = losses['loss_query_rank'].detach()
    return losses, stats


class RTDETRDecoderQuerySupplement(RTDETRDecoder):
    innovation_schedule = True

    def __init__(self, nc=80, ch=(512, 1024, 2048), hd=256, nq=300, ndp=4, nh=8, ndl=6, params=None):
        super().__init__(nc, ch, hd, nq, ndp, nh, ndl)
        self.settings = QuerySupplementSettings(**(params or {}))
        self._epoch = 0
        self._step = 0
        self._proposal_context = None

    def set_epoch(self, epoch, total_epochs):
        if epoch != self._epoch:
            self._step = 0
            self._epoch = epoch

    def forward(self, x, batch=None):
        if not self.training or batch is None or not self.settings.enabled(self._epoch) or not any(batch['gt_groups']):
            return super().forward(x, batch)

        context = self._proposal_context = {}
        try:
            output = super().forward(x, batch)
        finally:
            self._proposal_context = None
        batch['_query_supplement_context'] = context
        return output

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        context = self._proposal_context
        if context is None:
            return super()._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Same selection as the native head. Use one topk for the main output
        # and the supervision context so tied scores always use the same slots.
        anchors, valid = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid * feats)
        logits = self.enc_score_head(features)
        scores = logits.max(-1).values
        top = torch.topk(scores, self.num_queries, dim=1).indices
        rows = torch.arange(len(feats), device=top.device)[:, None]
        selected = features[rows, top]
        reference = self.enc_bbox_head(selected) + anchors[0, top]
        enc_boxes, enc_scores = reference.sigmoid(), logits[rows, top]
        context.update(features=features.detach(), feats=feats.detach(), shapes=shapes, anchors=anchors[0],
                       valid=valid[0, :, 0], scores=scores.detach(), top=top, references=reference.detach())

        embeddings, reference = selected.detach(), reference.detach()
        if dn_embed is not None:
            embeddings = torch.cat((dn_embed, embeddings), 1)
        if dn_bbox is not None:
            reference = torch.cat((dn_bbox, reference), 1)
        return embeddings, reference, enc_boxes, enc_scores

    def compute_ground_loss(self, targets, batch=None):
        context = targets.pop('_query_supplement_context', None)
        if context is None:
            return None
        if 'matches' not in context:
            raise RuntimeError('RTDETRDecoderQuerySupplement requires a query supplement loss plugin')
        decode_builder = context['decode_builder']
        if self.settings.decode_weight and decode_builder is None:
            # Earlier combined checkpoints used the rank plugin with a nonzero decode weight.
            from .query_supplement_decode_loss import build_query_decode_batch

            decode_builder = build_query_decode_batch
        work = build_query_supplement(self, context, targets, self.settings, self._step, decode_builder)
        self._step += 1
        if work is None:
            return None

        gains = context['loss_gain']
        if batch is not None and batch.get('_query_supplement_defer', False):
            batch['_query_supplement_work'] = work, gains
            return None

        losses, stats = self.compute_query_supplement(work, gains)
        if batch is not None:
            batch['_query_supplement_stats'] = stats
        return sum(losses.values())

    def compute_query_supplement(self, work, gains):
        if work.decode is not None:
            from .query_supplement_decode_loss import query_supplement_decode_losses

            return query_supplement_decode_losses(self, work, self.settings, gains)
        return query_supplement_losses(self, work, self.settings, gains)


class RTDETRDetectionLossQuerySupplement(RTDETRDetectionLoss):
    """Expose the native last-layer assignment without changing the base loss."""

    decode_builder = None

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        context = batch.get('_query_supplement_context')
        if context is None:
            return super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)

        boxes, scores = preds[0][-1].detach(), preds[1][-1].detach()

        def capture_match(_matcher, inputs, matches):
            # Match only the original last matching layer, never encoder/aux or DN.
            if inputs[0].data_ptr() == boxes.data_ptr() and inputs[1].data_ptr() == scores.data_ptr():
                context.update(matches=matches, last_boxes=boxes, last_scores=scores, loss_gain=self.loss_gain,
                               decode_builder=self.decode_builder)

        with self.matcher.register_forward_hook(capture_match):
            return super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)


# Keep the module names used by existing pickled RT-DETR checkpoints loadable.
sys.modules.setdefault('ultralytics.nn.extra_modules.decoder_query_supplement', sys.modules[__name__])
sys.modules.setdefault('ultralytics.models.utils.query_supplement', sys.modules[__name__])

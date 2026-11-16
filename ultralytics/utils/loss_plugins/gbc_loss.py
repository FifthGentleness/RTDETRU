"""Grid-to-Budget Coverage: train the native RT-DETR proposal head before Top-K."""

import math
from contextlib import ExitStack, contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.metrics import bbox_iou


def _exclusive_assignment(num_gt, rows, tokens, costs, *, cover_first=True):
    """Match legal edges only; each GT gets its own dummy and at most one real token."""
    if not rows.numel():
        return rows.new_full((num_gt,), -1)
    rows_cpu = rows.cpu().numpy()
    tokens_cpu, cols = np.unique(tokens.cpu().numpy(), return_inverse=True)
    values = costs.cpu().numpy().astype(np.float64, copy=False)
    n = len(tokens_cpu)
    cost = np.full((num_gt, n + num_gt), np.inf, dtype=np.float64)
    # With cover_first, one more dummy costs more than the sum of all real costs.
    dummy = (num_gt + 1) * (values.max() + 1.0) if cover_first else 1.0
    diag = np.arange(num_gt)
    cost[diag, n + diag] = dummy
    np.minimum.at(cost, (rows_cpu, cols), values)  # A native Top-K edge may repeat a grid edge.
    i, j = linear_sum_assignment(cost)
    real = j < n
    assignment = np.full(num_gt, -1, dtype=np.int64)
    assignment[i[real]] = tokens_cpu[j[real]]
    return torch.as_tensor(assignment, device=rows.device)


class RTDETRDetectionLossGBC(RTDETRDetectionLoss):
    """Replace only the encoder auxiliary loss, retaining the native decoder and DN objectives."""

    innovation_schedule = True

    def __init__(self, nc=80, enabled=True, geom_gain=0.75, slot_gain=0.50,
                 start_epoch=5, full_epoch=15, box_chunk=1024, use_vfl=True, **kwargs):
        super().__init__(nc=nc, use_vfl=use_vfl, **kwargs)
        if not all(math.isfinite(x) and x >= 0 for x in (geom_gain, slot_gain)):
            raise ValueError('GBC geom_gain and slot_gain must be finite and non-negative')
        if (not 0 <= start_epoch < full_epoch or int(start_epoch) != start_epoch
                or int(full_epoch) != full_epoch or box_chunk < 1 or int(box_chunk) != box_chunk):
            raise ValueError('GBC requires integer epochs with 0 <= start_epoch < full_epoch and box_chunk >= 1')
        if not self.aux_loss or self.use_uni_match:
            raise ValueError('GBC expects the default independent RT-DETR auxiliary matching')
        self.enabled = bool(enabled)
        self.geom_gain, self.slot_gain = float(geom_gain), float(slot_gain)
        self.start_epoch, self.full_epoch, self.box_chunk = int(start_epoch), int(full_epoch), int(box_chunk)
        self.mix = 0.0
        self.last_gbc_stats = {}

    def set_epoch(self, epoch, total_epochs=0):
        self.mix = min(1.0, max(0.0, (epoch + 1 - self.start_epoch) / (self.full_epoch - self.start_epoch)))

    @property
    def needs_encoder_pool(self):
        return self.enabled and self.mix > 0 and self.training and torch.is_grad_enabled()

    @contextmanager
    def capture_encoder(self, head):
        """Borrow existing tensors for one training forward; never change hook outputs."""
        pool = {'head': head}

        def capture_scores(_module, args, output):
            pool.update(features=args[0], logits=output)

        def capture_shapes(_module, args):
            # The native decoder receives the pre-selection projected features and their actual shapes.
            pool.update(shapes=tuple(tuple(hw) for hw in args[3]), anchor_dtype=args[2].dtype)

        with ExitStack() as stack:
            stack.callback(head.enc_score_head.register_forward_hook(capture_scores).remove)
            stack.callback(head.decoder.register_forward_pre_hook(capture_shapes).remove)
            yield pool

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None, encoder_pool=None):
        self.last_gbc_stats = {}
        if not self.needs_encoder_pool:
            return super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
        if encoder_pool is None or not {'features', 'logits', 'shapes', 'anchor_dtype'} <= encoder_pool.keys():
            raise RuntimeError('Active GBC needs the live encoder pool from RTDETRDetectionModel.loss()')

        boxes, scores = preds
        # Slice off only the ordinary encoder auxiliary stage; DN has no encoder stage.
        loss = super().forward((boxes[1:], scores[1:]), batch, dn_bboxes, dn_scores, dn_meta)
        if self.mix < 1.0:
            native = self._get_loss(boxes[0], scores[0], batch['bboxes'], batch['cls'], batch['gt_groups'])
            for name, item in native.items():
                loss[f'{name}_aux'] = loss[f'{name}_aux'] + (1.0 - self.mix) * item
        rep, geom, slot = self._encoder_loss(boxes[0], scores[0], batch, encoder_pool)
        loss.update(loss_gbc_rep=self.mix * rep, loss_gbc_geom=self.mix * self.geom_gain * geom,
                    loss_gbc_slot=self.mix * self.slot_gain * slot)
        return loss

    @staticmethod
    def _grid_edges(gt, shapes, valid):
        """List edges from GT-centred 3x3 windows in the native flattened scale order."""
        shift = torch.arange(-1, 2, device=gt.device)
        dy, dx = torch.meshgrid(shift, shift, indexing='ij')
        gt_id = torch.arange(len(gt), device=gt.device)[:, None].expand(-1, 9)
        rows, tokens, levels = [], [], []
        offset = 0
        for level, (h, w) in enumerate(shapes):
            x = (gt[:, 0] * w).long().clamp(0, w - 1)[:, None] + dx.reshape(1, 9)
            y = (gt[:, 1] * h).long().clamp(0, h - 1)[:, None] + dy.reshape(1, 9)
            ids = offset + y.clamp(0, h - 1) * w + x.clamp(0, w - 1)
            legal = (x >= 0) & (x < w) & (y >= 0) & (y < h) & valid[ids]
            rows.append(gt_id[legal])
            tokens.append(ids[legal])
            levels.append(torch.full_like(tokens[-1], level))
            offset += h * w
        return torch.cat(rows), torch.cat(tokens), torch.cat(levels)

    def _pool_boxes(self, ids, features, anchors, top_slot, top_boxes, box_head):
        """Reuse exact native boxes for selected rows; evaluate the same MLP for the rest."""
        slots = top_slot[ids]
        boxes = top_boxes[slots.clamp_min(0)].float()
        missing = torch.where(slots < 0)[0]
        if missing.numel():
            extra = []
            for chunk in ids[missing].split(self.box_chunk):
                extra.append((box_head(features[chunk]) + anchors[chunk]).sigmoid().float())
            boxes = boxes.index_copy(0, missing, torch.cat(extra))
        return boxes

    @staticmethod
    def _utility(q):
        return ((q[..., None] - q.new_tensor((0.50, 0.60, 0.75, 0.85))) / 0.04).sigmoid().mean(-1)

    @torch.no_grad()
    def _mine_image(self, gt, gt_cls, top_ids, top_boxes, top_scores, features, anchors, valid, shapes, hw, box_head):
        """Choose owners, backgrounds and budget opponents using detached evidence only."""
        n, k = len(valid), len(top_ids)
        device = valid.device
        top_slot = torch.full((n,), -1, dtype=torch.long, device=device)
        slots = torch.arange(k, device=device)
        top_slot[top_ids] = slots
        spatial = torch.zeros_like(valid)
        owned = torch.zeros_like(valid)
        primary = top_ids.new_full((len(gt),), -1)
        spare = primary.clone()
        main_q = gt.new_zeros(len(gt))
        top_q = main_q.clone()
        top_owner = primary.clone()
        q_top = bbox_iou(gt[:, None], top_boxes[None].float()).squeeze(-1)
        top_values, top_cls = top_scores.float().max(-1)
        if len(gt):
            g, j, level = self._grid_edges(gt, shapes, valid)
            spatial[j] = True
            # Keep two already-localized native candidates available to each GT.
            quality, top_pos = q_top.masked_fill(~valid[top_ids], -1.0).topk(min(2, k), dim=1)
            keep = quality >= 0.35
            g = torch.cat((g, torch.arange(len(gt), device=device)[:, None].expand_as(keep)[keep]))
            extra_ids = top_ids[top_pos[keep]]
            boundaries = top_ids.new_tensor([h * w for h, w in shapes]).cumsum(0)[:-1]
            level = torch.cat((level, torch.bucketize(extra_ids, boundaries, right=True)))
            j = torch.cat((j, extra_ids))
            if j.numel():
                pool_ids, inverse = torch.unique(j, sorted=True, return_inverse=True)
                candidate = self._pool_boxes(pool_ids, features, anchors, top_slot, top_boxes, box_head)
                edge_q = bbox_iou(candidate[inverse], gt[g]).flatten()
                px = gt.new_tensor((hw[1], hw[0]))
                side = (gt[:, 2:] * px).prod(-1).clamp_min(0).sqrt()
                # The size convention is the area-equivalent length after mapping to 640x640.
                side640 = gt[:, 2:].prod(-1).clamp_min(0).sqrt() * 640.0
                prefer = torch.bucketize(side640, side640.new_tensor((32.0, 96.0)), right=True)
                distance = ((anchors[j, :2].float().sigmoid() - gt[g, :2]) * px).norm(dim=-1)
                distance = (distance / (side[g] * 0.5).clamp_min(8.0)).clamp_max(4.0)
                cost = distance + 0.25 * (level - prefer[g]).abs() + 2.0 * self.mix * (1 - edge_q)
                primary = _exclusive_assignment(len(gt), g, j, cost)
                main_gt = torch.where(primary >= 0)[0]
                main_ids = primary[main_gt]
                owned[main_ids] = True
                main_box = candidate[torch.searchsorted(pool_ids, main_ids)]
                main_q[main_gt] = bbox_iou(main_box, gt[main_gt]).flatten()
                leftover = ~owned[j] & (primary[g] >= 0) & (main_q[g] < 0.75)
                spare = _exclusive_assignment(len(gt), g[leftover], j[leftover], cost[leftover])
                owned[spare[spare >= 0]] = True

            good = (gt_cls[:, None] == top_cls) & valid[top_ids] & (q_top >= 0.35)
            gi, si = torch.where(good)
            top_owner = _exclusive_assignment(len(gt), gi, si, 1.0 - self._utility(q_top[gi, si]), cover_first=False)
            present = torch.where(top_owner >= 0)[0]
            top_q[present] = q_top[present, top_owner[present]]

        max_top_q = q_top.amax(0) if len(gt) else gt.new_zeros(k)
        clear_top = valid[top_ids] & ~spatial[top_ids] & ~owned[top_ids] & (max_top_q <= 0.10)
        main_gt = torch.where(primary >= 0)[0]
        n_main = len(main_gt)
        max_bg = 2 * n_main if len(gt) else 64
        background = top_ids[:0]
        if max_bg:
            random_budget = min(64, max(1, max_bg // 4))
            eligible = torch.where(valid & ~spatial & ~owned & (top_slot < 0))[0]
            if eligible.numel():
                sample = eligible[torch.randperm(len(eligible), device=device)[:2 * random_budget]]
                if len(gt):
                    boxes = self._pool_boxes(sample, features, anchors, top_slot, top_boxes, box_head)
                    overlap = bbox_iou(boxes[:, None], gt[None]).squeeze(-1).amax(1)
                    sample = sample[overlap <= 0.10]
                background = sample[:random_budget]
            # Native Top-K rows are already ordered by score, highest first.
            background = torch.cat((top_ids[clear_top][:max_bg - len(background)], background))

        slot_gt = top_ids[:0]
        opponent = top_ids[:0]
        utility_gain = (self._utility(main_q) - self._utility(top_q)).clamp_min(0)
        if self.slot_gain and n_main:
            to_promote = (top_slot[primary[main_gt]] < 0) & (main_q[main_gt] >= 0.40)
            to_promote &= (main_q[main_gt] >= top_q[main_gt] + 0.03) & (utility_gain[main_gt] > 0)
            eligible = main_gt[to_promote]
            slot_gt = eligible[utility_gain[eligible].argsort(descending=True, stable=True)[:k]]
            if slot_gt.numel():
                used = torch.zeros(k, dtype=torch.bool, device=device)
                used[top_owner[top_owner >= 0]] = True
                quality, best_gt = q_top.topk(min(2, len(gt)), dim=0)
                other = quality[1] if len(gt) > 1 else 0.0
                winner_gt = best_gt[0]
                duplicate = ((quality[0] >= 0.50) & (other < 0.35) & (top_cls == gt_cls[winner_gt])
                             & (top_q[winner_gt] >= quality[0]))
                safe = valid[top_ids] & ~used & ~owned[top_ids] & (clear_top | duplicate)
                opponent = slots[safe].flip(0)[:len(slot_gt)]  # Lowest safe score first.

        return (main_gt, primary, spare, main_q, top_q, top_slot, background, slot_gt, opponent,
                utility_gain, top_values.amin())

    def _box_cost(self, boxes, gt, hw):
        px = boxes.new_tensor((hw[1], hw[0]))
        center = ((boxes[:, :2] - gt[:, :2]) * px) / (gt[:, 2:] * px).clamp_min(8.0)
        size = (boxes[:, 2:] * px).clamp_min(0.5).log() - (gt[:, 2:] * px).clamp_min(0.5).log()
        xy = F.smooth_l1_loss(center, torch.zeros_like(center), beta=0.25, reduction='none').mean(-1)
        wh = F.smooth_l1_loss(size, torch.zeros_like(size), beta=0.25, reduction='none').mean(-1)
        return (self.loss_gain['bbox'] * (boxes - gt).abs().sum(-1)
                + self.loss_gain['giou'] * (1 - bbox_iou(boxes, gt, GIoU=True).flatten()) + 0.5 * xy + 0.25 * wh)

    @staticmethod
    def _negative_vfl(logits):
        return 0.75 * logits.sigmoid().square() * F.softplus(logits)

    def _encoder_loss(self, top_boxes, top_scores, batch, pool):
        features, logits, shapes, head = (pool[key] for key in ('features', 'logits', 'shapes', 'head'))
        with torch.no_grad():
            anchors, valid = head._generate_anchors(shapes, dtype=pool['anchor_dtype'], device=features.device)
            anchors, valid = anchors[0], valid[0, :, 0]
            # Same dtype, reduction and Top-K call used by the native head (including ties).
            top_ids = torch.topk(logits.detach().max(-1).values, top_boxes.shape[1], dim=1).indices
        hw = batch['image_hw']
        # Link both existing proposal heads even if a local DDP batch has no usable candidates.
        zero = (top_boxes[..., 0].float().sum() + logits[:, 0, 0].float().sum()) * 0.0
        pos, other, neg, geom, slot = (zero,) * 5
        stats = dict(targets=len(batch['bboxes']), primary=0, satellite=0, offpool_primary=0,
                     offpool_satellite=0, background=0, slot_gt=0, slot_opponents=0,
                     top_owned_r75_count=zero.detach(), slot_gap_sum=zero.detach())
        gts = batch['bboxes'].detach().float().split(batch['gt_groups'])
        classes = batch['cls'].split(batch['gt_groups'])
        for b, (gt, cls) in enumerate(zip(gts, classes)):
            args = (gt, cls, top_ids[b], top_boxes[b], top_scores[b], features[b], anchors, valid, shapes, hw,
                    head.enc_bbox_head)
            (main_gt, primary, spare, main_q, top_q, top_slot, bg, slot_gt, opponent,
             utility_gain, cutoff) = self._mine_image(*args)
            main_ids = primary[main_gt]
            spare_gt = torch.where(spare >= 0)[0]
            spare_ids = spare[spare_gt]
            stats['primary'] += len(main_ids)
            stats['satellite'] += len(spare_ids)
            stats['background'] += len(bg)
            stats['offpool_primary'] = stats['offpool_primary'] + (top_slot[main_ids] < 0).sum()
            stats['offpool_satellite'] = stats['offpool_satellite'] + (top_slot[spare_ids] < 0).sum()
            stats['top_owned_r75_count'] = stats['top_owned_r75_count'] + (top_q >= 0.75).sum()
            if main_ids.numel():
                main_z = logits[b, main_ids].float()
                c = cls[main_gt, None]
                y = main_q[main_gt].clamp_min(0.25)
                pos = pos + (y * F.binary_cross_entropy_with_logits(main_z.gather(1, c).flatten(), y,
                                                                    reduction='none')).sum()
                other = other + self._negative_vfl(main_z).scatter(1, c, 0.0).sum()
                ids, truth = torch.cat((main_ids, spare_ids)), torch.cat((main_gt, spare_gt))
                boxes = self._pool_boxes(ids, features[b], anchors, top_slot, top_boxes[b], head.enc_bbox_head)
                weight = torch.where(spare[truth] >= 0, 1.5, 1.0)
                cost = self._box_cost(boxes, gt[truth], hw) / weight
                geom = geom + cost[:len(main_ids)].sum() + 0.5 * cost[len(main_ids):].sum()
            if bg.numel():
                neg = neg + self._negative_vfl(logits[b, bg].float()).sum()
            if slot_gt.numel():
                z = logits[b, primary[slot_gt], cls[slot_gt]].float()
                gain = utility_gain[slot_gt]
                slot = slot + (0.5 * F.softplus((cutoff + 0.20 - z) / 0.5) * gain).sum()
                if opponent.numel():
                    rival = top_scores[b, opponent].float().max(-1).values
                    slot = slot + (0.25 * F.softplus((rival + 0.20 - z[:len(rival)]) / 0.5) * gain[:len(rival)]).sum()
                stats['slot_gt'] += len(z)
                stats['slot_opponents'] += len(opponent)
                stats['slot_gap_sum'] = stats['slot_gap_sum'] + (cutoff - z.detach()).sum()

        den = max(stats['primary'], 1)
        rep = self.loss_gain['class'] * ((pos + 0.5 * other) / den + 0.5 * neg / max(stats['background'], 1))
        geom, slot = geom / den, slot / den
        self.last_gbc_stats = {**stats, 'mix': self.mix, 'rep': rep.detach(),
                               'geom': geom.detach(), 'slot': slot.detach()}
        return rep, geom, slot

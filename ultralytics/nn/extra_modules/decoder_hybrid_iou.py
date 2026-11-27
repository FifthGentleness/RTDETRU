"""Hybrid One-to-Many (O2M) training group + Detached Box-Aligned IoU Head.

Designed for the Base-AIFI supervision/head track (orthogonal to AIFI changes):
1. Hybrid O2M Training Group (zero inference overhead):
   - During training only, selects an extra scale-balanced O2M query group
     (75% from F3 stride-8 for dense small-object coverage, 25% from F4-F5) and
     runs it alongside [DN | O2O] in the shared decoder under a strict 3-way
     isolated self-attention ``attn_mask``.
   - Applies scale-aware multi-positive assignment (small targets get up to
     ``o2m_k`` repeats, larger targets fewer, proportionally capped on crowded
     images so every GT receives positives) via standard VFL + L1 + GIoU.
   - Completely skipped during evaluation/inference (0 extra params/latency in eval).
2. Box-Aligned Detached IoU Head (fixes same-class quality-score inversion):
   - Operates ONLY after the final O2O decoder layer's predicted boxes.
   - Samples 9 box-aligned boundary/center points (4 corners, 4 edge midpoints,
     center) from detached F3 features + detached final query hidden states,
     keeping main backbone/FPN/decoder gradients strictly isolated.
   - Supervised on all final O2O boxes against their best same-class GT IoU.
   - At eval/inference, calibrates final output confidence as
     ``p_cls ** (1 - alpha) * p_iou ** alpha`` while keeping raw logits intact.

Both components have independent YAML toggles for clean single-factor ablations:
- ``o2m_queries=0`` disables Hybrid O2M;
- ``iou_weight=0`` / ``iou_alpha=0`` disables Detached IoU Head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ultralytics.utils.loss import VarifocalLoss
from ultralytics.utils.metrics import bbox_iou

from ..modules.head import RTDETRDecoder
from ..modules.transformer import DeformableTransformerDecoder
from ..modules.utils import inverse_sigmoid

__all__ = ('RTDETRDecoderHybridIoU',)


def _pairwise_cxcywh_iou(boxes1, boxes2):
    """Compute pairwise IoU between [N, 4] and [M, 4] normalized cxcywh boxes."""
    wh1 = boxes1[:, 2:].clamp(min=1e-4)
    wh2 = boxes2[:, 2:].clamp(min=1e-4)
    tl1, br1 = boxes1[:, :2] - 0.5 * wh1, boxes1[:, :2] + 0.5 * wh1
    tl2, br2 = boxes2[:, :2] - 0.5 * wh2, boxes2[:, :2] + 0.5 * wh2
    lt = torch.maximum(tl1.unsqueeze(1), tl2.unsqueeze(0))
    rb = torch.minimum(br1.unsqueeze(1), br2.unsqueeze(0))
    inter_wh = (rb - lt).clamp(min=0.0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    area1 = (wh1[:, 0] * wh1[:, 1]).unsqueeze(1)
    area2 = (wh2[:, 0] * wh2[:, 1]).unsqueeze(0)
    union = (area1 + area2 - inter).clamp(min=1e-6)
    return inter / union


class DeformableTransformerDecoderWithFeat(DeformableTransformerDecoder):
    """DeformableTransformerDecoder that also returns the final hidden states."""

    def forward(self, embed, refer_bbox, feats, shapes, bbox_head, score_head, pos_mlp,
                attn_mask=None, padding_mask=None):
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        last_output = output
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))

            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
                last_output = output
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                last_output = output
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls), last_output


class DetachedBoxAlignedIoUHead(nn.Module):
    """Detached 9-point box-boundary feature readout for localization quality prediction."""

    def __init__(self, hidden_dim=256, proj_dim=64):
        super().__init__()
        self.feat_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, proj_dim, 1, bias=False),
            nn.GroupNorm(8, proj_dim),
            nn.SiLU(),
        )
        self.region_mlp = nn.Sequential(
            nn.Linear(9 * proj_dim, 128),
            nn.SiLU(),
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        # 9 canonical box coordinates: center, 4 edge midpoints, 4 corners
        pts = torch.tensor([
            [0.0, 0.0],
            [-0.5, 0.0], [0.5, 0.0], [0.0, -0.5], [0.0, 0.5],
            [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5],
        ], dtype=torch.float32)
        self.register_buffer('stencil', pts, persistent=False)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.feat_proj[0].weight)
        nn.init.xavier_uniform_(self.region_mlp[0].weight)
        nn.init.zeros_(self.region_mlp[0].bias)
        nn.init.xavier_uniform_(self.query_mlp[0].weight)
        nn.init.zeros_(self.query_mlp[0].bias)
        nn.init.xavier_uniform_(self.head[1].weight)
        nn.init.zeros_(self.head[1].bias)
        nn.init.normal_(self.head[3].weight, std=1e-2)
        nn.init.zeros_(self.head[3].bias)

    def forward(self, pred_boxes, query_feat, f3_map):
        """Predict per-query IoU logits from strictly detached inputs.

        Args:
            pred_boxes (Tensor): [B, Q, 4] normalized cxcywh boxes.
            query_feat (Tensor): [B, Q, C] final decoder hidden states.
            f3_map (Tensor): [B, C, H3, W3] projected F3 feature map.

        Returns:
            Tensor: [B, Q, 1] predicted IoU logits.
        """
        boxes = pred_boxes.detach()
        q_in = query_feat.detach()
        f3_in = f3_map.detach()

        bs, nq, _ = boxes.shape
        proj = self.feat_proj(f3_in)  # [B, proj_dim, H3, W3]
        locs = boxes[..., :2].unsqueeze(2) + boxes[..., 2:4].unsqueeze(2) * self.stencil.to(boxes.dtype).view(1, 1, 9, 2)
        grids = (2.0 * locs - 1.0).clamp(-1.0, 1.0)
        sampled = F.grid_sample(
            proj,
            grids.to(proj.dtype),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )  # [B, proj_dim, Q, 9]
        reg_feat = sampled.permute(0, 2, 3, 1).reshape(bs, nq, -1)
        combined = torch.cat((self.region_mlp(reg_feat), self.query_mlp(q_in)), dim=-1)
        return self.head(combined)


class RTDETRDecoderHybridIoU(RTDETRDecoder):
    """RT-DETR Decoder with Hybrid O2M Training Group + Detached Box-Aligned IoU Head."""

    innovation_schedule = True

    def __init__(self, nc, ch, hd=256, nq=300, ndp=4, nh=8, ndl=3,
                 o2m_queries=300, o2m_weight=0.35, o2m_k=4,
                 iou_alpha=0.35, iou_weight=1.0, warmup_epochs=20):
        super().__init__(nc, ch, hd, nq, ndp, nh, ndl)

        # Preserve decoder layers and weights while returning final hidden states
        old_decoder = self.decoder
        new_decoder = DeformableTransformerDecoderWithFeat(hd, old_decoder.layers[0], ndl, eval_idx=old_decoder.eval_idx)
        new_decoder.layers = old_decoder.layers
        self.decoder = new_decoder

        self.o2m_queries = int(o2m_queries)
        self.o2m_weight = float(o2m_weight)
        self.o2m_k = int(o2m_k)
        self.iou_alpha = float(iou_alpha)
        self.iou_weight = float(iou_weight)
        self.warmup_epochs = max(int(warmup_epochs), 1)

        self._current_epoch = None
        self._total_epochs = None
        self._o2m_cache = None
        self._iou_cache = None

        if self.iou_weight > 0.0 or self.iou_alpha > 0.0:
            self.iou_head = DetachedBoxAlignedIoUHead(hidden_dim=hd, proj_dim=64)
        else:
            self.iou_head = None

    def set_epoch(self, epoch, total_epochs=0):
        self._current_epoch = int(epoch)
        self._total_epochs = int(total_epochs)

    def _schedule_scale(self):
        if self._current_epoch is None:
            return 1.0
        return min(1.0, max(0.0, (self._current_epoch + 1) / self.warmup_epochs))

    def _select_o2m_queries(self, features, anchors, enc_outputs_scores, shapes):
        """Select scale-balanced O2M candidate queries (75% F3 small-object grid, 25% F4-F5)."""
        bs, total_tok, _ = features.shape
        device = features.device
        h3, w3 = shapes[0]
        n_f3 = h3 * w3
        max_scores = enc_outputs_scores.detach().amax(dim=-1)  # [B, total_tok]

        k_total = min(self.o2m_queries, total_tok)
        k_f3 = min(int(round(k_total * 0.75)), n_f3)
        k_hi = min(k_total - k_f3, total_tok - n_f3)
        if k_f3 + k_hi < k_total:
            k_f3 = min(k_total - k_hi, n_f3)

        idx_parts = []
        if k_f3 > 0:
            idx_parts.append(max_scores[:, :n_f3].topk(k_f3, dim=1).indices)
        if k_hi > 0:
            idx_parts.append(max_scores[:, n_f3:].topk(k_hi, dim=1).indices + n_f3)
        o2m_ind = torch.cat(idx_parts, dim=1) if len(idx_parts) > 1 else idx_parts[0]

        batch_ind = torch.arange(bs, device=device)[:, None]
        o2m_feat = features[batch_ind, o2m_ind]
        o2m_anc = anchors[:, o2m_ind.reshape(-1)].view(bs, k_total, 4)
        o2m_refs = (self.enc_bbox_head(o2m_feat) + o2m_anc).detach()
        if not self.learnt_init_query:
            o2m_embed = o2m_feat.detach()
        else:
            q_idx = torch.arange(k_total, device=device) % self.num_queries
            o2m_embed = self.tgt_embed.weight[q_idx].unsqueeze(0).expand(bs, -1, -1)
        return o2m_embed, o2m_refs

    def forward(self, x, batch=None):
        from ultralytics.models.utils.ops import get_cdn_group

        self._o2m_cache = None
        self._iou_cache = None

        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        f3_map = proj_feats[0]
        feats = torch.cat([f.flatten(2).permute(0, 2, 1) for f in proj_feats], dim=1)
        shapes = [[f.shape[2], f.shape[3]] for f in proj_feats]

        bs = len(feats)
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)
        enc_outputs_scores = self.enc_score_head(features)

        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype, device=feats.device).unsqueeze(-1).repeat(
            1, self.num_queries).view(-1)

        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors
        enc_bboxes = refer_bbox.sigmoid()
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        embed = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features

        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embed = embed.detach()

        dn_embed, dn_bbox, dn_mask, dn_meta = get_cdn_group(
            batch, self.nc, self.num_queries, self.denoising_class_embed.weight, self.num_denoising,
            self.label_noise_ratio, self.box_noise_scale, self.training and batch is not None,
        )
        dn_num = 0 if dn_embed is None else dn_embed.shape[1]
        if dn_embed is not None:
            embed = torch.cat((dn_embed.to(embed.dtype), embed), dim=1)
            refer_bbox = torch.cat((dn_bbox.to(refer_bbox.dtype), refer_bbox), dim=1)

        # Append isolated Hybrid O2M query group only during training
        o2m_len = 0
        attn_mask = dn_mask
        if self.training and batch is not None and self.o2m_queries > 0 and self.o2m_weight > 0.0:
            o2m_embed, o2m_refs = self._select_o2m_queries(features, anchors, enc_outputs_scores, shapes)
            o2m_len = o2m_embed.shape[1]
            main_len = dn_num + self.num_queries
            total_len = main_len + o2m_len
            full_mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=embed.device)
            if dn_mask is not None:
                full_mask[:main_len, :main_len] = dn_mask
            # Strictly isolate [DN + O2O] from O2M both ways
            full_mask[:main_len, main_len:] = True
            full_mask[main_len:, :main_len] = True
            attn_mask = full_mask
            embed = torch.cat((embed, o2m_embed.to(embed.dtype)), dim=1)
            refer_bbox = torch.cat((refer_bbox, o2m_refs.to(refer_bbox.dtype)), dim=1)

        dec_bboxes, dec_scores, last_output = self.decoder(
            embed, refer_bbox, feats, shapes, self.dec_bbox_head, self.dec_score_head,
            self.query_pos_head, attn_mask=attn_mask,
        )

        if o2m_len > 0:
            main_len = dn_num + self.num_queries
            o2m_boxes = dec_bboxes[:, :, main_len:, :]
            o2m_scores = dec_scores[:, :, main_len:, :]
            dec_bboxes = dec_bboxes[:, :, :main_len, :]
            dec_scores = dec_scores[:, :, :main_len, :]
            last_output = last_output[:, :main_len, :]
            self._o2m_cache = (o2m_boxes, o2m_scores)

        # Detached Box-Aligned IoU prediction on final O2O predictions only
        o2o_final_boxes = dec_bboxes[-1, :, dn_num:, :]
        o2o_final_feat = last_output[:, dn_num:, :]
        iou_logits = None
        if self.iou_head is not None:
            iou_logits = self.iou_head(o2o_final_boxes, o2o_final_feat, f3_map)
            if self.training and batch is not None and self.iou_weight > 0.0:
                self._iou_cache = (iou_logits, o2o_final_boxes.detach(), dec_scores[-1, :, dn_num:, :].detach())

        raw = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return raw

        p_cls = dec_scores.squeeze(0).sigmoid()
        eff_alpha = self.iou_alpha * self._schedule_scale()
        if iou_logits is not None and eff_alpha > 0.0:
            p_iou = iou_logits.sigmoid().to(p_cls.dtype)
            cal_scores = p_cls.clamp_min(1e-6).pow(1.0 - eff_alpha) * p_iou.clamp_min(1e-6).pow(eff_alpha)
        else:
            cal_scores = p_cls
        y = torch.cat((dec_bboxes.squeeze(0), cal_scores), dim=-1)
        return y if self.export else (y, raw)

    def _compute_o2m_loss(self, targets):
        """Scale-biased one-to-many matching loss on the isolated O2M decoder branch."""
        o2m_boxes, o2m_scores = self._o2m_cache
        ndl, bs, n_q, _ = o2m_scores.shape
        device = o2m_scores.device
        gt_groups = targets['gt_groups']
        if sum(gt_groups) == 0:
            return None

        gt_bboxes_all = targets['bboxes'].float()
        gt_cls_all = targets['cls'].long()
        h_img, w_img = targets.get('image_hw', (640, 640))

        # Build per-image repeated GT sets (small objects get more repeats, capped <= n_q)
        rep_bboxes_list, rep_cls_list, rep_groups = [], [], []
        offset = 0
        for g_cnt in gt_groups:
            if g_cnt == 0:
                rep_groups.append(0)
                offset += g_cnt
                continue
            gb = gt_bboxes_all[offset:offset + g_cnt]
            gc = gt_cls_all[offset:offset + g_cnt]
            offset += g_cnt

            px_area = (gb[:, 2] * w_img) * (gb[:, 3] * h_img)
            small_mask = px_area < (32.0 ** 2)
            repeats = torch.where(small_mask, self.o2m_k, max(2, self.o2m_k // 2))
            total_rep = int(repeats.sum().item())
            if total_rep > n_q:
                # Proportionally scale down so all GTs keep >=1 copy within n_q budget
                scale = n_q / float(total_rep)
                repeats = torch.clamp((repeats.float() * scale).floor().long(), min=1)
                if int(repeats.sum().item()) > n_q:
                    repeats = torch.ones_like(repeats)
                    gb = gb[:n_q]
                    gc = gc[:n_q]
                    repeats = repeats[:n_q]

            rep_b = torch.repeat_interleave(gb, repeats, dim=0)
            rep_c = torch.repeat_interleave(gc, repeats, dim=0)
            rep_bboxes_list.append(rep_b)
            rep_cls_list.append(rep_c)
            rep_groups.append(len(rep_b))

        rep_bboxes = torch.cat(rep_bboxes_list, dim=0)
        rep_cls = torch.cat(rep_cls_list, dim=0)

        # Match each O2M layer using RT-DETR matching cost (class=2, bbox=5, giou=2)
        total_o2m = o2m_scores.new_zeros((), dtype=torch.float32)
        for l_idx in range(ndl):
            pb = o2m_boxes[l_idx].float()
            ps = o2m_scores[l_idx].float()
            ps_prob = ps.detach().sigmoid().reshape(-1, self.nc)[:, rep_cls]
            neg_c = 0.75 * (ps_prob ** 2.0) * (-(1.0 - ps_prob + 1e-8).log())
            pos_c = 0.25 * ((1.0 - ps_prob) ** 2.0) * (-(ps_prob + 1e-8).log())
            c_cls = pos_c - neg_c
            pb_flat = pb.detach().reshape(-1, 4)
            c_l1 = (pb_flat.unsqueeze(1) - rep_bboxes.unsqueeze(0)).abs().sum(-1)
            c_giou = 1.0 - bbox_iou(pb_flat.unsqueeze(1), rep_bboxes.unsqueeze(0), xywh=True, GIoU=True).squeeze(-1)
            cost = (2.0 * c_cls + 5.0 * c_l1 + 2.0 * c_giou)
            cost[cost.isnan() | cost.isinf()] = 0.0
            cost = cost.view(bs, n_q, -1).cpu()

            gt_offsets = torch.as_tensor([0, *rep_groups[:-1]]).cumsum(0)
            b_idx_list, q_idx_list, g_idx_list = [], [], []
            for bi, c_mat in enumerate(cost.split(rep_groups, dim=-1)):
                if rep_groups[bi] == 0:
                    continue
                qi, gi = linear_sum_assignment(c_mat[bi])
                b_idx_list.append(torch.full((len(qi),), bi, dtype=torch.long))
                q_idx_list.append(torch.as_tensor(qi, dtype=torch.long))
                g_idx_list.append(torch.as_tensor(gi, dtype=torch.long) + gt_offsets[bi])

            if not q_idx_list:
                continue
            b_idx = torch.cat(b_idx_list).to(device)
            q_idx = torch.cat(q_idx_list).to(device)
            g_idx = torch.cat(g_idx_list).to(device)

            matched_pb = pb[b_idx, q_idx]
            matched_gb = rep_bboxes[g_idx]
            num_pos = max(len(matched_gb), 1)

            one_hot = torch.zeros((bs, n_q, self.nc + 1), dtype=torch.int64, device=device)
            targets_cls = torch.full((bs, n_q), self.nc, dtype=torch.int64, device=device)
            targets_cls[b_idx, q_idx] = rep_cls[g_idx]
            one_hot.scatter_(2, targets_cls.unsqueeze(-1), 1)
            one_hot = one_hot[..., :-1]

            gt_scores = torch.zeros((bs, n_q), dtype=torch.float32, device=device)
            gt_scores[b_idx, q_idx] = bbox_iou(matched_pb.detach(), matched_gb, xywh=True).squeeze(-1).clamp(0.0, 1.0)
            gt_scores_oh = gt_scores.unsqueeze(-1) * one_hot

            cls_l = VarifocalLoss.forward(ps, gt_scores_oh, one_hot) / (num_pos / n_q)
            l1_l = 5.0 * F.l1_loss(matched_pb, matched_gb, reduction='sum') / num_pos
            giou_l = 2.0 * (1.0 - bbox_iou(matched_pb, matched_gb, xywh=True, GIoU=True)).sum() / num_pos
            total_o2m = total_o2m + (cls_l + l1_l + giou_l)

        return (self.o2m_weight * self._schedule_scale() / ndl) * total_o2m

    def _compute_iou_head_loss(self, targets):
        """Supervise DetachedBoxAlignedIoUHead on all final O2O queries."""
        iou_logits, pred_boxes, pred_scores = self._iou_cache
        bs, nq, _ = iou_logits.shape
        device = iou_logits.device
        gt_groups = targets['gt_groups']
        gt_bboxes_all = targets['bboxes'].float()
        gt_cls_all = targets['cls'].long()

        logits_f = iou_logits.float().squeeze(-1)  # [B, Q]
        pred_cls = pred_scores.argmax(dim=-1)      # [B, Q]

        iou_targets = logits_f.new_zeros((bs, nq))
        weights = logits_f.new_full((bs, nq), 0.25)

        offset = 0
        num_fg = 0
        for bi, g_cnt in enumerate(gt_groups):
            if g_cnt == 0:
                continue
            gb = gt_bboxes_all[offset:offset + g_cnt]
            gc = gt_cls_all[offset:offset + g_cnt]
            offset += g_cnt

            pb = pred_boxes[bi].float()
            iou_mat = _pairwise_cxcywh_iou(pb, gb)  # [Q, g_cnt]
            same_cls = (pred_cls[bi].unsqueeze(1) == gc.unsqueeze(0))
            same_iou = torch.where(same_cls, iou_mat, iou_mat.new_zeros(()))
            max_same_iou = same_iou.amax(dim=-1).clamp(0.0, 1.0)
            max_any_iou = iou_mat.amax(dim=-1).clamp(0.0, 1.0)

            # High weight on foreground boxes (same-class IoU > 0.1), downweight background
            fg_mask = max_any_iou > 0.1
            iou_targets[bi] = torch.where(fg_mask, max_same_iou, logits_f.new_zeros(()))
            weights[bi] = torch.where(fg_mask, logits_f.new_full((nq,), 1.0), logits_f.new_full((nq,), 0.25))
            num_fg += int(fg_mask.sum().item())

        bce = F.binary_cross_entropy_with_logits(logits_f, iou_targets, reduction='none')
        denom = max(num_fg, 1)
        return self.iou_weight * (bce * weights).sum() / denom

    def compute_ground_loss(self, targets, batch=None):
        """Combine Hybrid O2M and Detached IoU Head auxiliary training losses."""
        o2m_ready = self._o2m_cache is not None
        iou_ready = self._iou_cache is not None
        total = None

        if o2m_ready:
            o2m_loss = self._compute_o2m_loss(targets)
            self._o2m_cache = None
            if o2m_loss is not None:
                total = o2m_loss

        if iou_ready:
            iou_loss = self._compute_iou_head_loss(targets)
            self._iou_cache = None
            if iou_loss is not None:
                total = iou_loss if total is None else total + iou_loss

        return total

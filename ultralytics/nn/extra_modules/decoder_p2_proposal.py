"""Independent stride-4 proposals for the RT-DETR matching queries."""

import torch
from torch import nn
from torch.nn import functional as F

from ultralytics.utils.loss import VarifocalLoss
from ultralytics.utils.metrics import bbox_iou

from ..modules.conv import Conv
from ..modules.head import RTDETRDecoder
from ..modules.utils import bias_init_with_prob, inverse_sigmoid

__all__ = ('RTDETRDecoderP2Proposal',)


class RTDETRDecoderP2Proposal(RTDETRDecoder):
    """Decode the original F3–F5 queries together with separately ranked P2 proposals."""

    def __init__(self, nc, ch, hd, nq, ndp, nh, ndl, p2_queries=32, p2_dim=96):
        if len(ch) != 4 or p2_queries < 1:
            raise ValueError('P2 proposals require [F3, F4, F5, P2] and a positive P2 query count')
        super().__init__(nc, ch[:3], hd, nq, ndp, nh, ndl)
        self.p2_queries = p2_queries
        self.p2_proj = Conv(ch[3], p2_dim, 1)
        self.p2_semantics = Conv(ch[0], p2_dim, 1)
        self.p2_mix = Conv(p2_dim, p2_dim, 3, g=p2_dim)
        self.p2_score = nn.Conv2d(p2_dim, nc, 1)
        self.p2_box = nn.Conv2d(p2_dim, 4, 1)
        self.p2_embed = nn.Sequential(nn.Linear(p2_dim, hd), nn.LayerNorm(hd))
        self._dense_proposals = None
        nn.init.constant_(self.p2_score.bias, bias_init_with_prob(0.01))
        nn.init.zeros_(self.p2_box.weight)
        nn.init.zeros_(self.p2_box.bias)

    def _make_p2_proposals(self, p2, f3):
        height, width = p2.shape[-2:]
        semantic = F.interpolate(self.p2_semantics(f3), size=(height, width), mode='nearest')
        fine = self.p2_mix(self.p2_proj(p2) + semantic)
        scores = self.p2_score(fine).flatten(2).transpose(1, 2)
        offsets = self.p2_box(fine).flatten(2).transpose(1, 2).float()

        device = fine.device
        y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
        grid = torch.stack((x, y), dim=-1).reshape(1, -1, 2) + 0.5
        spatial_scale = offsets.new_tensor((width, height))
        centers = (grid + 2 * offsets[..., :2].tanh()) / spatial_scale
        size_prior = offsets.new_tensor((2.0, 3.0)) / spatial_scale
        refs = torch.cat((inverse_sigmoid(centers), offsets[..., 2:] + inverse_sigmoid(size_prior)), dim=-1)

        count = min(self.p2_queries, height * width)
        indices = scores.amax(dim=-1).topk(count, dim=1).indices
        batch_ids = torch.arange(p2.shape[0], device=device)[:, None]
        content = self.p2_embed(fine.flatten(2).transpose(1, 2)[batch_ids, indices])
        return content, refs, scores, (batch_ids, indices)

    def forward(self, x, batch=None):
        """Keep P2 proposal selection independent and decode both sources jointly."""
        from ultralytics.models.utils.ops import get_cdn_group

        self._dense_proposals = None
        feats, shapes = self._get_encoder_input(x[:3])
        embed, refs, enc_boxes, enc_scores = self._get_decoder_input(feats, shapes)
        fine_embed, fine_refs, fine_scores, selected = self._make_p2_proposals(x[3], x[0])
        fine_boxes = fine_refs.sigmoid()
        fine_selected = fine_refs[selected].to(refs)
        enc_boxes = torch.cat((enc_boxes, fine_boxes[selected].to(enc_boxes)), dim=1)
        enc_scores = torch.cat((enc_scores, fine_scores[selected]), dim=1)
        if self.training and batch is not None:
            self._dense_proposals = (fine_scores, fine_boxes, x[3].shape[-2:])
            fine_selected = fine_selected.detach()
        embed = torch.cat((embed, fine_embed), dim=1)
        refs = torch.cat((refs, fine_selected), dim=1)

        dn_embed, dn_boxes, mask, dn_meta = get_cdn_group(
            batch, self.nc, embed.shape[1], self.denoising_class_embed.weight, self.num_denoising,
            self.label_noise_ratio, self.box_noise_scale, self.training and batch is not None)
        if dn_embed is not None:
            embed = torch.cat((dn_embed, embed), dim=1)
            refs = torch.cat((dn_boxes, refs), dim=1)

        boxes, scores = self.decoder(embed, refs, feats, shapes, self.dec_bbox_head, self.dec_score_head,
                                     self.query_pos_head, attn_mask=mask)
        raw = boxes, scores, enc_boxes, enc_scores, dn_meta
        if self.training:
            return raw
        predictions = torch.cat((boxes.squeeze(0), scores.squeeze(0).sigmoid()), dim=-1)
        top = predictions[..., 4:].amax(dim=-1).topk(self.num_queries, dim=1).indices
        predictions = predictions.gather(1, top.unsqueeze(-1).expand(-1, -1, predictions.shape[-1]))
        return predictions if self.export else (predictions, raw)

    @staticmethod
    def _proposal_matches(boxes, image_ids, batch_size, height, width):
        """Assign up to four nearest P2 centers per GT, with one owner per cell."""
        limits = image_ids.new_tensor((width, height))
        centers = boxes[:, :2] * limits - 0.5
        offsets = image_ids.new_tensor(((0, 0), (1, 0), (0, 1), (1, 1)))
        cells = (centers.floor().long().unsqueeze(1) + offsets).clamp_min(0).minimum(limits - 1)
        distances = (cells - centers.unsqueeze(1)).square().sum(dim=-1).flatten()
        keys = ((image_ids[:, None] * height + cells[..., 1]) * width + cells[..., 0]).flatten()

        total_sites = batch_size * height * width
        best_distance = distances.new_full((total_sites,), float('inf'))
        best_distance.scatter_reduce_(0, keys, distances, reduce='amin')
        owner_ids = torch.arange(len(boxes), device=keys.device).repeat_interleave(len(offsets))
        unmatched = len(boxes)
        candidates = torch.where(distances == best_distance[keys], owner_ids, unmatched)
        owners = keys.new_full((total_sites,), unmatched)
        owners.scatter_reduce_(0, keys, candidates, reduce='amin')
        positive_sites = (owners != unmatched).nonzero().flatten()
        return positive_sites, owners[positive_sites]

    def compute_ground_loss(self, targets, batch=None):
        """Supervise the P2 candidate pool, including cells outside the selected queries."""
        proposals, self._dense_proposals = self._dense_proposals, None
        if proposals is None:
            return None
        scores, boxes, (height, width) = proposals
        scores = scores.float().contiguous()
        gt = targets['bboxes'].float()
        sizes = gt[:, 2:] * gt.new_tensor((4 * width, 4 * height))
        tiny = (sizes.prod(dim=1) < 32**2) & (sizes.amin(dim=1) < 8)
        gt = gt[tiny]
        classes = targets['cls'][tiny]
        sites, owners = self._proposal_matches(gt, targets['batch_idx'][tiny], len(scores), height, width)
        pred_boxes, gt_boxes = boxes.flatten(0, 1)[sites], gt[owners]
        quality = bbox_iou(pred_boxes.detach(), gt_boxes, xywh=True).flatten().clamp_min(0.05)
        soft_scores = torch.zeros_like(scores)
        soft_scores.view(-1, self.nc)[sites, classes[owners]] = quality
        denominator = max(len(sites), 1)
        cls_loss = VarifocalLoss.forward(scores, soft_scores, (soft_scores > 0).float()) * scores.shape[1]
        l1 = F.l1_loss(pred_boxes, gt_boxes, reduction='sum')
        giou = (1 - bbox_iou(pred_boxes, gt_boxes, xywh=True, GIoU=True)).sum()
        return 0.25 * (cls_loss + 5 * l1 + 2 * giou) / denominator

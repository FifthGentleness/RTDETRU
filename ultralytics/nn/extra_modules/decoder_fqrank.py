"""Final-layer localization quality teaches encoder candidate ranking.
Inspired by D-FINE's localization self-distillation, not an FDR/GO-LSD reproduction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.head import RTDETRDecoder
from ultralytics.models.utils.ops import HungarianMatcher
from ultralytics.utils.metrics import bbox_iou

__all__ = ['RTDETRDecoderFQRank']


class RTDETRDecoderFQRank(RTDETRDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality_head = nn.Sequential(nn.Linear(self.hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.quality_head[-1].weight)
        nn.init.zeros_(self.quality_head[-1].bias)
        self.register_buffer('rank_alpha', torch.tensor(0.5))
        self.alpha_max = 0.5
        self.ramp_fraction = 0.2
        self.explore_max = 0.1
        self.explore_end = 0.5
        self.explore_ratio = 0.1
        self.innovation_schedule = True
        self.quality_weight = 0.1
        self.teacher = 'final'  # ablation: 'encoder'
        self.detach_quality_features = False
        self._cache = None
        self._cache_enabled = False
        self._quality_matcher = HungarianMatcher(cost_gain={'class': 2, 'bbox': 5, 'giou': 2})

    def configure(self, params):
        allowed = {'alpha_max', 'ramp_fraction', 'explore_max', 'explore_end',
                   'quality_weight', 'teacher', 'detach_quality_features'}
        if set(params) - allowed:
            raise ValueError(f'Unknown FQRank parameters: {set(params) - allowed}')
        for key, value in params.items():
            setattr(self, key, value)
        if (self.alpha_max < 0 or self.ramp_fraction <= 0 or
                not 0 <= self.explore_max < 1 or not 0 <= self.explore_end <= 1 or
                self.quality_weight < 0 or self.teacher not in ('final', 'encoder')):
            raise ValueError('Invalid FQRank parameters')
        self.rank_alpha.fill_(self.alpha_max)
        self.explore_ratio = self.explore_max

    def set_epoch(self, epoch, total_epochs):
        progress = epoch / max(total_epochs - 1, 1)
        self.rank_alpha.fill_(self.alpha_max * min(progress / self.ramp_fraction, 1.0))
        self.explore_ratio = self.explore_max if progress < self.explore_end else 0.0

    def forward(self, x, batch=None):
        self._cache = None
        self._cache_enabled = self.training and batch is not None
        try:
            output = super().forward(x, batch)
            if self._cache is not None:
                boxes, scores, enc_boxes, enc_scores, dn = output
                n_dn = dn['dn_num_split'][0] if dn is not None else 0
                self._cache += (boxes[-1, :, n_dn:].detach(), scores[-1, :, n_dn:].detach(),
                                enc_boxes.detach(), enc_scores.detach())
            return output
        finally:
            self._cache_enabled = False

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        b = feats.shape[0]
        anchors, valid = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid * feats)
        scores = self.enc_score_head(features)
        quality = self.quality_head(features.detach() if self.detach_quality_features else features).squeeze(-1)
        rank = scores.sigmoid().amax(-1).clamp_min(1e-12).log() + self.rank_alpha * F.logsigmoid(quality)
        rank = rank.masked_fill(~valid.squeeze(-1), float('-inf'))
        n = self.num_queries
        indices = rank.topk(n, dim=1).indices
        if self._cache_enabled and self.explore_ratio > 0:
            count = min(round(n * self.explore_ratio), max(int(valid.sum().item()) - n, 0))
            if count:
                # Uniform sample from valid non-top-N candidates, without replacement.
                noise = torch.rand_like(rank).masked_fill(~valid.squeeze(-1), -1)
                noise.scatter_(1, indices, -1)
                explore = noise.topk(count, dim=1).indices
                indices = torch.cat((indices[:, :n-count], explore), 1)
        batch_indices = torch.arange(b, device=feats.device)[:, None]
        selected = features[batch_indices, indices]
        reference = self.enc_bbox_head(selected) + anchors[0][indices]
        enc_boxes, enc_scores = reference.sigmoid(), scores[batch_indices, indices]
        if self._cache_enabled:
            self._cache = (quality[batch_indices, indices],)
        embeddings = self.tgt_embed.weight[None].expand(b, -1, -1) if self.learnt_init_query else selected
        if self.training:
            reference = reference.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_bbox is not None:
            reference = torch.cat((dn_bbox, reference), 1)
        if dn_embed is not None:
            embeddings = torch.cat((dn_embed, embeddings), 1)
        return embeddings, reference, enc_boxes, enc_scores

    def compute_ground_loss(self, targets, batch=None):
        cache, self._cache = self._cache, None  # release graph before EMA/deepcopy
        if not self.training or cache is None:
            return None
        logits, final_boxes, final_scores, enc_boxes, enc_scores = cache
        boxes, scores = (enc_boxes, enc_scores) if self.teacher == 'encoder' else (final_boxes, final_scores)
        matches = self._quality_matcher(boxes.contiguous(), scores.contiguous(), targets['bboxes'], targets['cls'], targets['gt_groups'])
        labels = torch.zeros_like(logits)
        positive = torch.zeros_like(logits, dtype=torch.bool)
        for i, (src, dst) in enumerate(matches):
            if src.numel():
                labels[i, src] = bbox_iou(boxes[i, src], targets['bboxes'][dst], xywh=True).flatten()
                positive[i, src] = True
        losses = F.binary_cross_entropy_with_logits(logits.float(), labels.float(), reduction='none')
        pos = losses[positive].mean() if positive.any() else logits.sum() * 0
        neg = losses[~positive].mean() if (~positive).any() else logits.sum() * 0
        return self.quality_weight * (pos + 0.25 * neg)

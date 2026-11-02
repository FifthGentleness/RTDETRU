"""Centerness-guided encoder query selection for the RT-DETR decoder.

Adapted to this repository's RT-DETR query path from PF-RPN (CVPR 2026):
https://openaccess.thecvf.com/content/CVPR2026/html/Tang_Prompt-Free_Universal_Region_Proposal_Network_CVPR_2026_paper.html
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.head import RTDETRDecoder

__all__ = ('RTDETRDecoderCGQS',)


class RTDETRDecoderCGQS(RTDETRDecoder):
    """Select decoder queries using class confidence and predicted centerness.

    The selected encoder boxes, denoising queries, and decoder layers retain the
    standard RT-DETR behavior. Centerness is supervised on all encoder tokens,
    including tokens that do not enter the decoder.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = max(self.hidden_dim // 4, 32)
        self.enc_centerness_head = nn.Sequential(
            nn.Linear(self.hidden_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        # A constant 0.5 centerness leaves the original class-score ranking
        # unchanged at initialization.
        nn.init.zeros_(self.enc_centerness_head[-1].weight)
        nn.init.zeros_(self.enc_centerness_head[-1].bias)
        self._cgqs_logits = None
        self._cgqs_shapes = None
        self._cgqs_cache_enabled = False

    def forward(self, x, batch=None):
        # DetectionModel performs a dummy forward with batch=None at build time.
        # Caching that graph would make the model unsafe to deepcopy for EMA.
        self._cgqs_cache_enabled = self.training and batch is not None
        try:
            return super().forward(x, batch)
        finally:
            self._cgqs_cache_enabled = False

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        bs = feats.shape[0]
        anchors, valid_mask = self._generate_anchors(
            shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)
        enc_outputs_scores = self.enc_score_head(features)
        center_logits = self.enc_centerness_head(features).squeeze(-1)

        # Joint confidence is computed before top-k, so previously unselected
        # encoder tokens can now enter the decoder. Invalid anchors cannot win.
        class_confidence = enc_outputs_scores.max(-1).values.float().sigmoid()
        selection_score = class_confidence * center_logits.float().sigmoid()
        selection_score = selection_score.masked_fill(~valid_mask.squeeze(-1), float('-inf'))
        topk_ind = selection_score.topk(self.num_queries, dim=1).indices
        top_k_features = features.gather(
            1, topk_ind.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
        top_k_anchors = anchors.expand(bs, -1, -1).gather(
            1, topk_ind.unsqueeze(-1).expand(-1, -1, 4))
        enc_scores = enc_outputs_scores.gather(
            1, topk_ind.unsqueeze(-1).expand(-1, -1, self.nc))

        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors
        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)

        embeddings = (self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
                      if self.learnt_init_query else top_k_features)
        if self.training:
            if self._cgqs_cache_enabled:
                self._cgqs_logits = center_logits
                self._cgqs_shapes = tuple(tuple(s) for s in shapes)
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        else:
            self._cgqs_logits = None
            self._cgqs_shapes = None
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)
        return embeddings, refer_bbox, enc_bboxes, enc_scores

    def compute_ground_loss(self, targets, batch=None):
        """FCOS-style centerness targets on encoder grids, with tiny-box fallback.

        Called by the existing auxiliary-ground-loss collector in tasks.py.
        Each GT is guaranteed a positive on the finest encoder grid even when
        the box is smaller than one grid cell, as is common in VisDrone.
        """
        if not self.training or self._cgqs_logits is None:
            return None
        logits, shapes = self._cgqs_logits, self._cgqs_shapes
        self._cgqs_logits = self._cgqs_shapes = None
        bs, n = logits.shape
        device = logits.device
        centers = []
        for h, w in shapes:
            ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
            xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
            gy, gx = torch.meshgrid(ys, xs, indexing='ij')
            centers.append(torch.stack((gx.flatten(), gy.flatten()), dim=-1))
        centers = torch.cat(centers)
        assert centers.shape[0] == n

        labels = torch.zeros((bs, n), device=device, dtype=torch.float32)
        boxes, batch_idx = targets['bboxes'], targets['batch_idx']
        for b in range(bs):
            gt = boxes[batch_idx == b].float()
            gt = gt[torch.isfinite(gt).all(-1) & (gt[:, 2:] > 0).all(-1)]
            if gt.numel() == 0:
                continue
            # Bound the temporary GT-by-token tensors on small GPUs.
            for chunk in gt.split(16):
                x, y, w, h = chunk.unbind(-1)
                left = centers[:, 0].unsqueeze(0) - (x - w / 2).unsqueeze(1)
                right = (x + w / 2).unsqueeze(1) - centers[:, 0].unsqueeze(0)
                top = centers[:, 1].unsqueeze(0) - (y - h / 2).unsqueeze(1)
                bottom = (y + h / 2).unsqueeze(1) - centers[:, 1].unsqueeze(0)
                inside = (left > 0) & (right > 0) & (top > 0) & (bottom > 0)
                center_x = left.clamp_min(0).minimum(right.clamp_min(0)) / (
                    left.maximum(right).clamp_min(1e-6))
                center_y = top.clamp_min(0).minimum(bottom.clamp_min(0)) / (
                    top.maximum(bottom).clamp_min(1e-6))
                centerness = (center_x * center_y).sqrt() * inside
                labels[b] = torch.maximum(labels[b], centerness.amax(dim=0))

                # When no grid point is inside a tiny GT, supervise its closest
                # P3 point instead of silently dropping the object.
                missing = ~inside.any(dim=1)
                if missing.any():
                    fine_h, fine_w = shapes[0]
                    nearest_x = (x[missing] * fine_w - 0.5).round().long().clamp(0, fine_w - 1)
                    nearest_y = (y[missing] * fine_h - 0.5).round().long().clamp(0, fine_h - 1)
                    labels[b, nearest_y * fine_w + nearest_x] = 1.0

        prediction = logits.float().sigmoid()
        positive = labels > 0
        negative = ~positive
        loss_pos = (prediction[positive] - labels[positive]).abs().mean() if positive.any() else prediction.sum() * 0
        loss_neg = prediction[negative].mean() if negative.any() else prediction.sum() * 0
        return loss_pos + 0.25 * loss_neg

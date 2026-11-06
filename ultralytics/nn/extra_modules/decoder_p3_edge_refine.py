"""Bounded four-edge refinement inspired by ArgMatch (ICCV 2025) local correlation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.head import RTDETRDecoder
from ..modules.transformer import DeformableTransformerDecoder

__all__ = ['RTDETRDecoderP3EdgeRefine']


class P3EdgeRefiner(nn.Module):
    """Read inside/outside P3 features around each predicted edge."""

    innovation_schedule = True

    def __init__(self, hidden_dim, num_queries, proj_dim=16, stride=8, min_side_px=4.0, max_size_px=48.0,
                 max_shift_px=4.0, temperature=0.25, max_strength=1.0, start_epoch=0, full_epoch=1):
        super().__init__()
        if (proj_dim < 1 or stride < 1 or not 0 < min_side_px < max_size_px or max_shift_px <= 0
                or temperature <= 0 or not 0 <= max_strength <= 1 or not 0 <= start_epoch < full_epoch):
            raise ValueError('Invalid P3 edge refinement parameters')
        self.p3_proj = nn.Conv2d(hidden_dim, proj_dim, 1, bias=False)
        self.query_proj = nn.Linear(hidden_dim, proj_dim, bias=False)
        self.num_queries, self.stride = num_queries, stride
        self.min_side_px, self.max_size_px = min_side_px, max_size_px
        self.max_shift_px, self.temperature = max_shift_px, temperature
        self.max_strength, self.strength = max_strength, max_strength
        self.start_epoch, self.full_epoch = start_epoch, full_epoch
        # Side order is left, top, right, bottom. Use integer buffers to preserve
        # the symmetric prior and stencil when the checkpoint is cast to FP16.
        self.register_buffer('normals', torch.tensor([[-1, 0], [0, -1], [1, 0], [0, 1]]), persistent=False)
        self.register_buffer('steps', torch.arange(-2, 3), persistent=False)
        self.register_buffer('prior', torch.tensor([1, 2, 4, 2, 1]), persistent=False)

    def set_epoch(self, epoch, total_epochs=0):
        progress = (epoch + 1 - self.start_epoch) / (self.full_epoch - self.start_epoch)
        self.strength = self.max_strength * min(1.0, max(0.0, progress))

    @staticmethod
    def p3_from_decoder_inputs(feats, shapes):
        """Restore the first (stride-8) level from the projected decoder inputs."""
        h, w = shapes[0]
        return feats[:, :h*w].transpose(1, 2).reshape(feats.shape[0], feats.shape[-1], h, w)

    def forward(self, boxes, query, feats, shapes):
        if self.strength == 0:
            return boxes
        # RT-DETR prepends denoising queries; only the matching suffix is changed.
        dn = boxes.shape[1] - self.num_queries
        p3 = self.p3_from_decoder_inputs(feats, shapes)
        refined = self.refine(boxes[:, dn:], query[:, dn:], p3)
        return torch.cat((boxes[:, :dn], refined), dim=1) if dn else refined

    def refine(self, boxes, query, p3):
        """Refine normalized cxcywh; also usable with cached features in a frozen probe."""
        if self.strength == 0:
            return boxes
        b, nq, _ = boxes.shape
        h, w = p3.shape[-2:]
        feature_wh = p3.new_tensor([w, h], dtype=torch.float32)
        image_wh = feature_wh * self.stride
        box_scale = image_wh.repeat(2)

        # Detach only the stencil coordinates; the original boxes retain their
        # regression gradient through the additive residual below.
        center, size = (boxes.detach().float() * box_scale).split(2, dim=-1)
        normal_size = size[..., [0, 1, 0, 1]]
        tangent_size = size[..., [1, 0, 1, 0]]
        normal = self.normals.float()
        tangent = normal.flip(-1).abs()
        edge = center.unsqueeze(-2) + 0.5 * normal_size.unsqueeze(-1) * normal
        radius = (0.25 * normal_size).clamp(max=self.max_shift_px)
        gap = (0.125 * normal_size).clamp(max=2.0)
        normal_shift = 0.5 * radius[..., None, None] * self.steps[:, None] + gap[..., None, None] * self.steps[::4] * 0.5
        tangent_shift = 0.25 * tangent_size[..., None] * self.steps[1:4]
        points = (edge[..., None, None, None, :]
                  + normal_shift[:, :, :, :, None, :, None] * normal[:, None, None, None, :]
                  + tangent_shift[:, :, :, None, :, None, None] * tangent[:, None, None, None, :])
        grid = points * (2.0 / image_wh) - 1.0
        valid = (grid.abs() <= 1.0 - feature_wh.reciprocal()).flatten(2).all(dim=-1)
        valid = valid & (size.amin(dim=-1) >= self.min_side_px) & (size.prod(dim=-1) < self.max_size_px ** 2)
        # Sample only from the valid pixel-center region. Replace the entire
        # stencil for rejected boxes so its values cannot affect the residual.
        grid = torch.where(valid[..., None, None, None, None, None], grid, 0.0)

        features = self.p3_proj(p3).float()
        readout = F.grid_sample(features, grid.reshape(b, nq, 120, 2), align_corners=False)
        readout = readout.permute(0, 2, 3, 1).reshape(b, nq, 4, 5, 3, 2, -1)
        inside, outside = F.normalize(readout, dim=-1, eps=1e-6).unbind(-2)
        query = F.normalize(self.query_proj(query).float(), dim=-1, eps=1e-6)[:, :, None, None, None, :]
        contrast = ((inside - outside) * query).sum(dim=-1)
        boundary = 1.0 - (inside * outside).sum(dim=-1)
        logits = (contrast + 0.25 * boundary).mean(dim=-1) / self.temperature + self.prior.float().log()
        p = logits.softmax(dim=-1)
        # Compute by symmetric pairs so a flat stencil gives zero displacement.
        offset = radius * ((p[..., 4] - p[..., 0]) + 0.5 * (p[..., 3] - p[..., 1]))
        left, top, right, bottom = offset.unbind(-1)
        delta = torch.stack((0.5 * (right-left), 0.5 * (bottom-top), right+left, bottom+top), dim=-1)
        delta = torch.where(valid.unsqueeze(-1), delta * (self.strength / box_scale), 0.0)
        return boxes + delta.to(boxes.dtype)


class _P3EdgeDecoder(DeformableTransformerDecoder):
    """Use the unmodified decoder and read the final hidden state locally."""

    def __init__(self, original, refiner):
        nn.Module.__init__(self)
        self.layers = original.layers
        self.num_layers, self.hidden_dim, self.eval_idx = original.num_layers, original.hidden_dim, original.eval_idx
        self.final_box_refiner = refiner

    def forward(self, embed, refer_bbox, feats, shapes, bbox_head, score_head, pos_mlp,
                attn_mask=None, padding_mask=None):
        if self.final_box_refiner.strength == 0:
            return super().forward(embed, refer_bbox, feats, shapes, bbox_head, score_head, pos_mlp,
                                   attn_mask=attn_mask, padding_mask=padding_mask)

        # The baseline applies the last bbox head to the final query before
        # producing b0. Capture that query without hooks or module state.
        final_query = None
        heads = list(bbox_head)
        final_head = heads[-1]

        def capture_final_query(query):
            nonlocal final_query
            final_query = query
            return final_head(query)

        heads[-1] = capture_final_query
        boxes, scores = super().forward(embed, refer_bbox, feats, shapes, heads, score_head, pos_mlp,
                                        attn_mask=attn_mask, padding_mask=padding_mask)
        refined = self.final_box_refiner(boxes[-1], final_query, feats, shapes)
        return torch.cat((boxes[:-1], refined.unsqueeze(0))), scores


class RTDETRDecoderP3EdgeRefine(RTDETRDecoder):
    """Add the P3 readout after the final matching boxes, preserving the decoder state layout."""

    def __init__(self, nc=80, ch=(512, 1024, 2048), hd=256, nq=300, ndp=4, nh=8, ndl=6,
                 refine_params=None, **kwargs):
        super().__init__(nc, ch, hd, nq, ndp, nh, ndl, **kwargs)
        if self.decoder.eval_idx != ndl - 1:
            raise ValueError('P3 edge refinement requires evaluation at the final decoder layer')
        self.decoder = _P3EdgeDecoder(self.decoder, P3EdgeRefiner(hd, nq, **(refine_params or {})))

"""Use P3 readouts before box regression in the last one or two RT-DETR layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.head import RTDETRDecoder
from ..modules.transformer import DeformableTransformerDecoder
from ..modules.utils import inverse_sigmoid

__all__ = ['RTDETRDecoderP3EdgeRead']


class P3EdgeRead(nn.Module):
    """A shared P3 projection, four-edge readout, and optional layer-2 context readout."""

    def __init__(self, hidden_dim, use_context=False, proj_dim=32, stride=8, min_side_px=4.0,
                 max_size_px=48.0, max_radius_px=4.0, max_offset_px=2.0, max_strength=0.25,
                 max_context_strength=0.15):
        super().__init__()
        if (proj_dim < 1 or stride < 1 or not 0 < min_side_px < max_size_px
                or not 0 < max_offset_px <= max_radius_px
                or not 0 < max_strength <= 1 or not 0 < max_context_strength <= 1):
            raise ValueError('Invalid P3 edge read parameters')

        self.stride = stride
        self.min_side_px, self.max_size_px = min_side_px, max_size_px
        self.max_radius_px, self.max_offset_px = max_radius_px, max_offset_px
        self.use_context = use_context
        # The fuse layers below start at zero, so a fixed gain still starts with no readout residual.
        self.strength = max_strength
        self.context_strength = max_context_strength if use_context else 0.0

        self.p3_proj = nn.Conv2d(hidden_dim, proj_dim, 1, bias=False)
        self.edge_query = nn.Linear(hidden_dim, proj_dim, bias=False)
        self.edge_offset = nn.Sequential(nn.Linear(5 * proj_dim + 4, 64), nn.SiLU(), nn.Linear(64, 1))
        self.edge_fuse = nn.Linear(4 * (3 * proj_dim + 1), hidden_dim, bias=False)
        nn.init.zeros_(self.edge_offset[-1].weight)
        nn.init.zeros_(self.edge_offset[-1].bias)
        nn.init.zeros_(self.edge_fuse.weight)
        self.register_buffer('normals', torch.tensor([[-1, 0], [0, -1], [1, 0], [0, 1]]), persistent=False)
        self.register_buffer('edge_ids', torch.eye(4), persistent=False)

        if use_context:
            self.context_query = nn.Linear(hidden_dim, proj_dim, bias=False)
            self.context_fuse = nn.Linear(3 * proj_dim + 2, hidden_dim, bias=False)
            nn.init.zeros_(self.context_fuse.weight)
            grid = torch.tensor([(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)])
            self.register_buffer('context_grid', grid, persistent=False)
            self.register_buffer('context_prior', 4 - 2 * grid.abs().amax(dim=-1) - grid.abs().prod(dim=-1),
                                 persistent=False)

    def project_p3(self, feats, shapes):
        """Restore and project the first level of the decoder's existing memory once."""
        h, w = shapes[0]
        p3 = feats[:, :h * w].transpose(1, 2).reshape(feats.shape[0], feats.shape[-1], h, w)
        return self.p3_proj(p3).float()

    def _geometry(self, boxes, p3):
        h, w = p3.shape[-2:]
        feature_wh = p3.new_tensor([w, h])
        image_wh = feature_wh * self.stride
        center, size = (boxes.float() * image_wh.repeat(2)).split(2, dim=-1)
        eligible = (size.amin(dim=-1) >= self.min_side_px) & (size.prod(dim=-1) < self.max_size_px ** 2)
        return center, size, eligible, 2.0 / image_wh, 1.0 - feature_wh.reciprocal()

    @staticmethod
    def _sample(p3, grid):
        b, n = grid.shape[:2]
        values = F.grid_sample(p3, grid.reshape(b, n, -1, 2), align_corners=False)
        return values.permute(0, 2, 3, 1).reshape(*grid.shape[:-1], p3.shape[1])

    def _edge_grid(self, edge, normal_shift, tangent_shift, grid_scale):
        normal = self.normals.float()
        tangent = normal.flip(-1).abs()
        points = (edge[..., None, None, :] + normal_shift[..., :, None, None] * normal[:, None, None, :]
                  + tangent_shift[..., None, :, None] * tangent[:, None, None, :])
        return points * grid_scale - 1.0

    def _read_edges(self, p3, grid, limit, eligible):
        # Reject each invalid edge independently; an image-border edge cannot pollute the other three.
        valid = (grid.abs() <= limit).flatten(-3).all(dim=-1) & eligible
        grid = torch.where(valid[..., None, None, None], grid, 0.0)
        return self._sample(p3, grid).mean(dim=-2), valid  # mean over the two tangential positions

    def read_edges(self, query, boxes, p3):
        """Read 32 scout points, move the read locations, then read 16 fine points."""
        center, size, eligible, grid_scale, limit = self._geometry(boxes, p3)
        normal_size = size[..., [0, 1, 0, 1]]
        tangent_size = size[..., [1, 0, 1, 0]]
        edge = center.unsqueeze(-2) + 0.5 * normal_size.unsqueeze(-1) * self.normals
        radius = (0.25 * normal_size).clamp(min=0, max=self.max_radius_px)
        tangent_shift = tangent_size.unsqueeze(-1) * p3.new_tensor([-0.25, 0.25])

        scout_steps = radius.unsqueeze(-1) * p3.new_tensor([-1.0, -0.5, 0.5, 1.0])
        scout_grid = self._edge_grid(edge, scout_steps, tangent_shift, grid_scale)
        scout, valid = self._read_edges(p3, scout_grid, limit, eligible.unsqueeze(-1))
        q = self.edge_query(query).unsqueeze(-2).expand(-1, -1, 4, -1)
        edge_ids = self.edge_ids.to(q.dtype).expand(*q.shape[:-1], 4)
        evidence = torch.cat((scout.flatten(-2).to(q.dtype), q, edge_ids), dim=-1)
        offset = self.edge_offset(evidence).squeeze(-1).float().tanh() * radius.clamp(max=self.max_offset_px)

        fine_steps = offset.unsqueeze(-1) + radius.unsqueeze(-1) * p3.new_tensor([-0.5, 0.5])
        fine_grid = self._edge_grid(edge, fine_steps, tangent_shift, grid_scale)
        fine, valid = self._read_edges(p3, fine_grid, limit, valid)
        inside, outside = fine.unbind(dim=-2)
        relative_offset = (offset / radius.clamp_min(1e-6)).unsqueeze(-1)
        detail = torch.cat((inside, outside, inside - outside, relative_offset), dim=-1)
        detail = torch.where(valid.unsqueeze(-1), detail, 0.0).flatten(-2)
        return query + self.strength * self.edge_fuse(detail.to(query.dtype)).tanh()

    def read_context(self, query, boxes, p3):
        """Read a wider 3x3 neighborhood, using query similarity and a center prior."""
        center, size, eligible, grid_scale, limit = self._geometry(boxes, p3)
        offsets = 0.65 * self.context_grid.float()
        grid = (center.unsqueeze(-2) + size.unsqueeze(-2) * offsets) * grid_scale - 1.0
        valid = (grid.abs() <= limit).all(dim=-1) & eligible.unsqueeze(-1)
        values = self._sample(p3, torch.where(valid.unsqueeze(-1), grid, 0.0))
        values = torch.where(valid.unsqueeze(-1), values, 0.0)
        q = F.normalize(self.context_query(query).float(), dim=-1, eps=1e-6)
        keys = F.normalize(values, dim=-1, eps=1e-6)
        logits = 4.0 * (q.unsqueeze(-2) * keys).sum(dim=-1) + self.context_prior.float().log()
        weights = logits.masked_fill(~valid, -1e4).softmax(dim=-1) * valid
        pooled = (values * weights.unsqueeze(-1)).sum(dim=-2)
        central = values[..., 4, :]
        centroid = (weights.unsqueeze(-1) * offsets).sum(dim=-2)
        detail = torch.cat((pooled, central, pooled - central, centroid), dim=-1)
        return query + self.context_strength * self.context_fuse(detail.to(query.dtype)).tanh()


class _P3EdgeReadBoxHead(nn.Module):
    """Retain the existing regression linear layers and replace only their hidden activations."""

    def __init__(self, original):
        super().__init__()
        self.layers = original.layers

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.leaky_relu(layer(x), negative_slope=0.1)
        return self.layers[-1](x)


class _P3EdgeReadDecoder(DeformableTransformerDecoder):
    """Change inputs to selected bbox heads while keeping the original refinement loop."""

    def __init__(self, original, reader, num_queries):
        nn.Module.__init__(self)
        self.layers = original.layers
        self.num_layers, self.hidden_dim, self.eval_idx = original.num_layers, original.hidden_dim, original.eval_idx
        self.reader, self.num_queries = reader, num_queries

    def _regression_query(self, i, query, ref, p3):
        if i == self.eval_idx and self.reader.strength:
            read = self.reader.read_edges
        elif i == 1 and self.reader.context_strength:
            read = self.reader.read_context
        else:
            return query

        # Denoising queries are prepended. Keep the same layout for the loss and attention mask.
        dn = query.shape[1] - self.num_queries
        matching = read(query[:, dn:], ref[:, dn:], p3)
        return torch.cat((query[:, :dn], matching), dim=1) if dn else matching

    def forward(self, embed, refer_bbox, feats, shapes, bbox_head, score_head, pos_mlp,
                attn_mask=None, padding_mask=None):
        if not (self.reader.strength or self.reader.context_strength):
            return super().forward(embed, refer_bbox, feats, shapes, bbox_head, score_head, pos_mlp,
                                   attn_mask=attn_mask, padding_mask=padding_mask)

        p3 = self.reader.project_p3(feats, shapes)
        output = embed
        dec_bboxes, dec_cls = [], []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))
            box_query = self._regression_query(i, output, refer_bbox, p3)
            bbox = bbox_head[i](box_query)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox if i == 0 else torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)


class RTDETRDecoderP3EdgeRead(RTDETRDecoder):
    """Read P3 edges in layer 3, with an optional wider center read in layer 2."""

    def __init__(self, nc=80, ch=(512, 1024, 2048), hd=256, nq=300, ndp=4, nh=8, ndl=3,
                 read_params=None, **kwargs):
        super().__init__(nc, ch, hd, nq, ndp, nh, ndl, **kwargs)
        if ndl != 3 or self.decoder.eval_idx != 2 or len(ch) != 3:
            raise ValueError('P3EdgeRead requires three input scales and evaluation at decoder layer 3')
        self.dec_bbox_head[-1] = _P3EdgeReadBoxHead(self.dec_bbox_head[-1])
        self.decoder = _P3EdgeReadDecoder(self.decoder, P3EdgeRead(hd, **(read_params or {})), nq)

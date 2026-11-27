"""Sparse foreground-routed P2 proposals & iterative P2 deformable decoding (V2).

Fixes the bottlenecks of ``RTDETRDecoderP2Proposal`` (V1):
1. Keeps ``hd=256`` (never halves hidden dim like naive 4-scale P2 configs) and
   never pushes the 25,600 P2 tokens through ``enc_output`` / ``enc_score_head``.
2. Feeds foreground-routed stride-4 ``P2`` features into EVERY decoder layer's
   iterative sampling pool via ``P2SparseDeformAttn`` instead of only generating
   initial proposals on P2 and sampling exclusively from F3-F5.
3. Dual sparse routing:
   - Spatial foreground routing on P2 (guided by F3 encoder logits + P2 local
     foreground confidence, dilated 3x3, hard Top-K spatial mask keeping only
     ``p2_fg_ratio`` active foreground regions and zeroing pure background).
   - Query-level small-object / foreground routing inside each decoder layer:
     only the top ``p2_route_queries`` matching queries (scored by query
     content + reference box size prior) perform ``grid_sample`` on P2; DN
     queries and large/background queries keep the native F3-F5 path.
4. Deduplicated proposal selection:
   - 3x3 local-peak NMS on the P2 foreground map (prevents adjacent 4px cells
     from wasting slots on the same tiny object).
   - Cross-scale IoU deduplication against the base F3-F5 top-300 boxes
     (``dedup_iou``), reserving ``p2_queries`` slots for missed/under-covered
     small targets.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import VarifocalLoss
from ultralytics.utils.metrics import bbox_iou

from ..modules.conv import Conv
from ..modules.head import RTDETRDecoder
from ..modules.transformer import DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from ..modules.utils import _get_clones, bias_init_with_prob, inverse_sigmoid

__all__ = ('RTDETRDecoderP2ProposalV2',)


def _pairwise_cxcywh_iou(boxes1, boxes2):
    """Compute pairwise IoU between two batches of normalized cxcywh boxes.

    Args:
        boxes1 (Tensor): [B, N, 4] in [0, 1].
        boxes2 (Tensor): [B, M, 4] in [0, 1].

    Returns:
        Tensor: [B, N, M] IoU matrix.
    """
    xy1, wh1 = boxes1[..., :2], boxes1[..., 2:].clamp(min=1e-4)
    xy2, wh2 = boxes2[..., :2], boxes2[..., 2:].clamp(min=1e-4)
    tl1, br1 = xy1 - 0.5 * wh1, xy1 + 0.5 * wh1
    tl2, br2 = xy2 - 0.5 * wh2, xy2 + 0.5 * wh2
    lt = torch.maximum(tl1.unsqueeze(2), tl2.unsqueeze(1))
    rb = torch.minimum(br1.unsqueeze(2), br2.unsqueeze(1))
    inter_wh = (rb - lt).clamp(min=0.0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    area1 = (wh1[..., 0] * wh1[..., 1]).unsqueeze(2)
    area2 = (wh2[..., 0] * wh2[..., 1]).unsqueeze(1)
    union = (area1 + area2 - inter).clamp(min=1e-6)
    return inter / union


class P2SparseDeformAttn(nn.Module):
    """Query-sparse multi-head deformable sampling dedicated to stride-4 P2.

    Only the top ``k_route`` matching queries (prioritizing small reference
    boxes and foreground semantics) run ``grid_sample`` on the pre-routed P2
    feature map; remaining queries skip P2 sampling completely.
    """

    def __init__(self, d_model=256, n_heads=8, n_points=4, k_route=96):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f'd_model ({d_model}) must be divisible by n_heads ({n_heads})')
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_points = n_points
        self.k_route = int(k_route)

        # Size-aware sparse router over matching queries
        self.size_embed = nn.Linear(4, 32)
        self.router = nn.Sequential(
            nn.Linear(d_model + 32, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        # Learnable small-object prior weight (prefers sqrt(w*h) < 0.05)
        self.size_prior_weight = nn.Parameter(torch.tensor(8.0))

        # Deformable P2 sampling heads
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)
        self.out_proj = nn.Linear(d_model, d_model)
        # Small non-zero residual scale paired with small out_proj keeps step-0
        # perturbation minimal while passing clean gradients from step 0.
        self.res_scale = nn.Parameter(torch.full((d_model,), 0.2))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.size_embed.weight)
        nn.init.zeros_(self.size_embed.bias)
        nn.init.xavier_uniform_(self.router[0].weight)
        nn.init.zeros_(self.router[0].bias)
        nn.init.normal_(self.router[2].weight, std=1e-2)
        nn.init.constant_(self.router[2].bias, -0.5)

        nn.init.zeros_(self.sampling_offsets.weight)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 2).repeat(
            1, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, i, :] *= (i + 1) * 0.5
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.view(-1))

        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.normal_(self.out_proj.weight, std=1e-2)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, query_with_pos, refer_bbox, p2_value_heads, p2_hw):
        """Sparse P2 deformable readout for a [B, N_m, C] matching-query slice.

        Args:
            query_with_pos (Tensor): [B, N_m, C] query + positional embedding.
            refer_bbox (Tensor): [B, N_m, 4] normalized cxcywh reference boxes.
            p2_value_heads (Tensor): [B*H, D_h, H2, W2] pre-projected P2 values.
            p2_hw (Tuple[int, int]): (H2, W2).

        Returns:
            Tensor: [B, N_m, C] sparse P2 cross-attention delta (0 on unrouted).
        """
        bs, n_q, c = query_with_pos.shape
        k_q = min(self.k_route, n_q)
        if k_q <= 0:
            return query_with_pos.new_zeros(bs, n_q, c)

        wh = refer_bbox[..., 2:4].detach().clamp(min=1e-4)
        area_sqrt = (wh[..., 0] * wh[..., 1]).sqrt()
        size_feat = torch.stack(
            (wh[..., 0], wh[..., 1], area_sqrt, torch.log(area_sqrt + 1e-4)),
            dim=-1,
        )
        s_emb = self.size_embed(size_feat.to(query_with_pos.dtype))
        route_logits = self.router(torch.cat((query_with_pos, s_emb), dim=-1)).squeeze(-1)
        # Small-object size bias: boosts boxes smaller than ~32px (0.05)
        small_prior = ((0.05 - area_sqrt) * self.size_prior_weight).clamp(-4.0, 4.0).to(route_logits.dtype)
        route_prob = torch.sigmoid(route_logits + small_prior)

        top_prob, top_idx = route_prob.topk(k_q, dim=1)
        idx_q = top_idx.unsqueeze(-1).expand(-1, -1, c)
        idx_b = top_idx.unsqueeze(-1).expand(-1, -1, 4)
        q_sub = torch.gather(query_with_pos, 1, idx_q)       # [B, k_q, C]
        box_sub = torch.gather(refer_bbox, 1, idx_b)         # [B, k_q, 4]

        offsets = self.sampling_offsets(q_sub).view(bs, k_q, self.n_heads, self.n_points, 2)
        attn = self.attention_weights(q_sub).view(bs, k_q, self.n_heads, self.n_points)
        attn = F.softmax(attn, dim=-1)

        # Box-scaled deformable offsets around refer_bbox center
        scale_wh = box_sub[..., 2:4].clamp(min=1.0 / max(p2_hw)).unsqueeze(2).unsqueeze(3)
        locs = box_sub[..., :2].unsqueeze(2).unsqueeze(3) + offsets / self.n_points * scale_wh * 0.5
        grids = (2.0 * locs - 1.0).transpose(1, 2).reshape(bs * self.n_heads, k_q, self.n_points, 2)

        sampled = F.grid_sample(
            p2_value_heads,
            grids.to(p2_value_heads.dtype),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )  # [B*H, D_h, k_q, n_points]
        attn_w = attn.transpose(1, 2).reshape(bs * self.n_heads, 1, k_q, self.n_points).to(sampled.dtype)
        agg = (sampled * attn_w).sum(dim=-1).view(bs, c, k_q).transpose(1, 2)  # [B, k_q, C]

        out_sub = self.out_proj(agg.to(query_with_pos.dtype))
        out_sub = out_sub * (top_prob.unsqueeze(-1) * self.res_scale).to(out_sub.dtype)

        out = query_with_pos.new_zeros(bs, n_q, c)
        out.scatter_add_(1, idx_q, out_sub.to(out.dtype))
        return out


class DecoderLayerP2Sparse(DeformableTransformerDecoderLayer):
    """Standard DeformableTransformerDecoderLayer + sparse P2 iterative sampling."""

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0., act=nn.ReLU(),
                 n_levels=3, n_points=4, k_route=96):
        super().__init__(d_model, n_heads, d_ffn, dropout, act, n_levels, n_points)
        self.p2_attn = P2SparseDeformAttn(d_model=d_model, n_heads=n_heads, n_points=n_points, k_route=k_route)

    def forward(self, embed, refer_bbox, feats, shapes, padding_mask=None, attn_mask=None,
                query_pos=None, dn_num=0, p2_value_heads=None, p2_hw=None):
        # Self attention
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1),
                             attn_mask=attn_mask)[0].transpose(0, 1)
        embed = self.norm1(embed + self.dropout1(tgt))

        # Native 3-level [F3, F4, F5] cross-attention
        q_cross = self.with_pos_embed(embed, query_pos)
        tgt = self.cross_attn(q_cross, refer_bbox.unsqueeze(2), feats, shapes, padding_mask)

        # Iterative sparse P2 cross-attention on matching queries only
        if p2_value_heads is not None and p2_hw is not None:
            q_match = q_cross[:, dn_num:]
            box_match = refer_bbox[:, dn_num:]
            tgt_p2 = self.p2_attn(q_match, box_match, p2_value_heads, p2_hw)
            if dn_num > 0:
                pad_dn = tgt.new_zeros(tgt.shape[0], dn_num, tgt.shape[2])
                tgt = tgt + torch.cat((pad_dn, tgt_p2.to(tgt.dtype)), dim=1)
            else:
                tgt = tgt + tgt_p2.to(tgt.dtype)

        embed = self.norm2(embed + self.dropout2(tgt))
        return self.forward_ffn(embed)


class DeformableTransformerDecoderP2Sparse(DeformableTransformerDecoder):
    """Iterative decoder forwarding pre-projected sparse P2 value pool to each layer."""

    def forward(self, embed, refer_bbox, feats, shapes, bbox_head, score_head, pos_mlp,
                attn_mask=None, padding_mask=None, dn_num=0, p2_value_heads=None, p2_hw=None):
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output = layer(
                output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox),
                dn_num=dn_num, p2_value_heads=p2_value_heads, p2_hw=p2_hw,
            )

            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)


class RTDETRDecoderP2ProposalV2(RTDETRDecoder):
    """Repaired P2Proposal decoder: hd=256, sparse foreground-routed P2 sampling pool,
    query-level small-object routing, and cross-scale deduplicated proposal selection.

    Args:
        nc (int): Number of classes.
        ch (list[int]): Input channels ``[F3, F4, F5, P2]``.
        hd (int): Hidden dim (kept at 256).
        nq (int): Base F3-F5 query count (default 300).
        ndp (int): Deformable sampling points per head/level.
        nh (int): Attention heads.
        ndl (int): Decoder layers.
        p2_queries (int): Deduplicated P2 proposal queries appended to base ``nq``.
        p2_dim (int): Lightweight P2 fusion channel width before ``p2_val_proj``.
        p2_route_queries (int): Top-K matching queries allowed to sample P2 per layer.
        p2_fg_ratio (float): Fraction of P2 spatial pixels kept active as foreground.
        dedup_iou (float): Cross-scale IoU threshold to suppress P2 proposals already
            covered by base F3-F5 top-k boxes (<=0 disables cross-scale suppression).
        ground_loss_weight (float): Auxiliary tiny-GT P2 dense supervision weight
            (0.0 disables auxiliary loss for strict zero-loss-change compliance).
    """

    def __init__(self, nc, ch, hd=256, nq=300, ndp=4, nh=8, ndl=3,
                 p2_queries=32, p2_dim=128, p2_route_queries=96,
                 p2_fg_ratio=0.25, dedup_iou=0.65, ground_loss_weight=0.0):
        if len(ch) != 4 or p2_queries < 1:
            raise ValueError('RTDETRDecoderP2ProposalV2 requires [F3, F4, F5, P2] inputs and p2_queries >= 1')
        super().__init__(nc, ch[:3], hd, nq, ndp, nh, ndl)

        # Replace 3-level decoder layers with P2-sparse-aware layers (keeps hd=256)
        layer = DecoderLayerP2Sparse(hd, nh, d_ffn=1024, dropout=0., act=nn.ReLU(),
                                     n_levels=self.nl, n_points=ndp, k_route=p2_route_queries)
        self.decoder = DeformableTransformerDecoderP2Sparse(hd, layer, ndl, eval_idx=-1)
        # Re-init base decoder layer submodules identically to RTDETRDecoder
        for dec_layer in self.decoder.layers:
            dec_layer.cross_attn._reset_parameters()
            dec_layer.p2_attn._reset_parameters()

        self.p2_queries = int(p2_queries)
        self.p2_dim = int(p2_dim)
        self.p2_route_queries = int(p2_route_queries)
        self.p2_fg_ratio = float(p2_fg_ratio)
        self.dedup_iou = float(dedup_iou)
        self.ground_loss_weight = float(ground_loss_weight)

        # P2 semantic + local fusion (never passes through enc_output)
        self.p2_proj = Conv(ch[3], self.p2_dim, 1)
        self.p2_semantics = Conv(ch[0], self.p2_dim, 1)
        self.p2_mix = nn.Sequential(
            Conv(self.p2_dim, self.p2_dim, 3, g=self.p2_dim),
            Conv(self.p2_dim, self.p2_dim, 1),
        )
        # Projection into hd=256 sampling pool for iterative P2 deformable attention
        self.p2_val_proj = nn.Sequential(
            nn.Conv2d(self.p2_dim, hd, 1, bias=False),
            nn.BatchNorm2d(hd),
        )
        nn.init.xavier_uniform_(self.p2_val_proj[0].weight)

        # P2 proposal & foreground heads (initialized conservatively so F3 prior
        # guides early foreground localization before P2 residuals refine it)
        self.p2_score = nn.Conv2d(self.p2_dim, nc, 1)
        self.p2_box = nn.Conv2d(self.p2_dim, 4, 1)
        self.f3_prior_scale = nn.Parameter(torch.tensor(0.5))
        self.p2_embed = nn.Sequential(nn.Linear(self.p2_dim, hd), nn.LayerNorm(hd))
        self._dense_proposals = None

        nn.init.normal_(self.p2_score.weight, std=1e-2)
        nn.init.constant_(self.p2_score.bias, bias_init_with_prob(0.01) * 0.25)
        nn.init.zeros_(self.p2_box.weight)
        nn.init.zeros_(self.p2_box.bias)

    def _get_base_encoder_and_scores(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Run native top-nq query selection on [F3, F4, F5] and return F3 logits too."""
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
        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features

        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()

        h3, w3 = shapes[0]
        f3_scores = enc_outputs_scores[:, :h3 * w3, :].transpose(1, 2).reshape(bs, self.nc, h3, w3)
        return embeddings, refer_bbox, enc_bboxes, enc_scores, f3_scores

    def _build_p2_proposals_and_pool(self, p2, f3, f3_scores, base_enc_boxes):
        """Fuse F3->P2, apply foreground spatial routing & deduplicated proposal selection."""
        bs, _, h2, w2 = p2.shape
        device = p2.device

        semantic = F.interpolate(self.p2_semantics(f3), size=(h2, w2), mode='bilinear', align_corners=False)
        fused = self.p2_proj(p2) + semantic
        fine = fused + self.p2_mix(fused)  # [B, p2_dim, H2, W2]

        # F3 foreground prior upsampled to stride-4 P2 grid
        f3_logits_up = F.interpolate(f3_scores.detach(), size=(h2, w2), mode='bilinear', align_corners=False)
        p2_logits_map = self.p2_score(fine) + self.f3_prior_scale * f3_logits_up

        # Foreground spatial routing mask: dilate max foreground prob, keep top p2_fg_ratio
        fg_prob_p2 = p2_logits_map.detach().amax(dim=1, keepdim=True).sigmoid()
        fg_prob_f3 = f3_logits_up.amax(dim=1, keepdim=True).sigmoid()
        fg_score = 0.5 * fg_prob_p2 + 0.5 * fg_prob_f3
        fg_dilated = F.max_pool2d(fg_score, kernel_size=3, stride=1, padding=1)  # [B, 1, H2, W2]

        if 0.0 < self.p2_fg_ratio < 1.0:
            k_fg = max(self.p2_queries, int(round(h2 * w2 * self.p2_fg_ratio)))
            flat_fg = fg_dilated.flatten(1)
            top_fg_idx = flat_fg.topk(k_fg, dim=1).indices
            hard_fg_mask = torch.zeros_like(flat_fg)
            hard_fg_mask.scatter_(1, top_fg_idx, 1.0)
            spatial_mask = hard_fg_mask.view(bs, 1, h2, w2) * fg_dilated
        else:
            hard_fg_mask = fg_dilated.new_ones(bs, h2 * w2)
            spatial_mask = fg_dilated

        # Routed stride-4 P2 sampling pool [B*nh, head_dim, H2, W2]
        p2_val = self.p2_val_proj(fine) * spatial_mask.to(fine.dtype)
        p2_val_heads = p2_val.view(bs * self.nhead, self.hidden_dim // self.nhead, h2, w2)

        # Proposal boxes on P2 grid
        scores = p2_logits_map.flatten(2).transpose(1, 2)  # [B, H2*W2, nc]
        offsets = self.p2_box(fine).flatten(2).transpose(1, 2).float()
        y_g, x_g = torch.meshgrid(
            torch.arange(h2, device=device, dtype=offsets.dtype),
            torch.arange(w2, device=device, dtype=offsets.dtype),
            indexing='ij',
        )
        grid = torch.stack((x_g, y_g), dim=-1).reshape(1, -1, 2) + 0.5
        spatial_scale = offsets.new_tensor((w2, h2))
        centers = (grid + 2.0 * offsets[..., :2].tanh()) / spatial_scale
        size_prior = offsets.new_tensor((2.0, 3.0)) / spatial_scale
        refs = torch.cat((inverse_sigmoid(centers), offsets[..., 2:] + inverse_sigmoid(size_prior)), dim=-1)
        boxes = refs.sigmoid()

        # Deduplicated P2 proposal selection:
        # 1) Restrict to active foreground regions + 3x3 local peak (avoids 4px-adjacent dupes)
        raw_max_map = p2_logits_map.amax(dim=1, keepdim=True)
        is_peak = (raw_max_map == F.max_pool2d(raw_max_map, kernel_size=3, stride=1, padding=1)).flatten(1)
        active_fg = (hard_fg_mask > 0)
        sel_scores = raw_max_map.flatten(1)
        sel_scores = torch.where(active_fg & is_peak, sel_scores, sel_scores - 1e4)

        # 2) Suppress P2 candidates already well-covered by F3-F5 top-300 boxes
        cand_count = min(max(self.p2_queries * 8, 256), h2 * w2)
        cand_scores, cand_idx = sel_scores.topk(cand_count, dim=1)
        if self.dedup_iou > 0.0 and base_enc_boxes is not None:
            cand_boxes = torch.gather(boxes, 1, cand_idx.unsqueeze(-1).expand(-1, -1, 4)).detach()
            max_base_iou = _pairwise_cxcywh_iou(cand_boxes, base_enc_boxes.detach()).amax(dim=-1)
            cand_scores = torch.where(max_base_iou > self.dedup_iou, cand_scores - 1e4, cand_scores)

        count = min(self.p2_queries, cand_count)
        sub_pick = cand_scores.topk(count, dim=1).indices
        indices = torch.gather(cand_idx, 1, sub_pick)
        batch_ids = torch.arange(bs, device=device)[:, None]

        fine_flat = fine.flatten(2).transpose(1, 2)
        content = self.p2_embed(fine_flat[batch_ids, indices])
        return content, refs, boxes, scores, (batch_ids, indices), p2_val_heads, (h2, w2)

    def forward(self, x, batch=None):
        """Decode [F3, F4, F5] queries + deduplicated P2 proposals with sparse P2 iterative sampling."""
        from ultralytics.models.utils.ops import get_cdn_group

        self._dense_proposals = None
        feats, shapes = self._get_encoder_input(x[:3])
        embed, refs, enc_boxes, enc_scores, f3_scores = self._get_base_encoder_and_scores(feats, shapes)

        fine_embed, fine_refs, fine_boxes, fine_scores, selected, p2_val_heads, p2_hw = \
            self._build_p2_proposals_and_pool(x[3], x[0], f3_scores, enc_boxes)

        fine_selected_refs = fine_refs[selected].to(refs.dtype)
        enc_boxes = torch.cat((enc_boxes, fine_boxes[selected].to(enc_boxes.dtype)), dim=1)
        enc_scores = torch.cat((enc_scores, fine_scores[selected].to(enc_scores.dtype)), dim=1)

        if self.training and batch is not None:
            if self.ground_loss_weight > 0.0:
                self._dense_proposals = (fine_scores, fine_boxes, p2_hw)
            fine_selected_refs = fine_selected_refs.detach()

        embed = torch.cat((embed, fine_embed.to(embed.dtype)), dim=1)
        refs = torch.cat((refs, fine_selected_refs), dim=1)

        dn_embed, dn_boxes, mask, dn_meta = get_cdn_group(
            batch, self.nc, embed.shape[1], self.denoising_class_embed.weight, self.num_denoising,
            self.label_noise_ratio, self.box_noise_scale, self.training and batch is not None,
        )
        dn_num = 0 if dn_embed is None else dn_embed.shape[1]
        if dn_embed is not None:
            embed = torch.cat((dn_embed.to(embed.dtype), embed), dim=1)
            refs = torch.cat((dn_boxes.to(refs.dtype), refs), dim=1)

        boxes, scores = self.decoder(
            embed, refs, feats, shapes, self.dec_bbox_head, self.dec_score_head,
            self.query_pos_head, attn_mask=mask, dn_num=dn_num,
            p2_value_heads=p2_val_heads, p2_hw=p2_hw,
        )
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
        """Optional tiny-GT P2 dense supervision (disabled when ground_loss_weight <= 0)."""
        proposals, self._dense_proposals = self._dense_proposals, None
        if proposals is None or self.ground_loss_weight <= 0.0:
            return None
        scores, boxes, (height, width) = proposals
        scores = scores.float().contiguous()
        gt = targets['bboxes'].float()
        sizes = gt[:, 2:] * gt.new_tensor((4 * width, 4 * height))
        tiny = (sizes.prod(dim=1) < 32 ** 2) & (sizes.amin(dim=1) < 8)
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
        return self.ground_loss_weight * (cls_loss + 5 * l1 + 2 * giou) / denominator

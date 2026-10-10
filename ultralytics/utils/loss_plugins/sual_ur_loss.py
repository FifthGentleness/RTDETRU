# Ultralytics YOLO 馃悰, AGPL-3.0 license
"""
SUAL Probe B: UR-only (uncertainty-driven IoU reweighting) for RT-DETR.

Purpose: single-factor isolation of the UR component from the full SUAL loss
(ultralytics/utils/loss_plugins/sual_loss.py). If Probe B alone breaks
training (mAP ~ 0), the UR implementation in SUAL is the culprit; if Probe B
is healthy, the bug lives in the SLS / SQCL / UM components instead.

What is KEPT from baseline RT-DETR (identical to DETRLoss._get_loss_bbox):
    loss_bbox = gain_bbox * L1(cx, cy, w, h)
    loss_giou = gain_giou * (1 - GIoU)
    cls loss (VFL), Hungarian matching, DN plumbing: unchanged.

What Probe B changes (the ONLY delta vs baseline):
    loss_giou_i = (1 - GIoU_i) * (1 + rho * u_hat_i)     [main branch only]

Two fixes relative to the full SUAL implementation:

[F1] Fixed-scale uncertainty normalization (replaces per-batch max).
    SUAL used u_hat = std / std.max(), i.e. the weight of every matched pair
    is decided by whichever pair happens to have the largest inter-layer std
    in the current batch of 4 images - extremely noisy early in training.
    Here: u_hat = (std / u_ref).clamp(0, 1) with u_ref a detached EMA of the
    running mean inter-layer std (u_ref_init seeds it). The weight therefore
    lies in [1, 1 + rho] and moves smoothly with the training state.

[F2] DN-branch safety.
    SUAL computed u_hat from the main match indices but _get_loss_bbox is
    also called for the denoising branch (postfix='_dn') whose matching
    order is different; SUAL thus applied misaligned weights to DN losses.
    Here u_hat is applied ONLY to the main branch (postfix == '');
    the DN branch uses the plain baseline giou term.

Additional hygiene:
    - Uncertainty is estimated from detached decoder outputs (no gradient
      flows through u_ref or std), so UR cannot amplify gradients, only
      reweight them within [1, 1 + rho].
    - At validation time (no grad) u_hat is None: val losses are exactly
      baseline, keeping curves comparable.
    - No extra loss keys are introduced: results.csv columns stay identical
      to the baseline run for same-epoch comparison.

Registered as loss_name: 'sual_ur' in ultralytics/utils/loss_plugins.
"""

import torch
import torch.nn.functional as F

from ultralytics.models.utils.loss import DETRLoss, RTDETRDetectionLoss
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import bbox_iou

__all__ = ('RTDETRDetectionLossSUALUR',)


class RTDETRDetectionLossSUALUR(RTDETRDetectionLoss):
    """RT-DETR loss with UR-only uncertainty reweighting of the GIoU term.

    Difference vs baseline DETRLoss: the main-branch giou term of matched
    queries is multiplied by (1 + rho * u_hat), where u_hat in [0, 1] is a
    fixed-scale normalized inter-decoder-layer std of the matched predicted
    boxes. Training-only: no parameters, no inference cost.
    """

    def __init__(self,
                 nc=80,
                 loss_gain=None,
                 aux_loss=True,
                 use_fl=True,
                 use_vfl=True,
                 use_sl=False,
                 use_emasl=False,
                 use_svfl=False,
                 use_emasvfl=False,
                 use_mal=False,
                 sual_rho=1.0,        # rho: reweighting strength, weight in [1, 1+rho]
                 u_ref_init=0.02,     # seed for the EMA std reference (normalized xywh units)
                 u_ref_momentum=0.99, # EMA momentum of the std reference
                 u_log_every=200,     # print u_hat stats every N optimizer steps (0 = off)
                 **kwargs):
        super().__init__(nc=nc, loss_gain=loss_gain, aux_loss=aux_loss,
                         use_fl=use_fl, use_vfl=use_vfl, use_sl=use_sl,
                         use_emasl=use_emasl, use_svfl=use_svfl,
                         use_emasvfl=use_emasvfl, use_mal=use_mal, **kwargs)
        self.sual_rho = float(sual_rho)
        self.u_ref_init = float(u_ref_init)
        self.u_ref_momentum = float(u_ref_momentum)
        self.u_log_every = int(u_log_every)
        self._u_ref = self.u_ref_init   # detached scalar, EMA of mean inter-layer std
        self._u_hat = None              # [M] in [0, 1], main-branch matched order
        self._n_calls = 0

    # ------------------------------------------------------------------
    # uncertainty estimation (fixed-scale normalization, F1)
    # ------------------------------------------------------------------
    def _estimate_u_hat(self, pred_bboxes, match_indices):
        """Inter-layer std of matched predicted boxes, normalized by a
        detached EMA reference and clamped to [0, 1]. Returns None when not
        applicable (single decoder layer, no matches, or eval mode)."""
        if not torch.is_grad_enabled():
            return None
        if pred_bboxes.shape[0] < 2:
            return None
        idx, _ = self._get_index(match_indices)  # flatten to (batch_idx, query_idx)
        if idx is None or len(idx[0]) == 0:
            return None
        with torch.no_grad():
            per_layer = torch.stack([pred_bboxes[l][idx] for l in range(pred_bboxes.shape[0])])
            std = per_layer.std(0, unbiased=False).mean(-1)          # [M]
            # update the EMA reference (detached; never carries gradient)
            batch_mean = std.mean().item()
            self._u_ref = (self.u_ref_momentum * self._u_ref +
                           (1.0 - self.u_ref_momentum) * batch_mean)
            self._u_ref = max(self._u_ref, 1e-6)
            u_hat = (std / self._u_ref).clamp(0.0, 1.0)
        # diagnostic logging (cheap, every u_log_every calls)
        self._n_calls += 1
        if self.u_log_every > 0 and self._n_calls % self.u_log_every == 0:
            LOGGER.info(f'[SUAL-UR] u_ref={self._u_ref:.5f} '
                        f'u_hat mean={u_hat.mean().item():.3f} '
                        f'max={u_hat.max().item():.3f} '
                        f'weight range=[1.0, {1.0 + self.sual_rho * u_hat.max().item():.2f}]')
        return u_hat

    # ------------------------------------------------------------------
    # baseline bbox loss + UR reweight (main branch only, F2)
    # ------------------------------------------------------------------
    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=''):
        name_bbox = f'loss_bbox{postfix}'
        name_giou = f'loss_giou{postfix}'
        if len(gt_bboxes) == 0:
            zero = torch.tensor(0., device=self.device)
            return {name_bbox: zero, name_giou: zero}
        n = len(gt_bboxes)

        # ---- baseline L1 (identical to DETRLoss) ----
        loss_bbox = self.loss_gain['bbox'] * F.l1_loss(pred_bboxes, gt_bboxes, reduction='sum') / n

        # ---- baseline GIoU (identical to DETRLoss) ----
        giou_loss = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True).squeeze(-1)

        # ---- UR reweighting: main branch only (F2) ----
        if postfix == '' and self._u_hat is not None and len(self._u_hat) == n:
            giou_loss = giou_loss * (1.0 + self.sual_rho * self._u_hat)

        return {name_bbox: loss_bbox,
                name_giou: self.loss_gain['giou'] * giou_loss.sum() / n}

    # ------------------------------------------------------------------
    # forward: one matcher pass, u_hat from main matches, parent for the rest
    # ------------------------------------------------------------------
    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        pred_bboxes, pred_scores = preds
        self.device = pred_bboxes.device
        gt_cls, gt_bboxes, gt_groups = batch['cls'], batch['bboxes'], batch['gt_groups']

        match_indices = self.matcher(pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls,
                                     gt_groups, masks=None, gt_mask=None)

        # estimate once in the main-branch matched order; consumed by _get_loss_bbox
        self._u_hat = self._estimate_u_hat(pred_bboxes, match_indices)

        total_loss = DETRLoss.forward(self, pred_bboxes, pred_scores, batch, match_indices=match_indices)

        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta['dn_pos_idx'], dn_meta['dn_num_group']
            assert len(batch['gt_groups']) == len(dn_pos_idx)
            dn_match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch['gt_groups'])
            # _u_hat deliberately NOT reset: _get_loss_bbox ignores it for postfix='_dn'
            dn_loss = DETRLoss.forward(self, dn_bboxes, dn_scores, batch, postfix='_dn',
                                       match_indices=dn_match_indices)
            total_loss.update(dn_loss)
        else:
            total_loss.update({f'{k}_dn': torch.tensor(0., device=self.device)
                               for k in total_loss.keys()})

        self._u_hat = None  # clear after use; val forward passes have grad disabled anyway
        return total_loss

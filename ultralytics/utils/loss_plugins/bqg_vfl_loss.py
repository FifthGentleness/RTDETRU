"""Quality-preserving Brier-style gradient reweighting of final matching VFL.

Inspired by Uncertainty Weighted Gradients for Model Calibration, CVPR 2025:
https://github.com/Jinxu-Lin/BSCE-GRA
This sigmoid/IoU-aware adaptation is not categorical BSCE-GRA. No BlockSparse.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss


class RTDETRDetectionLossBQGVFL(RTDETRDetectionLoss):
    innovation_schedule = True

    def __init__(self, *args, lambda_max=0.25, warmup_epochs=30, ema_momentum=0.99, eps=1e-6, **kwargs):
        kwargs.setdefault('use_vfl', True)
        super().__init__(*args, **kwargs)
        if self.vfl is None or self.fl is None or any((self.sl,self.emasl,self.svfl,self.emasvfl,self.mal)):
            raise ValueError('BQG requires the original VFL + no-GT FocalLoss path')
        if not 0 <= lambda_max < 1 or warmup_epochs < 0 or not 0 <= ema_momentum < 1 or eps <= 0:
            raise ValueError('Invalid BQG strength, warmup, EMA momentum or epsilon')
        self.lambda_max, self.warmup_epochs = float(lambda_max), float(warmup_epochs)
        self.ema_momentum, self.eps = float(ema_momentum), float(eps)
        self.register_buffer('bqg_strength', torch.tensor(0. if warmup_epochs else float(lambda_max)))
        self.register_buffer('bqg_error_ema', torch.tensor(0.))
        # Float marker is also propagated by this repository's floating-buffer ModelEMA.
        self.register_buffer('bqg_initialized', torch.tensor(0.))
        self._aux_depth = 0

    def set_epoch(self, epoch, total_epochs=0):
        ramp = min(max(float(epoch), 0.) / self.warmup_epochs, 1.) if self.warmup_epochs else 1.
        self.bqg_strength.fill_(self.lambda_max * ramp)

    def _get_loss_aux(self, *args, **kwargs):
        # Base aux calls use the SAME postfix as the main loss. Postfix alone
        # cannot exclude encoder and intermediate decoder classification losses.
        self._aux_depth += 1
        try:
            return super()._get_loss_aux(*args, **kwargs)
        finally:
            self._aux_depth -= 1

    @torch.no_grad()
    def _row_weights(self, pred_scores, targets, gt_scores):
        matched = targets < self.nc
        p = pred_scores.detach().float().sigmoid()[matched]
        cls = targets[matched].long()
        quality = gt_scores.detach().reshape_as(targets)[matched].float()
        true_p = p.gather(1, cls[:, None]).squeeze(1)
        other = (p.square().sum(-1) - true_p.square()).clamp_min(0) / max(self.nc-1, 1)
        error = (true_p-quality).square() + other
        if self.training:
            stats = torch.stack((error.sum(), error.new_tensor(error.numel())))
            # Every rank participates, including ranks with zero GT. Synchronizing
            # only nonempty ranks would deadlock mixed empty/nonempty DDP batches.
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(stats)
            if stats[1] > 0:
                mean = stats[0] / stats[1]
                if self.bqg_initialized < 0.5:
                    self.bqg_error_ema.copy_(mean)
                    self.bqg_initialized.fill_(1.)
                else:
                    self.bqg_error_ema.mul_(self.ema_momentum).add_(mean, alpha=1-self.ema_momentum)
        weights = pred_scores.new_ones(targets.shape, dtype=torch.float32)
        if error.numel():
            mean = self.bqg_error_ema if self.bqg_initialized >= 0.5 else error.mean()
            raw = 1 + self.bqg_strength * (error / (mean+self.eps) - 1).clamp(-0.5, 1.)
            weights[matched] = raw / raw.mean()
        return weights

    def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=''):
        if postfix or self._aux_depth:
            return super()._get_loss_class(pred_scores, targets, gt_scores, num_gts, postfix)
        # Always participate in training statistics before the empty-GT fallback.
        row_weights = self._row_weights(pred_scores, targets, gt_scores)
        if not num_gts:
            return super()._get_loss_class(pred_scores, targets, gt_scores, num_gts, postfix)
        b, nq = pred_scores.shape[:2]
        one_hot = torch.zeros((b,nq,self.nc+1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        quality = gt_scores.reshape(b,nq,1) * one_hot
        # Keep the ORIGINAL differentiable VFL negative p^2 factor; only the
        # additional BQG row weights are detached.
        vfl_weight = 0.75 * pred_scores.sigmoid().pow(2.) * (1-one_hot) + quality * one_hot
        with torch.cuda.amp.autocast(enabled=False):
            elements = F.binary_cross_entropy_with_logits(pred_scores.float(), quality.float(), reduction='none')
            loss = (elements * vfl_weight * row_weights[..., None]).mean(1).sum()
        return {f'loss_class{postfix}': loss / (max(num_gts,1)/nq) * self.loss_gain['class']}

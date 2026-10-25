# DSAW: Dynamic Scale-Aware supervision Weighting for RT-DETR (loss plugin).
#
# 第三个创新点候选(损失函数方向, 非DETR系)。
# 灵感来源: Tiny object detection based on dynamic scale-awareness label
# assignment and contextual enhancement (DCNet), Pattern Recognition 172,
# 2026 (Zhang et al., 一区Top)。核心原则迁移: 监督强度不应对所有尺度
# 一视同仁, 且应随训练状态动态调节。
#
# 机制(单组件, 公理: 零新增loss项/不碰IoU度量/单组件纯净):
#   尺度因子   lam(s) = clip(1 - sqrt(w*h)/tau, 0, 1)      小目标->1, 大目标->0
#   动态因子   eta_t  = 1 - q_ema                          质量缺口: 早期大, 随训练自动收敛
#              (q_ema 为匹配对平均IoU的EMA跟踪, DCNet"动态"原则的参数免费实现)
#   分类支路   正样本VFL权重 q -> q * (1 + beta * lam * eta_t)   [recall主通道: 放大小目标分类梯度]
#   几何支路   逐样本 L1/GIoU 乘 (1 - alpha * lam)               [微小框几何监督温和化]
#
# 退化保证: beta=0 且 alpha=0 => 逐位等于基线 RTDETRDetectionLoss(VFL)。
# yaml loss_params:
#   beta=0.5  alpha=0.25  tau=0.03125  eta_momentum=0.999
#   q_ema_init=0.5  dynamic_eta=true (false=静态权重消融, eta恒1)
#
# 失败经验对照: SUAL/TSL死于多组件捆绑与回归侧重机制; 本方案为分类侧单组件
# 权重调度, 目标(q=IoU)与度量(GIoU/L1)完全不动, 与SR-MAL(目标质量轴)正交。

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.loss import FocalLoss
from ultralytics.utils.metrics import bbox_iou

__all__ = ('RTDETRDetectionLossDSAW',)


class RTDETRDetectionLossDSAW(RTDETRDetectionLoss):
    """DSAW: 尺度感知 + 训练动态的监督权重调度 (只改权重, 不改目标/度量)。

    Args:
        beta: 分类正样本放大强度 (0=关闭, 0.5默认, 1.0强版)。
        alpha: 几何温和化强度 (0=关闭, 0.25默认; 上限0.5, 防过度削弱)。
        tau: 尺度分界(归一化坐标), 1/32: 32px框lam=0, 16px~0.2, 8px~0.6。
        eta_momentum: 质量EMA动量(0.999 => 时间常数~1000次迭代)。
        q_ema_init: 质量EMA初值。
        dynamic_eta: true=动态因子(随训练收敛); false=静态消融(eta恒1)。
    """

    def __init__(self, nc=80, loss_gain=None, aux_loss=True, use_fl=True,
                 use_vfl=False, beta=0.5, alpha=0.25, tau=0.03125,
                 eta_momentum=0.999, q_ema_init=0.5, dynamic_eta=True, **kw):
        super().__init__(nc=nc, loss_gain=loss_gain, aux_loss=aux_loss,
                         use_fl=use_fl, use_vfl=use_vfl, **kw)
        self.beta = min(float(beta), 4.0)
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.eta_momentum = float(eta_momentum)
        self.q_ema = float(q_ema_init)
        self.dynamic_eta = bool(dynamic_eta)
        self.fl = FocalLoss()

    @property
    def eta(self):
        """动态因子 = 质量缺口 ∈ [0,1]: 训练早期(低质量)大, 后期自动趋0。"""
        if not self.dynamic_eta:
            return 1.0
        return max(0.0, min(1.0, 1.0 - self.q_ema))

    def _get_loss(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                  masks=None, gt_mask=None, postfix='', match_indices=None):
        """与父类相同流程, 额外构造: 质量EMA更新 + 逐样本权重图。"""
        if match_indices is None:
            match_indices = self.matcher(pred_bboxes, pred_scores, gt_bboxes,
                                         gt_cls, gt_groups,
                                         masks=masks, gt_mask=gt_mask)

        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes, gt_bboxes = pred_bboxes[idx], gt_bboxes[gt_idx]

        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device,
                             dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        w_cls_map = torch.ones([bs, nq], device=pred_scores.device)
        w_geo = None
        if len(gt_bboxes):
            iou = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1)
            gt_scores[idx] = iou
            # 尺度因子 lam: GT侧常数, 无梯度路径
            s = (gt_bboxes[:, 2] * gt_bboxes[:, 3]).clamp(min=0).sqrt()
            lam = (1.0 - s / self.tau).clamp(0.0, 1.0) if self.tau > 0 \
                else torch.zeros_like(s)
            # 动态因子: EMA跟踪匹配质量 (主/辅助层共用同一统计, 纯python侧状态)
            m = self.eta_momentum
            self.q_ema = m * self.q_ema + (1 - m) * float(iou.mean().detach())
            # 权重: 分类正样本放大 / 几何温和化
            w_cls_map[idx] = 1.0 + self.beta * lam * self.eta
            w_geo = 1.0 - self.alpha * lam

        loss = {}
        loss.update(self._get_loss_class_w(pred_scores, targets, gt_scores,
                                           len(gt_bboxes), w_cls_map, postfix))
        loss.update(self._get_loss_bbox_w(pred_bboxes, gt_bboxes, w_geo, postfix))
        return loss

    def _get_loss_class_w(self, pred_scores, targets, gt_scores, num_gts,
                          w_cls_map, postfix=''):
        """基线VFL的精确复写, 仅正样本权重 q -> q * w_cls_map。"""
        name_class = f'loss_class{postfix}'
        bs, nq = pred_scores.shape[:2]
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64,
                              device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        label = one_hot.float()
        gt_scores = gt_scores.view(bs, nq, 1) * label

        with torch.cuda.amp.autocast(enabled=False):
            weight = 0.75 * pred_scores.sigmoid().pow(2.0) * (1 - label) + \
                gt_scores * w_cls_map.unsqueeze(-1) * label
            loss_cls = (F.binary_cross_entropy_with_logits(
                pred_scores.float(), gt_scores.float(), reduction='none') *
                weight).mean(1).sum()

        if num_gts == 0:
            loss_cls = self.fl(pred_scores, label)
        else:
            loss_cls /= max(num_gts, 1) / nq
        return {name_class: loss_cls.squeeze() * self.loss_gain['class']}

    def _get_loss_bbox_w(self, pred_bboxes, gt_bboxes, w_geo, postfix=''):
        """基线 L1+GIoU 的精确复写, 仅逐样本乘 w_geo。"""
        name_bbox = f'loss_bbox{postfix}'
        name_giou = f'loss_giou{postfix}'
        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0., device=pred_bboxes.device)
            loss[name_giou] = torch.tensor(0., device=pred_bboxes.device)
            return loss

        if w_geo is None:
            w_geo = torch.ones(len(gt_bboxes), device=pred_bboxes.device)
        l1 = (pred_bboxes - gt_bboxes).abs().sum(-1)                     # [n]
        loss[name_bbox] = self.loss_gain['bbox'] * (l1 * w_geo).sum() / len(gt_bboxes)
        giou = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True,
                              GIoU=True).squeeze(-1)                     # [n]
        loss[name_giou] = self.loss_gain['giou'] * (giou * w_geo).sum() / len(gt_bboxes)
        return {k: v.squeeze() for k, v in loss.items()}

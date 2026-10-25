# SR-MAL: Scale-Robust Matchability-Aware Loss for RT-DETR (loss plugin).
#
# 第三个创新点(损失函数方向), 两个协同组件:
#   组件A (质量目标): 分类目标从纯 IoU 换成尺度鲁棒混合质量
#       q = lam(s) * NWD + (1 - lam(s)) * IoU,
#       lam(s) = clip(1 - sqrt(w*h)/tau, 0, 1)   (GT 尺度越小 lam 越大)
#     小目标(VisDrone 主力)的 IoU 高噪声 -> 用 Wasserstein 相似度(对 1px 偏移
#     平滑、尺度不变)托举分类目标; 大目标保持 IoU 精确边界监督。
#     在 MAL 框架 (DEIM, CVPR 2025) 下计算:
#       L_cls = BCE(p, q^gamma) * (p^gamma*(1-label) + label)
#     修复 MAL 原版在小目标上的系统性低分压制(q^gamma = IoU^2 对微小框过低,
#     DETR 无NMS纯靠分数排序 -> recall 腰斩的损失侧根源)。
#   组件B (分配-监督对齐): 匈牙利匹配 cost 的 giou 项替换为同一把混合尺子
#     cost_giou = 1 - ((1-lam_j)*GIoU + lam_j*NWD)   (lam_j 按 GT 列)
#     保证"谁被分配"与"被监督成什么"一致。DN/辅助层复用 _get_loss, 天然对齐。
#
# 超参数 (yaml loss_params):
#   gamma    : MAL 目标指数。1.0≈VFL式一次方目标, 2.0=DEIM默认, 2.5=更强排序判别
#   tau      : 尺度分界(归一化坐标)。1/32=0.03125: 32px框lam=0纯IoU, 16px≈0.2,
#              8px≈0.6; tau=0.0 => lam恒0 => 精确退化为原味 DEIM MAL(探针用)
#   nwd_const: Wasserstein 常数。坐标归一化到[0,1](=640px), 12.8/640≈0.02
#   align_matcher: 是否启用组件B
#
# 灵感来源(均已核实正式发表):
#   DEIM: DETR with Improved Matching for Fast Convergence, CVPR 2025 (MAL)
#   Uncertainty-Aware Gradient Stabilization for Small Object Detection,
#   ICCV 2025 (微小框几何度量高噪声的动机)
#
# 设计公理(本仓库18个loss插件失败复盘): 不新增辅助loss项(只替换现有项内部
# 目标/权重)、不动回归分支几何(GIoU+L1原样)、零推理开销零架构改动。

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.models.utils.ops import HungarianMatcher
from ultralytics.utils.loss import FocalLoss
from ultralytics.utils.metrics import bbox_iou, wasserstein_loss

__all__ = ('RTDETRDetectionLossSRMAL', 'SRMALHungarianMatcher')


def gt_scale_lambda(gt_bboxes, tau):
    """每个 GT 的 NWD 混合权重 lam ∈ [0,1], 按 GT 归一化尺度 sqrt(w*h) 线性衰减。

    tau<=0 时恒返回 0 (精确退化为纯 IoU 质量, 即原味 MAL 探针)。
    """
    if tau is None or tau <= 0:
        return torch.zeros_like(gt_bboxes[..., 0])
    s = (gt_bboxes[..., 2] * gt_bboxes[..., 3]).clamp(min=0).sqrt()
    return (1.0 - s / float(tau)).clamp(0.0, 1.0)


class SRMALHungarianMatcher(HungarianMatcher):
    """组件B: giou cost 替换为按 GT 尺度混合的 (GIoU, NWD) 相似度。"""

    def __init__(self, cost_gain=None, use_fl=True, tau=0.03125, nwd_const=0.02,
                 **kw):
        super().__init__(cost_gain=cost_gain, use_fl=use_fl, **kw)
        self.tau = float(tau)
        self.nwd_const = float(nwd_const)

    def forward(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                masks=None, gt_mask=None):
        bs, nq, nc = pred_scores.shape
        if sum(gt_groups) == 0:
            return [(torch.tensor([], dtype=torch.long),
                     torch.tensor([], dtype=torch.long)) for _ in range(bs)]

        # ---- 与父类一致: 类别 cost (focal 形式) ----
        pred_scores = pred_scores.detach().view(-1, nc)
        pred_scores = F.sigmoid(pred_scores) if self.use_fl else F.softmax(pred_scores, dim=-1)
        pred_bboxes = pred_bboxes.detach().view(-1, 4)

        pred_scores = pred_scores[:, gt_cls]
        if self.use_fl:
            neg_cost_class = (1 - self.alpha) * (pred_scores ** self.gamma) * \
                (-(1 - pred_scores + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - pred_scores) ** self.gamma) * \
                (-(pred_scores + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -pred_scores

        cost_bbox = (pred_bboxes.unsqueeze(1) - gt_bboxes.unsqueeze(0)).abs().sum(-1)

        # ---- 组件B: 尺度混合的 giou cost ----
        iou_sim = bbox_iou(pred_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(0),
                           xywh=True, GIoU=True).squeeze(-1)          # [nq, ng]
        nwd_sim = wasserstein_loss(pred_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(0),
                                   xywh=True, constant=self.nwd_const).squeeze(-1)  # [nq, ng]
        lam = gt_scale_lambda(gt_bboxes, self.tau).unsqueeze(0)        # [1, ng]
        cost_giou = 1.0 - ((1.0 - lam) * iou_sim + lam * nwd_sim)

        C = self.cost_gain['class'] * cost_class + \
            self.cost_gain['bbox'] * cost_bbox + \
            self.cost_gain['giou'] * cost_giou

        # ---- 与父类一致: 逐图线性分配 ----
        C[C.isnan() | C.isinf()] = 0.0
        C = C.view(bs, nq, -1).cpu()
        indices = [linear_sum_assignment(c[i])
                   for i, c in enumerate(C.split(gt_groups, -1))]
        gt_off = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        return [(torch.tensor(i, dtype=torch.long),
                 torch.tensor(j, dtype=torch.long) + gt_off[k])
                for k, (i, j) in enumerate(indices)]


class RTDETRDetectionLossSRMAL(RTDETRDetectionLoss):
    """SR-MAL: 尺度鲁棒匹配性监督 (组件A质量目标 + 组件B匹配对齐)。

    Args:
        gamma: MAL 目标指数 q^gamma (1.0≈VFL式, 2.0=DEIM默认, 2.5=强判别).
        tau: 尺度分界, 归一化坐标 (0=纯MAL探针, 1/32默认, 1/16温和, 1/64激进).
        nwd_const: Wasserstein 常数 (归一化坐标默认 0.02 = 12.8/640).
        align_matcher: 启用组件B (匹配 cost 与监督同尺子).
    """

    def __init__(self, nc=80, loss_gain=None, aux_loss=True, use_fl=True,
                 use_vfl=False, gamma=2.0, tau=0.03125, nwd_const=0.02,
                 align_matcher=True, **kw):
        super().__init__(nc=nc, loss_gain=loss_gain, aux_loss=aux_loss,
                         use_fl=use_fl, use_vfl=use_vfl, **kw)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.nwd_const = float(nwd_const)
        # 无 GT 图像分支需要 FocalLoss (与父类 fl 分支一致)
        self.fl = FocalLoss()
        if align_matcher:
            self.matcher = SRMALHungarianMatcher(
                cost_gain={'class': 2, 'bbox': 5, 'giou': 2},
                use_fl=True, tau=self.tau, nwd_const=self.nwd_const)

    # ---- 组件A: 混合质量目标 ----
    def _blended_quality(self, pred_bboxes, gt_bboxes):
        """已对齐的 (pred, gt) 对 -> q = lam*NWD + (1-lam)*IoU, lam 按 GT 尺度。"""
        iou = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1)
        nwd = wasserstein_loss(pred_bboxes.detach(), gt_bboxes, xywh=True,
                               constant=self.nwd_const).squeeze(-1)
        lam = gt_scale_lambda(gt_bboxes, self.tau)
        return lam * nwd + (1.0 - lam) * iou, lam

    def _get_loss(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                  masks=None, gt_mask=None, postfix='', match_indices=None):
        """与父类相同的流程, 仅 gt_scores 的构造换成尺度鲁棒混合质量。"""
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
        if len(gt_bboxes):
            q, _ = self._blended_quality(pred_bboxes, gt_bboxes)
            gt_scores[idx] = q

        loss = {}
        loss.update(self._get_loss_class(pred_scores, targets, gt_scores,
                                         len(gt_bboxes), postfix))
        loss.update(self._get_loss_bbox(pred_bboxes, gt_bboxes, postfix))
        return loss

    # ---- 组件A: MAL 形式分类损失 (gamma 可配) ----
    def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts,
                        postfix=''):
        name_class = f'loss_class{postfix}'
        bs, nq = pred_scores.shape[:2]
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64,
                              device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        label = one_hot.float()
        gt_scores = gt_scores.view(bs, nq, 1) * label        # 正样本处=质量q

        p = pred_scores.sigmoid()
        q_target = gt_scores.pow(self.gamma)
        weight = p.detach().pow(self.gamma) * (1.0 - label) + label
        loss_cls = (F.binary_cross_entropy_with_logits(
            pred_scores.float(), q_target.float(), reduction='none') *
            weight).mean(1).sum()

        if num_gts == 0:  # 与父类一致: 无GT图像退回 FocalLoss
            loss_cls = self.fl(pred_scores, label)
        else:
            loss_cls /= max(num_gts, 1) / nq
        return {name_class: loss_cls.squeeze() * self.loss_gain['class']}

"""Exercise the TCR plugin on CPU from the repository root: python tools/verify_dsaw_v8_tcr.py -v."""

from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics.utils.loss_plugins.dsaw_v4_loss import RTDETRDetectionLossDSAWV4
from ultralytics.utils.loss_plugins.dsaw_v8_tcr_loss import RTDETRDetectionLossDSAWV8TCR


RANK = 'loss_tcr_rank'
CROSS = 'loss_tcr_cross'


class FixedMatcher(nn.Module):
    """Pin ownership so score/IoU changes exercise the loss, not assignment changes."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, boxes, scores, gt_boxes, gt_cls, gt_groups, **kwargs):
        self.calls.append(boxes.data_ptr())
        offset = 0
        result = []
        for count in gt_groups:
            result.append((torch.arange(count), torch.arange(offset, offset + count)))
            offset += count
        return result


def make_inputs(boxes, probabilities, gt_groups, layers=3):
    """Every labeled image has one class-zero GT, centered at (0.25, 0.25)."""
    gt_box = (0.25, 0.25, 0.20, 0.20)
    box_stack = torch.tensor(boxes, dtype=torch.float32).repeat(layers, 1, 1, 1).requires_grad_()
    p = torch.tensor(probabilities, dtype=torch.float32)
    score_stack = torch.stack((p.logit(), torch.full_like(p, -6.0)), -1)
    score_stack = score_stack.repeat(layers, 1, 1, 1).requires_grad_()
    count = sum(gt_groups)
    return (box_stack, score_stack), {
        'bboxes': torch.tensor([gt_box] * count, dtype=torch.float32).reshape(-1, 4),
        'cls': torch.zeros(count, dtype=torch.long),
        'gt_groups': gt_groups,
        'image_hw': (640, 640),
    }


def centered(q):
    """Shrinking only width makes IoU to the labeled box equal to q."""
    return (0.25, 0.25, 0.20 * q, 0.20)


def denoising(gt_groups, layers):
    bs = len(gt_groups)
    return {
        'dn_bboxes': torch.tensor(centered(0.82)).repeat(layers, bs, 1, 1).requires_grad_(),
        'dn_scores': torch.zeros(layers, bs, 1, 2, requires_grad=True),
        'dn_meta': {'dn_pos_idx': [torch.arange(n) for n in gt_groups], 'dn_num_group': 1},
    }


def at_full_gain(**kwargs):
    criterion = RTDETRDetectionLossDSAWV8TCR(nc=2, use_vfl=True, **kwargs)
    criterion.set_epoch(29, 350)
    return criterion


class TestDSAWV8TCR(unittest.TestCase):
    def test_disabled_prewarm_and_eval_preserve_v4(self):
        preds, batch = make_inputs([[centered(0.84)], [centered(0.71)]], [[0.56], [0.68]], [1, 1])
        dn = denoising(batch['gt_groups'], len(preds[0]))
        base = RTDETRDetectionLossDSAWV4(nc=2, use_vfl=True)
        expected = base(preds, batch, **dn)
        tcr_criteria = (
            at_full_gain(rank_gain=0.0, cross_gain=0.0),
            RTDETRDetectionLossDSAWV8TCR(nc=2, use_vfl=True),
            at_full_gain().eval(),
        )
        for criterion in tcr_criteria:
            with self.subTest(training=criterion.training, rank_gain=criterion.rank_gain):
                actual = criterion(preds, batch, **dn)
                self.assertEqual(expected.keys(), actual.keys())
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                self.assertEqual(base.q_ema, criterion.q_ema)

    def test_rank_changes_only_selected_regular_logits(self):
        preds, batch = make_inputs(
            [[centered(0.84), centered(0.76), centered(0.93), (0.80, 0.80, 0.20, 0.20)]],
            [[0.56, 0.70, 0.78, 0.74]], [1],
        )
        dn = denoising(batch['gt_groups'], len(preds[0]))
        criterion = at_full_gain(cross_gain=0.0)
        matcher = FixedMatcher()
        criterion.matcher = matcher
        base = RTDETRDetectionLossDSAWV4(nc=2, use_vfl=True)
        base.matcher = FixedMatcher()
        expected = base(preds, batch, **dn)
        losses = criterion(preds, batch, **dn)
        boxes, scores = preds

        self.assertCountEqual(matcher.calls, [x.data_ptr() for x in boxes])
        self.assertEqual(set(losses) - set(expected), {RANK, CROSS})
        for key in expected:
            torch.testing.assert_close(losses[key], expected[key], rtol=0, atol=0)
        self.assertGreater(losses[RANK].item(), 0.0)
        scores_grad, boxes_grad, dn_grad, dn_box_grad = torch.autograd.grad(
            losses[RANK], (scores, boxes, dn['dn_scores'], dn['dn_bboxes']), allow_unused=True,
        )
        self.assertLess(scores_grad[-1, 0, 0, 0].item(), 0.0)  # winner rises
        self.assertGreater(scores_grad[-1, 0, 1, 0].item(), 0.0)  # known duplicate falls
        self.assertEqual(scores_grad[-1, 0, 2, 0].item(), 0.0)  # better unmatched is protected
        self.assertGreater(scores_grad[-1, 0, 3, 0].item(), 0.0)  # background falls
        self.assertTrue(torch.all(scores_grad[:-1] == 0))
        self.assertTrue(torch.all(scores_grad[..., 1] == 0))
        for grad in (boxes_grad, dn_grad, dn_box_grad):
            self.assertTrue(grad is None or torch.all(grad == 0))

    def test_cross_image_ranking_does_not_push_down_matched_positive(self):
        preds, batch = make_inputs([[centered(0.84)], [centered(0.71)]], [[0.56], [0.68]], [1, 1])
        criterion = at_full_gain(cross_gain=0.0)
        criterion.matcher = FixedMatcher()
        loss = criterion(preds, batch)[RANK]
        self.assertGreater(loss.item(), 0.0)
        grad, = torch.autograd.grad(loss, preds[1])
        self.assertLess(grad[-1, 0, 0, 0].item(), 0.0)
        self.assertEqual(grad[-1, 1, 0, 0].item(), 0.0)

    def test_crowded_neighbor_does_not_become_a_duplicate(self):
        # A candidate between two overlapping same-class GTs has no unique owner.
        preds, batch = make_inputs(
            [[centered(0.89), (0.29, 0.25, 0.20 * 0.88, 0.20), (0.269, 0.25, 0.17, 0.20)]],
            [[0.60, 0.55, 0.77]], [2],
        )
        batch['bboxes'][1, 0] = 0.29
        criterion = at_full_gain(cross_gain=0.0)
        criterion.matcher = FixedMatcher()
        losses = criterion(preds, batch)
        self.assertEqual(losses[RANK].item(), 0.0)

    def test_box_repair_reaches_threshold_inside_rank_guard_band(self):
        for q, active in ((0.740, True), (0.749, True), (0.751, False)):
            with self.subTest(iou=q):
                # IoU 0.80 produces a winner at .75 but not at .80 (guard=.015).
                preds, batch = make_inputs([[centered(0.80)], [centered(q)]], [[0.56], [0.68]], [1, 1])
                criterion = at_full_gain(rank_gain=0.0)
                criterion.matcher = FixedMatcher()
                loss = criterion(preds, batch)[CROSS]
                if not active:
                    self.assertEqual(loss.item(), 0.0)
                    continue
                self.assertGreater(loss.item(), 0.0)
                boxes_grad, scores_grad = torch.autograd.grad(loss, preds, allow_unused=True)
                # Increasing the lower-IoU box width moves it toward the GT.
                self.assertLess(boxes_grad[-1, 1, 0, 2].item(), 0.0)
                self.assertTrue(torch.all(boxes_grad[:-1] == 0))
                self.assertTrue(torch.all(boxes_grad[-1, 0] == 0))
                self.assertTrue(scores_grad is None or torch.all(scores_grad == 0))

    def test_empty_targets_produce_finite_zero_additions(self):
        preds, batch = make_inputs([[centered(0.80)]], [[0.9]], [0])
        losses = at_full_gain()(preds, batch)
        self.assertTrue(all(torch.isfinite(loss) for loss in losses.values()))
        self.assertEqual(losses[RANK].item(), 0.0)
        self.assertEqual(losses[CROSS].item(), 0.0)


if __name__ == '__main__':
    unittest.main()

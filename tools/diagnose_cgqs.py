"""Evaluate query ranking policies on one frozen CGQS EMA checkpoint.

Shares backbone/encoder features across policies. Uses repository AP matching,
640-square stretch preprocessing, and all 300 decoder predictions without NMS.
Only the in-memory decoder forward is replaced; training code is not modified.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ultralytics import RTDETR
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import ap_per_class, box_iou
from ultralytics.utils.ops import xywh2xyxy

POLICIES = ('joint', 'class_only', 'blend25')


def diagnostic_forward(self, x, batch=None):
    assert not self.training and batch is None
    feats, shapes = self._get_encoder_input(x)
    anchors, valid = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
    features = self.enc_output(valid * feats)
    scores = self.enc_score_head(features)
    center = self.enc_centerness_head(features).squeeze(-1).float().sigmoid()
    cls = scores.max(-1).values.float().sigmoid()
    ranking = {'joint': cls * center, 'class_only': cls,
               'blend25': cls * (0.75 + 0.25 * center)}
    all_boxes = (self.enc_bbox_head(features) + anchors).sigmoid()
    results = {}
    for policy in POLICIES:
        idx = ranking[policy].masked_fill(~valid.squeeze(-1), float('-inf')).topk(self.num_queries, 1).indices
        selected = features.gather(1, idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
        selected_anchors = anchors.expand(feats.shape[0], -1, -1).gather(1, idx.unsqueeze(-1).expand(-1, -1, 4))
        reference = self.enc_bbox_head(selected) + selected_anchors
        embed = self.tgt_embed.weight.unsqueeze(0).repeat(feats.shape[0], 1, 1) if self.learnt_init_query else selected
        boxes, logits = self.decoder(embed, reference, feats, shapes,
                                     self.dec_bbox_head, self.dec_score_head, self.query_pos_head)
        prediction = torch.cat((boxes.squeeze(0), logits.squeeze(0).sigmoid()), -1)
        results[policy] = {'prediction': prediction, 'idx': idx,
                           'initial': reference.sigmoid()}
    self._diagnostic = {'results': results, 'all_boxes': all_boxes, 'valid': valid,
                        'center': center, 'class': cls, 'shapes': shapes}
    # The surrounding model only needs this return value during the diagnostic.
    return results['joint']['prediction'], None


def ap_summary(stats, names):
    merged = [np.concatenate([s[i] for s in stats], axis=0) for i in range(4)]
    result = ap_per_class(*merged, plot=False, names=names)
    ap, ids = result[5], result[6]
    return {'mAP50': float(ap[:, 0].mean()), 'mAP75': float(ap[:, 5].mean()),
            'mAP50_95': float(ap.mean()), 'precision': float(result[2].mean()),
            'recall': float(result[3].mean()),
            'per_class': {names[int(k)]: {'AP50': float(ap[j, 0]),
                                        'AP50_95': float(ap[j].mean())}
                          for j, k in enumerate(ids)}}


def coverage(records, masks):
    values = np.concatenate(records)
    groups = np.concatenate(masks)
    return {name: {'count': int(mask.sum()), 'mean_best_iou': float(values[mask].mean()),
                   'R50': float((values[mask] >= 0.5).mean()),
                   'R75': float((values[mask] >= 0.75).mean())}
            for name, mask in [('all', np.ones(len(values), bool)), ('small', groups == 0),
                               ('medium', groups == 1), ('large', groups == 2)] if mask.any()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='runs/train/rtdetr_r18_DSAWACGAv5-P2SPDOKMFSV7-CGQS/weights/last.pt')
    parser.add_argument('--output', default='runs/diagnostics/cgqs_epoch77')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    weights = ROOT / args.weights
    initial_stat = weights.stat()
    with weights.open('rb') as f:
        digest = hashlib.sha256()
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    print('Loading checkpoint...', flush=True)
    wrapper = RTDETR(str(weights))
    net = wrapper.model.to(args.device).float().eval()
    head = net.model[-1]
    assert type(head).__name__ == 'RTDETRDecoderCGQS', type(head).__name__
    original_forward = head.forward
    epoch = int(wrapper.ckpt.get('epoch', -1)) + 1
    names = net.names
    print(f'Checkpoint epoch={epoch}, EMA={wrapper.ckpt.get("ema") is not None}, device={args.device}', flush=True)
    images = sorted((ROOT / 'dataset/visdrone/images/val').glob('*.jpg'))
    if args.limit:
        images = images[:args.limit]
    matcher = types.SimpleNamespace(iouv=torch.linspace(0.5, 0.95, 10, device=args.device))
    stats = {p: [] for p in POLICIES}
    geometry = {p: [] for p in POLICIES}
    final_geometry = {p: [] for p in POLICIES}
    all_geometry, masks, per_image = [], [], []
    center_samples, cls_samples, iou_samples = [], [], []
    levels = {p: [0] * head.nl for p in POLICIES}
    overlap = []
    selections_equal = []
    started = time.time()
    parity_error = None
    with torch.inference_mode():
        for i, file in enumerate(images):
            image = cv2.imread(str(file))
            if image is None:
                raise RuntimeError(f'Cannot read {file}')
            h, w = image.shape[:2]
            resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1))).to(args.device).float()[None] / 255
            label_file = ROOT / 'dataset/visdrone/labels/val' / (file.stem + '.txt')
            lines = label_file.read_text().strip().splitlines()
            labels = np.asarray([list(map(float, line.split())) for line in lines], np.float32).reshape(-1, 5)
            labels = np.unique(labels, axis=0)
            gt = torch.as_tensor(labels[:, 1:], device=args.device)
            gt_xyxy = xywh2xyxy(gt)
            gt_cls = torch.as_tensor(labels[:, 0], device=args.device)
            group = np.where(np.sqrt(labels[:, 3] * labels[:, 4]) * 640 < 32, 0,
                             np.where(np.sqrt(labels[:, 3] * labels[:, 4]) * 640 < 96, 1, 2))
            if i == 0:
                reference_prediction = net(tensor)[0]
                head.forward = types.MethodType(diagnostic_forward, head)
            prediction = net(tensor)[0]
            if i == 0:
                parity_error = float((reference_prediction - prediction).abs().max())
                print(f'Native joint-forward parity max abs error={parity_error:.8g}', flush=True)
                if parity_error > 1e-5:
                    raise RuntimeError('Diagnostic forward disagrees with native forward')
            cache = head._diagnostic
            valid = cache['valid'].squeeze()
            all_iou = box_iou(gt_xyxy * 640, xywh2xyxy(cache['all_boxes'][0]) * 640)
            best_all = all_iou[:, valid].max(1).values.cpu().numpy() if len(gt) else np.empty(0)
            all_geometry.append(best_all)
            masks.append(group)
            token_quality = all_iou.max(0).values if len(gt) else torch.zeros_like(valid, dtype=torch.float)
            valid_indices = valid.nonzero().flatten()
            sample = valid_indices[torch.linspace(0, len(valid_indices)-1, min(1000, len(valid_indices)), device=args.device).long()]
            center_samples.append(cache['center'][0, sample].cpu().numpy())
            cls_samples.append(cache['class'][0, sample].cpu().numpy())
            iou_samples.append(token_quality[sample].cpu().numpy())
            joint_idx = cache['results']['joint']['idx'][0]
            class_idx = cache['results']['class_only']['idx'][0]
            overlap_count = int(torch.isin(joint_idx, class_idx).sum())
            overlap.append(overlap_count / head.num_queries)
            selections_equal.append(overlap_count == head.num_queries)
            row = {'image': file.name, 'gt_count': len(gt), 'top300_overlap': overlap[-1]}
            boundaries = np.cumsum([hh*ww for hh, ww in cache['shapes']])
            for policy in POLICIES:
                result = cache['results'][policy]
                idx = result['idx'][0]
                ids = idx.cpu().numpy()
                for level in range(head.nl):
                    lo = boundaries[level-1] if level else 0
                    levels[policy][level] += int(((ids >= lo) & (ids < boundaries[level])).sum())
                best = all_iou[:, idx].max(1).values.cpu().numpy() if len(gt) else np.empty(0)
                geometry[policy].append(best)
                pred = result['prediction'][0]
                confidence, classes = pred[:, 4:].max(-1)
                order = confidence.argsort(descending=True)
                confidence, classes = confidence[order], classes[order]
                predicted_boxes = xywh2xyxy(pred[order, :4])
                # Match in original image coordinates, as RTDETRValidator does.
                native_scale = torch.tensor([w, h, w, h], device=args.device)
                iou = box_iou(gt_xyxy * native_scale, predicted_boxes * native_scale)
                correct = BaseValidator.match_predictions(matcher, classes, gt_cls, iou)
                stats[policy].append((correct.cpu().numpy(), confidence.cpu().numpy(),
                                      classes.cpu().numpy(), labels[:, 0]))
                final_iou = box_iou(gt_xyxy * 640, predicted_boxes * 640)
                final_best = final_iou.max(1).values.cpu().numpy() if len(gt) else np.empty(0)
                final_geometry[policy].append(final_best)
                row[policy+'_initial_R50'] = float((best >= .5).mean()) if len(gt) else None
                row[policy+'_initial_R75'] = float((best >= .75).mean()) if len(gt) else None
            per_image.append(row)
            del head._diagnostic
            if (i+1) % 25 == 0 or i+1 == len(images):
                print(f'{i+1}/{len(images)} images, {time.time()-started:.1f}s', flush=True)
    head.forward = original_forward
    if weights.stat().st_mtime_ns != initial_stat.st_mtime_ns or weights.stat().st_size != initial_stat.st_size:
        raise RuntimeError('Checkpoint changed during diagnosis; rerun on a stable checkpoint')
    center = np.concatenate(center_samples)
    quality = np.concatenate(iou_samples)
    cls = np.concatenate(cls_samples)
    groups = np.concatenate(masks)
    joint_best = np.concatenate(geometry['joint'])
    class_best = np.concatenate(geometry['class_only'])
    transitions = {}
    for threshold in (.5, .75):
        gained = (joint_best >= threshold) & (class_best < threshold)
        lost = (joint_best < threshold) & (class_best >= threshold)
        transitions[str(threshold)] = {name: {'gained_gt': int((gained & mask).sum()),
                                             'lost_gt': int((lost & mask).sum())}
                                      for name, mask in [('all', np.ones(len(groups), bool)),
                                                         ('small', groups == 0)]}
    summary = {'weights': str(weights), 'sha256': digest.hexdigest(), 'epoch_one_based': epoch,
               'EMA': wrapper.ckpt.get('ema') is not None, 'images': len(images),
               'gt_count': int(len(groups)), 'native_forward_max_abs_error': parity_error,
               'elapsed_seconds': time.time()-started,
               'notes': ['Same frozen checkpoint and shared encoder features across policies.',
                         'FP32; 640x640 stretch; repository AP matching, no NMS or confidence filtering.',
                         'Proposal coverage is class-agnostic and is not AP.',
                         'Small/medium/large use sqrt(area) at stretched 640: <32, 32-96, >=96.',
                         'Inference policy ablation does not isolate auxiliary-loss effects during training.'],
               'top300_overlap_mean': float(np.mean(overlap)),
               'images_with_identical_query_set': int(sum(selections_equal)),
               'centerness_sample': {'count': len(center), 'mean': float(center.mean()),
                   'std': float(center.std()), 'quantiles': np.quantile(center, [0,.1,.5,.9,1]).tolist(),
                   'mean_good_box_iou_ge_05': float(center[quality >= .5].mean()),
                   'mean_poor_box_iou_lt_01': float(center[quality < .1].mean()),
                   'pearson_with_box_iou': float(np.corrcoef(center, quality)[0,1]),
                   'class_pearson_with_box_iou': float(np.corrcoef(cls, quality)[0,1])},
               'joint_vs_class_coverage_transitions': transitions,
               'all_candidate_coverage': coverage(all_geometry, masks),
               'policies': {p: {'AP': ap_summary(stats[p], names),
                                'initial_coverage': coverage(geometry[p], masks),
                                'final_coverage': coverage(final_geometry[p], masks),
                                'selected_by_level': levels[p]} for p in POLICIES}}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (output / 'per_image.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    arrays = {}
    for p in POLICIES:
        for j, label in enumerate(('tp','confidence','class','target_class')):
            arrays[p+'_'+label] = np.concatenate([s[j] for s in stats[p]])
    arrays['gt_counts'] = np.asarray([r['gt_count'] for r in per_image])
    np.savez_compressed(output / 'ap_inputs.npz', **arrays)
    print(json.dumps({'epoch': epoch, 'overlap': summary['top300_overlap_mean'],
                      'AP': {p: summary['policies'][p]['AP'] for p in POLICIES}}, indent=2), flush=True)
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()

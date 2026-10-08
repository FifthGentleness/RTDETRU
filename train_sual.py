# -*- coding: utf-8 -*-
"""
Train RT-DETR V7 + SUAL loss (SLS scale branch + UGS-inspired uncertainty branch).

Usage:
    python train_sual.py                                   # full SUAL
    python train_sual.py --sqcl-weight 0.0 --um-weight 0.0 # SLS-only ablation
    python train_sual.py --rho 0.0 --um-weight 0.0         # no uncertainty reweight/UM
    python train_sual.py --nonuniform 2.0                  # non-uniform bins (UGS style)
    python train_sual.py --device 1 --name SUAL_full
"""
import argparse
import warnings

warnings.filterwarnings('ignore')

from ultralytics import RTDETR

MODEL_YAML = 'ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-SUAL.yaml'


def build_argparser():
    p = argparse.ArgumentParser('train_sual')
    p.add_argument('--model', type=str, default=MODEL_YAML, help='model yaml or pt path')
    p.add_argument('--name', type=str, default='SUAL_full', help='experiment name')
    p.add_argument('--device', type=str, default='', help='cuda device, e.g. 0 or 0,1 or cpu')
    p.add_argument('--resume', type=str, default='', help='path to last.pt to resume training')
    # ---- ablation switches ----
    p.add_argument('--no-sls', action='store_true',
                   help="drop the SLS branch: plain GIoU + no polar location term")
    p.add_argument('--no-sqcl', action='store_true', help='disable SQCL (lambda_c = 0)')
    p.add_argument('--no-um', action='store_true', help='disable UM (mu = 0)')
    # ---- hyperparameters ----
    p.add_argument('--loc-weight', type=float, default=None, help='lambda_L (polar location term)')
    p.add_argument('--sqcl-weight', type=float, default=None, help='lambda_c (SQCL term)')
    p.add_argument('--um-weight', type=float, default=None, help='mu (UM term)')
    p.add_argument('--rho', type=float, default=None, help='uncertainty reweighting strength')
    p.add_argument('--nonuniform', type=float, default=None, help='bin non-uniformity gamma')
    p.add_argument('--iou-type', type=str, default=None, choices=['giou', 'iou', 'ciou'])
    return p


def apply_loss_params(model, args):
    lp = model.model.yaml.setdefault('loss_params', {})
    if args.no_sls:
        # SLS branch off -> revert to the baseline regression form:
        # alpha weighting requires SLS, so flag it through loc_weight=0 + iou 'giou'
        # and let the user compare against the baseline run instead.
        lp['sls_loc_weight'] = 0.0
        print('[SUAL] --no-sls: polar location term disabled '
              '(note: scale-alpha weighting is inherent to the SLS IoU form; '
              'use the baseline V7 run as the no-SLS reference)')
    if args.no_sqcl:
        lp['sqcl_weight'] = 0.0
    if args.no_um:
        lp['um_weight'] = 0.0
    if args.loc_weight is not None:
        lp['sls_loc_weight'] = args.loc_weight
    if args.sqcl_weight is not None:
        lp['sqcl_weight'] = args.sqcl_weight
    if args.um_weight is not None:
        lp['um_weight'] = args.um_weight
    if args.rho is not None:
        lp['sual_rho'] = args.rho
    if args.nonuniform is not None:
        lp['sqcl_nonuniform'] = args.nonuniform
    if args.iou_type is not None:
        lp['sls_iou'] = args.iou_type
    active = {k: lp.get(k) for k in ('sls_iou', 'sls_loc_weight', 'sqcl_weight',
                                     'um_weight', 'sual_rho', 'sqcl_nonuniform')}
    print(f'[SUAL] active loss_params: {active}')


if __name__ == '__main__':
    args = build_argparser().parse_args()

    model = RTDETR(args.model)
    apply_loss_params(model, args)

    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=350,
                patience=0,
                batch=4,
                workers=4,
                pretrained=False,
                resume=args.resume,
                device=args.device,
                project='runs/train',
                name=args.name,
                )

# -*- coding: utf-8 -*-
"""
Train RT-DETR V7 + Tri-Scale Loss (SD + PGDE + SARD).

Usage:
    python train_triscale.py                                  # full Tri-Scale
    python train_triscale.py --no-sard                        # ablation: SD + PGDE
    python train_triscale.py --no-pgde --no-sard              # ablation: SD only
    python train_triscale.py --pgde-weight 3.0                # tune a hyperparameter
    python train_triscale.py --device 1 --name TSL_full       # custom run

The loss is selected by loss_name / loss_params inside the model yaml; the CLI
flags below mutate model.yaml['loss_params'] before training, so a single yaml
serves all ablation variants (criterion is built lazily at the first loss call).
"""
import argparse
import warnings

warnings.filterwarnings('ignore')

from ultralytics import RTDETR

MODEL_YAML = 'ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-TSL.yaml'


def build_argparser():
    p = argparse.ArgumentParser('train_triscale')
    p.add_argument('--model', type=str, default=MODEL_YAML, help='model yaml or pt path')
    p.add_argument('--name', type=str, default='TSL_full', help='experiment name')
    p.add_argument('--device', type=str, default='', help='cuda device, e.g. 0 or 0,1 or cpu')
    p.add_argument('--resume', type=str, default='', help='path to last.pt to resume training')
    # ---- Tri-Scale ablation switches ----
    p.add_argument('--no-sd', action='store_true', help='disable SD (area-adaptive regression)')
    p.add_argument('--no-pgde', action='store_true', help='disable PGDE (Gaussian map supervision)')
    p.add_argument('--no-sard', action='store_true', help='disable SARD (importance distillation)')
    # ---- Tri-Scale hyperparameters ----
    p.add_argument('--sd-strength', type=float, default=None, help='SD rescheduling strength k')
    p.add_argument('--pgde-weight', type=float, default=None, help='loss_pgde gain')
    p.add_argument('--distill-weight', type=float, default=None, help='loss_distill gain')
    p.add_argument('--pgde-grids', type=int, nargs='+', default=None, help='PGDE map grids, e.g. 64 32 16')
    return p


def apply_loss_params(model, args):
    """Mutate loss_params inside the model yaml before the criterion is built."""
    lp = model.model.yaml.setdefault('loss_params', {})
    if args.no_sd:
        lp['use_sd'] = False
    if args.no_pgde:
        lp['use_pgde'] = False
    if args.no_sard:
        lp['use_sard'] = False
    if args.sd_strength is not None:
        lp['sd_strength'] = args.sd_strength
    if args.pgde_weight is not None:
        lp['pgde_weight'] = args.pgde_weight
    if args.distill_weight is not None:
        lp['distill_weight'] = args.distill_weight
    if args.pgde_grids is not None:
        lp['pgde_grids'] = args.pgde_grids
    active = {k: lp.get(k) for k in ('use_sd', 'use_pgde', 'use_sard',
                                     'sd_strength', 'pgde_weight', 'distill_weight')}
    print(f"[TriScale] active loss_params: {active}")


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

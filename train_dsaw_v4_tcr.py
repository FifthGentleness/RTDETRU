"""从完整 DSAWv4 的最佳权重启动 TCR 微调，公共 train.py 保持原样。"""

import argparse
from pathlib import Path

from ultralytics import RTDETR


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='runs/train/DSAW-v4-TCR/weights/best.pt',
                        help='复制过来的完整 DSAWv4 权重路径')
    parser.add_argument('--name', default='', help='新训练结果的目录名')
    parser.add_argument('--device', default='0', help='CUDA 设备号，例如 0')
    parser.add_argument('--control', action='store_true', help='从同一份权重启动不加 TCR 的对照组')
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.is_file():
        parser.error(f'找不到权重：{weights}。请用 --weights 指定实际 .pt 路径。')

    model_dir = 'ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-DSAW-'
    model_yaml = model_dir + ('v4.yaml' if args.control else 'v8-TCR.yaml')
    model = RTDETR(model_yaml).load(str(weights))
    source = model.ckpt.get('ema') or model.ckpt['model']
    if source.yaml.get('loss_name') != 'dsaw_v4':
        parser.error('请选择完整 DSAWv4 的权重，不能用 IoUOnly 或已经加入 TCR 的权重。')
    if any(source.yaml.get(k) != model.model.yaml.get(k) for k in ('nc', 'scales', 'backbone', 'head')):
        parser.error('权重的网络结构与当前 FSV7 模型配置不一致。')

    if not args.control:
        # 前两轮只用 v4，第 3–7 轮逐步开启 TCR。只设置当前运行，不改原 YAML。
        model.model.yaml['loss_params'].update(start_epoch=2, full_epoch=7)

    name = args.name or ('DSAW-v4-Control-FT' if args.control else 'DSAW-v4-TCR-FT')
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=25,
                patience=0,
                batch=4,
                workers=4,
                pretrained=False,
                resume=False,
                device=args.device,
                project='runs/train',
                name=name,
                exist_ok=False,
                seed=0,
                optimizer='AdamW',
                lr0=2e-5,
                lrf=1.0,
                warmup_epochs=200,
                warmup_bias_lr=0.0,
                )

"""用指定的 RT-DETR 模型配置和初始权重启动微调。"""

import argparse
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', '--cfg', type=Path, required=True, metavar='YAML',
                        help='RT-DETR 模型 YAML 路径')
    parser.add_argument('--weights', type=Path, required=True, metavar='PT',
                        help='用于初始化微调的 .pt 权重路径')
    parser.add_argument('--data', type=Path, default='dataset/data.yaml',
                        help='数据集 YAML 路径（默认：%(default)s）')
    parser.add_argument('--name', default='', help='新训练结果的目录名')
    parser.add_argument('--device', default='0', help='CUDA 设备号，例如 0')
    args = parser.parse_args()

    for option, file, suffixes in (
        ('--model', args.model, ('.yaml', '.yml')),
        ('--weights', args.weights, ('.pt',)),
        ('--data', args.data, ('.yaml', '.yml')),
    ):
        if file.suffix.lower() not in suffixes or not file.is_file():
            parser.error(f'{option} 需要一个存在的 {" / ".join(suffixes)} 文件：{file}')

    from ultralytics import RTDETR

    model = RTDETR(str(args.model)).load(str(args.weights))

    model.train(data=str(args.data),
                cache=False,
                imgsz=640,
                epochs=50,
                patience=0,
                batch=4,
                workers=4,
                pretrained=False,
                resume=False,
                device=args.device,
                project='runs/train',
                name=args.name or f'{args.model.stem}-FT',
                exist_ok=False,
                seed=0,
                )

import warnings
warnings.filterwarnings('ignore')
import sys
import torch
from ultralytics import RTDETR
from ultralytics.utils.torch_utils import model_info

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python get_single_yaml_param_and_flops.py <yaml_path> [imgsz]")
        print("Example: python get_single_yaml_param_and_flops.py ultralytics/cfg/models/rt-detr/rtdetr-SOEP.yaml")
        print("Example: python get_single_yaml_param_and_flops.py ultralytics/cfg/models/rt-detr/rtdetr-SOEP.yaml 640")
        sys.exit(1)

    yaml_path = sys.argv[1]
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 640

    model = RTDETR(yaml_path)
    model.fuse()
    n_l, n_p, n_g, flops = model_info(model.model, imgsz=imgsz)

    print(f"\n{'='*60}")
    print(f"Config:  {yaml_path}")
    print(f"Input:   {imgsz}x{imgsz}")
    print(f"GFLOPs:  {flops:.2f}")
    print(f"Params:  {n_p:,}")
    print(f"Layers:  {n_l}")
    print(f"Gradients: {n_g:,}")
    print(f"{'='*60}")
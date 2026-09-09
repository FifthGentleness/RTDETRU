import warnings
warnings.filterwarnings('ignore')
import sys
import torch
from copy import deepcopy
from ultralytics import RTDETR
from ultralytics.utils.torch_utils import model_info, get_flops, get_flops_with_torch_profiler, de_parallel

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python get_single_yaml_param_and_flops.py <yaml_path> [imgsz]")
        print("Example: python get_single_yaml_param_and_flops.py ultralytics/cfg/models/rt-detr/rtdetr-SOEP.yaml")
        print("Example: python get_single_yaml_param_and_flops.py ultralytics/cfg/models/rt-detr/rtdetr-SOEP.yaml 640")
        sys.exit(1)

    yaml_path = sys.argv[1]
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 640

    model = RTDETR(yaml_path)

    n_l, n_p, n_g, flops = model_info(model.model, imgsz=imgsz)

    if flops == 0:
        flops = get_flops_with_torch_profiler(model.model, imgsz=imgsz)
        if flops == 0:
            try:
                m = de_parallel(model.model)
                p = next(m.parameters())
                stride = 640
                im = torch.zeros((1, 3, stride, stride), device=p.device)
                with torch.profiler.profile(with_flops=True) as prof:
                    m(im)
                flops_raw = sum(x.flops for x in prof.key_averages()) / 1e9
                imgsz_list = [imgsz, imgsz]
                flops = flops_raw * imgsz_list[0] / stride * imgsz_list[1] / stride
            except Exception as e:
                print(f"Profiler fallback failed: {e}")
                flops = 0

    print(f"\n{'='*60}")
    print(f"Config:  {yaml_path}")
    print(f"Input:   {imgsz}x{imgsz}")
    print(f"GFLOPs:  {flops:.2f}")
    print(f"Params:  {n_p:,}")
    print(f"Layers:  {n_l}")
    print(f"Gradients: {n_g:,}")
    print(f"{'='*60}")
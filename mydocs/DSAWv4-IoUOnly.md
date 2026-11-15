# DSAWv4-IoUOnly：单独验证 IoU 混合

保持 FSV7 网络、匈牙利匹配和原训练分支不变。IoUOnly 复用原 dsaw_v4 的回归逻辑，将 beta、alpha_l1 和旧版 L1 回退值 alpha 设为 0；alpha_iou 继续为 0。独立插件直接调用原生分类损失，因此无 GT 批次也保留原生 Focal Loss 的归一化。同步新 YAML、新插件和损失注册表即可，无需改 train.py 或公共 loss.py。

- VFL：保留匹配 IoU 软标签和原 VFL 正负样本权重，去掉 v4 额外的正样本增权。
- L1：保留原 cx、cy、w、h 的 L1 及损失系数，逐框权重恢复为 1。
- IoU：q 为匹配框的普通 IoU（不反传），mu=sigmoid((q−0.45)/0.05)，几何项为 mu×(1−GIoU)+(1−mu)×InnerInterpIoU。inner_ratio=0.8，动态插值不变；q_ema 不再影响损失权重。

| 组别 | VFL 额外增权 beta | L1 减权 alpha_l1 | IoU 损失 |
|---|---:|---:|---|
| 原 FSV7 | 0 | 0 | 纯 GIoU |
| DSAWv4-IoUOnly | 0 | 0 | 原 v4 混合 |
| 完整 DSAWv4 | 0.5 | 0.25 | 原 v4 混合 |

原 FSV7 对 IoUOnly 回答「混合单独有没有收益」；IoUOnly 对完整 v4 衡量恢复 v4 整套加权后的净差异，不能区分分类和 L1 各自的贡献。若训练含整批无 GT 的样本，这一对还包含默认分类归一化与原 v4 空批次归一化的差异；原 FSV7 对 IoUOnly 不受该差异影响。如果只能跑两组，优先选前一对。

## 用原 train.py 启动

以下命令在训练机的项目根目录运行。三组使用相同初始化、种子、数据、增强、学习率和训练轮数。主看同轮窗口的整体 AP50–95 和 AP75，辅看整体 AP50 和短边 <8、8–16 px 的分组指标。分组像素按 640 输入计算。也可报告同一选模规则下的最佳权重。

~~~powershell
python train.py --model "ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7.yaml" --name "FSV7-IoUAblation-GIoU" --device 1
python train.py --model "ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-DSAW-v4-IoUOnly.yaml" --name "FSV7-DSAW-v4-IoUOnly" --device 1
python train.py --model "ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-DSAW-v4.yaml" --name "FSV7-DSAW-v4-Full" --device 1
~~~

## 可选：沿用复制权重的续训方式

这只回答「检查点之后，哪种损失更有用」。纯 GIoU 与 IoUOnly 必须从同一个完整检查点的两个副本继续，沿用同一优化器状态和训练计划。不能把续训成绩和从头训练的对照当作单因素比较。

先把原检查点复制到相应新目录的 weights/last.pt。在训练机的项目根目录，把下面整段粘贴到 PowerShell，执行一次以修改副本。对不同组更换 group，分别处理各自的副本。请只加载自己信任的训练权重。

~~~powershell
@'
from pathlib import Path
import torch
from ultralytics.utils import yaml_load

group = "iou_only"  # "iou_only", "giou", "full_v4"
tags = {"iou_only": "IoUOnly", "giou": "GIoU", "full_v4": "Full"}
run = (Path("runs/train") / ("FSV7-DSAW-v4-" + tags[group] + "-Resume")).resolve()
pt = run / "weights" / "last.pt"
cfg = yaml_load("ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-DSAW-v4-IoUOnly.yaml")

ckpt = torch.load(pt, map_location="cpu", weights_only=False)
assert ckpt["epoch"] >= 0 and ckpt.get("optimizer") is not None, "Need a complete resumable checkpoint"
assert ckpt["model"].yaml.get("loss_name") == "dsaw_v4", "Start from a copy of the original v4 checkpoint"
assert ckpt["train_args"]["epochs"] > ckpt["epoch"] + 1, "No epochs left in the original plan"

for model in (ckpt["model"], ckpt.get("ema")):
    if model is None:
        continue
    y = dict(model.yaml)
    if group == "iou_only":
        y["loss_name"] = cfg["loss_name"]
        switches = ("beta", "alpha", "alpha_l1", "alpha_iou")
        y["loss_params"] = {**y["loss_params"], **{k: cfg["loss_params"][k] for k in switches}}
    elif group == "giou":
        y.pop("loss_name", None)
        y.pop("loss_params", None)
    model.yaml = y

criterion = ckpt["model"].init_criterion()
if group == "iou_only":
    assert criterion.beta == criterion.alpha_l1 == criterion.alpha_iou == 0
ckpt["train_args"].update(model=str(pt), project=str(run.parent), name=run.name, save_dir=str(run))
tmp = pt.with_suffix(".tmp")
torch.save(ckpt, tmp)
tmp.replace(pt)
print("Loss:", type(criterion).__name__)
print("Next epoch:", ckpt["epoch"] + 2, "/", ckpt["train_args"]["epochs"])
print("Output:", run)
'@ | python -
~~~

然后运行项目原入口，例如：

~~~powershell
python train.py --resume "runs/train/FSV7-DSAW-v4-IoUOnly-Resume/weights/last.pt" --device 1
~~~

resume 从 pt 内部读取模型和输出目录，不依赖外部 args.yaml。不要复制原实验的 results.csv，以免继承旧曲线或重复轮次。IoUOnly 仍显示原来的三个损失列；混合项位于 giou_loss 列，这是日志名称。因为 resume 会继承旧 best fitness，且新目录没有历史 best.pt，如果本组没超过旧最佳值，用实际同轮 last.pt 比较。

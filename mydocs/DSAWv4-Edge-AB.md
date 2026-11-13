# DSAWv4-Edge：在新目录从 v4 检查点续训

B 组保留原 DSAWv4 的 VFL、L1 减权、IoU 混合和匈牙利匹配。只给最后一层常规查询的匹配框加四边 Smooth L1：640 输入上 sqrt(w×h)<32 px、短边 [8,16) px、匹配 IoU [0.60,0.85)。四边误差分别按 GT 宽、高归一化，Smooth L1 beta=0.05，系数 0.7，分母是该批最后一层的全部匹配对数。编码器、辅助解码器和去噪分支不加此项。

## 在训练机器准备 B 组

同步独立插件、损失注册表和 B 组模型 YAML。使用项目原有的 train.py；不需要专用训练脚本，也不修改原生 loss.py。

在训练机的仓库根目录运行下面的一次性配置操作。先建立新实验目录下的 weights 子目录，把原 v4 的完整检查点复制成其中的 last.pt；改好代码第一行路径。源权重应保有 optimizer 和 epoch（被裁剪的推理权重不能恢复优化器）。以下只重写新目录里的副本。

~~~python
from pathlib import Path
import torch
from ultralytics.utils import yaml_load

pt = Path("runs/train/DSAWv4-Edge-B/weights/last.pt").resolve()
cfg = yaml_load("ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-DSAW-v4-Edge.yaml")
ckpt = torch.load(pt, map_location="cpu", weights_only=False)
assert ckpt["epoch"] >= 0 and ckpt.get("optimizer") is not None, "需要含优化器的完整检查点"
assert ckpt["model"].yaml.get("loss_name") == "dsaw_v4", "副本应来自原 DSAWv4"

for model in (ckpt["model"], ckpt.get("ema")):
    if model is not None:
        model.yaml = {
            **model.yaml,
            "loss_name": cfg["loss_name"],
            "loss_params": {
                **model.yaml["loss_params"],
                **{k: v for k, v in cfg["loss_params"].items() if k.startswith("edge_")},
            },
        }

run = pt.parent.parent
ckpt["train_args"].update(save_dir=str(run), project=str(run.parent), name=run.name)
# 可选：只想追加 25 轮时启用；默认沿用检查点原有总轮数和学习率设置。
# ckpt["train_args"]["epochs"] = ckpt["epoch"] + 1 + 25
assert ckpt["train_args"]["epochs"] > ckpt["epoch"] + 1, "原计划已结束，请设置新的总轮数"
tmp = pt.with_suffix(".tmp")
torch.save(ckpt, tmp)
tmp.replace(pt)
print("B 组检查点：", pt)
print("下一轮：", ckpt["epoch"] + 2, "总轮数：", ckpt["train_args"]["epochs"])
~~~

在训练机的仓库根目录，用原入口恢复：

~~~shell
python train.py --resume runs/train/DSAWv4-Edge-B/weights/last.pt --device 0
~~~

关键点：当前仓库从 .pt 的 model.yaml 读取损失配置，并用 train_args.save_dir 决定写到哪里。只复制权重、只改 YAML 文件、或命令里另传 --model / --name 都不能替代上述两处配置。启动后检查日志里的续训起始轮和输出目录。若拿同一检查点做 A/B 对照，A 也用新目录副本、只改输出目录（以及与 B 相同的总轮数，如有改动），保留 dsaw_v4。这样两组继承同样的优化器、EMA 和训练计划；它和先前提出的“重新创建优化器、低学习率微调”是不同实验。若 B 没超过历史 best fitness，新目录可能只生成 last.pt；对照时优先用两组同轮的 last.pt，并比较后几轮的 AP50–95。

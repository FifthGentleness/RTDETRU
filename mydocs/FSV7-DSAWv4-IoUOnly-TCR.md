# FSV7：在 DSAWv4-IoUOnly 上添加 TCR

新损失从 `dsaw_v4_iou_only` 继承，保留原生 VFL（包括整批无 GT 时的分类路径）、原生 L1，以及 v4 的 GIoU / InnerInterpIoU 混合。`beta`、`alpha`、`alpha_l1`、`alpha_iou` 都为 0。TCR 沿用[原方案](FSV7-DSAWv8-TCR方案.md)的排序和跨阈值定位规则，只在训练时加到末层常规 query；不改匹配器、编码器辅助监督、DN、AIFI、解码器和推理打分。

| 组别 | 基础损失 | 排序系数 | 跨阈值纠框系数 |
|---|---|---:|---:|
| A · IoUOnly | IoUOnly | 0 | 0 |
| B · IoUOnly-TCR-RankOnly | IoUOnly | 0.25 | 0 |
| C · IoUOnly-TCR | IoUOnly | 0.25 | 0.50 |

第 1–15 轮与 IoUOnly 一致，第 16–30 轮线性开启 TCR。用 A→B 看排序贡献，用 B→C 看纠框有没有额外收益。只跑两组时，A→C 测整体 TCR。以末层 mAP50–95 为主，结合 AP75 和短边 8–16 px 组的末层同类 R75，观察 AP50 是否下滑。旧 `dsaw_v8_tcr` 配置和旧权重仍对应完整 v4，不能当这次的对照。

在仓库根目录沿用 `train.py --model`。PowerShell：

```powershell
$stem = 'ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-DSAW-v4-IoUOnly'
python train.py --model "$stem.yaml" --name FSV7-DSAWv4-IoUOnly-Control --device 0
python train.py --model "$stem-TCR-RankOnly.yaml" --name FSV7-DSAWv4-IoUOnly-TCR-RankOnly --device 0
python train.py --model "$stem-TCR.yaml" --name FSV7-DSAWv4-IoUOnly-TCR --device 0
```

如果想直接用已训练的 IoUOnly 权重，A/B/C 都从训练机上的**同一份 IoUOnly 检查点**载入权重，开新的优化器和输出目录。用 `RTDETR(各组 YAML).load(同一份 IoUOnly 权重)` 后再 `.train(...)`，不要直接对 IoUOnly 的 `.pt` 用 `--resume` 来开启 TCR，后者会恢复旧 YAML。若采用 25 轮短筛，B/C 可在 `.train()` 前设置 `model.model.yaml['loss_params'].update(start_epoch=2, full_epoch=7)`，A 保持纯 IoUOnly，各组使用相同的训练设置。中断后继续同一组则用该组自己的 `last.pt` 和 `train.py --resume`。

验证入口：`python tools/verify_dsaw_v8_tcr.py -v`。新旧基线都检查关闭 TCR 时的逐项损失、空 GT、匹配调用和末层梯度；新配置额外检查网络图和基础损失参数与 IoUOnly 完全一致。

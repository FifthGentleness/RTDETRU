# Tri-Scale Loss（三尺度损失协同）设计与使用说明

> **v1.1 修订（2026-10-08）**：首轮训练发现前期收敛过慢（epoch6 mAP50-95 0.031 vs baseline 0.116，
> giou 梯度膨胀 ~3.7x）。修订：sd_strength 1.0→0.5、sd_giou_max 4.0→2.0、新增
> ux_warmup_iters=8000（PGDE/SARD 权重在前 ~5 个 epoch 线性 ramp-in，val 的 no_grad
> 前向不推进计数）。已训的早期 run 建议弃掉重跑。

> 生成日期：2026-10-08 ｜ 基线：rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7（V7）

## 1. 总体设计

三个**正交的、纯损失侧**组件，零推理开销（模型结构与 V7 完全一致：17.6M 参数 / 75.3 GFLOPs）：

| 维度 | 组件 | 解决的问题 | 思想来源 |
|------|------|-----------|---------|
| 回归侧（梯度质量） | **SD**：面积自适应 L1 调度 + GIoU 面积增益 | 小目标位置梯度被 w/h 项淹没、IoU 梯度弱 | AAAI 2025, Scale-based Dynamic Loss (Yang et al., 39(9):9202-9210) |
| 监督密度侧 | **PGDE**：无参数多尺度高斯分布图对齐 | 小目标正样本稀疏 → 有效监督稀疏 | CVPR 2025, Feature Information Driven Position Gaussian Distribution Estimation (Bian et al., pp.30376-30386) |
| 监督质量侧 | **SARD**：结构重要性加权的跨层自蒸馏 | 小目标区域的监督被均匀稀释 | CVPR 2026, Structure-Aware Representation Distillation (Liu et al., pp.34775-34783) |

### 与已有 `sd` 插件（loss_plugins/sd_loss.py）的关系
- 已有 `sd` 插件 = AAAI 2025 SDIoU 的忠实移植，**只替换 GIoU 项**（β 调度在 CIoU 内部）。
- 本 TriScale 的 SD 维度 = **L1 项内部**的 x/y–w/h 调度 + GIoU 的面积自适应增益，**不替换 IoU 度量**。
- 两者互斥（都作用于回归项），论文消融可对比：`baseline → +SDIoU(已有) → +TriScale(本方案)`。

## 2. 文件清单

| 文件 | 作用 |
|------|------|
| `ultralytics/models/utils/tri_scale_loss.py` | 核心：`TriScaleDetectionLoss(RTDETRDetectionLoss)` |
| `ultralytics/utils/loss_plugins/__init__.py` | 注册表新增 `'tri_scale'` 一行 |
| `ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-TSL.yaml` | 模型配置（结构同 V7 + loss_name/loss_params） |
| `train_triscale.py` | 训练脚本（CLI 消融开关） |

接入机制：`RTDETRDetectionModel.init_criterion()` 读取 yaml 的 `loss_name: tri_scale` → `loss_plugins.build_loss` → `TriScaleDetectionLoss`。**tasks.py 与原损失文件零改动**；不带 `loss_name` 的 yaml 走原路径（已回归测试）。

## 3. 各组件细节

### 3.1 SD（`_get_loss_bbox` 重写）
- 归一化面积 `a = w·h`，对数调度 `t = clip((log a − log a_min)/(log a_max − log a_min), 0, 1)`（0=极小，1=大）。
- L1：小目标 x/y 权重↑（`w_pos = 1 + k(1−t)`），大目标 w/h 权重↑（`w_scale = 1 + k·t`）。
- GIoU：逐对增益 `g = clamp((a_ref/a)^γ, 1, g_max)`，小目标 IoU 梯度弱的问题被面积增益补偿。
- 锚点默认对应 640 输入：a_min≈8px 框、a_max≈180px 框、a_ref≈80px 框。

### 3.2 PGDE（`_get_loss_pgde`，无参数设计）
- 把每个 GT 框栅格化为幅度为 1 的高斯斑（σ = 0.25·max(w,h)，下限 2.5 格），叠加成混合分布图；匹配到的预测框同法栅格化。
- 在 64/32/16 三个栅格尺度上做 soft-Dice 对齐 → 稀疏 query 监督变成稠密空间监督。
- 与原论文差异（论文中需写明）：原 PGDE 用信息熵图 + 辅助增强头；本实现是**无辅助参数的查询级分布对齐**，梯度直接回流到框参数与特征，推理零开销。

### 3.3 SARD（`_get_loss_distill`）
- Teacher = 最后一个 decoder 层（detach，分类 logits 过温度 T=2）；Student = 全部辅助层（含 encoder one2one 层）。
- 损失 = 重要性加权的 soft-BCE（分类）+ smooth-L1（框），按匹配对逐一加权。
- 重要性图（每 GT，归一化到 [0.3, 1]）：边界显著性 s1（越小越边界主导）+ 几何复杂度 s2（|log 宽高比|）+ 局部结构变化 s3（GT 高斯图上环绕 8 采样点的 std/mean，与 PGDE 图共享先验）。

## 4. 使用

```bash
# 完整 TriScale
D:\Anaconda\envs\RTDETR\python.exe train_triscale.py --device 1 --name TSL_full

# 消融
python train_triscale.py --no-sard              # SD + PGDE
python train_triscale.py --no-pgde              # SD + SARD
python train_triscale.py --no-pgde --no-sard    # 仅 SD
python train_triscale.py --pgde-weight 3.0 --distill-weight 1.5   # 调权重
```

## 5. 推荐消融表（论文 Table 结构）

| # | SD | PGDE | SARD | 说明 |
|---|----|----|----|------|
| 0 | – | – | – | V7 baseline |
| 1 | ✔ | – | – | 回归侧单独 |
| 2 | – | ✔ | – | 监督密度单独 |
| 3 | – | – | ✔ | 蒸馏单独 |
| 4 | ✔ | ✔ | – | 双维度 |
| 5 | ✔ | ✔ | ✔ | 完整 TriScale |
| 6 | ✔(SDIoU 替换) | ✔ | ✔ | 与已有 sd 插件对比变体（可选） |

补充消融：`sd_strength ∈ {0.5, 1, 2}`；`pgde_grids` 单尺度 vs 三尺度；`sard_imp_min ∈ {0, 0.3, 0.5}`；分 APs/APm/APl 报告（收益应集中在 APs）。

## 6. 风险与调参提示
- PGDE 的 `pgde_sigma_scale` 是最敏感超参：σ 过大 → 图糊、判别性下降；过小 → 梯度稀疏。建议先扫 {0.2, 0.25, 0.35}。
- 蒸馏权重过大可能抑制 GT 监督：若 `loss_distill` 数值量级明显大于 `loss_class`，先降 `distill_weight` 至 0.5。
- 训练初期匹配质量差，PGDE 图对齐噪声大，属正常现象；可观察 20 epoch 后 `loss_pgde` 是否稳定下降。
- 训练显存/耗时增加 <5%（高斯图计算在 N≤数百框、G≤64 下开销极小）。

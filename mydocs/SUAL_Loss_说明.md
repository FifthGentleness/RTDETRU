# SUAL：Scale-Uncertainty-Aware Localization 损失设计说明

> 生成日期：2026-10-08 ｜ 基线：rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7（V7）
> 结构与 V7 完全一致（17.6M 参数 / 75.3 GFLOPs），**纯损失侧改动，零推理开销**

## 1. 总公式

按匹配对 i（Hungarian 匹配后的 query-GT 对，xywh 归一化）：

```
L_SUAL = Σᵢ w_u,i · (1 − αᵢ · IoUᵢ)      ← 尺度×不确定性 加权 IoU 项（替换原 GIoU 项）
       + λ_L · L_loc_polar               ← SLS 极坐标中心位置惩罚（替换原 L1 项）
       + λ_c · L_SQCL                    ← UGS 式分类定位 CE（新增）
       + μ  · L_UM                       ← 跨层方差最小化（新增）

w_u,i = 1 + ρ·ûᵢ     （ûᵢ ∈ [0,1]：跨 decoder 层预测框 std 的批内归一化）
αᵢ    = (min(A_p,A_gt) + dis) / (max(A_p,A_gt) + dis)，dis = ((A_p−A_gt)/2)²
```

L_cls / 匈牙利匹配代价 / DN 分支管道全部继承原实现，未改动。

## 2. 两个来源的映射（诚实区分：忠实移植 vs 思想适配）

### [1] Scale 分支 —— SLS（CVPR 2024, Liu et al., pp.17491-；官方 BIT-RuiLiu/MSHNet）→ **忠实移植**
| SLS 原文（分割掩码形式） | SUAL 框形式 |
|---|---|
| α 权重：基于 \|A_p\| 与 \|A_gt\| 的 min/max+方差（Eq.3），尺度差距大→α 小→损失大 | A = 框面积 w·h；**梯度经 A_p 回流**，主动缩小预测/GT 尺度差 |
| L_L 位置惩罚：中心点极坐标形式（角度项 4/π²·Δθ² + 径向比 min(r)/max(r)），比 L1/L2 更具区分性 | 完全同式，中心取预测/GT 框中心，r 为到图像原点距离 |
| 1 − α·SoftIoU | 1 − α·GIoU（默认；可切 'iou'/'ciou'） |

⚠️ 论文表述注意：SLS 原文是红外小目标分割任务。你引用时写"将 SLS 的尺度权重与极坐标位置惩罚移植到框回归形式"，属于合理的跨任务移植，与 RT-DETR/VisDrone 同领域论文无撞车。

### [2] Uncertainty 分支 —— UGS（ICCV 2025, Sun et al., pp.8407-8417）→ **思想适配（非复制）**
UGS 原文需要专用分类式定位头（非均匀离散标签）+ UR 扰动精修模块，属于架构改动。SUAL 在**零参数**前提下适配其三个核心思想：

| UGS 原文 | SUAL 适配 |
|---|---|
| 定位重表述为分类、离散化标签→有界、置信度驱动梯度 | **SQCL**：坐标 → 可微分数值 bin 索引（γ 幂映射，γ>1 时近 0 处 bin 更密，即 UGS 的非均匀量化思想）→ 核 softmax 分布 → 与 GT 分布做 CE。梯度有界且远离目标时饱和（已验证：CE 5.67→22.68 饱和） |
| UM：不确定性最小化 | 跨 decoder 层匹配框预测方差最小化（面积反比加权，小目标优先稳定） |
| UR：不确定性引导精修（扰动式模块） | **仅借思想**：不确定性驱动的 IoU 项加权（ρ·û）。原 UR 模块需架构支持，未复制——论文中必须如实写明这是 UR 的损失级替代 |

⚠️ 撞车提示：UGS 原文在 VisDrone 上有 DINO-5scale +2.6 AP 的结果。你的差异化必须写清：(a) 检测器是 RT-DETR；(b) 无分类定位头/UR 模块，是参数-free 的分布 CE + 跨层方差实现；(c) 与 SLS 融合为双分支形式是其原文没有的。

## 3. 文件清单

| 文件 | 作用 |
|---|---|
| `ultralytics/utils/loss_plugins/sual_loss.py` | 核心 `RTDETRDetectionLossSUAL` |
| `ultralytics/utils/loss_plugins/__init__.py` | 注册 `'sual'` |
| `ultralytics/cfg/models/rt-detr/...-SUAL.yaml` | 配置（结构同 V7 + loss_name: sual） |
| `train_sual.py` | 训练脚本（CLI 消融/调参） |

## 4. 使用

```powershell
D:\Anaconda\envs\RTDETR\python.exe train_sual.py --device 1 --name SUAL_full

# 消融
python train_sual.py --sqcl-weight 0 --um-weight 0            # 仅 SLS 分支
python train_sual.py --rho 0 --um-weight 0                    # SLS + SQCL（去不确定性加权/UM）
python train_sual.py --sqcl-weight 0                          # SLS + UM
python train_sual.py --nonuniform 2.0                         # UGS 非均匀 bin
python train_sual.py --iou-type iou                           # 严格 SLS 原文 IoU 形式
```

## 5. 推荐消融表

| # | Scale 分支 | SQCL | UM/ρ | 说明 |
|---|---|---|---|---|
| 0 | – | – | – | V7 baseline |
| 1 | ✔ | – | – | SLS only（λ_c=0, μ=0, ρ=0） |
| 2 | – | ✔ | – | 仅 SQCL（需另出变体，见备注） |
| 3 | ✔ | ✔ | – | + 分类定位 |
| 4 | ✔ | ✔ | ✔ | 完整 SUAL |
| 5 | ✔ | ✔ | ✔ | + nonuniform γ=2.0 |
| 6 | – | ✔ | ✔ | 纯 uncertainty 分支（对照） |

分 APs/APm/APl 报告；另报 `loss_sual_um` 收敛曲线（方差抑制的直接证据）。

## 6. 风险与调参
- **`loss_sual_ce` 有常数熵下界**（收敛后 ≈ 每坐标 ~1.9，无梯度，无害），但绝对值较大属正常；若训练初期压过主损失，降 `sqcl_weight` 至 0.5。
- `sqcl_sigma` 建议扫 {0.5, 1.0, 2.0}：σ 小→分布锐、梯度强、饱和快。
- `sqcl_nonuniform=2.0` 时 v→0 处索引导数放大（设计使然：小框更精细），若不稳定回退 1.0。
- 极坐标位置项的"原点在图像左上角"继承自 SLS 原文；若担心图像位置偏置，可消融 `sls_loc_weight=0` 对比。
- SUAL 与 `tri_scale`/`sd` 插件互斥（都替换回归项），一次实验只挂一个 loss_name。

# 损失函数全解析：原生 RT-DETR 与 DSAW 系列（v1–v6）

> 生成日期: 2026-10-28
> 代码位置:
> - 原生: `ultralytics/models/utils/loss.py` (`DETRLoss` → `RTDETRDetectionLoss`)
> - v1: `ultralytics/utils/loss_plugins/dsaw_loss.py`
> - v2: `ultralytics/utils/loss_plugins/dsaw_v2_loss.py`
> - v3: `ultralytics/utils/loss_plugins/inner_interpiou_dsaw3_loss.py` (+ `inner_interpiou_loss.py`)
> - v4: `ultralytics/utils/loss_plugins/dsaw_v4_loss.py`
> - v5: `ultralytics/utils/loss_plugins/dsaw_v5_loss.py`
>
> 继承链: `RTDETRDetectionLoss` → `RTDETRDetectionLossDSAW(v1)` → `…V2` → { `…InnerInterpIoUDSAW3(v3)`, `…V4` } → `…V5`
> 每一版只做增量修改，且所有新参数归零后可逐位退化为原生损失。

---

## 0. 公共背景（所有损失共享的流程）

六个损失函数共享同一条数据流，区别只发生在"匹配之后的监督方式"上：

```
预测: pred_bboxes [b, 300, 4](xywh归一化), pred_scores [b, 300, nc]
  │
  ├─① HungarianMatcher: 300 个 query 与该图 GT 一对一最优指派
  │     代价 C = 2·C_cls + 5·C_L1 + 2·C_GIoU      (v1~v4 匹配器与原生完全相同)
  │
  ├─② 取出匹配对 (B_i, B_gt,i), i = 1..N
  │
  ├─③ 分类支路: VFL, 对全部 300×nc 个 (query,类) 位置计算
  ├─④ 回归支路: L1 + IoU 系, 只对 N 个匹配对计算
  │
  ├─⑤ aux loss: 前 L-1 层 decoder 输出重复 ③④ (深监督)
  └─⑥ dn loss: 去噪 query 分支, 答案已知, 不需要匹配 (postfix='_dn')
```

约定: 坐标均为**归一化 xywh**（除以图像边长，640px 图上 1px ≈ 0.0015625）。

---

## 1. 原生 RT-DETR 损失（基线）

### 1.1 总体形式

```
L = 1·L_cls(VFL) + 5·L_bbox(L1) + 2·L_giou(GIoU)
```

loss_gain = `{class: 1, bbox: 5, giou: 2}`。

### 1.2 匹配器（HungarianMatcher, 逐图线性指派）

```
C(q, g) = 2·C_cls(q,g) + 5·C_L1(q,g) + 2·C_GIoU(q,g)
```

| 符号 | 含义 |
|---|---|
| C_cls | focal 式分类代价: focal α=0.25, γ=2（use_fl 时） |
| C_L1 = \|B_q − B_g\|₁ | 逐坐标 L1 代价 |
| C_GIoU = 1 − GIoU | 广义 IoU 代价 |

匹配端对目标尺度完全无感——这是整个 DSAW 系列针对的结构性缺口。

### 1.3 分类支路: Varifocal Loss

软标签构造:

```
target(q,c) = IoU(B_matched, B_gt)   若 (q,c) 是匹配正样本位置
target(q,c) = 0                      其余全部位置
```

VFL 计算（α=0.75, γ=2.0）:

```
weight = 0.75·σ(p)²·(1−label) + q·label
L_cls  = Σ BCE_with_logits(p, target)·weight    (类维均值, query 维求和)
       ÷ max(num_gts,1)/nq × 1
```

| 符号 | 含义 |
|---|---|
| p | 分类 logits 的 sigmoid |
| label | one-hot 正样本指示 |
| q | 匹配对 IoU —— **同时是 soft label 和正样本权重** |
| 0.75·σ(p)² | 负样本降权（VFL 特性: 只降负样本，不降正样本） |

关键性质: 正样本权重只取决于"匹配质量"，与目标大小无关。小目标 IoU 难做高 → 分类权重小 → 梯度弱，天然处于劣势。

### 1.4 回归支路: L1 + GIoU

```
L_bbox = 5 · (1/N)·Σ_i |B_i − B_gt,i|₁
L_giou = 2 · (1/N)·Σ_i (1 − GIoU(B_i, B_gt,i))
```

| 符号 | 含义 |
|---|---|
| L1 项 | 梯度恒定（与框大小无关）; 小框相对过强 |
| GIoU | IoU − 最小外接框空隙/外接框面积; 补上无重叠时的梯度 |
| N | 当前 batch 匹配对总数 |

两个支路均逐样本等权，无尺度感知、无质量感知。

### 1.5 辅助损失与去噪

- **aux**: 前 L−1 层 decoder 输出各算一遍三件套并累加（共享最后一层的匹配结果）;
- **_dn**: 去噪 query 分支，对应关系构造时已知，不需要匹配，同样三件套。

---

## 2. DSAW-v1（Dynamic Scale-Aware Weighting）

> 设计哲学: **零新增损失项、只调权重**。度量（L1/GIoU/VFL 目标）与匹配器一个都不动。

### 2.1 两个调度因子

**尺度因子 lam**（GT 侧常数，无梯度路径）:

```
lam_i = clip(1 − √(w_i·h_i) / τ, 0, 1)
```

| 符号 | 含义 | 取值 |
|---|---|---|
| w_i, h_i | GT 框归一化宽高 | — |
| τ (tau) | 尺度分界: √(wh) ≥ τ 时 lam=0 | 1/32 ≈ 0.03125（= 32px） |

数值示例（640px 图）: 32px→0, 16px→0.2, 8px→0.6, 4px→0.8。**干预强度随目标变小线性增大**。

**动态因子 eta**（训练状态，自动衰减）:

```
q_ema ← m·q_ema + (1−m)·mean(IoU_matched)     (detach, 纯统计)
eta   = 1 − q_ema
```

| 符号 | 含义 | 取值 |
|---|---|---|
| m (eta_momentum) | EMA 动量, 时间常数≈1000 iter | 0.999 |
| q_ema_init | q_ema 初始值 | 0.5（→ 初始 eta=0.5） |
| eta | 质量缺口: 训练早期大, 收敛后→0 | 动态 |

### 2.2 三个支路

**分类（只放大正样本）**:

```
w_cls,i = 1 + β·lam_i·eta          β = 0.5
VFL 正样本权重: q_i → q_i·w_cls,i     (负样本权重不变)
```

| 符号 | 作用 |
|---|---|
| β | 放大强度上限调节（0=关闭）; 8px 框早期约 ×1.3 |
| lam·eta | 小目标 × 训练早期 → 放大最多; 主攻召回/mAP50 |

**几何（L1 与 GIoU 共用一个软化权重）**:

```
w_geo,i = 1 − α·lam_i               α = 0.25     (注意: 不乘 eta)
L_bbox = 5·(1/N)·Σ L1_i·w_geo,i
L_giou = 2·(1/N)·Σ (1−GIoU_i)·w_geo,i
```

| 符号 | 作用 |
|---|---|
| α | 温和化强度（上限 0.5）; L1 梯度恒定, 小框相对过冲, 最多削 25% |

### 2.3 与原生损失的区别（仅此三处）

1. VFL 正样本权重乘 `1 + β·lam·eta`;
2. L1 逐样本乘 `1 − α·lam`;
3. GIoU 逐样本乘 `1 − α·lam`。

退化保证: β=0 且 α=0 → 逐位等于原生。新增超参: β, α, τ, m, q_ema_init, dynamic_eta。
新增运行时状态: q_ema 一个标量。

### 2.4 实验结论（基座 DSAWACGAv5-P2SPDOKMFSV7, VisDrone）

同 epoch 对齐全程领先基座: mAP50 +0.5~1.3pt, mAP50-95 +0.4~0.7pt（ep10~75）, 75ep 后差距未收窄。**系列中唯一确定性正向的主损失调度**。注意 eta 动态衰减意味着优势集中在前期, 终值需跑满 350ep 验证。

---

## 3. DSAW-v2（聚焦尺度 + 动态几何退火）

在 v1 基础上改两处:

### 3.1 改动一: 聚焦指数 lam_pow

```
lam_i = clip(1 − √(w_i·h_i)/τ, 0, 1) ** lam_pow      lam_pow = 2.0
```

| 符号 | 含义 |
|---|---|
| lam_pow | 幂次聚焦: >1 时权重向真正微小的目标集中（16px: 0.2→0.04, 8px: 0.6→0.36） |

效果: 中等小目标（16~24px）的干预被大幅压缩, 干预集中于 ≤10px 的极小目标。

### 3.2 改动二: 几何权重也退火（dynamic_geo）

```
w_cls = 1 + β·lam·eta                    (同 v1)
w_geo = 1 − α·lam·geo_eta                geo_eta = eta (dynamic_geo=true) else 1
```

| 符号 | 含义 |
|---|---|
| dynamic_geo | true=几何软化随训练退火（早期温和、后期恢复全强度）; false=恒定（退回 v1 行为） |

设计逻辑: v1 的几何软化是恒定的, 训练后期小目标定位已改善时仍被削 25% 几何梯度, 不合理; v2 让刹车也随质量缺口收敛。

### 3.3 超参汇总

| 参数 | 默认 | 作用 |
|---|---|---|
| lam_pow | 2.0 | 尺度聚焦指数（=1 退回 v1） |
| dynamic_geo | true | 几何软化是否随 eta 退火 |

### 3.4 实验结论

ep20 与 v1 打平（0.1913 vs 0.1897）, ep29 反而低于 v1（0.2197 vs 0.2219）。增益在噪声量级, **lam² 过度偏置可能在中期反噬**。单独不构成有效改进, 但其机制被 v3/v4/v5 继承。

---

## 4. DSAW-v3（InnerInterpIoU-DSAW3: 换几何度量）

> 定位: DSAW-v2 权重 + 把 GIoU 整体替换为 Inner-InterpIoU。loss-only, 架构不变。

### 4.1 权重部分（与 v2 完全一致）

```
lam = clip(1 − √(wh)/τ, 0, 1)^2
eta = 1 − q_ema
w_cls = 1 + β·lam·eta
w_geo = 1 − α·lam·eta
```

### 4.2 几何度量: Inner-InterpIoU

```
α_i = 1 − IoU_i                    (逐样本动态插值系数, detach)
B_a = α_i·B_i + (1−α_i)·B_gt,i     (预测框向 GT 插值的辅助框)

L_II = (1 − IoU(B_i, B_gt))
     + (1 − IoU(B_a, B_gt))        (主项: Dynamic-InterpIoU)
     + IoU(B_i, B_gt) − IoU_inner(r)   (辅助项: Inner-IoU)
```

| 符号 | 含义 | 取值 |
|---|---|---|
| α_i (dynamic_interp) | 插值系数随 IoU 动态变化: IoU 低 → B_a 更靠近 GT → 梯度更强 | true |
| interp_alpha | 静态插值系数（dynamic_interp=false 时） | 0.99 |
| r (inner_ratio) | 双方框以中心为准缩放比例; r<1 边界监督更锐利 | 0.8 |
| IoU_inner(r) | 缩放后辅助框之间的 IoU | — |

机制: 主项给低 IoU 对更强的拉力（冲 mAP50/召回）; 辅助项 r=0.8 缩小框 → 对边界误差更敏感（适合小目标精修）。

### 4.3 超参汇总

| 参数 | 默认 | 作用 |
|---|---|---|
| inner_ratio | 0.8 | Inner 框缩放比（消融格点 0.7/0.8/1.0/1.25; =1 退化为纯 D-InterpIoU） |
| dynamic_interp | true | 插值系数是否逐样本动态 |
| interp_alpha | 0.99 | 静态插值系数（动态关闭时使用） |

### 4.4 实验结论

ep29: mAP50 0.3706（同期最好, +0.7pt vs 基座）, 但 mAP50-95 0.2167 反而低于 v2（0.2197）。**典型 mAP50/mAP50-95 trade-off**: 低 IoU 强拉提升粗定位, 但缺少高 IoU 精修 → 精度受损。冲 mAP50/AR 可保留, 冲 mAP50-95 不划算。此短板直接催生了 v4 的分段设计。

---

## 5. DSAW-v4（分段几何 + 解耦权重）—— 系列 loss 端最优

### 5.1 权重部分

继承 v2 的 lam（含 lam_pow=2）与 eta, 分类支路不变:

```
w_cls,i = 1 + β·lam_i·eta
```

几何权重**解耦为两个**:

```
w_geo_l1,i  = 1 − α_l1·lam_i·geo_eta        α_l1 = 0.25
w_geo_iou,i = 1 − α_iou·lam_i·geo_eta       α_iou = 0  → 恒为 1
```

| 符号 | 含义 | 设计理由 |
|---|---|---|
| α_l1 | L1 支软化强度 | L1 梯度恒定, 小框需要刹车（保留 v1/v2 机制） |
| α_iou | IoU 支软化强度 | IoU 型损失梯度天然 ∝ 1/面积, 已尺度自适应; 再软化 = 二次削弱 → v3 高 IoU"梯度真空"的根源。置 0 = 放行精修梯度 |

### 5.2 几何度量: IoU 门控的分段混合（核心创新）

```
μ_i = sigmoid( (IoU_i^detach − c) / T )
L_geo,i = μ_i·(1 − GIoU_i) + (1 − μ_i)·InnerInterpIoU_i
```

| 符号 | 含义 | 取值 |
|---|---|---|
| IoU_i^detach | 匹配对 IoU（detach, 门控不回传梯度） | — |
| c (blend_center) | 切换点: IoU≈0.45 时 μ=0.5 | 0.45 |
| T (blend_temp) | sigmoid 温度（过渡平滑度; IoU 0.3~0.6 之间平滑过渡） | 0.05（下限 1e-4） |
| μ→0 (低 IoU) | 用 Inner-InterpIoU: 强拉, 保留 v3 的 mAP50/召回优势 | — |
| μ→1 (高 IoU) | 用 GIoU: 干净精修, 无内框膨胀偏置, 修 v3 短板 | — |

### 5.3 Inner-InterpIoU 子项超参

同 v3: inner_ratio=0.8, dynamic_interp=true, interp_alpha=0.99。

### 5.4 超参汇总（新增 vs v2）

| 参数 | 默认 | 作用 |
|---|---|---|
| alpha_l1 | None→继承 alpha(0.25) | L1 支软化强度 |
| alpha_iou | 0.0 | IoU 支软化强度（0=关闭, 解耦的关键） |
| blend_center | 0.45 | 分段切换点（1.5→纯 v3; 0→纯 GIoU/v2 几何） |
| blend_temp | 0.05 | 切换平滑度 |
| inner_ratio | 0.8 | Inner 框缩放比 |
| dynamic_interp | true | 插值系数动态 |
| interp_alpha | 0.99 | 静态插值系数 |

### 5.5 退化检查

| 参数组合 | 等价 |
|---|---|
| blend_center=1.5 | μ≈0 → 纯 v3 几何 |
| blend_center=0 | μ≈1 → 纯 v2 几何（GIoU） |
| β=0, α_l1=0, blend_center=0 | 逐位等于原生基线 |

### 5.6 一句话总结

> **分类**: 小目标放大（β·lam·η）冲召回; **L1**: 小目标刹车（α_l1·lam·η）防过冲; **IoU**: 低 IoU 段 InnerInterpIoU 强拉、高 IoU 段 GIoU 精修（μ 分段）且不做尺度软化——三个通道各管一段, 互不重复削弱。

### 5.7 实验结论

早期信号全系列最好: ep10 mAP50 +1.4pt / mAP50-95 +0.8pt vs 基座, ep15 mAP50-95 0.1770（同期最高）。当前只训了 15ep, **最高优先级动作: 续训到 ≥75ep 与 v1 对照, 再决定是否跑满 350ep**。

---

## 6. DSAW-v5（v4 + 匹配端 NWD）—— 已验证负收益

### 6.1 设计动机

微小 GT 与所有 query 的 GIoU≈0 → cost_giou 几乎打平 → Hungarian 指派在 iter 间翻转（纯梯度噪声）。NWD（Normalized Wasserstein Distance）对无重叠小框仍随中心/尺寸偏移平滑变化, 想用它给匹配器一个可区分、抗翻转的信号。

### 6.2 损失分支

**完全继承 v4, 一字不改**（分类权重、L1 刹车、分段几何全部相同）。

### 6.3 匹配端改动（唯一改动）

把匹配器替换为 `DSAWV5HungarianMatcher`, GIoU 代价改为尺度门控的 GIoU/NWD 混合:

```
lam_j = clip(1 − √(w_j·h_j)/τ, 0, 1)^lam_pow      (GT 侧, 与 loss 端同一门控)
w_j   = nwd_weight · lam_j
cost_giou = 1 − ( (1−w_j)·GIoU + w_j·NWD )
C = 2·C_cls + 5·C_L1 + 2·cost_giou                (cost_gain 与基线完全一致)
```

NWD 定义（`wasserstein_loss`, 归一化坐标）:

```
NWD = exp( −√((x1−x2)² + (y1−y2)²) / (2·nwd_const) ) 的 2-Wasserstein 形式
```

| 符号 | 含义 | 取值 |
|---|---|---|
| nwd_const | Wasserstein 常数（控制 NWD 随距离衰减的速度） | 0.02 |
| nwd_weight | lam=1（最小目标）时的最大混合强度 | 1.0 |
| lam_j | 与 loss 端同源的尺度门（τ=1/32, 平方） | 设计文档称"assignment-supervision consistency" |

退化保证: nwd_weight=0 或 τ=0 → lam=0 → 匹配器逐位等于基线（= v4）。

### 6.4 实验结论: 负收益, 建议放弃

v5 − v4（干净的匹配端净效应）:

- ep2~4 微弱领先（+0.001~0.002, 噪声级）;
- **ep5 起连续 11 个 epoch 全面落后, 差距从 −0.003 扩大到 −0.005**（mAP50 同样 −0.005 左右）;
- v5 到 ep45 仍略低于 v1（−0.0015）——v4 loss + NWD matcher 连 v1 的原始 loss 都没追上。

失败机理分析:

1. **lam² 门控使干预几乎失效**: 16px 框权重仅 4%, 只有 ≤8px 框拿到 >36% 权重。VisDrone 主体是 10~30px 框, 恰落在"太弱基本无作用"区间;
2. **前提站不住**: 匹配 cost 里 C_cls（2×）和 C_L1（5×）本就能区分 query, "pure tie"被高估;
3. **同源信号重复计数**: 匹配端复用 loss 端的 lam, 两个通道对"小目标往哪拉"的隐含假设不一致, 中期互相拖累;
4. **NWD 盲区**: 无重叠时退化为"中心最近优先", 忽略长宽比/形状一致性, 形状精修阶段（中后期）开始拖后腿——与"早期略好、随后持续变差"的曲线形态吻合。

若仍想验证匹配端方向: ① 先统计微小 GT 的 assignment 翻转率验证前提; ② 在干净基座上单独测（不带 v4 loss）; ③ 只在无重叠时混 NWD（`w = lam·(1−iou_sim)`）; ④ 去掉 lam_pow 或减小 nwd_weight; ⑤ 与 v1 曲线比 ≥75ep。

---

## 7. 数值对照表

### 7.1 尺度因子 lam（640px 图, τ=1/32）

| 框尺寸 (px) | v1 (线性) | v2/v3/v4/v5 (平方) |
|---|---|---|
| 32 | 0 | 0 |
| 24 | 0.125 | 0.016 |
| 16 | 0.2 | 0.04 |
| 12 | 0.4 | 0.16 |
| 8 | 0.6 | 0.36 |
| 4 | 0.8 | 0.64 |

### 7.2 动态因子 eta（m=0.999, 约 1000 iter 时间常数）

训练早期 IoU 低 → eta 大（干预强）; 随 q_ema 收敛 eta→0。初始 q_ema=0.5 → 初始 eta=0.5。

---

## 8. 超参速查总表（六损失并排）

| 参数 | 原生 | v1 | v2 | v3 | v4 | v5 | 作用 |
|---|---|---|---|---|---|---|---|
| loss_gain class/bbox/giou | 1/5/2 | 同 | 同 | 同 | 同 | 同 | 三支路总权重 |
| matcher cost cls/L1/giou | 2/5/2 | 同 | 同 | 同 | 同 | 同(giou 混 NWD) | 匹配代价 |
| β | — | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 分类正样本放大强度 |
| α | — | 0.25 | 0.25 | 0.25 | —(拆分) | —(拆分) | 几何软化强度（v1~v3 单一） |
| α_l1 | — | — | — | — | 0.25 | 0.25 | L1 支软化 |
| α_iou | — | — | — | — | 0 | 0 | IoU 支软化（0=解耦放行） |
| τ | — | 1/32 | 1/32 | 1/32 | 1/32 | 1/32(+匹配端) | 尺度分界（32px） |
| lam_pow | — | 1(无) | 2 | 2 | 2 | 2 | 尺度聚焦指数 |
| eta_momentum | — | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 质量EMA动量 |
| q_ema_init | — | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | EMA 初值 |
| dynamic_eta | — | true | true | true | true | true | eta 是否动态 |
| dynamic_geo | — | — | true | true | true | true | 几何软化是否随eta退火 |
| blend_center | — | — | — | — | 0.45 | 0.45 | 分段切换点 |
| blend_temp | — | — | — | — | 0.05 | 0.05 | 切换平滑度 |
| inner_ratio | — | — | — | 0.8 | 0.8 | 0.8 | Inner 框缩放比 |
| dynamic_interp | — | — | — | true | true | true | 插值系数动态 |
| interp_alpha | — | — | — | 0.99 | 0.99 | 0.99 | 静态插值系数 |
| nwd_const | — | — | — | — | — | 0.02 | NWD 距离常数 |
| nwd_weight | — | — | — | — | — | 1.0 | NWD 最大混合强度 |

---

## 9. 演进图与有效性总结

```
原生 ──(只调权重: +β放大/−α刹车)──> v1 ──(lam², 几何退火)──> v2 ──(GIoU→InnerInterpIoU)──> v3
                                                        │
                                   (分段几何+L1/IoU解耦) │
                                                        v
                                                       v4 ──(匹配端NWD)──> v5 ✗
```

| 版本 | 有效? | 证据 | 结论 |
|---|---|---|---|
| v1 | **有效** | ep10~75 全程领先, +0.4~0.7pt mAP50-95 | 跑满 350ep 验证终值 |
| v2 | 无效（噪声级） | ep29 与 v1 打平/略低 | 机制被 v3/v4 继承 |
| v3 | 取决于指标 | mAP50 +0.7pt, mAP50-95 −0.3pt | mAP50/mAP50-95 trade-off |
| v4 | **最有希望** | ep10 mAP50 +1.4pt（全系列最快） | 续训 ≥75ep 对照 |
| v5 | **负收益** | ep5~15 连续落后 v4, 差距扩大 | 放弃或按隔离方案重做 |

**方法论提醒**: 各消融 run 的 epoch 数不等（15~350）, 任何绝对值比较必须同 epoch 对齐; 单 seed 下 ±0.3pt 属随机波动, v1 的 +0.4~0.7pt 已接近可信边缘, 跑满 + 重复 seed 是确认"有用"的最低要求。

---

## 10. DSAW-v6（v4 loss + 课程化无重叠门控 NWD 匹配器）—— 匹配端重设计

> 定位: 吸取 v5 教训后在 v4 基础上的匹配端重设计。**loss 分支逐位继承 v4**，唯一改动是匹配器。
> 代码: `ultralytics/utils/loss_plugins/dsaw_v6_loss.py`
> 配置: `ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-DSAW-v6.yaml`

### 10.1 v5 教训 → v6 修复对照

| v5 失败模式 | v6 对策 |
|---|---|
| ① NWD 在中后期（形状精修期）持续扰动 | **课程化**: 混合权重乘 η（质量缺口），训练前期强、收敛后自动归零 |
| ② NWD 稀释了 GIoU 本可工作的区间的形状监督 | **无重叠门控**: 仅对 plain IoU < 0.1 的 (query, GT) 对混 NWD；有重叠的对保持纯 GIoU |
| ③ lam² 门控覆盖不足（16px→4%），VisDrone 主体 10~30px 无干预 | **匹配端独立线性门**: matcher_lam_pow=1，覆盖 16~24px 框 |
| ④ 匹配端与 loss 端复用同一 lam（信号纠缠） | **门控解耦**: matcher_tau / matcher_lam_pow 独立超参 |
| ⑤ "GIoU cost 打平"前提从未验证 | **内置诊断计数器**: last_w_mean / last_gate_frac / last_w_active_mean |

### 10.2 匹配器公式（唯一改动）

```
lam_j   = clip(1 − √(w_j·h_j)/matcher_tau, 0, 1) ** matcher_lam_pow   (GT 侧, 无梯度)
eta_t   = 1 − q_ema                    (一步延迟: 匹配器在 loss 刷新 q_ema 前运行; EMA
                                         时间常数≈1000 iter, 延迟可忽略)
gate_ij = 1[ IoU_raw(pred_i, gt_j) < overlap_thresh ]                  (detach)
w_ij    = nwd_weight · lam_j · eta_t · gate_ij
cost_giou = 1 − ( (1−w_ij)·GIoU + w_ij·NWD )
C = 2·C_cls + 5·C_L1 + 2·cost_giou      (cost_gain 与基线完全一致)
```

### 10.3 新增超参（vs v4）

| 参数 | 默认 | 作用 |
|---|---|---|
| nwd_const | 0.02 | Wasserstein 距离常数（归一化坐标） |
| nwd_weight | 0.5 | lam=1, eta=1, gate=1 时的最大混合强度（0=退化为 v4） |
| overlap_thresh | 0.1 | 无重叠门阈值: IoU ≥ 此值的对保持纯 GIoU |
| matcher_tau | 0.03125 | 匹配端独立尺度分界（默认复用 loss 的 τ） |
| matcher_lam_pow | 1.0 | 匹配端独立聚焦指数（线性, 与 loss 的平方解耦） |

### 10.4 退化保证与验证结果

- nwd_weight=0 或 matcher_tau=0 → w=0 → **逐位等于 v4**（合成数据实测 max_diff=0.00e+00, BIT-EXACT PASS）;
- 前向验证: loss 三支路数值正常, eta 跨迭代正确同步, 诊断计数器工作正常。

### 10.5 实验建议

1. 先跑 75ep 与 v1/v4 曲线同 epoch 对齐（不要与 350ep 的绝对值比较）;
2. 训练中抽查 `matcher.last_gate_frac / last_w_active_mean` 验证前提（门控触发率、实际混合强度）;
3. 若 v6 仍 < v4 → 匹配端方向封存, 资源转 v4 续训;
4. 若 v6 > v4 → 补"原生loss + v6匹配器"消融, 确认收益来自匹配端本身而非与 v4 loss 的交互。

### 10.6 演进图（更新）

```
原生 ──(只调权重)──> v1 ──(lam²,几何退火)──> v2 ──(InnerInterpIoU)──> v3
                                                        │
                                                        v
                                                       v4 ──┬──(NWD匹配, 恒定)──> v5 ✗
                                                            └──(NWD匹配, 课程化+无重叠门+解耦门)──> v6
```

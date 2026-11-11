# RT-DETR 损失函数分类清单

范围：当前仓库 RT-DETR 检测训练，核对 `loss_name` 注册表、实际实现和模型 YAML。共 **44 个可选入口、69 份配置**。该数不含原版、不含模型钩子额外加入的损失。以 `rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7` 的原版损失为对照；结构名里的 `DSAWACGAv5` 与损失名 `dsaw_v5` 是两件事。

- **匹配前**：[HungarianMatcher](../ultralytics/models/utils/ops.py#L12) 默认成本系数为分类 2、L1 5、GIoU 2，用于分配 query 和 GT。
- **匹配后**：[原版检测损失](../ultralytics/models/utils/loss.py#L14) 为 VFL、L1、GIoU，默认系数为 1、5、2；[模型入口](../ultralytics/nn/tasks.py#L456) 按 YAML 选择[注册项](../ultralytics/utils/loss_plugins/__init__.py)。即使替换 GIoU，返回键通常仍叫 `loss_giou`。
- 表中「—」表示该主分支未改；「原版」表示未改 Hungarian 成本或配对策略，例外单独写明。只调系数、改公式、改分类软标签和附加监督分别标注。分类分组以普通匹配的主损失为准，DN 另列。

## DSAW 系列：三个主损失分支均有改动（以完整版本为准）

| `loss_name`（链接到源码） | 分类 `loss_class` | L1 `loss_bbox` | GIoU `loss_giou` | 匹配 / 额外监督 |
|---|---|---|---|---|
| [`dsaw`](../ultralytics/utils/loss_plugins/dsaw_loss.py) | VFL 正样本按尺寸和动态因子加权；保留 IoU 软标签 | 按目标尺寸减轻小框惩罚 | 保留 GIoU 公式，按目标尺寸减轻小框惩罚 | 原版 |
| [`dsaw_v2`](../ultralytics/utils/loss_plugins/dsaw_v2_loss.py) | 尺寸门控平方化，动态加权正样本 | 按尺寸及 EMA 质量差距退火 | 保留 GIoU 公式，按尺寸及 EMA 质量差距退火 | 原版 |
| [`inner_interpiou_dsaw3`](../ultralytics/utils/loss_plugins/inner_interpiou_dsaw3_loss.py) | 沿用 v2 | 沿用 v2 | InnerInterpIoU 替换 GIoU，保留动态几何降权 | 原版 |
| [`dsaw_v4`](../ultralytics/utils/loss_plugins/dsaw_v4_loss.py) | 沿用 v2 | 沿用 v2 型门控；本 YAML `alpha_l1=.25` | 按普通 IoU 的软门控混合 GIoU 与 InnerInterpIoU；本 YAML `alpha_iou=0`，不再额外按尺寸降权 | 原版 |
| [`dsaw_v5`](../ultralytics/utils/loss_plugins/dsaw_v5_loss.py) | 沿用 v4 | 沿用 v4 | 沿用 v4 | 小框匹配的几何成本混合 GIoU 与 NWD |
| [`dsaw_v6`](../ultralytics/utils/loss_plugins/dsaw_v6_loss.py) | 沿用 v4 | 沿用 v4 | 沿用 v4 | NWD 由低重叠和尺寸门控，随质量变化退场 |
| [`dsaw_v7`](../ultralytics/utils/loss_plugins/dsaw_v7_loss.py) | 沿用 v4 | 沿用 v4 | 沿用 v4 | 在 v6 基础上加 EMA 质量预热 |

### DSAW 消融和版本关系

- `DSAW-cls`：`alpha=0`，只改分类。 `DSAW-geo`：`beta=0`，只改 L1、GIoU 的权重。
- `DSAW-b1`：将分类系数 `beta` 从 0.5 提到 1；`DSAW-static`：关闭 EMA 动态因子，尺寸加权仍保留。
- v4 的 IoU 门控中心为 0.45、温度为 0.05。v5–v7 相比 v4 的新增改动都在匹配阶段；NWD 没有替换匹配后的 GIoU 回归项。

## 仅改分类分支（11 个入口）

| `loss_name`（链接到源码） | 分类 `loss_class` | L1 `loss_bbox` | GIoU `loss_giou` | 匹配 / 额外监督 |
|---|---|---|---|---|
| [`bqg_vfl`](../ultralytics/utils/loss_plugins/bqg_vfl_loss.py) | 用 Brier 型误差相对 EMA 计算匹配 query 行权重；只用于最后一层普通匹配分类 | — | — | 原版 |
| [`peq_vfl`](../ultralytics/utils/loss_plugins/peq_vfl_loss.py) | 融合普通 IoU 与扰动后的平均 IoU，作为 VFL 正样本质量目标 | — | — | 原版 |
| [`sr_mal`](../ultralytics/utils/loss_plugins/sr_mal_loss.py) | VFL → MAL；正样本质量用按尺寸混合的 IoU / NWD | — | — | 主配置另改几何匹配成本；`noAlign` 用原匹配，`pureMAL` 的 NWD 权重为零 |
| [`xfcw`](../ultralytics/utils/loss_plugins/xfcw_loss.py) | 支持按类别调整 VFL 正样本权重；现有两个 YAML 全为 1，实际等同原版 | — | — | 原版 |
| [`xcas`](../ultralytics/utils/loss_plugins/xcas_loss.py) | 后期衰减 `loss_class*` 权重 | — | — | 原版 |
| [`xvfla`](../ultralytics/utils/loss_plugins/xmisc_loss.py) | VFL 负样本系数 α 从 0.75 降为 0.5 | — | — | 原版 |
| [`xnegg`](../ultralytics/utils/loss_plugins/xb6_loss.py) | VFL 负样本 γ 从 2 升至 3 | — | — | 原版 |
| [`xls`](../ultralytics/utils/loss_plugins/xb6_loss.py) | 平滑 VFL 正样本质量标签 | — | — | 原版 |
| [`xqclamp`](../ultralytics/utils/loss_plugins/xdeep_loss.py) | VFL 正样本 IoU 软标签封顶 0.9 | — | — | 原版 |
| [`xfcwcas`](../ultralytics/utils/loss_plugins/xb8_loss.py) | 按类加权加后期衰减；当前 YAML 类权重全为 1，只有后期衰减实际生效 | — | — | 原版 |
| [`xsizeq`](../ultralytics/utils/loss_plugins/xb8_loss.py) | 提高小框正样本 IoU 软标签，封顶为 1 | — | — | 原版 |

## 仅改 L1 分支（2 个入口）

| `loss_name`（链接到源码） | 分类 `loss_class` | L1 `loss_bbox` | GIoU `loss_giou` | 匹配 / 额外监督 |
|---|---|---|---|---|
| [`xbbr`](../ultralytics/utils/loss_plugins/xmisc_loss.py) | — | 仅把权重从 5 调为 6 | — | 原版 |
| [`xsl1`](../ultralytics/utils/loss_plugins/xdeep_loss.py) | — | L1 → Smooth L1，β=1 | — | 原版 |

## 仅改 GIoU 分支（15 个入口）

| `loss_name`（链接到源码） | 分类 `loss_class` | L1 `loss_bbox` | GIoU `loss_giou` | 匹配 / 额外监督 |
|---|---|---|---|---|
| [`shape_iou`](../ultralytics/utils/loss_plugins/shape_iou_loss.py) | — | — | Shape-IoU，按目标形状调整中心和宽高惩罚 | 原版 |
| [`sd`](../ultralytics/utils/loss_plugins/sd_loss.py) | — | — | SDLoss，按目标像素面积调度 CIoU 的距离和形状项；面积 ≥81 px² 时退回 CIoU | 原版 |
| [`inner_siou`](../ultralytics/utils/loss_plugins/inner_siou_loss.py) | — | — | Inner-SIoU，辅助框重叠结合 SIoU 距离、角度与形状约束 | 原版 |
| [`inner_mpdiou`](../ultralytics/utils/loss_plugins/inner_mpdiou_loss.py) | — | — | Inner-MPDIoU，辅助框重叠结合对应角点距离 | 原版 |
| [`interpiou`](../ultralytics/utils/loss_plugins/interpiou_loss.py) | — | — | InterpIoU，混合原框与插值框重叠 | 原版 |
| [`inner_interpiou`](../ultralytics/utils/loss_plugins/inner_interpiou_loss.py) | — | — | InnerInterpIoU，动态插值结合固定比例的辅助框重叠 | 原版 |
| [`focaler_interpiou`](../ultralytics/utils/loss_plugins/focaler_interpiou_loss.py) | — | — | FocalerInterpIoU，插值重叠加 IoU 区间映射 | 原版 |
| [`da_inner_interpiou`](../ultralytics/utils/loss_plugins/da_inner_interpiou_loss.py) | — | — | DAInterpIoU，辅助框比例随 IoU 变化 | 原版 |
| [`inner_piou`](../ultralytics/utils/loss_plugins/inner_piou_loss.py) | — | — | Inner-PIoU，此处 PIoU 指 Powerful-IoU，叠加辅助框重叠 | 原版 |
| [`inner_piou2`](../ultralytics/utils/loss_plugins/inner_piou_loss.py) | — | — | Inner-PIoUv2，辅助框重叠加非单调聚焦 | 原版 |
| [`thermal_inner_iou`](../ultralytics/utils/loss_plugins/thermal_inner_iou_loss.py) | — | — | ThermalInnerIoU，高斯扩散软重叠加 Inner 修正 | 原版 |
| [`asiou_inner`](../ultralytics/utils/loss_plugins/asiou_inner_loss.py) | — | — | ASIoU 与 Inner 辅助框修正 | 原版 |
| [`aiou_inner`](../ultralytics/utils/loss_plugins/aiou_inner_loss.py) | — | — | AIoU 的自适应边界对齐加 Inner 修正 | 原版 |
| [`xgbr`](../ultralytics/utils/loss_plugins/xgbr_loss.py) | — | — | 仅把 GIoU 权重从 2 调为 3 | 原版 |
| [`sual_ur`](../ultralytics/utils/loss_plugins/sual_ur_loss.py) | — | — | SUAL-B：按跨解码层预测的不确定度重加权 GIoU；作用于普通主层和辅助层，不含 DN | 成本原版；辅助层复用末层配对 |

## 组合与可调权重（5 个入口）

| `loss_name`（链接到源码） | 分类 `loss_class` | L1 `loss_bbox` | GIoU `loss_giou` | 匹配 / 额外监督 |
|---|---|---|---|---|
| [`mal_shape`](../ultralytics/utils/loss_plugins/mal_shape_loss.py) | VFL → MAL | — | GIoU → Shape-IoU | 原版 |
| [`mal_interpiou`](../ultralytics/utils/loss_plugins/mal_interpiou_loss.py) | VFL → MAL；当前 YAML 融合原始与插值后的质量目标 | — | GIoU → D-InterpIoU | 原版 |
| [`sual`](../ultralytics/utils/loss_plugins/sual_loss.py) | — | 原 L1 加极坐标中心惩罚；L1 本体保留 | 按面积比例和不确定度修改重叠项 | 成本原版；辅助层复用末层配对。另加坐标分布 CE 和跨层框方差惩罚 |
| [`tri_scale`](../ultralytics/utils/loss_plugins/tri_scale_loss.py) | 当前配置不改 VFL；支持额外分类蒸馏，默认关闭 | 按尺寸调整中心与宽高坐标的 L1 权重 | 按尺寸重加权 GIoU | 成本原版；辅助层复用末层配对。另加 PGDE 密度对齐和 SARD 自蒸馏 |
| [`xgain`](../ultralytics/utils/loss_plugins/xgain_loss.py) | `CLS15`/`CLSGIOU`：1 → 1.5 | 支持配置；现有 5 份 YAML 均未改 L1 | `GIOU15`/`CLSGIOU`：2 → 1.5 | 原版 |

## 只改 DN 或增加独立监督（4 个入口）

| `loss_name`（链接到源码） | 分类 `loss_class` | L1 `loss_bbox` | GIoU `loss_giou` | 匹配 / 额外监督 |
|---|---|---|---|---|
| [`xdnw`](../ultralytics/utils/loss_plugins/xdnw_loss.py) | 仅 DN 分类乘 0.5 | 仅 DN L1 乘 0.5 | 仅 DN GIoU 乘 0.5 | 普通三项和匹配原版 |
| [`neighbor_overlap`](../ultralytics/utils/loss_plugins/neighbor_overlap_loss.py) | — | — | — | 原版；另加 `loss_neighbor_overlap`，约束同类相邻框过度重叠 |
| [`query_supplement`](../ultralytics/utils/loss_plugins/query_supplement_loss.py) | — | — | — | 复用原匹配结果；额外加 `loss_query_rank`，监督未进入 Top-K 的候选 query |
| [`query_supplement_decode`](../ultralytics/utils/loss_plugins/query_supplement_decode_loss.py) | — | — | — | 原匹配成本及主损失不变；另加排序及补充 query 的 `loss_query_decode`（含 VFL、L1、GIoU） |

## 注册表之外的附加训练损失

[模型训练损失汇总](../ultralytics/nn/tasks.py#L517) 还可向主损失字典追加下列项。它们由对应结构和权重启用，不是对 VFL、L1 或 GIoU 的直接替换。

| 返回键 | 触发分支和监督内容 |
|---|---|
| `loss_route_distill` | AIFI BlockSparse V8/V10 的路由 KL 蒸馏；V19 的路由负载均衡 |
| `loss_ground` | AIFI ForegroundGuided 的前景 BCE；RegisterMemory / RegisterBlockSparse 的 GT register BCE；FoundationDistill 的离线教师特征余弦监督；解码器 CGQS 的中心度监督、FQRank 的额外质量 BCE |
| `loss_qga` | QGA 解码器的查询 Gram 矩阵正则 |

[基础损失文件](../ultralytics/models/utils/loss.py#L14) 另外保留 Slide、EMA Slide、Slide VFL、EMA Slide VFL、MAL 分类开关，和可选 NWD、Wise-IoU 回归分支。这些是底层实现或可选分支，未作为单独的 `loss_name` YAML 纳入统计；MAL 的现有可选方案见表。未找到独立的 RNWD-CIoU 入口。

## 69 份 YAML 索引

下列链接保留实际文件路径，标签省略共有前缀 `rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-`。同一个入口的消融、权重和结构配置保留在同一行。

| `loss_name` | YAML |
|---|---|
| `dsaw` | [DSAW-b1](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-DSAW-b1.yaml)、[DSAW-cls](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-DSAW-cls.yaml)、[DSAW-geo](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-DSAW-geo.yaml)、[DSAW-static](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-DSAW-static.yaml)、[DSAW](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-DSAW.yaml) |
| `dsaw_v2` | [DSAW-v2](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-DSAW-v2.yaml) |
| `inner_interpiou_dsaw3` | [InnerInterpIoUDSAW3](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-InnerInterpIoUDSAW3.yaml) |
| `dsaw_v4` | [DSAW-v4](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-DSAW-v4.yaml) |
| `dsaw_v5` | [DSAW-v5](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-DSAW-v5.yaml) |
| `dsaw_v6` | [DSAW-v6](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-DSAW-v6.yaml) |
| `dsaw_v7` | [DSAW-v7](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-DSAW-v7.yaml) |
| `bqg_vfl` | [BQG-VFL](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-BQG-VFL.yaml) |
| `peq_vfl` | [PEQVFL](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-PEQVFL.yaml) |
| `sr_mal` | [SRMAL-g1p0](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-SRMAL-g1p0.yaml)、[SRMAL-g1p5](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-SRMAL-g1p5.yaml)、[SRMAL-g2p5](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-SRMAL-g2p5.yaml)、[SRMAL-noAlign](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-SRMAL-noAlign.yaml)、[SRMAL-pureMAL](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-SRMAL-pureMAL.yaml)、[SRMAL-t16](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-SRMAL-t16.yaml)、[SRMAL-t64](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-SRMAL-t64.yaml)、[SRMAL](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-SRMAL.yaml) |
| `xfcw` | [X004-FCW](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X004-FCW.yaml)、[X081-BS-FCW](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X081-BS-FCW.yaml) |
| `xcas` | [X008-CAS](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X008-CAS.yaml)、[X082-BS-CAS](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X082-BS-CAS.yaml) |
| `xvfla` | [X029-VFLA05](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X029-VFLA05.yaml) |
| `xnegg` | [X055-NegG3](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X055-NegG3.yaml)、[X084-BS-NegG](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X084-BS-NegG.yaml) |
| `xls` | [X056-LS01](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X056-LS01.yaml)、[X085-BS-LS](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X085-BS-LS.yaml) |
| `xqclamp` | [X063-QClamp](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X063-QClamp.yaml)、[X072-BS-QClamp](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X072-BS-QClamp.yaml) |
| `xfcwcas` | [X075-FCWCAS](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X075-FCWCAS.yaml) |
| `xsizeq` | [X078-SizeQ](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X078-SizeQ.yaml)、[X083-BS-SizeQ](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X083-BS-SizeQ.yaml) |
| `xbbr` | [X028-BBR6](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X028-BBR6.yaml)、[X100-BS-XBBR](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X100-BS-XBBR.yaml) |
| `xsl1` | [X068-SmoothL1](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X068-SmoothL1.yaml)、[X086-BS-SL1](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X086-BS-SL1.yaml) |
| `shape_iou` | [ShapeIoU](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-ShapeIoU.yaml) |
| `sd` | [SDLoss](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-SDLoss.yaml) |
| `inner_siou` | [InnerSIoU](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-InnerSIoU.yaml) |
| `inner_mpdiou` | [InnerMPDIoU](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-InnerMPDIoU.yaml) |
| `interpiou` | [InterpIoU](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-InterpIoU.yaml) |
| `inner_interpiou` | [InnerInterpIoU](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-InnerInterpIoU.yaml) |
| `focaler_interpiou` | [FocalerInterpIoU](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-FocalerInterpIoU.yaml) |
| `da_inner_interpiou` | [DAInterpIoU](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-DAInterpIoU.yaml) |
| `inner_piou` | [InnerPIoU](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-InnerPIoU.yaml) |
| `inner_piou2` | [InnerPIoU2](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-InnerPIoU2.yaml) |
| `thermal_inner_iou` | [ThermalInnerIoU](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-ThermalInnerIoU.yaml) |
| `asiou_inner` | [ASIoUInnerIoU](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-ASIoUInnerIoU.yaml) |
| `aiou_inner` | [AIoUInnerIoU](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-AIoUInnerIoU.yaml) |
| `xgbr` | [X018-GBR](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X018-GBR.yaml)、[X099-BS-XGBR](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X099-BS-XGBR.yaml) |
| `sual_ur` | [SUAL-B](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-SUAL-B.yaml) |
| `mal_shape` | [MalShape](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-MalShape.yaml) |
| `mal_interpiou` | [MalInterpIoU](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-MalInterpIoU.yaml) |
| `sual` | [SUAL](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-SUAL.yaml) |
| `tri_scale` | [TSL](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-TSL.yaml) |
| `xgain` | [X037-CLS15](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X037-CLS15.yaml)、[X038-GIOU15](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X038-GIOU15.yaml)、[X045-CLSGIOU](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X045-CLSGIOU.yaml)、[X087-BS-CLS15](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X087-BS-CLS15.yaml)、[X088-BS-GIOU15](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X088-BS-GIOU15.yaml) |
| `xdnw` | [X047-DNw05](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X047-DNw05.yaml)、[X089-BS-DNw](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-X089-BS-DNw.yaml) |
| `neighbor_overlap` | [NeighborOverlap](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-NeighborOverlap.yaml) |
| `query_supplement` | [QuerySupplement](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-QuerySupplement.yaml) |
| `query_supplement_decode` | [QuerySupplementDecode](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-QuerySupplementDecode.yaml) |

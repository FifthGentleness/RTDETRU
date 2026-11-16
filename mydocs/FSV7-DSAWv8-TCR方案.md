# DSAWv8-TCR：面向 AP 阈值的排序与定位损失

> 本文及本文配置保留完整 DSAWv4 作为历史基线。新实验使用 [DSAWv4-IoUOnly + TCR](FSV7-DSAWv4-IoUOnly-TCR.md)，沿用同一套 TCR 规则。

**建议只押一版：在原 FSV7 的 DSAWv4 上，对最后一层常规 query 增加「同类违序排序 + 近阈值定位纠错」。** 高分未匹配误检要降分；已匹配、也可能在较宽松阈值下计为 TP 的框，保留其分类监督，只让有把握的近阈值框修定位。两路用同一套违序样本。保留标准 AIFI、解码器、300 query、原 Hungarian matcher、编码器辅助层及 DN。推理沿用原分类分数和框，无额外头、候选或前向操作。

## 1. 哪些结果决定了方向

以下为同一验证集、同一 640×640/FP32/全部 300 query 评估协议。整体 AP 是项目评估的 mAP50–95；像素分组使用 640 输入尺寸的小目标（√面积 <32 px）。比较的是各自实际检查点，尚非配对重复实验。

| 检查点 | AP50–95 | AP50 | AP75 | 640 小目标 AP | 短边 8–16 px AP75 |
|---|---:|---:|---:|---:|---:|
| 原 FSV7 best，e265 | 31.9426% | 51.4588% | 33.0507% | 25.464% | — |
| DSAWv4 best，e309 | **32.0465%** | **51.7127%** | **33.1551%** | 25.374% | 30.3021% |
| DSAWv4-Edge last，e350 | 31.8536% | 51.3633% | 32.9696% | 25.151% | 29.8552% |

- **v4 的进步主要体现在较宽松的阈值。** 相对基线，AP50 +0.254 pp，总 AP +0.104 pp，AP75 +0.104 pp，小目标 AP −0.090 pp。v4 e306–315 同轮均值领先基线 0.334 pp，但 e251–270 的对比方向相反。32.189% 的 BlockSparse 是不同 AIFI 的外部参考，不纳入本次模型改动或配对对照。
- **框更准，分数没一定排对。** v4 的 ≥0.5 分数预测有 28,686 个（基线 27,641），这些框按原评估在 IoU 0.75 的 TP 比例为 59.38%（基线 60.98%）。Edge 的 8–16 px 几何 R75 +0.261 pp、同类 R75 +0.101 pp，但该组 AP75 掉 0.447 pp；原 Edge `best.pt` 和 v4 `best.pt` 的 871 项 EMA 张量完全一致。这个续训结果不能单独归因到 Edge 项，更不能拿继承的 best 当改进结果。
- **有排序空间，但不能把诊断当涨点。** v4 的 GT 重排对固定框、类和原 TP 分配重新计 AP，AP75 +9.47 pp、整体 AP +7.24 pp；它不会重跑去重匹配。四边最多纠正 0.5 px 的另一反事实为 AP75 +3.14 pp。两种诊断的约束不同。实际 Edge 没兑现几何反事实，优先解决哪些框排在前面。
- **历史试验给出三条约束。** v1/v2/v5 的小框重权或 NWD 有早段收益，v7 跑到 e280 后同预算峰值比基线低 0.198 pp，没有支持再全局改匹配。InnerInterpIoU 单独跑到 e299，历史峰值 31.870%；NeighborOverlap 跑满 350，峰值 31.662%，说明单独替换回归度量、或只在已匹配邻居框间加约束，现有收益有限。BQG-VFL、PEQVFL、SRMAL 和 SUAL 等同时重塑质量监督的试验无法稳定兑现 AP，首版保留 v4 的 VFL 正样本 IoU 标签和正负目标，新增项只触发于实测分数违序。

证据：[v4 完整复评与对照](../runs/diagnostics/fsv7_dsaw_v4_best/report.md)、[Edge 实际检查点复评](../runs/diagnostics/fsv7_dsaw_v4_best/edge_checkpoints_report.md)、[历史复盘](FSV7_全量训练结果与失败复盘_20261106.md)。R75 是每个 GT 至少有一个满足条件的框，未考察置信度和一对一去重；它不等于 AP75。

本次另外做了两项**只读离线回放**，均用 v4 best、548 张图、640×640、FP32 和原末层匹配：

- **严格换配空间很小。** 17,750 个匹配框未达 .75；只有 **344 个（1.94%）** 存在能跨 .75 的未匹配候选，同时满足 argmax 同类、分数 ≥.10、最佳 GT 是当前目标且比次优 GT 高 ≥.10 IoU、比原匹配框高 ≥.05 IoU。8–16 px 组为 120 / 4,935。因此这版保护更优候选，不更换匈牙利分配。
- **跨图违序的候选面明显更广。** seed 0 将图随机分为 4 图小批，只在 32,490 个预测 argmax 等于其匹配 GT 真类的框中比同类、不同图。优者 IoU ≥.55、IoU 差 ≥.08、至少有一个正式阈值夹在二者之间且各离阈值 ≥.015；要求优者分数反而更低。15,213 个框曾是逆序中的高质量框；9,483 个低质量框距 [.60,.85] 的某个被跨过阈值仅 .015–.060。这是构对限额之前的机会计数，不是 TCR 的实际训练触发率或 AP 增量。严格同 GT 重复框的高分抢位覆盖远少于跨图违序，所以同类跨图配对是主项，重复框是补充。

## 2. 为什么这个结构适合切这个位置

| 路径 | 实际结构 | 对损失设计的影响 |
|---|---|---|
| 主干 | 名称含 `r18`，实际是 stride-2 Conv + `C2f_DSAWACGA_v5`。P2/P3/P4/P5 通道 128/256/384/384，步长 4/8/16/32。v5 块结合 Haar 调制的多核 DWConv、Focused Linear Attention 和 StarFFN。 | 有局部细节和上下文表达，但结构本身并不保证输出分数随框质量有序。 |
| 细节融合 | P2 四相位 SPD 后投到 128 通道，与全通道 P3 和上采样 Y4 拼成 640；先 1×1 混合，再按 160/480 分流。160 路加 DCFM、31 大核、FCA→SCA→FGM 及 FreqScale，并在块内保留残差。两路重拼、RepC3 得 f3，PAN 再传至 f4/f5。 | P2 信息已进入检测路径，但融合后是 stride 8。损失可教已有候选分清同类邻居，无法恢复特征或候选中不存在的对象。 |
| AIFI / decoder | 标准 AIFI 只在 P5（640 输入下 20×20，投影至 256 通道）做全局交互。解码读取 f3/f4/f5，无独立 P2 输入；编码器按最大类别 logit 取 Top-300，3 层 decoder 精修。项目验证对每个 query 取最大类别分数，保留 300 框，按分数排，无 NMS。 | 同类高分误检和重复预测会影响 AP。损失插件只见已入选的 Top-300 及各阶段输出，不直接监督全量未入选位置。 |

短边 8–16 px 的几何 R75：基线全池 **65.154 → Top-300 58.791 → 末层 61.321%**，v4 为 **63.859 → 57.489 → 60.792%**。差距在选点前已有。TCR 的目标是把已有好框排前并修复部分临界框；如果仍缺覆盖，不能通过加大排序系数补回来。

源码：[FSV7 模型 YAML](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7.yaml#L31)、[DSAWACGAv5](../ultralytics/nn/extra_modules/dsawacga_v5.py#L166)、[CCFFP2V7](../ultralytics/nn/extra_modules/hybrid_encoder_p2_spd_okm_fs_v7.py#L223)、[Top-300](../ultralytics/nn/modules/head.py#L335)、[验证后处理](../ultralytics/models/rtdetr/val.py#L93)。

## 3. 完整损失：同一个违序，按身份给不同梯度

$\mathcal L = \mathcal L_{\mathrm{v4}} + r(e)\,[\lambda_r\mathcal L_{\mathrm{rank}}+\lambda_b\mathcal L_{\mathrm{cross}}]$。$\mathcal L_{\mathrm{v4}}$ 包括原 VFL、小框正样本增权、L1 减权、低 IoU 偏 InnerInterpIoU / 高 IoU 偏 GIoU、辅助分支和 DN。新项只接末层常规 query；两条额外梯度分别只接当前分类 logits 和框输出。共享特征仍会收到两路梯度。

记原末层 Hungarian 匹配得到 $i\to g_i$，真类 $c_i$，$q_i=\mathrm{IoU}(b_i,g_i)$，$z_{ic}$ 为原分类 logit。逐图计算矩阵 $Q_{jg}=\mathrm{IoU}(b_j,g)$。只用 detached FP32 的框、分数、IoU 构对；仅在计算最终损失时取回可求导的 logit / 对齐 IoU。阈值 $T=\{.50,.55,\dots,.95\}$，容错带 $\epsilon=.015$。这是 AP 的局部代理，训练批内排序不等价于全验证集 AP。

### 3.1 谁应该排在谁前面

每个阈值 $t$，找 $q_i\ge t+\epsilon$ 且当前 argmax 类为真类的匹配 query 作胜者。只比较同类；当 $z_{ic}-z_{jc}<m=.20$ 时构成违序或间隔不足。每个胜者、每个阈值，每种竞争者各取至多一个风险最高者（真类 logit 最大）；同一竞争者同一阈值被选中超过两次时，保留间隔缺口最大的两对。M 类从同一个违序池分别按分类条件和定位条件取样，两路的取样不受另一项系数开关影响。

| 竞争者 | 候选约束 | 分类排序梯度 |
|---|---|---|
| **D：确定重复框** | 同图、未匹配且预测类为 $c_i$；最贴近 $g_i$，全类 GT 中第一与第二 IoU 差 ≥.10；对 $g_i$ 的 IoU ≥.50，且至少比其主匹配低 .03；不能达到其它同类 GT 的 $t-\epsilon$。 | 推高胜者、压低重复框的 $c_i$ logit。 |
| **F：明确背景误检** | 未匹配且预测类为 $c_i$，与本图所有 GT 的最大 IoU ≤.10，$p_{jc}\ge .20$；可跨图同类比较。空图允许提供这种负例。 | 推高胜者、压低误检的 $c_i$ logit。 |
| **M：低质量匹配框** | 已匹配到另一个同类 GT、当前预测类正确，$q_j\ge .40$、$p_{jc}\ge .20$，且对本图同类 GT 的最大 IoU <$t$；同图或跨图。分类排序收紧到最大 IoU ≤$t-\epsilon$。 | 只推高胜者，**停止对竞争者 logit 的梯度**。后者仍受原 VFL 正样本监督，以免损伤 AP50。 |

其它未匹配框不进入新增排序，尤其是比主匹配更准的框、有歧义的密集邻居和潜在属于另一个 GT 的框。不改变它们在原 VFL 里的背景身份。也不通过交换 query 把一个 GT 重新分给其它框。

每对的正则项为 $h=T_s\,\mathrm{softplus}((m-z_{ic}+\tilde z_{jc})/T_s)$，$T_s=.5$。$\tilde z_{jc}$ 在 D/F 为实时值，在 M 停梯度。M 路只有在胜者当前分数低于 $\min(q_i+.03,.98)$ 时计算排序项，避免为追过一个过度自信的框把好框也推得过度自信；定位触发保留。三种竞争者分别累积，$\mathcal L_{\mathrm{rank}}=\sum h/[3|T|\max(N_{\mathrm{match}},1)]$。对多阈值都排错的对自然有更强信号；分母使用全部匹配数，避免激活对很少时系数突增。

### 3.2 只给能跨线的真实目标加定位梯度

从 M 的违序池选 $t\in[.60,.85]$、$t-.06\le q_j<t$ 的边缘框。**定位支路在 $t$ 以下保留监督**，避免框一进入分类排序的容错区就失去跨线梯度。每个 query 至多取一个阈值；多个合格阈值时选最近的上方阈值。用真实匹配 GT 的可求导 IoU 计算：

$$
\mathcal L_{\mathrm{cross}}=\frac{1}{\max(N_{\mathrm{match}},1)}\sum_{(j,t)\in\mathcal R}\operatorname{ReLU}\left(t+0.01-\mathrm{IoU}(b_j,g_j)\right).
$$

例如 AP75 下，好框 IoU 0.81、得分 0.56，另一已匹配框 IoU 0.71、得分 0.68：排序项推好框分数，定位项把后者的框往 0.76 推，不压它的正类分数。如果后者是同一 GT 的较差未匹配框，才直接压分，不给它新增回归训练。这是和 Edge 的关键区别：看「会抢排名、能跨阈值」再给局部 IoU 梯度，而非对某个尺寸组普遍加四边惩罚。

初始建议：$\lambda_r=0.25,\lambda_b=0.50$。微调前 2 轮系数为零，接下来 5 轮线性升到 1；全程重训建议前 15 轮为零，再用 15 轮升到 1。先在几个训练批上量新项对末层输出的梯度：有效批里起始目标约为各自 v4 对应项梯度范数的 5–20%；系数只在这次标定中调，锁定后跑配对。没有正样本或有效对时新项为零。预先记录触发率；如果有效对很少，先排查采样覆盖。

## 4. 接线与实验判定

- **代码位置：** [TCR 插件](../ultralytics/utils/loss_plugins/dsaw_v8_tcr_loss.py)继承 `RTDETRDetectionLossDSAWV4`，由[注册表](../ultralytics/utils/loss_plugins/__init__.py)的 `dsaw_v8_tcr` 选择。插件仿照已有损失，在首次主 `_get_loss` 复用原匹配；encoder、辅助解码层和 DN 按原流程计算。沿用 `innovation_schedule` / `set_epoch`；验证和预热不构对。`loss_tcr_rank` 和 `loss_tcr_cross` 纳入训练总 loss，原 CSV 仍按仓库流程记录三项主 loss。训练时可读取 `model.criterion.last_tcr_stats` 查看最近训练批的分支有效对数、参与几何修复的框数和未乘系数的损失值。可运行 `python tools/verify_dsaw_v8_tcr.py -v` 检查开关、梯度和分支隔离。
- **先验证监督打在什么框上。** 在 v4 权重上回放，分 D/F/M 统计真正违序、跨线门触发、误压更优候选（应为零），按类和 `<8` / `8–16` / `≥16` px 分桶；重算修改评分后的真实匹配可作为调试，只改 AP 输入数组的原 oracle 不作本版验收。训练同时监控几何 R75、同类 R75、固定分数段的 TP75 占比、同类排序反转、重复 FP，以及 AP50 的召回护栏。不同阈值的预测归属可能变化，不声称按 rank pair 计数等价于 TP/FP 数。
- **三组配对：** A 原 v4，B 只开 rank，C TCR 全量。三组从同一 v4 `best.pt` 的 EMA 权重重新初始化优化器，保持 FSV7 图、seed、数据、增强、batch 和 EMA 设置一致；首筛 25 轮、AdamW `lr0=2e-5`，短微调用同一 warmup 计划，`warmup_bias_lr` 设为 0 或与 `lr0` 同阶。重开优化器的三组不能和历史 resume 曲线作配对。GPU 预算只够两条跑 A/C；有信号再补 B。
- **继续门槛：** 用同轮末 10 轮均值，C 相对 A 的总 AP 至少 +0.30 pp、AP75 至少 +0.40 pp，AP50 不低于 −0.20 pp，并检查短边 `<8` 的召回及各类 AP。数值是算力门槛，非预报。B 同样只靠排序就达到门槛而 C 没增加收益，选 B；C 若只有 AP75 提升而整体 AP 不涨，暂不替代 v4。过门槛再做至少 3 个配对 seed 的完整训练，按固定选模规则报告 AP/AP50/AP75 及差值。只有固定权重排序改善、几何覆盖不变时，可支持排序贡献；若 C 相对 B 改善近阈值跨线及总 AP，才支持定位纠错贡献。

排序监督已有 [Rank & Sort Loss](https://openaccess.thecvf.com/content/ICCV2021/html/Oksuz_Rank__Sort_Loss_for_Object_Detection_and_Instance_Segmentation_ICCV_2021_paper.html) 等先例，[Stable Matching](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Detection_Transformer_with_Stable_Matching_ICCV_2023_paper.pdf) 也研究过定位质量与 DETR 分类／匹配结合。这里的创新主张限于 **FSV7 原一对一身份保护下，用同一批多阈值违序对，将未匹配误检导向分类排序、将可跨线匹配框导向几何纠错**。阈值参考 [COCO API](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)；仓库的匹配与最大预测数仍按当前项目评估。

## 5. 启动训练

[完整 TCR](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-DSAW-v8-TCR.yaml) 与 [RankOnly](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-DSAW-v8-TCR-RankOnly.yaml) 都沿用仓库的 `train.py --model` 入口。以下在仓库根目录运行。默认 350 轮：前 15 轮仅 v4，第 16–30 轮线性加入 TCR。

```powershell
python train.py --model ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-DSAW-v8-TCR.yaml --name FSV7-DSAWv8-TCR --device 0
python train.py --model ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-DSAW-v8-TCR-RankOnly.yaml --name FSV7-DSAWv8-TCR-RankOnly --device 0
```

首轮 25 轮配对使用同一个 Ultralytics `RTDETR` 训练接口；`train.py` 本身固定 350 轮，不传它未定义的参数。下面存为临时 Python 文件，在仓库根目录执行；依次把 `group` 改成 `A`、`B`、`C`。三组都从同一份 v4 best 加载权重，再新建优化器；短实验仅改 TCR 两组的开启节奏（前 2 轮关闭，第 3–7 轮爬坡）。

```python
from pathlib import Path

from ultralytics import RTDETR

if __name__ == '__main__':
    group = 'C'  # A: v4; B: RankOnly; C: Full TCR
    suffix = {'A': 'v4', 'B': 'v8-TCR-RankOnly', 'C': 'v8-TCR'}[group]
    cfg = Path('ultralytics/cfg/models/rt-detr') / f'rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-DSAW-{suffix}.yaml'
    weights = Path('runs/train/rtdetr_r18_DSAWACGAv5-P2SPDOKMFSV7-DSAW-v4/weights/best.pt')
    model = RTDETR(str(cfg)).load(str(weights))
    if group != 'A':
        model.model.yaml['loss_params'].update(start_epoch=2, full_epoch=7)
    model.train(data='dataset/data.yaml', imgsz=640, batch=4, workers=4, device='0',
                epochs=25, patience=0, optimizer='AdamW', lr0=2e-5, lrf=1.0,
                warmup_epochs=200, warmup_bias_lr=0.0, amp=False, seed=0,
                cache=False, close_mosaic=0, resume=False,
                project='runs/train', name=f'FSV7-TCR-pair-{group}')
```

这里 `.load()` 只把旧权重迁到新 YAML 对应的模型。首轮配对不要对 v4 `best.pt` 使用 `--resume`，否则会恢复旧模型 YAML。若某个 TCR 运行中断，恢复该组自己的 `last.pt` 时仍用原 `train.py --resume`。

## 附：当前可核对的 loss-only 训练记录

扫描本地 `runs/train` 原始 CSV，且 YAML 网络图与基础 FSV7 一致。AP 峰值为 %，Δ 为 pp；同预算取基线相同轮数内峰值；末窗按轮对齐，最多十轮。短程结果描述当前预算，不能推断长训上限。只有配置没有 CSV（如 DSAWv4-IoUOnly、DSAW cls/geo/static/b1）不计入。FQRank/QuerySupplementDecode 修改解码或另跑解码，只作为机制旁证。

| 变体 | 记录轮数 | AP 峰值（轮次） | 同预算峰值 Δ基线 | 末窗 Δ基线 |
|---|---:|---:|---:|---:|
| 原 FSV7 | 350 | 31.943 (265) | 0 | 0 |
| DSAW | 75 | 28.242 (75) | +0.367 | +0.175 |
| DSAW-v2 | 29 | 21.968 (29) | +0.161 | +0.181 |
| InnerInterpIoUDSAW3 | 33 | 22.627 (33) | −0.081 | −0.052 |
| DSAW-v4 | 315 | 32.046 (309) | +0.103 | +0.334 |
| DSAW-v4-Edge | 41（续训 e310–350） | 32.020 (310) | — | +0.053 |
| DSAW-v5 | 45 | 25.210 (45) | +0.517 | +0.338 |
| DSAW-v6 | 5 | 9.595 (5) | −0.469 | −0.112 |
| DSAW-v7 | 280 | 31.745 (259) | −0.198 | −0.222 |
| ASIoUInnerIoU | 29 | 21.589 (29) | −0.218 | −0.177 |
| BQG-VFL | 160 | 30.055 (160) | −0.937 | −0.821 |
| FocalerInterpIoU | 54 | 25.362 (54) | −0.634 | −0.707 |
| GCD | 207 | 31.182 (198) | −0.279 | −0.301 |
| InnerInterpIoU | 299 | 31.870 (280) | −0.073 | −0.031 |
| InterpIoU | 22 | 19.903 (22) | +0.288 | +0.411 |
| JitteringIoU | 55 | 25.938 (55) | −0.204 | −0.204 |
| MGIoU-HBB | 72 | 27.591 (72) | −0.180 | −0.053 |
| MGIoU-HBB-IoUScale | 50 | 25.391 (50) | −0.053 | +0.002 |
| MGIoU-HBB-X2 | 34 | 23.432 (34) | +0.506 | +0.433 |
| MGIoU-HBB-X4 | 27 | 20.257 (27) | −0.880 | −0.776 |
| NeighborOverlap | 350 | 31.662 (308) | −0.281 | −0.335 |
| PEQVFL | 45 | 23.816 (45) | −0.877 | −0.802 |
| SDLoss | 34 | 23.213 (34) | +0.287 | +0.218 |
| ShapeIoU | 160 | 30.966 (155) | −0.026 | +0.148 |
| SUAL | 3 | 0.000 (1) | −6.436 | −3.643 |
| SUAL-B | 18 | 13.984 (18) | −4.147 | −3.593 |
| TSL | 28 | 15.235 (28) | −6.300 | −5.807 |
| X078-SizeQ | 74 | 27.304 (74) | −0.471 | −0.581 |
| SRMAL | 13 | 11.476 (13) | −4.582 | −4.554 |

Edge CSV 有继承的 v4 历史及回卷追加轮次，仅取最后连续续训段。若只看原 v4 已记录范围之后的 e316–350，Edge 峰值为 e323 的 31.973%。缺少同检查点纯 v4 的平行续训，末窗仅作描述。

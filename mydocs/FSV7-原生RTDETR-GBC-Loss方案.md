# FSV7 第三创新点：GBC-Loss（选点前的实例覆盖与配额损失）

**推荐只押一版：从基础 FSV7 的原生 RT-DETR 损失出发，完整替换编码器的 Top-300 辅助损失。让整个 P3/P4/P5 网格中的位置都能通过 GT 空间邻域竞争训练资格，每个 GT 只推一个主代表去竞争第 300 个名额，另一个候选可以学定位。** 已有 AIFI、解码器各层、一对一末层监督、DN 和实际选点规则沿用原版。这个方案不继承 DSAWv4/TCR 的权重、回归或排序目标。

## 1. 为什么从选点之前改

原生 FSV7 的历史峰值 **31.943%**，DSAWv4 **32.047%**，约 **+0.104 pp**。同协议复评下，v4 的 AP50 增加约 0.254 pp，AP75 仅增加约 0.104 pp，640 输入的小目标分组 AP −0.090 pp。更换几何度量或调权的多组试验没有给出大幅、可重复的长训收益：InnerInterpIoU 峰值 31.870%，NeighborOverlap 跑满峰值 31.662%，DSAWv7 跑到 e280 同预算峰值比原版低 0.198 pp。SUAL、SRMAL 等早期改多路分类/定位/匹配的试验出现大幅落后，不能简单归因，但也不值得照搬。Edge 在 8–16 px 的几何覆盖略增，分组 AP75 却掉约 0.447 pp。本地未找到 TCR 或 IoUOnly 的正式训练 CSV。[28 项原始轨迹对照](FSV7-DSAWv8-TCR方案.md#附当前可核对的-loss-only-训练记录) · [v4 复评](../runs/diagnostics/fsv7_dsaw_v4_best/report.md) · [Edge 复评](../runs/diagnostics/fsv7_dsaw_v4_best/edge_checkpoints_report.md)

更直接的证据来自**原生 FSV7 best**，548 张图、38,759 个 GT（其中 **33,280** 个小目标），640×640 拉伸，按每个 GT 的最大 IoU 计算几何覆盖：

| 分组 | 编码器全池 | 原 Top-300 初始框 | 末层 |
|---|---:|---:|---:|
| 小目标 R50 | 90.08% | 84.96% | 85.94% |
| 小目标 R75 | 53.21% | 48.40% | 50.73% |
| 小目标短边 8–16 px R75 | 65.154% | 58.791% | 61.321% |

两处缺口不同：有的 GT 全池已经有好框，却没被原分数选中；也有的 GT 即使搜全池也没有准确框。因此要同时学习**框能不能拟合**和**每个实例能不能拿到名额**。现成的 128 图冻结权重反事实中，用 GT 每图最多换 3 个槽位，小目标初始 R75 +1.255 pp，末层只 +0.418 pp，总 AP 只 +0.074 pp。它不能证明更大规模选点没有用，但不支持只做分数搬运。[全池覆盖](../runs/diagnostics/query_selection_baseline/report.md) · [短边组](../runs/diagnostics/fsv7_dsaw_v4_best/report.md) · [冻结换位](../runs/diagnostics/fsv7_query_swap/summary_128.json)

本次又直接用这 548 图的标注验证空间分配：短边 <8 px 的 15,220 个 GT 中，**29.7%** 与其它 GT 落在同一 P3 格，**40.5%** 的所在格中心不在框内。按原 anchor 有效掩码逐图求最大基数匹配，P3 只给所在单格时，38,759 个 GT 中只能给 **35,314** 个分到独占位置；改用 P3 3×3 邻域后，**38,759** 个均有独占位置，短边 <8 px 也全覆盖。P3/P4/P5 都给 3×3 同样全覆盖。这个检查没有用模型分数、预测框或 300 名额；它验证的是空间候选图可行，不能换算为检测召回。P4/P5 在方案中主要补尺度，避免给所有目标硬挤 P3。

## 2. 结构上动哪里

| 结构 | 640 输入下的真实路径 | 损失含义 |
|---|---|---|
| DSAWACGAv5 主干 | 名称含 `r18`，实际 backbone 是 stride-2 Conv + `C2f_DSAWACGA_v5`，在 P2/P3/P4/P5 输出 128/256/384/384 通道。块结合 Haar 调制的多核 DWConv、Focused Linear Attention 和 StarFFN。 | P2 保有高分辨率细节，但细节存在不代表低排名位置得到了直接分类和回归监督。 |
| P2SPDOKMFSV7 融合 | P2 stride 4 经四相位 SPD 转 stride 8，与 P3、上采样 Y4 拼到 640 通道，先混合再分为 160/480。160 路用 DCFM、31 核和频率/空间增强，重拼后形成 f3，再经 PAN 形成 f4/f5。 | 极小目标没有独立的 P2 query；密集邻居需要在已有 f3/f4/f5 网格中学到不同的候选。 |
| AIFI 与候选筛选 | 标准 AIFI 只在 P5 做全局交互；f3/f4/f5 是 80²+40²+20²=8,400 个位置。原头先算全部位置的类别 logits，再按每位置最大类分数进 Top-300（约 3.6%）；原生 encoder 框只对入选位置计算。 | 原损失没有为池外位置提供直接的 proposal 分类或定位梯度；共享特征可以间接受益。训练时初始参考框和特征还会 detach，末层附加正样本不能补上这条直接路径。 |
| 原训练损失 | 每个常规监督层默认独立 Hungarian（分类/框 L1/GIoU 成本 2:5:2），匹配对用 VFL 的 IoU 软目标与 5×L1、2×GIoU；DN 使用已知的对应索引。 | 只替换 encoder 辅助分支；解码各层保留原生的一对一去重目标。 |

源码：[模型 YAML](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7.yaml#L31) · [融合块](../ultralytics/nn/extra_modules/hybrid_encoder_p2_spd_okm_fs_v7.py#L223) · [选点](../ultralytics/nn/modules/head.py#L335) · [原生损失](../ultralytics/models/utils/loss.py#L36)。

## 3. 损失设计与首版参数

令 $L_{E}^{\rm orig}$ 为原 encoder Top-300 辅助损失，$L_D^{\rm orig}$ 为三层解码和所有 DN 损失。实现以 `RTDETRDetectionLoss` 为父类，目标为：

$$
L=L_D^{\rm orig} + (1-r)L_E^{\rm orig}
 + r\underbrace{[L_{\rm rep}+0.75 L_{\rm geom}+0.50 L_{\rm slot}]}_{L_E^{\rm GBC}}.
$$

轮 1–5 使用原生损失，轮 6–15 线性接管，第 15 轮末完整替换。不会长程并行保留两套互相冲突的 encoder 目标。空间图从第一次开启就允许零 IoU 候选，不用置信度作为入图门槛；只有争名额项依赖已学出的质量。上述系数是首跑配置，不是已调优结果。

### 3.1 网格归属：先每个目标一个，再给难例一个备用

1. 对每个 GT 在每个实际尺度取中心所在格及其 3×3 邻域，过滤无效 anchor，不要求格点中心在框内。另纳入该 GT 在现有 Top-300 中 IoU ≥.35 的最多两个最高 IoU 候选，以保留已学到的优质框。各 GT 的 token 编号取并集，P3/P4/P5 用全局偏移区分；未入选 token 用同一个 `enc_bbox_head` 与原 anchor 公式算预测框。
2. 每图做一次最大基数、最小成本分配：先给尽可能多的 GT 分到一个独占主代表，再在同一基数下择优。候选成本用 $C_{gj}=d_{gj}+0.25|\ell_j-\ell^*(g)|+2r\,[1-\mathrm{IoU}(b_j,g)]$。$d_{gj}$ 为 anchor 中心至 GT 中心的像素距离，分母取 $\max(8,\sqrt{w_gh_g}/2)$，截断到 4；$\ell^*$ 用 640 口径 √面积 32/96 px 划分 P3/P4/P5 优先级。实际训练用图像宽高还原像素，若改变输入尺寸须同步转换该口径。合法边有限成本，缺配用 dummy，先保证覆盖数。成本和分配停梯度，**不用分类分数决定谁成为正样本**。
3. 对主代表 IoU <.75 的 GT，在所有主代表锁定之后，用未占用候选再做一次独占分配，每个目标最多一个备用、权重 0.5。主代表同时做分类、定位、名额竞争；备用**只回归**，从 encoder 的正负分类采样中忽略。没有备用就用主代表，不能抢其它 GT 的位置。

例如两个短边都小于 8 px 的行人落在同一 P3 格：单格只能给一个，3×3 可以把中心格和邻格分别分给两个目标，再独立训练两个框。重点是先给两个实例各一个代表，然后才允许某个难例拥有第二个定位候选。

### 3.2 先训练好框，再给每个实例一个名额

**框的供给 $L_{\rm geom}$。** 对主代表和备用用原 5×L1 + 2×(1−GIoU)，外加 $0.5\operatorname{Huber}(e_{xy};0.25)+0.25\operatorname{Huber}(e_{wh};0.25)$，使小框的中心/宽高信号在归一化坐标下不被吞掉。$e_{xy}=(\hat c-c_g)/\max((w_g,h_g),8\text{px})$ 逐坐标计算，$e_{wh}=\log(\max((\hat w,\hat h),0.5\text{px})/\max((w_g,h_g),0.5\text{px}))$，坐标均先转为输入像素。Huber 表示 Smooth L1 的 beta 参数；各自两个坐标取均值。GIoU 与中心项在不相交时仍工作；8 px 分母下限避免极小框的像素误差无限放大。每个 GT 内按主 1、备用 0.5 加权平均，再按有主代表的 GT 数归一。

**主代表分类 $L_{\rm rep}$。** 沿用原生 VFL 形式，但改用新分配；主代表真类目标取 $\max(0.25,\mathrm{IoU}_{\rm detach})$，不让刚分到的零 IoU 代表丢掉正类梯度。其它类照常做负标签。负 token 从当前 Top-300 的明确背景与少量全图有效随机背景中抽取，每图至多 2:1 的负 token/主代表，空目标图采 64 个。明确背景要求不是任何 GT 的空间候选、不是主代表或备用，且该候选框对所有 GT 的 IoU ≤.10；其它位置忽略。正项按正数、负项按抽样负数分别归一，负项初值权重 0.5。这样全图背景不会靠数量压倒正类，也不会把备用训练成互相打架的分类目标。

**配额损失 $L_{\rm slot}$。** 每图用原生入选 300 行的 `max_class_logit` 最小值作为真实门槛 $\theta$，停梯度，也不自行改变无效位置在原选点中的处理。在原 Top-300 框上只保留 argmax 类正确且 IoU ≥.35 的边，用带 dummy 的最大权匹配最大化总 $U(q)$，记每个 GT 的当前已入选质量 $q_g^{\rm top}$（无匹配为 0）。主代表未入选、其 IoU ≥.40，且优于当前独占覆盖至少 .03 时，给它一次争名额。权重用 $w_g=\operatorname{sg}([U(q_g)-U(q_g^{\rm top})]_+)$；$U(q)$ 为在 $t\in\{.50,.60,.75,.85\}$ 下的四个 $\sigma((q-t)/.04)$ 的均值。无增益就不挤掉现有覆盖。单图标注超过 300 时，几何仍为所有可分配 GT 提供训练；门槛项只保留覆盖增益最高的至多 300 个不同 GT。

设 $z_g$ 是主代表的**真类**原 logit，$H(x)=0.5\operatorname{softplus}(x/0.5)$。对每个可争取名额的 GT：

$$
\ell_{\rm slot}(g)=w_g\,[ H(\theta+0.20-z_g) + 0.5 H(z_{\rm out}+0.20-z_g)] .
$$

第二项只有在入选池里能确认存在可释放名额时才算。对手可为明确背景，或满足以下全部条件的重复框：不属于任何新主/备用，也未在 Top-300 独占匹配中被使用；其最佳 GT 与预测同类、IoU ≥.50，且对其它所有 GT 的 IoU <.35；另一个已入选、已分给该 GT 的同类框至少有同等 IoU。按覆盖增益由高到低给 GT 选对手，从门槛以上最靠线者开始，每个对手最多使用一次。$z_{\rm out}$ 表示该对手的最大类 logit，保留梯度以主动释放名额；重复对手不作为 $L_{\rm rep}$ 的普通背景。门槛项在无安全对手时仍可工作。$L_{\rm slot}$ 除以全部主代表数，不按少数激活样本放大。备用不做 gate；没有额外覆盖收益时不拉升新候选。验证中用真实原 Top-300 重算覆盖，不用训练代理当作 AP。

## 4. 实现边界与区分已有尝试

- **仅 YAML 和原 criterion 参数看不到全池。** 现有 `RTDETRDetectionModel.loss` 只拿到 `[encoder_top300, dec1, dec2, dec3]`，需要一处仅训练期的 loss 数据桥。激活时在 `loss()` 调用原 `predict()` 的局部范围内用 hook 读取原 `enc_score_head` 的输入特征和输出 logits，并从实际 f3/f4/f5 读取形状。hook 用 `try/finally` 撤销，不改返回值、不给模块挂常驻带图缓存，不把 head 注册成 criterion 子模块。用原有效 anchor 生成函数；对 GT 邻域 token 的去重并集分块、停梯度评估框以分配，再对实际主/备用复用原 bbox MLP 做可反传的计算；已入选框可复用原输出。640 的 8,400 是例子，不能写死。验证、导出、关闭时不安装 hook。[现有 loss 接线](../ultralytics/nn/tasks.py#L456)
- **保留原仓库入口。** 实现时加独立 loss 插件、registry、同网络图 YAML，以及上述受插件开关控制的窄接线；`train.py`、AIFI 模块、解码器源码及其前向、真实选点和推理输出都不改。开启方案会增加训练期稀疏 bbox MLP、匹配时间和显存，需实测其开销；不存在额外解码、推理参数或 FLOPs。用同权重比较挂钩前后原预测逐值一致。用梯度探针核实**实际池外**正 token 的分类、框和融合特征收到梯度，确认不是在重放 Top-300。
- **与本地前案的区别。** QuerySupplementDecode 每图最多补 3 个、旧 rank 路径只看高排位池外候选且特征断梯度，另做一次小解码；其落后在启动补位前也存在，不能用结果单独否定池外监督。CGQS/FQRank 修改打分或附加质量头。GBC 直接在原有 proposal 头上学空间上未入选且未必已有好框的候选，既优化框又释放真实名额。TCR 的末层排序和 DSAW 的几何门控都不进入这个新损失。[旧补位代码](../ultralytics/utils/loss_plugins/query_supplement_loss.py) · [旧实验复盘](FSV7_全量训练结果与失败复盘_20261106.md#querysupplementdecode)
- **与论文方法的区别。** [Co-DETR](https://openaccess.thecvf.com/content/ICCV2023/html/Zong_DETRs_with_Collaborative_Hybrid_Assignments_Training_ICCV_2023_paper.html)、[RT-DETRv3](https://openaccess.thecvf.com/content/WACV2025/papers/Wang_RT-DETRv3_Real-Time_End-to-End_Object_Detection_with_Hierarchical_Dense_Positive_Supervision_WACV_2025_paper.pdf) 已做稠密训练，[MS-DETR](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_MS-DETR_Efficient_DETR_Training_with_Mixed_Supervision_CVPR_2024_paper.html) 已做不额外加 query 的混合监督，[OTA](https://openaccess.thecvf.com/content/CVPR2021/papers/Ge_OTA_Optimal_Transport_Assignment_for_Object_Detection_CVPR_2021_paper.pdf) 已做全局标签分配。本项目可主张的贡献是：**复用原生 proposal 头，按跨 GT 独占归属扩大定位样本；通过原模型真实出现的 Top-300 分数门槛，把一实例一份的预算目标与多候选定位训练接起来。** 不宣称稠密监督或标签分配本身是首创。

## 5. 首跑与淘汰规则

1. **前瞻配对。** A 原生 FSV7，B 完整 GBC。使用相同原生网络初始化、VisDrone、640、batch 4、AdamW、增强和 seed；推荐同 350 轮计划，在 e60/e100 看阶段结果；同配时固定学习率日程，不改成另一种 60 轮日程。不要用 v4 best 做起点。过门再补单独的 `slot_gain=0` 组，辨别候选回归与名额项各自贡献。历史最优 v4 只是外部参考。
2. **先核对因果链。** 记录独占主/备用的分配覆盖、池外正例比例、正类分数与第 300 线差、全池几何 R75、Top-300 **一对一同类** R75、末层 AP75 和重复 FP；分 `<8 px`、`8–16 px`、中大目标和密集图。典型失败是「分数过线但框仍差」「新框到了选点却被末层丢掉」「多给小目标的名额牺牲中大目标」。分别用这些指标确认，避免看整体 loss 猜测机制。
3. **涨点才继续。** e60/e100 的后 10 轮与原生同轮比较；推荐作为继续门槛：小目标 Top-300 独占同类 R75 **+1.0 pp**、总 AP **+0.50 pp**、AP75 **+0.60 pp**，同时检查 AP50 下滑不超过 0.20 pp、重复 FP 未明显上升。这里是投入标准，不是预计收益。长训至少 3 个配对 seed，以总 AP 作为主终点，比较对应验证协议的 v4 历史参考。只有全池 R75 涨但 Top-300 没涨，先查过线/负例；Top-300 涨而 AP 不涨，先查新增 query 在末层的类别与去重，不能把候选覆盖率当成功。

## 6. 仓库实现与启动方式

[完整 GBC YAML](../ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-GBC.yaml) 通过 `loss_name: gbc` 选择 [GBC 插件](../ultralytics/utils/loss_plugins/gbc_loss.py)。在仓库根目录用现有入口跑同种子对照：

```powershell
python train.py --model ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7.yaml --name FSV7-Native-Control --device 0
python train.py --model ultralytics/cfg/models/rt-detr/rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-GBC.yaml --name FSV7-GBC --device 0
```

默认第 1–5 轮为原生损失，第 6–15 轮切换 encoder 辅助损失，`geom_gain=0.75`、`slot_gain=0.50`。只消融配额项可在 GBC 配置副本中将 `slot_gain` 设为 0；`enabled: false` 完全回到原生损失。中断后用 `train.py --resume` 恢复 GBC 自己的检查点。

额外的 `loss_gbc_rep`、`loss_gbc_geom`、`loss_gbc_slot` 计入训练总损失，CSV 仍沿用原三列显示规则。训练时 `model.criterion.last_gbc_stats` 提供最近一批的主/备用及池外数、配额触发数、可释放对手数和门槛差。`top_owned_r75_count` 是训练用独占匹配的计数，不是最终 AP 或按单一阈值重算的覆盖。额外的训练耗时与显存需要用正式配置实测。

可运行 `python tools/verify_gbc_loss.py -v`，检查原生退化、DN 分支、显式池外梯度、空标签、CUDA AMP 和模型推理一致性。当前未启动正式训练，AP 收益仍待配对实验验证。

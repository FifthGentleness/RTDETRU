# 六、主干替换、Stem 改进与 P2 采样增强配置（J–Z 组）

## 共性说明

本组共 65 个配置文件（rtdetr-KAN … rtdetr-WaveletPool），全部基于 RT-DETR-l（`scales: l: [1.00, 1.00, 1024]`），绝大多数以默认架构为基线：Backbone = ConvNormLayer×3 + MaxPool 的 stem + `Blocks(BasicBlock)` 四阶段残差主干（P2–P5，通道 64/128/256/512），Neck = HybridEncoder 的 CCFF（AIFI + 上采样/Concat/RepC3 自顶向下 + lateral 1×1），PAN（3×3 下采样 + Concat/RepC3），解码器统一 `RTDETRDecoder [nc, 256, 300, 4, 8, 3]`。改进方式是"外科手术式"单点替换：只换一个块、一个卷积、一个融合算子或一段拓扑，其余结构逐层保持不变，因此对比实验的消融变量非常干净。

按改进位置可归纳为七大类：(1) 整主干替换（SwinTransformer / TransNeXt / VanillaNet / MobileMamba，backbone 段压缩为一行包装器，层级索引整体前移）；(2) Backbone 残差块内算子替换（`BasicBlock_XXX` 变体，主干拓扑不变）；(3) Stem / 下采样替换（LoGStem、SRFD、RepStem、WaveletPool、LDConv）；(4) ELAN/重参数化主干重构（RepNCSPELAN 系、OREPA、RGC、RepHMS）与双分支混合主干（MutilBackbone 三变体）；(5) Neck 结构级改造（MAFPN 系、ReCalibrationFPN 三尺度版）；(6) Neck 融合算子/块替换（MFM、SDI、WFU、SNI+GSConvE、PST、CSP_PAC、SDFM/PSFM、RepC3 系变体、PSConv）；(7) 轻量化 AIFI（Pola 系列 8 个）与 P2 采样增强（SOEP 系列 5 个）。

模块实现集中在 `ultralytics/nn/extra_modules/block.py`（KAN/LDConv/LFE/LoGStem/MANet/MBRConv/OREPA/PAC/PST/SNI/GSConvE/RGC/SBA/SDI/SRFD/DRFD/WFU/Star/WaveletPool 等几乎全部通用块），其余分布在 `transformer.py`（Pola 系列、AIFI）、`RFAConv.py`、`wtconv2d.py`、`shiftwise_conv.py`（SWC）、`rep_block.py`（WDBB）、`mhafyolo.py`（RepHMS）、`RepStem.py`、`DCMPNet.py`（MFM）、`mona.py`、`semnet.py`（SEFN）、`filc.py`（FMFFN）、`transMamba.py`（SpectralEnhancedFFN）、`EVSSM.py`（EDFFN）、`kan_convs/kagn_conv.py`、`mobileMamba/`、以及 `nn/backbone/`（SwinTransformer、VanillaNet、TransNeXt、rmt.py 的 RetBlock）。源码注释保留了论文章节标记（ICLR2025 PolaFormer、CVPR2025 Mona、CVPR2025 DynamicTanh、CVPR2025 StarNet、ECCV2024 WTConv、AAAI2024 OmniKernel、FasterNet、RetNet 等），便于追溯出处。

### 6.1 整主干替换（Transformer / Mamba / 极简主干）

这 4 个配置把整个 CNN 主干替换为外部主干，backbone 段只剩一行包装器（输出 P2–P5 多尺度特征），head 的 CCFF 结构完全不动；由于主干层数减少，head 层索引整体前移（如 head 从 layer 5 开始）。

| 配置文件 | 改进位置 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-SwinTransformer | 整主干 | `SwinTransformer_Tiny`（nn/backbone/SwinTransformer.py） | 分层窗口多头自注意力 + 移位窗口（shifted window）建模，四阶段金字塔输出多尺度特征，用层次化 Transformer 换掉 CNN 主干以获得全局感受野。 |
| rtdetr-TransNeXt | 整主干 | `transnext_micro`（nn/backbone/TransNeXt/） | TransNeXt 聚合注意力（Aggregated Attention：像素级关注的滑动窗口注意力 + 卷积仿射校准）+ 多尺度 patch embedding，micro 最小规格。 |
| rtdetr-VanillaNet | 整主干 | `vanillanet_5`（nn/backbone/VanillaNet.py） | "极简主义"主干：每阶段仅一层（深度）卷积 + 激活函数雕刻（训练时多分支、推理折叠为单卷积），5 阶段 dims 128→1024，追求极低推理延迟。 |
| rtdetr-MobileMamba | 整主干 | `MobileMamba_T2`（extra_modules/mobileMamba/mobilemamba.py） | 全局分支（Mamba/SSM 选择性扫描，global_ratio 0.8/0.7/0.6）与局部分支（大核深度卷积 k=7/5/3）混合的三阶段轻量主干，embed_dim 144/272/368，线性复杂度全局建模。 |

### 6.2 Backbone 残差块算子替换（BasicBlock 系变体）

主干拓扑（四阶段 `Blocks`、通道、P 层级）完全不变，只把 `BasicBlock` 的 `branch2a/branch2b` 换成新算子，是对"残差块内 3×3 卷积"这一最小单元的消融。

| 配置文件 | 改进位置 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-KAN | Backbone P4/P5 块 | `BasicBlock_KAN` + `KAGNConv2DLayer`（kan_convs/kagn_conv.py） | branch2b 换成 KAN 卷积：标准 3×3 基卷积上叠加可学习多项式基（degree=3）的 Kolmogorov–Arnold 非线性映射，分组计算 + InstanceNorm，增强逐通道非线性表达；仅替换 P4/P5 两阶段（YAML 传 `KAGNConv2DLayer` 名称参数）。 |
| rtdetr-Ortho | Backbone 全部 4 阶段 | `BasicBlock_Ortho` | 在残差块尾部叠加 Gram-Schmidt 正交变换（`F_C_A`）+ 正交注意力 + SE 激励，约束特征各维度正交以抑制通道冗余；每阶段 height 参数 64/32/16/8 控制正交基规模。 |
| rtdetr-PConv | Backbone 全部 4 阶段 | `BasicBlock_PConv`（Partial_conv3，FasterNet） | 部分卷积：仅对 1/4 通道做 3×3 卷积、其余通道恒等直通，同算力下加大深度（split_cat 前向），极致轻量。 |
| rtdetr-PConv-Rep | Backbone 全部 4 阶段 | `BasicBlock_PConv_Rep`（Partial_conv3_Rep） | PConv 的重参数化版本：训练期多分支、推理期融合，兼得部分卷积的效率与重参数化的容量。 |
| rtdetr-Star | Backbone 全部 4 阶段 | `BasicBlock_Star`（Star_Block，CVPR2025 StarNet） | branch2b 换星运算块：`ReLU(W1x) ⊙ W2x` 逐元素相乘把特征投到 3× 高维隐空间再投影回原维度，7×7 DWConv 前置，乘法隐式高维使浅层网络获得强表达。 |
| rtdetr-SWC | Backbone 全部 4 阶段 | `BasicBlock_SWC`（ReparamLargeKernelConv，shiftwise_conv.py） | Shift-wise 卷积：用小核平移拼接逼近大核（P2 用 9、其余 7）的重参数化组卷积（groups=ch_out//16），低开销扩大感受野。 |
| rtdetr-VSS | Backbone P4/P5 块 | `BasicBlock_VSS`（VSSBlock，mamba_vss.py，VMamba） | branch2b 换 Visual State Space 块：2D 选择性扫描（SS2D）四向交叉扫描，线性复杂度长程依赖建模；仅 P4/P5 两阶段（YAML 第 6/7 层）。 |
| rtdetr-WDBB | Backbone 全部 4 阶段 | `BasicBlock_WDBB`（WideDiverseBranchBlock，rep_block.py） | branch2a/2b 均换多样化分支卷积（3×3、1×1、池化等多分支并行，训练后可折叠为单卷积），宽化多样性表征。 |
| rtdetr-WTConv | Backbone 块 + Neck 下采样 | `BasicBlock_WTConv`（WTConv2d，wtconv2d.py） | 小波卷积：Haar/db1 小波分解后在低频/高频子带分别做小核深度卷积再逆变换，用小核获得指数级感受野；PAN 的两个 downsample_convs 也换成 `WTConv2d [256,5,2]`。 |
| rtdetr-RFAConv | Backbone 全部 4 阶段 | `BasicBlock_RFAConv`（RFAConv.py） | 感受野注意力卷积：AvgPool+组卷积为每个感受野生成独立权重，与展开（rearrange）后的特征逐点相乘再卷积，替代卷积核"全图共享"的均匀加权。 |
| rtdetr-RFCAConv | Backbone 全部 4 阶段 | `BasicBlock_RFCAConv` | RFAConv 基础上叠加坐标注意力（CA：H/W 方向池化编码位置），感受野权重进一步方向感知。 |
| rtdetr-RFCBAMConv | Backbone 全部 4 阶段 | `BasicBlock_RFCBAMConv` | RFAConv 基础上叠加 BAM 式注意力（max+avg 池化拼接 → 3×3 conv → Sigmoid 空间图 + SE 通道权重），空间-通道双重加权。 |

- rtdetr-KAN 与 rtdetr-VSS 是"半替换"：只动深层 P4/P5，浅层保持标准 BasicBlock（小目标特征提取不受影响，成本可控）。
- rtdetr-WTConv 的改进横跨 Backbone 与 Neck（同一下采样位置换成小波下采样），属于块替换 + 采样替换的复合配置。

### 6.3 Stem 与下采样改进（LoGStem / SRFD / RepStem / WaveletPool / LDConv）

Stem 是输入端 4× 下采样（原为 ConvNormLayer×3 + MaxPool），这组配置把它整体替换；部分配置同步改造 Neck 下采样，形成"入口-出口"一致的无损/增强采样链。

| 配置文件 | 改进位置 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-LoGStem | Stem | `LoGStem [64]`（block.py，LEGNet） | LoG（高斯-拉普拉斯）滤波器提取边缘 → 深度卷积+2× 下采样 → 高斯平滑残差 + BN → DRFD_LoG 完成 4× 下采样，一步到位且保留边缘/细节，利于小目标。 |
| rtdetr-SRFD | Stem + Neck 下采样 | `SRFD [64]` + `DRFD [256]` | 深度细节保留下采样：7×7 卷积强化后走"深度卷积步长 2 + Cut 像素重排 + MaxPool"三路并行再 1×1 融合，分两级到 4×；Neck 的两个下采样换 DRFD（同思路双路径），信息损失最小化。 |
| rtdetr-RepStem | Stem | `RepStem [64]`（RepStem.py） | 三个 MobileOneBlock（多分支重参数化卷积：k×k 与多个 1×1 分支训练、推理折叠为单卷积）串联完成 4× 下采样，精度-推理延迟折中。 |
| rtdetr-WaveletPool | Stem/上采样/Neck 下采样 | `WaveletPool` + `WaveletUnPool` | 固定 Haar 小波滤波器（无参数）做 2× 下采样，频域无损、抗混叠；Neck 上采样用逆小波（WaveletUnPool）、下采样用 WaveletPool，替换全部 3 处。 |
| rtdetr-LDConv | Stem 降采样 + Neck 下采样 | `LDConv [64,6,2]` / `LDConv [256,6,2]` | 线性变形卷积：(N,1) 形卷积按步长 N 排布 + 偏移预测网络生成任意采样点位置，用"线性核"灵活逼近变形感受野；替换 stem 的 MaxPool 与 PAN 两个 3×3 下采样。 |

### 6.4 ELAN / 重参数化主干重构（RepNCSPELAN 系 / OREPA / RGC / RepHMS）

这组把默认 ResNet 主干整体换成 YOLO 系高效主干（ELAN 结构、ADown 下采样、重参数化卷积），同时把 Neck 的 fpn/pan 融合块从 RepC3 同步换成同族 ELAN 块，形成"主干-颈部一致化"的改造。

| 配置文件 | 改进位置 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-RepNCSPELAN | 全主干 + Neck 块 | `RepNCSPELAN4`（block.py） | YOLOv9 风格：Conv/ADown 下采样 + RepNCSPELAN4（cv1 切半 → 两级 RepNCSP+3×3 Conv 链 → 四分支 Concat），重参数化 CSP-ELAN；Neck 四个融合块同步替换。主干 ADown 通道 128/256/512。 |
| rtdetr-RepNCSPELAN-CAA | 全主干 + Neck 块 | `RepNCSPELAN4_CAA` | 在 RepNCSPELAN4 输出前加 CAA（Context Anchor Attention：AvgPool + 1×11 水平/垂直条形深度卷积 + Sigmoid），以条形大核对齐长上下文引导特征；主干更窄（ADown 128/256/512，块 [128,64,32]→[512,256,128]）。 |
| rtdetr-OREPANCSPELAN | 全主干 + Neck 块 | `OREPANCSPELAN4` | RepNCSPELAN4 内部瓶颈换 OREPA（在线多分支重参数化卷积：多尺度卷积分支 + 池化分支，训练并行、推理折叠），保持全宽 ADown 256/512/1024。 |
| rtdetr-RGCSPELAN | 全主干 | `RGCSPELAN`（Rep-Ghost CSP-ELAN） | ELAN 线性堆叠 + RepConv 过渡 + 1×1 收尾的自定义 CSP-ELAN；主干通道设计 128/256/384/384（P4/P5 收窄为 384），P5 块 repeats=3，head 索引随之偏移。 |
| rtdetr-RepHMS | 全主干 | `RepHMS`（mhafyolo.py，MHAF-YOLO） | 重参数化异质多尺度块：宽/深/核可配的 DepthBottleneckUniv2 多支路（核 3 与 5/7/9 组合，深度可分卷积），四阶段 [2,3,3,2] 深度，兼顾多尺度与部署效率。 |

### 6.5 双分支混合主干（MutilBackbone 系列）

主干改为"两套异构分支并行 + 每阶段融合"：C2f 分支（YOLO 风格梯度流）与 HGBlock 分支（YOLOv7/ELAN-G 风格轻量块，light 模式）并行下采样，逐阶段用融合算子对齐输出（64/128/256），head 不变。

| 配置文件 | 改进位置 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-MutilBackbone-DAF | 主干融合算子 | `DynamicAlignFusion`（block.py） | 两支路 1×1 对齐通道后拼接 3×3 卷积，用可学习标量权重（clamp 至 ±1）动态加权两路再融合，梯度可学的自适应对齐。 |
| rtdetr-MutilBackbone-HAFB | 主干融合算子 | `HAFB [64/128/256, group]` | 层次注意力融合块：每支路特征分别经局部（2×2）与全局（4×4）LocalGlobalAttention 提取，三路拼接 squeeze 后 RepConv 融合（P3 层开分组卷积）。 |
| rtdetr-MutilBackbone-MSGA | 主干融合算子 | `MultiScaleGatedAttn` | 多尺度门控注意力：两支路对齐到 min 通道，MultiscaleFusion 提多尺度上下文，1×1 生成 2 通道选择图对 x/g 互导门控（内容引导），轻量跨支路筛选。 |

### 6.6 Neck/FPN 结构级重构（MAFPN 系 / ReCalibrationFPN 系）

这一组不是换算子，而是重写 CCFF 拓扑：MAFPN 系手工展开多分支非对称 FPN/PAN，ReCalibrationFPN 系用 SBA 链式再校准取代 Upsample+Concat，并扩展检测层数（P2 或 P6）。

| 配置文件 | 改进位置 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-MAFPN | Neck 整体拓扑 | 手写 MAFPN-Y（多分支非对称 FPN/PAN） | Y5 先旁路 3×3 下采样生成第二条 P5 支路；每级 Concat 汇聚 3~4 路（backbone 直连、上层上采样、同级/邻级下采样），P3/P4/P5 三支路独立 RepC3 后再交叉下采样融合，输出 [23,27,31]（128/256/256），跨层信息通路更密。 |
| rtdetr-MFMMAFPN | Neck 整体拓扑 | MAFPN 拓扑 + `MFM` 融合 | 拓扑与 MAFPN 完全一致，所有 Concat（含 4 输入级）替换为 MFM 门控多流融合（SK 风格逐通道 softmax 加权），融合由"无参拼接"升级为"自适应加权"。 |
| rtdetr-ReCalibrationFPN-P345 | Neck 融合方式 | `SBA`（Re-CalibrationFPN） | 摒弃 Upsample+Concat：SBA 把高层/低层特征各自 1×1 压缩到半通道，Sigmoid 门控后互为残差补充（`x + x·g + (1-g)·对方门控上采样`），双向再校准后拼接 3×3 融合；FPN/PAN 全部串联 SBA。 |
| rtdetr-ReCalibrationFPN-P2345 | Neck + 检测层数 | `SBA` 链 + P2 层 | SBA 链向下延伸至 backbone P2/4（层 4，64 通道），再回程三次 SBA，形成 P2–P5 四级输出；解码器 hidden_dim 降为 128（`[nc, 128, 300, ...]`）以抵消 P2 带来的 token 膨胀。 |
| rtdetr-ReCalibrationFPN-P3456 | Backbone + Neck + 检测层数 | `SBA` 链 + P6 阶段 | 主干追加第 5 个 `Blocks [512, BasicBlock, 2, 5]` 得 P6/64，SBA 链从 P6 逐级下探到 P3 再回程，输出 P3–P6 四级（YAML 内注释仍沿用 "P2,P3,P4,P5" 字样，实为 P3-P6），hidden_dim 保持 256。 |

- ReCalibrationFPN 三个变体共享同一 SBA 融合器，仅"覆盖尺度范围"不同（P345 基线 / P2345 小目标 / P3456 大目标），可直接横向对比尺度扩展的收益。

### 6.7 Neck 融合算子替换（Concat / 上采样 / 下采样替换）

CCFF 拓扑不变，只把某类算子替换为更强的融合/重采样模块，是最小侵入的 Neck 改进。

| 配置文件 | 改进位置 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-MFM | 全部 Concat | `MFM [256]`（DCMPNet.py） | 多流融合模块：各输入 1×1 对齐通道后堆叠为 (B,K,C,H,W)，全局池化→MLP→Softmax 生成每流逐通道权重后加权和（SKNet 式选择），支持 2~3 输入。 |
| rtdetr-SDI | 全部 Concat | `SDI` | 语义-细节注入（Semantics and Detail Infusion）：各输入 GSConv 对齐到目标尺度后逐元素连乘（`ans = ∏ GSConv(x_i)`），以乘性方式让语义与细节相互调制。 |
| rtdetr-WFU | FPN 段两处融合 | `WFU`（小波融合上采样） | 对大分辨率特征做 Haar 分解（a/h/v/d 子带），高频 h+v+d 经 RB 卷积、低频 a 与小特征拼接经通道变换，逆小波重建；仅替换 FPN 自顶向下两处（PAN 段仍为 Concat）。 |
| rtdetr-RFPN | 上采样 + 下采样 | `SNI [2]` + `GSConvE` | "Rethinking FPN" 组合：上采样用 SNI（软最近邻插值，nearest×1/4 系数实现次邻点软对齐），下采样用 GSConvE（GSConv 增强：标准+深度卷积并行，单模块生成多感受野纹理特征）。 |
| rtdetr-PST | FPN 两个融合块 | `PST [256,2,0.5]`（金字塔稀疏 Transformer） | cv1/cvup 把当前拼接特征与上层特征（fpn_blocks.0 用 Y5、fpn_blocks.1 用主干 P4）对齐到 128 隐藏通道，n=2 个 PSAttnBlock 做跨尺度稀疏注意力（top-k token 筛选 + 逐窗注意力）后拼接输出；仅替换 fpn_blocks，PAN 仍是 RepC3。 |
| rtdetr-PACAPN | 上采样 + 下采样 + 融合块 | `AttentionUpsample/AttentionDownsample` + `CSP_PAC`（PAC-APN） | 上/下采样换成注意力门控双分支重采样（全局池化 Hardsigmoid 通道门控 × [转置卷积‖上采样+1×1] 或 [步长卷积‖最大池化+1×1]）；融合块换 CSP_PAC（CSP + 并行空洞卷积 PAC），多感受野自顶向下/自底向上。 |

### 6.8 每层并行主干特征融合（SDFM / PSFM）

标准 Neck 输出后，额外从 backbone P3/P4/P5 拉 1×1 投影支路，与 Neck 同尺度输出逐级融合后再送解码器，形成"主干直连-颈部输出"双路精修。

| 配置文件 | 改进位置 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-SDFM | Decoder 前逐层融合 | `SDFM`（superficial detail fusion module） | backbone 特征与 neck 输出经通道注意力（平均池化 SE 式 Recalibrate）+ 空间注意力（1×1 conv）两路加权融合，再 channel_agg 降回单分支，强化表层细节。 |
| rtdetr-PSFM | Decoder 前逐层融合 | `PSFM`（profound semantic fusion module，DenseLayer+GEFM） | 两路输入各经 DenseLayer（1×1 瓶颈 + 密集连接），GEFM 用 DSConv 生成 Q/K/V 做交叉注意力融合（gamma 残差、softmax 归一），语义级深度融合；P3/P4/P5 各一路。 |

- 两者的共同结构：解码器输入从 `[19,22,25]` 改为融合分支 `[29,30,31]`（`c2 = ch[f[1]]`，即融合输出通道=neck 输出 256）。

### 6.9 Neck 融合块 / 卷积替换（RepC3 系变体 + PSConv）

CCFF 中 fpn_blocks / pan_blocks 的 RepC3（或 PAN 下采样卷积）替换为其他块，是最常见的"块消融"位置。

| 配置文件 | 改进位置 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-KANC3 | Neck 全部融合块 | `KANC3` | RepC3 的 3×3 卷积换成 KAGNConv2DLayer（KAN 多项式卷积），全 head 4 块统一替换。 |
| rtdetr-LFEC3 | Neck 全部融合块 | `LFEC3`（LFE，ELAN-SR） | RepC3 内部换 LFE：ShiftConv2d（移位大核）+ 扩张 FFN，源自高效长程注意力超分网络，低频长程增强。 |
| rtdetr-MBRConv3C3 | Neck 全部融合块 | `MBRConv3C3`（MBRConv3） | RepC3 内部换多分支重参数化卷积（3×3 大核 + 1×1 + BN 分支训练、推理折叠单卷积），3×3 版本。 |
| rtdetr-MBRConv5C3 | Neck 全部融合块 | `MBRConv5C3`（MBRConv5） | 同上，5×5 核版本，与 3×3 版构成核尺寸消融对。 |
| rtdetr-RetBlockC3 | Neck 全部融合块 | `RetBlockC3 [.,.,'chunk',8]`（RetBlock，backbone/rmt.py） | RepC3 内部换 RetNet 保留注意力：chunkwise 线性递归模式 + 2D 相对位置编码（RelPos2d），线性复杂度全局交互。 |
| rtdetr-MAN | Neck 全部融合块 | `MANet [256/128, False, 2, 3]` | 多聚合网络：1×1 双路（cv_block_1 逐点、cv_block_2 DWConv 隐式路径）+ chunk 双路 + n=3 个 Bottleneck 递归链，输出 (4+n) 流拼接，聚合度高于 RepC3。 |
| rtdetr-MAN-Faster | Neck 全部融合块 | `MANet_FasterBlock` | MANet 内 Bottleneck 换 Faster_Block（FasterNet PConv 空间混合 + MLP），轻量化聚合单元。 |
| rtdetr-MAN-FasterCGLU | Neck 全部融合块 | `MANet_FasterCGLU` | Faster_Block 的 MLP 换 ConvolutionalGLU（1×1 门控线性单元 + 3×3 DWConv），空间门控 FFN。 |
| rtdetr-MAN-Star | Neck 全部融合块 | `MANet_Star` | MANet 内 Bottleneck 换 Star_Block（星运算：ReLU(W1x)⊙W2x 隐式高维），乘法聚合单元。 |
| rtdetr-PSConv | PAN 两个下采样 | `PSConv [256,3,2]`（风车形卷积） | 非对称 padding 四方向（1×k 横/竖各两条 + k×1）卷积并行 + 2×2 拼接卷积，形成"风车"状多方向感受野；替换 PAN 的两个 3×3 下采样卷积，下采样同时保留多向细节。 |

- MANet 系列的通道设计注意点：fpn_blocks.1（X3）输出显式设为 128（其余三块为 256），与基线 RepC3 全 256 不一致，属低通道瓶颈设计（`MANet [128, False, 2, 3]`）。
- MAN 四变体共用同一外壳、只换单元（Bottleneck / Faster_Block / Faster_Block_CGLU / Star_Block），是与 MBRConv3/5 同风格的受控消融。

### 6.10 SOEP 系列：P2 采样增强（Sample-and-Optimize Embedding P2）

SOEP 系列（5 个）是针对小目标的 P2 信息注入方案：不把 P2 加为独立检测层（避免解码器 token 爆炸），而是把 backbone P2/4 特征经 SPDConv（space-to-depth 4× 通道展开 + 3×3 卷积，128 通道）无损降采样后并入 X3（fpn_blocks.1 的输出级），再过优化算子（CSPOmniKernel 或 PST）后才进入 PAN；同时把 Y4 lateral 1×1 收窄为 128 通道控制成本。五个变体共用该底座，仅替换"融合算子"与"上/下采样算子"：

| 配置文件 | P2 注入 | 优化算子 | 融合算子 | 上采样 | 下采样 |
|---|---|---|---|---|---|
| rtdetr-SOEP | SPDConv(P2)+Concat | `CSPOmniKernel` | Concat | nn.Upsample | Conv 3×2 |
| rtdetr-SOEP-MFM | SPDConv(P2)+MFM(3输入) | `CSPOmniKernel` | `MFM`（全部替换 Concat） | nn.Upsample | Conv 3×2 |
| rtdetr-SOEP-RFPN | SPDConv(P2)+Concat | `CSPOmniKernel` | Concat | `SNI` | `GSConvE` |
| rtdetr-SOEP-MFM-RFPN | SPDConv(P2)+MFM(3输入) | `CSPOmniKernel` | `MFM` | `SNI` | `GSConvE` |
| rtdetr-SOEP-PST | SPDConv(P2)+Concat | `PST`（跨尺度稀疏注意力，与主干 P4 交互） | Concat | nn.Upsample | Conv 3×2 |

- 共同底座：`SNI/上采样 → SPDConv [128] → (MFM|Concat) → CSPOmniKernel|PST → RepC3` 得 X3；Y4 lateral `Conv [128,1,1]`；解码器仍 P3/P4/P5、hidden 256。
- `CSPOmniKernel`（AAAI2024 OmniKernel）：CSP 分支只取 1/4 通道进 OmniKernel（1×31/31×1 条形 + 31×31 大核 + 1×1 全部为组卷积 + FFT 式全局分支），大感受野补足 P2 细节的结构化整理，其余 3/4 通道恒等，开销极低。
- 组合逻辑：MFM 轴（拼接→门控加权）与重采样轴（常规→SNI+GSConvE）正交可叠加；PST 变体则把"优化算子"从卷积式换成跨尺度注意力式，是与主干 P4 直接交互的唯一变体。

### 6.11 Pola 系列：轻量化 AIFI（Polarized self-attention + FFN/归一化/适配器组合）

Pola 系列（8 个）只改一处：把 HybridEncoder 里的 AIFI（标准多头自注意力 + 1×1 FCN FFN + LayerNorm）替换为 `TransformerEncoderLayer_Pola_*`（transformer.py），主干、CCFF、PAN、解码器全部不动。所有变体的公共底座是 `PolaLinearAttention`（ICLR2025 PolaFormer）：极化线性注意力——q 与门控 g 联合投影、kv 低秩分解、逐头可学习 `power`/`scale` 极化权重 + 固定 (20,20) 位置编码 + 深度卷积特征补偿，把 AIFI 的 O(N²) 自注意力降为线性复杂度（AIFI 输入为 20×20 网格，故 hw 参数与之一致）。其余自由度集中在 FFN 类型、归一化方式与适配器：

| 配置文件 | 注意力 | FFN | 归一化 | Mona 适配器 |
|---|---|---|---|---|
| rtdetr-Pola | PolaLinearAttention | 标准 1×1 FCN FFN（fc1→GELU→fc2） | LayerNorm ×2 | 无 |
| rtdetr-Pola-CGLU | PolaLinearAttention | `ConvolutionalGLU`（1×1 门控双分支 + 3×3 DWConv，带 shortcut） | LayerNorm ×2 | 无 |
| rtdetr-Pola-FMFFN | PolaLinearAttention | `FMFFN`（1×1 FFN + 窗口频率调制 WindowFrequencyModulation，filc.py） | LayerNorm ×2 | 无 |
| rtdetr-Pola-SEFN | PolaLinearAttention | `SEFN`（semnet.py：空间引导 FFN，接收注意力前特征 src 作 spatial 先验，含 2× 下采样-上采样增强支路） | LayerNorm ×2 | 无 |
| rtdetr-Pola-SEFN-Mona | PolaLinearAttention | SEFN | LayerNorm ×2 | `Mona` ×2（norm1 后 mona1、输出前 mona2） |
| rtdetr-Pola-SEFN-Mona-DyT | PolaLinearAttention | SEFN | `DynamicTanh` ×2（DyT：α·tanh(βx)，替代 LayerNorm） | Mona ×2 |
| rtdetr-Pola-SEFFN-Mona-DyT | PolaLinearAttention | `SpectralEnhancedFFN`（transMamba.py：DWConv 后 rfft2 频域内逐通道仿射加权再逆变换） | DynamicTanh ×2 | Mona ×2 |
| rtdetr-Pola-EDFFN-Mona-DyT | PolaLinearAttention | `EDFFN`（EVSSM：8×8 patch 分块 rFFT 幅度谱可学习滤波） | DynamicTanh ×2 | Mona ×2 |

- `Mona`（CVPR2025）：紧凑视觉适配器——1×1 降到 64 通道 → 多尺度深度卷积组（3/5/7 平均 + 1×1 投影，残差）→ 1×1 升回，带 LayerNorm 与可学习缩放（初始 1e-6），在冻结/轻量主干上做低成本微调。
- `DynamicTanh`（CVPR2025 DyT）：以逐通道可学习 α/β 的 `α·tanh(βx)` 替代 LayerNorm，去除归一化统计、保留幅度压制。
- 组合矩阵呈"阶梯式消融"：FFN 维度从标准→门控卷积（CGLU）→频域（FMFFN/SEFFN/EDFFN）→空间引导（SEFN）；归一化 LN→DyT 与适配器无→Mona 只在 SEFN 及以后叠加，用于分离"FFN 增强"与"微调适配"两类收益。

### 本组统计

- 配置总数：65（rtdetr-KAN 至 rtdetr-WaveletPool，全部 `.yaml`，scale 均为 l）。
- 按类别：整主干替换 4（SwinTransformer / TransNeXt / VanillaNet / MobileMamba）；Backbone 残差块算子替换 12（KAN、Ortho、PConv、PConv-Rep、Star、SWC、VSS、WDBB、WTConv、RFAConv、RFCAConv、RFCBAMConv）；Stem/采样改进 5（LoGStem、SRFD、RepStem、WaveletPool、LDConv）；ELAN/重参数化主干 5（RepNCSPELAN、RepNCSPELAN-CAA、OREPANCSPELAN、RGCSPELAN、RepHMS）；MutilBackbone 双分支主干 3（DAF、HAFB、MSGA）；Neck 结构级重构 5（MAFPN、MFMMAFPN、ReCalibrationFPN-P345/-P2345/-P3456）；Neck 融合算子替换 6（MFM、SDI、WFU、RFPN、PST、PACAPN）；逐层主干-颈部融合 2（SDFM、PSFM）；Neck 块/卷积替换 10（KANC3、LFEC3、MBRConv3C3、MBRConv5C3、RetBlockC3、MAN、MAN-Faster、MAN-FasterCGLU、MAN-Star、PSConv）；SOEP 系列 5；Pola 系列 8。
- 解码器统一性：除 ReCalibrationFPN-P2345（hidden_dim 128，P2–P5 四层）外，其余全部保持 `RTDETRDecoder [nc, 256, 300, 4, 8, 3]`；AIFI 除 Pola 系列 8 个外全部保留原版。
- 检测层数扩展仅出现在 ReCalibrationFPN 系（P2 或 P6），SOEP 系选择"P2 注入而非 P2 检测层"的轻量路线，两者构成本组小目标策略的对照。
- 复合改进横跨多位置的配置：WTConv（Backbone+Neck 下采样）、LDConv（Stem+Neck 下采样）、SRFD（Stem+Neck 下采样）、WaveletPool（Stem+上采样+Neck 下采样）、RepNCSPELAN 系（主干+Neck 块同步替换）。
- 已知文件内不一致点：MANet 系 X3 输出 128 通道（其余 256）；ReCalibrationFPN-P3456 注释写 "P2,P3,P4,P5" 实为 P3–P6；rtdetr-RFPN/MFMMAFPN 等注释 "cat backbone P3" 与实际索引（P4）不符，均不影响解析。

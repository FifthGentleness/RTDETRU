## 五、卷积算子、重参数化与特征融合改进配置（A–I 组）

本组共 92 个配置文件，覆盖 rtdetr-AKConv 至 rtdetr-IELC3、rtdetr-attention（含 rtdetr-CA-HSFPN、rtdetr-CAA-HSFPN 两个同族配置）。所有配置均基于 RT-DETR-l（`scales: l: [1.00, 1.00, 1024]`，nc=80），保留了 baseline 的总体架构：Decoder 固定为 `RTDETRDecoder, [nc, 256, 300, 4, 8, 3]`（300 查询、4 层解码、8 头、3 层 FFN）。

**baseline 结构（各配置的对照参考）**：Backbone 为 3 层 `ConvNormLayer`（32/32/64 通道）+ `nn.MaxPool2d` 的 P1/P2 下采样 stem，之后 4 个 `Blocks`（内部为 ResNet 式 `BasicBlock`，通道 64→128→256→512，对应 P2–P5）；Head 为经典 CCFF：`input_proj`（1×1 Conv，channel=256，bias=False）→ `AIFI, [1024, 8]`（仅作用于 P5）→ 自顶向下 FPN（`nn.Upsample` + `Concat` + `RepC3, [256, 0.5]`）+ 自底向上 PAN（`Conv, [256, 3, 2]` 下采样 + `Concat` + `RepC3`），输出 X3/F4/F5 三尺度给 Decoder。绝大多数改进只动其中一处，对照阅读 yaml 时以 `[from, repeats, module, args]` 中模块 token 与层级编号（# 4–# 7 为 backbone stage）为锚点即可快速定位。

**改进位置分布规律**（本组统计口径）：
- **Backbone stage 内替换卷积/块**：通过 `Blocks, [ch, BasicBlock_XXX, ...]` 的 `BasicBlock_XXX` 变体实现，如 AKConv、DBB、DCNv2/3/4、DEConv、DRB、Conv3XC、DualConv、Faster_Block 系、ContextGuided、AggregatedAtt、attention（AFGC）等。多数只替换 P5 stage（DCN 系列、DySnake），替换全部 4 个 stage 的有 AKConv、DEConv、DualConv、DBB、DeepDBB、DRB、Conv3XC、Faster 系、ContextGuided。
- **Backbone 整体替换**：rtdetr-CSwomTramsformer（CSWin_tiny）、rtdetr-EfficientFormerv2、rtdetr-EfficientViT、rtdetr-FCM，以及采用 YOLOv9 式 "Conv stem + ADown + NCSPELAN4" 结构的 Conv3XCNCSPELAN / DBBNCSPELAN / DRBNCSPELAN / CSP-FreqSpatial / CSP-MutilScaleEdgeInformation* / CSP-PMSFA。
- **Neck 融合节点替换**：把 `Concat` 换成专用融合模块（CAB、GDSAFusion、ContextGuideFusionModule、DPCF、CAFMFusion/CGAFusion 位于尾部），或把整个 CCFF 重写为 BiFPN/HSFPN/FDPN/CGRFPN/CFPT 等结构，或把 `RepC3` 换成同形异核块（DBBC3、DRBC3、Conv3XCC3、GCConvC3、IELC3、DWRC3、FDConvC3、Converse2DC3）。
- **上/下采样替换**：FPN 侧 `nn.Upsample` → CARAFE / DySample / EUCB / EUCB_SC；PAN 侧 `Conv, [256,3,2]` 下采样 → HWD / FourierConv / GCConv / ContextGuidedBlock_Down。

### 5.1 卷积算子替换类（Backbone BasicBlock 内换核）

这一类通过 `BasicBlock_XXX` 变体把 ResNet BasicBlock 的 `branch2a/branch2b` 常规 3×3 卷积换成特殊算子， aiming 于更强的空间几何建模或更低的计算量。实现集中在 `ultralytics/nn/extra_modules/block.py`。

| 配置文件 | 改进位置 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-AKConv | Backbone 全部 4 个 stage（BasicBlock_AKConv，branch2a/2b 均替换） | AKConv | 任意核卷积（Adaptive Kernel Conv）：p_conv 从特征预测 N 个采样点偏移，经双线性插值在任意形状（初始为 (N,1) 直线排列）的位置卷积，突破固定方形核的形状约束 |
| rtdetr-DCNV2 | Backbone 仅 P5 stage（BasicBlock_DCNv2，branch2b 替换） | DCNv2 | 可变形卷积 v2：conv_offset_mask 同时预测 offset 与调制 mask（sigmoid），`torchvision.deform_conv2d` 执行，增强形变目标建模 |
| rtdetr-DCNV2-Dynamic | Backbone 仅 P5 stage（BasicBlock_DCNv2_Dynamic） | DCNv2_Dynamic | 在 DCNv2 的 offset/mask 生成分支前插入 MPCA 注意力（DCNv2_Offset_Attention），让偏移预测受通道-空间注意力引导 |
| rtdetr-DCNV3 | Backbone 仅 P5 stage（BasicBlock_DCNv3） | DCNv3 | InternImage 的 DCNv3（`ops_dcnv3` 编译算子）：组内共享 offset + 归一化调制权重 + depthwise 分离，精度/效率优于 v2 |
| rtdetr-DCNV4 | Backbone 仅 P5 stage（BasicBlock_DCNv4） | DCNv4 | DCNv4（`DCNv4_op` 编译算子）：稀疏内存访问的更快前向，聚合强度进一步提升，是 DCN 系列的最终版本 |
| rtdetr-DySnake | Backbone 仅 P5 stage（BasicBlock_DySnakeConv） | DySnakeConv | 动态蛇形卷积：标准 3×3 与沿 x/y 轴蛇形移动采样点的 DSConv 三路 concat（输出 3 倍通道），针对管状/细长边界目标 |
| rtdetr-DualConv | Backbone 全部 4 个 stage（BasicBlock_DualConv） | DualConv | 3×3 分组卷积（g=4）与 1×1 逐点卷积并行、逐元素相加，兼顾空间特征与通道混合并降低参数量 |
| rtdetr-ContextGuided | Backbone 全部 4 个 stage（BasicBlock_ContextGuided）+ PAN 下采样换 ContextGuidedBlock_Down | ContextGuidedBlock | CGNet 上下文引导卷积：局部卷积 F_loc（DW）+ 周围上下文 F_sur（膨胀 DW）拼接，再经 FGlo（全局 SE 式）精调 |
| rtdetr-ContextGuidedDown | 仅 PAN 两个下采样节点（#20/#23）换 ContextGuidedBlock_Down | ContextGuidedBlock_Down | 3×3 s2 降采样后做 F_loc/F_sur 局部-上下文卷积 + FGlo 全局注意力，替代普通 stride-2 Conv，下采样时保留上下文信息 |

- DCN 系列四配置互为对照实验：v2 → v2-Dynamic（offset 加注意力）→ v3 → v4，均只动 P5，是研究"形变感受野对检测精度贡献"的典型纵向对比组。
- rtdetr-ContextGuided 是本组唯一"backbone 卷积 + neck 下采样"双位置联动的卷积算子配置。

### 5.2 重参数化卷积类（训练多分支、部署等价合并）

这类配置在训练时使用多分支/多核结构丰富梯度路径，推理时通过 `switch_to_deploy`/`get_equivalent_kernel_bias` 合并为单个等效卷积，精度近似无损、零推理开销。实现同样在 block.py，多处含 `switch_to_deploy`。

| 配置文件 | 改进位置 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-DBB | Backbone 全部 4 个 stage（BasicBlock_DBB，branch2a/2b 均替换） | DiverseBranchBlock | 多样化分支块：k×k 卷积 + 1×1 卷积 + 平均池化分支 + 恒等映射，BN 后可加和等效为单 3×3 卷积 |
| rtdetr-DeepDBB | Backbone 全部 4 个 stage（BasicBlock_DeepDBB） | DeepDiverseBranchBlock | DBB 的"深"版本：多分支内再串联子分支（DBB×深层堆叠），表达力更强，部署同样合并 |
| rtdetr-DRB | Backbone 全部 4 个 stage（BasicBlock_DRB，branch2b 替换） | DilatedReparamBlock(7) | UniRepLKNet 空洞重参数化块：多个不同膨胀率并行分支合并为等效 7×7 大核，扩大感受野且推理不变慢 |
| rtdetr-Conv3XC | Backbone 全部 4 个 stage（BasicBlock_Conv3XC，branch2a/2b 替换） | Conv3XC | SPAN 轻量重参数化卷积：1×1 短路 + 1×1→3×3→1×1 串联主路，`switch_to_deploy` 合并为单 3×3（含 BN 融合） |
| rtdetr-DBBC3 | Neck 的 4 个 RepC3（fpn/pan_blocks）→ DBBC3 | DBBC3 | RepC3 结构中 `m` 序列换成 DiverseBranchBlock(3×3)，重参数化引入 CCFF 融合块 |
| rtdetr-DRBC3 | Neck 的 4 个 RepC3 → DRBC3 | DRBC3 | RepC3 的 `m` 序列换成 DilatedReparamBlock(5)，等效大核重参数化用于多尺度融合 |
| rtdetr-Conv3XCC3 | Neck 的 4 个 RepC3 → Conv3XCC3 | Conv3XCC3 | RepC3 的 `m` 序列换成 Conv3XC，把 SPAN 重参数化卷积引入颈部 |
| rtdetr-GCConv | PAN 两个下采样 Conv → GCConv | GCConv | Rep 风格多分支门控卷积：3×3×2 + 1×1 + BN 恒等四路并行，`get_equivalent_kernel_bias` 合并为单 3×3；提供 deploy 模式 |
| rtdetr-GCConvC3 | Neck 的 4 个 RepC3 → GCConvC3 | GCConvC3 | RepC3 的 `m` 序列换成 GCConv，同上机制的颈部版本 |
| rtdetr-DEConv | Backbone 全部 4 个 stage（BasicBlock_DEConv，branch2b 替换） | DEConv | 细节增强卷积（RCF/CD 混合）：中心差分、角度差分、水平/垂直差分 + 普通卷积五路权重直接相加合并（`switch_to_deploy`），强化边缘梯度响应 |

- **Backbone 整体替换为 YOLOv9 式重参数化结构**（三个 NCSPELAN 配置）：stem 改为 Conv+Conv，stage 间下采样换 `ADown`（平均池化/maxpool 双半通道），主干块为 `RepNCSPELAN4`（ELAN 双分支 + 重参数化 Conv）：

| 配置文件 | 主干块 | 差异 |
|---|---|---|
| rtdetr-Conv3XCNCSPELAN | Conv3XCNCSPELAN4 | RepNCSPELAN4 的 cv2/cv3 内部 NCSP 模块换 Conv3XC 重参数化卷积 |
| rtdetr-DBBNCSPELAN | DBBNCSPELAN4 | NCSP 模块换 DiverseBranchBlock 系列（DBBNCSP） |
| rtdetr-DRBNCSPELAN | DRBNCSPELAN4 | NCSP 模块换 DilatedReparamBlock 系列（DRBNCSP），且 yaml 传入核尺寸参数（13→5 随层级递减）；Neck 的 RepC3 同步换成 DRBNCSPELAN4 |

### 5.3 频域、小波与滤波类

利用 FFT / 小波 / 频率滤波先验替换下采样或增强特征，多用于保留高频细节、获得全局感受野。

| 配置文件 | 改进位置 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-FourierConv | PAN 两个下采样 Conv → FourierConv（输入尺寸 [40,40]/[20,20]） | FourierConv + DeepSparse | FFT 到频域后乘以可学习复数权重（实部/虚部参数）再 IFFT，等价于频域全局滤波，随后 3×3 Conv 融合；用于 MRI 重建的全局感受野块 |
| rtdetr-HWD | PAN 两个下采样 Conv → HWD | HWD | Haar 小波下采样（DWTForward）：LL/HL/LH/HH 四个子带 concat 后 1×1 Conv，替代 stride 卷积，下采样时无损保留高频纹理 |
| rtdetr-FDT | AIFI 之后插入（#9 位置，dim=8 头） | FDT | 频域-空间双域交替 Transformer：RSA（频域旋转/窗口注意力）与 GSA（通道注意力）交替堆叠且共享注意力权重（shared_depth），源自小波特征增强网络 |
| rtdetr-FreqFFPN | FPN 融合节点替换：Upsample+Concat+RepC3 → FreqFusion | FreqFusion | 频率自适应特征融合（TPAMI）：对低分辨率特征生成 ALPF 低通核上采样、对高分辨率特征生成 AHPF 高通核，抑制混叠后再融合 |
| rtdetr-CSP-FreqSpatial | Backbone 整体替换（Conv stem + CSP_FreqSpatial ×4） | CSP_FreqSpatial（C2f 内嵌 FreqSpatial） | FreqSpatial 块：ScharrConv 边缘/频域分支与两层空间卷积并行（时域卷积+频域滤波），CSP 框架下多尺度频率-空间联合建模 |

### 5.4 注意力与 Transformer 类

分两个流派：(a) **AIFI 替换/增强**——把 encoder 中的标准 AIFI 换成其他注意力结构；(b) **Backbone stage 级注意力**——在 `BasicBlock` 尾部或整体 backbone 上加注意力。另有两个 Backbone 级换装的外部 Transformer 主干。

| 配置文件 | 改进位置 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-AdditiveTokenMixer | AIFI → TransformerEncoderLayer_AdditiveTokenMixer | AdditiveTokenMixer | 加性 token 混合：q、k 分别经空间+通道两路操作后相加，(q+k) 经 DWConv 后与 v 逐元素相乘输出；免 qk^T 矩阵乘的轻量注意力 |
| rtdetr-ASSR | AIFI → AttentiveLayer([[20,20]]) | AttentiveLayer（mambaIR） | 图像复原用的窗口自注意力块：W-MHSA + 空间/通道 FFN + layer-scale 残差，替换 AIFI 的全局自注意力（需固定输入分辨率 20×20@P5） |
| rtdetr-CAB | FPN/PAN 四个 Concat 节点 → CrossAttentionBlock | CrossAttentionBlock | 跨模态交叉注意力：一层特征出 Q，另一层出 K/V，实现 FPN 上采特征与 backbone 特征的相互引导（CFNet 风格） |
| rtdetr-CascadedGroupAttention | AIFI → TransformerEncoderLayer_LocalWindowAttention | LocalWindowAttention | EfficientViT 级联分组注意力：特征按头分组后逐头级联计算窗口注意力，前一头输出作为后一头输入，大幅降低计算量 |
| rtdetr-AggregatedAtt | Backbone P4/P5 stage（BasicBlock_AggregatedAtt，yaml 注明 sr_ratio P4=2、P5=1） | TransNeXt_AggregatedAttention | TransNeXt 聚合注意力：相对位置编码（CPB）+ 聚合嵌入 + 空间缩减（sr_ratio），骨干内嵌入像素级聚焦注意力 |
| rtdetr-attention | Backbone P4/P5 stage（BasicBlock_Attention） | AFGCAttention | 自适应细粒度通道注意力：自适应核的一维卷积生成通道权重 + Mix（频域混合）调制，代码中 CoordAtt/BiLevelRouting/ELA/SimAM 被注释、当前启用 AFGC |
| rtdetr-ETB | AIFI 之后插入（#9） | ETB | 频率-空间纠缠块（ECCV2024 FSEL）：Attention_F（频域）与 Attention_S（空间）双注意力并行相加，再接 FFN，用于伪装目标检测 |
| rtdetr-DTAB | AIFI 位置（#9） | DTAB | 双路三重注意力（tsdn）：通道支（DilatedMDTA 通道自注意力 + FFN）与空间支（DilatedOCA 重叠交叉窗口注意力 + FFN）串联 |
| rtdetr-IELC3 | Neck 的 4 个 RepC3 → IELC3 | IEL（IELC3） | 简单门控前馈（CVPR2025 HVI 的 IEL）：DWConv 后 Tanh 激活双分支相乘，作为轻量非线性门控替换 RepC3 内 Bottleneck |
| rtdetr-CTrans | Backbone P3/P4/P5 输出后接 ChannelTransformer（#8），GetIndexOutput 拆回三尺度 | ChannelTransformer | CTNet 跨层通道 Transformer：三尺度各自 patch embedding 后共享 Encoder（通道+空间注意力），再 Reconstruct 回原分辨率，残差相加 |
| rtdetr-CSwomTramsformer | Backbone 整体替换为 CSWin_tiny（单层 from） | CSWin_tiny | CSWin Transformer 主干：十字形（cross-shaped）窗口自注意力，横向/纵向条带窗口内注意力兼顾局部与长程依赖 |
| rtdetr-EfficientFormerv2 | Backbone 整体替换为 efficientformerv2_s0 | EfficientFormerV2-s0 | 4D（卷积式）注意力与全局 MHSA 混合的移动端 Transformer 主干，低延迟高精度 |
| rtdetr-EfficientViT | Backbone 整体替换为 EfficientViT_M0 | EfficientViT-M0 | 级联分组注意力主干（与 rtdetr-CascadedGroupAttention 同源思想做成完整 backbone），GPU 上高吞吐 |
| rtdetr-DGCST | Neck 的 4 个 RepC3 → DGCST | DGCST | 动态分组卷积 Shuffle Transformer：1/4 通道走深度可分离分组卷积 + 通道 shuffle 重排，其余直通，末端残差，模拟 token mixing |
| rtdetr-DGCST2 | Neck 的 4 个 RepC3 → DGCST2 | DGCST2 | DGCST 改进版：分组卷积换成 DGSM（门控空间混合机制），其余结构相同 |
| rtdetr-CSP-PMSFA | Backbone 整体替换（Conv stem + CSP_PMSFA ×4） | CSP_PMSFA（C2f 内嵌 PMSFA） | 部分多尺度特征聚合：3×3、5×5(g)、7×7(g) 分组金字塔卷积逐级下采样聚合，再 1×1 输出，多尺度核覆盖 |

- rtdetr-CSwomTramsformer / EfficientFormerv2 / EfficientViT 三个配置 backbone 只有 1 行（`- [-1, 1, CSWin_tiny, []]` 等），内部多尺度输出由 backbone 类自身 yield 多级特征实现，input_proj 从 2/3/5 号索引取 P3/P4/P5。
- rtdetr-ASSR 名字源自论文复现路径（AttentiveLayer 隶属 mambaIR 复原框架），**并非重参数化方法**，实际改进位置在 AIFI。

### 5.5 颈部与多尺度特征融合类

本组最大的一类：或替换 FPN/PAN 中的融合算子（Concat → 注意力融合），或整体重写颈部拓扑（BiFPN、HSFPN、FDPN、CGRFPN、CFPT），或增加尾部跨尺度融合。

| 配置文件 | 改进位置 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-ASF | FPN Concat → Zoom_cat；尾部加 ScalSeq+Add（#26-#27） | Zoom_cat / ScalSeq | ASF（Attentional Scale Sequence）：Zoom_cat 用 max+avg 池化/插值对齐三尺度再拼接；ScalSeq 把 P3/P4/P5 对齐后沿 3D 卷积通道维聚合再 MaxPool，生成尺度序列特征与 X3 相加 |
| rtdetr-ASF-Dynamic | 同 ASF，ScalSeq → DynamicScalSeq | DynamicScalSeq | 聚合 P4/P5 时改用 DySample（lp）动态上采样替代 nearest 插值，上采样核随内容自适应 |
| rtdetr-ASF-P2 | ASF 基础上引入 P2 层参与融合与检测 | ScalSeq ×2 + Add | 新增 #28-#33：X3 上采样与 backbone P2（#4）Concat/RepC3，再与 P3 经 ScalSeq+Add 融合；通道全降为 128，Decoder 接 4 尺度（#33 P2 级、#27、#22、#25），decoder 维度 128 |
| rtdetr-BIMAFPN | 颈部整体重写为 BiFPN（7 个 Fusion 节点、头通道 192） | Fusion(bifpn) + RepC3 | BiFMW 加权融合：可学习融合权重 ReLU 归一化后逐输入加权求和；含 P3↔P4↔P5 双向重复与跨层直连（#17、#22 等中间节点被二次利用） |
| rtdetr-EMBSFPN | BiFPN 结构 + node 换 CSP_MSCB + 上采样换 EUCB | CSP_MSCB / EUCB / Fusion(bifpn) | EUCB 上采样（Upsample+DWConv+channel shuffle+PWConv）；MSCB 多尺度卷积块按层级配 1-3-5 / 3-5-7 / 5-7-9 核，头通道 128 |
| rtdetr-EMBSFPN-SC | 同 EMBSFPN | EUCB_SC / CSP_MSCB_SC | EUCB 加 Shift_channel_mix（通道移位混合，SPConv 风格），MSCB 同步升级为 SC 版本 |
| rtdetr-HSFPN | 颈部重写为两层筛选式 FPN（HSFPN 结构） | ChannelAttention_HSFPN + ConvTranspose2d | 每层特征经坐标注意力（H/W 方向池化）生成权重图，转置卷积上采样后与上层相乘筛选（Multiply）再相加融合，"高层筛选低层" |
| rtdetr-ELA-HSFPN | 同 HSFPN 结构 | ELA_HSFPN | 筛选算子换 ELA（高效长程注意力：H/W 方向 1D 池化 + 7×1 Conv + GroupNorm），比 CA 更轻的条带注意力 |
| rtdetr-CA-HSFPN | 同 HSFPN 结构 | CA_HSFPN | 筛选算子用 CA 坐标注意力（reduction=8 变体） |
| rtdetr-CAA-HSFPN | 同 HSFPN 结构 | CAA_HSFPN | 筛选算子用 CAA（context anchor attention 类条带注意力，flag 参数版） |
| rtdetr-HSPAN | HSFPN 之上补全 PAN 下行路径（#27-#40） | ChannelAttention_HSFPN | 在 HSFPN 的自顶向下筛选后，再经 Conv s2 下采样逐级与 Y4/Y5 做 Multiply+Add 筛选融合，形成完整双向 HSFPN |
| rtdetr-HS-FPN | 颈部整体重写（CVPR2024 HS-FPN） | HFP + SDP | HFP：DCT 域空间/通道交互对每层特征做高低频筛选；SDP：P5/P4 级语义作为 K/V 对下层 patch 级 cross-attention 注入语义依赖，再逐级 Add/Conv 汇聚 |
| rtdetr-CDFA | AIFI 后插入（#11） | ContrastDrivenFeatureAggregation | 对比驱动特征聚合（CDFA）：Haar 小波把特征分为前景/背景（fg/bg）两组，分别预测局部窗口内注意力权重，以背景对比增强前景聚合 |
| rtdetr-CGAFusion | 尾部新增 P3/P4/P5 与 X3/F4/F5 的融合节点（#29-#31） | CGAFusion | 上下文引导注意力融合：通道注意力 + 空间注意力联合生成像素注意力图，`initial + pattn*x + (1-pattn)*y` 门控加权双路特征 |
| rtdetr-CAFMFusion | 同 CGAFusion 位置（#29-#31） | CAFMFusion | CAFM（3D 卷积实现的通道自注意力，头温度参数）+ 像素注意力门控，同为双路加权和融合，注意力更偏全局通道相关性 |
| rtdetr-ContextGuideFPN | 四个 Concat 节点 → ContextGuideFusionModule | ContextGuideFusionModule | SEAttention 对拼接特征生成两组权重，交叉调制两路输入（`x0*w0 + x1` 与 `x1*w1 + x0`）再拼接，上下文互导融合 |
| rtdetr-DPCF | FPN 两个 Concat 节点 → DPCF | DPCF | 双路径特征组合（SAMamba）：高低层特征各分 4 组 chunk，AdaptiveCombiner 逐组自适应组合后 tail_conv 融合，组级选择性互补 |
| rtdetr-GDSAFusion | 四个 Concat 节点 → GDSAFusion | GDSAFusion（OverLoCK） | 动态稀疏注意力融合：query/key 生成动态卷积核权重 + DilatedReparamBlock 大核 lepe + SE 门控，融合同时做空间动态加权和通道校准 |
| rtdetr-CSFCN | 尾部：P5 → CFC_CRB，P3+P5 → SFC_G2（#26/#27） | CFC_CRB / SFC_G2 | CFC：PSP 金字塔 key/value 与全图 query 做跨尺度上下文注意力 + 局部注意力；SFC：预测 offset 对内容/语义双特征做对齐采样融合（语义特征图上采样对齐） |
| rtdetr-CGRFPN | 颈部整体重写（#11-#28） | PyramidContextExtraction / RCM / FuseBlockMulti / DynamicInterpolationFusion | 金字塔上下文提取（PyramidPoolAgg + RCA 频率选择注意力）→ 每层 RCM（MetaNeXtBlock）筛选 → FuseBlockMulti 高低层互导门控（h_sigmoid）→ DynamicInterpolationFusion 动态插值逐级上传，最后 PAN 收尾 |
| rtdetr-FDPN | 颈部整体重写为 Focus Diffusion Pyramid（#11-#24） | FocusFeature | FocusFeature：P4 主节点把上层上采样、本层 1×1、下层 ADown 三路对齐 concat，再并行 5/7/9/11 四种大核 DW 卷积求和聚焦；两次迭代后加二次 PAN |
| rtdetr-FDPN-DASI | 同 FDPN，FocusFeature → DASI | DASI | 维度感知选择性整合（HCFNet）：高低层特征各 chunk 4 组，Bag 单元逐组在低层/高层/本层间门控选择，再 tail_conv + 残差 |
| rtdetr-GLSA | input_proj 与 lateral_conv（#10/#12/#17）替换为 GLSA | GLSA | 全局-局部空间聚合：特征分半，一路 ContextBlock 全局上下文、一路 GLSAConvBranch 局部细节，拼接 1×1 融合，增强投影特征的表达 |
| rtdetr-GlobalEdgeInformationTransfer1 | 输入先提边缘（#4-#7），backbone 各 stage 后 ConvEdgeFusion 融合 | MutilScaleEdgeInfoGenetator / ConvEdgeFusion | SobelConv 提取边缘 → 逐级 maxpool 得 P3/P4/P5 边缘金字塔；ConvEdgeFusion 将边缘与 stage 特征 concat → 1×1 → 3×3 → 1×1 融合（仅 backbone 内） |
| rtdetr-GlobalEdgeInformationTransfer2 | 同 1，输出前再融合一次边缘（#33-#35） | ConvEdgeFusion | neck 输出的 P3/P4/P5 与原始边缘金字塔再次 ConvEdgeFusion 后进 Decoder，边缘信息"全程直通" |
| rtdetr-GlobalEdgeInformationTransfer3 | 同 1，边缘在 FPN/PAN 中间层逐级注入（#27/#31/#35） | ConvEdgeFusion | 边缘金字塔在 X3、F4、F5 生成节点逐步融合，替代 1/2 的"末端一次性"策略 |
| rtdetr-FDConvC3 | Neck 的 4 个 RepC3 → FDConvC3 | FDConv | 频域动态卷积（CVPR2025）：把卷积核映射到 Fourier 域做频带分解与动态调制（KSM 核选择 + FBM 频域分支），单核参数自适应多频响应 |
| rtdetr-DWRC3 | Backbone P5 stage（`Blocks` 位置换 `- [-1, 3, DWRC3, [512, 2]]`） | DWRC3（RepC3 内嵌 DWR） | DWR 膨胀率残差：3×3 降维后并行 d1/d3/d5 三支膨胀卷积，concat 1×1 融合 + 残差，分级扩大感受野 |
| rtdetr-DWRC3-DRB | 同上 → DWRC3_DRB | DWR_DRB | DWR 中 d3/d5 两支换 DilatedReparamBlock(5/7)，膨胀卷积重参数化为等效大核，感受野更大且推理更快 |

- rtdetr-ASF-P2 是本组**唯一引入 P2 高分辨率检测层**的配置（检测头 4 尺度、通道 128），大目标/小目标极端分布场景可用；代价是 P2 级计算量与显存显著上升。
- rtdetr-CFPT / rtdetr-CFPT-P3456 虽以 P3456 命名，但 CFPT 本体为三尺度层间注意力：

| 配置文件 | 改进位置 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-CFPT | 颈部整体重写：input_proj 后接 CrossLayerChannelAttention + CrossLayerSpatialAttention（#13/#17），GetIndexOutput 拆分后与原特征残差 | CrossLayerChannelAttention / CrossLayerSpatialAttention（CLFT） | 层间通道注意力（P3/P4/P5 层间 token 交换的通道注意力）+ 层间空间注意力（跨层窗口注意力）两步跨层交互，金字塔层级间互补信息直接流动 |
| rtdetr-CFPT-P3456 | CFPT 基础上尾部新增 P6 层（#24 Conv 512 s2），Decoder 4 尺度 | 同上 + Conv | 增加更大感受野的 P6/64 检测层，面向超大目标场景；同样保持 256 通道 |

### 5.6 上采样 / 下采样替换类

只替换 FPN 自顶向下路径的两个 `nn.Upsample`（#11/#16），结构最轻、对比最干净。

| 配置文件 | 改进位置 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-CARAFE | FPN 两个 Upsample → CARAFE | CARAFE | 内容感知重组上采样：特征压缩 → 编码出 (scale·k_up)² 重装核 → PixelShuffle → softmax 归一化，对上采样邻域加权重组，核随内容自适应 |
| rtdetr-DySample | FPN 两个 Upsample → DySample, [2, 'lp'] | DySample | 动态点采样上采样：1×1 Conv 预测每组采样点 offset（lp 风格，基于初始位置网格），亚像素级内容感知插值，零额外分辨率开销 |
| rtdetr-EUCB | FPN 两个 Upsample → EUCB | EUCB | 频率增强上采样块（EMO）：Upsample → DWConv → channel shuffle → PWConv，上采样同时做通道重排与混合 |
| rtdetr-EUCB-SC | FPN 两个 Upsample → EUCB_SC | EUCB_SC | EUCB 附加 Shift_channel_mix（通道移位混合）进一步打散通道冗余 |

### 5.7 边缘信息增强类

面向"边界/轮廓信息在网络深层丢失"的问题，以 Sobel/Scharr 算子或边缘增强分支显式注入边缘特征。

| 配置文件 | 改进位置 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-CSP-MutilScaleEdgeInformationEnhance | Backbone 整体替换（Conv stem + CSP_MutilScaleEdgeInformationEnhance ×4） | MutilScaleEdgeInformationEnhance | 多尺度池化金字塔（bins 3/6/9/12）逐级 1×1+DW 提取，每支接 EdgeEnhancer（边缘增强算子），上采样回原尺寸后与局部分支 concat 融合 |
| rtdetr-CSP-MutilScaleEdgeInformationSelect | 同上 | MutilScaleEdgeInformationSelect | 与 Enhance 结构相同，但在 concat 后插入 DualDomainSelectionMechanism（空间-通道双域选择）先筛选再融合，抑制噪声边缘 |
| rtdetr-DWRC3 / rtdetr-DWRC3-DRB | Backbone P5（见 5.5 表） | DWR / DWR_DRB | 通过多膨胀率分支间接增强大尺度上下文与边缘响应，与边缘类同属"感受野/细节"改进（见 5.5 表格说明） |

- GlobalEdgeInformationTransfer1/2/3 是同一思想的三种注入策略对照（backbone 内 / 输出前 / 颈部逐级），见 5.5 表格。
- rtdetr-CSP-FreqSpatial 的 ScharrConv 分支亦属边缘先验（归入频域类，见 5.3）。

### 5.8 超图 / 高阶关联与特殊架构类

| 配置文件 | 改进位置 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-HyperACE | AIFI 后引入 YOLOv13 的 HyperACE（#12，repeats=2）+ FullPAD_Tunnel 全线门控注入（#15-#17、#23、#28、#32、#36） | HyperACE / FullPAD_Tunnel | HyperACE：FuseModule 融合 P3/P4/P5 后，两条 C3AH 超图自适应关联分支（超边建模高阶关联）与 DSConv 低阶分支并行；FullPAD_Tunnel 用可学习标量门控 `x + gate·enhanced` 把增强特征回注各层 |
| rtdetr-HyperCompute | AIFI 后语义收集（#11-#15 多尺度 AvgPool 拼接）→ HyperComputeModule（超图卷积）+ MANet（#17/#18），再回注 FPN | HyperComputeModule / HyPConv / MANet | Hyper-YOLO 超图计算：按特征距离阈值（threshold=10）建超图，消息传递聚合跨尺度语义，MANet 多支路聚合后并入 CCFF |
| rtdetr-HyperCompute-MFM | 同 HyperCompute，全部 Concat 节点换 MFM | MFM | 多特征混合：拼接后按尺度堆叠，GAP+MLP 生成每尺度权重 softmax 加权求和，替代简单 concat |
| rtdetr-FCM | Backbone 整体替换（FCM_3/FCM_2/FCM_1/FCM + Pzconv），架构变为 P2–P4 两尺度检测 | FCM 系列 / Pzconv | AAAI2025 FCM 频率-通道混合块：特征按 1/4 分割，空间与通道注意力交叉调制两支；Pzconv 为 3/5/7 多核 DW 串联金字塔；Decoder 仅接 #22/#25（P3、P4），通道 128，输出尺度与 baseline 不同 |
| rtdetr-FCM 的特殊性 | — | — | 唯一一个"降尺度"配置：无 P5 检测层、decoder 通道 128，检测头只有 2 尺度（P3/P4），面向小目标为主的数据集 |

### 5.9 其余值得注意的配置

| 配置文件 | 改进位置 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-Converse2D | FPN 两个 Upsample → Converse2D, [256, 3, 2] | Converse2D | "逆卷积"上采样算子（ConverseNet）：以 softmax 归一化的深度核做相关-逆置卷积（converse convolution），scale=2 上采样同时保持深度可分离的低参数量 |
| rtdetr-Converse2DC3 | Neck 的 4 个 RepC3 → Converse2DC3 | Converse2DC3 | RepC3 内部 Bottleneck 的卷积换 Converse2D（block.py `Bottleneck_Converse` 同款机制），把逆卷积算子引入融合块 |

### 本组统计

- **配置总数**：92 个（任务清单 90 个 + 同族 rtdetr-CA-HSFPN、rtdetr-CAA-HSFPN），全部为 RT-DETR-l 规模、P3-P5（或 P2/P3-P4-P5、P3-P6）三/四尺度输出。
- **按改进位置分布**：Backbone stage 内替换卷积/块约 20 个；Backbone 整体替换 10 个（CSwomTramsformer、EfficientFormerv2、EfficientViT、FCM、CSP-FreqSpatial、CSP-MutilScaleEdgeInformation*2、CSP-PMSFA、Conv3XC/DBB/DRB-NCSPELAN 3 个）；Neck 融合节点/结构替换约 40 个（本组主体）；上/下采样替换 7 个（CARAFE、DySample、EUCB×2、Converse2D、HWD、FourierConv 位置类）；AIFI 替换/增强 6 个（AdditiveTokenMixer、ASSR、ETB、DTAB、CascadedGroupAttention、CSP-FreqSpatial 部分重叠）。
- **改进模块家族对照组**（适合消融对比）：DCN v2/v2-Dynamic/v3/v4 四连；Faster / Faster-Rep / Faster-EMA / Faster-Rep-EMA 四连；HSFPN / ELA-HSFPN / CA-HSFPN / CAA-HSFPN / HSPAN 五连筛选算子；DBB / DeepDBB / DRB / Conv3XC / GCConv 重参数化五连及其 C3/NCSPELAN 颈部移植版；GlobalEdgeInformationTransfer 1/2/3 三种注入策略；ASF / ASF-Dynamic / ASF-P2 渐进组合。
- **结构变动最大的配置**：rtdetr-FCM（两尺度、无 P5）、rtdetr-ASF-P2（四尺度、引入 P2）、rtdetr-CFPT-P3456（四尺度、引入 P6）、rtdetr-HyperACE / HyperCompute（超图注入 + 门控回注，节点数最多）。
- **易混淆点**：rtdetr-ASSR 实际是 AIFI 注意力替换（mambaIR AttentiveLayer），与重参数化无关；rtdetr-attention 的 BasicBlock_Attention 当前启用 AFGCAttention（其余注意力被注释）；rtdetr-FDConvC3 的 FDConv 实现在 `FDConv_initialversion.py`；DCNv3/v4 依赖编译算子（`ops_dcnv3`、`DCNv4_op`），离线环境构建需注意。

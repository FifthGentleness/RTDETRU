## 四、Backbone C2f 块替换改进配置

### 共性说明

本组共 **92 个配置**（`rtdetr-C2f-*.yaml`），是本项目中规模最大的一批改进配置。统一模式为：以 `rtdetr-C2f-AP.yaml` 为模板，把 RT-DETR-l 的 Backbone 中 4 处 C2f 模块整体替换为 `C2f_XXX` 改进块，其余结构一字不改。Backbone 通道序列保持 `Conv(64,P1) → Conv(128,P2) → C2f_XXX(128) → Conv(256,P3) → C2f_XXX(256) → Conv(384,P4) → C2f_XXX(384) → Conv(384,P5) → C2f_XXX(384, n=3)`，深度系数、通道宽度（scales `l: [1.00, 1.00, 1024]`）与neck/decoder 均不变。唯一例外是 `rtdetr-C2f-DIMB-HyperACE.yaml`：backbone 用 `C2f_DCMB` 的同时把 Neck 也换成了 YOLOv13 的 HyperACE（超图计算 + C3AH）与 FullPAD_Tunnel 门控残差全管线分发结构，属于"骨干+颈部"双重改造配置。

`C2f_XXX` 包装类的实现高度一致（全部位于 `ultralytics/nn/extra_modules/block.py`）：继承标准 `C2f`，仅把 `self.m`（Bottleneck 序列）替换为对应改进块（单个 token-mixer 块或包了一层 Bottleneck 的算子），因此各配置间参数量差异主要来自内嵌块本身。内嵌块分两类封装：一类是 MetaFormer 风格整块（LayerNorm/BatchNorm + token mixer + FFN，自带残差与 layer scale），直接作为 C2f 的瓶颈单元；另一类仅替换 Bottleneck 中的卷积算子（如 FDConv、DSA、wConv），保留 `cv1/cv2` 双卷积瓶颈形态。

**组合后缀含义**（由代码确认）：
- `-CGLU`：把块内 FFN/MLP 替换为 ConvolutionalGLU（卷积门控线性单元：1×1 升维 → DW 3×3 + GELU 门控 → 1×1 降维），增强空间感知的前馈变换；
- `-EDFFN`：FFN 替换为 EDFFN（EVSSM 的高效动态前馈网络：1×1 升维 → DW 卷积 → GELU 门控乘 → 按 8×8 patch 做 FFT 并用可学习频域参数调制后再逆变换）；
- `-EPGO`：在 SHSA 单头注意力上增加门控分支（1×1→ReLU→1×1→Sigmoid）动态预测保留数 K，对注意力矩阵做 top-K 截断，形成动态稀疏注意力；
- `-KAN`：FFN 替换为 KAN 网络（可学习样条激活函数替代固定激活）；
- `Faster` 前缀：内嵌 FasterNet 风格的部分卷积 PConv（仅对 1/4 通道做 3×3 卷积）以压缩计算量。

**部分配置带非常规 YAML 参数**：`C2f_GlobalFilter` / `C2f_DynamicFilter` 传入 size（160/80/40/20）以匹配各层特征图大小的频域复数权重；`C2f_FourierConv` 传入 [160,160]→[20,20] 的尺寸对；`C2f_RMB` 传入 [40,40]/[20,20] 的 inp_shape（MaIR 的输入形状，用于构建扫描索引）；`C2f_wConv` 传入 [0.9] 的 den 系数（控制加权核的衰减分布）。

**分类总览**：全部 92 个配置可归为六大类——(1) 卷积结构增强类（多尺度核/异形卷积/重参数化卷积，16 个）；(2) 注意力与前馈增强类（窗口/单头/多粒度注意力 + GLU 前馈，33 个）；(3) 频域与全局滤波类（FFT 滤波/小波/热传导，19 个）；(4) 状态空间模型类（Mamba/SSM 变体，15 个）；(5) 门控 CNN 类（MambaOut 及算子替换系列，6 个）；(6) 动态卷积混合与组合改造类（3 个）。

---

### 4.1 卷积结构增强类（16 个）

以新型卷积算子为核心改造 Bottleneck：涵盖异形/风车卷积、多尺度大核、门控卷积、逆卷积以及"训练多分支、推理单分支"的重参数化卷积。共同特点是保持纯卷积形态、部署友好，主要增益来自更灵活的感受野形状与更强的局部特征表达。

| 配置文件 | 改进模块 | 机制与作用 |
|---|---|---|
| rtdetr-C2f-AP.yaml | C2f_AP（APBottleneck） | 非对称填充"风车卷积"瓶颈（红外小目标检测）：4 个零填充分支按 (2,0,2,0)/(0,2,0,2) 等四种旋转对称方式填充后各做 3×3 卷积产出 1/4 通道，拼接后融合，以极低成本获得旋转等变的多方向感受野 |
| rtdetr-C2f-PKI.yaml | C2f_PKIModule（PKIModule） | PKINet 多核聚合块：3/5/7/9/11 五种核的深度卷积并行提取多尺度位置关键信息，串联 CAA 上下文锚定注意力（AvgPool+1×5+5×1 条带卷积）引导远距离上下文 |
| rtdetr-C2f-MSM.yaml | C2f_MSM（MultiShapeKernel） | 多形状核模块：多尺度池化层（DeepPoolLayer）+ 立方/条形注意力 + 多形状卷积核聚合不同几何形态的特征，适合形变目标 |
| rtdetr-C2f-LSBlock.yaml | C2f_LSBlock（LSBlock, CVPR2025 LSNet） | LSNet 轻量块：3×3 深度卷积 + BN + GELU + 两个 1×1 卷积的极简"大-小核分离"瓶颈，参数量与 FLOPs 极低 |
| rtdetr-C2f-gConv.yaml | C2f_gConv（gConvBlock） | 门控卷积块：深度卷积分支 Wv 与门控分支 W3g 逐元素相乘实现自适应特征选择，冗余信息被门控抑制 |
| rtdetr-C2f-wConv.yaml | C2f_wConv（Bottleneck_wConv，weightedConvolution2.0） | 加权卷积：把可学习卷积权重与固定类小波核 Φ（对称系数向量 alfa 的外积，den=0.9）相乘调制，以零参数代价注入带通频率先验 |
| rtdetr-C2f-GCConv.yaml | C2f_GCConv（Bottleneck_GCConv） | 全局上下文卷积：1×1 与 3×3 双分支并行的简化自注意力式全局建模卷积，支持部署时重参数化融合为单 3×3 卷积 |
| rtdetr-C2f-Converse2D.yaml | C2f_Converse2D（Bottleneck_Converse） | ConverseNet（图像修复）的 Converse2D"逆卷积"算子：与转置卷积互逆的深度相关运算，以深度可分离方式建模与普通卷积相反的特征关联方向 |
| rtdetr-C2f-ConverseB.yaml | C2f_ConverseB（ConverseBlock） | 同源的完整 ConverseBlock：LayerNorm + Converse2D 逆卷积注意力 + 前馈的整块结构，作用同上但封装层级更高 |
| rtdetr-C2f-UniConvBlock.yaml | C2f_UniConvBlock（UniConvNet） | UniConvNet 统一卷积块：ConvMod 卷积式注意力（DW 卷积生成 K/V）+ 以 DCNv3 为核心算子的 MLP，用可变形卷积统一局部与全局建模 |
| rtdetr-C2f-SMPCGLU.yaml | C2f_SMPCGLU（SMPCGLU） | SMP（结构重参数化多路径卷积）+ CGLU 门控前馈的组合块：多分支卷积重参数化与门控线性单元叠加 |
| rtdetr-C2f-EBlock.yaml | C2f_EBlock（EBlock, CVPR2025 DarkIR） | 图像修复编码块：1×1 升维 + 多膨胀率深度卷积分支（dilation=1）+ 通道注意力 SCA + 简化前馈，强调高效恢复高频细节 |
| rtdetr-C2f-DBlock.yaml | C2f_DBlock（DBlock, DarkIR） | 同源的解码块：dilation=[1,2,3] 三条膨胀深度卷积分支并联 + SCA 通道注意力，多膨胀率扩大感受野 |
| rtdetr-C2f-MBRConv3.yaml | C2f_MBRConv3（Bottleneck_MBRConv3, ICCV2025 MobileIE） | 多分支重参数化 3×3 卷积：训练时 3×3/1×1/3×1/1×3 四种卷积各配 BN 共 8 路并联，推理时数学等价融合为单个 3×3 卷积，零推理开销增容 |
| rtdetr-C2f-MBRConv5.yaml | C2f_MBRConv5（Bottleneck_MBRConv5） | 同上机制的 5×5 核版本，更大原生感受野，同样训练多分支、推理融合为单分支 |
| rtdetr-C2f-RCB.yaml | C2f_RCB（RepConvBlock, CVPR2025 OverLoCK） | OverLoCK 残差卷积块：DilatedReparamBlock（多膨胀率小核训练并行、部署融合为单个大核深度卷积）+ SEModule + GRN 归一化 MLP，兼得大感受野与部署效率 |

### 4.2 注意力与前馈增强类（33 个）

以各种轻量注意力（单头/窗口/多尺度/多粒度/条带/空间门控）作为 token mixer，或以 GLU 门控前馈增强 MLP。此类配置数量最多，多数来自 2024-2025 年视觉主干论文，核心思路是"部分通道做注意力 + 门控前馈"以压低计算量的同时保留全局交互。

| 配置文件 | 改进模块 | 机制与作用 |
|---|---|---|
| rtdetr-C2f-CAMixer.yaml | C2f_CAMixer（CAMixer） | 内容自适应混合器：预测器（PredictorLG）按内容动态决定各窗口用可变形注意力还是卷积处理，并预测采样偏移，"简单区域走卷积、复杂区域走注意力" |
| rtdetr-C2f-CFBlock.yaml | C2f_CFBlock（CFBlock） | 卷积注意力块：以卷积生成 K/V 的 ConvolutionalAttention（中间通道 64 降维）+ MLP，DropPath 残差结构，计算量低于标准 MHSA |
| rtdetr-C2f-ConvAttn.yaml | C2f_ConvAttn（Bottleneck_ConvAttn, ICCV2025 ESC） | 卷积注意力：仅对 1/4 通道施加 13×13 大核卷积注意力（可学习平面滤波器集合），1×1 聚合，是大核注意力的高效实现 |
| rtdetr-C2f-ESC.yaml | C2f_ESC（ESCBlock, ICCV2025 ESC） | 高效自校准块：窗口自注意力 + 两级"卷积注意力 + ConvFFN"串联 + PLK 大核滤波器几何集成，兼顾窗口全局与大核局部校准 |
| rtdetr-C2f-IEL.yaml | C2f_IEL（IEL, CVPR2025 HVI-CIDNet） | 强度增强层：1×1 升维 → DW 卷积 → 通道对分、各自经 Tanh 门控 DW 卷积后逐元素相乘，用强度门控调制特征（原为 CIDNet 的 FFN 替代件） |
| rtdetr-C2f-SHSA.yaml | C2f_SHSA（SHSABlock, EfficientViT） | 单头自注意力块：仅对部分通道（pdim=32）做单头 QK 注意力、其余通道直通，DW 卷积残差 + 轻量 FFN，线性复杂度全局建模 |
| rtdetr-C2f-SHSA-EPGO.yaml | C2f_SHSA_EPGO | SHSA + EPGO：增加门控分支动态预测保留数 K，对注意力矩阵做 top-K 截断为动态稀疏注意力，抑制无关 token 干扰 |
| rtdetr-C2f-SHSA-CGLU.yaml | C2f_SHSA_CGLU | SHSA 块的 FFN 换成 ConvolutionalGLU 门控前馈版本 |
| rtdetr-C2f-SHSA-EPGO-CGLU.yaml | C2f_SHSA_EPGO_CGLU | 同时叠加 EPGO 动态稀疏注意力与 CGLU 门控前馈，是 SHSA 系列的完全体 |
| rtdetr-C2f-SMAFB.yaml | C2f_SMAFB（SMAFormerBlock） | SMAFormer 块：自混合注意力 SMA（通道混合调制 Q/K/V）+ 高效 MLP（E_MLP，含 DW 卷积），以调制方式融合通道与空间注意力 |
| rtdetr-C2f-SMAFB-CGLU.yaml | C2f_SMAFB_CGLU | SMAFormer 块 + ConvolutionalGLU 前馈版本 |
| rtdetr-C2f-MSMHSA-CGLU.yaml | C2f_MSMHSA_CGLU（MSMHSA_CGLU） | 多尺度多头自注意力 + CGLU：以多尺度池化产生分层 Q/K/V 再做 MHSA，前馈用卷积门控线性单元 |
| rtdetr-C2f-AddutuveBlock.yaml | C2f_AdditiveBlock（CAS-ViT） | 加性 token 混合块：注意力以 dwc(q+k)·v 的加性形式替代乘性 QK^T，避免昂贵的矩阵乘，配合 MetaFormer 式前馈 |
| rtdetr-C2f-AddutuveBlock-CGLU.yaml | C2f_AdditiveBlock_CGLU | 同上加性注意力的 CGLU 前馈版本 |
| rtdetr-C2f-MogaBlock.yaml | C2f_MogaBlock（MogaBlock, MogaNet） | 多阶门控聚合块：多阶深度卷积（0/1/2 阶泰勒式分层感受野）做门控空间聚合 + 通道聚合 FFN（ChannelAggregationFFN），多粒度信息融合 |
| rtdetr-C2f-FMB.yaml | C2f_FMB（FMB, ECCV2024 SMFANet） | 特征混合块：SMFA 空间多尺度特征注意力（局部 DW 卷积 + 全局方差项）+ PCFN 部分卷积前馈，恢复型主干的高效基本单元 |
| rtdetr-C2f-KAT.yaml | C2f_KAT（Kat） | KAT 注意力块：用 KAN（可学习样条激活）替代 QKV 线性投影的自注意力 + FFN，提升注意力变换的表达能力 |
| rtdetr-C2f-Faster-KAN.yaml | C2f_Faster_KAN（Faster_Block_KAN） | FasterNet 部分卷积 PConv 空间混合 + KAN 卷积作为 MLP 的极简高效块，兼顾 PConv 速度与 KAN 非线性表达 |
| rtdetr-C2f-LGLB.yaml | C2f_LGLB（LGLBlock, MobileUViT） | 局部-全局块：LocalAgg 局部聚合（DW 卷积 + 逐点卷积）后接带空间缩减比（sr_ratio）的自注意力，先局部细节后全局交互 |
| rtdetr-C2f-LWGA.yaml | C2f_LWGA（LWGA_Block, LWGANet） | 局部-窗口-全局注意力组：通道四等分，分别经 PA 点注意力、LA 局部注意力、MRA 中程注意力与 GA 全局注意力（按 stage 选配），多距离感受野并行 |
| rtdetr-C2f-LEGM.yaml | C2f_LEGM（LEGM, CVPR2024 DCMPNet） | 局部边缘引导混合块：窗口注意力（含 shift 重标定 rescale/rebias 归一化）+ MLP 残差结构，边缘感知的特征混合 |
| rtdetr-C2f-PPA.yaml | C2f_PPA（PPA, HCFNet） | 并行化补丁感知注意力：多核深度卷积金字塔并行提取 + 通道/空间双重注意力加权融合 |
| rtdetr-C2f-HDRAB.yaml | C2f_HDRAB | 高动态范围注意力块（去雾网络）：CAB 通道注意力块 + 双残差结构与多级池化，增强高低动态范围特征的响应 |
| rtdetr-C2f-RAB.yaml | C2f_RAB（RAB, RADN 去噪网络） | 残差注意力块：卷积残差 + CAB 通道注意力 + SAB 空间注意力的 CBAM 式双重加权 |
| rtdetr-C2f-LFEM.yaml | C2f_LFEM（LFE_Module） | 轻量局部特征增强模块：Conv_Extra 额外卷积 + Scharr 边缘算子 + 高斯平滑组合，显式注入边缘与纹理先验 |
| rtdetr-C2f-DSA.yaml | C2f_DSA（Bottleneck_DSA） | 可变形空间注意力 DSA：1×1 投影 + GELU 后经 DSCNX/DSCNY 一对动态蛇形卷积生成空间注意力图门控原特征，再 1×1 投影并残差；只替换 Bottleneck 的 cv2 |
| rtdetr-C2f-DSAN.yaml | C2f_DSAN（DSAN） | DSA 的 MetaFormer 整块化：BatchNorm + DSA 可变形空间注意力 + Mlp 前馈 + layer scale 残差 |
| rtdetr-C2f-DSAN-EDFFN.yaml | C2f_DSAN_EDFFN | 同上，但前馈换成 EDFFN（FFT 域门控动态前馈），注意力与前馈均带频域/形变增强 |
| rtdetr-C2f-DTAB.yaml | C2f_DTAB（DTAB, AAAI2025 盲点去噪网络） | 双轴 Transformer 块：通道维 MDTA 注意力 + 膨胀重叠交叉注意力 DilatedOCA（窗口重叠+膨胀采样）与各自 FFN 级联 |
| rtdetr-C2f-Strip.yaml | C2f_Strip（StripBlock） | 条带注意力块：沿 H/W 方向的 1×k 条带 MLP（StripMlp）建模长条形上下文，适合细长目标（车道、河道等） |
| rtdetr-C2f-StripCGLU.yaml | C2f_StripCGLU | 条带注意力 + CGLU 门控前馈的组合版本 |
| rtdetr-C2f-ELGCA.yaml | C2f_ELGCA（ELGCA_EncoderBlock） | 扩展局部-全局通道注意力编码块：多头局部门控注意力与全局通道注意力（ELGCA_MLP）级联，先局部再全局 |
| rtdetr-C2f-ELGCA-CGLU.yaml | C2f_ELGCA_CGLU | ELGCA 编码块的 CGLU 前馈版本 |

### 4.3 频域与全局滤波类（19 个）

在傅里叶/小波/分数阶频域进行特征变换或滤波，用全局频谱操作替代/增强空间卷积。共同机制是 `rfft2 → 可学习复数权重调制 → irfft2`，天然具备全局感受野且计算量为 O(N log N)；`vHeat` 则以热传导方程的频域解实现全局建模。

| 配置文件 | 改进模块 | 机制与作用 |
|---|---|---|
| rtdetr-C2f-FCA.yaml | C2f_FCA（FCA, FcaNet） | 频率通道注意力：用多组 DCT 频率分量替代单一 GAP 生成通道描述子，再经 1×1 卷积加权，比 SE 注意力捕获更丰富的通道频率信息 |
| rtdetr-C2f-FFCM.yaml | C2f_FFCM（Fused_Fourier_Conv_Mixer） | 融合傅里叶卷积混合器：FourierUnit 频域变换与空域卷积分支并联融合，频域补全局、空域补局部 |
| rtdetr-C2f-SFHF.yaml | C2f_SFHF（SFHF_Block） | 空间-频率混合块：局部 Token 混合器 + SFHF_FourierUnit（空间频率混合傅里叶单元）+ 全局 Token 混合器三级级联 |
| rtdetr-C2f-FourierConv.yaml | C2f_FourierConv（Bottleneck_FourierConv，MRI 重建） | 傅里叶卷积：DeepSparse 稀疏频域滤波 + 1×1 卷积，yaml 逐层传入特征图尺寸 [160,160]→[20,20] 以对齐频域权重，获得全局感受野 |
| rtdetr-C2f-GlobalFilter.yaml | C2f_GlobalFilter（GlobalFilterBlock, GFNet） | GFNet 全局滤波器：可学习复数频域权重（size 与各层特征图 160/80/40/20 匹配）在 rfft2 域直接调制幅相，前馈用 ConvolutionalGLU |
| rtdetr-C2f-DynamicFilter.yaml | C2f_DynamicFilter（DynamicFilter） | 动态频域滤波器：路由 MLP 依输入内容生成 num_filters=4 组复数频域权重的混合系数，实现"输入自适应的全局滤波 + 逐点变换" |
| rtdetr-C2f-JDPM.yaml | C2f_JDPM（JDPM, ECCV2024 FSEL 伪装检测） | 联合域感知模块：空域与频域并行分支分别感知纹理与全局模式，联合调制融合，强化隐蔽目标的频率线索 |
| rtdetr-C2f-ETB.yaml | C2f_ETB（ETB） | 高效 Transformer 块：空间注意力 Attention_S 与频域注意力 Attention_F 双分支并联，互补空域结构与频谱全局性 |
| rtdetr-C2f-FDT.yaml | C2f_FDT（FDT） | 频域 Transformer：窗口内 FFT 频域注意力（Attention_F）+ 空间注意力（GSA/RSA）混合，共享深度残差 |
| rtdetr-C2f-FMABlock.yaml | C2f_FMABlock（FMABlock, 超分网络） | 傅里叶调制注意力块：FConvMod（FFT 频域注意力 + 条件位置编码 CPE + 多头 layer scale）+ MixFFN（5×5/7×5 动态核 DW 卷积前馈） |
| rtdetr-C2f-FDConv.yaml | C2f_FDConv（Bottleneck_FDConv, CVPR2025） | 频域动态卷积：傅里叶域频带分解（FBM 多频带自适应调制）+ 核空间调制（KSM），把固定卷积核变成频带可组合的动态核 |
| rtdetr-C2f-PFDConv.yaml | C2f_PFDConv（Bottleneck_PFDConv） | FDConv 的部分通道（Partial）变体：仅部分通道走频域动态卷积，压缩计算量 |
| rtdetr-C2f-FasterFDConv.yaml | C2f_FasterFDConv（FasterFDConv） | FDConv 的 FasterNet 式变体：部分卷积结构 + 频域动态卷积，速度优先 |
| rtdetr-C2f-SFSConv.yaml | C2f_SFSConv（Bottleneck_SFSConv, ICLR2025） | 尺度-频率序列卷积：PWC 逐点卷积分半后，一路走 SPU 空间单元、一路走 FrFT/FrGT 分数阶傅里叶/伽博滤波单元，拼接聚合，实现尺度序列与频域联合建模 |
| rtdetr-C2f-PSFSConv.yaml | C2f_PSFSConv（Bottleneck_PSFSConv） | SFSConv 的部分通道变体，降低频域分支开销 |
| rtdetr-C2f-FasterSFSConv.yaml | C2f_FasterSFSConv（FasterSFSConv） | SFSConv 的 FasterNet 式部分卷积变体，速度优先 |
| rtdetr-C2f-vHeat.yaml | C2f_Heat（HeatBlock, CVPR2025 vHeat） | 热传导视觉主干块：Heat2D 在频域解析求解 2D 热传导方程（可学习热扩散率），以"热量扩散"形式实现全局建模，附带频率嵌入与层缩放 |
| rtdetr-C2f-HFERB.yaml | C2f_HFERB（HFERB, ICCV2023 CRAFT） | 高频增强残差块：显式高频提取分支与空域分支融合，用频域视角强化边缘/纹理等高频细节 |
| rtdetr-C2f-FAT.yaml | C2f_FAT（FAT_Block, ICLR2024 FTIC） | 频域自适应 Token 块：Swin 式窗口注意力 + 频域窗口注意力 FDWA（W/SW 交替移位），在窗口内引入频域 token 交互 |

### 4.4 状态空间模型类（Mamba/SSM，15 个）

以选择性扫描 SSM（SS2D 及其简化版）做线性复杂度的全局序列建模，多为 2024-2025 年轻量视觉 Mamba 论文（GroupMamba、MambaVision、MobileMamba、VSSD、EfficientViM、TinyVIM、MaIR 等）。共同特点是把特征图展开为序列做多方向扫描，计算量随分辨率线性增长，并普遍采用通道分组、部分扫描等手段降低开销。

| 配置文件 | 改进模块 | 机制与作用 |
|---|---|---|
| rtdetr-C2f-GroupMamba.yaml | C2f_GroupMamba（GroupMambaLayer） | 分组 Mamba：通道四等分各用一个 SS2D 独立扫描（降低状态维度），SE 式门控聚合四组输出，线性复杂度分组全局建模 |
| rtdetr-C2f-GroupMambaBlock.yaml | C2f_GroupMambaBlock（Block_mamba） | 同源的完整 Mamba 块封装（GroupMamba 论文的 Block 形态），含归一化与前馈 |
| rtdetr-C2f-MambaVision.yaml | C2f_MambaVision（MambaVisionBlock, NVIDIA） | Mamba 块与自注意力块串联：先 SS2D 选择性扫描提全局，再 MHSA 精修，兼顾 Mamba 效率与注意力的精确对齐能力 |
| rtdetr-C2f-MobileMamba.yaml | C2f_MobileMambaBlock（MobileMambaBlock, CVPR2025） | 移动端 Mamba 块：DW 卷积 + FFN 包裹"全局 SSM 分支（global_ratio=0.25）+ 局部多核卷积分支"的三角效率设计，面向高分辨率低延时 |
| rtdetr-C2f-VSSD.yaml | C2f_VSSD（VMAMBA2Block, VSSD） | VMambaV2/MLLA 风格块：2D 选择性扫描 + 非线性简化（去掉冗余归一化）的高效视觉状态空间块 |
| rtdetr-C2f-GLVSS.yaml | C2f_GLVSS（GL_VSS, TGRS2025 UMFormer） | 全局-局部视觉状态空间：MambaBlock 堆栈全局扫描 + LocalFeature 局部特征支路 + FeatureEmbedding 嵌入融合 |
| rtdetr-C2f-SAVSS.yaml | C2f_SAVSS（SAVSSBlock） | 空间聚合视觉状态空间块：SSM 扫描前做空间聚合注意力，引导扫描聚焦显著区域 |
| rtdetr-C2f-EVS.yaml | C2f_EVS（EVS, EVSSM） | 高效视觉状态空间块：SS2D 扫描 + EDFFN 频域门控前馈，并带分块（grid）处理机制以控制大分辨率下的显存 |
| rtdetr-C2f-EfficientVIM.yaml | C2f_EfficientVIM（EfficientViMBlock, CVPR2025） | EfficientViM：HSM-SSD 层级单步多感受野 SSM + 前后两个 DW 卷积，Sigmoid(alpha) 四段可学习残差混合，单次扫描的极致轻量 |
| rtdetr-C2f-EfficientVIM-CGLU.yaml | C2f_EfficientVIM_CGLU | 同上，前馈换成 ConvolutionalGLU 门控前馈 |
| rtdetr-C2f-TVIM.yaml | C2f_TVIM（TViMBlock, TinyVIM） | TinyVIM token 混合块：轻量 2D 选择性扫描（d_state=8, ssm_ratio=1.0）+ MLP，是 EfficientViM 的姊妹设计 |
| rtdetr-C2f-TransMamba.yaml | C2f_TransMamba（TransMambaBlock） | Transformer-Mamba 混合块：Transformer 分支（自注意力+频谱增强 FFN）与 Mamba 分支并联，输出拼接后 1×1 融合 |
| rtdetr-C2f-CSI.yaml | C2f_CSI（CSI） | 通道-空间交互块：通道四等分各接 Mamba（d_model=C/4）+ MLP，再按通道逐一交织重排（channel shuffle）融合，部分通道做 SSM 以省算力 |
| rtdetr-C2f-SBSM.yaml | C2f_SBSM（SBSM, WACV2025 SEMNet） | 蛇形双向序列建模：MambaLayer 扫描（带位置编码）+ SEFN 语义增强前馈，双向序列建模捕捉长程依赖 |
| rtdetr-C2f-RMB.yaml | C2f_RMB（RMB, CVPR2025 MaIR） | MaIR 残差混合块：VMM 层级多向 2D 选择性扫描（scan_len=8）+ MLP，yaml 传入各层 inp_shape，相邻块交替 shift_size=0/2 做窗口移位 |

### 4.5 门控 CNN 类（MambaOut 系列，6 个）

MambaOut（CVPR2025）的核心论断是"不需要序列扫描的视觉任务可用门控 CNN 复现 Mamba 收益"。其 GatedCNNBlock 形态为：LayerNorm → 1×1 升维两份（门控 g 与恒等 i）→ 对部分通道做深度卷积 → GELU(g)·(i,c) → 1×1 降维 + DropPath 残差。本系列把其中唯一的卷积算子换成不同论文的算子，形成"同一骨架、六种内核"的对照实验组。

| 配置文件 | 改进模块 | 机制与作用 |
|---|---|---|
| rtdetr-C2f-mambaout.yaml | C2f_MambaOut（GatedCNNBlock_BCHW） | MambaOut 原版门控 CNN 块：门控乘 + 7×7 部分通道深度卷积，BCHW 布局免序列化重排 |
| rtdetr-C2f-mambaout-UniRepLK.yaml | C2f_MambaOut_UniRepLK（GatedUniRepLKBlock_BCHW） | 门控分支内嵌 UniRepLKNetBlock：多膨胀率重参数化大核深度卷积 + SE + GRU 式 FFN，扩大门控分支感受野 |
| rtdetr-C2f-mambaout-FDConv.yaml | C2f_MambaOut_FDConv（GatedFDConvBlock_BCHW） | 门控分支内嵌 FDConv（CVPR2025 频域动态卷积），给门控 CNN 加频域动态核 |
| rtdetr-C2f-mambaout-DSA.yaml | C2f_MambaOut_DSA（GatedDSABlock_BCHW） | 门控分支内嵌 DSA 可变形空间注意力（动态蛇形卷积对门控），空间形变自适应 |
| rtdetr-C2f-mambaout-LSConv.yaml | C2f_MambaOut_LSConv（GatedLSConvBlock_BCHW, CVPR2025 LSNet） | 门控分支内嵌 LSNet 大-小核分离卷积（LSConv），轻量大感受野 |
| rtdetr-C2f-mambaout-SFSC.yaml | C2f_MambaOut_SFSC（GatedSFSCBlock_BCHW） | 门控分支内嵌 SFSConv（ICLR2025 尺度-频率序列卷积），空频联合建模 |

### 4.6 动态卷积混合与组合改造类（3 个）

以"动态 Inception 多分支卷积 + 动态路由"为核心，其中一例叠加颈部改造，是本组唯一的骨干+颈部联合配置。

| 配置文件 | 改进模块 | 机制与作用 |
|---|---|---|
| rtdetr-C2f-DIMB.yaml | C2f_DCMB（DynamicIncMixerBlock） | 动态 Inception 混合块：DynamicInceptionDWConv2d 多分支不同核深度卷积 + 动态路由权重自适应选择分支，layer scale 残差结构 |
| rtdetr-C2f-DIMB-KAN.yaml | C2f_DCMB_KAN（DynamicCIncMixerBlock_KAN） | 同上 mixer + KAN 网络（可学习样条激活）替代 MLP 前馈，双料表达增强 |
| rtdetr-C2f-DIMB-HyperACE.yaml | C2f_DCMB + HyperACE neck | 双重改造：backbone 用 DCMB 动态 Inception 块，neck 换 YOLOv13 HyperACE（C3AH 超图自适应混合 + 全尺度 Hypercompute）与 FullPAD_Tunnel 门控残差，在 FPN/PAN 各级注入超图增强特征后再送 RTDETRDecoder |

### 本组统计

- **配置总数**：92 个（全部为 RT-DETR-l 骨干 C2f 替换，head 标准型 91 个 + HyperACE 联合改造 1 个）。
- **模块类别数**：6 大类（卷积结构增强 16、注意力与前馈增强 33、频域与全局滤波 19、状态空间模型 15、门控 CNN 6、动态卷积混合与组合 3），涉及独立内嵌模块约 70 种。
- **组合后缀使用**：-CGLU 10 个、-EDFFN 1 个、-EPGO 2 个、-KAN 2 个（均为"注意力/卷积骨干块 + 增强前馈"的对照设计）。
- **同骨架多内核对照**：SHSA 4 个（SHSA / EPGO / CGLU / EPGO+CGLU）、MambaOut 6 个、FDConv 4 个（FDConv/PFDConv/FasterFDConv/MambaOut-FDConv）、SFSConv 4 个（SFSConv/PSFSConv/FasterSFSConv/MambaOut-SFSC），适合做消融对比实验。

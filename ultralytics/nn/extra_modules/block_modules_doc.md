# block.py 模块解析文档

> 文件路径: `ultralytics/nn/extra_modules/block.py`
> 总模块数: 630
> 生成日期: 2026-09-03

---

## 目录

1. [HGBlock 系列 (GhostConv/RepConv)](#1-hgblock-系列)
2. [Dilation-wise Residual (DWR)](#2-dilation-wise-residual-dwr)
3. [OrthoNets 正交网络](#3-orthonets-正交网络)
4. [DCNv2 可变形卷积 v2](#4-dcnv2-可变形卷积-v2)
5. [DCNv2_Dynamic 动态偏移可变形卷积](#5-dcnv2_dynamic-动态偏移可变形卷积)
6. [DCNv3 可变形卷积 v3](#6-dcnv3-可变形卷积-v3)
7. [iRMB 系列倒残差移动块](#7-irmb-系列倒残差移动块)
8. [ResNet18 Attention 注意力模块](#8-resnet18-attention-注意力模块)
9. [DySnakeConv 动态蛇形卷积](#9-dysnakeconv-动态蛇形卷积)
10. [FasterBlock 快速模块](#10-fasterblock-快速模块)
11. [AKConv 任意核卷积](#11-akconv-任意核卷积)
12. [RFAConv 系列感受野注意力卷积](#12-rfaconv-系列感受野注意力卷积)
13. [Conv3XC / SPAB Swift参数免费注意力](#13-conv3xc--spab-swift参数免费注意力)
14. [DilatedReparamBlock / UniRepLKNetBlock](#14-dilatedreparamblock--unireplknetblock)
15. [DRB / DBB 重参数化系列](#15-drb--dbb-重参数化系列)
16. [DualConv / EDLAN 双路卷积](#16-dualconv--edlan-双路卷积)
17. [Attentional Scale Sequence Fusion](#17-attentional-scale-sequence-fusion)
18. [SlimNeck (GSConv)](#18-slimneck-gsconv)
19. [TransNeXt AggregatedAttention](#19-transnext-aggregatedattention)
20. [SDI 语义细节注入](#20-sdi-语义细节注入)
21. [RepVGGBlock](#21-repvggblock)
22. [GOLD-YOLO 系列](#22-gold-yolo-系列)
23. [DCNv4 可变形卷积 v4](#23-dcnv4-可变形卷积-v4)
24. [HS-FPN 注意力系列](#24-hs-fpn-注意力系列)
25. [DySample 动态上采样](#25-dysample-动态上采样)
26. [CARAFE 内容感知重组上采样](#26-carafe-内容感知重组上采样)
27. [HWD 半小波下采样](#27-hwd-半小波下采样)
28. [SWC / VSS / LVMB 系列状态空间模型](#28-swc--vss--lvmb-系列)
29. [YOLOv9 系列 (RepN/ADown)](#29-yolov9-系列)
30. [BiFPN Fusion](#30-bifpn-fusion)
31. [ContextGuidedBlock 上下文引导块](#31-contextguidedblock-上下文引导块)
32. [PAC-APN 并行空洞卷积](#32-pac-apn-并行空洞卷积)
33. [DGSM / DGCST 动态组卷积混洗Transformer](#33-dgsm--dgcst-动态组卷积混洗transformer)
34. [RTM Retention块](#34-rtm-retention块)
35. [PKIModule / FADC 频率自适应空洞卷积](#35-pkimodule--fadc-频率自适应空洞卷积)
36. [FocusFeature / PPA / Deep Feature Downsampling](#36-focusfeature--ppa--deep-feature-downsampling)
37. [CFC / SFC / CAFM 上下文空间特征校准](#37-cfc--sfc--cafm-上下文空间特征校准)
38. [RGCSPELAN / ConvolutionalGLU](#38-rgcspelan--convolutionalglu)
39. [SDFM / GEFM / PSFM 语义融合模块](#39-sdfm--gefm--psfm-语义融合模块)
40. [StarNet Star_Block](#40-starnet-star_block)
41. [KAN Kolmogorov-Arnold网络](#41-kan-kolmogorov-arnold网络)
42. [ContextGuideFusionModule](#42-contextguidefusionmodule)
43. [DEConv 去雾增强卷积](#43-deconv-去雾增强卷积)
44. [SMPCGLU / Heat (vHeat)](#44-smpcglu--heat-vheat)
45. [SBA / PSA](#45-sba--psa)
46. [WaveletPool / WaveletUnPool](#46-waveletpool--waveletunpool)
47. [CSP_PTB 部分Transformer块](#47-csp_ptb-部分transformer块)
48. [GLSA 全局-局部空间聚合](#48-glsa-全局-局部空间聚合)
49. [SPDConv 空间到深度卷积](#49-spdconv-空间到深度卷积)
50. [OmniKernel / CSPOmniKernel](#50-omnikernel--cspomnikernel)
51. [WTConv 小波卷积](#51-wtconv-小波卷积)
52. [RCE / RCM / PCE 矩形自校准模块](#52-rce--rcm--pce-矩形自校准模块)
53. [SMFANet (FMB / SMFA / PCFN)](#53-smfanet-fmb--smfa--pcfn)
54. [gConv / LDConv](#54-gconv--ldconv)
55. [AdditiveBlock / MSCB / MutilScale系列](#55-additiveblock--mscb--mutilscale系列)
56. [MogaBlock / SHSA / SMAFormer](#56-mogablock--shsa--smaformer)
57. [DynamicAlignFusion / EdgeEnhancer](#57-dynamicalignfusion--edgeenhancer)
58. [Fourier系列 (FFCM / SFHF / FreqSpatial)](#58-fourier系列-ffcm--sfhf--freqspatial)
59. [HDRAB / RAB / LFE 边缘特征模块](#59-hdrab--rab--lfe-边缘特征模块)
60. [HyperComputeModule / MANet / HFERB](#60-hypercomputemodule--manet--hferb)
61. [JDPM / ETB / FDT / WFU](#61-jdpm--etb--fdt--wfu)
62. [PSConv / APBottleneck](#62-psconv--apbottleneck)
63. [ELGCA / Strip系列](#63-elgca--strip系列)
64. [MultiScalePCA / FSA](#64-multiscalepca--fsa)
65. [KAT / KAN Transformer](#65-kat--kan-transformer)
66. [DynamicInception / GlobalFilter / DynamicFilter](#66-dynamicinception--globalfilter--dynamicfilter)
67. [HAFB / MambaOut / EfficientVIM](#67-hafb--mambaout--efficientvim)
68. [Mamba系列 (SAVSS / VSSD / TVIM / GroupMamba / MambaVision)](#68-mamba系列)
69. [CrossAttentionBlock / IEL / RCB / FAT / LEGM](#69-crossattentionblock--iel--rcb--fat--legm)
70. [LFEA / LFEM / LoG系列](#70-lfea--lfem--log系列)
71. [FDConv / SFSConv / DSAN / DSA / RMB / SNI](#71-fdconv--sfsconv--dsan--dsa--rmb--sni)
72. [FCM / Pzconv / PST (PointSetTransformer)](#72-fcm--pzconv--pst)
73. [FourierConv / wConv / GLVSS / ESC](#73-fourierconv--wconv--glvss--esc)
74. [MBRConv / ConvAttn / VSSD / TVIM](#74-mbrconv--convattn--vssd--tvim)
75. [DPCF / CSI / UniConvBlock / LGLB / ConverseNet / GCConv / CFBlock / FMABlock / LWGA](#75-dpcf--csi--uniconvblock--lglb--conversenet--gcconv--cfblock--fmablock--lwga)

---

## 1. HGBlock 系列

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `Ghost_HGBlock` | L122 | PPHGNetV2的HG块，使用GhostConv或普通Conv构建，通过squeeze-excitation结构聚合多分支特征 |
| `RepLightConv` | L145 | 轻量级重参数化卷积，由1x1 Conv + RepConv(k)组成 |
| `Rep_HGBlock` | L162 | PPHGNetV2的HG4G块，使用RepLightConv或普通Conv，结构与Ghost_HGBlock类似但使用重参数化卷积 |

---

## 2. Dilation-wise Residual (DWR)

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `DWR` | L189 | 空洞残差模块，使用不同膨胀率(d=1,3,5)的3x3卷积捕获多尺度上下文信息，通过残差连接保持信息流 |
| `DWRC3` | L208 | 基于RepC3结构的DWR变体，将Bottleneck替换为DWR模块，支持stride=2下采样 |
| `C3_DWR` | L222 | C3结构中使用DWR作为子模块 |
| `C2f_DWR` | L228 | C2f结构中使用DWR作为子模块 |

---

## 3. OrthoNets 正交网络

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `GramSchmidtTransform` | L259 | Gram-Schmidt正交变换，生成正交滤波器用于特征压缩，通过单例模式缓存 |
| `Attention_Ortho` | L282 | 正交注意力机制，通过迭代Gram-Schmidt变换将空间信息压缩为通道向量 |
| `BasicBlock_Ortho` | L294 | ResNet BasicBlock + 正交注意力 + SE通道激励，在残差分支上应用正交压缩和通道注意力 |
| `BottleNeck_Ortho` | L343 | ResNet BottleNeck + 正交注意力 + SE通道激励，expansion=4的瓶颈结构 |
| `Bottleneck_Ortho` | L401 | YOLO风格Bottleneck + 正交注意力 + SE通道激励 |
| `C3_Ortho` | L424 | C3结构中使用Bottleneck_Ortho |
| `C2f_Ortho` | L430 | C2f结构中使用Bottleneck_Ortho |

---

## 4. DCNv2 可变形卷积 v2

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `DCNv2` | L439 | 标准可变形卷积v2实现，通过学习偏移量和调制掩码实现自适应感受野，使用torchvision的deform_conv2d算子 |
| `Bottleneck_DCNV2` | L505 | 标准Bottleneck，将第二个卷积替换为DCNv2 |
| `C3_DCNv2` | L513 | C3结构中使用Bottleneck_DCNV2 |
| `C2f_DCNv2` | L519 | C2f结构中使用Bottleneck_DCNV2 |
| `BasicBlock_DCNv2` | L524 | ResNet BasicBlock，将branch2b替换为DCNv2 |
| `BottleNeck_DCNv2` | L530 | ResNet BottleNeck，将branch2b替换为DCNv2 |

---

## 5. DCNv2_Dynamic 动态偏移可变形卷积

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `DCNv2_Offset_Attention` | L540 | 带MPCA注意力的偏移量生成模块，对偏移量进行注意力加权以提升偏移质量 |
| `DCNv2_Dynamic` | L554 | 动态可变形卷积v2，使用DCNv2_Offset_Attention生成偏移量，比标准DCNv2的偏移更具自适应性 |
| `Bottleneck_DCNV2_Dynamic` | L611 | Bottleneck + DCNv2_Dynamic |
| `C3_DCNv2_Dynamic` | L619 | C3 + DCNv2_Dynamic |
| `C2f_DCNv2_Dynamic` | L625 | C2f + DCNv2_Dynamic |
| `BasicBlock_DCNv2_Dynamic` | L630 | BasicBlock + DCNv2_Dynamic |
| `BottleNeck_DCNv2_Dynamic` | L636 | BottleNeck + DCNv2_Dynamic |

---

## 6. DCNv3 可变形卷积 v3

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `DCNV3_YOLO` | L646 | DCNv3的YOLO适配封装，处理通道不匹配时的stem_conv，进行NCHW↔NHWC格式转换 |
| `Bottleneck_DCNV3` | L665 | Bottleneck + DCNV3_YOLO |
| `C3_DCNv3` | L673 | C3 + DCNV3_YOLO |
| `C2f_DCNv3` | L679 | C2f + DCNV3_YOLO |
| `BasicBlock_DCNv3` | L684 | BasicBlock + DCNV3_YOLO |
| `BottleNeck_DCNv3` | L690 | BottleNeck + DCNV3_YOLO |

---

## 7. iRMB 系列倒残差移动块

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `iRMB` | L700 | 倒残差移动块(Inverted Residual Mobile Block)，结合窗口自注意力+深度卷积+SE注意力，支持多种配置(注意力前置/后置、v投影等) |
| `iRMB_Cascaded` | L787 | iRMB的级联变体，使用LocalWindowAttention替代手动窗口注意力实现 |
| `iRMB_DRB` | L832 | iRMB + DilatedReparamBlock变体，将局部卷积替换为空洞重参数化块以扩大感受野 |
| `iRMB_SWC` | L919 | iRMB + Shift-wise Conv变体，将局部卷积替换为ReparamLargeKernelConv以获取大核感受野 |
| `C3_iRMB` | L1006 | C3 + iRMB |
| `C2f_iRMB` | L1012 | C2f + iRMB |
| `BasicBlock_iRMB` | L1017 | BasicBlock + iRMB |
| `BottleNeck_iRMB` | L1023 | BottleNeck + iRMB |
| `C3_iRMB_Cascaded` | L1029 | C3 + iRMB_Cascaded |
| `C2f_iRMB_Cascaded` | L1035 | C2f + iRMB_Cascaded |
| `BasicBlock_iRMB_Cascaded` | L1040 | BasicBlock + iRMB_Cascaded |
| `BottleNeck_iRMB_Cascaded` | L1046 | BottleNeck + iRMB_Cascaded |
| `C3_iRMB_DRB` | L1052 | C3 + iRMB_DRB |
| `C2f_iRMB_DRB` | L1058 | C2f + iRMB_DRB |
| `BasicBlock_iRMB_DRB` | L1063 | BasicBlock + iRMB_DRB |
| `BottleNeck_iRMB_DRB` | L1069 | BottleNeck + iRMB_DRB |
| `C3_iRMB_SWC` | L1078 | C3 + iRMB_SWC |
| `C2f_iRMB_SWC` | L1084 | C2f + iRMB_SWC |
| `BasicBlock_iRMB_SWC` | L1089 | BasicBlock + iRMB_SWC |
| `BottleNeck_iRMB_SWC` | L1095 | BottleNeck + iRMB_SWC |

---

## 8. ResNet18 Attention 注意力模块

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `BasicBlock_Attention` | L1108 | ResNet BasicBlock + AFGCAttention注意力，在残差分支后应用注意力机制增强特征 |
| `BottleNeck_Attention` | L1149 | ResNet BottleNeck + CoordAtt注意力，expansion=4 |
| `HGBlock_Attention` | L1196 | HGBlock + CoordAtt注意力，在squeeze卷积前应用坐标注意力 |
| `Bottleneck_Attention` | L1221 | YOLO风格Bottleneck + CoordAtt注意力 |
| `C2f_Attention` | L1239 | C2f + Bottleneck_Attention |
| `C3_Attention` | L1245 | C3 + Bottleneck_Attention |

---

## 9. DySnakeConv 动态蛇形卷积

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `Bottleneck_DySnakeConv` | L1254 | Bottleneck + DySnakeConv，使用动态蛇形卷积替代标准卷积，输出通道x3后用1x1卷积压缩 |
| `C3_DySnakeConv` | L1266 | C3 + Bottleneck_DySnakeConv |
| `C2f_DySnakeConv` | L1272 | C2f + Bottleneck_DySnakeConv |
| `BasicBlock_DySnakeConv` | L1277 | BasicBlock + DySnakeConv |
| `BottleNeck_DySnakeConv` | L1286 | BottleNeck + DySnakeConv |

---

## 10. FasterBlock 快速模块

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `Partial_conv3` | L1299 | 部分卷积(Partial Convolution)，仅对1/4通道进行3x3卷积，其余通道保持不变，大幅减少计算量 |
| `Faster_Block` | L1326 | 快速模块，由Partial_conv3空间混合 + MLP组成，支持layer_scale和drop_path |
| `Faster_Block_EMA` | L1382 | Faster_Block + EMA注意力，在MLP输出后应用EMA高效多尺度注意力 |
| `Partial_conv3_Rep` | L1402 | 使用RepConv的部分卷积变体 |
| `Faster_Block_Rep` | L1408 | 使用Partial_conv3_Rep的Faster_Block变体 |
| `Faster_Block_Rep_EMA` | L1418 | Faster_Block_Rep + EMA注意力 |
| `C3_Faster` | L1428 | C3 + Faster_Block |
| `C2f_Faster` | L1434 | C2f + Faster_Block |
| `C3_Faster_EMA` | L1439 | C3 + Faster_Block_EMA |
| `C2f_Faster_EMA` | L1445 | C2f + Faster_Block_EMA |
| `C3_Faster_Rep` | L1450 | C3 + Faster_Block_Rep |
| `C2f_Faster_Rep` | L1456 | C2f + Faster_Block_Rep |
| `C3_Faster_Rep_EMA` | L1461 | C3 + Faster_Block_Rep_EMA |
| `C2f_Faster_Rep_EMA` | L1467 | C2f + Faster_Block_Rep_EMA |
| `BasicBlock_PConv` | L1472 | BasicBlock + Partial_conv3 |
| `BottleNeck_PConv` | L1482 | BottleNeck + Partial_conv3 |
| `BasicBlock_PConv_Rep` | L1493 | BasicBlock + Partial_conv3_Rep |
| `BottleNeck_PConv_Rep` | L1503 | BottleNeck + Partial_conv3_Rep |
| `BasicBlock_Faster_Block` | L1514 | BasicBlock + Faster_Block |
| `BasicBlock_Faster_Block_Rep` | L1520 | BasicBlock + Faster_Block_Rep |
| `BasicBlock_Faster_Block_EMA` | L1526 | BasicBlock + Faster_Block_EMA |
| `BasicBlock_Faster_Block_Rep_EMA` | L1532 | BasicBlock + Faster_Block_Rep_EMA |
| `BottleNeck_Faster_Block` | L1538 | BottleNeck + Faster_Block |
| `BottleNeck_Faster_Block_EMA` | L1547 | BottleNeck + Faster_Block_EMA |
| `BottleNeck_Faster_Block_Rep` | L1556 | BottleNeck + Faster_Block_Rep |
| `BottleNeck_Faster_Block_Rep_EMA` | L1565 | BottleNeck + Faster_Block_Rep_EMA |

---

## 11. AKConv 任意核卷积

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `AKConv` | L1578 | 任意核卷积(Arbitrary Kernel Convolution)，通过学习采样偏移量实现任意形状的卷积核，使用双线性插值进行特征重采样 |
| `Bottleneck_AKConv` | L1713 | Bottleneck + AKConv |
| `C3_AKConv` | L1722 | C3 + Bottleneck_AKConv |
| `C2f_AKConv` | L1728 | C2f + Bottleneck_AKConv |
| `BasicBlock_AKConv` | L1733 | BasicBlock + AKConv |
| `BottleNeck_AKConv` | L1740 | BottleNeck + AKConv |

---

## 12. RFAConv 系列感受野注意力卷积

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `Bottleneck_RFAConv` | L1750 | Bottleneck + RFAConv(感受野注意力卷积) |
| `C3_RFAConv` | L1760 | C3 + Bottleneck_RFAConv |
| `C2f_RFAConv` | L1766 | C2f + Bottleneck_RFAConv |
| `BasicBlock_RFAConv` | L1771 | BasicBlock + RFAConv |
| `BottleNeck_RFAConv` | L1778 | BottleNeck + RFAConv |
| `Bottleneck_RFCBAMConv` | L1784 | Bottleneck + RFCBAMConv(感受野通道位注意力卷积) |
| `C3_RFCBAMConv` | L1793 | C3 + Bottleneck_RFCBAMConv |
| `C2f_RFCBAMConv` | L1799 | C2f + Bottleneck_RFCBAMConv |
| `BasicBlock_RFCBAMConv` | L1804 | BasicBlock + RFCBAMConv |
| `BottleNeck_RFCBAMConv` | L1811 | BottleNeck + RFCBAMConv |
| `Bottleneck_RFCAConv` | L1817 | Bottleneck + RFCAConv(感受野坐标注意力卷积) |
| `C3_RFCAConv` | L1826 | C3 + Bottleneck_RFCAConv |
| `C2f_RFCAConv` | L1832 | C2f + Bottleneck_RFCAConv |
| `BasicBlock_RFCAConv` | L1837 | BasicBlock + RFCAConv |
| `BottleNeck_RFCAConv` | L1844 | BottleNeck + RFCAConv |

---

## 13. Conv3XC / SPAB Swift参数免费注意力

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `Conv3XC` | L1854 | Swift 3x3卷积，通过1x1→3x3→1x1的bottleneck结构+skip连接实现，支持部署时融合为单个3x3卷积 |
| `SPAB` | L1917 | Swift Parameter-free Attention Block，使用Conv3XC构建，通过sigmoid自注意力实现无参数注意力机制 |
| `Bottleneck_Conv3XC` | L1938 | Bottleneck + Conv3XC |
| `C3_Conv3XC` | L1948 | C3 + Bottleneck_Conv3XC |
| `C2f_Conv3XC` | L1954 | C2f + Bottleneck_Conv3XC |
| `C3_SPAB` | L1959 | C3 + SPAB |
| `C2f_SPAB` | L1965 | C2f + SPAB |
| `BasicBlock_Conv3XC` | L1970 | BasicBlock + Conv3XC |
| `BottleNeck_Conv3XC` | L1977 | BottleNeck + Conv3XC |
| `Conv3XCC3` | L1983 | RepC3 + Conv3XC |

---

## 14. DilatedReparamBlock / UniRepLKNetBlock

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `DilatedReparamBlock` | L1993 | 空洞重参数化块(UniRepLKNet)，将多个不同膨胀率的小核深度卷积在训练时并行，部署时融合为单个大核卷积 |
| `UniRepLKNetBlock` | L2070 | 通用大核网络块，包含大核深度卷积(DilatedReparamBlock) + SE + FFN(GRU+Linear)，支持部署模式融合 |
| `C3_UniRepLKNetBlock` | L2180 | C3 + UniRepLKNetBlock |
| `C2f_UniRepLKNetBlock` | L2186 | C2f + UniRepLKNetBlock |

---

## 15. DRB / DBB 重参数化系列

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `Bottleneck_DRB` | L2191 | Bottleneck + DilatedReparamBlock |
| `C3_DRB` | L2199 | C3 + Bottleneck_DRB |
| `C2f_DRB` | L2205 | C2f + Bottleneck_DRB |
| `BasicBlock_DRB` | L2210 | BasicBlock + DilatedReparamBlock |
| `BottleNeck_DRB` | L2216 | BottleNeck + DilatedReparamBlock |
| `DRBC3` | L2225 | RepC3 + DilatedReparamBlock |
| `DWR_DRB` | L2235 | DWR + DilatedReparamBlock组合 |
| `DWRC3_DRB` | L2254 | DWRC3 + DilatedReparamBlock |
| `C3_DWR_DRB` | L2268 | C3 + DWR_DRB |
| `C2f_DWR_DRB` | L2274 | C2f + DWR_DRB |
| `BasicBlock_DBB` | L2279 | BasicBlock + DBB(多样化分支块) |
| `BottleNeck_DBB` | L2286 | BottleNeck + DBB |
| `BasicBlock_WDBB` | L2292 | BasicBlock + WDBB(宽DBB) |
| `BottleNeck_WDBB` | L2299 | BottleNeck + WDBB |
| `BasicBlock_DeepDBB` | L2305 | BasicBlock + DeepDBB(深层DBB) |
| `BottleNeck_DeepDBB` | L2312 | BottleNeck + DeepDBB |
| `Bottleneck_DBB` | L2322 | YOLO Bottleneck + DBB |
| `C2f_DBB` | L2329 | C2f + Bottleneck_DBB |
| `C3_DBB` | L2334 | C3 + Bottleneck_DBB |
| `DBBC3` | L2340 | RepC3 + DBB |

---

## 16. DualConv / EDLAN 双路卷积

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `DualConv` | L2350 | 双路卷积，并行使用分组卷积(GroupConv)和逐点卷积(PointwiseConv)，输出为两者之和 |
| `EDLAN` | L2373 | 高效双路层聚合网络，由两个串联DualConv组成 |
| `CSP_EDLAN` | L2381 | CSP结构的EDLAN，类似C2f但使用EDLAN作为子模块 |
| `BasicBlock_DualConv` | L2402 | BasicBlock + DualConv |
| `BottleNeck_DualConv` | L2408 | BottleNeck + DualConv |

---

## 17. Attentional Scale Sequence Fusion

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `Zoom_cat` | L2418 | 多尺度缩放拼接，将大/中/小三个尺度特征对齐到中等尺度后拼接 |
| `ScalSeq` | L2430 | 尺度序列融合，使用3D卷积在尺度维度上融合P3/P4/P5特征 |
| `DynamicScalSeq` | L2461 | 动态尺度序列融合，使用DySample替代最近邻插值进行上采样 |
| `Add` | L2495 | 逐元素相加融合模块 |
| `asf_channel_att` | L2502 | 自适应尺度融合通道注意力，使用1D卷积生成通道权重 |
| `asf_local_att` | L2520 | 自适应尺度融合局部注意力，使用strip注意力分别沿H/W方向生成空间权重 |
| `asf_attention_model` | L2551 | 自适应尺度融合注意力模型，组合通道注意力和局部注意力 |

---

## 18. SlimNeck (GSConv)

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `GSConv` | L2568 | GSConv(Slim-Neck)，将标准卷积分解为逐点卷积+深度卷积，并通过channel shuffle混合信息 |
| `GSBottleneck` | L2592 | GSConv瓶颈块，使用GSConv构建lighting分支 + 1x1 shortcut |
| `GSBottleneckC` | L2606 | 便宜GSConv瓶颈块，使用DWConv作为shortcut |
| `VoVGSCSP` | L2612 | VoVNet风格GSConv CSP模块 |
| `VoVGSCSPC` | L2628 | 便宜VoVGSCSP，使用GSBottleneckC |

---

## 19. TransNeXt AggregatedAttention

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `BasicBlock_AggregatedAtt` | L2639 | BasicBlock + TransNeXt聚合注意力 |
| `BottleNeck_AggregatedAtt` | L2676 | BottleNeck + TransNeXt聚合注意力 |
| `Bottleneck_AggregatedAttention` | L2727 | YOLO Bottleneck + TransNeXt聚合注意力 |
| `C2f_AggregatedAtt` | L2738 | C2f + Bottleneck_AggregatedAttention |
| `C3_AggregatedAtt` | L2743 | C3 + Bottleneck_AggregatedAttention |

---

## 20. SDI 语义细节注入

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `SDI` | L2752 | 语义与细节注入模块(Semantics and Detail Infusion)，使用GSConv对不同尺度特征进行通道对齐 |

---

## 21. RepVGGBlock

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `RepVGGBlock` | L2783 | RepVGG重参数化块，训练时使用3x3+1x1+identity三分支，部署时融合为单个3x3卷积 |

---

## 22. GOLD-YOLO 系列

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `SimFusion_3in` | L2924 | 简单3输入融合，将三个尺度特征对齐后拼接+1x1融合 |
| `SimFusion_4in` | L2946 | 简单4输入融合，将四个尺度特征对齐后拼接 |
| `IFM` | L2966 | 内部融合模块(Inner Fusion Module)，使用RepVGGBlock进行特征融合 |
| `h_sigmoid` | L2979 | 硬sigmoid激活函数: ReLU6(x+3)/6 |
| `InjectionMultiSum_Auto_pool` | L2987 | 多尺度注入求和模块，使用自适应池化对齐尺度后进行加权融合 |
| `PyramidPoolAgg` | L3040 | 金字塔池化聚合，将多尺度特征池化到统一尺寸后拼接+1x1卷积 |
| `Mlp` | L3084 | MLP模块，使用1x1 Conv + 深度卷积 + ReLU6 |
| `DropPath` | L3104 | 随机深度(Drop Path)，用于残差分支的随机丢弃 |
| `GOLDYOLO_Attention` | L3115 | GOLD-YOLO注意力，使用QKV多头注意力机制 |
| `top_Block` | L3148 | GOLD-YOLO顶层块，包含注意力 + MLP |
| `TopBasicLayer` | L3169 | GOLD-YOLO顶层基础层，堆叠多个top_Block |
| `AdvPoolFusion` | L3189 | 自适应池化融合，将两个特征通过池化对齐后拼接 |

---

## 23. DCNv4 可变形卷积 v4

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `DCNV4_YOLO` | L3212 | DCNv4的YOLO适配封装，支持通道不匹配的stem_conv |
| `Bottleneck_DCNV4` | L3229 | Bottleneck + DCNV4_YOLO |
| `C3_DCNv4` | L3237 | C3 + Bottleneck_DCNV4 |
| `C2f_DCNv4` | L3243 | C2f + Bottleneck_DCNV4 |
| `BasicBlock_DCNv4` | L3248 | BasicBlock + DCNV4_YOLO |
| `BottleNeck_DCNv4` | L3254 | BottleNeck + DCNV4_YOLO |

---

## 24. HS-FPN 注意力系列

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `ChannelAttention_HSFPN` | L3264 | HS-FPN通道注意力，使用AvgPool+MaxPool双路径SE注意力 |
| `ELA_HSFPN` | L3285 | HS-FPN高效局部注意力，沿H/W方向使用1D卷积+GroupNorm生成空间权重 |
| `h_sigmoid` | L3303 | 硬sigmoid激活(重复定义) |
| `h_swish` | L3312 | 硬swish激活: x * h_sigmoid(x) |
| `CA_HSFPN` | L3320 | HS-FPN坐标注意力，沿H/W方向编码位置信息后生成注意力权重 |
| `CAA_HSFPN` | L3353 | HS-FPN上下文锚定注意力，使用AvgPool+1x5+5x1深度卷积生成注意力 |
| `Multiply` | L3370 | 逐元素相乘模块 |

---

## 25. DySample 动态上采样

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `DySample` | L3381 | 动态采样上采样器，通过学习采样偏移量实现内容自适应的上采样，支持'lp'和'pl'两种风格，比转置卷积更灵活 |

---

## 26. CARAFE 内容感知重组上采样

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `CARAFE` | L3460 | 内容感知特征重组(CARAFE)，通过预测重组核实现内容自适应的上采样，能聚合大范围上下文信息 |

---

## 27. HWD 半小波下采样

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `HWD` | L3504 | 半小波下采样(Haar Wavelet Downsampling)，使用Haar小波变换进行下采样，同时保留低频和高频信息 |

---

## 28. SWC / VSS / LVMB 系列状态空间模型

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `Bottleneck_SWC` | L3525 | Bottleneck + Shift-wise Conv(ReparamLargeKernelConv) |
| `C3_SWC` | L3533 | C3 + Bottleneck_SWC |
| `C2f_SWC` | L3539 | C2f + Bottleneck_SWC |
| `BasicBlock_SWC` | L3544 | BasicBlock + Shift-wise Conv |
| `BottleNeck_SWC` | L3550 | BottleNeck + Shift-wise Conv |
| `Bottleneck_VSS` | L3563 | Bottleneck + VSSBlock(视觉状态空间块) |
| `C3_VSS` | L3569 | C3 + Bottleneck_VSS |
| `C2f_VSS` | L3575 | C2f + Bottleneck_VSS |
| `C3_LVMB` | L3580 | C3 + LVMB(大视觉Mamba块) |
| `C2f_LVMB` | L3586 | C2f + LVMB |
| `BasicBlock_VSS` | L3591 | BasicBlock + VSSBlock |
| `BottleNeck_VSS` | L3597 | BottleNeck + VSSBlock |

---

## 29. YOLOv9 系列 (RepN/ADown)

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `RepConvN` | L3610 | YOLOv9重参数化卷积N，支持多分支(3x3, 1x1, identity)训练，部署时融合 |
| `RepNBottleneck` | L3712 | YOLOv9重参数化瓶颈块，使用RepConvN |
| `DBBNBottleneck` | L3724 | RepNBottleneck + DBB |
| `OREPANBottleneck` | L3730 | RepNBottleneck + OREPA |
| `DRBNBottleneck` | L3736 | RepNBottleneck + DilatedReparamBlock |
| `Conv3XCNBottleneck` | L3742 | RepNBottleneck + Conv3XC |
| `RepNCSP` | L3748 | YOLOv9重参数化CSP模块 |
| `DBBNCSP` | L3761 | RepNCSP + DBB |
| `OREPANCSP` | L3767 | RepNCSP + OREPA |
| `Conv3XCNCSP` | L3773 | RepNCSP + Conv3XC |
| `DRBNCSP` | L3779 | RepNCSP + DilatedReparamBlock |
| `RepNCSPELAN4` | L3785 | YOLOv9重参数化CSP-ELAN4模块 |
| `DBBNCSPELAN4` | L3805 | RepNCSPELAN4 + DBB |
| `OREPANCSPELAN4` | L3811 | RepNCSPELAN4 + OREPA |
| `DRBNCSPELAN4` | L3817 | RepNCSPELAN4 + DilatedReparamBlock |
| `Conv3XCNCSPELAN4` | L3823 | RepNCSPELAN4 + Conv3XC |
| `ADown` | L3829 | YOLOv9自适应下采样，将通道分为两半分别用AvgPool+Conv和MaxPool+Conv处理 |

---

## 30. BiFPN Fusion

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `Fusion` | L3848 | 多特征融合模块，支持weight(加权求和)/adaptive(自适应权重)/concat(拼接)/bifpn(BiFPN加权)/SDI五种融合方式 |

---

## 31. ContextGuidedBlock 上下文引导块

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `FGlo` | L3890 | 全局细化模块，使用SE-like通道注意力细化局部+上下文联合特征 |
| `ContextGuidedBlock` | L3910 | 上下文引导块(CG Block)，1x1降维 → 局部特征(3x3 DW) + 周围上下文(膨胀3x3 DW) → 全局细化，支持残差连接 |
| `ContextGuidedBlock_Down` | L3945 | 上下文引导下采样块，在下采样同时应用上下文引导，通道翻倍 |
| `C3_ContextGuided` | L3982 | C3 + ContextGuidedBlock |
| `C2f_ContextGuided` | L3988 | C2f + ContextGuidedBlock |
| `BasicBlock_ContextGuided` | L3993 | BasicBlock + ContextGuidedBlock |
| `BottleNeck_ContextGuided` | L3999 | BottleNeck + ContextGuidedBlock |

---

## 32. PAC-APN 并行空洞卷积

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `ParallelAtrousConv` | L4012 | 并行空洞卷积，使用不同膨胀率(d=1,2,3)的3x3卷积并行提取多尺度特征 |
| `CSP_PAC` | L4024 | CSP + ParallelAtrousConv |
| `AttentionUpsample` | L4040 | 注意力上采样，使用全局门控+双分支(ConvTranspose+Upsample)上采样 |
| `AttentionDownsample` | L4063 | 注意力下采样，使用全局门控+双分支(Conv+MaxPool)下采样 |

---

## 33. DGSM / DGCST 动态组卷积混洗Transformer

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `DGSM` | L4090 | 动态组卷积混洗模块，1x1投影 → 分组深度卷积 → channel shuffle → 1x1融合 |
| `DGCST` | L4114 | 动态组卷积混洗Transformer，使用DGSM+残差连接+FFN |
| `DGCST2` | L4144 | DGCST变体，使用DGSM替代简单分组卷积 |

---

## 34. RTM Retention块

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `RetBlockC3%` | L4167 | RepC3 + RetBlock(Retention机制)，使用相对位置编码的并行Retention替代自注意力 |
| `C3_RetBlock` | L4182 | C3 + RetBlock，支持chunk/gn两种Retention实现 |
| `C2f_RetBlock` | L4204 | C2f + RetBlock |

---

## 35. PKIM2Module / FADC 频率自适应空洞卷积

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `GSiLU` | L4226 | 门控SiLU激活函数 |
| `PKIModule_CAA` | L4235 | 位置关键信息模块+CAA注意力 |
| `PKIModule` | L4250 | 位置关键信息模块，使用深度卷积+1x1卷积提取位置关键信息 |
| `C3_PKIModule` | L4286 | C3 + PKIModule |
| `C2f_PKIModule` | L4292 | C2f + PKIModule |
| `RepNCSPELAN4_CAA` | L4297 | RepNCSPELAN4 + CAA上下文锚定注意力 |
| `BasicBlock_FADC` | L4322 | BasicBlock + AdaptiveDilatedConv(频率自适应空洞卷积) |
| `BottleNeck_FADC` | L4328 | BottleNeck + AdaptiveDilatedConv |
| `Bottleneck_FADC` | L4334 | YOLO Bottleneck + AdaptiveDilatedConv |
| `C3_FADC` | L4342 | C3 + Bottleneck_FADC |
| `C2f_FADC` | L4348 | C2f + Bottleneck_FADC |

---

## 36. FocusFeature / PPA / Deep Feature Downsampling

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `FocusFeature` | L4357 | 聚焦特征模块，将三个尺度特征对齐后使用多核深度卷积(5,7,9,11)聚合 |
| `C3_PPA` | L4389 | C3 + PPA(并行化补丁感知注意力) |
| `C2f_PPA` | L4395 | C2f + PPA |
| `Cut` | L4404 | 切片下采样，将特征图按步长2切片为4份后拼接+1x1融合，类似SPDConv |
| `SRFD` | L4420 | 浅层特征下采样(Shallow Residual Feature Downsampling)，7x7增强→2x下采样(ConvD+CutD)→4x下采样(ConvD+MaxD+CutD) |
| `DRFD` | L4479 | 深层特征下采样(Deep Residual Feature Downsampling)，使用CutD+ConvD+MaxD三分支下采样 |

---

## 37. CFC / SFC / CAFM 上下文空间特征校准

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `PSPModule` | L4518 | 金字塔池化模块(PSP)，使用多尺度自适应平均池化(1,2,3,6)提取全局上下文 |
| `LocalAttenModule` | L4540 | 局部注意力模块，使用1x1→3x3+Tanh生成局部空间注意力 |
| `CFC_CRB` | L4569 | 上下文特征校准-通道重校准块 |
| `SFC_G2` | L4625 | 空间特征校准G2，使用组卷积进行空间特征校准 |
| `SpatialAttention_CGA` | L4691 | CGA空间注意力，使用Avg+Max → 7x7 Conv生成空间权重 |
| `ChannelAttention_CGA` | L4704 | CGA通道注意力，使用GAP → SE结构生成通道权重 |
| `PixelAttention_CGA` | L4720 | CGA像素注意力，使用7x7分组卷积生成逐像素权重 |
| `CGAFusion` | L4736 | 上下文引导注意力融合，组合空间+通道+像素注意力进行双特征融合 |
| `CAFM` | L4757 | 卷积与注意力融合模块，结合3D深度卷积(局部)和多头自注意力(全局) |
| `CAFMFusion` | L4811 | CAFM融合，使用CAFM+像素注意力进行双特征融合 |

---

## 38. RGCSPELAN / ConvolutionalGLU

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `RGCSPELAN` | L4832 | 重参数化Ghost CSP-ELAN，使用RepConv+普通Conv构建ELAN结构 |
| `ConvolutionalGLU` | L4866 | 卷积门控线性单元(GLU)，使用1x1→深度3x3+GELU→1x1，将GLU引入CNN架构 |
| `Faster_Block_CGLU` | L4897 | Faster_Block + ConvolutionalGLU替代MLP |
| `C3_Faster_CGLU` | L4946 | C3 + Faster_Block_CGLU |
| `C2f_Faster_CGLU` | L4952 | C2f + Faster_Block_CGLU |
| `BasicBlock_Faster_Block_CGLU` | L4957 | BasicBlock + Faster_Block_CGLU |
| `BottleNeck_Faster_Block_CGLU` | L4963 | BottleNeck + Faster_Block_CGLU |

---

## 39. SDFM / GEFM / PSFM 语义融合模块

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `SDFM` | L4976 | 语义细节融合模块(Semantic Detail Fusion Module)，用于RGB-Depth多模态特征融合 |
| `GEFM` | L5024 | 全局增强融合模块(Global Enhancement Fusion Module) |
| `DenseLayer` | L5065 | 密集连接层，使用DSConv构建密集连接块 |
| `PSFM` | L5090 | 深层语义融合模块(Profound Semantic Fusion Module)，使用DenseLayer+GEFM进行双模态融合 |

---

## 40. StarNet Star_Block

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `Star_Block` | L5108 | StarNet星形块，使用7x7深度卷积 → 星形运算(f1*f2) → 7x7深度卷积，高效的非线性特征变换 |
| `Star_Block_CAA` | L5128 | Star_Block + CAA上下文锚定注意力 |
| `C3_Star` | L5143 | C3 + Star_Block |
| `C2f_Star` | L5149 | C2f + Star_Block |
| `C3_Star_CAA` | L5154 | C3 + Star_Block_CAA |
| `C2f_Star_CAA` | L5160 | C2f + Star_Block_CAA |
| `BasicBlock_Star` | L5165 | BasicBlock + Star_Block |
| `BottleNeck_Star` | L5171 | BottleNeck + Star_Block |

---

## 41. KAN Kolmogorov-Arnold网络

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `Bottleneck_KAN` | L5197 | Bottleneck + KAN卷积(支持FastKAN/KAN/KALN/KACN/KAGN五种变体) |
| `C3_KAN` | L5204 | C3 + Bottleneck_KAN |
| `C2f_KAN` | L5210 | C2f + Bottleneck_KAN |
| `BasicBlock_KAN` | L5215 | BasicBlock + KAN卷积 |
| `BottleNeck_KAN` | L5221 | BottleNeck + KAN卷积 |
| `KANC3` | L5227 | RepC3 + KAN卷积 |

---

## 42. ContextGuideFusionModule

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `ContextGuideFusionModule` | L5237 | 上下文引导融合模块，使用SE注意力对双特征进行交叉加权融合 |

---

## 43. DEConv 去雾增强卷积

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `Bottleneck_DEConv` | L5262 | Bottleneck + DEConv(去雾增强卷积) |
| `C3_DEConv` | L5271 | C3 + Bottleneck_DEConv |
| `C2f_DEConv` | L5277 | C2f + Bottleneck_DEConv |
| `BasicBlock_DEConv` | L5282 | BasicBlock + DEConv |
| `BottleNeck_DEConv` | L5288 | BottleNeck + DEConv |

---

## 44. SMPCGLU / Heat (vHeat)

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `SMPCGLU` | L5301 | SMP卷积门控线性单元 |
| `C3_SMPCGLU` | L5322 | C3 + SMPCGLU |
| `C2f_SMPCGLU` | L5328 | C2f + SMPCGLU |
| `Mlp_Heat` | L5337 | Heat MLP，使用2D频率域实现的MLP |
| `LayerNorm2d` | L5357 | 2D层归一化 |
| `Heat2D` | L5364 | vHeat 2D热传导算子，在频率域实现2D热传导方程求解，用于全局建模 |
| `HeatBlock` | L5469 | vHeat块，使用Heat2D + MLP，支持频率嵌入和层缩放 |
| `C3_Heat` | L5540 | C3 + HeatBlock |
| `C2f_Heat` | L5546 | C2f + HeatBlock |

---

## 45. SBA / PSA

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `SBA` | L5561 | 选择性双向注意力模块(Selective Bidirectional Attention)，在高低特征间进行双向门控融合 |
| `PSA_Attention` | L5600 | 部分自注意力(Partial Self-Attention)，使用QKV注意力+位置编码(PE) |
| `PSA` | L5628 | 部分自注意力模块，将通道分为两部分：一部分保持不变，一部分应用自注意力+FFN |

---

## 46. WaveletPool / WaveletUnPool

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `WaveletPool` | L5652 | 小波池化下采样，使用Haar小波变换进行下采样，保留低频(LL)子带 |
| `WaveletUnPool` | L5672 | 小波反池化上采样，将低频子带上采样回原始分辨率 |

---

## 47. CSP_PTB 部分Transformer块

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `MHSA_CGLU` | L5696 | 多头自注意力+卷积GLU |
| `PartiallyTransformerBlock` | L5714 | 部分Transformer块，对部分通道应用MHSA_CGLU，其余通道保持不变 |
| `CSP_PTB` | L5734 | CSP + PartiallyTransformerBlock |

---

## 48. GLSA 全局-局部空间聚合

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `ContextBlock` | L5763 | 上下文块，使用注意力池化或平均池化提取全局上下文，通过通道加/乘进行特征细化 |
| `GLSAChannelAttention` | L5871 | GLSA通道注意力，AvgPool+MaxPool双路径SE |
| `GLSASpatialAttention` | L5890 | GLSA空间注意力，Avg+Max → 7x7 Conv |
| `GLSAConvBranch` | L5907 | GLSA卷积分支，多级深度卷积+通道/空间注意力 |
| `GLSA` | L5940 | 全局-局部空间聚合模块，将通道分为两半分别用局部卷积分支和全局上下文块处理 |

---

## 49. SPDConv 空间到深度卷积

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `SPDConv` | L5972 | 空间到深度卷积(Space-to-Depth Conv)，将特征图按步长2切片为4份拼接(通道x4)后接3x3卷积，保留下采样过程中的细粒度信息 |

---

## 50. OmniKernel / CSPOmniKernel

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `FGM` | L5988 | 傅里叶门控调制(Fourier Gated Modulation)，在频域进行门控调制，通过可学习α/β参数混合频域和空域信息 |
| `OmniKernel` | L6014 | 全核网络块(AAAI-24)，结合多方向深度卷积(1x31, 31x1, 31x31, 1x1) + 频域通道注意力(FCA) + 空间通道注意力(SCA) + FGM，实现全方向全尺度特征提取 |
| `CSPOmniKernel` | L6062 | CSP结构的OmniKernel，将通道分为OmniKernel分支和恒等分支，兼顾效率和性能 |

---

## 51. WTConv 小波卷积

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `BasicBlock_WTConv` | L6078 | BasicBlock + WTConv2d(小波卷积，ECCV-24) |
| `BottleNeck_WTConv` | L6084 | BottleNeck + WTConv2d |

---

## 52. RCE / RCM / PCE 矩形自校准模块

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `PyramidPoolAgg_PCE` | L6094 | 金字塔池化聚合(简化版)，将多尺度特征池化到统一尺寸后拼接 |
| `ConvMlp` | L6105 | 卷积MLP，使用1x1 Conv实现MLP保持空间维度 |
| `RCA` | L6130 | 矩形通道注意力(Rectangular Channel Attention)，使用1D strip卷积沿H/W方向生成注意力 |
| `RCM` | L6162 | 矩形自校准模块(Rectangular Calibration Module)，结合RCA + ConvMlp + 深度卷积的自校准块 |
| `multiRCM` | L6201 | 多层RCM串联 |
| `PyramidContextExtraction` | L6209 | 金字塔上下文提取，使用PyramidPoolAgg_PCE + multiRCM |
| `FuseBlockMulti` | L6222 | 多尺度融合块，使用h_sigmoid门控进行高低特征融合 |
| `DynamicInterpolationFusion` | L6242 | 动态插值融合，使用1x1 Conv + 双线性插值进行跨尺度特征融合 |

---

## 53. SMFANet (FMB / SMFA / PCFN)

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `DMlp` | L6255 | 双分支MLP，使用深度3x3+1x1实现 |
| `PCFN` | L6272 | 部分卷积前馈网络(Partial Conv Feed-Forward Network)，对部分通道应用3x3卷积 |
| `SMFA` | L6298 | 空间多尺度特征注意力(Spatial Multi-scale Feature Attention)，结合局部深度卷积+全局方差+DMlp |
| `FMB` | L6324 | 特征混合块(Feature Mixing Block)，SMFA + PCFN，ECCV-24 SMFANet核心模块 |
| `C2f_FMB` | L6336 | C2f + FMB |

---

## 54. gConv / LDConv

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `gConvBlock` | L6346 | 门控卷积块，使用Wv(深度卷积) * W3g(门控)实现自适应特征选择 |
| `gConvC3` | L6383 | RepC3 + gConvBlock |
| `C2f_gConv` | L6389 | C2f + gConvBlock |
| `LDConv` | L6398 | 线性动态卷积(Linear Dynamic Convolution)，通过学习采样偏移实现任意形状卷积核，与AKConv类似但使用不同的重采样策略 |

---

## 55. AdditiveBlock / MSCB / MutilScale系列

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `Mlp_CASVIT` | L6537 | CASVIT的MLP模块 |
| `SpatialOperation` | L6555 | 空间操作模块 |
| `ChannelOperation` | L6569 | 通道操作模块 |
| `LocalIntegration` | L6581 | 局部积分模块 |
| `AdditiveTokenMixer` | L6598 | 加性Token混合器，使用加性注意力替代乘性注意力 |
| `AdditiveBlock` | L6627 | 加性块，使用AdditiveTokenMixer + MetaFormerBlock |
| `AdditiveBlock_CGLU` | L6649 | AdditiveBlock + CGLU变体 |
| `C2f_AdditiveBlock` | L6654 | C2f + AdditiveBlock |
| `C2f_AdditiveBlock_CGLU` | L6659 | C2f + AdditiveBlock_CGLU |
| `EUCB` | L6669 | 扩展上采样卷积块(Expanded Up-Sampling Conv Block) |
| `MSDC` | L6698 | 多尺度深度卷积(Multi-Scale Depthwise Conv) |
| `MSCB` | L6724 | 多尺度卷积块(Multi-Scale Conv Block)，使用多个不同核大小的深度卷积 |
| `CSP_MSCB` | L6788 | C2f + MSCB |
| `MutilScal` | L6798 | 多尺度特征提取模块 |
| `Mutilscal_MHSA` | L6836 | 多尺度多头自注意力模块 |

---

## 56. MogaBlock / SHSA / SMAFormer

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `MSMHSA_CGLU` | L6882 | 多尺度多头自注意力+CGLU |
| `C2f_MSMHSA_CGLU` | L6900 | C2f + MSMHSA_CGLU |
| `PMSFA` | L6909 | 位置感知多尺度特征注意力 |
| `CSP_PMSFA` | L6929 | C2f + PMSFA |
| `ElementScale` | L6939 | 元素缩放模块，可学习的逐元素缩放 |
| `ChannelAggregationFFN` | L6953 | 通道聚合前馈网络，MogaBlock的核心FFN |
| `MultiOrderDWConv` | L7023 | 多阶深度卷积，使用多阶泰勒展开的深度卷积 |
| `MultiOrderGatedAggregation` | L7093 | 多阶门控聚合模块 |
| `MogaBlock` | L7165 | MogaBlock(多阶门控聚合块)，结合MultiOrderDWConv + ChannelAggregationFFN |
| `C2f_MogaBlock` | L7241 | C2f + MogaBlock |
| `Conv2d_BN` | L7250 | Conv2d + BatchNorm2d序列 |
| `Residual` | L7274 | 残差连接模块 |
| `SHSA_GroupNorm` | L7282 | SHSA的GroupNorm变体 |
| `SHSABlock_FFN` | L7290 | SHSA块的FFN |
| `SHSA` | L7301 | 自混合自注意力(Self-Hybrid Self-Attention) |
| `SHSA_EPGO` | L7332 | SHSA + EPGO(扩展位置生成优化) |
| `SHSABlock` | L7377 | SHSA块 |
| `SHSABlock_EPGO` | L7387 | SHSA + EPGO块 |
| `C2f_SHSA` | L7397 | C2f + SHSABlock |
| `C2f_SHSA_EPGO` | L7402 | C2f + SHSABlock_EPGO |
| `SHSABlock_CGLU` | L7407 | SHSA + CGLU块 |
| `SHSABlock_EPGO_CGLU` | L7417 | SHSA + EPGO + CGLU块 |
| `C2f_SHSA_CGLU` | L7427 | C2f + SHSABlock_CGLU |
| `C2f_SHSA_EPGO_CGLU` | L7432 | C2f + SHSABlock_EPGO_CGLU |
| `Modulator` | L7441 | 调制器，SMAFormer的核心组件 |
| `SMA` | L7554 | 自混合注意力(Self-Mixed Attention) |
| `E_MLP` | L7576 | 高效MLP |
| `SMAFormerBlock` | L7608 | SMAFormer块，SMA + E_MLP |
| `SMAFormerBlock_CGLU` | L7627 | SMAFormer + CGLU块 |
| `C2f_SMAFB` | L7647 | C2f + SMAFormerBlock |
| `C2f_SMAFB_CGLU` | L7652 | C2f + SMAFormerBlock_CGLU |

---

## 57. DynamicAlignFusion / EdgeEnhancer

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `DynamicAlignFusion` | L7661 | 动态对齐融合，使用可学习偏移进行特征对齐后融合 |
| `EdgeEnhancer` | L7702 | 边缘增强器，增强特征图中的边缘信息 |
| `MutilScaleEdgeInformationEnhance` | L7714 | 多尺度边缘信息增强 |
| `MutilScaleEdgeInformationSelect` | L7740 | 多尺度边缘信息选择 |
| `CSP_MutilScaleEdgeInformationEnhance` | L7767 | C2f + 多尺度边缘信息增强 |
| `CSP_MutilScaleEdgeInformationSelect` | L7772 | C2f + 多尺度边缘信息选择 |

---

## 58. Fourier系列 (FFCM / SFHF / FreqSpatial)

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `FourierUnit` | L7781 | 傅里叶单元，在频域进行特征变换 |
| `Freq_Fusion` | L7818 | 频率融合模块 |
| `Fused_Fourier_Conv_Mixer` | L7861 | 融合傅里叶卷积混合器，结合频域和空域特征 |
| `C2f_FFCM` | L7913 | C2f + Fused_Fourier_Conv_Mixer |
| `SFHF_FFN` | L7922 | 空间-频率混合前馈网络 |
| `TokenMixer_For_Local` | L7968 | 局部Token混合器 |
| `SFHF_FourierUnit` | L7989 | 空间-频率混合傅里叶单元 |
| `TokenMixer_For_Gloal` | L8029 | 全局Token混合器 |
| `SFHF_Mixer` | L8055 | 空间-频率混合器 |
| `SFHF_Block` | L8100 | 空间-频率混合块 |
| `C2f_SFHF` | L8130 | C2f + SFHF_Block |
| `ScharrConv` | L8140 | Scharr边缘检测卷积，使用Scharr算子检测边缘 |
| `FreqSpatial` | L8183 | 频率-空间特征模块 |
| `CSP_FreqSpatial` | L8228 | C2f + FreqSpatial |

---

## 59. HDRAB / RAB / LFE 边缘特征模块

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `DeepPoolLayer` | L8237 | 深度池化层，多级池化提取特征 |
| `dynamic_filter` | L8269 | 动态滤波器，学习自适应滤波核 |
| `cubic_attention` | L8316 | 立方注意力，沿三个方向进行注意力计算 |
| `spatial_strip_att` | L8331 | 空间条形注意力，沿条形方向计算注意力 |
| `MultiShapeKernel` | L8369 | 多形状核模块，使用不同形状的卷积核 |
| `C2f_MSM` | L8383 | C2f + MultiShapeKernel |
| `CAB` | L8392 | 通道注意力块 |
| `HDRAB` | L8408 | 高动态范围注意力块(HDR Attention Block) |
| `C2f_HDRAB` | L8468 | C2f + HDRAB |
| `ChannelPool` | L8473 | 通道池化，沿通道维度进行Avg+Max池化 |
| `SAB` | L8480 | 空间注意力块 |
| `RAB` | L8493 | 残差注意力块(Residual Attention Block) |
| `C2f_RAB` | L8524 | C2f + RAB |
| `MeanShift` | L8533 | 均值偏移卷积 |
| `ShiftConv2d0` | L8544 | 移位卷积v0，通过空间移位实现零参数卷积 |
| `ShiftConv2d1` | L8568 | 移位卷积v1 |
| `ShiftConv2d` | L8591 | 移位卷积，组合多个移位操作 |
| `LFE` | L8608 | 局部特征提取(Local Feature Extraction) |
| `LFEC3` | L8632 | RepC3 + LFE |
| `SobelConv` | L8642 | Sobel边缘检测卷积 |
| `MutilScaleEdgeInfoGenetator` | L8662 | 多尺度边缘信息生成器 |
| `ConvEdgeFusion` | L8678 | 卷积边缘融合 |

---

## 60. HyperComputeModule / MANet / HFERB

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `FrequencyProjection` | L8715 | 频率投影，将特征投影到不同频率子带 |
| `ChannelProjection` | L8743 | 通道投影 |
| `SpatialProjection` | L8776 | 空间投影 |
| `DynamicPosBias` | L8798 | 动态位置偏置，可学习的位置编码 |
| `Spatial_Attention` | L8838 | 空间注意力，多尺度空间注意力机制 |
| `Spatial_Frequency_Attention` | L8954 | 空间-频率注意力，在空域和频域同时进行注意力计算 |
| `Channel_Transposed_Attention` | L9146 | 通道转置注意力，沿通道维度进行转置注意力计算 |
| `FrequencyGate` | L9229 | 频率门控，在频域进行门控选择 |
| `DFFN` | L9252 | 动态前馈网络(Dynamic FFN) |
| `FCA` | L9285 | 频率通道注意力(Frequency Channel Attention) |
| `C2f_FCA` | L9320 | C2f + FCA |
| `C2f_CAMixer` | L9329 | C2f + CAMixer(通道注意力混合器) |
| `MANet` | L9338 | 多聚合网络(Multi-Aggregation Network) |
| `MANet_FasterBlock` | L9361 | MANet + FasterBlock |
| `MANet_FasterCGLU` | L9366 | MANet + FasterCGLU |
| `MANet_Star` | L9371 | MANet + Star_Block |
| `MessageAgg` | L9376 | 消息聚合模块 |
| `HyPConv` | L9396 | 超参数卷积(Hyper-Parameter Conv) |
| `HyperComputeModule` | L9412 | 超计算模块，使用HyPConv进行超参数化计算 |
| `GlobalExtraction` | L9441 | 全局特征提取 |
| `ContextExtraction` | L9468 | 上下文特征提取 |
| `MultiscaleFusion` | L9509 | 多尺度融合 |
| `MultiScaleGatedAttn` | L9524 | 多尺度门控注意力 |
| `HFERB` | L9590 | 高频增强残差块(High-Frequency Enhancement Residual Block) |
| `C2f_HFERB` | L9619 | C2f + HFERB |
| `C2f_DTAB` | L9628 | C2f + DTAB(动态Token注意力块) |

---

## 61. JDPM / ETB / FDT / WFU

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `JDPM` | L9637 | 联合域感知模块(Joint Domain Perception Module)，在空域和频域同时进行特征感知 |
| `C2f_JDPM` | L9710 | C2f + JDPM |
| `FeedForward` | L9715 | 标准前馈网络 |
| `Attention_F` | L9760 | 频域注意力 |
| `Attention_S` | L9793 | 空域注意力 |
| `ETB` | L9838 | 高效Transformer块(Efficient Transformer Block) |
| `C2f_ETB` | L9853 | C2f + ETB |
| `GSA` | L9862 | 全局自注意力(Global Self-Attention) |
| `RSA` | L9904 | 残差自注意力(Residual Self-Attention) |
| `FDT` | L9956 | 频域Transformer(Frequency Domain Transformer) |
| `C2f_FDT` | L10006 | C2f + FDT |
| `HaarWavelet` | L10011 | Haar小波变换模块 |
| `WFU` | L10044 | 小波特征上采样(Wavelet Feature Upsampling) |

---

## 62. PSConv / APBottleneck

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `PSConv` | L10082 | 逐点分组卷积(Pointwise-Group Convolution) |
| `APBottleneck` | L10102 | 非对称池化瓶颈块(Asymmetric Pooling Bottleneck) |
| `C2f_AP` | L10123 | C2f + APBottleneck |
| `HaarWaveletConv` | L10132 | Haar小波卷积，将小波变换与卷积结合 |
| `ContrastDrivenFeatureAggregation` | L10169 | 对比驱动特征聚合，通过对比学习进行特征聚合 |

---

## 63. ELGCA / Strip系列

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `ELGCA_MLP` | L10265 | 扩展局部全局通道注意力MLP |
| `ELGCA` | L10284 | 扩展局部全局通道注意力(Extended Local-Global Channel Attention) |
| `ELGCA_EncoderBlock` | L10330 | ELGCA编码器块 |
| `C2f_ELGCA` | L10355 | C2f + ELGCA_EncoderBlock |
| `ELGCA_CGLU` | L10360 | ELGCA + CGLU变体 |
| `C2f_ELGCA_CGLU` | L10385 | C2f + ELGCA_CGLU |
| `StripMlp` | L10394 | 条形MLP，沿条形方向进行MLP计算 |
| `Strip_Block` | L10414 | 条形块 |
| `Strip_Attention` | L10431 | 条形注意力，沿条形方向计算注意力 |
| `StripBlock` | L10448 | 条形块，结合条形注意力和FFN |
| `C2f_Strip` | L10468 | C2f + StripBlock |
| `StripCGLU` | L10473 | 条形CGLU |
| `C2f_StripCGLU` | L10492 | C2f + StripCGLU |

---

## 64. MultiScalePCA / FSA

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `MultiScalePCA` | L10501 | 多尺度位置通道注意力(Multi-Scale Position Channel Attention) |
| `MultiScalePCA_Down` | L10550 | 多尺度位置通道注意力下采样版本 |
| `Adaptive_global_filter` | L10598 | 自适应全局滤波器 |
| `SpatialAttention` | L10627 | 空间注意力模块(标准CBAM空间注意力) |
| `FSA` | L10642 | 频率选择注意力(Frequency Selection Attention) |

---

## 65. KAT / KAN Transformer

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `KAN` | L10663 | Kolmogorov-Arnold网络实现 |
| `KatAttention` | L10702 | KAT注意力(KAN Attention)，使用KAN替代QKV线性投影 |
| `LayerScale` | L10752 | 层缩放，可学习的逐层缩放因子 |
| `Kat` | L10766 | KAT块，使用KatAttention + FFN |
| `C2f_KAT` | L10815 | C2f + Kat |
| `Faster_Block_KAN` | L10820 | Faster_Block + KAN卷积 |
| `C2f_Faster_KAN` | L10870 | C2f + Faster_Block_KAN |

---

## 66. DynamicInception / GlobalFilter / DynamicFilter

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `DynamicInceptionDWConv2d` | L10879 | 动态Inception深度卷积，多分支深度卷积+动态路由 |
| `DynamicInceptionMixer` | L10905 | 动态Inception混合器 |
| `DynamicIncMixerBlock` | L10923 | 动态Inception混合器块 |
| `C2f_DCMB` | L10942 | C2f + DynamicIncMixerBlock |
| `DynamicCIncMixerBlock_KAN` | L10947 | 动态Inception混合器块 + KAN |
| `C2f_DCMB_KAN` | L10967 | C2f + DynamicCIncMixerBlock_KAN |
| `GlobalFilter` | L10976 | 全局滤波器，在频域进行全局滤波 |
| `GlobalFilterBlock` | L10989 | 全局滤波器块 |
| `C2f_GlobalFilter` | L11004 | C2f + GlobalFilterBlock |
| `StarReLU` | L11024 | 星形ReLU激活: x * ReLU(x) |
| `DynamicFilterMlp` | L11043 | 动态滤波器MLP |
| `DynamicFilter` | L11070 | 动态滤波器，在频域进行动态滤波 |
| `C2f_DynamicFilter` | L11120 | C2f + DynamicFilter |

---

## 67. HAFB / MambaOut / EfficientVIM

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `HAFB` | L11131 | 混合注意力融合块(Hybrid Attention Fusion Block) |
| `C2f_SAVSS` | L11166 | C2f + SAVSSBlock(自注意力视觉状态空间块) |
| `C2f_MambaOut` | L11175 | C2f + GatedCNNBlock_BCHW(MambaOut核心块) |
| `GatedUniRepLKBlock_BCHW` | L11180 | 门控UniRepLK块，结合UniRepLK大核卷积+门控机制 |
| `C2f_MambaOut_UniRepLK` | L11216 | C2f + GatedUniRepLKBlock_BCHW |
| `C2f_EfficientVIM` | L11225 | C2f + EfficientViMBlock(高效视觉Mamba块) |
| `C2f_EfficientVIM_CGLU` | L11230 | C2f + EfficientViMBlock_CGLU |

---

## 68. Mamba系列

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `Shift_channel_mix` | L11248 | 移位通道混合 |
| `EUCB_SC` | L11269 | 扩展上采样卷积块(移位卷积版) |
| `MSCB_SC` | L11299 | 多尺度卷积块(移位卷积版) |
| `CSP_MSCB_SC` | L11366 | C2f + MSCB_SC |
| `C2f_GroupMamba` | L11255 | C2f + GroupMambaLayer(分组Mamba层) |
| `C2f_GroupMambaBlock` | L11260 | C2f + Block_mamba(Mamba块) |
| `C2f_MambaVision` | L11269 | C2f + MambaVisionBlock(MambaVision块) |
| `C2f_GLVSS` | L12864 | C2f + GL_VSS(全局局部视觉状态空间) |
| `C2f_VSSD` | L13248 | C2f + VMAMBA2Block(VSSD块) |
| `C2f_TVIM` | L13257 | C2f + TViMBlock(TinyVIM块) |

---

## 69. CrossAttentionBlock / IEL / RCB / FAT / LEGM

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `CrossAttentionBlock` | L11377 | 交叉注意力块，在两个特征间进行交叉注意力计算 |
| `IEL` | L11424 | 内部增强层(Internal Enhancement Layer) |
| `C2f_IEL` | L11448 | C2f + IEL |
| `IELC3` | L11453 | RepC3 + IEL |
| `C2f_RCB` | L11463 | C2f + RCB(残差卷积块) |
| `C2f_FAT` | L11472 | C2f + FAT(频率自适应Token) |
| `C2f_LEGM` | L11481 | C2f + LEGM(局部边缘全局混合) |
| `C2f_MobileMamba` | L11490 | C2f + MobileMambaBlock(移动Mamba块) |

---

## 70. LFEA / LFEM / LoG系列

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `Conv_Extra` | L11499 | 额外卷积模块 |
| `Scharr4` | L11509 | Scharr边缘检测算子 |
| `Gaussian` | L11536 | 高8Gaussian高斯模糊卷积 |
| `LFEA` | L11566 | 局部特征增强注意力(Local Feature Enhancement Attention) |
| `LFE_Module` | L11588 | 局部特征提取模块 |
| `C2f_LFEM` | L11623 | C2f + LFE_Module |
| `DRFD_LoG` | L116B28 | 深层特征下采样 + LoG(Laplacian of Gaussian) |
| `LoGFilter` | L11656 | LoG滤波器(Laplacian of Gaussian) |
| `LoGStem` | L11685 | LoG特征提取茎干，使用LoG滤波器进行初始特征提取 |
| `C2f_SBSM` | L11716 | C2f + SBSM(空间带状状态混合) |
| `C2f_LSBlock` | L11725 | C2f + LSBlock(大核分离块) |

---

## 71. FDConv / SFSConv / DSAN / DSA / RMB / SNI

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `GatedLSConvBlock_BCHW` | L11730 | 门控大核分离卷积块 |
| `C2f_MambaOut_LSConv` | L11766 | C2f + GatedLSConvBlock_BCHW |
| `C2f_TransMamba` | L11775 | C2f + TransMambaBlock(Transformer-Mamba混合块) |
| `C2f_EVS` | L11784 | C2f + EVS(高效视觉状态空间) |
| `C2f_EBlock` | L11793 | C2f + EBlock(增强块) |
| `C2f_DBlock` | L11798 | C2f + DBlock(解码块) |
| `Bottleneck_FDConv` | L11807 | Bottleneck + FDConv(频率域卷积) |
| `C2f_FDConv` | L11814 | C2f + Bottleneck_FDConv |
| `GatedFDConvBlock_BCHW` | L11819 | 门控频率域卷积块 |
| `C2f_MambaOut_FDConv` | L11855 | C2f + GatedFDConvBlock_BCHW |
| `Partial_FDConv` | L11860 | 部分频率域卷积 |
| `FasterFDConv` | L11887 | 快速频率域卷积 |
| `Bottleneck_PFDConv` | L11943 | Bottleneck + Partial_FDConv |
| `C2f_PFDConv` | L11950 | C2f + Bottleneck_PFDConv |
| `C2f_FasterFDConv` | L11955 | C2f + FasterFDConv |
| `FDConvC3` | L11960 | RepC3 + FDConv |
| `C2f_DSAN` | L11978 | C2f + DSAN(深度可分离注意力网络) |
| `C2f_DSAN_EDFFN` | L11983 | C2f + DSAN_EDFFN(扩展深度可分离注意力网络) |
| `GatedDSABlock_BCHW` | L11988 | 门控深度可分离注意力块 |
| `C2f_MambaOut_DSA` | L12024 | C2f + GatedDSABlock_BCHW |
| `Bottleneck_DSA` | L12029 | Bottleneck + DSA(深度可分离注意力) |
| `C2f_DSA` | L12036 | C2f + Bottleneck_DSA |
| `C2f_RMB` | L12045 | C2f + RMB(残差移动块) |
| `SNI` | L12054 | 语义噪声注入(Semantic Noise Injection) |
| `GSConvE` | L12069 | 扩展GSConv |
| `Bottleneck_SFSConv` | L12098 | Bottleneck + SFS_Conv(空间频率分离卷积) |
| `C2f_SFSConv` | L12105 | C2f + Bottleneck_SFSConv |
| `GatedSFSCBlock_BCHW` | L12110 | 门控空间频率分离卷积块 |
| `C2f_MambaOut_SFSC` | L12146 | C2f + GatedSFSCBlock_BCHW |
| `Partial_SFSConv` | L12151 | 部分空间频率分离卷积 |
| `FasterSFSConv` | L12178 | 快速空间频率分离卷积 |
| `Bottleneck_PSFSConv` | L12234 | Bottleneck + Partial_SFSConv |
| `C2f_PSFSConv` | L12241 | C2f + Bottleneck_PSFSConv |
| `C2f_FasterSFSConv` | L12246 | C2f + FasterSFSConv |

---

## 72. FCM / Pzconv / PST (PointSetTransformer)

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `Channel` | L12278 | 通道特征提取 |
| `Spatial` | L12295 | 空间特征提取 |
| `FCM_3` | L12309 | 全通道混合器v3 |
| `FCM_2` | L12332 | 全通道混合器v2 |
| `FCM_1` | L12356 | 全通道混合器v1 |
| `FCM` | L12381 | 全通道混合器(默认版本) |
| `Pzconv` | L12407 | Pz卷积，并行深度卷积+1x1卷积 |
| `PSAttn` | L12438 | 点集注意力(Point-Set Attention)，在点集上计算注意力 |
| `PSAttnBlock` | L12589 | 点集注意力块 |
| `PST` | L12667 | 点集Transformer(Point-Set Transformer) |
| `DeepSparse` | L12776 | 深度稀疏模块 |

---

## 73. FourierConv / wConv / GLVSS / ESC

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `FourierConv` | L12796 | 傅里叶卷积，在频域进行卷积操作 |
| `Bottleneck_FourierConv` | L12808 | Bottleneck + FourierConv |
| `C2f_FourierConv` | L12815 | C2f + Bottleneck_FourierConv |
| `wConv2d` | L12824 | 小波卷积2d，使用小波变换进行卷积 |
| `Bottleneck_wConv` | L12848 | Bottleneck + wConv2d |
| `C2f_wConv` | L12855 | C2f + Bottleneck_wConv |
| `C2f_ESC` | L12873 | C2f + ESCBlock(高效空间卷积块) |
| `Bottleneck_ConvAttn` | L12878 | Bottleneck + ConvAttn(卷积注意力) |
| `C2f_ConvAttn` | L12884 | C2f + Bottleneck_ConvAttn |

---

## 74. MBRConv / ConvAttn / VSSD / TVIM

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `MBRConv3` | L12893 | 多分支重参数化卷积v3，使用3x3核 |
| `MBRConv5` | L13043 | 多分支重参数化卷积v5，使用5x5核 |
| `Bottleneck_MBRConv3` | L13208 | Bottleneck + MBRConv3 |
| `C2f_MBRConv3` | L13215 | C2f + Bottleneck_MBRConv3 |
| `Bottleneck_MBRConv5` | L13220 | Bottleneck + MBRConv5 |
| `C2f_MBRConv5` | L13227 | C2f + Bottleneck_MBRConv5 |
| `MBRConv3C3` | L13232 | RepC3 + MBRConv3 |
| `MBRConv5C3` | L13238 | RepC3 + MBRConv5 |
| `AdaptiveCombiner` | L13266 | 自适应组合器，学习多特征的组合权重 |
| `DPCF` | L13279 | 深度部分卷积融合(Depthwise Partial Conv Fusion) |

---

## 75. DPCF / CSI / UniConvBlock / LGLB / ConverseNet / GCConv / CFBlock / FMABlock / LWGA

| 模块名 | 行号 | 作用 |
|--------|------|------|
| `C2f_CSI` | L13306 | C2f + CSI(通道空间交互) |
| `C2f_UniConvBlock` | L13315 | C2f + UniConvBlock(通用卷积块) |
| `C2f_LGLB` | L13324 | C2f + LGLBlock(局部全局局部块) |
| `C2f_ConverseB` | L13333 | C2f + ConverseBlock(逆卷积块) |
| `Bottleneck_Converse` | L13338 | Bottleneck + Converse2D |
| `C2f_Converse2D` | L13345 | C2f + Bottleneck_Converse |
| `Converse2DC3` | L13350 | RepC3 + Converse2D |
| `Bottleneck_GCConv` | L13360 | Bottleneck + GCConv(全局上下文卷积) |
| `C2f_GCConv` | L13367 | C2f + Bottleneck_GCConv |
| `GCConvC3` | L13372 | RepC3 + GCConv |
| `C2f_CFBlock` | L13382 | C2f + CFBlock(上下文频率块) |
| `C2f_FMABlock` | L13391 | C2f + FMABlock(频率调制注意力块) |
| `C2f_LWGA` | L13400 | C2f + LWGA_Block(局部窗口全局注意力块) |

---

## 模块分类统计

| 类别 | 数量 | 说明 |
|------|------|------|
| HGBlock系列 | 3 | PPHGNetV2的HG块变体 |
| DWR系列 | 4 | 空洞残差模块 |
| OrthoNets系列 | 7 | 正交注意力网络 |
| DCNv2系列 | 6 | 可变形卷积v2 |
| DCNv2_Dynamic系列 | 7 | 动态偏移可变形卷积 |
| DCNv3系列 | 6 | 可变形卷积v3 |
| DCNv4系列 | 6 | 可变形卷积v4 |
| iRMB系列 | 20 | 倒残差移动块及变体 |
| Attention系列 | 6 | 注意力增强ResNet块 |
| DySnakeConv系列 | 5 | 动态蛇形卷积 |
| FasterBlock系列 | 22 | 快速模块及变体 |
| AKConv系列 | 5 | 任意核卷积 |
| RFAConv系列 | 14 | 感受野注意力卷积 |
| Conv3XC/SPAB系列 | 11 | Swift参数免费注意力 |
| UniRepLKNet系列 | 4 | 通用大核网络 |
| DRB/DBB系列 | 18 | 重参数化系列 |
| DualConv系列 | 5 | 双路卷积 |
| ASSF系列 | 7 | 注意力尺度序列融合 |
| SlimNeck系列 | 5 | GSConv轻量颈部 |
| AggregatedAtt系列 | 5 | TransNeXt聚合注意力 |
| GOLD-YOLO系列 | 12 | GOLD-YOLO融合模块 |
| HS-FPN系列 | 7 | 高效特征金字塔注意力 |
| 上下采样系列 | 4 | DySample/CARAFE/HWD |
| SSM系列 | 12 | 状态空间模型(VSS/Mamba) |
| YOLOv9系列 | 17 | 重参数化CSP-ELAN |
| Fusion系列 | 1 | BiFPN多特征融合 |
| ContextGuided系列 | 7 | 上下文引导块 |
| PAC系列 | 4 | 并行空洞卷积 |
| DGCST系列 | 3 | 动态组卷积混洗 |
| RTM系列 | 3 | Retention机制 |
| FADC系列 | 7 | 频率自适应空洞卷积 |
| 下采样系列 | 5 | 深层特征下采样 |
| CFC/CAFM系列 | 10 | 上下文空间特征校准 |
| GLU系列 | 6 | 卷积门控线性单元 |
| 语义融合系列 | 4 | 多模态语义融合 |
| Star系列 | 8 | StarNet星形块 |
| KAN系列 | 6 | Kolmogorov-Arnold网络 |
| vHeat系列 | 8 | 热传导算子网络 |
| PSA系列 | 3 | 部分自注意力 |
| 小波系列 | 2 | 小波池化/反池化 |
| GLSA系列 | 5 | 全局-局部空间聚合 |
| SPDConv | 1 | 空间到深度卷积 |
| OmniKernel系列 | 3 | 全核网络(AAAI-24) |
| WTConv系列 | 2 | 小波卷积(ECCV-24) |
| PCE系列 | 7 | 矩形自校准模块(ECCV-24) |
| SMFANet系列 | 5 | 空间多尺度特征注意力 |
| gConv/LDConv系列 | 3 | 门控卷积/线性动态卷积 |
| MogaBlock系列 | 2 | 多阶门控聚合 |
| SHSA系列 | 10 | 自混合自注意力 |
| SMAFormer系列 | 6 | 自混合注意力Former |
| 边缘系列 | 6 | 边缘增强/选择 |
| Fourier系列 | 16 | 傅里叶频域模块 |
| HDRAB/RAB系列 | 5 | 高动态范围/残差注意力 |
| LFE系列 | 8 | 局部特征提取 |
| Hyper/MANet系列 | 22 | 超计算/多聚合网络 |
| JDPM/ETB/FDT系列 | 11 | 联合域感知/高效Transformer |
| PSConv系列 | 5 | 逐点分组卷积 |
| ELGCA/Strip系列 | 12 | 扩展局部全局通道注意力 |
| MultiScalePCA系列 | 5 | 多尺度位置通道注意力 |
| KAT系列 | 7 | KAN Transformer |
| DynamicInception系列 | 7 | 动态Inception混合器 |
| GlobalFilter系列 | 4 | 全局/动态滤波器 |
| MambaOut系列 | 7 | MambaOut核心块 |
| Mamba系列 | 10 | 各类Mamba变体 |
| 交叉注意力系列 | 8 | 交叉注意力/内部增强 |
| LFEA/LoG系列 | 11 | 局部特征增强/LoG滤波 |
| FDConv/SFSConv系列 | 25 | 频率域卷积系列 |
| FCM/PST系列 | 12 | 全通道混合器/点集Transformer |
| FourierConv/wConv系列 | 8 | 傅里叶卷积/小波卷积 |
| MBRConv系列 | 8 | 多分支重参数化卷积 |
| 其他尾系列 | 13 | CSI/UniConv/Converse/GCConv等 |

---

## 关键模块详解

### SPDConv (空间到深度卷积)

SPDConv是一种保信息的下采样方法。传统下采样(如stride=2的卷积或池化)会丢失信息，SPDConv通过将特征图按步长2在空间维度上切片为4个子图，然后在通道维度拼接，使得空间信息被完整保留在通道维度中，再通过3x3卷积进行通道压缩。

```
输入: [B, C, H, W]
切片: x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]
拼接: [B, 4C, H/2, W/2]
卷积: Conv(4C → ouc, k=3) → [B, ouc, H/2, W/2]
```

### OmniKernel (全核网络块)

OmniKernel来自AAAI-2024论文，核心思想是同时利用多个方向和尺度的卷积核进行特征提取：
1. **多方向深度卷积**: 1x31(水平), 31x1(垂直), 31x31(方形), 1x1(逐点)
2. **频域通道注意力(FCA)**: 在频域进行通道注意力计算
3. **空间通道注意力(SCA)**: 全局平均池化后进行通道注意力
4. **傅里叶门控调制(FGM)**: 在频域进行门控调制

### CSPOmniKernel

CSP结构的OmniKernel，将输入通道分为两部分：
- OmniKernel分支(占e比例): 使用OmniKernel处理
- 恒等分支(占1-e比例): 直接传递
最后通过1x1卷积融合两个分支，兼顾特征变换能力和计算效率。

### DilatedReparamBlock (空洞重参数化块)

UniRepLKNet的核心组件，在训练时使用一个大核深度卷积+多个不同膨胀率的小核深度卷积并行，部署时通过数学等价变换将所有分支融合为单个大核深度卷积，实现训练时的多尺度感受野和部署时的高效率。

### iRMB (倒残差移动块)

结合了三种关键机制：
1. **窗口自注意力**: 在局部窗口内进行多头自注意力
2. **深度卷积**: 局部空间特征提取
3. **SE注意力**: 通道注意力增强

支持多种配置：注意力前置/后置、v投影、不同局部卷积类型等。

### ContextGuidedBlock (上下文引导块)

CG Block来自CGNet，核心思想是同时利用局部特征和周围上下文：
1. 1x1卷积降维
2. **局部特征(F_loc)**: 3x3深度卷积(d=1)
3. **周围上下文(F_sur)**: 3x3深度卷积(d=2，膨胀卷积)
4. 拼接局部+上下文 → BN+Act
5. **全局细化(F_glo)**: SE-like通道注意力
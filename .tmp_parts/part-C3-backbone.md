## 七、基线配置、r50 系列复验与整体主干替换

本节覆盖 `ultralytics/cfg/models/rt-detr/` 目录中以小写字母开头的三类配置：**基线配置（14 个）**、**r50 系列模块复验变体（44 个 + r50-m 共 45 个）**、**整体主干替换与颈部补充（29 个）**，合计 87 个 yaml。

通用约定（全目录一致）：
- RT-DETR 三段式架构 = Backbone（PResNet/HGNetV2/替换主干）→ HybridEncoder 颈部（AIFI + CCFF 双向融合）→ RTDETRDecoder。
- ResNet 系主干统一由 `Blocks` 模块组装，参数 `[ch_out, block_type, block_nums, stage_num, act, variant, ...]`；`block_type` 是核心改进点（BasicBlock/BottleNeck 及其 `BottleNeck_XXX`/`BasicBlock_XXX` 变体）。残差分支的 `branch2b`（第二个 3×3 卷积）是最常见的替换位置。
- `RTDETRDecoder` 参数为 `[nc, hd=256, nq=300, ndp=4, nh=8, ndl=6, d_ffn=1024, eval_idx=-1, ...]`（modules/head.py:172）。多数配置显式写出 `[nc, 256, 300, 4, 8, X]`，X 即解码器层数 ndl；r50-m 额外写 `1024, 2`（d_ffn=1024、eval_idx=2）。
- 颈部 CCFF：3 个 input_proj（1×1 Conv，`False` 表示无 BN+激活）→ AIFI 作用于 P5 → 自顶向下 lateral_conv + Concat + RepC3（fpn_blocks）→ 自底向上 downsample_conv（3×3 s2）+ Concat + RepC3（pan_blocks）→ 解码器吃 P3/P4/P5 三层。

### 7.1 基线配置

#### 7.1.1 PResNet 主干系列（r18/r34/r50/r50-m/r101）

主干结构：`ConvNormLayer(32,3,2)` → `ConvNormLayer(32,3,1)` → `ConvNormLayer(64,3,1)` → `nn.MaxPool2d(3,2,1)`（共 4 层 stem，P1→P2）→ 4 个 `Blocks` 段。全部 `scales: l: [1.00, 1.00, 1024]`（r101 为 2048）。

| 配置文件 | 主干（Blocks 段） | 颈部/解码器要点 |
|---|---|---|
| rtdetr-r18.yaml | BasicBlock 2-2-2-2（ResNet-18），relu | RepC3 `[256, 0.5]`（内层缩放 0.5）；AIFI `[1024, 8]`；解码器 `[nc,256,300,4,8,3]`（ndl=3，配合轻量主干） |
| rtdetr-r34.yaml | BasicBlock 3-4-6-3（ResNet-34） | 同 r18 结构，解码器 ndl=4 |
| rtdetr-r50.yaml | BottleNeck 3-4-6-3（ResNet-50，expansion=4） | RepC3 `[256]`（无缩放）；解码器 ndl=6（官方 RT-DETR-R50 规格） |
| rtdetr-r50-m.yaml | BottleNeck 3-4-6-3（同 r50 主干） | RepC3 `[256, 0.5]`（颈部内层减半）；解码器 `[nc,256,300,4,8,6,1024,2]`，即 eval_idx=2——推理只取 6 层解码器的第 3 层输出，是"中型"推理开销变体 |
| rtdetr-r101.yaml | BottleNeck 3-4-23-3（ResNet-101） | max_channels 2048；头部通道升为 384，AIFI `[2048, 8]`，RepC3 `[384]`；解码器 ndl=6 |

#### 7.1.2 HGNetV2 主干系列（l/x 及变体）

| 配置文件 | 主干 | 颈部/解码器要点 |
|---|---|---|
| rtdetr-l.yaml | HGNetV2-L：HGStem[32,48] + HGBlock×4 段（stage1 [48,128,3]、stage2 [96,512,3]、stage3 [192,1024,5]×3 块 light=True、stage4 [384,2048,5]），段间 DWConv 下采样 | 头部 256 通道，AIFI `[1024, 8]`；RepC3 `[256]`；解码器 `[nc]`（全默认：hd=256, nq=300, ndp=4, nh=8, ndl=6, d_ffn=1024） |
| rtdetr-x.yaml | HGNetV2-X：更宽更深（stage1 [64,128]、stage2 [128,512]×2、stage3 [256,1024]×5、stage4 [512,2048]×2） | 头部 384 通道，AIFI `[2048, 8]`，RepC3 `[384]`；`scales: x: [1.00, 1.00, 2048]`；解码器 `[nc]` |
| rtdetr-l-GhostHGNetV2.yaml | 同 rtdetr-l，但 HGBlock → `Ghost_HGBlock`（block.py:122，用 GhostConv 构建堆叠卷积），且 stage3/4 卷积核由 5 改为 1 | 颈部与 rtdetr-l 完全一致；轻量化主干变体（GhostConv 线性廉价运算生成冗余特征图） |
| rtdetr-l-RepHGNetV2.yaml | 同 rtdetr-l，但 HGBlock → `Rep_HGBlock`（block.py:162，堆叠卷积换成 RepConv 重参数化卷积），k 保持 3 | 颈部与 rtdetr-l 一致；重参数化主干变体（训练多分支、推理融合为单路卷积，速度更快） |
| rtdetr-l-attention.yaml | 同 rtdetr-l，仅 stage4 最后一个 HGBlock → `HGBlock_Attention`（block.py:1196，在 sc/ec 挤压-激发前插入 `CoordAtt` 坐标注意力） | 颈部与 rtdetr-l 一致；是"AIFI 之外再给主干末端加注意力"的对照基线 |

#### 7.1.3 P2 / P6 多尺度与 timm 接口

| 配置文件 | 结构要点 |
|---|---|
| rtdetr-p2.yaml | r18 主干（BasicBlock 2-2-2-2）；在标准 P3-P5 之外引入 P2 层（backbone 第 4 层输出，1/4 分辨率）：颈部共 4 个 input_proj + 3 个 fpn_blocks + 3 个 pan_blocks，解码器输入 P2/P3/P4/P5 四层，hd=128；RepC3 均 `[·, 0.5]`，解码器 `[nc,128,300,4,8,3]`。面向小目标 |
| rtdetr-p6-CTrans.yaml | r18 主干再加深一段 P6/64（`Blocks [512, BasicBlock, 2, 5]`）；P3-P6 四层先送入 `ChannelTransformer`（CTrans，跨尺度通道注意力，extra_modules/CTrans.py:335），经 `GetIndexOutput` 按 index 拆回 4 层；随后 4 尺度 CCFF（3 个 fpn_blocks + 3 个 pan_blocks）；解码器吃 P3-P6，`[nc,256,300,4,8,3]` |
| rtdetr-timm.yaml | timm 主干通用接口模板：backbone 仅一行 `- [-1, 1, mobilenetv3_small_100, [False]]`（占位示例，可替换为任意 timm 模型名，经 tasks.py 的 timm 加载器输出 4 个尺度）；颈部 RepC3 `[256, 0.5]`，解码器 ndl=3（轻量版头部） |
| rtdetr-r50-timm.yaml | 同一 timm 模板，但头部按 r50 规格配置：RepC3 `[256]`、解码器 ndl=6。两文件本身只差颈部规格，主干同为 mobilenetv3_small_100 占位 |

### 7.2 r50 系列变体（模块复验）

**规律**：`rtdetr-r50-XXX.yaml` 与 `rtdetr-XXX.yaml` 使用完全相同的改进模块，区别仅在主干——r50 版把改进模块装进 **BottleNeck（ResNet-50，3-4-6-3）**，而无 r50 前缀版本装进 **BasicBlock（ResNet-18 形，2-2-2-2）**。二者互为"同一模块在两种主干深度上的复验"。解码器均 `[nc,256,300,4,8,6]`、RepC3 `[256]`，除非下表备注说明。

#### 7.2.1 全 4 个 stage 替换（BottleNeck_XXX 全程生效）

| 配置文件 | 对应模块配置 | 改进模块与机制 |
|---|---|---|
| rtdetr-r50-AKConv.yaml | 同 rtdetr-AKConv | BottleNeck_AKConv：AKConv 任意核形卷积（不规则采样坐标，卷积核参数量/形状可变）替换 branch2b |
| rtdetr-r50-Conv3XC.yaml | 同 rtdetr-Conv3XC | BottleNeck_Conv3XC：SPAN（Swift Parameter-free Attention Network）的 Conv3XC 重参数化 3×3 卷积；各 stage 附带 64/32/16/8 通道参数 |
| rtdetr-r50-DBB.yaml | 同 rtdetr-DBB | BottleNeck_DBB：Diverse Branch Block 多样分支块（训练多分支，推理合并单卷积） |
| rtdetr-r50-WDBB.yaml | 同 rtdetr-WDBB | BottleNeck_WDBB：加宽版 Diverse Branch Block |
| rtdetr-r50-DeepDBB.yaml | 同 rtdetr-DeepDBB | BottleNeck_DeepDBB：深层 Diverse Branch Block 变体 |
| rtdetr-r50-DEConv.yaml | 同 rtdetr-DEConv | BottleNeck_DEConv：细节增强卷积（普通+中心差分/角度差分卷积并联，可重参数化，DEA-Net） |
| rtdetr-r50-DRB.yaml | 同 rtdetr-DRB | BottleNeck_DRB：Diverse Receptive field Block（多膨胀率大核并行，DRBNet） |
| rtdetr-r50-DualConv.yaml | 同 rtdetr-DualConv | BottleNeck_DualConv：双卷积（分组卷积与 1×1 卷积并行共享输入） |
| rtdetr-r50-Faster.yaml | 同 rtdetr-Faster | BottleNeck_Faster_Block：FasterNet Faster Block（PConv 部分卷积，只对 1/4 通道做 3×3） |
| rtdetr-r50-Faster-Rep.yaml | 同 rtdetr-Faster-Rep | BottleNeck_Faster_Block_Rep：Faster Block + 重参数化 |
| rtdetr-r50-Faster-EMA.yaml | 同 rtdetr-Faster-EMA | BottleNeck_Faster_Block_EMA：Faster Block + EMA 高效多尺度注意力 |
| rtdetr-r50-Faster-Rep-EMA.yaml | 同 rtdetr-Faster-Rep-EMA | BottleNeck_Faster_Block_Rep_EMA：三者叠加（PConv + 重参数化 + EMA） |
| rtdetr-r50-faster-CGLU.yaml | 同 rtdetr-faster-CGLU | BottleNeck_Faster_Block_CGLU：PConv + ConvolutionalGLU 门控 FFN（block.py:4866） |
| rtdetr-r50-PConv.yaml | 同 rtdetr-PConv | BottleNeck_PConv：部分卷积 |
| rtdetr-r50-PConv-Rep.yaml | 同 rtdetr-PConv-Rep | BottleNeck_PConv_Rep：部分卷积 + 重参数化 |
| rtdetr-r50-RFAConv.yaml | 同 rtdetr-RFAConv | BottleNeck_RFAConv：感受野注意力卷积（RFAConv），各 stage 附 64/32/16/8 参数 |
| rtdetr-r50-RFCAConv.yaml | 同 rtdetr-RFCAConv | BottleNeck_RFCAConv：RFAConv + CA 坐标注意力 |
| rtdetr-r50-RFCBAMConv.yaml | 同 rtdetr-RFCBAMConv | BottleNeck_RFCBAMConv：RFAConv + CBAM 注意力 |
| rtdetr-r50-Star.yaml | 同 rtdetr-Star | BottleNeck_Star：StarNet 星型运算块（两分支元素乘实现隐式高维映射） |
| rtdetr-r50-SWC.yaml | 同 rtdetr-SWC | BottleNeck_SWC：Shift-wise Conv 移位式卷积；kernel 9/7/7/7 逐 stage 配置 |
| rtdetr-r50-iRMB.yaml | 同 rtdetr-iRMB | BottleNeck_iRMB：EMO（ICCV2023）倒残差移动块 iRMB（窗口注意力 + 深度卷积 + SE） |
| rtdetr-r50-iRMB-Cascaded.yaml | 同 rtdetr-iRMB-Cascaded | BottleNeck_iRMB_Cascaded：iRMB 注意力部分改为 LocalWindowAttention（级联组窗口注意力） |
| rtdetr-r50-iRMB-DRB.yaml | 同 rtdetr-iRMB-DRB | BottleNeck_iRMB_DRB：iRMB 局部卷积换 DilatedReparamBlock（多膨胀率重参数化大核）；dw_ks=13/11/9/7 |
| rtdetr-r50-iRMB-SWC.yaml | 同 rtdetr-iRMB-SWC | BottleNeck_iRMB_SWC：iRMB 局部卷积换 ReparamLargeKernelConv（大核重参数化，UniRepLKNet 风格）；dw_ks=7/7/5/5 |
| rtdetr-r50-ContextGuided.yaml | 同 rtdetr-ContextGuided | BottleNeck_ContextGuided（CGNet 上下文引导卷积：局部+邻域+全局三分支）；**注意颈部也被改**：pan 路径的 2 个下采样 Conv 换成 `ContextGuidedBlock_Down`（与 rtdetr-ContextGuided 一致） |
| rtdetr-r50-attention.yaml | 同 rtdetr-attention | BottleNeck_Attention（"resnet attention" 系列）：BottleNeck 主体后接 `CoordAtt` 坐标注意力（block.py:1149） |
| rtdetr-r50-Ortho.yaml | 同 rtdetr-Ortho | BottleNeck_Ortho：OrthoNet 正交卷积（约束特征正交性）；各 stage 附 64/32/16/8 参数 |
| rtdetr-r50-KAN.yaml | 同 rtdetr-KAN | BottleNeck_KAN：KAN 卷积（Kolmogorov-Arnold 网络，样条激活）+ KAGNConv2DLayer；**仅 P4/P5 两段**替换（P2/P3 保持 BottleNeck） |
| rtdetr-r50-VSS.yaml | 同 rtdetr-VSS | BottleNeck_VSS：Visual State Space 块（Mamba SS2D 选择性扫描）；**仅 P4/P5 两段**替换 |
| rtdetr-r50-iRMB（其余 3 个组合见上） | — | 见 7.3.3 小节 iRMB 系列详解 |

#### 7.2.2 仅 P5（末段）替换

| 配置文件 | 对应模块配置 | 改进模块与机制 |
|---|---|---|
| rtdetr-r50-DCNV2.yaml | 同 rtdetr-DCNV2 | BottleNeck_DCNv2：可变形卷积 v2（可学习调制标量 + offset）替换 P5 段 |
| rtdetr-r50-DCNV2-Dynamic.yaml | 同 rtdetr-DCNV2-Dynamic | BottleNeck_DCNv2_Dynamic：DCNv2 的动态版（offset 生成网络随输入动态变化） |
| rtdetr-r50-DCNV3.yaml | 同 rtdetr-DCNV3 | BottleNeck_DCNv3：DCNv3（InternImage，组透明归一化 + 更高采样效率） |
| rtdetr-r50-DCNV4.yaml | 同 rtdetr-DCNV4 | BottleNeck_DCNv4：DCNv4（更快显存访存、更高吞吐的 DCNv3 升级） |
| rtdetr-r50-DySnake.yaml | 同 rtdetr-DySnake | BottleNeck_DySnakeConv：动态蛇形卷积（沿目标边界自适应移动的卷积点列，细长/管状目标） |
| rtdetr-r50-fadc.yaml | 同 rtdetr-fadc | BottleNeck_FADC：频率自适应膨胀卷积（ICCV2023，按频域信息自适应膨胀率与采样位置） |
| rtdetr-r50-AggregatedAtt.yaml | 同 rtdetr-AggregatedAtt | BottleNeck_AggregatedAtt：TransNeXt 聚合注意力（像素聚焦注意力 + 空间缩减 sr_ratio）。**注意与 rtdetr-AggregatedAtt 的差异**：r50 版仅 P5 段使用（`[512, BottleNeck_AggregatedAtt, 3, 5, 'relu', 20, 1]`，P2-P4 保持普通 BottleNeck）；rtdetr 版在 P4、P5 两段使用（40,2 / 20,1） |

#### 7.2.3 主干不变、改进在颈部的 r50 版（与对应配置头部逐层一致）

| 配置文件 | 对应模块配置 | 颈部改进与机制 |
|---|---|---|
| rtdetr-r50-DGCST.yaml | 同 rtdetr-DGCST | CCFF 中 4 个 RepC3 全部换成 `DGCST`（Dynamic Group Convolution Shuffle Transformer：1×1 投影后 1/4 通道分组卷积 + 通道 shuffle，再接双 1×1 残差） |
| rtdetr-r50-RetBlockC3.yaml | 同 rtdetr-RetBlockC3 | RepC3 → `RetBlockC3 [256, 1, 'chunk', 8]`（RetNet 多尺度保留机制 retention 的 chunk 模式，block.py:4167） |
| rtdetr-r50-PACAPN.yaml | 同 rtdetr-PACAPN | 上采样/下采样 Conv 换 `AttentionUpsample`/`AttentionDownsample`，fpn/pan 块换 `CSP_PAC`（CSP 并行空洞卷积，PAC-APN 注意力金字塔网络） |
| rtdetr-r50-ASF.yaml | 同 rtdetr-ASF | Concat → `Zoom_cat`（尺度对齐拼接，input_proj 从 2 个变 1 个），末端 `ScalSeq`（注意力尺度序列融合）+ `Add` 增强 P3 输出（ASF-YOLO 的检测头增强） |
| rtdetr-r50-bifpn.yaml | 同 rtdetr-bifpn | 整个 CCFF 换成 BiFPN：`fusion_mode: bifpn`、`node_mode: RepC3`、`head_channel: 256`，含 P2→P3 额外自底向上路径，8 个 `Fusion[bifpn]` 节点 |
| rtdetr-r50-RepNCSPELAN.yaml | 同 rtdetr-RepNCSPELAN | **特例：主干不再是 ResNet**。整个 backbone 换成 YOLOv9 ELAN 风格：`Conv stem → RepNCSPELAN4 → ADown → RepNCSPELAN4 → …`（通道 128→1024），颈部 RepC3 全部换成 `RepNCSPELAN4`（重参数化 CSP-ELAN）。对应 rtdetr-RepNCSPELAN（同结构但通道更小 64→512），r50 版是放大宽度版本 |

#### 7.2.4 仅 stem 替换

| 配置文件 | 对应模块配置 | 机制 |
|---|---|---|
| rtdetr-r50-SRFD.yaml | 同 rtdetr-SRFD | stem（3×ConvNormLayer+MaxPool）→ `SRFD`（block.py:4420）：7×7 卷积强化特征后两级"深度卷积 + Cut 切半 + 池化"三分支融合，1 次前向直接下采样 4 倍到 P2，同时减少下采样信息损失。主干 4 段 BottleNeck 保持不变，层号整体前移 |

### 7.3 整体主干替换

这一组用完整新主干替换 PResNet/HGNetV2。除 mamba 系（自带新颈部）外，其余均为"一行主干 + 标准 HybridEncoder"结构（timm 风格接口，主干输出 P2-P5 四个尺度，颈部取 P3/P4/P5），解码器 `[nc,256,300,4,8,3]`、RepC3 `[256, 0.5]`。

#### 7.3.1 模块化主干（单行配置，timm 风格）

| 配置文件 | 主干模块 | 机制与作用 |
|---|---|---|
| rtdetr-convnextv2.yaml | `convnextv2_atto`（backbone/convnextv2.py） | ConvNeXt V2：深度可分离 7×7 卷积 + LayerNorm + GRN 全局响应归一化；FCMAE 掩码自监督预训练框架；atto 为最小规格。现代卷积基线 |
| rtdetr-fasternet.yaml | `fasternet_t0`（backbone/fasternet.py） | FasterNet（CVPR2023）：PConv 部分卷积——仅对 1/n 通道做常规卷积、其余通道原样通过，在冗余计算上省时延；t0 为最小版 |
| rtdetr-mambaout.yaml | `mambaout_femto`（backbone/MambaOut.py） | MambaOut：GatedCNNBlock——大核深度卷积 + 门控机制（无 SSM/选择性扫描）；论文结论"视觉任务无长序列依赖则不需要 Mamba"，是与 mamba 系对照的门控 CNN 基线 |
| rtdetr-lsknet.yaml | `lsknet_t`（backbone/lsknet.py） | LSKNet（遥感检测）：大选择性核——两个串联深度卷积（含大核）+ 空间选择机制动态加权核分支，高效获得大感受野 |
| rtdetr-lsnet.yaml | `lsnet_t`（backbone/lsnet.py） | LSNet（"See Large, Focus Small"）：LDB 大核感知 + SKA 小核聚焦（Spatial Kernel Attention，含 Triton 实现），大核提取全局上下文、小核注意力精细化 |
| rtdetr-mobilenetv4.yaml | `MobileNetV4ConvSmall`（backbone/mobilenetv4.py） | MobileNetV4：Universal Inverted Bottleneck（多核深度卷积任意组合）+ Mobile MQA 混合注意力；移动端效率设计 |
| rtdetr-overlock.yaml | `overlock_xt`（backbone/overlock.py） | OverLoCK（arXiv 2502.20087）："Overview-first, Look-Closely-next" 卷积网络——动态大核卷积 LDC + Top/Base 全局上下文建模 + GDSA 聚焦细化注意力，依赖 natten |
| rtdetr-pkinet.yaml | `PKINET_T`（backbone/pkinet.py） | PKINet（CVPR2025，遥感小目标）：多边形核 Inception 网络——并行多尺度多边形（含旋转方向）卷积核 + CAA 上下文锚点注意力；T 为 tiny |
| rtdetr-repvit.yaml | `repvit_m0_9`（backbone/repvit.py） | RepViT：把 ViT 的轻量设计搬回 CNN——MobileNetV3 式架构 + RepVGG 重参数化大核卷积作 token mixer，部署时可融合 |
| rtdetr-rmt.yaml | `RMT_T`（backbone/rmt.py） | RMT（Retentive Networks Meet Vision Transformers）：将 RetNet 的多尺度保留（MSRA）注意力引入视觉，兼顾长程建模与并行推理 |
| rtdetr-starnet.yaml | `starnet_s2 [False]`（backbone/starnet.py） | StarNet（CVPR2024 "Rewrite the Stars"）：以元素乘（star 操作）替代求和实现隐式高维特征空间，极简高效 |
| rtdetr-unireplknet.yaml | `unireplknet_a`（backbone/UniRepLKNet.py） | UniRepLKNet：RepLKNet 升级的多模态通用大核网络（GRN + 大核重参数化卷积），音频/点云/图像统一感知 |

#### 7.3.2 轻量主干 + 标准颈部（改在 r18 形残差主干上）

| 配置文件 | 改进位置 | 机制与作用 |
|---|---|---|
| rtdetr-fadc.yaml | P5 段 BasicBlock → `BasicBlock_FADC` | 频率自适应膨胀卷积：对频域分解后的特征自适应选择膨胀率（ICCV2023 FADC），大物体/低频处用大膨胀率扩大感受野 |
| rtdetr-faster-CGLU.yaml | 全 4 段 BasicBlock → `BasicBlock_Faster_Block_CGLU` | FasterNet PConv 部分卷积做空间混合 + ConvolutionalGLU 门控 FFN，兼顾效率与门控表达力 |
| rtdetr-wConv.yaml | 颈部 2 个下采样 Conv → `wConv2d [256, 3, [0.9], 2]`（block.py:12824） | Weighted Convolution 2.0：卷积权重乘以固定外积矩阵 Phi（由参数 den=0.9 生成的中心加权核），零参数代价强化中心权重 |

#### 7.3.3 iRMB 系列（4 个变体，全 4 段替换）

iRMB（inverted Residual Mobile Block）出自 EMO（ICCV2023 "Rethinking Mobile Block for Efficient Attention-based Models"）：窗口注意力（或恒等）→ 深度卷积 → SE → 1×1 投影的倒残差结构，是"注意力+卷积"轻量混合块的代表作。四个配置的差异全部在 `branch2b` 使用的块（block.py:700-1105）：

| 配置文件 | 核心块 | 与原版 iRMB 的差异 |
|---|---|---|
| rtdetr-iRMB.yaml | `iRMB` | 原版：滑动窗口自注意力（window_size=7）+ 固定 3×3 深度卷积 + SE |
| rtdetr-iRMB-Cascaded.yaml | `iRMB_Cascaded` | 注意力部分换成 `LocalWindowAttention`（级联式窗口注意力，先窗口后通道分组细化） |
| rtdetr-iRMB-DRB.yaml | `iRMB_DRB` | 局部卷积换成 `DilatedReparamBlock`（DRBNet 多膨胀率并行大核，训练并行、推理重参数化）；逐 stage dw_ks=13/11/9/7 |
| rtdetr-iRMB-SWC.yaml | `iRMB_SWC` | 局部卷积换成 `ReparamLargeKernelConv`（UniRepLKNet 风格大核重参数化，分组深度卷积）；逐 stage dw_ks=7/7/5/5 |

（r50 系列对应版 `rtdetr-r50-iRMB[-Cascaded|-DRB|-SWC].yaml` 将同样的块装进 BottleNeck，DRB/SWC 的 dw_ks 参数与 rtdetr 版完全相同。）

#### 7.3.4 Mamba-YOLO 主干（T/B/L 三档）

| 配置文件 | 主干 | 机制与作用 |
|---|---|---|
| rtdetr-mamba-T.yaml / -B / -L | `SimpleStem → VSSBlock_YOLO ×n → VisionClueMerge → … → SPPF`（extra_modules/mamba_yolo.py:506） | Mamba-YOLO 主干：VSS（Visual State Space）2D 选择性扫描块做 token mixer，`VisionClueMerge` 做视觉线索引导的尺度合并，SPPF 收尾；**颈部也换**为 XSSBlock PAFPN（mamba_yolo.py:439，SS2D 驱动的双向融合块）再接 RTDETRDecoder `[nc,256,300,4,8,3]`。三档 scale：T `[0.33,0.25,1024]`（约 6.1M 参数）、B `[0.33,0.50,1024]`（约 21.8M）、L `[0.67,0.75,768]`（约 57.6M）。唯一连颈部一起换掉的主干替换组 |

#### 7.3.5 颈部增强型"主干"配置（r18 主干 + 输出增强）

| 配置文件 | 改进位置 | 机制与作用 |
|---|---|---|
| rtdetr-fsa.yaml | input_proj 之前：对 AIFI 前的 P5 及 backbone P4/P3 各插入一个 `FSA`（通道 20/40/80） | FSA（BIBM2024 医学分割 SFDA-Net，block.py:10642）：频域自适应全局滤波 Adaptive_global_filter + 空间注意力相加，双域增强后再进 CCFF |
| rtdetr-mpcafsa.yaml | 每个融合节点：`MultiScalePCA`（上采样路径）/`MultiScalePCA_Down`（下采样路径）后接 `FSA` | MultiScalePCA（block.py:10501）做跨尺度主成分加权聚合，FSA 做频-空双域增强，逐节点替换"Concat→RepC3"中的融合方式 |
| rtdetr-msga.yaml | CCFF 之后：backbone P3/P4/P5（各 1×1 投影到 256）与对应 FPN 输出成对送入 `MultiScaleGatedAttn`（block.py:9524，Gold-YOLO 聚合-分发式门控注意力） | 多尺度门控注意力：用主干原始特征门控调制颈部输出，得到最终三层解码器输入；主干与 CCFF 本身保持标准结构 |
| rtdetr-gConvC3.yaml | 颈部 4 个 RepC3 → `gConvC3 [256, 0.5]`（block.py:6383） | gConvBlock 出自图像去雾工作（"Rethinking Performance Gains in Image Dehazing Networks"）：门控卷积块替换 RepC3 内部的 Bottleneck 序列 |

### 7.4 颈部补充（slimneck / goldyolo / ASF 组合）

| 配置文件 | 组合内容 | 机制与作用 |
|---|---|---|
| rtdetr-slimneck.yaml | 纯 Slim-Neck | 颈部所有 input_proj/lateral/downsample 的 Conv → `GSConv`（标准卷积 + DWConv 混洗串联，block.py:2568），4 个 RepC3 → `VoVGSCSP`（GSConv 版 CSP，block.py:2612）；以更低 FLOPs 保持特征融合质量 |
| rtdetr-slimneck-ASF.yaml | Slim-Neck + ASF | 在 slimneck 基础上：上采样路径的 Concat → `Zoom_cat`（先对齐尺度再拼接，input_proj 由 3 个减为 2 个），末端增加 `ScalSeq [256]` + `Add`（backbone P3/P4/P5 注意力序列融合后与 FPN 输出相加，ASF-YOLO 检测头增强） |
| rtdetr-slimneck-WFU.yaml | Slim-Neck + WFU | 在 slimneck 基础上：上采样路径的 Concat → `WFU`（Weighted Fusion Unit，block.py:10044，对高层语义上采样特征与低层细节特征做加权融合后再拼接），保留 3 个 input_proj |
| rtdetr-goldyolo.yaml | Gold-YOLO 颈部（NeurIPS2023） | 聚合-分发（Gather-and-Distribute）机制：`SimFusion_4in` 全局聚合 backbone 四层 + `IFM`（transformer 信息融合）→ 两个 `SimFusion_3in` + `InjectionMultiSum_Auto_pool` 注入路径 → `PyramidPoolAgg [352,2]` + `TopBasicLayer` 全局层 → `AdvPoolFusion` 收集-分发回 P3/P4/P5。保留 AIFI，替换其后整个 CCFF |
| rtdetr-goldyolo-asf.yaml | Gold-YOLO + ASF | 与 rtdetr-goldyolo 完全一致的 28 层，末端追加 `ScalSeq [256]` + `Add`（第 29-30 层；`asf_attention_model` 一行已注释停用）——P3 路径再加 ASF 式增强 |
| rtdetr-ASF.yaml / rtdetr-ASF-P2 / rtdetr-ASF-Dynamic | （非本组，供对照） | ASF 模块的三种用法，其中 r50 复验版为 `rtdetr-r50-ASF.yaml`（7.2.3） |

### 本组统计

- **覆盖配置文件 87 个**：基线 14（r18/r34/r50/r50-m/r101/l/x/p2/p6-CTrans/l-GhostHGNetV2/l-RepHGNetV2/l-attention/timm/r50-timm）+ r50 系列 44（AKConv…iRMB-SWC）+ 主干替换与颈部补充 29（mamba-T/B/L、mambaout、convnextv2、fasternet、fadc、faster-CGLU、fsa、gConvC3、goldyolo、goldyolo-asf、iRMB×4、lsknet、lsnet、mobilenetv4、mpcafsa、msga、overlock、pkinet、repvit、rmt、slimneck×3、starnet、unireplknet、wConv）。
- 结构规律：**"改进模块 × 主干深度"双复验**——44 个 r50 配置中 37 个是把 `rtdetr-XXX` 的 BasicBlock 级改进升格为 BottleNeck 级（含 stage 覆盖范围三种：全 4 段 27 个、仅 P4/P5 2 个（KAN/VSS）、仅 P5 7 个）；5 个颈部型配置（DGCST/RetBlockC3/PACAPN/ASF/bifpn）头部逐层对应；2 个结构性特例（RepNCSPELAN 换 ELAN 主干、SRFD 换 stem）。
- 通道/解码器规格随主干匹配：r18 系解码器 ndl=3、RepC3 内层 0.5 缩放；r50 系 ndl=6、RepC3 满宽；r50-m 以 eval_idx=2 折中推理开销；r101 头部 384 通道、AIFI 宽 2048。
- 颈部补充共 4 条技术路线：GSConv/VoVGSCSP 轻量化（slimneck×3）、Gold-YOLO 聚合-分发（×2）、ASF 尺度序列融合（可与 slimneck/goldyolo 叠加）、BiFPN 加权融合；另有输出端增强（FSA/MSGA/MultiScalePCA）与融合块替换（gConvC3/DGCST/RetBlockC3/CSP_PAC）。
- 唯一"连颈部一起换"的主干替换是 Mamba-YOLO 三档（T/B/L）；其余主干替换均保留标准 HybridEncoder，便于横向对比同一颈下不同主干的收益。

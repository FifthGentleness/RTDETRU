## 三、AIFI 尺度内注意力改进配置

AIFI（Attention-based Intra-scale Feature Interaction）是 RT-DETR HybridEncoder 内部唯一的 Transformer 全局注意力模块：对 backbone 输出的最高层 S5/P5 特征（20×20=400 token）做一次"尺度内自注意力 + FFN"的编码器层交互，随后才由 CCFF（FPN+PAN 的 RepC3 融合）做跨尺度特征融合。它对应本仓库八位置改进体系（见 mydocs/AIFI改进位置详解.md）中的位置 1（注意力模式）、2（Token 集合）、3（QKV/打分）、5（FFN）、6（归一化/残差）、7（辅助监督）等抓手的总入口。

改进动机集中在三点：(1) 标准 MHSA 对 N 个 token 是 O(N²) 打分，AIFI 是全网唯一的算力级全局注意力，任何稀疏化/线性化/量化改造都能直接降低编码器开销；(2) P5 分辨率低，VisDrone 类小目标只占 0.5~2 个 token，全局 softmax 被背景/大目标 token 主导，小目标信号容易被"平均掉"——因此大量变体围绕稀疏路由、代理 token、局部窗口、频域增强、register/检索记忆展开；(3) FFN、位置编码、归一化等子结构从未在原版 RT-DETR 中被触碰，属于低成本试错区。

本目录（`ultralytics/cfg/models/rt-detr/`）下共 51 个"只替换 AIFI"的配置，分两族：

- **第三方参考族（l 基线）23 个**：`rtdetr-AIFI-*.yaml`（22 个）+ `rtdetr-AIFIRepBN.yaml`，2025/10 引入的外部改进库菜单（BiliBili 魔鬼面具等），**全部未训练**，仅作机制参考。除 `rtdetr-AIFIRepBN.yaml` 命名无连字符外均为 `rtdetr-AIFI-XXX` 格式。
- **自研实验族（r18 基线）28 个**：`rtdetr-r18-AIFI-*.yaml`（22 个连字符式）+ `rtdetr-r18-AIFI_*.yaml`（6 个下划线式）+ 1 个无扩展名文件 `rtdetr-r18-AIFI-DiffAttnv2yaml`。多数有 VisDrone 训练记录，其中 **AIFI_BlockSparse（0.3219）是唯一超过 baseline（0.3194）的变体**。

两族 yaml 骨架完全一致：PResNet 风格 backbone（ConvNormLayer×3 + MaxPool + `Blocks/BasicBlock` 四阶段，P3/P4/P5 输出）→ 第 9 层 AIFI（唯一改动点，`d_model=1024, heads=8`）→ CCFF（input_proj + RepC3 的 FPN/PAN）→ RTDETRDecoder(300 query)。文件名前缀 l/r18 仅表示对照的训练基线（本 fork 中 r18 用 ResNet-18 预训练权重的 PResNet-BasicBlock 骨架）；`nc: 80` 为 COCO 对照、`nc: 10` 为 VisDrone 实验。有三个配置让 AIFI 额外接收多尺度输入：MuViT 吃 `[P3,P4,P5]` 三层、BlockSparseV18 吃 `[P4 guide, P5 query]`、UPLiFT 吃 `[P3 guide, P5 value]`。

---

### 3.1 第三方参考族 · 注意力模式替换（l 基线，12 个）

这组配置把 AIFI 里的全局 MHSA 整体换成社区已有的高效/稀疏/多尺度注意力（位置 1/3），FFN 与残差拓扑保持原状（fc1/fc2 的 Conv2d 形式 FFN）。全部未训练，作为机制菜单参考。

| 配置文件 | 基线 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-AIFI-ASSA.yaml | l | TransformerEncoderLayer_ASSA | Adaptive Sparse Transformer（CVPR 2024 图像复原）：token 划成 4×4 窗口（shift_size=2 移位窗口），窗口内用 `WindowAttention_sparse` 做稀疏注意力，FFN 换成 FRFN（频域路由 FFN），按内容在窗口内选择性计算。 |
| rtdetr-AIFI-DAttention.yaml | l | TransformerEncoderLayer_DAttention | DAT 可变形注意力：offset 分支（dwconv→2 通道偏移图）为每个 query 预测少量变形采样点，只在采样点处做 QK 打分，复杂度从 O(N²) 降到 O(N·K)，且采样点天然对齐目标形状。 |
| rtdetr-AIFI-DHSA.yaml | l | TransformerEncoderLayer_DHSA | Dynamic-range Histogram Self-Attention：qkv 后接 3×3 dwconv，每头对 Q/K/V 做直方图式动态范围归一化，并用 `softmax_1`（分母+1）替代标准 softmax，弱化注意力过度集中，抑制背景主导。 |
| rtdetr-AIFI-DPB.yaml | l | TransformerEncoderLayer_DPB | CrossFormer 式 DPB 注意力：全量 MHSA 上叠加 MLP 生成的**动态相对位置偏置**（2D 相对坐标 → 每头 bias 加进 logits），group_size=(20,20)，比静态 sincos 更自适应地编码 token 间相对位置。 |
| rtdetr-AIFI-HiLo.yaml | l | TransformerEncoderLayer_HiLo | LITv2 高低频分离注意力：头按 α=0.5 分两半——高频头在 2×2 窗口内做局部注意力保留边缘/纹理细节，低频头对 AvgPool 降采样后的 K/V 做全局注意力，以 O(N·N/r) 近似全局结构。 |
| rtdetr-AIFI-MSLA.yaml | l | TransformerEncoderLayer_MSLA | 多尺度线性注意力：通道 4 等分，各经 3/3、5/5、7/7、9/9 dwconv 后共享 LinearAttention（核分解，O(N)），四路输出按可学习 `scale_weights` 加权融合。 |
| rtdetr-AIFI-MSMHSA.yaml | l | TransformerEncoderLayer_MSMHSA | Mutilscal_MHSA：Q 取原分辨率 token，K/V 取自 `MutilScal`（多膨胀率空洞卷积+池化降采样）的压缩 token，Q@K^T 复杂度随 KV 分辨率下降而降低；输出再叠一路 SE 式通道门控残差。 |
| rtdetr-AIFI-SHSA.yaml | l | TransformerEncoderLayer_SHSA | SHViT 单头自注意力：只取 25% 通道（pdim）经 GroupNorm 后算单头注意力（qk 降维至 50%），其余 75% 通道直通拼接后 1×1 proj 融合（proj BN 零初始化），大幅省算力。 |
| rtdetr-AIFI-TSSA.yaml | l | TransformerEncoderLayer_TSSA | Token Statistics Self-Attention（ToST）：完全不算 QK^T——对 L2 归一化的 w 求平方和，经可学习温度 softmax 得 token 重要性分布 Pi，`attn = 1/(1+Pi·w²)`，用一阶/二阶统计量替代 pairwise 相似度，O(N)。 |
| rtdetr-AIFI-EfficientAdditive.yaml | l | TransformerEncoderLayer_EfficientAdditiveAttnetion | SwiftFormer 高效加性注意力：Q/K L2 归一化后用可学习全局向量 w_g 给 query 打重要性分 A（跨 token softmax 归一化），G=ΣA·query 得全局上下文向量，输出=Proj(G⊙K)+Q，加性打分替代 softmax 矩阵，O(N)。 |
| rtdetr-AIFI-EPGO.yaml | l | TransformerEncoderLayer_EPGO | ACM MM 2025 门控稀疏注意力：conv 式 QKV（key 降维 attn_ratio=0.5），门控子网（1×1→ReLU→1×1→Sigmoid）预测 dynamic_k=N·gate 均值，按行 top-k 保留注意力分、其余置 -inf 后 softmax——注意力图的动态稀疏化。 |
| rtdetr-AIFI-SHSA-EPGO.yaml | l | TransformerEncoderLayer_SHSA_EPGO | SHSA 与 EPGO 的合体：单头、仅 25% 通道参与，注意力行内再经门控 dynamic_k top-k 稀疏化，双重省算力。 |

### 3.2 第三方参考族 · FFN 增强（l 基线，4 个）

注意力保持标准 `nn.MultiheadAttention` 不动，只替换 AIFI 的逐 token 变换（位置 5），给纯逐点 MLP 补空间/频域归纳偏置。

| 配置文件 | 基线 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-AIFI-DML.yaml | l | AIFI_DML | 标准 MHSA + MixFFN（IJCV 2024 SRConvNet）：FFN 隐层拆两半分别过 5×5、7×7 动态深度卷积（DyConv，16 组核加权），门控式融合后 1×1 投影回，用大核 dwconv 补局部空间信息。 |
| rtdetr-AIFI-EDFFN.yaml | l | AIFI_EDFFN | 标准 MHSA + Enhanced Diversified FFN：1×1 升维+3×3 dwconv 门控 FFN 之后再做 8×8 patch 级 2D rFFT，用可学习频域参数（逐频点加权）滤波后 irFFT 回时域，频域增强特征多样性。 |
| rtdetr-AIFI-SEFFN.yaml | l | AIFI_SEFFN | 标准 MHSA + SpectralEnhancedFFN：膨胀 3×3 dwconv 门控 FFN 中插入整图 2D rFFT，对频谱做逐通道可学习仿射（weight/bias）后 irFFT 回来再门控融合，频域修整 FFN 特征。 |
| rtdetr-AIFI-SEFN.yaml | l | AIFI_SEFN | 标准 MHSA + SEFN（WACV 2024 SEMNet）：FFN 输入 1×1 升到 2 倍隐层+3×3 dwconv 后二分门控；另有一路输入特征 2× 下采样→双 3×3 conv→上采样的空间上下文支路与 FFN 隐层 concat 融合，空间先验注入逐 token 变换。 |

### 3.3 第三方参考族 · 归一化 / 位置编码 / 适配器（l 基线，4 个）

不碰注意力与 FFN 主体，改 AIFI 的"外围件"：归一化（位置 6）、位置编码（位置 2）、PEFT 适配器。

| 配置文件 | 基线 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-AIFI-DyT.yaml | l | AIFI_DyT | Dynamic Tanh（"Transformers without Normalization"）替代 LayerNorm：`tanh(α·x)·w+b`，α 为可学习全局标量，逐通道 w/b 仿射——用逐元素非线性近似归一化的"压分布"作用，省掉均值/方差统计。 |
| rtdetr-AIFIRepBN.yaml | l | AIFI_RepBN | RepBN 归一化重参数化：norm1/norm2 换成 `LinearNorm(LN→BN, step=60000)`——训练时逐层混合 LayerNorm 与 BatchNorm（后期逐步切换到 BN），推理时 BN 可合并回线性层，纯归一化改进。 |
| rtdetr-AIFI-LPE.yaml | l | AIFI_LPE [1024,8,400] | 位置编码替换（位置 2）：把固定 2D sincos 换成 `nn.Embedding(400, 1024)` 可学习位置编码（fmap_size=400=P5 token 数），让模型自己学位置先验。 |
| rtdetr-AIFI-Mona.yaml | l | AIFI_Mona | Mona 适配器（CVPR 2025 MonaVis，PEFT 风格）：在 attention 输出后与 FFN 输出后各插一个 Mona——1×1 降到 64 维 → 多算子 `MonaOp`（多尺度卷积组）→ GELU → 1×1 升回，γ 初始 1e-6 的近零残差，作为轻量微调插件而非结构改造。 |

### 3.4 第三方参考族 · 多模块组合叠加（l 基线，3 个）

在同一层内把上述机制纵向堆叠，考察"注意力+FFN+adapter+归一化"的组合收益。

| 配置文件 | 基线 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-AIFI-ASSA-SEFN.yaml | l | TransformerEncoderLayer_ASSA_SEFN | ASSA 窗口稀疏注意力（位置 1）+ SEFN 门控/空间支路 FFN（位置 5）的一级组合。 |
| rtdetr-AIFI-ASSA-SEFN-Mona.yaml | l | TransformerEncoderLayer_ASSA_SEFN_Mona | 在上一组合基础上，norm1 之后、FFN 之后各挂一个 Mona 适配器，即位置 1+5+4/6 的二级叠加。 |
| rtdetr-AIFI-ASSA-SEFN-Mona-DyT.yaml | l | TransformerEncoderLayer_ASSA_SEFN_Mona_DyT | 顶配版：ASSA 稀疏注意力 + SEFN FFN + Mona×2 适配器 + norm1/norm2 全部换成 DynamicTanh（位置 1+5+6+适配器全家桶）。 |

- 组合链路：`ASSA →(+SEFN)→ ASSA-SEFN →(+Mona×2)→ ASSA-SEFN-Mona →(+DyT 归一化)→ ASSA-SEFN-Mona-DyT`，是第三方族里唯一成体系的消融式叠加序列；DyT 版本里 DynamicTanh 以 `channels_last=False` 作用在 BCHW 特征上，且 Mona 与 SEFN 都保留零/近零初始化残差，保证 step-0 行为接近标准层。

---

### 3.5 自研族 · 位置 1：块稀疏注意力主线（r18 基线，8 个文件 / 7 个模块）

自研实验的主战场。V1（BlockSparse）借鉴 sparse-vggt（CVPR 2026）的块稀疏全局注意力：token 分成 4×4=16 token 的块，每 query 块用**均值池化块描述子**算块级相似度并选 top-4 相关 key 块（外加自身块），生成块级 mask 展开成 token 级掩码，稠密矩阵上 masked softmax——只在块内/选中块间计算 QK^T，复杂度从 O(N²) 降到 O(N·(topk+1)·b)（b=块大小），保全局感受野。**是全部 60+ 实验中唯一赢 baseline 的模块（VisDrone 0.3194→0.3219，+0.25pt）**；后续 V16~V19 是围绕它的四种修正尝试，训练结果全部未超 V1。

| 配置文件 | 基线 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-r18-AIFI-BlockSparse.yaml | r18 | AIFI_BlockSparse | V1 原版：上述 top-k 块路由稀疏注意力 + sincos PE + 原 FFN/残差。实证 0.3219，唯一正收益。 |
| rtdetr-r18-AIFI-BlockSparseV16.yaml | r18 | AIFI_BlockSparseV16 | V1 全局分支 + AIFI-C 局部分支**并行**双分支：局部分支为 EfficientViT CascadedGroupAttention（7×7 窗口内逐头组级联），经零初始化逐通道门控 γ 注入，`out = tokens + attn_global + γ·local`。教训：γ 与 CGA 自带零初始化 proj BN 双重门控互相锁死梯度（分支死锁）。 |
| rtdetr-r18-AIFI-BlockSparseV17.yaml | r18 | AIFI_BlockSparseV17 | V16 的**串行**修正：先 AIFI-C 局部精修（residual，靠 CGA 原生零初始化 BN 保证 step-0 等价 V1），再把精修后的 token 送进 V1 全局块稀疏注意力——精修输入同时改善均值池化路由描述子的质量。 |
| rtdetr-r18-AIFI-BlockSparseV18.yaml | r18 | AIFI_BlockSparseV18 | CSB-AIFI 跨尺度版（双输入 `[6,-1]`：P4 guide + P5 query）：P5 query 块按均值池化描述子对 **P4 的 40×40=1600 token 块**做 top-k 路由，稠密 masked 交叉注意力；跨分支复用 V1 的 qkv/proj 权重（零新增注意力参数），仅加一个 Linear(c_p4→c) 输入投影和一个零初始化标量门控；P4 的 K 带 pos、V 不带 pos（保平移等变）。让小目标所在的高分辨率 P4 token 直接参与 AIFI 语义交换。 |
| rtdetr-r18-AIFI-BlockSparseV19.yaml | r18 | AIFI_BlockSparseV19 | LB-AIFI 负载均衡版：把块路由建模为 MoE 路由问题——加一个恒等初始化的 Linear 路由器、软路由概率以零初始化标量偏置进 token logits（直通梯度）、训练期挂 Switch-Transformer 式负载均衡辅助损失（`N·Σf_i·P_i`，权重 0.1，走既有 aux-loss 管线），解决 V1"均匀天空/路面块浪费预算、目标密集块饿死"的问题；可选 k 退火默认关闭。推理图与 V1 同构。 |
| rtdetr-r18-AIFI-TGHBSA.yaml | r18 | AIFI_TGHBSA | TG-HBSA 拓扑引导超图桥接：分支 A 保留 V1 块稀疏（块内交互+top-k 空间路由）；分支 B 把 S5 划成 2D 空间块作为超图顶点，E 个可学习超边经注意力生成软关联矩阵 S[B,E,M]（带距离衰减空间先验）做超图卷积，门控残差注入——为块稀疏补上"任意远块间高阶关系"，防过平滑（借鉴 Hyper-YOLO，NeurIPS 2024）。 |
| rtdetr-r18-AIFI-RABSA.yaml | r18 | AIFI_RABSA [1024,8,4,4,16,true,true,0.0] | Register-Anchored Bridged Sparse Attention：**位置 1+2 融合**——块稀疏核心原封不动（V1 mask 原样复用），新增 16 个无位置 register token（全局均值先验+可学习 embedding 的内容自适应初始化），每次 forward 被全部图像 token 稠密"写入"一次，然后作为**每个稀疏 query 块都可见的常驻 KV**，恢复块稀疏丢掉的"块→register→块"两跳全局路径；输出侧再经零初始化逐通道门控（LayerScale 式）读回 register 上下文（use_readout=true），step-0 与 BlockSparse 完全等价。 |

- 三个特殊配置的补充：**V18** 是唯一多输入 AIFI（P4 跨尺度引导），本质是"位置 1 稀疏 × 跨尺度 token 集合"的组合；**V19** 是唯一引入训练期辅助损失的路由修正（负载均衡 aux loss 经 `route_distill_loss` 属性由 tasks.py 收集）；**RABSA** 是八位置体系中"已验证机制（位置 1）× 最高潜力机制（位置 2）"的正式融合方案，也是下一步算力的主攻方向（目标冲 0.325+）。
- 实证结论（mydocs）：V2~V19 的 18 个衍生全部未超 V1——"简单 top-k 块路由就是甜区"，此位置边际收益已榨干，后续应把 BlockSparse 当既定主干去叠加其他位置。

### 3.6 自研族 · 位置 1：代理中继与 token 压缩（r18 基线，4 个）

思路是引入少量"中转 token"把 O(N²) 变成两段 O(N·M)，或用聚类中心压缩全局交换。

| 配置文件 | 基线 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-r18-AIFI-AgentAttention.yaml | r18 | AIFI_AgentAttention [1024,8,49] | Agent Attention（ECCV 2024）：引入 49 个可学习 agent token，注意力走两段 Q→A→K/V——agent 先从 N=400 个 token 聚合、query 再从 agent 检索，代价 O(N·M+M·N)；保留 softmax 形式但形成秩 ≤49 的低秩瓶颈。实证 0.2920@111ep，负收益。 |
| rtdetr-r18-AIFI-AgentAttentionV2.yaml | r18 | AIFI_AgentAttentionV2 [1024,8,144] | V2 修复版，三处修正：① Q/K 含 pos、**V 不含 pos**（DETR 惯例，防 bbox 回归位置偏置）；② agent 从 49 提到 144（12×12，压缩比 8.2:1→2.8:1，缓解小目标信号在低秩瓶颈中的丢失）；③ 补上原论文的 DWConv(V) 局部细节支路。实证 0.3165@320ep，仍低于 baseline。 |
| rtdetr-r18-AIFI-BA.yaml | r18 | AIFI_BA | Boltzmann Attention-Sampling（CVPR 2025 小目标方向）：能量估计器给每个 token 打能量分（平坦背景=高能冗余、小目标边界=低能显著），温度预测器按场景输出 τ，按 P∝exp(-E/τ) 的 Boltzmann 分布采样保留 K 个低能 token，Q(全量)×K/V(采样子集) 做稀疏注意力（训练期用 Gumbel-Max 保证可微），复杂度 O(N·K)。实证 0.3017@137ep，负收益。 |
| rtdetr-r18-AIFI_ContextCluster.yaml | r18 | AIFI_ContextCluster [1024,64] | 内容自适应聚类交换（Context Clusters, ICCV 2023 + Slot Attention）：64 个可学习锚点经若干轮"token 软分配（余弦相似度）→锚点加权均值重估"形成内容自适应聚类；K×K 的锚点间轻量注意力完成全局交换（代价可忽略）；聚类特征经分配矩阵广播回 token，输出投影零初始化。用自适应分组替代 pairwise 打分。实证 0.1719@16ep，早早中止。 |

### 3.7 自研族 · 位置 1/3：量化与打分函数修正（r18 基线，4 个文件 / 3 个模块）

不动信息流结构，只改"相似度怎么算/多省一位"——打分函数修正型改进的三个探针，实证全部落败。

| 配置文件 | 基线 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-r18-AIFI-BinaryAttention.yaml | r18 | AIFI_BinaryAttention | One-bit QK Attention（CVPR 2026）：仅对 Q/K 用符号函数二值化（sign-STE 直通梯度），V 与输出投影保持全精度（对小目标特征更安全）；已修复 pos 泄漏 bug（pos 只进 Q/K，不进残差流）。大幅省 QK^T 的访存与算力。实证 0.2419@52ep，最惨烈探针之一。 |
| rtdetr-r18-AIFI-QKNorm.yaml | r18 | AIFI_QKNorm [1024,8,1.0] | QK-Norm（ViT-22B，arXiv:2302.05442）：Q/K 投影后做 L2 归一化并配每头可学习尺度 s_q/s_k（可学习温度），把 attention logits 压到统一量级，防止高范数大目标/背景 token 饱和 softmax、饿死小目标梯度；仍是全精度全长度 MHSA。实证 0.3143@275ep（充分训练），-0.5pt。 |
| rtdetr-r18-AIFI-DiffAttn.yaml | r18 | AIFI_DiffAttn [1024,8,0.2] | Differential Transformer（ICLR 2025）：每头拆两组 Q/K，`Attn=(softmax(Q₁K₁ᵀ)-λ·softmax(Q₂K₂ᵀ))·V`——两路注意力图共享的"共模"背景/大目标响应在相减中抵消，差分信号相对放大弱小目标响应；λ 逐头可学习、sigmoid 参数化、初始 0.2（warm start）；全精度全长度，pos 只进 Q/K。实证 0.2587@60ep 中止。 |
| rtdetr-r18-AIFI-DiffAttnv2yaml（无扩展名） | r18 | AIFI_DiffAttn [1024,8,0.0] | 同一个 `AIFI_DiffAttn` 类的复验版：与上一文件**逐字节等价、唯一差异是 dropout 从 0.2 改为 0.0**（lambda_init 仍默认 0.2）。用于排除 dropout 干扰后复测差分注意力；实证中止。 |

### 3.8 自研族 · 位置 1/2：局部-全局混合、跨尺度与序列建模替代（r18 基线，4 个）

在"谁能看谁"之外，这组改的是注意力的空间组织或干脆换掉注意力机制。

| 配置文件 | 基线 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-r18-AIFI-Neighborhood.yaml | r18 | AIFI_Neighborhood [1024,8,7,1] | 门控局部-全局 AIFI：全局分支保留原版全精度 MHSA（γ_g=1 不动）；并行加 Neighborhood Attention（NAT, CVPR 2023）局部分支——每个 token 只与 7×7 邻域做窗口内精确 softmax（F.unfold 纯 PyTorch 实现，可膨胀），零初始化门控 γ_l=0 预热注入。全局 softmax 会把邻域结构平均掉，局部分支保住聚簇小目标的成对精确匹配。实证 0.3134@244ep，未过线。 |
| rtdetr-r18-AIFI-C.yaml | r18 | TransformerEncoderLayer_LocalWindowAttention [1024,4] | AIFI-C（DCI-DETR 同款，EfficientViT CVPR 2023）：7×7 局部窗口内的 CascadedGroupAttention——每头只管 1/4 通道、逐头级联复用前面头的 K/V、q 上带 dwconv，proj BN 零初始化；窗口内分组注意力省显存省算力。未训练。 |
| rtdetr-r18-AIFI-MuViT.yaml | r18 | AIFI_MuViT [256,1024,8]（三输入 `[P3,P4,P5]`） | MuViT 式多尺度坐标对齐注意力（CVPR 2026）：P3/P4/P5 全部 token 嵌入共享归一化 2D 坐标系（坐标式 sincos）+ 可学习尺度嵌入区分层；P5 作 query 层、三层作 K/V，P5 的 400 query 直接 attend 到高分辨率的 2100+ P3/P4 token——不动编码器分辨率的前提下把小目标细节引入 AIFI。实证 0.2486@46ep，中止。 |
| rtdetr-r18-AIFI_SelectiveScan.yaml | r18 | AIFI_SelectiveScan [1024] | SS-AIFI：用 VMamba（NeurIPS 2024）风格的 2D 选择性扫描（SS2D）**替代 softmax 注意力**——token 沿行/逆行/列/逆列 4 方向展平，过输入相关的选择性 SSM（Δ/B/C 门控逐 token 决定吸收多少全局上下文），线性复杂度 O(N)；纯 PyTorch 分块并行扫描实现（无自定义 CUDA 核）。未训练。 |

### 3.9 自研族 · 频域/能量引导的注意力输入增强（r18 基线，2 个）

不省算力，而是把"信号质量"做进注意力：Q/K 或注意力分支先经频域/频率分解提纯，再进标准 MHSA。

| 配置文件 | 基线 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-r18-AIFI-SET.yaml | r18 | AIFI_SET | Spectral-Enhanced AIFI（SET, CVF Open Access 小目标检测）：2D rFFT → 3×3 ComplexConv2d（跨频点空间平滑）→ σ(real) 得 [0,1] 增益掩码 → 频域 Hadamard 乘 → irFFT，再经 1×1+Sigmoid 门控得到提纯特征 X_purified；**Q/K 用提纯特征**（高信噪比小目标响应驱动注意力权重），V 保留原特征（不破坏大目标语义）。实证 0.2903@113ep，负收益。 |
| rtdetr-r18-AIFI-DFA.yaml | r18 | AIFI_DFA | Frequency-Aware Affinity（CVPR 2026 弱监督分割方向）：标准残差+FFN 结构下，把单路全局注意力拆成低频亲和分支（保物体级语义）+ 高频亲和分支（强化边缘/纹理证据），两路 AffinityAttention 结果融合输出——用频率分解给小目标的边缘证据留专用通道。实证 0.3091@189ep，负收益。 |

### 3.10 自研族 · 位置 4：输出融合与高分辨率细节救援（r18 基线，3 个）

原版注意力路径完全不动，另挂一条零初始化门控的细节/增强分支——与 BlockSparse 完全正交、约 10 行改动的低成本叠加位。

| 配置文件 | 基线 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-r18-AIFI-InvDet.yaml | r18 | AIFI_InvDet | InvDet 式目标感知信息保持（CVPR 2026 红外小目标）：原 AIFI 残差不动，加 TARM 式细节分支——avg_pool 分出低频结构、x−low 为高频残差，高频经以低频为条件的 Sigmoid 门（3×3 conv）选择性保留，低频乘可学习增益 (1+low_gain)；再由 GCTM 式几何/内容容差权重（局部残差能量+内容范数）软加权分支强度，防止平坦背景主导。 |
| rtdetr-r18-AIFI-UPLiFT.yaml | r18 | AIFI_UPLiFT [256,1024,8]（双输入 `[5,8]`：P3 guide + P5 value） | UPLiFT 式高分辨率引导（CVPR 2026）：P3 特征经 LocalAttender（guide 图生成紧凑注意力图、value 在固定局部邻域内池化，num_connected=5）上采出高分辨率细节，池化回 P5 尺寸后经零初始化门控残差注入 AIFI——让 P5 的注意力"看到"P3 级细节而不移动编码器分辨率。 |
| rtdetr-r18-AIFI-HOPS.yaml | r18 | AIFI_HOPS [1024,8] | HOPS 式注意力感知过滤 + 亲和引导增强（CVPR 2026 开放词表分割）：AFM 部分——用 AIFI 各头注意力图与 objectness 先验做逐层投票过滤（注意力>层均值投票 + 类置信度阈值）筛出前景 token；AEM 部分——亲和图引导的上下文传播增强输出分支；主 MHSA/FFN 残差流保留。改的是注意力图的后处理与输出增强。 |

### 3.11 自研族 · 位置 2：Token 集合扩展与跨图记忆（r18 基线，2 个）

打破"参与注意力的只有 400 个图像 token"的默认假设——这是八位置体系中唯一"机制全新且完全空白"的象限（RABSA 是它与位置 1 的融合版，见 3.5）。

| 配置文件 | 基线 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-r18-AIFI_RegisterMemory.yaml | r18 | AIFI_RegisterMemory [1024,8,16,10] | RM-AIFI（"Vision Transformers Need Registers", ICLR 2024）：16 个无位置 amodal register token 与图像 token 拼接后**平等参与全量自注意力**（双向 attend），吸收全局上下文、消解注意力 artifact；主注意力路径零改动。训练期附加任务语义接地：register 池化输出对 batch GT 的图像级类别存在向量做 BCE（经 compute_ground_loss 收集，pos_weight=10），逼 register 携带检测语义。原方案中的 decoder-query EMA 蒸馏已简化掉。未训练——位置 7 辅助监督的"最后一搏"。 |
| rtdetr-r18-AIFI_RetrievalAugmented.yaml | r18 | AIFI_RetrievalAugmented [1024,8] | RA-AIFI（RAT, NeurIPS 2024）：非参数 FIFO 记忆库（8192 容量 MoCo 式队列）跨 batch 存储无位置 P5 内容特征；每个 token 检索 top-k 相似历史条目，用共享 qkv/proj 权重做交叉注意力（零新增注意力参数），检索上下文经零初始化门控加回图内注意力输出——**全网唯一能跨图像积累/传递知识的机制**，对稀有/小目标类别的"数据集级先验"理论上可行。未训练。 |

### 3.12 自研族 · 位置 7：训练期辅助监督（r18 基线，2 个）

前向结构与原生 AIFI 完全相同或近同，创新活在训练信号里；推理零开销。实证上这是死亡率最高的位置（与 DETR 匹配损失梯度打架的嫌疑）。

| 配置文件 | 基线 | 改进模块 | 机制与作用 |
|---|---|---|---|
| rtdetr-r18-AIFI_ForegroundGuided.yaml | r18 | AIFI_ForegroundGuided [1024,8] | FG-AIFI（Focus-DETR/Sparse DETR 思路）：小 MLP 前景打分器给每个 P5 token 打分，训练期用投影到 P5 网格的 GT 框（token 中心落入 GT 内为正）做带 pos_weight=5 的 BCE 监督；sigmoid 分数经零初始化门控作为**加性注意力 logits 先验**（保序、不丢 token）；推理期仅用打分器，零额外监督。实证 0.3032@136ep，负收益。 |
| rtdetr-r18-AIFI_FoundationDistill.yaml | r18 | AIFI_FoundationDistill [1024,8] | SD-AIFI 基础模型蒸馏：离线用冻结 SAM ViT-B/DINOv2 抽取训练图稠密特征存 .npy（FeatureStore 惰性 LRU 加载）；训练期 AIFI token 与双线性对齐后的基础模型特征做余弦损失（Linear(ref_dim→c) 对齐头经 compute_ground_loss 收集）；前向图与原生 AIFI 完全相同，推理可丢对齐头、零开销。实证 0.2685@61ep，中止。 |

---

### 本组统计

本组共 **51 个配置**（第三方参考族 l 基线 23 个：注意力替换 12 + FFN 增强 4 + 归一化/PE/adapter 4 + 组合 3；自研族 r18 基线 28 个：块稀疏主线 7 + 代理/聚类压缩 4 + 量化打分 3 模块/4 文件 + 局部全局/跨尺度/扫描 4 + 频域能量 2 + 输出融合分支 3 + Token 集合 2 + 辅助监督 2），涉及 **50 个独立模块 token**（DiffAttn 两文件共用一个类），全部改动集中于第 9 层 AIFI、backbone 与 CCFF 零改动；按八位置体系分布约为位置 1（30 文件，59%）、位置 4（6 文件）、位置 5（8 文件）、位置 2（4 文件）、位置 3（6 文件）、位置 6（5 文件）、位置 7（4 文件，占比可跨类重叠）；训练结果上唯一正收益为 AIFI_BlockSparse（0.3219 vs baseline 0.3194，+0.25pt），第三方 23 个全部未训练，自研族其余探针均负收益或中止。

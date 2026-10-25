## 八、自研骨干与编码器创新主线

> 本组是 RTDETRU 项目最核心的自研创新部分，共 **21 个配置文件**，覆盖两条独立创新维度（Backbone 骨干创新 + Encoder 编码器创新）及其组合。模块实现位于 `ultralytics/nn/extra_modules/`：`dsawacga.py` / `dsawacga_v2.py` ~ `dsawacga_v7.py`、`dsadoc_v10.py` / `dsadoc_v11.py`、`saa.py` / `saav2.py`、`csfh.py`、`hybrid_encoder_p2_spd_okm.py`、`hybrid_encoder_p2_spd_okm_v2.py`、`hybrid_encoder_p2_spd_okm_fs_v5/v6/v7.py`、`dlff.py`。
>
> 注意：`mydocs/RTDETRU模块与配置总结.txt` 生成于 2026-08-31，仅覆盖到 DSAWACGA v1/v2、DSADOC v10/v11、编码器基础版/V5/V6；v3–v7、SAA/SAAv2、DLFF 与编码器 V7/V2 为后续新增，本节以**当前代码为准**（个别 YAML 头注释与代码不一致处已逐条标注）。

### 8.1 总体设计与公共骨架

**双维度创新体系**：Backbone 创新与 Encoder 创新完全解耦，可自由组合（见 8.5）。

**Backbone 创新分两代**：

- **第一代（branch2a 替换线，预训练权重兼容）**：保持 PResNet-18vd 骨架不动，仅把每个 stage 最后一个 BasicBlock 的 `branch2a`（3x3 卷积）替换为自研模块，`branch2b`（3x3）与残差 shortcut 完全不动，前向语义为 `act(branch2b(module(x)) + short(x))`。stage 输出通道/stride 不变（64/128/256/512 @ P2/4~P5/32），因此与 PResNet 预训练权重兼容，且能和任何 P2 编码器无冲突组合。代表：DSAWACGA(v1/v2)、DSADOC_v10/v11、SAA。
- **第二代（CSFH 框架重构线，整体换骨架）**：放弃 PResNet，改用"两级 stride=2 Conv 下采样 + C2f 容器"的 CSFH 式骨干：`Conv(3→64,s2) → Conv(64→128,s2) → C2f_XXX(128) → Conv(128→256,s2) → C2f_XXX(256) → Conv(256→384,s2) → C2f_XXX(384) → Conv(384→384,s2) → C2f_XXX(384)×3`，通道布局 **128/256/384/384**（P2/P3/P4/P5）。核心块为 Transformer 风格双阶段结构（Mixer + FFN + LayerScale），逐版本迭代出 DSAWACGAv3→v7 与 SAAv2。CSFH 本身来自 CSFPR-RTDETR 论文（移植实现），是这一代的"框架供体 + 对照基线"。

**第一代公共主干骨架**（v1/v2/DSADOC/SAA 配置共享，layer 0~3）：

| 层 | 模块 | 输出（640 输入） | 尺度 |
|---|---|---|---|
| L0 | ConvNormLayer(3→32, 3x3, s2) | B,32,320,320 | P1/2 |
| L1 | ConvNormLayer(32→32, 3x3, s1) | B,32,320,320 | — |
| L2 | ConvNormLayer(32→64, 3x3, s1) | B,64,320,320 | — |
| L3 | nn.MaxPool2d(3,2,1) | B,64,160,160 | P2/4 |

4 个 stage（`Blocks*` 容器，每 stage 2 个 block）：L4 stage2 64ch@160（首 block stride=1）、L5 stage3 128ch@80（P3/8）、L6 stage4 256ch@40（P4/16）、L7 stage5 512ch@20（P5/32）。替换模块的 `dim = ch_out`，创新支路通道 `cr = dim//2`（32/64/128/256）。

**两种检测头**：

- 标准头（12 个配置）：input_proj(P3/P4/P5)→AIFI(1024,8头)→FPN/PAN(RepC3, 256ch)→RTDETRDecoder([X3,F4,F5], nc=10, hidden=256, 300 queries, 4/8/3 层)。
- P2 头（9 个配置）：backbone 4 个 stage 全部输出（return_idx [0,1,2,3]），P2/P3/P4 原始特征直连（input_proj[0..2]=Identity），仅 P5 做 Conv1x1+BN 投影 512→256；CCFF 的 P3 融合路径被重构为 CCFFP2 系列（见 8.4），f3 由 P2/P3/Y4 三路融合而来，小目标特征保留更完整，代价是 160×160 的 P2 参与计算。

**公共约定**：VisDrone-DET 数据集 `nc=10`，输入 640×640；`scales: l: [1.00, 1.00, 1024]`（无缩放）；头/FPN/PAN 用 ultralytics `Conv`（Conv2d bias=False + BN），与 rtdetr_pytorch 的 ConvNormLayer 数学等价；`RepC3` 与 rtdetr_pytorch `CSPRepLayer` 数学等价（RepVggBlock == RepConv bn=False）。

**实验得出的硬约束**（后续所有模块设计均遵守）：

- DSA 模块内部禁止残差连接（避免退化为恒等映射）；PDCBlock 内部 stride=1 时禁止残差（外层 CCFFBlock 已有 identity 直通 + 三分支求和双保底）；
- 禁止 scale 缩放因子压缩模块输出（过拟合）；DCC 的 fusion_conv 必须用标量权重（→3）而非逐通道权重（低维逐通道 softmax 过拟合）；
- 训练轮次 350。

### 8.2 DSAWACGA 系列（v1→v7 演进 + 机制详解）

DSAWACGA = **D**ynamic **S**cale **A**ttention（多尺度空间注意力）+ **WA**velet（小波）+ **C**onvolutional **G**ating **A**ttention。系列横跨两代骨干，是本项目的第一创新主线。

#### 8.2.1 v1→v7 演进总表

| 版本 | 配置 / 模块文件 | 骨干容器 | 局部分支 | 全局分支 | 融合方式 | 一句话定位 |
|---|---|---|---|---|---|---|
| v1 | rtdetr-r18-DSAWACGA.yaml / dsawacga.py | PResNet branch2a 替换（64/128/256/512） | DSA 5 分支（DW3/5/7 + 膨胀d4 + Scharr 边缘），5 路 softmax | WACGA：可学习 Haar DWT 4 子带处理（Sobel H/V、Laplacian 固定核）→ Sigmoid 子带融合 → 双线性上采样为注意力图 M，再与 4 头 L2 归一化 QKV 温度注意力输出相乘 | cross_gate（双路 GAP→1x1+Sigmoid 全局门控）+ 1x1 融合 | 初版：空间+小波双创新支路 |
| v2 | rtdetr-r18-DSAWACGAv2.yaml / dsawacga_v2.py | 同上 | DSA 重设计为 6 分支（DW1/3/5/7/9/11），删 Scharr 与膨胀卷积 | WaveletModulation（纯小波注意力图，**删除 QKV 注意力**） | `A ⊙ fdsa + fdsa`（SME-DETR DPF 式调制+残差，等价 (1+A)⊙fdsa）；cross_gate 删除 | 小波调制成形版 |
| v3 | rtdetr-r18-DSAWACGAv3.yaml / dsawacga_v3.py | **CSFH 框架**（C2f_DSAWACGA，128/256/384/384） | DSA 6 分支（独立算子，占满 dim） | WaveletGlobal：DWT 4 子带 → BN → FPE → 动态分组加权（softmax+分组 1x1+einsum）→ GELU → 双线性近似重建 | Mixer 内 cat→GELU→SE→ca_conv；块内 β/γ LayerScale | 迁入 CSFH 双阶段块框架 |
| v4 | rtdetr-r18-DSAWACGAv4.yaml / dsawacga_v4.py | C2f_DSAWACGA_v4 | WaveletModulatedDSA（(1+A)⊙fdsa，占满 dim，删 Conv3x3 支路） | 回归 CSFH FourierUnit（FFT 真全局感受野），替换 v3 的 WaveletGlobal（DWT level=1 仅 2×2 感受野） | 同 v3（SE + ca_conv） | 局部调制化 + 全局换回 FFT |
| v5 | rtdetr-r18-DSAWACGAv5.yaml / dsawacga_v5.py | C2f_DSAWACGA_v5 | WaveletModulatedDSA（同 v4） | FocusedLinearAttention（线性注意力 + focusing factor + v 路 DWConv） | SE；FFN 换 StarFFN | 全局换线性注意力，FFN 换 Star 操作 |
| v6 | rtdetr-r18-DSAWACGAv6.yaml / dsawacga_v6.py | C2f_DSAWACGA_v6 | DSA + 残差（x + DSA(x)） | GlobalBranch：DWT→4 子带各自处理（LL 1x1 / LH DW(1,3) / HL DW(3,1) / HH 1x1+Tanh）→ **可学习 HaarIDWT 精确重建** → SE → FLA | SE；FFN 带 identity 残差 | 全子带重建版（由"仅 LL 重建"的旧 v6 升级） |
| v7 | rtdetr-r18-DSAWACGAv7.yaml / dsawacga_v7.py | C2f_DSAWACGA_v7 | WaveletModulatedDSA（同 v5/v4） | Frequency_Convolution（AFNO/FourierSR：块对角复数线性谱卷积 + 复数 ReLU 激活 + softshrink 谱稀疏化 + 残差） | SE；FFN 同 StarFFN | 全局换显式谱滤波算子 |

#### 8.2.2 v1（DSAWACGA）机制详解

半通道分流设计：输入 `x [B,dim,H,W]` 沿通道 chunk(2)，**前半（cr 通道）走创新支路，后半走 Conv3x3+BN+ReLU 局部支路**，两路 Concat 后交 branch2b 融合。以 stage5（dim=512, cr=256, 20×20）为例：

- **DSA 分支（5 路）**：`f1=DW3x3, f2=DW5x5, f3=DW7x7, f4=DW3x3(dilation=4)`，`fs=Scharr 边缘幅值 √(gx²+gy²+1e-8)`（固定 Scharr 算子分组卷积 buffer，使用时减去空间均值）；5 路 cat → `weight_conv 1x1(cr*5→5)` → softmax(dim=1) 得空间自适应权重 → 加权求和 → `channel_mix 1x1` 输出 fdsa。
- **WACGA 分支**（与 DSA 同输入）：`LearnableHaarDWT`（dec_lo/dec_hi 为 nn.Parameter，Haar 系数初始化，外积构造 2x2 核，stride=2 分组卷积一次得 4 子带）→ LL 子带 `ya_proj 1x1(cr→cr/4)+ll_conv DW3x3`；LH 子带 `yh_conv(1,3)+固定 Sobel 水平核`；HL 子带 `yv_conv(3,1)+固定 Sobel 垂直核`；HH 子带 `Tanh+yd_proj 1x1+固定 Laplacian 核`；4 子带 cat → `subband_fusion DW3x3+Sigmoid` → 双线性上采样回原尺寸得小波注意力图 M。另一路 `qkv 1x1(cr→3cr)+DW3x3` → 4 头注意力（q/k 先 L2 归一化，temperature 可学习）→ `project_out 1x1` → `fwacga = out * M`。
- **cross_gate 交叉门控**：GAP(fdsa)、GAP(fwacga) cat → `1x1(2cr→2cr)+Sigmoid` 得门控权重 → `fused = cat([fdsa,fwacga]) * gate` → `dsawacga_fusion 1x1(2cr→cr)+BN+SiLU`。

各 stage 内部 BHW：stage2 cr=32@160×160、stage3 cr=64@80×80、stage4 cr=128@40×40、stage5 cr=256@20×20。

#### 8.2.3 v2 关键改动（对照代码，YAML 头注释部分过时）

- **cross_gate 删除**：全局门控"空间盲"（对 VisDrone 小目标有害）、与融合层功能冗余、Sigmoid(0,1) 纯抑制；每 stage 省 2·cr² 参数（4 stage 合计约 174K）。
- **DSA 重设计**：5 分支 → 6 分支（DW1x1/3/5/7/9/11，6 路 softmax）。删除 Scharr（固定核，与可学习分支 softmax 竞争时易被抑制）与膨胀卷积（由更大标准核替代，消除网格伪影）。
- **WACGA → WaveletModulation**：删除整套 QKV 注意力，只保留 DWT 4 子带处理 + Sigmoid 子带融合 + 双线性上采样，输出空间-通道注意力图 A∈[0,1]。
- **融合范式**（对齐 SME-DETR DPF）：`dsawacga_out = A ⊙ fdsa + fdsa`，残差保证 DSA 信息不丢失，小波调制做空间-通道选择性增强。
- **勘误**：该 yaml 头注释写的"fused by DIRECT concat + Conv1x1+BN+SiLU"是早期方案描述，与当前代码不符（代码中已无 dsawacga_fusion 层）；txt 总结文档同样停留在该旧版描述。**以 dsawacga_v2.py 代码为准**。
- v2 的 DSAWACGAv2 模块同时是 v3 C2f 块内 WaveletModulatedDSA 的雏形（WaveletModulation/LearnableHaarDWT 被 v3–v7 复用 import）。

#### 8.2.4 v3：迁入 CSFH 框架（第二代起点）

- 骨干整体换为 CSFH 框架（Conv 64/128 起步 + C2f 容器，通道 128/256/384/384），`C2f_DSAWACGA` 用 `DSAWACGAv2Block` 替换 C2f 的 Bottleneck：`cv1(c1→2c) → split(2 chunks) → [Block×n] → cat → cv2`。
- `DSAWACGAv2Block` 镜像 SFHF_Block 双阶段结构：Stage1 `x' = Mixer(BN1(x)) * β + x`；Stage2 `x'' = FFN(BN2(x')) * γ + x'`；**β/γ 为 (1,dim,1,1) 零初始化 LayerScale**，训练初期近似恒等映射，保护主干特征。
- `DSAWACGA_Mixer`：`conv_init(dim→2dim) → chunk(2)` → chunk1 走 DSA（6 分支，占满 dim，无内部再分）、chunk2 走 `TokenMixer_For_Global`（conv_init→2dim + **WaveletGlobal** + 残差 + conv_fina→dim）；cat → GELU → **SE 通道注意力**（GAP→Conv→ReLU→Conv→Sigmoid 逐通道门控）→ `ca_conv 2dim→dim`。
- `WaveletGlobal`（v3 全局核心，镜像 CSFH FourierUnit 的动态分组加权但换成 DWT）：DWT 4 子带 → LL: `ya_proj 1x1(dim→dim/4)+DW3x3`、LH: `Conv(1,3)+固定 Sobel 水平`、HL: `Conv(3,1)+固定 Sobel 垂直`、HH: `Tanh+1x1+固定 Laplacian` → cat → `subband_fusion DW3x3` → `BN → FPE(DW3x3+残差)` → 动态权重 `1x1(dim→groups)+softmax` → `fdc 分组 1x1(dim→dim×groups)` → einsum 按组加权 → GELU → **双线性插值近似重建**（非精确 IDWT）。
- `DSAWACGA_FFN`：`conv_init(dim→2dim) → split(4, 各 dim/2) → [identity, DW3, DW5, DW7] → cat → GELU → conv_fina(2dim→dim)`，与 CSFH 的 SFHF_FFN 完全相同。
- 相对 CSFH 的卖点：DWT 局部时频分析保留空间位置与方向子带（FFT 混合所有方向），更适合小目标；两个 mixer 分支均占满 dim 无内部容量浪费。

#### 8.2.5 v4–v7 逐版差异（均在 C2f 双阶段框架内，只换 Mixer 内组件 / FFN）

- **v4**（C2f_DSAWACGA_v4）：
  - 局部：WaveletModulatedDSA 替换独立 DSA——DSA 为"内容提供者"、WaveletModulation 为"调制器"，`(1+A) ⊙ fdsa` 比 v3 的并行 cat 耦合更紧；DSA 占满 dim（v2 时代 dim 的一半给 Conv3x3 支路属容量浪费），Conv3x3 支路删除（与 DSA 的 dwconv3、FFN 冗余）。
  - 全局：FourierUnit（FFT，真全局 H×W 感受野）替换 WaveletGlobal（DWT level=1 等效感受野仅 2×2，"全局"名不副实）——这是对 v3 全局分支的纠偏。
- **v5**（C2f_DSAWACGA_v5）：
  - 全局：FocusedLinearAttention（q/k 先 `relu(·)^focusing_factor(f=3)` 再做线性复杂度 kv 聚合，v 路 + DW5 空间偏置）替换 FourierUnit，TokenMixer 外层带 identity 残差。
  - FFN：**StarFFN**（Star 操作）——`fc1(dim→2dim) → split(x1,x2)`；x1 再 split 后分别 DW3/DW5 → cat → GELU 得空间感知门控，`star = GELU(x1') * x2` → `fc2 + x 残差`。相对 SFHF_FFN：非线性由多项式核（乘性 Star）替代 GELU（加性 Cat），多尺度只作用于门控路。
  - **勘误**：该 yaml 头注释写的"CA=FrequencySpatialAttention、FFN=MultiScaleGatedFFN、全局=FourierUnit"在当前代码中**不存在**（全仓库无这两个类名），头注释为过时方案；代码实际为 SE + StarFFN + FocusedLinearAttention。
- **v6**（C2f_DSAWACGA_v6）：Mixer 结构改为"低频/高频分治"：
  - 局部：`x_local + DSA(x_local)`（多尺度局部 + 显式残差）；
  - 全局：`GlobalBranch` = LearnableHaarDWT → 4 子带**各自方向化处理**（LL: 1x1 全局结构；LH: DW(1,3) 水平边缘；HL: DW(3,1) 垂直边缘；HH: 1x1+Tanh 轻量对角纹理）→ `LearnableHaarIDWT_Full`（ConvTranspose2d 伴随算子，正交 Haar 下**精确逆变换**）→ SE → FLA → `x_se + fla(x_se)`；
  - 相对"旧 v6（仅 LL 重建、高频置零）"的升级点：IDWT 保留对小目标关键的高频边缘信息；相对 CSFH 的 FFT：DWT 提供方向性分解。FFN 换回 [id,DW3,DW5,DW7] 结构并带 identity 残差。
- **v7**（C2f_DSAWACGA_v7）：只动全局分支——`Frequency_Convolution`（自适应傅里叶神经算子 AFNO，FourierSR 改造）替换 v5 的 FocusedLinearAttention：rfft2(norm=ortho) → reshape 成 num_blocks=8 个块 → **块对角复数线性** `einsum('bkihw,kio->bkohw')`（参数高效、保持复数结构）→ 复数 ReLU（real/imag 各自可学习权重组合，物理意义明确）→ **softshrink 谱稀疏化**（阈值 0.01，显式剔除噪声频点）→ irfft2 → 残差。相对 CSFH FourierUnit：无 BN/FPE/动态 softmax，靠显式谱滤波；相对 v5 线性注意力：真频域算子。
- 本配置 Decoder 行为 `[nc]` 单参（其余用默认 hidden=256/queries=300 等），与其他配置 `[nc,256,300,4,8,3]` 显式写法等价。

#### 8.2.6 演进脉络小结

- 融合范式：cross_gate 全局门控（v1）→ 直接调制 + 残差（v2 起，验证门控冗余）。
- 局部分支：5 分支+Scharr+膨胀（v1）→ 6 分支标准核（v2 起，固定）→ 由"独立 DSA"升级为"小波调制 DSA"（v4 起，固定）。
- 全局分支：小波注意力图（v1/v2 嵌在 WACGA 内）→ WaveletGlobal 近似重建（v3）→ FFT 真全局（v4）→ 线性注意力（v5）→ DWT 精确重建（v6）→ 显式谱卷积（v7），在"频域 vs 线性注意力"间反复对照实验。
- 容器：PResNet branch2a 替换（v1/v2，保预训练）→ C2f 双阶段块（v3–v7，自由重设计，靠 β/γ 零初始化保稳定）。

### 8.3 DSADOC 系列（v10/v11）

DSADOC = DSA + **DOC**（**D**WT **O**mnidirectional **C**alibration，小波域校准）。与 DSAWACGA 同属第一代 branch2a 替换线，区别仅在第二创新支路：WACGA（小波注意力）换成 DOC/DCC（小波域子带校准 + FiLM 调制）。

#### 8.3.1 DSADOC_v10 机制详解（dsadoc_v10.py）

半通道分流与 v1 完全一致（前半 cr 通道创新支路 + 后半 Conv3x3 支路，dsadoc_fusion 1x1+BN+SiLU，cross_gate 门控）。DSA 分支与 v1 相同（DW3/5/7 + 膨胀 d4 + Scharr，5 路 softmax + channel_mix）。DCC 分支（DOC 核心，以 stage5 cr=256 为例）：

- 全局描述子：`favg=GAP(xs)`、`fmax=GMP(xs)`；`HaarDWT`（**固定不可学习**，偶数尺寸直接相位下采样组合，奇数先 reflect pad）得 ll/lh/hl/hh 各 [B,cr,H/2,W/2]；`dwt_ll=GAP(ll)`；三者 cat → `fusion_conv 1x1(3cr→3)+softmax` 得**标量权重** α/β/γ → `global_feat = α·favg + β·fmax + γ·dwt_ll`（标量权重而非逐通道是实验硬约束，防低维逐通道 softmax 过拟合）。
- 子带校准：LL 用全 4 子带 cat 的全向空间特征 `spatial_mix_ll 1x1(4cr→cr)+ReLU`；LH/HL/HH 各用高频 3 子带 cat 的方向空间特征 `spatial_mix_xx 1x1(3cr→cr)+ReLU`；`cal_xx = cal_conv_xx(子带) + global_feat + spatial_xx`（逐子带 1x1 + 全局描述子广播 + 空间特征三元相加）。
- 重建与调制：4 个 cal 子带 → `HaarIDWT` 精确重组回 [B,cr,H,W] → `cal_norm GroupNorm(1)` → `film_gen 1x1(cr→2cr)` 拆成 γ_raw/β_raw → **FiLM 调制**：`film_gamma = 1 + s·tanh(γ_raw)`、`film_beta = s·tanh(β_raw)`（s = film_scale 标量参数，**零初始化 = 恒等启动**）→ `fdcc = xs * film_gamma + film_beta`。
- 融合：`fused = cat([fdsa,fdcc]) * cross_gate` → dsadoc_fusion → 与 conv 支路 cat 输出。

#### 8.3.2 DSADOC_v11（dsadoc_v11.py，去门控 + DSA 升级）

- DSA 升级为 DSAWACGAv2 式 6 分支（DW1/3/5/7/9/11，6 路 softmax），删 Scharr 与膨胀卷积（理由同 v2）。
- DOC/DCC 保留 1:1（HaarDWT → 子带校准 → HaarIDWT → GroupNorm → FiLM）。
- **cross_gate 删除**：`cat([fdsa,fdcc])` 直接进 `dsadoc_fusion 1x1+BN+SiLU`（省约 174K 参数），半通道分流与 conv_path 保留。
- 对照实验关系：v10（带门控 + 旧 DSA）↔ v11（去门控 + 新 DSA），与 DSAWACGA v1↔v2 的消融逻辑同构。

### 8.4 P2SPDOKM 编码器系列（基础版 → V5 → V6 → V7）

Encoder 创新主线：引入 P2/4 层 + 重构 CCFF 的 P3 融合路径。P2SPDOKM = **P2** + **SPD**（SPDConv 无损下采样）+ **OKM**（OKNet 大核/频域模块）。所有版本容器数据流一致：

```
P2[B,64,160,160] ──SPDConv──> [B,128,80,80] ─┐
Y4[B,256,40,40] ──Upsample×2──> [B,256,80,80] ─┤ cat [B,512,80,80] → cv1(1x1+BN+SiLU) → split(ratio=0.25)
P3[B,128,80,80] ──────────────────────────────┘        ├─ 创新半区 [B,128,80,80] → 创新块
                                                       └─ 恒等半区 [B,384,80,80]
cat [B,512,80,80] → cv2 → RepC3(512→256) → f3 [B,256,80,80]
```

`_SPDConv`（space-to-depth）：x 取 4 个相位子格 cat 成 [B,4C,H/2,W/2] → ConvNormLayer 3x3 → [B,C_out,H/2,W/2]，无损下采样（不丢小目标信息），是 P2(160×160) 进入 P3 尺度(80×80) 的桥梁。

#### 8.4.1 演进总表

| 版本 | 配置 / 模块文件 | 创新半区算子 | 关键差异 | 状态 |
|---|---|---|---|---|
| 基础版 | rtdetr-r18-P2SPDOKM.yaml / hybrid_encoder_p2_spd_okm.py | BottleNect(128, k=31)（OKNet 原版 verbatim 移植） | FCA→SCA→FGM（fft norm='backward'）+ 4 路大核 DW（1x31/31x1/31x31/1x1），`out = x + 大核和 + x_sca` 残差求和 → ReLU → out_conv；**无 DCFM/FreqScale**，无末端 BN | 起点基线 |
| V5 | rtdetr-r18-P2SPDOKMFSV5.yaml / ..._fs_v5.py | CCFFBlockV5(128) 三分支求和 | 局部 DCFMV5（**PDCBlockV5 带内部残差**）+ 大核 OKNetLargeKernel(31) + 全局 FreqScale→SCA→FGM(ortho)→global_out；`out = fuse_out(small+large+global)` | 次优 |
| V6 | rtdetr-r18-P2SPDOKMFSV6.yaml / ..._fs_v6.py | CCFFBlock(128) | 同 V5，但 **PDCBlock stride=1 去内部残差**（输出纯差分特征）；fs_init_scale=1e-5 | **最终版（最优）** |
| V7 | rtdetr-r18-P2SPDOKMFSV7.yaml / ..._fs_v7.py | CCFFBlockV7(128) | **恢复 identity 残差**：`out = ReLU(x + DCFM(x) + LKB(in_conv(x)) + FCA→SCA→FGM→FreqScale) → 1x1`；fs_init_scale=1e-2，FGM ortho，FreqScale 降级为 FGM 输出的二段精修（warmup） | 修正版 |
| V2（通道适配） | （随 DSAWACGAv5 组合使用）/ hybrid_encoder_p2_spd_okm_v2.py | BottleNect(160) | 新增 `spd_out_ch` 参数，SPDConv 输出与 ch_p3 解耦，适配 128/256/384/384 骨干通道 | 适配版 |

> 勘误：txt 总结记录的"V7 = DCFM 输出后加 BN，最差已删除"对应的是一个**已废弃的旧 V7 尝试**；当前仓库 `hybrid_encoder_p2_spd_okm_fs_v7.py` 的 V7 是"identity 残差保留版 V6"，`rtdetr-r18-P2SPDOKMFSV7.yaml` 即其配置。实验结论应表述为：V6（纯差分）> V5（带残差）> 旧 V7（DCFM+BN，已删）。

#### 8.4.2 CCFFBlock（V5/V6）三分支详解

以创新半区输入 x [B,128,80,80] 为例，三个并行分支求和后 fuse：

- **局部分支（DCFM，差分卷积模块）**：`PDCBlock('cv')`（常规卷积路 VC）与 `PDCBlock('cd')`（中心差分路 CDC，`y − θ·中心聚合`，θ=0.875）并行；每路为 `PDCConv 3x3 DW + BN → ReLU → 1x1`；两路 cat → `attention_fc(1x1+BN+ReLU+1x1 → 2 通道)` → softmax → 逐点加权求和（**AWF 自适应权重融合**）。之后 `cat([x, DCFM(x)]) → small_fuse 1x1(2sc→sc)+BN+SiLU`。
- **大核分支（OKNetLargeKernel, k=31）**：`1x1+GELU → 4 路 DW 大核（1x31 / 31x1 / 31x31 / 1x1）求和 → BN → ReLU`。
- **全局频域分支（FreqScale→SCA→FGM）**：`FreqScale(sc, group=16, filters=4, base=14, reweight_ratio=0.25, init_scale=1e-5)`：1x1+BN → rfft2(ortho) [B,128,80,41] → GAP 路由（Linear+StarReLU+Linear → [B,16,4] tanh）→ 复数权重插值到频谱尺寸 einsum 加权 → irfft2；再 `SCA`（池化通道注意力）→ `_FGM`（频域两段：M 频谱交互 + 门控残差 / R 内容路由复数重加权）；→ `global_out 1x1+BN+SiLU`。
- **汇总**：`out = small_out + large_out + global_out → fuse_out 1x1+BN+SiLU`。
- **V5 vs V6 唯一差异**（PDCBlock stride=1 时）：V5 `y = conv2(ReLU(BN(PDCConv(x)))) + x`（内部残差）；V6 去掉 `+ x`，输出纯差分特征，x 只经 `cat([x, DCFM(x)])` 出现一次（差分作为增量信息）。**V6 > V5**：差分特征幅值本身携带边缘强弱语义。
- **V7 与 V6 的差异**（fs_v7 头注释给出 V6 回归根因）：V6 的 `fuse_out(small_fuse + large + global)` 使 x 必须穿过两个随机初始化 1x1，identity 残差丢失（训练初期信息保持约 30–40%），而基础版 BottleNect 的 `x + 大核 + 全局` 有直通梯度。V7 恢复 `out = x + DCFM(x) + LKB(in_conv(x)) + FCA→SCA→FGM→FreqScale`：DCFM 纯差分直接相加不再 cat 混合；大核与全局共享 `in_conv`（同 BottleNect）；全局串行改为 FCA→SCA→FGM（kaiming 初始即有效）+ FreqScale 精修（init_scale 1e-5→1e-2 加速 warmup）；FGM norm backward→ortho 更数值稳定。

#### 8.4.3 DLFF（dlff.py，V7 的论文化命名）

`dlff.py` 是 `CCFFP2V7` 的 1:1 论文级重命名实现（结构完全一致，仅换名），命名对照：

| 论文名 | 含义 | 工程名 |
|---|---|---|
| DLFF | Difference-Large-kernel-Frequency Fusion | CCFFP2V7 |
| DLFB | DLF Fusion Block（三分支 + identity 块） | CCFFBlockV7 |
| S2D | Space-to-Depth 无损下采样 | _SPDConv |
| DCB | Difference Convolution Branch（VC + CDC + AWF） | DCFM |
| LKB | Large-Kernel Branch（1xK/Kx1/KxK/1x1 DW） | 4 路大核 DW |
| SPT | Shared Point-wise Transform（1x1+GELU，LKB/FGB 共享） | in_conv |
| FGB | Frequency-Guided Branch（FCA→SCA→FGM：M 调制 + R 精修） | FCA/SCA/FGM |
| FSR | FreqScale 二段精修（warmup） | FreqScale |
| IR | Identity Residual：`out = x + DCB + LKB + FGB` | identity 直通 |

DLFF 类支持 `spd_out_ch` 参数（默认 = ch_p3），R18 骨干时 S2D(64→128)、concat=512、split 128/384、DLFB(128)。

#### 8.4.4 CCFFP2V2（通道适配版，hybrid_encoder_p2_spd_okm_v2.py）

为 DSAWACGAv5/v7 骨干通道（P2=128, P3=256, P4/P5=384）适配：新增 `spd_out_ch` 把 SPDConv 输出与 ch_p3 解耦——P2 128ch 直连（Identity），`SPDConv(128→128)`（4×128=512 → Conv3x3 → 128）；P3 256ch 保持原始全信息；concat = 128 + 256 + 256(Y4) = **640**；split(0.25) → 创新半区 **160** + 恒等半区 480；创新块 BottleNect(160, k=31)；RepC3(640→256)。FPN/PAN 全 256ch，与原版 RT-DETR 相同。

### 8.5 组合配置（Backbone 创新 × Encoder 创新）

组合原理：Backbone 创新只改 stage 末块内部结构，输出通道/stride 不变（64/128/256/512 或 128/256/384/384），Encoder 消费这些原始特征；两段 YAML 层索引完全对齐即可无冲突拼接。

| 配置文件 | 骨干创新 | 编码器创新 | 说明（索引对齐关系） |
|---|---|---|---|
| rtdetr-r18-DSAWACGA-P2SPDOKMFSV6.yaml | BlocksDSAWACGA（v1，branch2a→DSAWACGA，64/128/256/512） | CCFFP2V6@L15 `[4,5,14]`（P2=64/P3=128/Y4=256） | 8.1 骨架 L0–L7 + P2 头 L8–L22；SDPConv(64→128)、concat 512、split 128/384、CCFFBlock(128)；超参 large_kernel=31, fs_group=16, fs_num_filters=4, fs_base_size=14, split_ratio=0.25, dcfm_theta=0.875, fs_reweight_ratio=0.25, fs_init_scale=1e-5 |
| rtdetr-r18-DSADOCv10-P2SPDOKMFSV6.yaml | BlocksDSADOCv10（branch2a→DSADOC_v10） | CCFFP2V6@L15 `[4,5,14]` | 与上行完全同构，仅骨干容器换名；DSADOC 与 DSAWACGA 的 stage 输出一致，组合无需改任何索引/通道 |
| rtdetr-r18-DSAWACGAv5-P2SPDOKMv2.yaml | C2f_DSAWACGA_v5（CSFH 框架，128/256/384/384） | CCFFP2V2@L17 `[2,4,16]`（spd_out_ch=128） | 第二代骨干首次接 P2 编码器：P2=128 直连 → SPDConv(128→128)；P3=256 直连；P4=384 经 L13 Conv1x1→256、P5=384 经 L9 Conv1x1→256；concat 640、split 160/480、BottleNect(160)；Decoder@L24 `[17,20,23]` |
| rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7.yaml | C2f_DSAWACGA_v5（同上） | CCFFP2V7@L17 `[2,4,16]`（spd_out_ch=128, fs_init_scale=1e-2） | 与上行同通道设计，创新块换为 CCFFBlockV7（identity 残差 + DCFM 直接增量 + FreqScale 精修）；是 105 组合实验矩阵 `DSAWACGAv5-P2SPDOKMFSV7-*` 的母版（该矩阵不在本组） |

两组通道体系的关键区别：前两个组合沿用 PResNet 通道（P2=64），SPDConv 输出默认 = ch_p3 = 128；后两个组合 P2=128/P3=256，必须显式传 `spd_out_ch=128` 并用 640 通道 concat——这正是 CCFFP2V2/V7 引入 spd_out_ch 参数的动机。

### 8.6 其他自研模块配置（SAA/SAAv2、CSFH、DLFF）

#### 8.6.1 SAA（rtdetr-r18-SAA.yaml，saa.py）——SME-DETR 骨干移植（第一代对照基线）

来自 SME-DETR（IEEE TGRS 2025）的 Scale-Aware Attentional 骨干，沿用 PResNet branch2a 替换策略（`BlocksSAA` 容器，64/128/256/512），作为自研 DSAWACGA/DSADOC 系列的外部对照：

- **DPF（Dual-Path Fusion）= DSA + DCA**，替换 branch2a：
  - DSA（Dynamic Scale-Aware，Fig.2(e)）：`Conv3x3 → 5 分支多尺度 DWConv(3,5,7,9,11) + softmax 标量权重 → Conv1x1`（式(1)）；
  - DCA（Directional Channel Attention，Fig.2(d)）：`AGAP → 水平/垂直 1D 卷积 Conv1xK / ConvKx1 → DWConv+PWConv+Sigmoid` 通道注意力（式 2–4）；
  - DPF 输出 = `Conv1x1(DSA(x) + DSA(x) * DCA(x))`；SAA 模块再做一次 `split(Identity + DPF) → Concat` 半分流。
- **分 stage 的 K 递增设计**（YAML 第 6 个参数 n）：stage2 n=0(K=11)、stage3 n=1(K=13)、stage4 n=2(K=15)、stage5 n=3(K=17)，`K = 11 + 2n`，越深的 stage 通道依赖越长。
- 注意 SAA 的 DSA 是 5 分支无 Scharr 无 DW1x1——正是自研 v2"6 分支 DSA"的改造对象之一。

#### 8.6.2 SAAv2（rtdetr-r18-SAAv2.yaml，saav2.py）——SAA 的 CSFH 框架化重设计

- 骨干换为 CSFH 框架（128/256/384/384），`C2f_DPF` 容器：`cv1 → split → [DPFBlock×n] → cat → cv2`。
- `DPFBlock` 镜像 SFHF_Block 双阶段：Stage1 `DPF(BN1(x))*β + x`、Stage2 `DPF_FFN(BN2(x'))*γ + x'`，β/γ 零初始化。
- DPF 本体不变（DSA + DCA 融合）；`DPF_FFN` 用 4 分支 DW(3/5/7/9)（相对 SFHF_FFN 的 3/5/7 多一档）。
- 每 stage 显式传 dpf_n：`C2f_DPF(128,0)/C2f_DPF(256,1)/C2f_DPF(384,2)/C2f_DPF(384,3)×3`，K=11+2dpf_n 同 SAA。
- 定位：与 DSAWACGAv3 平行的"另一注意力内核（DSA+DCA）塞进同一 CSFH 框架"的对照实验。

#### 8.6.3 CSFH（rtdetr-r18-CSFH.yaml，csfh.py）——第二代框架供体与对照基线

来自 CSFPR-RTDETR 论文的 Cross-Scale Feature Hybridization 骨干（项目 1:1 移植），是与 v3–v7/SAAv2 逐组件对照的参照系（详见 `mydocs/CSFH骨干网络架构与数据流解析.md`）：

- 骨干：Conv(3→64,s2)→Conv(64→128,s2)→`CSFH_Block(128)`→Conv(128→256)→`CSFH_Block(256)`→Conv(256→384)→`CSFH_Block(384)`→Conv(384→384)→`CSFH_Block(384)×3`；P3/P4/P5 = 256/384/384ch。
- `CSFH_Block`（=C2f_SFHF）：C2f 容器内 `SFHF_Block` 替换 Bottleneck；SFHF_Block = norm1→`SFHF_Mixer`→×β(零初始化)+x → norm2→`SFHF_FFN`→×γ+x。
- `SFHF_Mixer`：conv_init(dim→2dim)→split → 局部 `TokenMixer_For_Local`（chunk 后两半分别 3x3 DW dilation=1/2，concat）+ 全局 `TokenMixer_For_Gloal`（1x1→2dim → `SFHF_FourierUnit` → 残差 → 1x1 回 dim）→ cat → GELU → SE 通道注意力 → ca_conv。
- `SFHF_FourierUnit`：rfft2 → real/imag 拼接实数化 → BN → FPE（频域 DW3x3+残差）→ 动态组权重（4 组 softmax）→ 分组 1x1 → einsum → GELU → 重组复数 → irfft2。
- `SFHF_FFN`：split4 → [identity, DW3, DW5, DW7] → cat → GELU → 1x1。
- P5 用 3 个 SFHF 串行（concat 通道 960 → 压回 384）。与自研差异总表见 DSAWACGAv3 文档：局部分支（2 分支膨胀 DW vs 6 分支 DSA）、全局分支（FFT FourierUnit vs WaveletGlobal/DWT）两处不同，容器/块结构/FFN/SE/LayerScale 全同。

#### 8.6.4 DLFF（rtdetr-r18-DLFF.yaml，dlff.py）——编码器创新的论文出口

即 8.4.3 的 CCFFP2V7 论文化命名配置：纯 PResNet-18vd 骨干（`Blocks` 容器，无骨干创新）+ P2 头，`DLFF@L15 [4,5,14]`（S2D 64→128、concat 512、split 128/384、DLFB(128, k=31, fs_init_scale=1e-2)）。YAML 头注释完整记录了 V6→V7 的 identity 残差修复逻辑（V6 训练初期信息保持约 30–40% vs V7/base 约 90%+），是编码器创新主线对外的论文命名版本。

### 本组统计

- **配置总数 21**：DSAWACGA v1–v7（7）、DSADOC v10/v11（2）、P2SPDOKM 基础版/V5/V6/V7（4）、组合配置 4（DSAWACGA-P2SPDOKMFSV6、DSADOCv10-P2SPDOKMFSV6、DSAWACGAv5-P2SPDOKMv2、DSAWACGAv5-P2SPDOKMFSV7）、SAA/SAAv2（2）、CSFH（1）、DLFF（1）。
- **骨干容器分三类**：branch2a 替换线（PResNet 64/128/256/512，BlocksXxx 容器）5 个 —— DSAWACGA v1/v2、DSADOC v10/v11、SAA；CSFH 框架线（C2f 容器，128/256/384/384）7 个 —— DSAWACGAv3–v7、SAAv2、CSFH；vanilla PResNet（Blocks 容器）+ P2 编码器 5 个 —— P2SPDOKM/FSV5/FSV6/FSV7/DLFF；另有组合配置 4 个（骨干段分别复用前两类容器）。合计 21。
- **头型分布**：标准 P3/P4/P5 头 12 个（DSAWACGA×7、DSADOC×2、SAA、SAAv2、CSFH）；P2 头 9 个（P2SPDOKM 系 4、组合 4、DLFF 1）。
- **关键版本结论**：Backbone 侧 cross_gate 被验证冗余（v2/v11 去门控）；DSA 定型为 6 分支（1,3,5,7,9,11）；小波调制 `(1+A)⊙fdsa` 成为 v4–v7 局部分支固定范式；全局分支在 DWT/FFT/线性注意力/谱卷积间迭代。Encoder 侧 V6（纯差分）为最优、V7（identity 残差保留）为修正版、旧 V7（DCFM+BN）已废弃；P2/P3/P4 原始直连 + 仅 P5 投影为固定设计。
- **模块文件与配置一一对应**：每个 yaml 的骨干容器 token（BlocksDSAWACGA/BlocksDSAWACGAv2/C2f_DSAWACGA(_v4/_v5/_v6/_v7)/BlocksDSADOCv10/BlocksDSADOCv11/BlocksSAA/C2f_DPF/CSFH_Block）与编码器 token（CCFFP2/CCFFP2V5/CCFFP2V6/CCFFP2V7/CCFFP2V2/DLFF）均可在 extra_modules 对应文件中检索。

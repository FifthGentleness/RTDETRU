# RT-DETR (RTDETRU) PResNet BasicBlock 变体改进位置分析

> 分析范围：`ultralytics/nn/extra_modules/block.py` 中 r18 系列的 33 个 `BasicBlock_XXX` 变体
> 依据：逐个核实源码中 branch2a / branch2b 的实际替换行为
> 对照配置：`rtdetr-r18-*.yaml` 系列（创新块在 layer 4~7 骨干 stage 内）

---

## 一、PResNet-18 BasicBlock 原版结构（参照基准）

```text
BasicBlock (每 stage 2 个 block)
├── branch2a: ConvNormLayer(ch_in, ch_out, 3, stride) + BN + ReLU
├── branch2b: ConvNormLayer(ch_out, ch_out, 3, 1) + BN
├── shortcut: stride>1 或通道不匹配时 AvgPool + 1x1 Conv（variant='d'）
└── forward:  act( branch2b(branch2a(x)) + short(x) )
```

### 每个 stage 内 2 个 block 的关键差异

| 位置 | stride | 通道 | 恒等残差 | branch2a 约束 |
|---|---|---|---|---|
| **block1（首块）branch2a** | 2（stage2 为 1） | ch_in ≠ ch_out | 无 | 需支持下采样 + 通道变化 |
| block1 branch2b | 1 | 相同 | 无 | — |
| **block2（末块）branch2a** | 1 | 相同 | 有 | 无约束，最自由 |
| block2 branch2b | 1 | 相同 | 有 | — |

### Blocks 容器的替换机制

yaml 写法 `Blocks, [64, BasicBlock_XXX, 2, 2, 'relu']` 中，变体类作为 `block` 类型传入，
容器在循环里 **给 stage 内每个位置都实例化同一个变体类**（首块、末块一视同仁）：

```python
for i in range(count):          # count=2
    self.blocks.append(block(
        ch_in, ch_out,
        stride=2 if i == 0 and stage_num != 2 else 1,  # 只有首块(i=0)有stride
        shortcut=False if i == 0 else True,             # 只有首块无恒等残差
        ...))
```

→ **库里所有变体都是 "stage 级全量替换"（8 个 block 全换），
   不区分首块/末块。**

---

## 二、33 个变体的具体改进位置

### 类别 1：branch2a + branch2b 都替换（8 个）

这些算子支持 stride / 通道变化，所以两个 3×3 位置全换（首末块共 4 处 3×3 全换，全网络 16 处）。

| 变体 | branch2a 替换为 | branch2b 替换为 |
|---|---|---|
| AKConv | AKConv(ch_in, ch_out, stride) | AKConv(ch_out, ch_out) |
| RFAConv | RFAConv(ch_in, ch_out, 3, stride) | RFAConv(ch_out, ch_out, 3) |
| RFCAConv | RFCAConv(ch_in, ch_out, 3, stride) | RFCAConv(ch_out, ch_out, 3) |
| RFCBAMConv | RFCBAMConv(ch_in, ch_out, 3, stride) | RFCBAMConv(ch_out, ch_out, 3) |
| Conv3XC | Conv3XC(s=stride) | Conv3XC |
| DBB | DiverseBranchBlock(stride) | DiverseBranchBlock |
| WDBB | WideDiverseBranchBlock(stride) | WideDiverseBranchBlock |
| DeepDBB | DeepDiverseBranchBlock(stride) | DeepDiverseBranchBlock |

```text
替换后:  act( branch2b'( branch2a'(x) ) + short(x) )
         其中 branch2a' 和 branch2b' 均为新算子（含首块带 stride 的 branch2a'）
```

### 类别 2：只替换 branch2b，branch2a 保持原版 ResNet（22 个）★ 最主流

这些算子大多不支持 stride / 通道变化（如 WTConv2d 源码中 `assert in_channels == out_channels`），
而首块 branch2a 带 stride=2 且通道变化，所以**所有 block 的 branch2a（含末块的）都保留标准卷积**，
只换恒为 stride=1、通道不变的 branch2b（全网络 8 处）。

| 变体 | branch2b 替换为 | 算子特点 |
|---|---|---|
| DCNV2 | DCNv2(ch_out, ch_out, 3) | 可变形卷积 v2 |
| DCNV2-Dynamic | DCNv2_Dynamic | 动态可变形卷积 |
| DCNV3 | DCNV3_YOLO | 可变形卷积 v3 |
| DCNV4 | DCNV4_YOLO | 可变形卷积 v4 |
| DySnake | DySnakeConv + 1×1 融合 | 动态蛇形卷积 |
| PConv | Partial_conv3 + BN + ReLU | 部分卷积（FasterNet） |
| PConv-Rep | Partial_conv3_Rep + BN + ReLU | 重参数化部分卷积 |
| Faster | Faster_Block | FasterNet 轻量块 |
| Faster-Rep | Faster_Block_Rep | 重参数化版 |
| Faster-EMA | Faster_Block_EMA | + EMA 注意力 |
| Faster-Rep-EMA | Faster_Block_Rep_EMA | 重参数化 + EMA |
| faster-CGLU | Faster_Block_CGLU | + 卷积 GLU |
| DRB | DilatedReparamBlock(k=7) | 空洞重参数块（UniRepLKNet） |
| DualConv | DualConv(g=4) | 3×3+5×5 双卷积 |
| SWC | ReparamLargeKernelConv | 大核重参数卷积（SLaK） |
| VSS | VSSBlock | Mamba 视觉状态空间块 |
| ContextGuided | ContextGuidedBlock | 上下文引导块 |
| fadc | AdaptiveDilatedConv | 频率自适应空洞卷积 |
| Star | Star_Block | StarReLU 星型运算块 |
| KAN | choose_kan(KAGNConv2DLayer 等) | Kolmogorov-Arnold 网络 |
| DEConv | DEConv | 差分边缘算子（脉冲神经网络差分卷积） |
| WTConv | WTConv2d | 小波大感受野卷积（ECCV-24） |
| iRMB | iRMB | 高效线性注意力块 |
| iRMB-Cascaded | iRMB_Cascaded | 级联版 |
| iRMB-DRB | iRMB_DRB(dw_ks) | + 空洞重参数 |
| iRMB-SWC | iRMB_SWC(dw_ks) | + 大核重参数 |

```text
替换后:  act( branch2b'( branch2a(x) ) + short(x) )
                                 ^^^^^^^^ 原版 ResNet 3×3（首末块都是）
```

### 类别 3：两个 branch 都不换，branch2b 之后插入注意力（3 个）

| 变体 | 插入模块 | 插入位置 |
|---|---|---|
| attention | AFGCAttention | branch2b 之后、残差相加之前 |
| Ortho | OrthoAttention + SE 激励 | 同上 |
| AggregatedAtt | TransNeXt_AggregatedAttention | 同上 |

```text
结构:  act( attention( branch2b( branch2a(x) ) ) + short(x) )
                    ^^^^^^^^ branch2a/branch2b 完全是原版 ResNet
```

---

## 三、汇总统计

| 改进策略 | 数量 | 代表变体 | 替换范围（全网络） |
|---|---|---|---|
| branch2a + branch2b 全换 | 8 | AKConv、DBB 系列 | 16 处 3×3 |
| 只换 branch2b | 22 | DCN 系列、WTConv、iRMB | 8 处 3×3 |
| 都不换 + 插注意力 | 3 | attention、Ortho | 8 处注意力插入 |
| **合计** | **33** | | |

### 为什么没有"只替换 branch2a"的变体？（约束交叉导致的空集）

| 算子类型 | 约束 | 结果 |
|---|---|---|
| 不支持 stride 的算子（22 个） | 进不了首块 branch2a | → 只能退守 branch2b |
| 支持 stride 的算子（8 个） | 展示型配置追求最大收益 | → 两个 branch 都换 |
| 注意力模块（3 个） | 作用于最终特征 | → 插在 branch2b 之后 |

"只换 branch2a" 需要同时满足"算子能处理 stride/通道变化"且"只想换一个位置"——
这两个条件在该库的场景里从不共存。且容器是全量替换机制，本身不支持选块。

---

## 四、与自研模块（DSAWACGA / DSADOC）的对比

| 维度 | 库里变体（33 个） | DSAWACGA / DSADOC |
|---|---|---|
| **替换哪些 block** | stage 内全部 2 个（全网络 8 个） | **只有末 block**（全网络 4 个） |
| **branch2a** | 全换（8 个）或全保留（22+3 个） | **只换末块的 branch2a**（独一份） |
| **branch2b** | 30 个变体动它 | **保留原版**（承担两路异构特征融合） |
| **改进性质** | 算子级原位替换（换更高级的卷积） | 结构级重构（半通道分流 + 多分支并行 + 跨域融合） |
| **通道处理** | 直通（ch_in → ch_out） | 半通道 chunk 分流：前半创新支路 ∥ 后半 Conv3×3 支路 |
| **stride 约束** | 首块 branch2a 带 stride=2 逼算子适配 | 末块 branch2a 恒 stride=1、ch_in==ch_out，无此约束 |
| **预训练权重兼容** | 8~16 处全变，stage 内权重基本不可复用 | 仅 4 处变化，其余全部直接加载 |
| **容器机制** | `Blocks`（全量替换，不选块） | `BlocksDSAWACGA`（`if i < count-1` 走原生块，仅末块用创新块） |

### 为什么"末块 branch2a"是最优位置

1. **约束天然满足**：stride=1、ch_in==ch_out，模块无需投影/下采样适配
2. **branch2b 成为天然融合适配器**：`cat([dsadoc_out, conv_out])` 的两路异构特征由紧随其后的原版 branch2b 3×3 融合
3. **增强直达输出**：末块输出即 stage 最终输出（P2/P3/P4/P5），直接被编码器消费，不被后续处理稀释
4. **残差保底**：末块有恒等 shortcut，训练初期梯度畅通
5. **特征成熟度合适**：Scharr 边缘 / 小波子带等先验作用于已被 block1 充分处理的成熟特征

```text
自研模块结构（每 stage 末块，以 stage5 为例）:

  x [B,512,20,20]
   ├─ DSAWACGA / DSADOC (替换 branch2a)   ← 半通道分流 + DSA + WACGA/DOC 多分支
   │        ↓ [B,512,20,20]
   ├─ branch2b (原版 3×3)                 ← 融合两路异构特征
   │        ↓
   └─ + x (恒等残差) → ReLU → [B,512,20,20]
```

---

## 五、r50 镜像说明

以上 33 个变体均有对应的 `rtdetr-r50-*` 版本（r50 的 stage 有 4 个 block，替换范围相应扩大为
每 stage 4 个 block 的对应位置，替换策略与 r18 完全相同）。

---

*生成日期：2026-08-31*
*源码依据：`ultralytics/nn/extra_modules/block.py`（逐类核实）、`ultralytics/nn/modules/block.py`（Blocks 容器）*

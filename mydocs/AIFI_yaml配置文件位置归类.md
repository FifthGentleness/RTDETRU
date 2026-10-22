# RT-DETR AIFI 配置文件归类总结（按 8 个改进位置）

> 统计范围：`ultralytics/cfg/models/rt-detr/` 下全部 87 个 AIFI 相关 yaml。
>
> 两个家族：
> - **第三方参考**：`rtdetr-AIFI-XXX.yaml`（2025/10 引入的外部改进库，共 23 个，**全部未训练**，仅作机制菜单参考）
> - **自研实验**：`rtdetr-r18-*.yaml`（含 baseline 系与 DSAWACGAv5-P2SPDOKMFSV7 系两套）
>
> 8 个位置定义见同目录《AIFI改进位置详解.md》：
> 1 注意力模式 | 2 Token集合 | 3 QKV/打分 | 4 输出融合 | 5 FFN | 6 归一化/残差 | 7 辅助监督 | 8 层数/宽度配置

---

## 位置 1：注意力模式 —— "谁能看谁"（约 30 个，挖得最狠）

### 自研（有训练记录）

| YAML | 机制 | 结果 |
|---|---|---|
| AIFI-BlockSparse | 块 top-k 稀疏路由（sparse-vggt, CVPR 2026） | ✅ **0.3219，唯一赢 baseline(0.3194)** |
| AIFI-BlockSparseV2~V19 | 18 个衍生（改路由/块大小/门控） | ❌ 全灭，无一超原版 |
| AIFI-AgentAttention | agent token 中继（ECCV 2024 风格） | ❌ 0.2920@111ep |
| AIFI-AgentAttentionV2 | V2 重构版 | ❌ 0.3165@320ep |
| AIFI-Neighborhood | 邻域窗口注意力 | ❌ 0.3134@244ep |
| AIFI-BA | 桥接注意力（agent 思路变体） | ❌ 0.3017@137ep |
| AIFI-BinaryAttention | 二值化注意力 | ❌ 0.2419@52ep |
| AIFI-DFA | 动态聚焦注意力 | ❌ 0.3091@189ep |
| AIFI-SET | 稀疏增强 transformer | ❌ 0.2903@113ep |
| AIFI-MuViT | 多尺度 ViT 注意力 | ❌ 0.2486@46ep |
| AIFI_ContextCluster | 聚类 token 交换 | ❌ 0.1719@16ep |
| AIFI-C | EfficientViT 级联组注意力 + 局部窗口（DCI-DETR 同款） | 未训练 |
| AIFI_SelectiveScan | SS2D 选择性扫描替代注意力（VMamba 风格，纯 PyTorch） | 未训练 |
| AIFI-HOPS（AFM 部分） | 注意力感知过滤（改注意力图后处理） | 未训练 |

### 第三方参考（未训练）

| YAML | 机制 |
|---|---|
| rtdetr-AIFI-ASSA | 自适应稀疏自注意力 |
| rtdetr-AIFI-DAttention | 可变形注意力 |
| rtdetr-AIFI-DHSA | 层级稀疏注意力 |
| rtdetr-AIFI-DPB | 细节感知注意力 |
| rtdetr-AIFI-HiLo | 高低频分离注意力（LITv2） |
| rtdetr-AIFI-MSLA | 多尺度线性注意力 |
| rtdetr-AIFI-MSMHSA | 多尺度多头注意力 |
| rtdetr-AIFI-SHSA | 单头注意力 + 组归一化 |

**实证结论：位置 1 挤干。** 简单 top-k 块路由就是甜区，更复杂的稀疏/中继/局部模式全部负优化。

---

## 位置 2：Token 集合 —— "谁在场"（3 个，全部未训练，主攻方向）

| YAML | 机制 | 状态 |
|---|---|---|
| AIFI_RegisterMemory | +16 amodal register（ICLR 2024）+ GT 类别接地 BCE | 🔲 未训练 |
| AIFI_RetrievalAugmented | 跨图像 FIFO 检索记忆库（RAT, NeurIPS 2024），全网唯一跨图知识机制 | 🔲 未训练 |
| AIFI_RegisterBlockSparse | **位置 1+2 融合**：register 稠密枢纽（豁免稀疏路由）+ 块稀疏核心 | 🔲 刚建好，待训 |
| rtdetr-AIFI-LPE | 可学习位置编码（第三方） | 未训练 |

---

## 位置 3：QKV/打分函数 —— 放弃

| YAML | 机制 | 结果 |
|---|---|---|
| AIFI-QKNorm | QK 归一化稳定 logits | ❌ 0.3143@275ep |
| AIFI-DiffAttn | 差分注意力消噪 | ❌ 0.2587@60ep |
| AIFI-DiffAttnv2 | V2 修正版 | ❌ 中止 |
| rtdetr-AIFI-EfficientAdditive | 加性打分替代 softmax | 第三方 |
| rtdetr-AIFI-TSSA | token 统计量打分 | 第三方 |

**实证结论：两个探针充分/中止均落后，打分函数修正型改进天花板低。**

---

## 位置 4：输出/分支融合 —— 自研少量、正交可叠加

| YAML | 机制 | 状态 |
|---|---|---|
| AIFI-InvDet | 原路径不动，零初始化门控挂细节分支（TARM 高通门控+低通增益，红外小目标思路） | 未训练 |
| AIFI-UPLiFT | P3 高分辨率细节经 LocalAttender 上采后零初始化残差回 P5 | 未训练 |
| AIFI-HOPS（AEM 部分） | 亲和引导增强输出分支 | 未训练 |
| rtdetr-AIFI-SEFN | SE 特征增强支路 | 第三方 |
| rtdetr-AIFI-EPGO | SHSA + 门控输出 | 第三方 |

**定位：与 BlockSparse 完全正交、改动 ~10 行的零初始化门控是现成叠加点。**

---

## 位置 5：FFN —— 自研空白，参考全在第三方

| YAML | 机制 |
|---|---|
| rtdetr-AIFI-DML | Dense MLP |
| rtdetr-AIFI-EDFFN | 增强多样化 FFN |
| rtdetr-AIFI-SEFFN | SE 加权 FFN |
| rtdetr-AIFI-MSMHSA（CGLU 部分） | 门控线性单元 FFN |
| rtdetr-AIFI-Mona | 多操作 adapter 插件（MonaVis 风格 PEFT） |
| rtdetr-AIFI-ASSA-SEFN / -Mona / -DyT | 第三方组合变体（1+5/6 叠加） |

**定位：ConvFFN（dwconv 补局部归纳偏置）是未动过的低风险抓手。**

---

## 位置 6：归一化/残差 —— 自研空白

| YAML | 机制 |
|---|---|
| rtdetr-AIFI-RepBN | 重参数化 BN 替代 LN |
| rtdetr-AIFI-DyT | Dynamic Tanh 替代归一化 |
| rtdetr-AIFI-SHSA（GroupNorm 部分） | 组归一化 |

**定位：±0.1pt 级稳定性工具，不单独开实验，定型后微调用。**

---

## 位置 7：训练期辅助监督 —— 5 探针全灭，最后一搏在 RM

| YAML | 机制 | 结果 |
|---|---|---|
| AIFI_ForegroundGuided | 注意力注入前景先验 + GT 监督 | ❌ 0.3032@136ep |
| AIFI_FoundationDistill | 基础模型蒸馏 | ❌ 0.2685@61ep |
| （RegisterMemory 的 grounding） | GT 类别存在 BCE（只监督 16 register 池化） | 🔲 未验证 |
| （QGA / SDLoss / TSL / SUAL 等 decoder/loss 侧） | 见训练 run 记录 | ❌ 全部中止 |

**实证结论：辅助 loss 易与 DETR 匹配损失梯度打架；RM 接地最温和，若再败则永久关闭。**

---

## 位置 8：层数/宽度配置 —— 空白

| YAML | 机制 |
|---|---|
| AIFI-BlockSparse-P5x512 | P5 通道 384→512 加宽后再进 BlockSparse（宽度实验） |
| （无） | ×2 层 / P4 下放 / P4+P5 异构双层 均未建 yaml |

**定位：组合拳阶段用算力换精度的选项。**

---

## 总结统计

```
位置    自研yaml  第三方yaml  训练过   有正收益
 1        ~20        8        16个     1个(BlockSparse)
 2         3         1         0        —
 3         3         2         3        0
 4         3         2         0        —
 5         0         6         0        —
 6         0         3         0        —
 7         2         0         2        0
 8         1         0         1        0
```

**核心结论：**
1. 87 个 yaml 中约 1/3 堆在位置 1，但唯一正收益只有 BlockSparse（+0.25pt），18 个衍生全灭——此位置边际收益趋零。
2. 位置 2（Token 集合）是"空白 + 机制全新"的唯一象限，RM/RA/RBS 是下一步算力去向。
3. 位置 4/5 是 BlockSparse 的正交叠加点（零初始化门控、ConvFFN），低成本可试。
4. 位置 3/7 实证判死刑；位置 6/8 留作定型后的微调与组合拳。

# CSFH 骨干网络架构与数据流解析

> 依据源码：
> - `ultralytics/nn/extra_modules/csfh.py`
> - `ultralytics/cfg/models/rt-detr/rtdetr-r18-CSFH.yaml`
>
> 本文以当前 YAML 的实际配置为准。张量顺序统一写作 **BCHW**，即 `(batch, channels, height, width)`。若只说 **B/H/W**，则表示 batch、高度、宽度三个维度；CSFH 内部大量变换只改变通道 `C`，基本不改变 `B/H/W`。

---

## 1. 结论速览

CSFH 骨干网络由 **下采样 `Conv` + 保持分辨率的 `CSFH_Block`** 交替组成：

```text
Input: (B, 3, H, W)
  │
  ├─ Conv 3→64,  3×3, stride=2        → P1/2
  ├─ Conv 64→128, 3×3, stride=2       → P2/4
  ├─ CSFH_Block(128) ×1               → 保持 /4 分辨率
  │
  ├─ Conv 128→256, 3×3, stride=2      → P3/8
  ├─ CSFH_Block(256) ×1               → 保持 /8 分辨率
  │
  ├─ Conv 256→384, 3×3, stride=2      → P4/16
  ├─ CSFH_Block(384) ×1               → 保持 /16 分辨率
  │
  ├─ Conv 384→384, 3×3, stride=2      → P5/32
  └─ CSFH_Block(384) ×3               → 保持 /32 分辨率
```

CSFH 的核心思想可以概括为：

1. **C2f/CSP 结构保留一路直连特征，另一路做深度增强**；
2. 深度增强路径使用 **SFHF_Block**；
3. SFHF_Block 同时建模：
   - **局部空域细节**：不同膨胀率的深度卷积；
   - **全局频域信息**：FFT / Fourier Unit；
   - **通道重要性**：通道注意力；
   - **多尺度前馈变换**：3/5/7 感受野的 FFN；
4. 每个 SFHF_Block 使用可学习残差系数 `β`、`γ`，且初始为 0，使块初始时近似恒等映射，训练更稳定。

---

## 2. 骨干网络逐层 B/C/H/W 变化

以常见输入尺寸 **640×640**、`B=1` 为例：

| 层号 | 模块 | 主要作用 | 输入 shape | 输出 shape | 尺度 |
|---:|---|---|---|---|---:|
| Input | 图像输入 | RGB 图像 | `(B, 3, 640, 640)` | — | /1 |
| 0 | `Conv(3→64, k=3, s=2)` | 初始下采样，提取低级边缘/纹理 | `(B, 3, 640, 640)` | `(B, 64, 320, 320)` | /2 |
| 1 | `Conv(64→128, k=3, s=2)` | 第二次下采样，形成 P2 特征 | `(B, 64, 320, 320)` | `(B, 128, 160, 160)` | /4 |
| 2 | `CSFH_Block(128, n=1)` | 在 /4 尺度做局部+全局混合增强 | `(B, 128, 160, 160)` | `(B, 128, 160, 160)` | /4 |
| 3 | `Conv(128→256, k=3, s=2)` | 下采样到 P3 | `(B, 128, 160, 160)` | `(B, 256, 80, 80)` | /8 |
| 4 | `CSFH_Block(256, n=1)` | 中低层语义增强 | `(B, 256, 80, 80)` | `(B, 256, 80, 80)` | /8 |
| 5 | `Conv(256→384, k=3, s=2)` | 下采样到 P4 | `(B, 256, 80, 80)` | `(B, 384, 40, 40)` | /16 |
| 6 | `CSFH_Block(384, n=1)` | 中高层语义增强 | `(B, 384, 40, 40)` | `(B, 384, 40, 40)` | /16 |
| 7 | `Conv(384→384, k=3, s=2)` | 下采样到 P5 | `(B, 384, 40, 40)` | `(B, 384, 20, 20)` | /32 |
| 8 | `CSFH_Block(384, n=3)` | 最高层密集 SFHF 增强，3 个 SFHF Block 串行 | `(B, 384, 20, 20)` | `(B, 384, 20, 20)` | /32 |

### 任意输入尺寸的通用表达

若输入为 `(B, 3, H, W)`，且 `H/W` 可被 32 整除，则：

| 输出 | shape | 分辨率 |
|---|---|---:|
| P1 | `(B, 64, H/2, W/2)` | /2 |
| P2 | `(B, 128, H/4, W/4)` | /4 |
| P3 | `(B, 256, H/8, W/8)` | /8 |
| P4 | `(B, 384, H/16, W/16)` | /16 |
| P5 | `(B, 384, H/32, W/32)` | /32 |

注意：`x0` 与 Fourier Unit 输出相加时都是 `2D` 通道；随后 `conv_fina` 将其压缩回 `D` 通道，作为 global 分支输出。

- **B 不变**：骨干网络全程保持 batch 维不变。
- **H/W 只在 `stride=2` 的 `Conv` 中减半**。
- **`CSFH_Block` 本身不下采样**：它增强特征，但保持空间尺寸不变。
- SFHF 中的 Fourier 变换会临时进入频域，宽度变为 `W/2+1`，但逆 FFT 后恢复为 `H×W`。

---

## 3. CSFH_Block：C2f 外壳的具体作用

`CSFH_Block` 继承自 `C2f`，但把 `C2f` 中的普通 `Bottleneck` 替换为 `SFHF_Block`。

### 3.1 数据流

设输入为：

```text
x: (B, C, H, W)
```

`C2f` 的隐藏通道为：

```text
D = C / 2      # e = 0.5
```

前向流程为：

```text
x
│
├─ cv1: 1×1 Conv
│   C → 2D
│
├─ chunk(2)
│   ├─ y0: (B, D, H, W)   # 直连分支，不经过 SFHF
│   └─ y1: (B, D, H, W)   # 增强分支
│
├─ SFHF_Block × n
│   y1 → SFHF_1(y1) → SFHF_2(...) → ...
│
├─ concat
│   [y0, SFHF_1(y1), ..., SFHF_n(...)]
│
└─ cv2: 1×1 Conv
    输出通道恢复为 C
```

公式化：

```text
z = cv1(x)                    # (B, 2D, H, W)
y0, y1 = chunk(z, 2)          # 各 (B, D, H, W)
y_i = SFHF_i(y_{i-1})         # i = 1..n
out = cv2(cat(y0, y1, ..., y_n))
```

### 3.2 Shape 变化

以一个输入输出通道都为 `C`、重复次数为 `n` 的 `CSFH_Block` 为例：

| 步骤 | 操作 | 输出 shape | B/H/W 是否变化 |
|---|---|---|---|
| 输入 | — | `(B, C, H, W)` | — |
| `cv1` | 1×1 Conv，`C→2D`，即 `C→C` | `(B, C, H, W)` | 不变 |
| `chunk` | 分成两半 | `y0=(B,D,H,W)`, `y1=(B,D,H,W)` | 不变 |
| `SFHF_Block` | 每个块输入输出均为 `D` | `(B, D, H, W)` | 不变 |
| `concat` | 直连分支 + n 个 SFHF 输出 | `(B, (n+2)D, H, W)` | 不变 |
| `cv2` | 1×1 Conv，恢复输出通道 | `(B, C, H, W)` | 不变 |

### 3.3 各阶段的内部通道

| 骨干位置 | 外层输出 `C` | 隐藏通道 `D=C/2` | SFHF 数量 `n` | concat 通道数 |
|---|---:|---:|---:|---:|
| P2 `/4` | 128 | 64 | 1 | 192 |
| P3 `/8` | 256 | 128 | 1 | 384 |
| P4 `/16` | 384 | 192 | 1 | 576 |
| P5 `/32` | 384 | 192 | 3 | 960 |

因此：

- P5 的 `CSFH_Block(384, n=3)` 内部会串行执行 **3 个 SFHF_Block**；
- 最后将 `5 × 192 = 960` 通道 concat 后，经 1×1 Conv 压回 384 通道。

---

## 4. SFHF_Block：核心增强单元

### 4.1 作用

`SFHF_Block` 是 CSFH 的核心。它包含两个残差子阶段：

1. **Token Mixing 阶段**
   - `norm1`
   - `SFHF_Mixer`
   - 残差连接，系数 `β`

2. **FFN 阶段**
   - `norm2`
   - `SFHF_FFN`
   - 残差连接，系数 `γ`

前向逻辑：

```text
identity = x

x_mixer = SFHF_Mixer(norm1(x))
x = x_mixer * β + identity

identity = x
x_ffn = SFHF_FFN(norm2(x))
x = x_ffn * γ + identity
```

### 4.2 Shape 变化

设 SFHF_Block 的输入通道为 `D`：

| 步骤 | 输出 shape | B/H/W 是否变化 |
|---|---|---|
| 输入 | `(B, D, H, W)` | — |
| `norm1` | `(B, D, H, W)` | 不变 |
| `SFHF_Mixer` | `(B, D, H, W)` | 不变 |
| `β` 残差 | `(B, D, H, W)` | 不变 |
| `norm2` | `(B, D, H, W)` | 不变 |
| `SFHF_FFN` | `(B, D, H, W)` | 不变 |
| `γ` 残差 | `(B, D, H, W)` | 不变 |
| 输出 | `(B, D, H, W)` | 不变 |

`β` 和 `γ` 的 shape 是：

```text
(1, D, 1, 1)
```

它们分别对 mixer 和 FFN 的输出做逐通道缩放。初始值为 0，因此初始状态下：

```text
SFHF_Block(x) ≈ x
```

这有利于避免新加模块在训练开始时破坏主干特征。

---

## 5. SFHF_Mixer：局部分支 + 全局分支 + 通道注意力

### 5.1 数据流

设输入为：

```text
x: (B, D, H, W)
```

流程：

```text
x
│
├─ 1×1 Conv: D → 2D
│
├─ split
│   ├─ local 分支:  D 通道
│   │   └─ TokenMixer_For_Local
│   │       - D/2 通道用 3×3 DWConv，dilation=1
│   │       - D/2 通道用 3×3 DWConv，dilation=2
│   │       - concat 回 D 通道
│   │
│   └─ global 分支: D 通道
│       └─ TokenMixer_For_Gloal
│           - 1×1 Conv 扩展到 2D
│           - SFHF_FourierUnit 频域全局建模
│           - 1×1 Conv 压回 D
│
├─ concat local + global
│   2D 通道
│
├─ GELU
│
├─ Channel Attention
│   全局平均池化 → MLP → Sigmoid → 通道门控
│
└─ 1×1 Conv
    2D → D
```

### 5.2 作用分解

| 子模块 | 具体作用 |
|---|---|
| `conv_init` | 先把通道从 `D` 扩展到 `2D`，为局部/全局双分支提供足够容量 |
| `TokenMixer_For_Local` | 用深度卷积捕获局部空间结构，膨胀率 1 和 2 提供两种局部感受野 |
| `TokenMixer_For_Gloal` | 通过 Fourier Unit 在频域建模长距离依赖/全局结构 |
| `cat(local, global)` | 将局部细节和全局上下文并列融合 |
| `GELU` | 非线性变换 |
| `ca` | 通道注意力，自适应增强重要通道、抑制次要通道 |
| `ca_conv` | 将融合后的 `2D` 通道压缩回 `D` 通道 |

### 5.3 Shape 变化

| 步骤 | 输出 shape | B/H/W 是否变化 |
|---|---|---|
| 输入 | `(B, D, H, W)` | — |
| `conv_init` | `(B, 2D, H, W)` | 不变 |
| split 后 local | `(B, D, H, W)` | 不变 |
| split 后 global | `(B, D, H, W)` | 不变 |
| local mixer 输出 | `(B, D, H, W)` | 不变 |
| global mixer 输出 | `(B, D, H, W)` | 不变 |
| concat | `(B, 2D, H, W)` | 不变 |
| GELU | `(B, 2D, H, W)` | 不变 |
| 通道注意力门控 | `(B, 2D, H, W)` | 不变 |
| `ca_conv` | `(B, D, H, W)` | 不变 |

---

## 6. TokenMixer_For_Local：局部多感受野分支

### 6.1 数据流

输入：

```text
x: (B, D, H, W)
```

流程：

```text
x
├─ chunk(2)
│   ├─ x1: (B, D/2, H, W)
│   └─ x2: (B, D/2, H, W)
│
├─ x1 → 3×3 depthwise Conv, dilation=1
├─ x2 → 3×3 depthwise Conv, dilation=2
│
└─ concat(cd1, cd2)
```

### 6.2 Shape 变化

| 步骤 | 输出 shape | B/H/W 是否变化 |
|---|---|---|
| 输入 | `(B, D, H, W)` | — |
| `chunk(2)` | 两个 `(B, D/2, H, W)` | 不变 |
| `CDilated_1` | `(B, D/2, H, W)` | 不变 |
| `CDilated_2` | `(B, D/2, H, W)` | 不变 |
| concat | `(B, D, H, W)` | 不变 |

### 6.3 设计意义

- depthwise convolution 参数量低；
- `dilation=1` 捕获较近邻局部结构；
- `dilation=2` 扩大感受野，捕获稍远处的局部上下文；
- 两半通道并行处理后 concat，兼顾效率和局部多尺度信息。

---

## 7. TokenMixer_For_Gloal：全局频域分支

> 源码类名为 `TokenMixer_For_Gloal`，从语义看应为 `Global` 的拼写问题，但不影响功能。

### 7.1 数据流

输入：

```text
x: (B, D, H, W)
```

流程：

```text
x
│
├─ conv_init: 1×1 Conv + GELU
│   D → 2D
│
├─ x0 = 当前特征
│
├─ SFHF_FourierUnit
│   频域全局建模，输出仍为 2D 通道
│
├─ conv_fina: 1×1 Conv + GELU
│   2D → D
│
└─ output = conv_fina(x + x0)
```

注意：`x0` 与 Fourier Unit 输出相加时都是 `2D` 通道；随后 `conv_fina` 将其压缩回 `D` 通道，作为 global 分支输出。

### 7.2 Shape 变化

| 步骤 | 输出 shape | B/H/W 是否变化 |
|---|---|---|
| 输入 | `(B, D, H, W)` | — |
| `conv_init` | `(B, 2D, H, W)` | 不变 |
| `x0` residual cache | `(B, 2D, H, W)` | 不变 |
| `SFHF_FourierUnit` | `(B, 2D, H, W)` | 最终不变 |
| `x + x0` | `(B, 2D, H, W)` | 不变 |
| `conv_fina` | `(B, D, H, W)` | 不变 |

### 7.3 设计意义

- 直接在空域做全局注意力通常开销较大；
- FFT 将空间信息转换到频域，天然具备全局感受野；
- Fourier Unit 可以捕获图像整体结构、周期性模式、长距离关系；
- 残差 `x + x0` 保留原始空间特征，避免频域变换完全替代原特征。

---

## 8. SFHF_FourierUnit：FFT 全局建模模块

### 8.1 数据流

该模块的输入来自 global mixer 的 `conv_init` 后特征。设输入为：

```text
x: (B, Q, H, W)
```

在 `TokenMixer_For_Gloal` 中：

```text
Q = 2D
```

流程：

```text
x
│
├─ rfft2
│   空间域 → 频域
│   (B, Q, H, W)
│   → complex (B, Q, H, Wf)
│   其中 Wf = W // 2 + 1
│
├─ real / imag 拼接
│   complex (B, Q, H, Wf)
│   → real tensor (B, 2Q, H, Wf)
│
├─ BatchNorm2d
│
├─ FPE: 3×3 depthwise Conv + residual
│
├─ 动态权重生成
│   weight: (B, groups=4, H, Wf)
│
├─ fdc: grouped 1×1 Conv
│
├─ einsum
│   按组加权求和
│
├─ GELU
│
├─ 重组为 complex
│   (B, Q, H, Wf)
│
└─ irfft2
│   频域 → 空间域
│   (B, Q, H, W)
```

### 8.2 Shape 变化

| 步骤 | shape | 说明 |
|---|---|---|
| 输入 | `(B, Q, H, W)` | 空间域特征 |
| `rfft2` | `(B, Q, H, Wf)` complex | `Wf = W//2 + 1` |
| real/imag 拼接 | `(B, 2Q, H, Wf)` | 实数张量 |
| BN | `(B, 2Q, H, Wf)` | 不变 |
| FPE | `(B, 2Q, H, Wf)` | residual，不变 |
| 动态权重 | `(B, 4, H, Wf)` | 每个空间频点有 4 组权重 |
| grouped conv 输出 | `(B, 8Q, H, Wf)` | 4 组，每组 `2Q` 通道 |
| einsum 加权求和 | `(B, 2Q, H, Wf)` | 组维度被加权融合 |
| 重组 complex | `(B, Q, H, Wf)` complex | real/imag 两路合并 |
| `irfft2` | `(B, Q, H, W)` | 恢复空间宽度和输入分辨率 |

### 8.3 设计意义

| 子模块 | 作用 |
|---|---|
| `rfft2` | 把特征从空间域转换到频域，获得全局视野 |
| real/imag 拼接 | 将复数频谱拆成实部和虚部，便于普通卷积处理 |
| `bn` | 稳定频域特征分布 |
| `fpe` | 频域局部位置增强，带残差 |
| `weight` | 根据频域内容动态生成组权重 |
| `fdc` | 分组频域通道变换 |
| `einsum` | 动态选择/融合不同频域变换组 |
| `irfft2` | 回到空间域，保持输入 shape |

---

## 9. Channel Attention：通道注意力

在 `SFHF_Mixer` 中，local 与 global 分支 concat 后得到：

```text
x: (B, 2D, H, W)
```

通道注意力流程：

```text
x
├─ AdaptiveAvgPool2d(1)
│   (B, 2D, H, W) → (B, 2D, 1, 1)
│
├─ Conv 1×1 + ReLU
│   2D → D
│
├─ Conv 1×1 + Sigmoid
│   D → 2D
│
└─ gate = attention(x)
└─ out = gate * x
```

### Shape 变化

| 步骤 | shape |
|---|---|
| 输入 | `(B, 2D, H, W)` |
| 全局平均池化 | `(B, 2D, 1, 1)` |
| 降维 | `(B, D, 1, 1)` |
| 升维 + Sigmoid | `(B, 2D, 1, 1)` |
| 广播相乘 | `(B, 2D, H, W)` |
| `ca_conv` 后 | `(B, D, H, W)` |

### 作用

- 每个通道得到一个 0~1 的重要性权重；
- 重要通道被放大，冗余或弱相关通道被抑制；
- 通道注意力只改变特征幅值，不改变 `B/H/W`，通道数最终由 `ca_conv` 压回 `D`。

---

## 10. SFHF_FFN：多尺度前馈网络

### 10.1 数据流

输入：

```text
x: (B, D, H, W)
```

流程：

```text
x
│
├─ conv_init: 1×1 Conv
│   D → 2D
│
├─ split 成 4 份
│   每份 D/2 通道
│
├─ branch 0: identity
├─ branch 1: 3×3 depthwise Conv
├─ branch 2: 5×5 depthwise Conv
├─ branch 3: 7×7 depthwise Conv
│
├─ concat
│   4 × D/2 = 2D
│
├─ GELU
│
└─ conv_fina: 1×1 Conv
    2D → D
```

### 10.2 Shape 变化

| 步骤 | 输出 shape | B/H/W 是否变化 |
|---|---|---|
| 输入 | `(B, D, H, W)` | — |
| `conv_init` | `(B, 2D, H, W)` | 不变 |
| split 后每个分支 | `(B, D/2, H, W)` | 不变 |
| identity / DWConv 分支 | `(B, D/2, H, W)` | 不变 |
| concat | `(B, 2D, H, W)` | 不变 |
| GELU | `(B, 2D, H, W)` | 不变 |
| `conv_fina` | `(B, D, H, W)` | 不变 |

### 10.3 设计意义

- 0 号分支保留原始信息；
- 3×3、5×5、7×7 depthwise 卷积提供不同感受野；
- 1×1 卷积负责通道扩张、分支混合和压缩；
- 相比普通 MLP，它能增强跨空间尺度的特征变换能力；
- 全部使用 stride=1、same padding，因此不改变 H/W。

---

## 11. CSFH 各阶段的数据模型/语义变化

| 阶段 | 分辨率 | 通道 | 语义作用 |
|---|---:|---:|---|
| Stem / P1 | /2 | 64 | 低级边缘、颜色、纹理 |
| P2 + CSFH | /4 | 128 | 细粒度局部纹理 + 初步全局结构信息 |
| P3 + CSFH | /8 | 256 | 中低层目标部件、局部形状、中等尺度上下文 |
| P4 + CSFH | /16 | 384 | 中高层语义，局部细节与全局频域信息进一步融合 |
| P5 + CSFH ×3 | /32 | 384 | 高层语义和全局上下文，连续 3 次 SFHF 增强，适合大目标和全局关系建模 |

数据抽象过程可概括为：

```text
RGB 像素
   ↓
低级纹理 / 边缘
   ↓
局部形状 / 部件
   ↓
跨尺度语义特征
   ↓
全局上下文 + 高层类别/目标语义
```

CSFH 的贡献在于：不是单纯堆叠卷积，而是在每个尺度上同时引入：

```text
局部空域细节 + 频域全局关系 + 通道选择 + 多尺度 FFN
```

因此它能缓解传统 CNN backbone 全局建模能力不足的问题。

---

## 12. 骨干输出与检测头的数据流衔接

当前 YAML 中，检测头主要使用骨干的三个输出：

| 来源层 | 分辨率 | 通道 | 在 head 中的角色 |
|---:|---:|---:|---|
| layer 4 | /8 | 256 | P3，小目标特征 |
| layer 6 | /16 | 384 | P4，中目标特征 |
| layer 8 | /32 | 384 | P5，大目标/全局语义特征 |

后续流程为：

```text
P5
 ├─ 1×1 Conv → 256
 ├─ AIFI 全局交互
 ├─ 上采样到 P4 尺寸
 ├─ 与 P4 投影特征 concat
 ├─ RepC3 融合
 │
 ├─ 上采样到 P3 尺寸
 ├─ 与 P3 投影特征 concat
 ├─ RepC3 融合 → X3
 │
 ├─ 下采样回 P4 尺寸
 ├─ 与 Y4 concat
 ├─ RepC3 融合 → F4
 │
 ├─ 下采样回 P5 尺寸
 ├─ 与 Y5 concat
 ├─ RepC3 融合 → F5
 │
 └─ X3/F4/F5 → RTDETRDecoder
```

以 640×640 输入为例，head 中主要 B/C/H/W 变化如下：

| 层号 | 模块 | 输出 shape | 说明 |
|---:|---|---|---|
| 9 | `Conv 1×1` | `(B, 256, 20, 20)` | P5 投影 |
| 10 | `AIFI` | `(B, 256, 20, 20)` | 内部临时变为 `(B, 400, 256)` token 序列 |
| 11 | `Conv 1×1` | `(B, 256, 20, 20)` | Y5 |
| 12 | `Upsample ×2` | `(B, 256, 40, 40)` | P5 → P4 尺寸 |
| 13 | `Conv 1×1` 来自 layer 6 | `(B, 256, 40, 40)` | P4 投影 |
| 14 | `Concat` | `(B, 512, 40, 40)` | Y5 + P4 |
| 15 | `RepC3` | `(B, 256, 40, 40)` | 融合 |
| 16 | `Conv 1×1` | `(B, 256, 40, 40)` | Y4 |
| 17 | `Upsample ×2` | `(B, 256, 80, 80)` | P4 → P3 尺寸 |
| 18 | `Conv 1×1` 来自 layer 4 | `(B, 256, 80, 80)` | P3 投影 |
| 19 | `Concat` | `(B, 512, 80, 80)` | Y4 + P3 |
| 20 | `RepC3` | `(B, 256, 80, 80)` | X3 |
| 21 | `Conv k=3,s=2` | `(B, 256, 40, 40)` | 下采样 |
| 22 | `Concat` | `(B, 512, 40, 40)` | 与 Y4 融合 |
| 23 | `RepC3` | `(B, 256, 40, 40)` | F4 |
| 24 | `Conv k=3,s=2` | `(B, 256, 20, 20)` | 下采样 |
| 25 | `Concat` | `(B, 512, 20, 20)` | 与 Y5 融合 |
| 26 | `RepC3` | `(B, 256, 20, 20)` | F5 |
| 27 | `RTDETRDecoder` | 推理输出 `(B, 300, 4+nc)` | `nc=10`，即 `(B, 300, 14)` |

推理时，`RTDETRDecoder` 输出：

```text
scores/bboxes: (B, 300, 4 + nc)
```

当前配置：

```text
nc = 10
num_queries = 300
```

所以最终预测张量为：

```text
(B, 300, 14)
```

其中每个 query 包含：

- 4 个边界框坐标值；
- 10 个类别分数。

---

## 13. 关键点总结

1. **CSFH_Block 不改变空间分辨率**
   - 输入 `(B, C, H, W)`；
   - 输出仍为 `(B, C, H, W)`；
   - 空间下采样只由前面的 stride=2 Conv 完成。

2. **CSFH_Block 内部通道先拆分、再增强、再融合**
   - `C → C`；
   - chunk 成两个 `D=C/2`；
   - 一路直连，一路进入 SFHF；
   - concat 后通过 1×1 Conv 压回 `C`。

3. **SFHF_Block 是局部 + 全局 + 通道 + 多尺度 FFN 的复合模块**
   - local：膨胀深度卷积；
   - global：FFT/Fourier Unit；
   - channel：Sigmoid 通道门控；
   - FFN：identity + 3×3 + 5×5 + 7×7。

4. **频域变换是临时的**
   - `rfft2` 后宽度变为 `W//2+1`；
   - `irfft2` 后恢复 `W`；
   - 因此模块最终保持输入 `H/W` 不变。

5. **P5 使用 3 个 SFHF_Block 串行**
   - 最高语义层获得更强的全局建模和特征重校准；
   - 对大目标、重叠目标和全局上下文理解更有利。

6. **骨干输出 P3/P4/P5**
   - P3: `(B, 256, H/8, W/8)`
   - P4: `(B, 384, H/16, W/16)`
   - P5: `(B, 384, H/32, W/32)`

7. **整体数据形态演变**
   ```text
   图像 (B,3,H,W)
     → 多尺度特征金字塔
     → 融合后的 X3/F4/F5
     → Transformer Decoder 查询序列
     → (B,300,4+nc) 检测结果
   ```



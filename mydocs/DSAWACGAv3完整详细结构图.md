# DSAWACGAv3 完整详细结构图

## 一、Backbone 整体结构（YAML 配置）

```
输入图像: [B, 3, H, W]
│
├─ Layer0: Conv(3→64, k=3, s=2)       # 0-P1/2      → [B, 64, H/2, W/2]
├─ Layer1: Conv(64→128, k=3, s=2)     # 1-P2/4      → [B, 128, H/4, W/4]
├─ Layer2: C2f_DSAWACGA(128)          # 2            → [B, 128, H/4, W/4]
│
├─ Layer3: Conv(128→256, k=3, s=2)    # 3-P3/8      → [B, 256, H/8, W/8]
├─ Layer4: C2f_DSAWACGA(256)          # 4            → [B, 256, H/8, W/8]
│
├─ Layer5: Conv(256→384, k=3, s=2)    # 5-P4/16     → [B, 384, H/16, W/16]
├─ Layer6: C2f_DSAWACGA(384)          # 6            → [B, 384, H/16, W/16]
│
├─ Layer7: Conv(384→384, k=3, s=2)    # 7-P5/32     → [B, 384, H/32, W/32]
└─ Layer8: C2f_DSAWACGA(384) ×3       # 8            → [B, 384, H/32, W/32]
```

## 二、Head 结构（HybridEncoder + Decoder）

```
Layer8 输出: [B, 384, H/32, W/32]  ← P5
│
├─ Layer9:  Conv(384→256, k=1, bias=False)   # input_proj.2  → [B, 256, H/32, W/32]
├─ Layer10: AIFI(256, dim_ff=1024, nhead=8)                    → [B, 256, H/32, W/32]
├─ Layer11: Conv(256→256, k=1, silu)         # Y5 lateral     → [B, 256, H/32, W/32]
│
│ ──── FPN ────
├─ Layer12: Upsample(nearest, ×2)                             → [B, 256, H/16, W/16]
├─ Layer13: Conv(Layer6→256, k=1, bias=False) # input_proj.1  → [B, 256, H/16, W/16]
├─ Layer14: Concat([Layer12, Layer13])                         → [B, 512, H/16, W/16]
├─ Layer15: RepC3(512→256, n=3)              # fpn_blocks.0   → [B, 256, H/16, W/16]
├─ Layer16: Conv(256→256, k=1, silu)         # Y4 lateral     → [B, 256, H/16, W/16]
│
├─ Layer17: Upsample(nearest, ×2)                             → [B, 256, H/8, W/8]
├─ Layer18: Conv(Layer4→256, k=1, bias=False) # input_proj.0  → [B, 256, H/8, W/8]
├─ Layer19: Concat([Layer17, Layer18])                         → [B, 512, H/8, W/8]
├─ Layer20: RepC3(512→256, n=3)              # fpn_blocks.1   → [B, 256, H/8, W/8]  ← X3
│
│ ──── PAN ────
├─ Layer21: Conv(256→256, k=3, s=2, silu)   # downsample.0   → [B, 256, H/16, W/16]
├─ Layer22: Concat([Layer21, Layer16])                         → [B, 512, H/16, W/16]
├─ Layer23: RepC3(512→256, n=3)              # pan_blocks.0   → [B, 256, H/16, W/16] ← F4
│
├─ Layer24: Conv(256→256, k=3, s=2, silu)   # downsample.1   → [B, 256, H/32, W/32]
├─ Layer25: Concat([Layer24, Layer11])                         → [B, 512, H/32, W/32]
├─ Layer26: RepC3(512→256, n=3)              # pan_blocks.1   → [B, 256, H/32, W/32] ← F5
│
└─ RTDETRDecoder([Layer20, Layer23, Layer26], nc=10, hidden=256, num_queries=300)
   输入: X3[B,256,H/8,W/8], F4[B,256,H/16,W/16], F5[B,256,H/32,W/32]
```

## 三、C2f_DSAWACGA（C2f 容器）

```
输入 x: [B, C_in, H, W]
│
├─ cv1: Conv1x1(C_in → 2C)              → [B, 2C, H, W]       C = C_out × e, e=0.5
├─ split(2 chunks)                       → x1: [B, C, H, W], x2: [B, C, H, W]
│
├─ x1 ─→ DSAWACGAv2Block(C) ─→ x1'     （若 n>1 则串联 n 个）
├─ x2 ─→ DSAWACGAv2Block(C) ─→ x2'
│   ...
│
├─ cat([x1', x2', ...])                 → [B, (n+1)×C, H, W]
└─ cv2: Conv1x1((n+1)×C → C_out)        → [B, C_out, H, W]

作用: C2f 的 CSP 梯度流结构，split 后各分支经 DSAWACGAv2Block 独立变换，cat 后压缩
```

## 四、DSAWACGAv2Block（Transformer 风格双阶段块）

```
输入 x: [B, dim, H, W]
│
│ ══════════ Stage 1: Attention ══════════
├─ norm1: BatchNorm2d(dim)               → [B, dim, H, W]
├─ DSAWACGA_Mixer(dim)                   → [B, dim, H, W]     ← 核心混合器
├─ × β (LayerScale, 初始化=0)            → [B, dim, H, W]
├─ + x (残差)                             → [B, dim, H, W]
│
│ ══════════ Stage 2: FFN ══════════
├─ norm2: BatchNorm2d(dim)               → [B, dim, H, W]
├─ DSAWACGA_FFN(dim)                     → [B, dim, H, W]     ← 多尺度前馈网络
├─ × γ (LayerScale, 初始化=0)            → [B, dim, H, W]
└─ + x (残差)                             → [B, dim, H, W]     ← 最终输出

作用: 模仿 Transformer 的 Attention+FFN 两阶段结构
      LayerScale(β/γ) 初始化为 0，训练初期为恒等映射，保证稳定性
```

## 五、DSAWACGA_Mixer（局部 DSA + 全局 WaveletGlobal 混合器）

```
输入 x: [B, dim, H, W]
│
├─ conv_init: Conv1x1(dim → 2dim)        → [B, 2dim, H, W]
├─ chunk(2, dim=1)
│   ├─ x_local:  [B, dim, H, W]
│   └─ x_global: [B, dim, H, W]
│
├─ x_local  → DSA(dim)                   → [B, dim, H, W]     ← 局部多尺度空间混合
├─ x_global → TokenMixer_For_Global(dim) → [B, dim, H, W]     ← 全局小波频域混合
│
├─ cat([local_out, global_out])           → [B, 2dim, H, W]
├─ GELU                                   → [B, 2dim, H, W]
│
│ ──── Channel Attention (SE) ────
├─ AdaptiveAvgPool2d(1)                   → [B, 2dim, 1, 1]
├─ Conv1x1(2dim → dim)                    → [B, dim, 1, 1]
├─ ReLU                                   → [B, dim, 1, 1]
├─ Conv1x1(dim → 2dim)                    → [B, 2dim, 1, 1]
├─ Sigmoid                                → [B, 2dim, 1, 1]    ∈ [0,1]
├─ × x (广播逐通道乘)                     → [B, 2dim, H, W]
│
└─ ca_conv: Conv1x1(2dim → dim)           → [B, dim, H, W]     ← 最终输出

作用: 局部(DSA) + 全局(WaveletGlobal) 双路混合，SE 通道注意力校准后压缩
```

## 六、DSA（Dynamic Scale Attention：6 分支多尺度局部混合）

```
输入 x: [B, dim, H, W]
│
├─ f0 = DWConv1x1(x,  groups=dim)        → [B, dim, H, W]     感受野 1×1
├─ f1 = DWConv3x3(x,  groups=dim)        → [B, dim, H, W]     感受野 3×3
├─ f2 = DWConv5x5(x,  groups=dim)        → [B, dim, H, W]     感受野 5×5
├─ f3 = DWConv7x7(x,  groups=dim)        → [B, dim, H, W]     感受野 7×7
├─ f4 = DWConv9x9(x,  groups=dim)        → [B, dim, H, W]     感受野 9×9
├─ f5 = DWConv11x11(x,groups=dim)        → [B, dim, H, W]     感受野 11×11
│
├─ cat([f0,f1,f2,f3,f4,f5], dim=1)       → [B, dim×6, H, W]
├─ weight_conv: Conv1x1(dim×6 → 6)       → [B, 6, H, W]
├─ Softmax(dim=1)                         → [B, 6, H, W]       6 个空间自适应权重
│
├─ 加权求和:
│   out = w0*f0 + w1*f1 + w2*f2 + w3*f3 + w4*f4 + w5*f5
│                                          → [B, dim, H, W]
│
└─ channel_mix: Conv1x1(dim → dim)        → [B, dim, H, W]     ← 最终输出

作用: 6 分支多尺度 DWConv 覆盖 1~11 感受野
      softmax 动态加权实现输入自适应的尺度选择
      channel_mix 做跨通道信息交换
```

## 七、TokenMixer_For_Global（全局分支：通道扩展 + WaveletGlobal + 残差）

```
输入 x: [B, dim, H, W]
│
├─ conv_init: Conv1x1(dim → 2dim) + GELU → [B, 2dim, H, W]     通道扩展
├─ x0 = x (保存残差)                       [B, 2dim, H, W]
│
├─ WaveletGlobal(2dim)                     → [B, 2dim, H, W]     ← 小波全局建模
│
├─ + x0 (残差连接)                          → [B, 2dim, H, W]
└─ conv_fina: Conv1x1(2dim → dim) + GELU  → [B, dim, H, W]     通道压缩

作用: 通道扩展增加容量 → WaveletGlobal 做小波域全局建模 → 残差保留 → 压缩回原通道
```

## 八、WaveletGlobal（小波全局建模核心）

```
输入 x: [B, dim, H, W]       （dim 实际为 2×原始dim，由 TokenMixer_For_Global 传入）
│
│ ══════════ 阶段1: DWT 分解 + 子带处理 ══════════
│
├─ LearnableHaarDWT(level=1)
│   ├─ ya (LL): [B, dim, H/2, W/2]    低频近似（全局结构）
│   ├─ yh (LH): [B, dim, H/2, W/2]    水平边缘
│   ├─ yv (HL): [B, dim, H/2, W/2]    垂直边缘
│   └─ yd (HH): [B, dim, H/2, W/2]    对角纹理
│
├─ LL 子带处理:
│   ├─ ya_proj: Conv1x1(dim → dim/4)       → [B, dim/4, H/2, W/2]
│   └─ ll_conv: DWConv3x3(dim/4, groups=dim/4) → [B, dim/4, H/2, W/2]
│
├─ LH 子带处理:
│   ├─ yh_conv: Conv(1,3)(dim → dim/4, groups=dim/4) → [B, dim/4, H/2, W/2]
│   └─ horizontal_conv: 固定Sobel水平核(DWConv3x3, 不可学习)  → [B, dim/4, H/2, W/2]
│
├─ HL 子带处理:
│   ├─ yv_conv: Conv(3,1)(dim → dim/4, groups=dim/4) → [B, dim/4, H/2, W/2]
│   └─ vertical_conv: 固定Sobel垂直核(DWConv3x3, 不可学习)    → [B, dim/4, H/2, W/2]
│
├─ HH 子带处理:
│   ├─ yd_act: Tanh()                                 → [B, dim, H/2, W/2]
│   ├─ yd_proj: Conv1x1(dim → dim/4)                  → [B, dim/4, H/2, W/2]
│   └─ diagonal_conv: 固定Laplacian核(DWConv3x3, 不可学习)    → [B, dim/4, H/2, W/2]
│
├─ cat([LL_proc, LH_proc, HL_proc, HH_proc], dim=1)   → [B, dim, H/2, W/2]
├─ subband_fusion: DWConv3x3(dim, groups=dim)          → [B, dim, H/2, W/2]
│
│ ══════════ 阶段2: 动态分组加权 ══════════
│
├─ BatchNorm2d(dim)                                    → [B, dim, H/2, W/2]
├─ FPE: DWConv3x3(dim, groups=dim) + 残差              → [B, dim, H/2, W/2]   位置增强
│
├─ weight: Conv1x1(dim → groups) + Softmax             → [B, groups, H/2, W/2] 动态组权重
├─ fdc: Conv1x1(dim → dim×groups, groups=G)            → [B, dim×G, H/2, W/2]
│      .view                                            → [B, G, dim, H/2, W/2]
├─ einsum(fdc, weight)                                 → [B, dim, H/2, W/2]    动态加权融合
├─ GELU                                                → [B, dim, H/2, W/2]
│
│ ══════════ 阶段3: 近似重建 ══════════
│
└─ F.interpolate(bilinear, size=(H, W))                → [B, dim, H, W]        ← 最终输出

作用: DWT 将特征分解为 4 个方向子带(LL/LH/HL/HH)，分别处理后融合
      动态分组加权实现输入自适应的频域变换选择
      bilinear 插值近似重建（非精确 IDWT）
```

### WaveletGlobal 子带处理详细参数

| 子带 | DWT 输出 | 第1步处理 | 第2步处理（固定核） | 输出 |
|------|----------|-----------|---------------------|------|
| LL (ya) | [B,dim,H/2,W/2] | ya_proj: Conv1x1→[B,dim/4,H/2,W/2] | ll_conv: DWConv3×3(可学习) | [B,dim/4,H/2,W/2] |
| LH (yh) | [B,dim,H/2,W/2] | yh_conv: Conv(1,3)→[B,dim/4,H/2,W/2] | horizontal_conv: Sobel水平(固定) | [B,dim/4,H/2,W/2] |
| HL (yv) | [B,dim,H/2,W/2] | yv_conv: Conv(3,1)→[B,dim/4,H/2,W/2] | vertical_conv: Sobel垂直(固定) | [B,dim/4,H/2,W/2] |
| HH (yd) | [B,dim,H/2,W/2] | Tanh→yd_proj: Conv1x1→[B,dim/4,H/2,W/2] | diagonal_conv: Laplacian(固定) | [B,dim/4,H/2,W/2] |

### 固定卷积核定义

| 名称 | 核值 | 作用 |
|------|------|------|
| horizontal (Sobel水平) | [[1,1,1],[0,0,0],[-1,-1,-1]] | 检测水平边缘 |
| vertical (Sobel垂直) | [[1,0,-1],[1,0,-1],[1,0,-1]] | 检测垂直边缘 |
| diagonal (Laplacian) | [[0,1,0],[1,-4,1],[0,1,0]] | 检测对角纹理 |

## 九、DSAWACGA_FFN（多尺度前馈网络）

```
输入 x: [B, dim, H, W]
│
├─ conv_init: Conv1x1(dim → 2dim)        → [B, 2dim, H, W]     通道扩展
├─ split(4 chunks, 各 dim/2)
│   ├─ x[0]: [B, dim/2, H, W]   ← identity（不处理，保留原始信息，内部跳跃连接）
│   ├─ x[1]: [B, dim/2, H, W]   → dw1: DWConv3×3(groups=dim/2) → [B, dim/2, H, W]
│   ├─ x[2]: [B, dim/2, H, W]   → dw2: DWConv5×5(groups=dim/2) → [B, dim/2, H, W]
│   └─ x[3]: [B, dim/2, H, W]   → dw3: DWConv7×7(groups=dim/2) → [B, dim/2, H, W]
│
├─ cat([x[0], x[1]', x[2]', x[3]'])      → [B, 2dim, H, W]
├─ GELU                                   → [B, 2dim, H, W]
└─ conv_fina: Conv1x1(2dim → dim)         → [B, dim, H, W]      ← 最终输出

作用: 多尺度 DWConv(3/5/7) + identity 跳跃连接
      x[0] 不处理保留扩展后原始特征，x[1~3] 分别用不同核大小提取多尺度局部特征
```

## 十、Channel Attention（SE 通道注意力，位于 DSAWACGA_Mixer 内）

```
输入 x: [B, 2dim, H, W]        （cat 后的局部+全局特征）
│
├─ AdaptiveAvgPool2d(1)                  → [B, 2dim, 1, 1]      全局平均池化
├─ Conv1x1(2dim → dim)                   → [B, dim, 1, 1]       通道压缩（瓶颈）
├─ ReLU                                  → [B, dim, 1, 1]       非线性
├─ Conv1x1(dim → 2dim)                   → [B, 2dim, 1, 1]      通道扩展
├─ Sigmoid                               → [B, 2dim, 1, 1]      ∈ [0,1] 通道权重
│
├─ × x (广播逐通道乘)                     → [B, 2dim, H, W]      逐通道重标定
└─ ca_conv: Conv1x1(2dim → dim)          → [B, dim, H, W]       通道降维

作用: SE 通道注意力：学习通道间依赖关系，重要通道增强，不重要通道抑制
      压缩比 r = 2（2dim → dim → 2dim）
```

## 十一、完整数据流总图（以 dim=128 为例）

```
输入: [B, 128, H, W]
│
╔══════════════════ DSAWACGAv2Block ═════════════════╗
║                                                      ║
║  ┌───── Stage 1: Attention ─────────────────────┐   ║
║  │  x: [B,128,H,W]                              │   ║
║  │  → BN → [B,128,H,W]                          │   ║
║  │  → DSAWACGA_Mixer:                           │   ║
║  │    conv_init: [B,128,H,W]→[B,256,H,W]        │   ║
║  │    chunk(2) → local[B,128,H,W], global 同    │   ║
║  │                                               │   ║
║  │    ┌─ DSA (局部): ─────────────────────────┐  │   ║
║  │    │  f0=DW1x1(x)   [B,128,H,W]            │  │   ║
║  │    │  f1=DW3x3(x)   [B,128,H,W]            │  │   ║
║  │    │  f2=DW5x5(x)   [B,128,H,W]            │  │   ║
║  │    │  f3=DW7x7(x)   [B,128,H,W]            │  │   ║
║  │    │  f4=DW9x9(x)   [B,128,H,W]            │  │   ║
║  │    │  f5=DW11x11(x) [B,128,H,W]            │  │   ║
║  │    │  cat→[B,768,H,W]→Conv1x1→Softmax       │  │   ║
║  │    │    → [B,6,H,W] (6个空间权重)            │  │   ║
║  │    │  加权求和 → [B,128,H,W]                 │  │   ║
║  │    │  channel_mix: Conv1x1 → [B,128,H,W]    │  │   ║
║  │    └────────────────────────────────────────┘  │   ║
║  │                                               │   ║
║  │    ┌─ TokenMixer_For_Global (全局): ────────┐  │   ║
║  │    │  conv_init: [B,128]→[B,256]+GELU       │  │   ║
║  │    │  x0 = x (残差) [B,256,H,W]             │  │   ║
║  │    │                                          │  │   ║
║  │    │  ┌─ WaveletGlobal(256): ─────────────┐  │  │   ║
║  │    │  │  DWT → LL[L,256,H/2,W/2]          │  │  │   ║
║  │    │  │        LH[L,256,H/2,W/2]          │  │  │   ║
║  │    │  │        HL[L,256,H/2,W/2]          │  │  │   ║
║  │    │  │        HH[L,256,H/2,W/2]          │  │  │   ║
║  │    │  │  LL→Conv1x1→DW3x3 → [L,64,H/2,W/2]│  │  │   ║
║  │    │  │  LH→Conv(1,3)→Sobel_H→[L,64,H/2,W/2]│  │  │   ║
║  │    │  │  HL→Conv(3,1)→Sobel_V→[L,64,H/2,W/2]│  │  │   ║
║  │    │  │  HH→Tanh→Conv1x1→Lapl→[L,64,H/2,W/2]│  │  │   ║
║  │    │  │  cat→[L,256,H/2,W/2]                │  │  │   ║
║  │    │  │  subband_fusion: DW3x3→[L,256,H/2,W/2]│  │  │   ║
║  │    │  │  BN→FPE+残差→[L,256,H/2,W/2]       │  │  │   ║
║  │    │  │  weight: Conv1x1→Softmax→[L,G,H/2,W/2]│  │  │   ║
║  │    │  │  fdc: Conv1x1→[L,256G,H/2,W/2]     │  │  │   ║
║  │    │  │    .view→[L,G,256,H/2,W/2]          │  │  │   ║
║  │    │  │  einsum→[L,256,H/2,W/2]→GELU        │  │  │   ║
║  │    │  │  interpolate(bilinear)→[L,256,H,W]  │  │  │   ║
║  │    │  └────────────────────────────────────┘  │  │   ║
║  │    │                                          │  │   ║
║  │    │  +x0(残差) → conv_fina: [B,256]→[B,128] │  │   ║
║  │    └──────────────────────────────────────────┘  │   ║
║  │                                               │   ║
║  │    cat([local,global]) → [B,256,H,W]          │   ║
║  │    → GELU                                     │   ║
║  │    → SE: Pool[B,256,1,1]→Conv→ReLU→Conv       │   ║
║  │      →Sigmoid→×x→[B,256,H,W]                  │   ║
║  │    → ca_conv: [B,256]→[B,128]                 │   ║
║  │                                               │   ║
║  │  → ×β + x (LayerScale残差)                    │   ║
║  └───────────────────────────────────────────────┘   ║
║                                                      ║
║  ┌───── Stage 2: FFN ──────────────────────────┐   ║
║  │  x: [B,128,H,W]                              │   ║
║  │  → BN → [B,128,H,W]                          │   ║
║  │  → DSAWACGA_FFN:                             │   ║
║  │    conv_init: [B,128]→[B,256]                 │   ║
║  │    split4 → 4×[B,64,H,W]                      │   ║
║  │      x[0]: identity (保留)                    │   ║
║  │      x[1]: DW3×3  → [B,64,H,W]               │   ║
║  │      x[2]: DW5×5  → [B,64,H,W]               │   ║
║  │      x[3]: DW7×7  → [B,64,H,W]               │   ║
║  │    cat → GELU → [B,256,H,W]                  │   ║
║  │    conv_fina: [B,256]→[B,128]                 │   ║
║  │                                               │   ║
║  │  → ×γ + x (LayerScale残差)                    │   ║
║  └───────────────────────────────────────────────┘   ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
│
输出: [B, 128, H, W]
```

## 十二、各模块功能总结表

| 模块 | 层级 | 核心作用 | 关键设计 | 与 CSFH 对应关系 |
|------|------|----------|----------|-------------------|
| **C2f_DSAWACGA** | 容器 | C2f CSP 梯度流 + DSAWACGAv2Block 串联 | split→多分支→cat→压缩 | = CSFH_Block（容器相同） |
| **DSAWACGAv2Block** | 块 | Transformer 风格双阶段：Attention + FFN | LayerScale(β/γ) 初始化为 0 | = SFHF_Block（结构相同） |
| **DSAWACGA_Mixer** | 混合器 | 局部 DSA + 全局 WaveletGlobal 融合 | split→双路→cat→SE→压缩 | ↔ SFHF_Mixer（局部/全局实现不同） |
| **DSA** | 局部分支 | 6 分支多尺度 DWConv + softmax 动态加权 | DWConv(1,3,5,7,9,11) + softmax + channel_mix | ≠ TokenMixer_For_Local（2-branch DilatedConv） |
| **TokenMixer_For_Global** | 全局分支 | 通道扩展 + 小波全局建模 + 残差 | dim→2dim→WaveletGlobal→残差→dim | ↔ TokenMixer_For_Gloal（FourierUnit→WaveletGlobal） |
| **WaveletGlobal** | 全局核心 | DWT 子带分解 + 动态分组加权 + 近似重建 | DWT→4子带处理→BN→FPE→softmax→grouped conv→einsum→GELU→bilinear | ≠ SFHF_FourierUnit（FFT→IFFT） |
| **DSAWACGA_FFN** | 前馈网络 | 多尺度非线性变换 | Split4→[id,DW3,DW5,DW7]→Cat→GELU | = SFHF_FFN（完全相同） |
| **SE (ca)** | 通道注意力 | 通道重要性重标定 | GAP→Conv→ReLU→Conv→Sigmoid→×x | = SFHF_Mixer.ca（完全相同） |
| **ca_conv** | 投影 | 通道降维回 dim | Conv1x1(2dim→dim) | = SFHF_Mixer.ca_conv（完全相同） |

## 十三、与 CSFH 的差异对照表

| 组件 | CSFH | DSAWACGAv3 | 是否相同 |
|------|------|------------|----------|
| C2f 容器 | C2f + SFHF_Block | C2f + DSAWACGAv2Block | ✅ 结构相同 |
| Block 双阶段 | SFHF_Block(Mixer+FFN+β/γ) | DSAWACGAv2Block(Mixer+FFN+β/γ) | ✅ 结构相同 |
| Mixer split+cat | SFHF_Mixer | DSAWACGA_Mixer | ✅ 框架相同 |
| **局部分支** | TokenMixer_For_Local: 2-branch DilatedDWConv(d=1,2) | **DSA: 6-branch DWConv(1,3,5,7,9,11) + softmax** | ❌ **不同** |
| **全局分支** | TokenMixer_For_Gloal → **FourierUnit(FFT)** | TokenMixer_For_Global → **WaveletGlobal(DWT)** | ❌ **不同** |
| **FFN** | SFHF_FFN: [id,DW3,DW5,DW7] | DSAWACGA_FFN: [id,DW3,DW5,DW7] | ✅ **完全相同** |
| **Channel Attention** | SE(2dim→dim→2dim) + ca_conv | SE(2dim→dim→2dim) + ca_conv | ✅ **完全相同** |
| LayerScale | β/γ zeros init | β/γ zeros init | ✅ 相同 |
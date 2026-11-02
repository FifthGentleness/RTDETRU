# 第三个创新点：三个独立实现

这是机制借鉴后的实验实现，尚无 VisDrone 完整训练精度。默认保留双创新点骨干、P2 融合、P3/P4/P5 解码输入及 300 query。所有路径相对仓库根目录。

## 配置与启动

配置目录：`ultralytics/cfg/models/rt-detr/`，文件均以 `rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7-` 开头。

| 方向 | 配置后缀 | 启动 scheme |
| --- | --- | --- |
| 标准 AIFI 门控 | AIFI-QHGate.yaml | qhgate |
| BlockSparse 门控对照 | AIFI-BlockSparseQHGate.yaml | sparse-qhgate |
| 解码器质量排序 | FQRank.yaml | fqrank |
| 期望质量 VFL | PEQVFL.yaml | peqvfl |

使用记录的基线 args.yaml（包含 AdamW、lr0=0.0001、lrf=1、350轮、batch=4、seed=0 等）启动，避免直接沿用 train.py 当前默认参数造成对照偏差：

```powershell
D:/Anaconda/envs/RTDETR/python.exe tools/train_third_innovations.py --scheme qhgate --device 0
D:/Anaconda/envs/RTDETR/python.exe tools/train_third_innovations.py --scheme fqrank --device 0
D:/Anaconda/envs/RTDETR/python.exe tools/train_third_innovations.py --scheme peqvfl --device 0
D:/Anaconda/envs/RTDETR/python.exe tools/train_third_innovations.py --scheme sparse-qhgate --device 0
```

加入 `--dry-run` 仅打印参数。`--epochs` / `--batch` 用于短程检查；正式实验保持基线设置。原 train.py 的 --resume 入口仍可用于从对应实验 last.pt 恢复。新脚本启动新实验，不自动加载其他实验权重。

## AIFI-QHGate

文件：`ultralytics/nn/extra_modules/aifi_qhgate.py`。
在逐头 SDPA 输出和输出投影之间乘以 `2*sigmoid(Linear(LN(content)))`。256通道/8头新增2056参数，LN无仿射参数。门控层零初始化，初始门值为1。保留标准AIFI参数名称、位置编码仅进入Q/K的约定，支持pre/post norm。

BlockSparse对照保持其原有位置编码/路由约定，门控使用未加位置编码的输入内容。包装projection时支持通过load_state_dict接收原BlockSparse的投影键。标准AIFI与BlockSparse基座不能混算增益；现有BlockSparse仍生成完整注意力矩阵，并非硬件稀疏加速。

来源：Gated Attention for Large Language Models，NeurIPS 2025。https://proceedings.neurips.cc/paper_files/paper/2025/hash/904e89bb4e632e75fb47f093b620b257-Abstract-Conference.html
2*sigmoid恒等起步是适配设计，不是原论文公式。

## FQRank

文件：`ultralytics/nn/extra_modules/decoder_fqrank.py`。
增加256→64→1质量头（16513参数）。使用 `log(max_class_probability) + alpha*log(sigmoid(quality))` 排序；无效锚点排除。最终层预测先去掉DN query，再独立按现有Hungarian代价匹配GT；匹配候选的质量目标为最终框与GT的IoU，未匹配候选目标为0。正负样本分开取均值，负项系数0.25，整体乘quality_weight。teacher框/分数停止梯度，默认质量头梯度可回到编码器特征。

训练前半程：270个排序候选+30个从有效非top-300候选中均匀无放回抽取的探索候选；后半程和推理使用完整top-300。探索候选参与正常检测监督。均匀探索是一种可检验选择，并不保证收益。

YAML顶层 `fqrank_params`：

- alpha_max: 0.5；ramp_fraction: 0.2（前20%训练线性升权）。
- explore_max: 0.1；explore_end: 0.5（探索比例与停止时点）。
- quality_weight: 0.1；teacher: final（可改encoder作消融）。
- detach_quality_features: false（改true可隔离共享特征梯度）。

关闭改动的对照：alpha_max=0、explore_max=0、quality_weight=0。只关闭排序仍保留质量训练，不等价于完整基线。
附加损失通过已有compute_ground_loss路由进入总loss；日志主三项不单列此项。推理不使用GT或匹配器，但仍需质量头计算。缓存消费后清除，以兼容EMA deepcopy。

来源：D-FINE，ICLR 2025。https://proceedings.iclr.cc/paper_files/paper/2025/hash/6cf58a87e3097e7d1f9be3e8693a93de-Abstract-Conference.html
仅借鉴较优定位结果指导较弱预测的思想，并非FDR/GO-LSD复现。

## PEQ-VFL

文件：`ultralytics/utils/loss_plugins/peq_vfl_loss.py`，注册名peq_vfl。
匹配正样本的质量目标变为 `(1-beta)*IoU + beta*mean(5个扰动IoU)`，五点为原框和中心向四方向平移。位移按实际输入宽高换算0.5像素，并限制在GT对应边长的10%内；中心裁剪保证框位于图像范围内。改变后的质量同时进入原VFL软标签和正样本权重；目标停止梯度。

默认beta=0.25，前20%训练线性升权。回归L1/GIoU、Hungarian代价、负类标签不变；主层、辅助层、编码器输出和DN均采用同一质量计算。beta=0、shift_px=0或max_shift_ratio=0时直接调用基线loss路径。手动调用criterion须在targets中提供image_hw=(H,W)；模型正常训练会自动提供。

来源：Uncertainty-Aware Gradient Stabilization for Small Object Detection，ICCV 2025。https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Uncertainty-Aware_Gradient_Stabilization_for_Small_Object_Detection_ICCV_2025_paper.html
这里借鉴不确定性动机，不是UGS离散定位/UM/UR复现。扰动幅度是超参数，并非测得的VisDrone标注误差。

## 调度、验证与限制

RTDETRTrainer向batch传递实际epoch/总轮数，模型loss更新PEQ调度，FQRank直接更新模块调度。支持正常resume的实际epoch；如改变总epochs，调度按新总轮数重新计算。自定义训练循环须主动调用set_epoch。未调用调度的独立推理使用保存的rank_alpha，独立PEQ criterion默认满强度。

```powershell
D:/Anaconda/envs/RTDETR/python.exe tools/test_third_innovations.py --full-model
```

测试覆盖：dense/sparse门控恒等退化与梯度，pre/post norm，FQRank推理选择退化、epoch调度、DN与空GT、质量头梯度、缓存清理、导出模式张量；PEQ关闭等价、回归不变、矩形输入、空GT、反向传播；四份完整YAML的构建、训练loss/反向、deepcopy和推理输出。
完整模型检查使用合成128×128输入；尚未跑350轮VisDrone，也未验证ONNX/TensorRT导出与实际延迟。覆盖率改善不能直接换算成AP。

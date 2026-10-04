# 分子特征正则

基因 VAE 在训练集表达数据上仅用重构 MSE 与 KL 预训练，不使用 spot 分类标签。
预训练后整个 VAE 冻结并保持 eval 模式，不参与分割阶段的优化或 EMA 更新。

分割训练将原始 crop 的解码特征按完整 spot 圆盘平均池化，再由图像投影头映射到
基因潜在维度。目标直接使用冻结编码器的后验均值 μ，不经过基因投影头。
每个 spot 等权参与单向余弦距离损失：

`L_molecular = mean(1 - cosine(image_projection, gene_mu))`

总损失保留分割、PFM 等已有损失及 MixUp/CutMix 项，只额外加入
`image.loss_weights.molecular * L_molecular`。分割阶段没有基因重构、KL 或分类损失。
分子正则权重由训练配置指定；设为 0 可关闭正则，并跳过训练步中的基因教师前向。
开启正则时 `vae.pretrain_epochs` 必须大于 0。

不再使用双向蒸馏、分类置信度门控、后验分类采样或基因 EMA。
`image.ema_decay` 仍用于图像模型的验证 EMA。推理流程不需要基因输入。

日志保留 `train/molecular`、`train/distillation/gene_to_image` 与 spot 数量。
`train/distillation/distillation_coverage` 只表示完整 spot 是否参与正则：有 spot 时为 1，
无 spot 时为 0，并非教师置信度。新增两侧 embedding 的逐维标准差均值用于观察
特征多样性；单 spot batch 的标准差为 0，不能单独据此判断塌缩。
旧的分类 AUC/F1、task loss、置信度和反向蒸馏日志不再生成。

新结构移除了基因投影头与分类头，需要重新预训练无监督教师；旧版教师或完整训练
checkpoint 不适用于此方案。图像模型结构和最终图像推理权重格式不变。
该正则约束图像表征与表达表征对齐，不保证分割提升或生物机制一致性。

## Spot 半径单位

训练配置使用 `dataset.spot_diameter_um` 指定物理直径，不再配置像素半径
`dataset.spot_radius`。DataModule 统一计算浮点像素半径：

`spot_radius_px = spot_diameter_um / (2 * resolution.target_mpp)`

Dataset 的完整 spot 筛选与 Distiller 的圆盘池化共用这个值。Distiller 再除以
`label_scale` 换算到特征网格，不额外乘原始图像的 MPP 或缩略图显示比例。
直径 100 μm、目标 1 MPP 时，图像半径仍为 50 px；在 7 MPP 特征网格上约为
7.14 px，因此当前分辨率下行为不变。`derive/dataset.yaml` 的物理直径设定应与
训练配置一致；`ingest` 中的固定 QC 标记半径不受此设置影响。

## 等价结构精简审计

此次审计覆盖入口、配置、数据集、模型、训练、动态传播和推理；只修改结构，
不调整超参数、张量计算公式、随机调用次序、模型参数注册顺序或训练阶段顺序。

| 区域 | 结构精简 |
| --- | --- |
| `main.py` / `pipeline.py` | 内联单次调度、路径和输出初始化包装；直接创建 CSVLogger，保留独立训练阶段函数。 |
| `modules/blocks.py` | 删除当前仓库没有实例化或外部调用的旧 ConvBlock、BlurPool、ResidualBlock、ModulatedBlock，以及对应导出。 |
| `modules/segmentation.py` | 删除无用导入和废弃注释，统一 ConditionalNorm 缩进，不改构造和前向计算。 |
| `training/segmentation_module.py` | 合并 MixUp/CutMix 的损失加权和日志代码，仍先执行 MixUp，再执行 CutMix。 |
| `inference/predictor.py` | 概率与传播特征共用完全相同的滑窗权重数组；内联单次上采样包装。 |
| `inference/metrics.py` | 内联仅调用一次的汇总包装，保留指标公式、字段顺序和 NaN/Inf 语义。 |
| `dynamic/visualization.py` | 复用已有的图像编码、叠色和面板函数，不增加公共工具模块，不改变保存文件。 |
| `dynamic/callback.py` | 内联仅调用一次的保存开关判断，保留动态更新时机和拒绝条件。 |

以下看似冗余的内容刻意保留，以避免改变行为或权重兼容性：

- ConvNeXtBlock 的 `norm` 虽未参与前向，仍注册参数，影响 state_dict、参数枚举和漂移统计。
- 路由诊断列表和无效构造参数未修改，模型接口与内部调用保持一致。
- 权重为 0 的 affinity、superpixel 和 PFM 项仍照常计算、记录；不跳过原来的计算图。
- No-KD 时仍创建并预训练 VAE、创建图像 projector，保留原来的流程和随机初始化消耗。
- VAE batch 和分割 batch 的标签字段保留；动态传播仍需要 spot 分类标签。
- EMA 更新公式、验证用 EMA / 最终测试用在线模型的原有口径均未修改。
- 数据加载、裁剪、染色与几何增强、spot 池化、随机游走和 SNIC 的算法保持原样。

旧网络块和入口内部包装没有仓库内调用；自定义外部脚本若导入这些已删除符号，
需改用当前模型或 `main.run` / `pipeline.run_fold`。当前图像权重结构保持不变。

### 审计中发现但未改变的行为

- EMA 参数在当前更新公式下先采用累计平均；修改公式会改变训练期验证结果，因此未调整。
- 验证集使用外层留出测试切片；不能用逐折最佳验证轮次作为独立测试结果，重构未调整划分。
- `augment.enabled` 控制图像及几何增强；PFM 屏蔽、MixUp、CutMix 仍由各自开关控制，未统一开关语义。
- Stem 步幅固定为 7；物理 spot 半径支持换算，不代表任意 target_mpp / label_mpp 组合都自动匹配网络网格。
- 漂移诊断逐 epoch 复制参数到 CPU，验证可视化再次读取专家掩码；缓存或降低频率虽能提速，
  但本次保留诊断与文件读取行为，没有引入缓存和新开关。

### 等价性检查范围

- 全部 Python 文件通过语法检查；Ruff 的 F 类检查及 `git diff --check` 通过。
- 所有 YAML 文件逐字节不变；模型、损失、数据集、随机游走算法定义的 AST 不变（忽略文档字符串）。
- 修改前快照与修改后代码在 CPU FP32 下对比 No-KD、MixUp、CutMix、联合蒸馏、空 spot、
  PFM 屏蔽和无效样本分支：初始化、损失、梯度、日志顺序、AdamW 更新、EMA、学习率调度、随机状态一致。
- 使用实际配置的图像网络检查参数与完整前向输出；滑窗概率/传播特征、指标字段、调度路径一致。
- 动态可视化 PNG、JPEG、JSON 文件逐字节一致；流程阶段与两次重置种子的顺序经 mock 检查一致。
- 未启动完整训练，也未验证 GPU BF16。完整 CPU BF16 前向在修改前后均因抗锯齿插值缺少
  BF16 内核而不可运行，该既有平台限制未通过修改算法绕过。

这些检查验证了覆盖路径的等价性，不构成对完整 GPU 多轮训练逐位复现的保证。

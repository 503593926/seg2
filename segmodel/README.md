# segmodel

当前唯一实验入口为 `her2st`：Stem 只变换通道、不做下采样，MoMu 条件 U-Net 在 1 mpp 网格输出特征，预测共享软组织状态，再解码分割与基因表达。使用患者隔离和混合监督训练。表达模板支持随机、NMF、K-means 及三种方式均分的混合初始化，并可选择继续训练或固定；状态预测与类别倾向仍随机初始化并参与训练。

## 网络与监督

```text
1 mpp H&E → Stem（3×3 Conv，stride=1，padding=1，3→64 + LayerNorm）
                              ↓
           预计算基础模型特征 → MoMu 条件 U-Net
                              ↓
              编码 1→4→16→64 mpp，逐层解码到 1 mpp
                              ↓
                        F（64 通道，1 mpp）
                              ↓ 状态编码器（1×1 Conv，64→64，GELU）
                   软状态 A（K=64，逐位置 softmax）与幅度 u
                         ├→ P = Σ_k A_k sigmoid(q_k) → 分割
                         └→ E_g = u Σ_k A_k B_kg → 非负线性表达图
```

PFM 的 CLS 与 patch 特征分别经过各自投影和共享投影，再用于 MoMu 的 Top-k 路由、通道调制和空间调制。`model.image.output_scale: 1` 表示 Stem 的步长为 1，只变换通道；输入为 1 mpp 时输出为 1 mpp。设置为 7 可使用原 7×7、stride=7 Stem，同时需将 `resolution.label_mpp` 设为 7。相邻编码层由 `downsample_factor: 4` 控制，使用 4×4、stride=4 卷积；解码端使用对应的 4×4、stride=4 转置卷积，并融合高分辨率跳连。896 crop 的空间尺寸为 `896→224→56→14→56→224→896`，最终输出保持 1 mpp。该参数设为 2 可使用两倍采样。状态编码器使用一层 1×1 卷积和 GELU，将 F 映射到 `tissue_state_dim` 通道（当前为 64），再由两个 1×1 头分别预测 K 通道状态和 1 通道幅度。每个位置的状态权重非负且和为 1。全局表达模板 B 为 K×G，每行由模板 logits 的 softmax 得到，默认可训练，也可固定；类别倾向 q 为 K 个可学习 logits。两种输出必须使用同一张状态图 A，没有独立分割捷径或分割修正头。

每个 stage 通过 `use_spatial` 统一控制内部所有 ConvNeXt block 的空间调制，global 调制始终保留。配置位于 `configs/model/base.yaml` 的 `image.encoder_spatial` 和 `image.decoder_spatial`，每个布尔值对应一个 stage。当前编码器顺序为 1、4、16、64 mpp，解码器顺序为 16、4、1 mpp；调整输出尺度或下采样倍数后，按新的 stage 尺度理解这些值。默认全部为 `false`，保持仅 global 调制。例如 `encoder_spatial: [false, true, true, false]` 只在编码器 4、16 mpp 两个 stage 开启空间调制。关闭的 stage 跳过空间插值、路由和调制；全部关闭时也跳过 patch 特征投影。开关随模型配置保存，独立推理时按相同设置重建。

幅度 u = G×softplus(amplitude_logits)，使初始化的逐基因表达保持 O(1)；由于模板逐行归一化，u 等于该位置所选 G 个基因预测表达之和，不代表真实细胞密度或原始 UMI 总数。类别 logits 使用 logsumexp 计算混合类别概率的 log odds，避免 clamp 概率截断梯度；模板混合与相关损失使用 FP32，兼容 bf16-mixed。

表达损失更新幅度、状态编码器和 MoMu，并在模板可训练时更新模板；分割损失更新类别倾向、状态编码器和 MoMu。共享状态实现任务耦合，但不会保证互相增益。学习到的状态也不能直接认定为细胞类型或真实细胞比例。1 mpp 是输出网格间距，spot 监督不能证明恢复了真实的 1 mpp 分子变化。当前仅接通细网格预测，尚未加入 cell 实例分割或 cell 聚合。

组织 mask 统一读取 `samples/<切片名>/spatial/tissue_mask.png`，不再回退到 `label_mpp` 旧目录。加载时直接读取组织 mask、伪标签、概率和 SNIC 缓存，不进行尺寸缩放。当前 1 mpp 流程由预处理保证这些数组与 H&E 和模型输出使用同一网格。

表达头输出当前训练 fold 所选基因的稠密非负线性相对表达。对每个有效 spot，在其圆盘内仅对组织像素求均值，再 `log1p`，与真实 CP10k + log1p 表达计算 MSE 和逐基因相关性损失。真实表达只用于监督；表达损失独立于类别种子和 RW 图是否可解。

每个 epoch 结束后，将当前模型的窗口 MoMu 特征 F 聚合到固定 SNIC 节点，并用该切片全部类别种子进行非可微随机游走传播。Lab 和 F 的节点均值分别标准化后计算边权，CPU 稀疏线性求解器计算癌概率。概率不高于 0.2 或不低于 0.8 的组织位置生成硬伪标签，其他位置忽略；无种子的连通分量保持 0.5。下一 epoch 的 crop 直接读取缓存的概率与监督范围，使用阈值化标签计算 BCE + Dice。

训练总损失对三项监督按 `training.image.loss_weights` 加权：

```text
L = w_seg × (BCE + Dice) + w_mse × MSE + w_pcc × mean(1 - PCC)
```

仅有表达的切片计算表达损失；有类别种子的切片还用整片传播的高置信伪标签计算分割损失。权重设为 0 即关闭对应监督。

PCC 在同一 crop 内沿 spot 维度逐基因计算，再对有效的 crop–基因项取平均，不跨切片混合相关性。每个 crop 默认至少需要 8 个有效 spot；少于该数量或真值标准差不超过 1e-6 的基因不贡献相关性损失，仍参与 MSE。常数预测不会被排除：对非恒定真值，相关性项为 1，并保留有限梯度。计算使用 FP32，可配合混合精度训练。

RW 目标使用预测网格上的 MoMu 特征 F；软状态和预测表达图不参与 RW 边权计算。

## 训练吞吐优化

共享状态结构训练时先在 spot 圆盘内聚合 `u × A`（K 通道），再乘 K×G 表达模板，得到 spot 的线性表达预测，随后计算 log1p、MSE 和 crop 内 PCC。由于模板不随空间位置变化，这与先解码稠密表达再求均值在数学上等价，避免训练时分配 B×G×H×W 基因场及其反向中间张量。整片验证/测试将窗口内 K 通道系数按逐像素 Hann 权重和归一化，在预测设备上用 FP64 流式累加到 spot，再转回 FP32、解码表达并计算指标。累加后除以整个 spot 圆盘内的组织像素数，保持先融合再取均值的定义；浮点求和顺序变化可能带来微小差异。仅在需要 dense 可视化时保留整片 K 通道系数，超过 256 MiB 的系数/特征拼接缓冲使用自动释放的临时文件映射。模型前向通过 `dense_expression=True` 返回稠密表达。

当前 1 mpp 使用 `crop_size: 896`（896 μm 视野）；训练和推理的 batch 大小分别由 `training.image.batch_size` 与 `dataset.inference_batch_size` 控制，训练通过 `accumulate_grad_batches` 控制每个优化器 step 累积的 batch 数。`gradient_checkpointing: true` 在训练反向时重算 ConvNeXt block、解码器上采样与跳连融合、共享状态预测头的激活，降低显存；融合与 block 分开 checkpoint，避免嵌套重算。沿用原 AMP 精度与随机数状态，不改变计算公式，以增加反向计算量换取显存。验证和推理不使用 checkpoint。训练 spot 聚合直接索引 NCHW 特征中的 batch 与像素位置，避免重排整幅特征的副本，仍先计算整幅 `u × A` 再按原权重聚合。相同网络深度在 1 mpp 下的局部卷积物理感受野比原 7 mpp 模式小，基础模型特征仍提供组织上下文。最终模型记录输出尺度与物理 MPP，可独立重建原生 1 mpp 输出。

BF16 混合精度下，通道 LayerNorm 使用原生算子，前向将仿射参数临时转为输入 dtype，内部用 FP32 累积，输出保持 BF16；条件归一化先在 FP32 中计算，再转回输入精度。条件调制系数和 LayerScale 的 gamma 按特征 dtype 参与乘加，因此骨干 stage 与 skip 特征保持 BF16。参数及其梯度仍为 FP32，状态概率、模板混合和表达损失继续使用 FP32。BF16 残差相加会产生舍入误差，初始化很小的 LayerScale 分支可能被舍入；梯度仍回传到原参数。训练与独立推理应使用一致的精度设置评估结果。PFM patch 投影在 autocast 下直接使用输入，避免先将 FP16 特征扩展到 FP32 再转 BF16；未启用 autocast 时保留 FP32 输入转换，CLS 均值继续使用 FP32。

位置状态熵日志在 detach 后按行分块计算，复用约 16 MiB 的 FP32 临时区，避免创建整幅状态图大小的 log 和乘积张量。统计公式与日志字段保持一致；分块求和可能改变日志的浮点末位，不参与训练损失，也不修改状态图。状态用量熵和平均幅度沿用原来的归约。

CLS 特征按非重叠 tile 网格直接读取，patch 特征可以继续重叠平均。训练和推理的窗口起点同时对齐标签、patch 和 CLS 网格；推理步长取不超过半窗的对齐值，至少为一个网格步长。当前四层编码器的总下采样倍数为 64，crop 必须同时满足模型与 CLS 网格的整除要求；CLS 步长为 224 时，crop 应为 448 的倍数。当前 crop=896、推理步长=448。未指定 `drop_path` 时，每个编码层使用 0，编码器和解码器层数按通道列表构建。

数据加载 worker 在几何增强后准备组织圆盘的像素索引和归一化权重，GPU 一次聚合整个 batch 的 spot。整片传播独立于训练计算图，训练步骤不再构造 RW 图或运行可微求解器。整片推理使用一个后台线程准备窗口，并通过 CUDA 拷贝流预取下一批锁页输入；按批次事件等待和记录使用流，保持原窗口顺序、batch 划分及融合权重，不增加预取配置。
HER2ST 训练 DataLoader 当前配置为：

```yaml
num_workers: 8
persistent_workers: true
prefetch_factor: 1
pin_memory: false
```

`num_workers=0` 时自动关闭持久化和预取。采样器把 epoch 与样本索引一起发送给 worker，避免预取使用旧的 epoch；有标注切片的癌概率、伪标签以及候选采样窗口使用共享存储，整片传播原地更新，因此 fork/spawn 常驻 worker 都能读取下一轮监督。固定超像素也使用共享存储，避免 spawn worker 复制整片数组。几何旋转和翻转先建立 NumPy 视图，最后合并成一次连续内存复制。训练结束时显式释放 worker，避免连续运行多个 fold 时保留整片数据。

大特征 batch 的多进程预取会同时占用共享内存和主进程中的锁页副本。WSL 的锁页内存可用量有限，HER2ST 使用 `prefetch_factor=1` 和 `pin_memory=false` 降低 Windows 主机内存压力；该开关不改变模型、损失或训练步数。通用 `dataset/base.yaml` 仍提供 `pin_memory=true`，可在其他环境中单独配置。

创建多进程训练 DataLoader 前，先在父进程调用 `cv2.setNumThreads(1)`，避免 fork 后调整继承的 OpenCV 并行池时死锁。每个训练 worker 初始化时再调用 `cv2.setNumThreads(1)` 和 `threadpool_limits(limits=1, user_api="blas")`，将 OpenCV 和已加载的 BLAS 线程池限制为单线程，并保留 `seed_everything(..., workers=True)` 的 Lightning 随机种子初始化。线程限制持续覆盖持久化 worker 的整个生命周期，可与启动前设置的 `OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OMP_NUM_THREADS=1` 配合使用。

## 目录

| 路径 | 职责 |
| --- | --- |
| `main.py` | Hydra 入口，按 fold 和随机种子组织实验 |
| `pipeline.py` | 准备数据、训练、保存最终模型、整片测试 |
| `configs/` | 数据、模型、训练、增强、RW 和实验配置 |
| `dataset/` | 切片加载、患者划分、裁剪、混合监督采样与增强 |
| `models/segmentation.py` | 完整图像与表达联合预测网络 |
| `modules/backbone.py` | Stem、ConvNeXt 编码器与解码器 |
| `modules/conditioning.py` | 基础模型特征投影、路由和条件调制 |
| `modules/tissue_states.py` | 共享软状态、表达模板、类别倾向与联合解码 |
| `dynamic/propagation.py` | 整片 RW 图构造、类别种子投票与 CPU 稀疏求解 |
| `modules/spot_pooling.py` | 稠密表达在 spot 圆盘内的组织均值 |
| `training/segmentation_module.py` | 损失接线、优化器、学习率、EMA 和验证日志 |
| `dynamic/callback.py` | 在固定超像素图上执行动态 RW，更新训练监督和采样窗口 |
| `training/losses.py` | 最终分割 BCE + Dice、表达 MSE 与逐基因 PCC |
| `inference/` | 滑窗拼接、分割/表达指标和结果可视化 |

## 配置入口

`configs/config.yaml` 默认选择 `experiment: her2st`，按原方式在 IDE 中运行 `main.py`。`configs/experiment/` 只保留 `her2st.yaml`，负责组合下面的配置文件：

| 配置文件 | 内容 |
| --- | --- |
| `configs/dataset/her2st.yaml` | 数据根目录、患者名单、自动发现切片和混合采样 |
| `configs/model/her2st.yaml` | 共享状态数、隐藏维度与原生输出尺度 |
| `configs/training/her2st.yaml` | HER2ST 表达 MSE/PCC 与分割损失权重 |
| `configs/augment/her2st.yaml` | 数据增强 |
| `configs/dynamic/her2st.yaml` | 整片 RW 传播开关与更新周期 |

各组的通用参数仍位于对应 `base.yaml`。

实验模式只保留 `patient_loocv` 和 `transfer`。`main.get_folds` 一次确定样本名单和路径，`run_fold` 统一执行数据准备、训练和最终测试；`FoldDataModule` 直接使用划分结果，不再重复扫描或生成 fold。患者留出每个患者训练一次；transfer 每个 seed 训练一次，再统一评估目标集所有切片。设置一个 seed 就只运行一次 transfer。

划分和 `split.json` 只保留 `train_names`、`test_names` 两个样本名单，分别包含全部训练、测试切片。每张切片的 `has_segmentation` 根据 `masks/training_mask.png` 是否存在确定；表达使用全部切片，分割监督、分割评估和动态传播只使用有标注的切片。采样所需的标注索引在训练 Dataset 内按此属性生成。

transfer 设置 `dataset.split_mode: transfer`、`ref_dataset`（源数据集）和 `tgt_dataset`（目标数据集）。切片分别从 `data/<数据集>/<target_mpp>mpp/samples/*/gene_counts.tsv.gz` 自动发现，有分割标注的切片通过 `masks/training_mask.png` 识别。表达文件默认读取 `data/<源集>_to_<目标集>/<源集>_to_<目标集>.h5ad`，文件名可通过 `gene_expression_file` 修改。该文件需使用源集确定的基因集合，`uns['train_names']` 记录源集训练切片，并包含源集、目标集的表达及坐标；切片名必须能区分源集与目标集。训练和模板初始化只使用源集，训练期间不读取目标集用于验证，训练结束后测试目标集。输出目录为 `seed_<seed>/<源集>_to_<目标集>/`。

表达监督变换设置在 `training.image.expression_transform`，HER2ST 为 `log1p`。MSE 权重为 `training.image.loss_weights.gene_prediction`，相关性权重为 `training.image.loss_weights.gene_correlation`。将后者设为 `0.0` 即关闭相关性监督。最少有效 spot 数设置在 `training.image.gene_correlation_min_spots`，默认 8，最低按 3 处理。基因数从训练表达矩阵读取。

表达模板初始化设置在 `configs/training/base.yaml` 的 `image.template_init.method`，支持 `random`、`nmf`、`kmeans` 和 `mix`。训练入口在每个 fold 开始时自动生成或读取缓存，无需单独运行脚本：

```yaml
template_init:
  method: mix  # 三种方式均分；也支持单独使用 random、nmf、kmeans
  trainable: true  # false：初始化后固定表达模板
  min_counts: 100  # 仅模板拟合过滤低计数 spot；0 关闭
  seed: 0
  max_iter: 1000
  kmeans_n_init: 10
  smoothing: 0.01
  cache_dir: outputs/template_initialization
```

数据初始化只读取当前 fold 的训练切片表达。HER2ST 的 log1p 数据先通过 expm1 还原为归一化后的线性表达；NMF 取 K×G 非负成分，K-means 在线性表达空间聚类并取各组的线性表达均值。所选基因全部为零的 spot 不参与拟合。模板逐行归一化，并混入配置比例的均匀分布，再以 log(template) 写入模板 logits。类别倾向和状态/幅度预测头继续使用随机初始化。

`template_init.min_counts` 默认 100，NMF、K-means 和 mix 拟合时仅使用 `obs['normalization_total_counts'] >= min_counts` 的训练 spot，阈值相等时保留，0 表示关闭此过滤。该字段记录归一化前、训练公共基因范围内的总 counts，不能从 CP10k/log1p 表达反推。启用过滤时若缺少该字段或训练计数非有限/为负，会明确报错。先过滤低 counts，再排除所选基因表达全零的 spot；若最终没有有效表达则停止并报告阈值。日志及初始化 JSON 记录输入数、低计数排除数、全零排除数和最终拟合数，阈值与过滤后表达内容进入缓存键，新配置自动使用新缓存。过滤仅作用于拟合输入，不修改 h5ad、训练/验证 spot、类别种子、采样或表达损失；random 初始化不读取计数或执行此过滤。

`mix` 要求 K >= 3，将模板数量尽量均分给 random、NMF、K-means；不能整除时余数依次分配给 random、NMF。模板按随机、NMF、K-means 的顺序合并。例如 K=32 时为 11+11+10，K=63 时为 21+21+21，K=64 时为 22+21+21。随机部分使用 `template_init.seed` 的独立随机数生成器，按与解码器默认初始化相同的 `Normal(0, 0.5)` logit 分布生成；NMF/K-means 在同一份经过上述过滤的训练表达上拟合。三部分均逐行归一化并应用相同 smoothing，因此同一 fold/参数可以跨模型训练 seed 复用完整混合模板。`max_iter` 分别限制两种拟合算法的迭代次数，`kmeans_n_init` 控制其中的 K-means 拟合。初始化 JSON 记录三种模板数量和各自的生成/拟合诊断；新 mix 缓存版本自动与旧的 NMF/K-means 各半缓存区分，单独 nmf/kmeans 的已有缓存仍可复用。

`template_init.trainable` 默认 `true`，模板随训练更新；设为 `false` 时，在创建 EMA 和优化器前冻结 `tissue_state_decoder.template_logits`，并将其排除在优化器参数组之外。四种初始化方式均支持该开关。固定模板时，表达损失仍通过模板更新状态/幅度头和骨干，类别倾向仍可训练；EMA 的模板也保持固定。该开关不改变拟合结果和缓存键，因此切换可训练性可以复用已有模板缓存；最终模型继续保存模板权重。

拟合仅在 CPU 上执行并限制线程数。缓存位于 `project_root` 下的 `template_init.cache_dir`，按方法、训练切片、基因顺序、表达数据内容、状态数、拟合参数和软件版本区分；初始化 seed 独立于模型训练 seed，相同 fold 和参数可跨训练 seed 复用。每个使用数据初始化的 fold 还会保存 `template_initialization.npz`（初始模板、基因/训练切片顺序）和 `template_initialization.json`（方法、拟合诊断与缓存信息）。模板在创建 EMA 前写入，普通模型和 EMA 从同一表达模板开始；最终模型仍按原接口保存和独立推理。

学习率在每次优化器更新后按 step 调度：先从 `0.1 × lr` 线性预热到 `lr`，再余弦衰减到 `0`。`training.image.warmup_steps` 只表示预热占总优化器更新次数的比例，取值范围 `[0, 1]`，实际预热步数为 `ceil(总步数 × warmup_steps)`；例如 `0.1` 表示前 10% 的训练步数用于预热，`0` 表示不预热，`1` 表示全程线性预热。总步数取 Lightning 的实际估计，考虑梯度累积、batch 限制和 `max_steps`。例如 2560 samples/epoch、batch size 16、100 epochs、`warmup_steps=0.1` 在单 GPU、无梯度累积时对应总计 16000 step、预热 1600 step。学习率不再使用 `warmup_epochs`；整片动态传播的周期和预热仍按 epoch 计数。

进度条通过 `training.image.progress_every_n_steps` 控制训练指标的显示间隔，默认每 100 个优化器 step 刷新一次，首个 step 和每轮末尾也刷新。显示当前 batch 的 Loss、Dice、IoU、表达 MSE 损失（`gene_prediction`）、相关性损失 1−PCC（`gene_correlation`）、有效 spot 数（`gene_spots`），以及全局 step 和科学计数法显示的当前学习率；表达损失仅显示已启用的项。梯度累积期间按优化器更新次数计数，epoch 汇总日志继续保留，验证指标仍在验证时更新。

当前默认模型设置：`tissue_state_count=64`、`tissue_state_dim=64`。G 从当前 fold 的训练表达矩阵读取。模板 `tissue_state_decoder.template_logits` 和类别倾向 `tissue_state_decoder.class_logits` 随模型权重保存，可分别 softmax/sigmoid 后查看。

在 `configs/model/her2st.yaml` 中设置共享软状态数 `tissue_state_count >= 2`。`tissue_state_dim` 是状态编码器的隐藏通道数，与模板数量 K 不同。

训练损失只有 `segmentation`、`gene_prediction` 和 `gene_correlation`，分别控制最终分割 BCE + Dice、表达 MSE 和逐基因 1−PCC。表达单任务将分割权重设为 0；共享状态的分割单任务将表达 MSE/PCC 都设为 0，仍构造相同状态和模板参数，没有表达监督时不能将模板解释为学到的分子模式。

## 整片动态传播

配置位于 `configs/dynamic/base.yaml`，HER2ST 通过 `configs/dynamic/her2st.yaml` 继承：

```yaml
enabled: true
warmup_epochs: 2
interval_epochs: 1
save_intermediate: true
```

`warmup_epochs` 指定首次更新的 epoch，当前为 2：前两轮使用离线伪标签，第 2 个 epoch 结束后首次传播，第 3 个 epoch 开始使用新监督。之后按 `interval_epochs` 更新，没有轮数上限，与验证周期和学习率预热独立。超像素固定使用预处理生成的 `label_mpp/<切片名>/snic/superpixels.npy`。每张切片首次更新时缓存图拓扑、标准化后的 Lab 节点均值、吸附后的 spot 与种子投票结果、可求解节点和滑窗权重和；恢复回调状态时重建这些缓存，训练结束后释放。每轮重新计算当前模型的节点特征、边权和 RW 概率。

仅处理当前 fold 中有类别种子的训练切片，使用普通模型当前的 F（`feature_key="features"`）进行整片滑窗推理。传播图使用分别标准化的 Lab/F 节点均值，`learned_weight=1.0` 按通道数平衡 F 与 Lab，`beta=1.0` 控制边权核。类别种子按 `max_snap_distance_um=100.0` 吸附到组织边缘，冲突种子不固定节点。无种子的连通分量保持不确定。超像素的尺寸和紧凑度由离线预处理配置决定。

有效更新只提交癌概率和 0/1/255 伪标签，并刷新候选训练窗口，超像素数组始终固定；只记录覆盖率比和重叠区翻转率，不设质量接受门槛。概率非有限或没有可用训练窗口时保留上一轮监督。更新留在训练内存中，不写回离线数据。动态回调由 `dynamic.enabled` 控制，独立于损失权重；分割权重为 0 时仍生成传播结果，启用分割训练需设置 `training.image.loss_weights.segmentation`。

动态传播不再分配或读写整片 H×W×C 特征场。每个窗口的 F 按原 Hann 权重除以逐像素的总权重后，在预测设备上用 FP64 累加到节点，最后按节点像素数求均值，仅将节点表传回 CPU。窗口位置、步长、融合权重和最终 1 mpp 超像素标签保持原定义；浮点求和顺序变化可能带来微小数值差异。模型概率图和 QC 输出保持原流程。RW 前向只计算 F 和分割 logits，跳过表达状态的 exp、幅度头和模板归一化；dense 表达可视化仍使用原来的整片系数拼接。

每次有效更新仅保存下面两个文件到 `<fold>/dynamic_pseudo_labels/epoch_0002/A1/` 等目录，`save_intermediate: false` 关闭保存：

| 文件 | 内容 |
| --- | --- |
| `overview.jpg` | 四面板：H&E 与种子、模型概率、RW 概率、实际提交的伪标签 |
| `metadata.json` | 覆盖率、翻转率、候选窗口数、固定超像素数量及传播参数 |

## 患者隔离与混合训练

HER2ST 在 `dataset.patient_names` 中填写患者字母，按列表顺序执行患者留出。表达切片从 `data_root/samples/*/gene_counts.tsv.gz` 自动发现，只使用所选患者；存在 `masks/training_mask.png` 的切片用于分割监督和评估，无需维护 `all_names` 或 `expression_names`。当前数据自动识别为 36 张表达切片、8 张标注切片。例如留出 C 患者时，C1–C6 全部排除在训练之外，C1 评估分割，C1–C6 都评估表达。表达预处理直接使用患者名单，按患者名保存 `gene_expression/patient_loocv/test_C.h5ad`，对应实验目录为 `seed_<seed>/test_C/`，不依赖该患者是否有分割标注。

每个表达文件保存全部切片在当前 fold 基因集合上的表达。`uns` 保留 `train_names`、`test_names`、`test_patient`、选基因和归一化参数，以及评估需要的 `missing_selected_genes_by_slide`；不再写入旧任务名、测试切片名、分割/表达名单及重复的缺失基因汇总。归一化前总 counts 继续保存在 `obs['normalization_total_counts']`，供表达模板初始化过滤低计数 spot。

其他患者的有标注切片和仅有表达切片混合采样，比例由 `dataset.labeled_sampling_fraction` 控制；设为 `null` 时从全部训练切片等概率抽取。仅表达切片没有类别种子，不贡献分割监督，但其有效 spot 参与表达损失。基因选择和归一化基因集合由训练患者决定，留出患者不参与选择。

## 指标与输出

验证与最终测试均以 H&E 和预计算图像特征为输入，不需要真实表达或类别种子。

- 分割：Dice、IoU 及混淆矩阵相关指标，不计算距离指标。
- `gene_prediction`：有效 spot × 基因观测上的表达 MSE。
- `gene_prediction_rmse`：表达 MSE 的平方根。
- `gene_prediction_pcc`：逐基因跨 spot 计算 PCC，再对可计算的基因取均值。
- `gene_prediction_pcc_max/min`：上述有效基因 PCC 的最大/最小值。
- 患者表达主指标合并该患者全部切片的有效 spot；同时保存各切片结果和宏平均。

进度条额外显示 `val/gene_prediction_pcc_macro`：先计算每张留出切片的逐基因平均 PCC，再对可计算的切片等权平均；每轮验证结束后更新，单切片时等于该切片的 PCC。它沿用验证指标定义，不是当前训练 batch 的相关性损失。

常数预测、常数真值及观测不足的基因不参与 PCC；缺失基因值不当作真实零表达。

训练日志分别记录未加权的 `train/gene_prediction`（MSE）和 `train/gene_correlation`（平均 1-PCC），后者越低越好；没有可计算的相关性项时为 0。总损失 `train/loss` 使用配置权重。另记录 `train/gene_spots`、启用的分割损失、整片传播覆盖率和翻转率记录。共享状态额外记录逐位置状态熵 `train/tissue_state_entropy`、batch 平均状态用量的熵 `train/tissue_state_usage_entropy` 和平均幅度 `train/expression_amplitude_mean`；两个熵均使用自然对数，取值范围约为 0 到 ln(K)。低用量熵提示少数状态占主导，高位置熵可能表示状态混合接近均匀，不能只凭熵判断质量。验证使用 EMA 模型，最终保存和测试使用普通模型。验证表达 MSE、RMSE、PCC 的定义保持不变，训练相关性项使用 crop 内统计，验证 PCC 使用整片或整患者统计。

每个 fold 保存 `config.yaml`、`split.json`、`metrics.csv`、`validation/`、`final_model.pt`、`test/<slide>/` 和 `metrics.json`。训练日志 `metrics.csv` 直接保存在 fold 目录。整片传播有效更新时额外保存 `dynamic_pseudo_labels/epoch_<轮次>/<训练切片>/`。有分割评估的切片保存概率图和掩码；表达评估保存 `gene_predictions.npz`。

Lightning 在 `fit()` 结束时会将模型移回 CPU。最终测试保存权重后，将普通模型恢复到 `trainer.strategy.root_device`，并复用 `trainer.precision_plugin.forward_context()`，使测试设备与计算精度和训练验证一致。开始测试时打印实际设备与精度；当前 GPU 配置使用 BF16 混合精度，参数和保存的权重仍为 FP32。

每轮验证结束后，`training.image.gene_visualization_top_n` 控制最高 PCC 基因的空间可视化，默认 10，设为 0 关闭。将当前留出患者所有有效 spot 合并，按逐基因 PCC 降序选择前 N 个；PCC 的观测掩码和常量排除规则与验证指标一致，可计算基因少于 N 时只显示实际可用数量。单张切片验证也支持。HER2ST 默认每个 epoch 验证一次；调整 `val_every_n_epochs` 后，可视化随验证周期生成。

结果位于 `validation/epoch_0001/gene_expression/`：`top_genes.csv` 保存该轮排名，`metadata.json` 保存排名口径、归一化范围和每张切片的 PCC。每张留出切片的每个入选基因只保存一张 `01_<gene>_dense_comparison_normalized.png`，三列分别为归一化的 spot 真值、spot 预测和 dense 预测。图像使用 EMA 验证结果，标题同时给出 pooled PCC 和该切片自身的 PCC。

对每个基因，真值、spot 预测和 dense 预测分别按所有留出切片的 min/max 映射到 0～1；同一来源的不同切片使用统一范围，保留切片间的相对差异。未观测 spot 不参与范围估计或绘图，dense 范围仅统计组织像素，常量范围映射为 0。色图使用 viridis，组织外显示白色。原始 min/max 记录在 `metadata.json` 的 `normalization` 和 `dense_normalization` 中，切片记录包含 `dense_mpp`、`dense_shape` 和唯一图片路径。归一化仅作用于绘图数组，PCC 排名、RMSE 和训练目标均使用原始表达。

dense 列来自滑窗融合后的原生预测网格，HER2ST 为 1 mpp，展示时缩放到对照面板；spot 真值仍只显示实际测量值。HER2ST 的 dense 绘图使用逐格 `log1p(线性预测)`，spot 预测使用 `log1p(圆盘内线性均值)`。当前共享状态模型仅暂存 CPU 状态系数及模板，排名后只解码入选基因；绘图结束即释放稠密缓存。绘图不增加模型推理；设置 `gene_visualization_top_n: 0` 时不缓存或生成这些图。

`final_model.pt` 包含图像模型权重、构造参数、滑窗设置和有序基因名称，可以重建模型并单独推理。

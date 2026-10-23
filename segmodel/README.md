# segmodel

当前唯一实验入口为 `her2st`：Stem 将 1 mpp 输入下采样到 2 mpp，MoMu 条件 U-Net 在 2 mpp 网格输出特征，预测共享软组织状态，再解码分割与基因表达。使用患者隔离和混合监督训练。表达模板仅使用 random 随机初始化，数量由 `model.image.tissue_state_count` 指定，全部参与训练。状态预测与类别倾向仍随机初始化并参与训练。

## 网络与监督

```text
1 mpp H&E → Stem（3×3 Conv，stride=2，padding=1，3→32 + LayerNorm）
                              ↓
           预计算基础模型特征 → MoMu 条件 U-Net
                              ↓
              编码 2→4→8→16 mpp，逐层解码到 2 mpp
                              ↓
                        F（32 通道，2 mpp）
                              ↓ 状态编码器（1×1 Conv，32→32，GELU）
                 软状态 A（默认 K=32，逐位置 softmax）与幅度 u
                         ├→ P = Σ_k A_k sigmoid(q_k) → 分割
                         └→ E_g = u Σ_k A_k B_kg → 非负线性表达图
```

PFM 的 CLS 与 patch 特征分别经过各自投影和共享投影，再用于 MoMu 的 Top-k 路由、通道调制和空间调制。`model.image.output_scale: 1` 表示 Stem 的步长为 1，只变换通道；输入为 1 mpp 时输出为 1 mpp。设置为 7 可使用原 7×7、stride=7 Stem，同时需将 `resolution.label_mpp` 设为 7。相邻编码层由 `downsample_factor: 2` 控制，使用 2×2、stride=2 卷积；解码端使用双线性插值放大 2 倍和 3×3 卷积，并融合高分辨率跳连。当前 896 crop 经 Stem 后的空间尺寸为 `448→224→112→56→112→224→448`，最终输出为 2 mpp。`output_scale: 2` 使用 3×3、stride=2、padding=1 的 Stem，将 1 mpp 输入映射到 2 mpp 输出。状态编码器使用一层 1×1 卷积和 GELU，将 F 映射到 `tissue_state_dim` 通道（当前为 32），再由两个 1×1 头分别预测 K 通道状态和 1 通道幅度。每个位置的状态权重非负且和为 1。全局表达模板 B 为 K×G，每行由模板 logits 的 softmax 得到；所有模板随机初始化并参与训练。类别倾向 q 为 K 个可学习 logits。两种输出必须使用同一张状态图 A，没有独立分割捷径或分割修正头。

每个 stage 通过 `use_spatial` 统一控制内部所有 ConvNeXt block 的空间调制，global 调制始终保留。配置位于 `configs/model/base.yaml` 的 `image.encoder_spatial` 和 `image.decoder_spatial`，每个布尔值对应一个 stage。当前编码器顺序为 2、4、8、16 mpp，解码器顺序为 8、4、2 mpp；调整输出尺度或下采样倍数后，按新的 stage 尺度理解这些值。当前 `encoder_spatial: [false, false, false, true]` 只在编码器 16 mpp stage 开启空间调制，解码器全部关闭。关闭的 stage 跳过空间插值、路由和调制；全部关闭时也跳过 patch 特征投影。开关随模型配置保存，独立推理时按相同设置重建。

幅度 u = G×softplus(amplitude_logits)，使初始化的逐基因表达保持 O(1)；由于模板逐行归一化，u 等于该位置所选 G 个基因预测表达之和，不代表真实细胞密度或原始 UMI 总数。类别 logits 使用 logsumexp 计算混合类别概率的 log odds，避免 clamp 概率截断梯度；模板混合与相关损失使用 FP32，兼容 bf16-mixed。

表达损失更新模板、幅度、状态编码器和 MoMu；分割损失更新类别倾向、状态编码器和 MoMu。共享状态实现任务耦合，但不会保证互相增益。学习到的状态也不能直接认定为细胞类型或真实细胞比例。1 mpp 是输出网格间距，spot 监督不能证明恢复了真实的 1 mpp 分子变化。当前仅接通细网格预测，尚未加入 cell 实例分割或 cell 聚合。

输入与输出 MPP 相同时，组织 mask 继续读取 `samples/<切片名>/spatial/tissue_mask.png`，SNIC/伪标签使用原 `label_mpp/<切片名>/` 缓存。输出更粗时，组织 mask、专家标签、SNIC 和伪标签统一读取 `label_mpp/<label_mpp>mpp/<切片名>/`。训练加载只检查网格尺寸，不进行在线缩放。H&E、PFM 特征和表达 h5ad 始终保持输入网格；表达 spot 坐标与半径由 `label_scale` 换算一次。

### 离线标签网格与 2 mpp 输出

恢复的 `derive/get_label_mpp.py` 为所有表达切片和有标注切片准备标签网格：H&E 使用面积缩小，组织/画布/有效 mask 和 0/1/255 专家标签使用最近邻，另生成仅供离线 RW 使用的坐标表和 `grid.json`。超像素和伪标签随后直接在该网格重新生成，尺寸、吸附距离与 spot 可视化半径均按 `label_mpp` 换算，保留原物理单位。

在项目根目录按以下顺序执行：

```bash
python -m derive.get_label_mpp
python -m derive.get_superpixel
python -m derive.get_pseudo_label
```

当前配置使用 1 mpp 输入、2 mpp 癌症与表达输出，需要保持 `config/resolution.yaml` 的 `label_mpp` 为 `2.0`，以及 `segmodel/configs/model/her2st.yaml` 的 `image.output_scale` 为 `2`，然后执行上面三步。2 mpp 文件写入 `data/her2st/1mpp/label_mpp/2mpp/<切片名>/`；输入图像、特征和表达 h5ad 继续复用。896×896 输入产生 448×448 的 logits、状态图和幅度；整片概率与表达系数保留原生网格，验证真值也使用此网格，可视化不再二次缩小。此前 2×2、stride=2 Stem 的权重与当前 3×3 Stem 尺寸不同，加载旧权重时需要重新初始化 Stem。

离线 `spatial_2mpp.tsv` 只供 SNIC/RW 流程；训练与表达评估仍读取 h5ad 中的输入分辨率坐标。表达圆盘聚合计算组织均值，网格变粗时不对表达数值乘除面积倍率。跨 MPP 比较分割指标时，应统一评估网格。脚本不会覆盖输入数据或其他 MPP 的缓存。

表达头输出当前训练 fold 所选基因的稠密非负线性相对表达。对每个有效 spot，在其圆盘内仅对组织像素求均值，再 `log1p`，与真实 CP10k + log1p 表达计算 MSE 和逐基因相关性损失。真实表达只用于监督；表达损失独立于类别种子和 RW 图是否可解。

每个 epoch 结束后，将当前模型的窗口 MoMu 特征 F 聚合到固定 SNIC 节点，并用该切片全部类别种子进行非可微随机游走传播。Lab 和 F 的节点均值分别标准化后计算边权，CPU 稀疏线性求解器计算癌概率。概率不高于 0.2 或不低于 0.8 的组织位置生成硬伪标签，其他位置忽略；无种子的连通分量保持 0.5。下一 epoch 的 crop 直接读取缓存的概率与监督范围，使用阈值化标签计算 BCE + Dice。

训练总损失对三项监督和一项状态用量正则按 `training.image.loss_weights` 加权：

```text
L = w_seg × (BCE + Dice) + w_mse × MSE + w_pcc × mean(1 - regularized_PCC)
    + w_balance × mean_k(max(0, log(alpha/K) - log(u_k)))
u_k = (1 / |T|) Σ_{p∈T} A_{k,p}
```

`T` 是 `batch["tissue"]`，覆盖所有组织像素，包括仅有表达的切片。`state_balance` 仅惩罚平均用量低于 `alpha/K` 的状态，达到下限后不再推动均匀分配。`training.image.state_balance_alpha` 默认 `0.5`，取值 `(0, 1]`；例如 K=32 时下限为 1.5625%。该参数控制用量下限，`loss_weights.state_balance` 独立控制惩罚强度，权重为 0 时关闭。损失非负且没有固定上界，与旧归一化 KL 的数值尺度不同，历史日志不能直接对比。统计从 `log_softmax` 在对数域归约，避免先 exp 再 clamp/log 导致极低用量状态失去恢复梯度；它仍不直接约束模板差异，也不能排除所有像素输出相同混合的解。

仅有表达的切片计算表达损失；有类别种子的切片还用整片传播的高置信伪标签计算分割损失。权重设为 0 即关闭对应项。

PCC 在 spot 表达完成 `log1p` 后，先将同一 slide 在当前 batch 中的全部有效 spot 合并，分别减去该 slide 的预测均值和真值均值，再将各 slide 的残差拼接，沿 spot 维度逐基因计算一次正则化 PCC，最后对有效基因的 `1-regularized_PCC` 取平均。两侧尺度使用 `sqrt(mean(residual²) + std_floor²)`，默认 `std_floor=0.02`，单位为 log1p 表达标准差；高方差时接近原 PCC，近零方差时平滑限制梯度。这样消除片间均值差异，保留片内变化；同一 slide 的多个 crop 共用均值，不是分别计算 crop PCC 再平均。默认每个 slide 在当前 batch 中合并后至少需要 8 个有效 spot，最低按 3 处理；不足门槛的 slide 组只跳过 PCC，仍参与 MSE。没有满足门槛的组时，PCC 损失为可反向传播的 0。真值残差均方不超过 1e-12 的基因不贡献相关性损失。片内恒定预测不会被排除：对有片内变化的真值，相关性项为 1，并保留经过软方差下限约束的梯度。MSE 仍使用原始 spot 表达，不做 slide 中心化。计算使用 FP32，可配合混合精度训练；梯度累积不扩大 PCC 的统计范围。

RW 目标使用预测网格上的 MoMu 特征 F；软状态和预测表达图不参与 RW 边权计算。

## 训练吞吐优化

共享状态结构训练时先在 spot 圆盘内聚合 `u × A`（K 通道），再乘 K×G 表达模板，得到 spot 的线性表达预测，随后计算 log1p、MSE 和按 slide 去均值后的 batch 正则化相关性损失。由于模板不随空间位置变化，这与先解码稠密表达再求均值在数学上等价，避免训练时分配 B×G×H×W 基因场及其反向中间张量。整片验证/测试将窗口内 K 通道系数按逐像素 Hann 权重和归一化，在预测设备上用 FP64 流式累加到 spot，再转回 FP32、解码表达并计算指标。累加后除以整个 spot 圆盘内的组织像素数，保持先融合再取均值的定义；浮点求和顺序变化可能带来微小差异。仅在需要 dense 可视化时保留整片 K 通道系数，超过 256 MiB 的系数/特征拼接缓冲使用自动释放的临时文件映射。模型统一返回状态、幅度和模板；训练与推理均先聚合 K 通道系数再解码表达，可视化按需解码入选基因。

当前 1 mpp 输入使用 `crop_size: 896`（896 μm 视野）；训练和推理的 batch 大小分别由 `training.image.batch_size` 与 `dataset.inference_batch_size` 控制，训练通过 `accumulate_grad_batches` 控制每个优化器 step 累积的 batch 数。`gradient_checkpointing: true` 在训练反向时重算 ConvNeXt block、解码器上采样与跳连融合、共享状态预测头的激活，降低显存；融合与 block 分开 checkpoint，避免嵌套重算。沿用原 AMP 精度与随机数状态，不改变计算公式，以增加反向计算量换取显存。验证和推理不使用 checkpoint。训练 spot 聚合直接索引 NCHW 特征中的 batch 与像素位置，避免重排整幅特征的副本，仍先计算整幅 `u × A` 再按原权重聚合。相同网络深度在 1 mpp 下的局部卷积物理感受野比原 7 mpp 模式小，基础模型特征仍提供组织上下文。最终模型记录输出尺度与物理 MPP，可独立重建相同网格的输出。

BF16 混合精度下，通道 LayerNorm 使用原生算子，前向将仿射参数临时转为输入 dtype，内部用 FP32 累积，输出保持 BF16；条件归一化先在 FP32 中计算，再转回输入精度。条件调制系数和 LayerScale 的 gamma 按特征 dtype 参与乘加，因此骨干 stage 与 skip 特征保持 BF16。参数及其梯度仍为 FP32，状态概率、模板混合和表达损失继续使用 FP32。BF16 残差相加会产生舍入误差，初始化很小的 LayerScale 分支可能被舍入；梯度仍回传到原参数。训练与独立推理应使用一致的精度设置评估结果。PFM patch 投影在 autocast 下直接使用输入，避免先将 FP16 特征扩展到 FP32 再转 BF16；未启用 autocast 时保留 FP32 输入转换，CLS 均值继续使用 FP32。

位置状态熵和 valid 掩膜上的状态用量熵在 detach 后按行分块计算，复用约 16 MiB 的 FP32 临时区，避免创建整幅状态图大小的 log 和乘积张量；两者都是诊断，不参与训练损失，也不修改状态图。`state_balance` 在共享状态头内按同一分块预算计算组织掩膜上的 `logsumexp`，跨块以 `logaddexp` 合并，返回 K 维对数用量和组织像素数。统计处于状态头的 checkpoint 内，不额外返回完整对数状态图；启用 checkpoint 时反向重算统计中间张量。空组织块采用有限的占位归约后丢弃，避免全 `-inf` 归约产生 NaN 梯度；全空掩膜的损失和梯度均为 0。权重为 0 时在 `no_grad` 下统计，只写日志。平均幅度沿用原来的归约。

CLS 特征按非重叠 tile 网格直接读取，patch 特征可以继续重叠平均。训练和推理的窗口起点同时对齐标签、patch 和 CLS 网格；推理步长取不超过半窗的对齐值，至少为一个网格步长。当前四层编码器含 Stem 的总下采样倍数为 16，crop 必须同时满足模型与 CLS 网格的整除要求；CLS 步长为 224 时，crop 应为 224 的倍数。当前 crop=896、推理步长=448。`drop_path` 在模型 YAML 中显式设置为 `[0.1, 0.1, 0.2, 0.3]`，长度必须与编码层数一致。

数据加载 worker 在几何增强后准备组织圆盘的像素索引和归一化权重，GPU 一次聚合整个 batch 的 spot。整片传播独立于训练计算图，训练步骤不再构造 RW 图或运行可微求解器。整片推理使用一个后台线程准备窗口，并通过 CUDA 拷贝流预取下一批锁页输入；按批次事件等待和记录使用流，保持原窗口顺序、batch 划分及融合权重，不增加预取配置。
整片推理按 GPU 复用 CUDA 拷贝流，避免每张切片新建流后留下多组输入缓存。提前结束推理时取消尚未执行的预取。验证和动态传播阶段的开始、结束都会等待 CUDA 工作完成并归还空闲显存缓存，同时输出 `allocated` 和回收前后的 `reserved`（GiB），用于区分活跃张量累积与空闲缓存扩张。清理仅发生在阶段边界，不在每个 batch 执行；模型参数、优化器状态和 CPU 的 RW 图缓存继续保留。

HER2ST 训练 DataLoader 当前配置为：

```yaml
num_workers: 8
persistent_workers: true
prefetch_factor: 1
pin_memory: false
```

`num_workers=0` 时自动关闭持久化和预取。采样器把 epoch 与样本索引一起发送给 worker，避免预取使用旧的 epoch；有标注切片的癌概率、伪标签以及候选采样窗口使用共享存储，整片传播原地更新，因此 fork/spawn 常驻 worker 都能读取下一轮监督。固定超像素也使用共享存储，避免 spawn worker 复制整片数组。几何旋转和翻转先建立 NumPy 视图，最后合并成一次连续内存复制。训练结束时显式释放 worker，避免连续运行多个 fold 时保留整片数据。

`dataset.slides_per_batch` 控制每个完整训练 batch 的不同切片数量，HER2ST 默认 `4`；它必须为正整数、不超过训练切片数，并整除 `training.image.batch_size`。例如 batch=8 时每张切片取 2 个 crop，batch=32 时每张取 8 个。设为 `null` 恢复每个 crop 独立随机选择切片。分组采样先选择互不重复的切片，再沿用 spot 引导采样，从同一切片的合法窗口中无放回选择起点；候选耗尽后允许再次使用，不强制窗口距离或无重叠，也不对重叠 crop 中的 spot 去重。`labeled_sampling_fraction` 在切片组层面分配数量，例如 4 张切片、比例 0.5 时选择 2 张有标注和 2 张仅表达切片；非整数数量随机舍入，某组切片不足时从另一组补齐。末尾不足一个 batch 时尽量均分，必要时减少切片数，保持 `samples_per_epoch` 不变。采样计划带有 epoch 和窗口坐标，持久化 worker 读取当前 RW 更新后的合法窗口和监督；现有 PCC 继续合并同一切片的有效 spot。

大特征 batch 的多进程预取会同时占用共享内存和主进程中的锁页副本。WSL 的锁页内存可用量有限，HER2ST 使用 `prefetch_factor=1` 和 `pin_memory=false` 降低 Windows 主机内存压力；该开关不改变模型、损失或训练步数。通用 `dataset/base.yaml` 仍提供 `pin_memory=true`，可在其他环境中单独配置。

创建多进程训练 DataLoader 前，先在父进程调用 `cv2.setNumThreads(1)`，避免 fork 后调整继承的 OpenCV 并行池时死锁。每个训练 worker 初始化时再调用 `cv2.setNumThreads(1)` 和 `threadpool_limits(limits=1, user_api="blas")`，将 OpenCV 和已加载的 BLAS 线程池限制为单线程，并保留 `seed_everything(..., workers=True)` 的 Lightning 随机种子初始化。线程限制持续覆盖持久化 worker 的整个生命周期，可与启动前设置的 `OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OMP_NUM_THREADS=1` 配合使用。

## 目录

| 路径 | 职责 |
| --- | --- |
| `main.py` | Hydra 入口，按 fold 和随机种子组织实验 |
| `pipeline.py` | 准备数据、训练、保存最终模型、整片测试 |
| `configs/` | 数据、模型、训练、增强、RW 和实验配置 |
| `dataset/` | 切片加载、患者划分、裁剪、混合监督采样与增强 |
| `dataset/loading.py` | 共用基础切片输入读取，训练另加载监督，推理另检查特征网格 |
| `dataset/sampling.py` | 按 epoch 生成可复现的切片分组和 crop 起点 |
| `models/segmentation.py` | 完整图像与表达联合预测网络 |
| `modules/backbone.py` | Stem、ConvNeXt 编码器与解码器 |
| `modules/conditioning.py` | 基础模型特征投影、路由和条件调制 |
| `modules/tissue_states.py` | 共享软状态、表达模板、类别倾向与联合解码 |
| `dynamic/propagation.py` | 整片 RW 图构造、类别种子投票与 CPU 稀疏求解 |
| `modules/spot_pooling.py` | 稠密表达在 spot 圆盘内的组织均值 |
| `training/segmentation_module.py` | Lightning 入口：前向、损失组合、日志和 EMA 更新 |
| `training/optim.py` | AdamW 参数分组与按更新步数执行的 warmup/cosine 调度 |
| `training/diagnostics.py` | 不参与反向传播的分块状态统计 |
| `training/validation.py` | `SlideValidation` 管理验证配置、本轮缓存、整片预测与表达汇总 |
| `dynamic/callback.py` | 在固定超像素图上执行动态 RW，更新训练监督和采样窗口 |
| `training/losses.py` | 最终分割 BCE + Dice、表达 MSE 与逐基因正则化 PCC |
| `inference/` | 滑窗拼接、分割/表达指标和结果可视化 |
| `inference/prefetch.py` | 后台窗口准备、CUDA 拷贝与预取资源释放 |

阅读训练代码可以从 `pipeline.run_fold` 进入，再看
`SegmentationModule.training_step`：一次基础前向 → 分割/表达/模板/状态损失 →
按配置顺序加权 → 日志。损失公式集中在 `training/losses.py`；
验证和日志统计独立于这条计算路径，Lightning 的钩子仍显式保留在训练模块中。

验证结果统一保存在 `module.validation.metrics` 和 `module.validation.expression`，每轮开始重置。
验证对象接收模型、数据和设备等明确参数，不读取训练模块或 Trainer 的内部属性；
Lightning 钩子负责选择 EMA、记录日志以及阶段边界的 CUDA 缓存清理。
`inference/visualization.py` 在同一文件内按整理观测、基因排名、稠密解码、绘图和落盘组织，
共享原来的排名口径、归一化范围和文件格式。训练入口专用于共享组织状态模型，
不再兼容缺少状态/表达输出的模型。

模型参数名称保持不变，模型构造参数全部由解析后的 YAML 提供并保存在 `final_model.pt` 的 `model_config` 中。骨干不再逐层传递未被读取的
`routing` 列表；全局和空间路由共用同一个 top-k softmax 函数。
`EpochSampler` 的原导入路径仍可使用，推荐从 `dataset.sampling` 导入。

重构数值对照工具为 `tools/check_segmodel_equivalence.py`，在项目根目录运行：

```bash
python -m tools.check_segmodel_equivalence --reference /path/to/before
python -m tools.check_segmodel_equivalence --reference /path/to/before --device cuda
python -m tools.check_segmodel_contracts
```

`before/` 应包含重构前完整的 `segmodel/`。工具使用合成数据和小网络，比较固定种子下的
输出、梯度、优化器及 EMA 状态、随机数状态、日志、采样和整片推理，并执行带梯度累积的
Lightning 训练。张量要求逐元素一致（`rtol=0, atol=0`）；这不等同于重跑完整患者实验。
文件对照覆盖 1/2 mpp 的训练及推理加载、网格错误检查，以及可视化 PNG、CSV、JSON 的完整字节。
验证缓存位置等已调整接口由对照脚本适配；已删除且无消费者的 `spot_count` 字段不参与比较。
接口检查另外覆盖完整配置与权重重建、联合 spot/超像素聚合及 Hann 缓存复用、
异常后恢复模型训练模式，以及 `spawn` 常驻 worker 接收更新后的伪标签。

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

transfer 设置 `dataset.split_mode: transfer`、`ref_dataset`（源数据集）和 `tgt_dataset`（目标数据集）。切片分别从 `data/<数据集>/<target_mpp>mpp/samples/*/gene_counts.tsv.gz` 自动发现，有分割标注的切片通过 `masks/training_mask.png` 识别。表达文件默认读取 `data/<源集>_to_<目标集>/<源集>_to_<目标集>.h5ad`，文件名可通过 `gene_expression_file` 修改。该文件需使用源集确定的基因集合，`uns['train_names']` 记录源集训练切片，并包含源集、目标集的表达及坐标；切片名必须能区分源集与目标集。训练只使用源集，训练期间不读取目标集用于验证，训练结束后测试目标集。输出目录为 `seed_<seed>/<源集>_to_<目标集>/`。

表达监督变换设置在 `training.image.expression_transform`，HER2ST 为 `log1p`。MSE 权重为 `training.image.loss_weights.gene_prediction`，相关性权重为 `training.image.loss_weights.gene_correlation`。将后者设为 `0.0` 即关闭相关性监督。当前 batch 中每个 slide 组的最少有效 spot 数设置在 `training.image.gene_correlation_min_spots`，默认 8，最低按 3 处理。相关性标准差软下限设置在 `training.image.gene_correlation_std_floor`，默认 0.02。基因数从训练表达矩阵读取。

表达模板数量设置在 `configs/model/base.yaml` 的 `image.tissue_state_count`：

```yaml
image:
  tissue_state_count: 32  # 状态数 K；至少为 2，全部参与训练
```

`tissue_state_count` 必须为至少 2 的整数，共享状态数 K 与其相等。模板 logits 在模型构建时按 `Normal(0, 0.5)` 初始化，使用本次训练的随机种子 `seeds`。相同训练 seed 可以复现初始化；不同训练 seed 会得到不同模板。模板逐行 softmax 后用于表达解码，普通模型与 EMA 从相同权重开始。

学习率在每次优化器更新后按 step 调度：先从 `0.1 × lr` 线性预热到 `lr`，再余弦衰减到 `0`。`training.image.warmup_steps` 只表示预热占总优化器更新次数的比例，取值范围 `[0, 1]`，实际预热步数为 `ceil(总步数 × warmup_steps)`；例如 `0.1` 表示前 10% 的训练步数用于预热，`0` 表示不预热，`1` 表示全程线性预热。总步数取 Lightning 的实际估计，考虑梯度累积、batch 限制和 `max_steps`。例如 2560 samples/epoch、batch size 16、100 epochs、`warmup_steps=0.1` 在单 GPU、无梯度累积时对应总计 16000 step、预热 1600 step。学习率不再使用 `warmup_epochs`；整片动态传播的周期和预热仍按 epoch 计数。

进度条通过 `training.image.progress_every_n_steps` 控制训练指标的显示间隔，默认每个优化器 step 刷新一次，首个 step 和每轮末尾也刷新。显示当前 batch 的 Loss、Dice、IoU、表达 MSE 损失（`gene_prediction`）、正则化相关性损失（`gene_correlation`）、有效 spot 数（`gene_spots`），以及全局 step 和科学计数法显示的当前学习率；表达损失仅显示已启用的项。梯度累积期间按优化器更新次数计数，epoch 汇总日志继续保留，验证指标仍在验证时更新。

当前默认模型设置：K=32、`tissue_state_dim=32`。G 从当前 fold 的训练表达矩阵读取。模板权重保存在 `tissue_state_decoder.template_logits`，对每行执行 softmax 得到表达模板。类别倾向通过 `tissue_state_decoder.class_logits.sigmoid()` 查看。最终模型保存状态总数 K，可通过保存的配置与权重独立重建。

在 `configs/model/base.yaml` 中设置 `image.tissue_state_count` 以确定共享状态数 K；`configs/model/her2st.yaml` 中的 `tissue_state_dim` 是状态编码器的隐藏通道数，与模板数量 K 不同。

训练损失包含 `segmentation`、`gene_prediction`、`gene_correlation`、`state_balance`、`template_orthogonal` 和 `template_entropy`，分别控制分割 BCE + Dice、表达 MSE、相关性损失、状态用量、模板去相关和模板熵。表达单任务将分割权重设为 0；共享状态的分割单任务将表达 MSE/PCC 都设为 0，仍构造相同状态和模板参数，没有表达监督时不能将模板解释为学到的分子模式。

## 整片动态传播

配置位于 `configs/dynamic/base.yaml`，HER2ST 通过 `configs/dynamic/her2st.yaml` 继承：

```yaml
enabled: true
warmup_epochs: 5
interval_epochs: 1
save_intermediate: true
```

`warmup_epochs` 指定首次更新的 epoch，当前为 5：前五轮使用离线伪标签，第 5 个 epoch 结束后首次传播，第 6 个 epoch 开始使用新监督。之后按 `interval_epochs` 更新，没有轮数上限，与验证周期和学习率预热独立。超像素固定使用当前标签网格的 `snic/superpixels.npy`，例如 2 mpp 使用 `label_mpp/2mpp/<切片名>/snic/superpixels.npy`。每张切片首次更新时缓存图拓扑、标准化后的 Lab 节点均值、吸附后的 spot 与种子投票结果、可求解节点和滑窗权重和；恢复回调状态时重建这些缓存，训练结束后释放。每轮重新计算当前模型的节点特征、边权和 RW 概率。

仅处理当前 fold 中有类别种子的训练切片，使用普通模型当前的 F进行整片滑窗推理。传播图使用分别标准化的 Lab/F 节点均值，`learned_weight=1.0` 按通道数平衡 F 与 Lab，`beta=1.0` 控制边权核。类别种子按 `max_snap_distance_um=100.0` 吸附到组织边缘，冲突种子不固定节点。无种子的连通分量保持不确定。超像素的尺寸和紧凑度由离线预处理配置决定。

有效更新只提交癌概率和 0/1/255 伪标签，并刷新候选训练窗口，超像素数组始终固定；只记录覆盖率比和重叠区翻转率，不设质量接受门槛。概率非有限或没有可用训练窗口时保留上一轮监督。更新留在训练内存中，不写回离线数据。动态回调由 `dynamic.enabled` 控制，独立于损失权重；分割权重为 0 时仍生成传播结果，启用分割训练需设置 `training.image.loss_weights.segmentation`。

动态传播不再分配或读写整片 H×W×C 特征场。每个窗口的 F 按原 Hann 权重除以逐像素的总权重后，在预测设备上用 FP64 累加到节点，最后按节点像素数求均值，仅将节点表传回 CPU。窗口位置、步长与融合权重保持原定义，F、概率和超像素统一使用配置的预测网格；浮点求和顺序变化可能带来微小数值差异。RW 前向只计算 F 和分割 logits，跳过表达状态的 exp、幅度头和模板归一化；dense 表达可视化仍使用原来的整片系数拼接。

每次有效更新仅保存下面两个文件到 `<fold>/dynamic_pseudo_labels/epoch_0005/A1/` 等目录，`save_intermediate: false` 关闭保存：

| 文件 | 内容 |
| --- | --- |
| `overview.jpg` | 四面板：H&E 与种子、模型概率、RW 概率、实际提交的伪标签 |
| `metadata.json` | 覆盖率、翻转率、候选窗口数、固定超像素数量及传播参数 |

## 患者隔离与混合训练

HER2ST 在 `dataset.patient_names` 中填写患者字母，按列表顺序执行患者留出。表达切片从 `data_root/samples/*/gene_counts.tsv.gz` 自动发现，只使用所选患者；存在 `masks/training_mask.png` 的切片用于分割监督和评估，无需维护 `all_names` 或 `expression_names`。当前数据自动识别为 36 张表达切片、8 张标注切片。例如留出 C 患者时，C1–C6 全部排除在训练之外，C1 评估分割，C1–C6 都评估表达。表达预处理直接使用患者名单，按患者名保存 `gene_expression/patient_loocv/test_C.h5ad`，对应实验目录为 `seed_<seed>/test_C/`，不依赖该患者是否有分割标注。

每个表达文件保存全部切片在当前 fold 基因集合上的表达。`uns` 保留 `train_names`、`test_names`、`test_patient`、选基因和归一化参数，以及评估需要的 `missing_selected_genes_by_slide`；不再写入旧任务名、测试切片名、分割/表达名单及重复的缺失基因汇总。归一化前总 counts 继续保存在 `obs['normalization_total_counts']`，用于数据检查。

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

训练日志分别记录未加权的 `train/gene_prediction`（MSE）和 `train/gene_correlation`（平均 1-regularized_PCC），后者越低越好；没有可计算的相关性项时为 0。总损失 `train/loss` 使用配置权重。另记录 `train/gene_spots`、启用的分割损失、整片传播覆盖率和翻转率记录。共享状态额外记录逐位置状态熵 `train/tissue_state_entropy`（按全部像素统计）、valid 像素上平均状态用量的熵 `train/tissue_state_usage_entropy` 和平均幅度 `train/expression_amplitude_mean`；两个熵均使用自然对数，取值范围约为 0 到 ln(K)。`train/state_balance` 是组织掩膜上的最低用量损失（所有状态用量达到 alpha/K 时为 0，无固定上界），只有权重非 0 时进入 `train/loss`；`train/state_effective_states = exp(H_tissue)` 是同一掩膜上的有效状态数。注意三种统计的掩膜不同：位置熵用全部像素，usage 熵用 valid 像素，`state_balance`/`state_effective_states` 用组织像素，不能直接横向比较。低用量熵提示少数状态占主导，高位置熵可能表示状态混合接近均匀，不能只凭熵判断质量；最低用量损失也可能被“每个像素都均匀混合”的退化解降低。验证使用 EMA 模型，最终保存和测试使用普通模型。验证表达 MSE、RMSE、PCC 的定义保持不变，训练相关性项使用 batch 内按 slide 中心化的残差，验证 PCC 使用整片或整患者统计。

每个 fold 保存 `config.yaml`、`split.json`、`metrics.csv`、`validation/`、`final_model.pt`、`test/<slide>/` 和 `metrics.json`。训练日志 `metrics.csv` 直接保存在 fold 目录。整片传播有效更新时额外保存 `dynamic_pseudo_labels/epoch_<轮次>/<训练切片>/`。有分割评估的切片保存概率图和掩码；表达评估保存 `gene_predictions.npz`。

Lightning 在 `fit()` 结束时会将模型移回 CPU。最终测试保存权重后，将普通模型恢复到 `trainer.strategy.root_device`，并复用 `trainer.precision_plugin.forward_context()`，使测试设备与计算精度和训练验证一致。开始测试时打印实际设备与精度；当前 GPU 配置使用 BF16 混合精度，参数和保存的权重仍为 FP32。

每轮验证结束后，`training.image.gene_visualization_top_n` 控制最高 PCC 基因的空间可视化，默认 5，设为 0 关闭。将当前留出患者所有有效 spot 合并，按逐基因 PCC 降序选择前 N 个；PCC 的观测掩码和常量排除规则与验证指标一致，可计算基因少于 N 时只显示实际可用数量。单张切片验证也支持。HER2ST 默认每个 epoch 验证一次；调整 `val_every_n_epochs` 后，可视化随验证周期生成。

结果位于 `validation/epoch_0001/gene_expression/`：`top_genes.csv` 保存该轮排名，`metadata.json` 保存排名口径、归一化范围和每张切片的 PCC。每张留出切片的每个入选基因只保存一张 `01_<gene>_dense_comparison_normalized.png`，三列分别为归一化的 spot 真值、spot 预测和 dense 预测。图像使用 EMA 验证结果，标题同时给出 pooled PCC 和该切片自身的 PCC。

对每个基因，真值、spot 预测和 dense 预测分别按所有留出切片的 min/max 映射到 0～1；同一来源的不同切片使用统一范围，保留切片间的相对差异。未观测 spot 不参与范围估计或绘图，dense 范围仅统计组织像素，常量范围映射为 0。色图使用 viridis，组织外显示白色。原始 min/max 记录在 `metadata.json` 的 `normalization` 和 `dense_normalization` 中，切片记录包含 `dense_mpp`、`dense_shape` 和唯一图片路径。归一化仅作用于绘图数组，PCC 排名、RMSE 和训练目标均使用原始表达。

dense 列来自滑窗融合后的原生预测网格，HER2ST 当前为 2 mpp，展示时缩放到对照面板；spot 真值仍只显示实际测量值。HER2ST 的 dense 绘图使用逐格 `log1p(线性预测)`，spot 预测使用 `log1p(圆盘内线性均值)`。当前共享状态模型仅暂存 CPU 状态系数及模板，排名后只解码入选基因；绘图结束即释放稠密缓存。绘图不增加模型推理；设置 `gene_visualization_top_n: 0` 时不缓存或生成这些图。

`final_model.pt` 包含图像模型权重、构造参数、滑窗设置和有序基因名称，可以重建模型并单独推理。

模型配置以 `configs/model/base.yaml` 和实验覆盖配置为准，构造函数不再提供隐式架构默认值。
训练 batch 只携带所需的监督与 spot 聚合索引；整片类别种子和超像素保留在 Dataset/Slide 中供动态传播使用。
概率、特征与表达系数共用同一份窗口范围和 Hann 权重和，推理仍保留 FP64 流式聚合、预取及文件映射缓冲。
独立加载新产物时使用 `SegmentationModel(**artifact["model_config"])` 后加载 `state_dict`。
旧产物若缺少 `drop_path`，需补入当时构造函数的值 `[0.1, 0.1, 0.2, 0.3]`；其余架构参数应取旧产物中的值。

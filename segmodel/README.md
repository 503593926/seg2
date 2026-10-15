# segmodel

当前唯一实验入口为 `her2st`：保持 MoMu 条件 U-Net，在其输出特征上预测共享软组织状态，再解码分割与基因表达，使用患者隔离和混合监督训练。表达模板支持随机、NMF、K-means 及 NMF/K-means 各半的混合初始化，并可选择继续训练或固定；状态预测与类别倾向仍随机初始化并参与训练。

## 网络与监督

```text
H&E + 预计算基础模型特征
              ↓
       ConvNeXt U-Net + MoMu
              ↓
              F（96 通道，7 mpp）
                       ↓
              状态编码器（1×1 Conv，96→96，GELU）
                       ↓
          软状态 A（K=32，逐位置 softmax）与幅度 u
                 ├→ P = Σ_k A_k sigmoid(q_k) → 分割
                 └→ E_g = u Σ_k A_k B_kg → 非负线性表达图
```

MoMu 的投影、Top-k 路由、通道调制和空间调制保持原结构。状态编码器仅使用一层 1×1 卷积和 GELU，将 F 从 96 通道映射到 96 通道，再由两个 1×1 头分别预测 K 通道状态和 1 通道幅度。每个位置的状态权重非负且和为 1。全局表达模板 B 为 K×G，每行由模板 logits 的 softmax 得到，默认可训练，也可固定；类别倾向 q 为 K 个可学习 logits。两种输出必须使用同一张状态图 A，没有独立分割捷径或分割修正头。

幅度 u = G×softplus(amplitude_logits)，使初始化的逐基因表达保持 O(1)；由于模板逐行归一化，u 等于该位置所选 G 个基因预测表达之和，不代表真实细胞密度或原始 UMI 总数。类别 logits 使用 logsumexp 计算混合类别概率的 log odds，避免 clamp 概率截断梯度；模板混合与相关损失使用 FP32，兼容 bf16-mixed。

表达损失更新幅度、状态编码器和 MoMu，并在模板可训练时更新模板；分割损失更新类别倾向、状态编码器和 MoMu。共享状态实现任务耦合，但不会保证互相增益。学习到的状态也不能直接认定为细胞类型或真实细胞比例。7 mpp 是输出网格间距，spot 监督不能证明恢复了真实的 7 mpp 分子变化。

表达头输出当前训练 fold 所选基因的稠密非负线性相对表达。对每个有效 spot，在其圆盘内仅对组织像素求均值，再 `log1p`，与真实 CP10k + log1p 表达计算 MSE 和逐基因相关性损失。真实表达只用于监督；表达损失独立于类别种子和 RW 图是否可解。

RW 在每个训练步中使用 crop 内全部有效类别种子，按 SNIC 超像素邻接、Lab 颜色和 F 的节点均值构造边权，以 Jacobi 预条件 CG 求解软标签。训练调用停止梯度，分割损失不通过目标侧更新 RW。只监督组织内概率不高于 0.2 或不低于 0.8 的位置，保留软概率数值。

训练总损失对启用项按 `training.image.loss_weights` 加权。HER2ST 默认同时训练分割和表达：

```text
L = 1.0 × RW_seg(P)
  + 1.0 × MSE + 0.5 × mean(1 - PCC)
```

仅有表达的切片计算表达损失；有类别种子的切片还计算分割损失。分割输出的系数由 `propagation` 控制，当前默认为 1.0。以上权重是联合训练的起点，需要分别比较单任务分割和单任务表达基线才能判断是否互相增益。

PCC 在同一 crop 内沿 spot 维度逐基因计算，再对有效的 crop–基因项取平均，不跨切片混合相关性。每个 crop 默认至少需要 8 个有效 spot；少于该数量或真值标准差不超过 1e-6 的基因不贡献相关性损失，仍参与 MSE。常数预测不会被排除：对非恒定真值，相关性项为 1，并保留有限梯度。计算使用 FP32，可配合混合精度训练。

RW 目标继续使用 MoMu 特征 F；软状态和预测表达图不参与 RW 边权计算。

## 训练吞吐优化

共享状态结构训练时先在 spot 圆盘内聚合 `u × A`（K 通道），再乘 K×G 表达模板，得到 spot 的线性表达预测，随后计算 log1p、MSE 和 crop 内 PCC。由于模板不随空间位置变化，这与先解码稠密表达再求均值在数学上等价，避免训练时分配 B×G×H×W 基因场及其反向中间张量。普通前向、验证、整片推理和 checkpoint 接口继续返回稠密表达。

数据加载 worker 在几何增强后准备组织圆盘的像素索引、归一化权重和 RW 节点/边/种子/连通分量。GPU 一次聚合整个 batch 的 spot；RW 仍使用当前前向的颜色和 F 计算边权，不缓存随模型变化的特征。CUDA PCG 使用脚本循环降低 Python 调度开销，每步检测并冻结收敛或失效的系统，每 8 步读取一次退出状态；最大迭代次数和容差保持原配置。CPU 求解和算子内部特征梯度接口保留。

HER2ST 训练 DataLoader 当前配置为：

```yaml
num_workers: 8
persistent_workers: true
prefetch_factor: 1
pin_memory: false
```

`num_workers=0` 时自动关闭持久化和预取。采样器把 epoch 与样本索引一起发送给 worker，避免预取使用旧的 epoch；有标注切片的超像素图使用共享存储，动态 SNIC 原地更新，因此 worker 复用后仍能读取新拓扑。几何旋转和翻转先建立 NumPy 视图，最后合并成一次连续内存复制。训练结束时显式释放 worker，避免连续运行多个 fold 时保留整片数据。

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
| `modules/propagation.py` | RW 图构造、类别种子投票与 PCG 求解 |
| `modules/spot_pooling.py` | 稠密表达在 spot 圆盘内的组织均值 |
| `training/segmentation_module.py` | 损失接线、优化器、学习率、EMA 和验证日志 |
| `training/dynamic_snic.py` | 定期用当前 F 重划分 SNIC，并保存整片 RW 诊断 |
| `training/losses.py` | 表达 MSE/逐基因 PCC、软标签 BCE + Dice 及可选图像/PFM 损失 |
| `inference/` | 滑窗拼接、分割/表达指标和结果可视化 |

## 配置入口

`configs/config.yaml` 默认选择 `experiment: her2st`，按原方式在 IDE 中运行 `main.py`。`configs/experiment/` 只保留 `her2st.yaml`，负责组合下面的配置文件：

| 配置文件 | 内容 |
| --- | --- |
| `configs/dataset/her2st.yaml` | 36 张表达切片、8 张标注评估切片、患者隔离和混合采样 |
| `configs/model/her2st.yaml` | 共享状态数、隐藏维度及旧结构对照开关 |
| `configs/training/her2st.yaml` | HER2ST 表达 MSE/PCC 与分割损失权重 |
| `configs/augment/her2st.yaml` | 数据增强 |
| `configs/propagation/her2st.yaml` | RW 软标签传播、动态 SNIC 开关与更新周期 |

各组的通用参数仍位于对应 `base.yaml`。

表达监督变换设置在 `training.image.expression_transform`，HER2ST 为 `log1p`。MSE 权重为 `training.image.loss_weights.gene_prediction`，相关性权重为 `training.image.loss_weights.gene_correlation`。将后者设为 `0.0` 即关闭相关性监督。最少有效 spot 数设置在 `training.image.gene_correlation_min_spots`，默认 8，最低按 3 处理。基因数从训练表达矩阵读取。

表达模板初始化设置在 `configs/training/base.yaml` 的 `image.template_init.method`，支持 `random`、`nmf`、`kmeans` 和 `mix`。训练入口在每个 fold 开始时自动生成或读取缓存，无需单独运行脚本：

```yaml
template_init:
  method: nmf  # kmeans：聚类均值；mix：一半 NMF、一半 K-means；random：随机初始化
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

`mix` 将状态数 K 分成 `ceil(K/2)` 个 NMF 成分和 `floor(K/2)` 个 K-means 中心；例如 K=32 时为 16+16，K=33 时为 17+16。两种算法都在同一份经过上述过滤的训练表达上拟合，分别生成所需数量的模板，再按 NMF 在前、K-means 在后的顺序合并，逐行归一化并应用相同 smoothing。`max_iter` 分别限制两种算法的迭代次数，`kmeans_n_init` 控制其中的 K-means 拟合。混合模式使用独立缓存，记录两种模板的数量和各自的拟合诊断。

`template_init.trainable` 默认 `true`，模板随训练更新；设为 `false` 时，在创建 EMA 和优化器前冻结 `tissue_state_decoder.template_logits`，并将其排除在优化器参数组之外。四种初始化方式均支持该开关。固定模板时，表达损失仍通过模板更新状态/幅度头和骨干，类别倾向仍可训练；EMA 的模板也保持固定。该开关不改变拟合结果和缓存键，因此切换可训练性可以复用已有模板缓存；最终模型继续保存模板权重。

拟合仅在 CPU 上执行并限制线程数。缓存位于 `project_root` 下的 `template_init.cache_dir`，按方法、训练切片、基因顺序、表达数据内容、状态数、拟合参数和软件版本区分；初始化 seed 独立于模型训练 seed，相同 fold 和参数可跨训练 seed 复用。每个使用数据初始化的 fold 还会保存 `template_initialization.npz`（初始模板、基因/训练切片顺序）和 `template_initialization.json`（方法、拟合诊断与缓存信息）。模板在创建 EMA 前写入，普通模型和 EMA 从同一表达模板开始；最终模型仍按原接口保存和独立推理。

学习率在每次优化器更新后按 step 调度：先从 `0.1 × lr` 线性预热到 `lr`，再余弦衰减到 `1e-6`。`training.image.warmup_steps` 只表示预热占总优化器更新次数的比例，取值范围 `[0, 1]`，实际预热步数为 `ceil(总步数 × warmup_steps)`；例如 `0.1` 表示前 10% 的训练步数用于预热，`0` 表示不预热，`1` 表示全程线性预热。总步数取 Lightning 的实际估计，考虑梯度累积、batch 限制和 `max_steps`。例如 2560 samples/epoch、batch size 16、100 epochs、`warmup_steps=0.1` 在单 GPU、无梯度累积时对应总计 16000 step、预热 1600 step。学习率不再使用 `warmup_epochs`；动态 SNIC 的周期和预热仍按 epoch 计数。

进度条通过 `training.image.progress_every_n_steps` 控制训练指标的显示间隔，默认每 100 个优化器 step 刷新一次，首个 step 和每轮末尾也刷新。显示当前 batch 的 Loss、Dice、IoU、表达 MSE 损失（`gene_prediction`）、相关性损失 1−PCC（`gene_correlation`）、有效 spot 数（`gene_spots`），以及全局 step 和科学计数法显示的当前学习率；表达损失仅显示已启用的项。梯度累积期间按优化器更新次数计数，epoch 汇总日志继续保留，验证指标仍在验证时更新。

当前默认模型设置：`tissue_state_count=32`、`tissue_state_dim=96`、`joint_prediction=false`。G 从当前 fold 的训练表达矩阵读取。模板 `tissue_state_decoder.template_logits` 和类别倾向 `tissue_state_decoder.class_logits` 随模型权重保存，可分别 softmax/sigmoid 后查看。

在 `configs/model/her2st.yaml` 中直接修改参数：

| 结构 | `tissue_state_count` | `joint_prediction` |
| --- | --- | --- |
| 共享软状态（当前默认） | `32`（也可比较 16/64） | `false` |
| 原 P₀ 条件/反馈结构 | `0` | `true` |
| 原分割 + 普通表达头 | `0` | `false` |

共享状态数必须至少为 2，两个联合结构互斥。`tissue_state_dim` 是编码器的隐藏通道数，与模板数量 K 不同。启用共享状态时，`phenotype_dim`、`molecular_dim`、`detach_phenotype`、`use_seg_condition` 和 `use_gene_feedback` 均不参与计算。

共享状态的表达单任务对照将全部分割损失设为 0；分割单任务对照将表达 MSE/PCC 都设为 0。后者仍会构造相同状态和模板参数，便于比较；没有表达监督时不能将模板解释为学到的分子模式。

下面两个交互开关用于原 P₀ 条件/反馈结构，需要先设置 `tissue_state_count: 0`、`joint_prediction: true`。该结构的初始/最终 RW 权重可设回 `propagation: 1.0`、`propagation_initial: 0.5`：


| 对照 | `use_seg_condition` | `use_gene_feedback` |
| --- | --- | --- |
| 共享 F 的两个独立预测分支 | `false` | `false` |
| 仅 P₀ → 表达 | `true` | `false` |
| 仅 Z → 分割 | `false` | `true` |
| 完整条件/反馈结构 | `true` | `true` |

`use_seg_condition: false` 将类别条件置零，保留表达编码器的层数、通道数和参数结构。`detach_phenotype: true` 仍读取 P₀，只阻断其梯度，不能代替这个开关。`use_gene_feedback: false` 跳过修正头，最终分割严格等于 P₀，导出的修正门控和残差为零；修正头参数保留，已训练的修正权重也不会生效。两个开关都会随 `final_model.pt` 保存。

关闭反馈后 P₁ 等于 P₀，`propagation` 与 `propagation_initial` 的监督相同，系数 1.0/0.5 对应分割总系数 1.5。比较四种交互模式时保持损失权重一致。共享编码器仍同时接受两个任务的梯度，关闭两个开关只关闭显式交互。

原条件/反馈结构做表达单任务对照时，关闭两个交互开关并将全部分割损失权重设为零，保留表达 MSE/PCC。做“相同联合结构但无表达监督”的容量对照时，保留 `joint_prediction: true`，将 MSE/PCC 权重都设为零；基因数仍由训练表达矩阵读取，不必手填。`tissue_state_count: 0`、`joint_prediction: false` 使用原分割路径和可选的普通线性表达头，表达分支深度不同，不能直接作为仅关闭 P₀ 条件的对照。

RW 配置位于 `configs/propagation/base.yaml`，包括核参数、求解精度和监督阈值。内部可微计算接口仍保留，当前训练固定停止目标梯度；门控幅度为 0，不训练传播边权。

## 动态 SNIC 与 RW 可视化

HER2ST 默认开启动态 SNIC。在 `configs/propagation/her2st.yaml` 设置：

```yaml
dynamic_snic:
  enabled: true
  warmup_epochs: 40
  interval_epochs: 20
```

完成第 40 轮训练后第一次更新，随后在第 60、80、100…轮结束时更新。第 40 轮生成的新划分从第 41 轮训练开始使用。这里的预热只控制 SNIC，与学习率预热独立；更新不依赖验证周期。

每次仅处理有分割监督的训练切片：使用普通模型当前的 MoMu 特征 F（显式指定 `feature_key="features"`）整片滑窗推理，将 F 在组织内逐通道标准化并与 Lab 颜色组合，重新运行 SNIC。不会使用软状态 A、预测基因图、真实表达或留出患者数据。通用参数在 `configs/propagation/base.yaml` 的 `dynamic_snic` 下：`superpixel_size_um=64.0`、`compactness=10.0`、`learned_weight=1.0`。最后一项仅调节 SNIC 特征，RW 的同名权重仍取外层配置。

新超像素图原地更新训练共享内存中的 `slide.superpixels`，下一轮 crop 使用新拓扑，RW 继续逐 crop 实时生成停止梯度的目标。不写回 `data/` 的预处理文件，不修改原始类别种子、表达目标或采样掩码。持久化 DataLoader worker 通过共享存储读取更新后的划分。关闭 RW 或将两项 RW 分割权重都设为零时，不注册动态 SNIC 回调。

同一次更新会在新划分上运行现有 RW 求解器，并保存到：

```text
<fold>/soft_labels/epoch_0040/A1/
<fold>/soft_labels/epoch_0060/A1/
...
```

| 文件 | 内容 |
| --- | --- |
| `superpixels.npy`、`superpixel_labels.png` | 新超像素编号数组及彩色图 |
| `superpixel_boundary_overlay.png` | H&E 上的红色 SNIC 边界；label MPP ≥ 4 时细线化，与静态 `snic/snic_boundary_overlay.png` 一致；PNG 无损保存 |
| `soft_probability.npy`、`soft_probability.png` | RW 连续概率；NPY 为原始 float32，PNG 为 uint16，除以 65535 还原 |
| `soft_probability_heatmap.png` | RW 概率热图，组织外白色 |
| `soft_pseudo_mask.png`、`soft_pseudo_mask_overlay.jpg` | 高置信 RW 伪标签；0 非癌、1 癌、255 忽略；蓝色非癌、红色癌 |
| `supervised_mask.png` | 白色为 RW 监督范围，黑色为忽略区域 |
| `spot_seeds_overlay.jpg` | 当前训练类别种子 |
| `model_probability.png`、`model_probability_heatmap.png` | 同一次推理的模型分割概率，供对照 |
| `overview.jpg`、`metadata.json` | 六面板总览、超像素数量、阈值、覆盖率和 PCG 等诊断 |

可视化的 RW 使用整片 F、全部该切片训练类别种子，以及与训练相同的求解器参数，在 CPU 上计算，避免额外占用整片图的 GPU 显存。训练仍在各 crop 上构图，两者的种子范围和标准化统计量不同，所以整片图用于诊断，不等于所有训练 crop 目标的拼接。若整片图不可解，保存的伪标签全部为忽略，元数据标记 `rw_solved=false`。

## 患者隔离与混合训练

HER2ST 的 36 张表达切片按患者划分；8 张有分割标注的切片决定 8 个评估 fold。例如留出 C 患者时，C1–C6 全部排除在训练之外，C1 评估分割，C1–C6 都评估表达。

其他患者的有标注切片和仅有表达切片混合采样，比例由 `dataset.labeled_sampling_fraction` 控制；设为 `null` 时从全部训练切片等概率抽取。仅表达切片没有类别种子，不贡献分割监督，但其有效 spot 参与表达损失。基因选择和归一化基因集合由训练患者决定，留出患者不参与选择。

## 指标与输出

验证与最终测试均以 H&E 和预计算图像特征为输入，不需要真实表达或类别种子。

- 分割：Dice、IoU、HD95。
- 原条件/反馈结构额外报告初始分割指标和 `refinement_dice_gain`（最终 Dice 减去初始 Dice）；共享状态结构不报告这些指标。
- `gene_prediction`：有效 spot × 基因观测上的表达 MSE。
- `gene_prediction_rmse`：表达 MSE 的平方根。
- `gene_prediction_pcc`：逐基因跨 spot 计算 PCC，再对可计算的基因取均值。
- `gene_prediction_pcc_max/min`：上述有效基因 PCC 的最大/最小值。
- 患者表达主指标合并该患者全部切片的有效 spot；同时保存各切片结果和宏平均。

常数预测、常数真值及观测不足的基因不参与 PCC；缺失基因值不当作真实零表达。

训练日志分别记录未加权的 `train/gene_prediction`（MSE）和 `train/gene_correlation`（平均 1-PCC），后者越低越好；没有可计算的相关性项时为 0。总损失 `train/loss` 使用配置权重。另记录 `train/gene_spots`、启用的分割损失、RW 的 PCG/覆盖率诊断。共享状态额外记录逐位置状态熵 `train/tissue_state_entropy`、batch 平均状态用量的熵 `train/tissue_state_usage_entropy` 和平均幅度 `train/expression_amplitude_mean`；两个熵均使用自然对数，取值范围约为 0 到 ln(K)。低用量熵提示少数状态占主导，高位置熵可能表示状态混合接近均匀，不能只凭熵判断质量。原结构仍记录修正门控和残差统计。验证使用 EMA 模型，最终保存和测试使用普通模型。验证表达 MSE、RMSE、PCC 的定义保持不变，训练相关性项使用 crop 内统计，验证 PCC 使用整片或整患者统计。

每个 fold 保存 `config.yaml`、`split.json`、`segmentation/`、`validation/`、`final_model.pt`、`test/<slide>/` 和 `metrics.json`。动态 SNIC 更新时额外保存 `soft_labels/epoch_<轮次>/<训练切片>/`。有分割评估的切片保存概率图和掩码；表达评估保存 `gene_predictions.npz`。仅原条件/反馈结构额外保存初始分割和修正门控。

每轮验证结束后，`training.image.gene_visualization_top_n` 控制最高 PCC 基因的空间可视化，默认 10，设为 0 关闭。将当前留出患者所有有效 spot 合并，按逐基因 PCC 降序选择前 N 个；PCC 的观测掩码和常量排除规则与验证指标一致，可计算基因少于 N 时只显示实际可用数量。单张切片验证也支持。HER2ST 默认每个 epoch 验证一次；调整 `val_every_n_epochs` 后，可视化随验证周期生成。

结果位于 `validation/epoch_0001/gene_expression/`：`top_genes.csv` 保存该轮排名，`metadata.json` 保存排名口径、色标范围和每张切片的 PCC；每张留出切片目录下有 `01_<gene>.png` 等真值/预测 spot 空间对照，以及 `overview.jpg` 总览。图像使用 EMA 验证结果，标题同时给出 pooled PCC 和该切片自身的 PCC；同一基因的真值、预测及不同切片共享从 0 到联合最大值的色标。缺失表达不画成零值。表达尺度沿用训练配置，HER2ST 为 log1p(CP10k)。绘图复用已经计算的 spot 预测与坐标，不增加模型推理。

每轮同时保存 `01_<gene>_normalized.png` 和 `overview_normalized.jpg`：对每个基因，分别在所有留出切片的有效真值和有效预测上计算 min/max，再用 `(value-min)/(max-min)` 映射到 0～1。同一来源的所有切片使用统一范围，以保留切片间的相对差异；未观测值不参与范围估计或绘图，常量范围映射为 0。该视图用于对比空间模式，原共享色标图继续保留以检查表达量偏差。各基因的原始 min/max 记录在 `metadata.json` 的 `normalization` 中。归一化仅作用于绘图数组，PCC 排名、RMSE 和训练目标均使用原始表达。

`final_model.pt` 包含图像模型权重、构造参数、滑窗设置和有序基因名称，可以重建模型并单独推理。

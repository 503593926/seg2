# 插值蒸馏

`interpolation_distillation` 分支增加训练期伪 spot 监督, 保留真实 spot 的分子蒸馏。
参考 `seg` 和 `segmentation` 中的规则网格布局, 在水平/竖直相邻实测 spot 的边中点
生成伪 spot, 并在完整 2×2 实测方格中心生成伪 spot。边中点有两个父节点,
权重各为 1/2; 方格中心有四个父节点, 权重各为 1/4。

位置和插值权重在 Dataset 初始化时按训练切片的 `x/y` 网格坐标构造, 不需要另跑
预处理脚本。不会跨缺失网格点外推。每个 crop 只保留圆盘完整位于窗口内、中心在
组织掩码内的伪 spot。父节点可以位于 crop 外, 但只来自同一张训练切片。
旋转和翻转同时作用于真实 spot 与伪 spot 坐标, 父节点索引和权重不变。

冻结 VAE 先编码真实父节点表达, 得到对角高斯后验的 μ 和 log σ²。
在伪 spot 位置, 以原有空间权重 w 构造父节点后验的局部混合, 再匹配其一阶矩
和对角二阶矩:

`mu_query = sum_k(weight_k * mu_parent_k)`

`variance_query = sum_k(weight_k * (variance_parent_k + (mu_parent_k - mu_query)^2))`

方差保留父节点自身的后验尺度, 并纳入父节点均值之间的差异。使用中心形式计算
方差以避免大均值下的相减消差, 在 FP32 下计算并将方差下限设为 1e-6。
均值公式仍是空间加权均值, 但插值目标不再只有一个点, 同时保留每维方差。
没有先平均表达再编码, 没有对插值目标做后验采样。
这里匹配的是邻域后验混合的矩, 不是独立随机变量加权平均的方差传播。

参考项目也采用 moment matching; 当前版本保留边端点/方格角点作为直接父节点,
不在边中点再引入附近方格中心。局部混合高斯是插值假设, 不是实测的伪 spot 后验;
当前只保留对角方差, 没有传播跨 latent 维度的协方差。

图像侧仍使用当前解码特征、物理 spot 半径对应的圆盘平均池化和投影头。
每个伪 spot 使用方差权重:

`weight_query = 1 / (1 + mean_dim(variance_query))`

插值损失为 `mean(weight_query * (1 - cosine(image_projection, mu_query)))`。
父节点后验尺度大或均值分歧大时, 对应插值监督减弱。权重不除以权重之和,
因此当整个 batch 的插值目标都不确定时, 总监督也会减弱。
此权重是利用后验尺度与邻域异质性的启发式, 不是经过校准的预测置信度。
VAE 的 KL 正则也会影响后验方差, 不能仅凭方差判断教师是否正确。
图像投影头仍是确定性特征头; 方差用于控制监督强度, 尚未要求图像侧预测分布。
无真实 spot 的 crop 也可通过窗口外的同切片父节点提供插值监督。
没有有效伪 spot 时, 插值损失返回可反向传播的零。

总损失额外加入 `experiment.training.image.loss_weights.interpolation * L_interpolation`。
默认权重为真实 spot `molecular: 0.5`、伪 spot `interpolation: 0.1`。
可以在原训练命令后追加:

```text
experiment.training.image.loss_weights.interpolation=0
```

关闭后不生成伪 spot, 并跳过插值教师前向。旧配置没有此权重时也按关闭处理。
仅开启插值蒸馏时仍要求 VAE 预训练轮数大于 0。推理继续只使用图像侧输入。

日志新增 `train/interpolation` 和 `train/distillation/interpolation_spots`。
前者已包含伪 spot 方差权重, 但不含配置中的总权重 0.1;
后者是当前 batch 内参与插值蒸馏的伪 spot 数。
`train/distillation/interpolation_std` 是所有伪 spot、所有 latent 维度的标准差均值;
`train/distillation/interpolation_weight` 是伪 spot 方差权重的均值。
重复采到同一伪 spot 时按 crop 视图分别计数, 与真实 spot 蒸馏的口径一致。
现有 epoch 日志按 crop batch size 加权取平均, 不是 epoch 累计数量。

轻量验证:

```bash
python -m unittest discover -s tests -p 'test_interpolation_distillation.py' -v
```

# 关系蒸馏

`relation` 分支保留逐点分子蒸馏, 并增加整个 batch 内的关系蒸馏, 包括跨切片配对。
冻结基因 VAE 继续提供后验均值 μ, 图像侧继续使用 spot 圆盘池化和现有投影头。

Dataset 返回 `slide_id` 和 `spot_ids`, 后者是当前 fold 表达矩阵的行索引。
几何增强只改变 spot 坐标, 不改变身份。同一个 spot 在重叠 crop 中的多个图像投影
先取平均, 基因目标也按身份合并。身份由 `(slide_id, spot_id)` 确定,
切片 ID 不用于限制配对。合并后, 在整个 batch 中计算所有不同 spot 的关系:

`L_relation = mean_{i<j}((cos(z_i, z_j) - cos(mu_i, mu_j))^2)`

所有有效 spot 对等权参与, 不再先按切片计算损失或对切片取平均。
不计算自身配对或同一 spot 的重复配对, 不按空间距离选择邻居。
整个 batch 少于两个不同 spot 时返回可反向传播的零损失。
关系计算使用 FP32, 教师目标不接收梯度。多 GPU 时配对范围是当前设备的本地 batch。

总损失增加 `training.image.loss_weights.relation * L_relation`。
当前配置权重是 `molecular: 0.5`、`relation: 0.0`, 关系蒸馏默认关闭。
可用 `relation: 0.1` 作为实验起点。
在现有启动命令后附加 Hydra 覆盖参数:

```text
experiment.training.image.loss_weights.relation=0       # 仅逐点蒸馏
experiment.training.image.loss_weights.relation=0.1     # 联合蒸馏
experiment.training.image.loss_weights.molecular=0 experiment.training.image.loss_weights.relation=0.1
experiment.training.image.loss_weights.molecular=0 experiment.training.image.loss_weights.relation=0
```

两项权重都为 0 时, 训练步跳过教师前向。任一项开启都要求 VAE 预训练轮数大于 0。
旧配置不含 `relation` 时按关闭处理。

日志新增 `train/relation`、`train/distillation/relation_pairs` 和
`train/distillation/relation_slides`。配对数按合并后的唯一 spot 计算, 每对只计一次;
切片数统计参与有效关系配对的所有切片, 单个 spot 的切片也能参与跨切片配对。
以上计数按现有日志机制汇总为 epoch 内 batch 均值。
它们用于确认混合切片 batch 是否提供了足够的关系监督。

验证命令: `python -m unittest discover -s tests -v`。

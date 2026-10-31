# 调参验证工具

训练与验证共用 `seg2.main.run()`，入口统一读取 `config.yaml`。训练期验证和训练后测试由 `evaluation/base.yaml` 控制，训练结束保存 `final_model.pt` 和 `final_ema_model.pt`。

需要训练期间观察验证结果时，将 `evaluation/base.yaml` 中的 `validation.enabled` 设为 `true`，然后运行 `python -m seg2.main` 或 `python -m seg2.validation`。两个入口使用同一实验、患者划分和输出目录结构。本目录的回调按周期保存临时模型文件，再启动独立测试进程。

验证开关及周期、权重选择位于 `seg2/configs/evaluation/base.yaml` 的 `validation` 下；裁块、batch size 和精度等共用参数位于同一文件的 `inference` 下。

patient_loocv 验证源数据的留出患者；transfer 跨数据集任务周期评估 target_dataset。回调使用被评估数据集自己的路径、MPP、spot 直径和特征目录。`evaluation.test.spot_classification` 同时控制验证与独立测试的 spot 二分类指标，macro 固定按切片平均。

验证回调只记录指标，训练使用固定轮数并保存末轮模型，不根据目标集成绩选择 checkpoint。若根据目标集指标挑选模型或调参，该数据集即承担验证集角色，最终评估需另留独立测试数据。

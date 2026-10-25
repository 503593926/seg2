# 调参验证工具

`python -m seg2.main` 只训练并保存 `final_model.pt`。

需要训练期间观察验证结果时，运行 `python -m seg2.validation`。本目录的回调按周期保存临时模型文件，再启动独立测试进程；训练代码没有验证钩子，也不导入测试代码。

验证设置位于 `seg2/configs/validation.yaml`。发布时可删除本目录及该配置文件，训练和独立测试仍可运行。

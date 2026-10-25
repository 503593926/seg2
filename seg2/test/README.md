# 独立测试

测试加载模型参数文件。测试输入、切片名单、网格和运行参数由测试命令指定，不读取训练配置、训练划分或 Trainer 状态。

```bash
python -m seg2.test \
  --artifact /path/to/final_model.pt \
  --data-root /path/to/test_data \
  --gene-path /path/to/test_expression.h5ad \
  --output /path/to/results \
  --slides A1 A2 \
  --feature-maps feature_ho1/tile_224_mpp_1_resize_1_overlap_1 \
  --target-mpp 1 --spot-diameter-um 100 \
  --crop-size 896 --inference-batch-size 8 \
  --device cuda:0 --precision bf16-mixed
```

Python 接口为 `seg2.test.run_test`。模型定义位于 `seg2/models/`，整片推理位于 `seg2/infer/`；二者都不依赖训练或测试流程。测试读取器位于 `data.py`，返回普通字典。

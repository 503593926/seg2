# 病理基础模型特征提取

所有提取器使用同一调用接口,模型只初始化一次:

```python
import numpy as np
from PIL import Image
from seg2.derive.pfm.uni2 import UNI2Extractor

extractor = UNI2Extractor(
    weights_root="/home/yfm/works/seg2/pretrained_model",
    batch_size=8,
    device="cuda",
    tile_size=448,
    resize=False,
    overlap_average=True,
    overlap_cls=True,
)
image = np.asarray(Image.open("he_1mpp.png").convert("RGB"))
feature_map, cls_map = extractor.extract(image)
```

## resize 与 tile_size

- `tile_size` 是从原图切取的正方形图块边长,默认 112。
- `resize=True`（默认）: 每块缩放为 224×224,保持各模型自己的插值方式和归一化参数。
- `resize=False`: 不缩放、不中心裁剪,直接输入 `tile_size×tile_size`。
- `overlap_average=True`（所有提取器均支持）: 以半个 tile 为步长
  重叠推理并平均同一空间位置的 patch 特征。
- `overlap_cls=False`（默认）: `cls_map` 保持原非重叠网格；设为 `True` 时
  保留所有半 tile 滑窗的 CLS，且要求同时启用 `overlap_average=True`。对于
  原始 `R×C` tile 网格，overlap CLS 的形状为 `(2R-1)×(2C-1)`。
  开启时 `tile_size` 和单个 tile 的 patch 网格边长必须为偶数；对较大图像，
  推理窗口数量约为非重叠模式的 4 倍。
- 不缩放时,输入边长必须能被模型 patch size 整除。448 和 672 对下表所有模型均适用。
- timm 模型启用 `dynamic_img_size=True`,保留权重对应的位置编码基准尺寸,
  推理时由模型插值位置编码。GenBio 使用自身的动态 RoPE 实现。
- 输入是 RGB `uint8` NumPy 数组。Dataset 沿用完整图块规则: 高宽应提前补齐到
  `tile_size` 的整数倍,否则右侧、底部不足一块的部分不参与提取。
- `extract(..., normalize=True)` 在输出前对每个位置的向量做 L2 归一化。
  这与图像预处理中的 mean/std 归一化是两件不同的事。

## 可用模型

| 模块 | Extractor | 权重目录 | Patch size | Patch 维度 | CLS/global 维度 |
|---|---|---|---:|---:|---:|
| `h_optimus_0` | `HO0Extractor` | `HOptimus0` | 14 | 1536 | 1536 |
| `h_optimus_1` | `HO1Extractor` | `HOptimus1` | 14 | 1536 | 1536 |
| `uni` | `UNIExtractor` | `UNI` | 16 | 1024 | 1024 |
| `uni2` | `UNI2Extractor` | `UNI2` | 14 | 1536 | 1536 |
| `virchow` | `VirchowExtractor` | `Virchow` | 14 | 1280 | 1280 |
| `virchow2` | `Virchow2Extractor` | `Virchow2` | 14 | 1280 | 1280 |
| `conch` | `CONCHExtractor` | `CONCH` | 16 | 768 | 512 |
| `genbio_pathfm` | `GenBioPathFMExtractor` | `GenBio_PathFM` | 16 | 4608 | 4608 |

CONCH 的 `cls_map` 存放官方 contrast attention pool 后、图文投影和 L2
归一化前的全局图像特征,对应 `encode_image(proj_contrast=False, normalize=False)`;
它不是 trunk 的原始 CLS token。空间特征来自视觉 trunk 的 patch tokens,
不使用失去原始空间网格含义的 caption queries。只加载这两类特征需要的视觉权重,
不需要安装整个 CONCH 包。
[官方视觉分支](https://github.com/mahmoodlab/CONCH/blob/main/conch/open_clip_custom/vision_tower.py)

CONCH 官方基准输入为 448,这里按统一接口在 `resize=True` 时使用 224;
使用 `tile_size=448, resize=False` 可保持其原生输入尺寸。
[官方模型配置](https://github.com/mahmoodlab/CONCH/blob/main/conch/open_clip_custom/model_configs/conch_ViT-B-16.json)

Virchow/Virchow2 的 `cls_map` 只保存 CLS,不拼接平均 patch 特征。
GenBio 使用本地官方 `forward_with_patches` 实现,R/G/B 分别通过单通道骨干后
拼接,因此两类特征都是 3×1536=4608 维。
[GenBio 官方说明](https://github.com/genbio-ai/genbio-pathfm)

## 输出形状与空间分辨率

设完整切块网格为 `(R, C)`、实际输入边长为 `S`、模型 patch size 为 `P`,
每块 patch 网格为 `G=S/P`:

- `feature_map`: `(R*G, C*G, D_patch)`。
- `cls_map`: 默认是 `(R, C, D_cls)`；`overlap_cls=True` 时是
  `(2R-1, 2C-1, D_cls)`。
- 两个输出都是 FP16 NumPy 数组。
- patch 网格间距为 `tile_size * 原图MPP / G`。CLS 网格间距默认是
  `tile_size * 原图MPP`，overlap CLS 则是 `tile_size * 原图MPP / 2`；
  两种 CLS 的窗口范围均为 `tile_size * 原图MPP`。

以 `tile_size=448`、1-MPP 图像为例:

| 模型 patch size | resize=True | resize=False |
|---|---|---|
| 14 | 每块 16×16 tokens,间距 28 µm | 每块 32×32 tokens,间距 14 µm |
| 16 | 每块 14×14 tokens,间距 32 µm | 每块 28×28 tokens,间距 16 µm |

`tile_size=672, resize=False` 时,每块分别为 48×48 或 42×42 tokens。
更大的 tile 会增加上下文范围和内存消耗;`batch_size` 需要按实际设备调整。

## 依赖与扩展

沿用项目的 torch、torchvision、timm、NumPy、Pillow。
GenBio 额外使用 safetensors 和 `transformers>=4.57.1,<5`,导入 `seg2/derive/pfm/GenBio_PathFM/`
中的官方 Python 文件,不执行远程代码下载。

```bash
python -m pip install 'transformers>=4.57.1,<5' safetensors
```

新模型继承 `FeatureMapExtractor`,通过 `get_patch_grid` 计算实际输入对应的网格,
通过 `create_preprocess` 构造统一的 resize 开关,再覆盖必要的 `_forward_tiles`。
该方法返回 `(patch_features, cls_features)`,形状分别为 `(B, G*G, D_patch)`
和 `(B, D_cls)`。公共实现不要求两者维度相同。

`seg2/derive/pfm_features.py` 已接入 resize 参数,输出目录和 metadata 都记录
该设置,避免相同 tile size 下两种输入模式的结果互相覆盖。

`seg2/derive/pfm_features.py` 从 `cfg.paths.weights_root` 读取权重根目录，并显式传入各提取器。直接构造提取器时必须提供 `weights_root`，各模型读取根目录下对应的模型子目录。

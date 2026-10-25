# XeHBC Rep1 数据接入

本目录只准备外部测试数据：H&E 输入、原始 cell 表达、真实细胞/细胞核边界及配准检查。

原始数据来源：
`/home/yfm/data/datasets/spatial_transcriptomics/xenium/hbc_10x_xenium_ffpe_rep1`。
使用 `raw/Xenium_FFPE_Human_Breast_Cancer_Rep1/inputs/` 的 H&E OME-TIFF 和配准 CSV，
以及 `outs/` 的原始表达矩阵、细胞坐标和边界。

在 `seg2/prepare.py` 中选择：

```python
dataset = "xenium"
ingest_stages = ("image", "cells", "qc")
derive_stages = ()
```

然后在项目根目录执行 `python -m seg2.prepare`。

默认产物写入项目的
`data/hbc_10x_xenium_ffpe_rep1/1mpp/samples/XeHBC_Rep1/`。
路径和样本名在 `seg2/configs/dataset/xenium.yaml` 中设置，由 `prepare.py` 统一加载配置。
分辨率参数集中在 `seg2/configs/prep/`，图像脚本读取
`prep.resolution` 下的 `target_mpp` 和 `tile_size`，
并据此确定相同的数据目录。

| 脚本 | 主要产物 |
|---|---|
| `image.py` | `spatial/he_1mpp.png`、`tissue_mask.png`、`canvas_mask.png`、`geometry.json` |
| `cells.py` | `cells.h5ad`、`cells_summary.json`、`spatial/cell_boundaries.parquet`、`nucleus_boundaries.parquet` |
| `qc.py` | `qc/overview.png` 整片 cell/nucleus 边界叠加图和 `summary.json` |

H&E 脚本读取 OME 物理分辨率，固定读取最高分辨率的 level 0，再一次缩放到
`prep.resolution.target_mpp`。按组织裁剪，并在右下补齐到 tile 网格。
`geometry.json` 保存全部变换：细胞微米坐标先除以 Xenium pixel size，再乘官方
H&E→Xenium 配准矩阵的逆矩阵，最后执行与图像相同的像素中心缩放及裁剪偏移。

`cells.h5ad` 保存全部 cell 和 313 个 Gene Expression feature 的原始稀疏 counts，
排除 control feature。细胞顺序按矩阵 barcode 对齐，保留原始 `cell_id`、组织/画布
标记和边界完整性；不选基因、不归一化、不按表达高低过滤细胞。
边界文件保留原始微米坐标，新增 H&E 浮点像素坐标和顶点顺序。

`qc/overview.png` 以无损 PNG 保存处理后 H&E 画布的完整尺寸，不缩放；当前为 1 mpp。
整片图绿色为 cell 边界，青色为 nucleus 边界。数值往返误差只能验证变换的自洽性，
还应查看叠加图确认实际配准。PFM 特征提取、模型训练和 cell 表达测试不在本目录执行。

参考：[10x 数据格式](https://cf.10xgenomics.com/supp/xenium/xenium_documentation.html)、
[10x 坐标变换](https://testkb.10xgenomics.com/s/article/35386990499853-How-can-I-convert-coordinates-between-H-E-image-and-Xenium-data)。

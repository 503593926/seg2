# Visium breast cancer Block A Section 1

数据集名称为 `visium_block_a_section_1`，原始目录为
`hbc_10x_block_a_section_1/raw/V1_Breast_Cancer_Block_A_Section_1/`。
直接读取其中的 `metadata.tsv`，不复制注释文件。

## 数据集准备

在 `seg2/prepare_dataset.py` 的 `main()` 中设置：

```python
dataset_name = "visium_block_a_section_1"
skip_features = False
```

顺序为 `counts → mpp → canvas → label_grid → pfm_features`。
`skip_features = True` 仅准备基础文件；每次运行都会重新生成对应阶段的输出。

- counts：读取 `filtered_feature_bc_matrix.h5` 的 Gene Expression，按 Ensembl ID 映射 HGNC，排除无法唯一映射的列，保留已映射的零表达列。复用共享 counts 处理。
- spots：读取无表头的 `tissue_positions_list.csv`，保留全部 4,992 个位置。`spot_id` 是原始 barcode，`selected` 来自 `in_tissue`，`has_expression` 按 H5 barcode 判定。组织内 3,798 个 barcode 与 counts 和注释必须完全一致，按 ID 合并。
- mpp：按 100 µm 相邻中心间距拟合六角点阵。拟合前使用 `array_x = array_col / 2`、`array_y = array_row * sqrt(3) / 2`，使两轴采用相同的间距单位。当前数据估计 MPP 为 X 约 0.366235、Y 约 0.364562 µm/px。`spot_diameter_fullres` 是显示用的估计直径，不用它除以 55 来估计 MPP。
- canvas：以只读方式映射原始 RGB `image.tif`，下采样到 1mpp，裁剪组织区域并保留所有选中 spot 的完整圆盘，再 padding 到 tile 大小。中心坐标按图像实际缩放比例转换并减去裁剪偏移。
- derive：复用 2mpp label_grid 和 PFM 特征提取，复用圆形 region 聚合。

主要基础输出为：

```text
data/visium_block_a_section_1/
├── gene_mapping.tsv
├── mpp.csv
└── 1mpp/
    ├── samples/visium_block_a_section_1/
    │   ├── gene_counts.tsv.gz
    │   └── spatial/
    │       ├── he_1mpp.png
    │       ├── spatial_1mpp.tsv
    │       ├── tissue_mask.png
    │       ├── canvas_mask.png
    │       └── spot_qc_overlay.jpg
    ├── label_mpp/2mpp/visium_block_a_section_1/spatial/
    └── feature_*/...
```

spatial 表保留原始 `array_row`、`array_col`、`source_pixel_x`、`source_pixel_y`。
`pixel_x`、`pixel_y` 是处理后 H&E 中的中心坐标，`spot_radius_px` 在 1mpp 为 27.5，2mpp 为 13.75。
组织外位置没有 counts，不补零，也不参与表达评估。

## 病理标签

注释来源：[SEDR BRCA1 metadata.tsv](https://github.com/JinmiaoChenLab/SEDR_analyses/blob/master/data/BRCA1/metadata.tsv)。
原文件 SHA256：`90aec08305574d0081e9b4705cd85f2ad4958e760f5d5a579bba35bfc469e91a`。
原始大类保存为 `annotation`，20 个细类保存为 `fine_annotation`。

| 原始大类 | spot 数 | spot_label |
| --- | ---: | ---: |
| Invasive | 1663 | 1 |
| Tumor | 827 | 1 |
| Healthy | 485 | 0 |
| Surrounding tumor | 823 | 255 |

这是本项目的二分类约定。边缘区域和未知类别标为 255，排除二分类，但可参与表达评估。
不生成像素分割真值、伪标签或超像素。

尺度参考：[10x spot 规格](https://www.10xgenomics.com/support/cn/software/space-ranger/3.1/getting-started/space-ranger-glossary)、
[Space Ranger 空间输出](https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/outputs/spatial-outputs)。

## 任务准备与训练

在 `seg2/prepare_task.py` 中设置：

```python
task_name = "her2st_to_visium_block_a_section_1"
```

在完整共有基因上归一化，仅使用 HER2ST 全切片选择最多 512 个 SVG，生成：

```text
data/task/her2st_to_visium_block_a_section_1/
├── gene_spec.json
├── her2st.h5ad
└── visium_block_a_section_1.h5ad
```

应先完成数据集和 PFM 特征准备，再生成任务 H5AD；重新运行 counts 会清理关联任务的表达文件。
在 `configs/config.yaml` 中选择 `experiment@_global_: her2st_to_visium_block_a_section_1` 后训练。
任务使用 `transfer`，模型基因数和顺序从任务 H5AD 获取。

测试先在 55 µm 捕获直径内聚合线性表达，再做 log1p 并计算 MSE/PCC。
分类独立聚合 canvas 有效圆盘内的癌症概率，仅对标签 0/1 计算 AUC/AP/F1/Acc。
表达沿用已有圆形区域的 tissue mask 规则，macro 固定按切片平均。

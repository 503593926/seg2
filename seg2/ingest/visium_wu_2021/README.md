# Wu et al. 2021 Visium breast cancer

数据集名称为 `visium_wu_2021`，包含六张外部测试切片。
原始目录为 `hbc_wu_2021/raw/<切片名>/`，逐切片读取 counts、坐标和 `metadata.csv`。

## 图像分辨率

当前使用未标注的 `spatial/tissue_hires_image.png`，最长边为 2,000 像素。
四张 CID 切片的原生 MPP 约为 4.4–4.5，两张 F 切片约为 7.6。
输出的 1mpp/2mpp 是重采样的物理坐标网格，原图细节仍受原生分辨率限制。
`images.pdf` 的病理着色图仅用于核对注释，不作为模型图像输入。

`mpp.csv` 每行对应一张切片：

- `image_source=hires` 记录当前图像来源。
- `fullres_mpp_x/y`：按 100 µm 邻居间距拟合六角点阵得到的 fullres MPP。
- `coordinate_scale`：原始 `tissue_hires_scalef`，将 fullres 坐标转换到 hires 图像。
- `mpp_x/y`：hires 原生 MPP，等于 `fullres_mpp_x/y / coordinate_scale`。
- `axis_*`、`affine_*`：点阵拟合及其残差信息。

空间表保留原始 fullres `source_pixel_x/y`。
生成 canvas 时先将坐标乘以 `coordinate_scale`，再按图像实际缩放比例换算，减去裁剪偏移，得到 `pixel_x/y`。
55 µm 捕获直径对应 1mpp 半径 27.5 像素、2mpp 半径 13.75 像素。

## Counts 与 barcode

本地 `filtered_count_matrix/` 中的三个 `.gz` 文件实际是明文，读取时依据 gzip 文件头判断压缩。
`features.tsv.gz` 只有一列 gene symbol；`matrix.mtx.gz` 是整数原始 counts，行对应基因、列对应 barcode。
共享 reader 同时支持真正的 gzip 文件，按矩阵维度核对 features 与 barcodes。

基因复用 HGNC symbol/alias 映射：无法唯一映射的列排除，映射到同一 HGNC 基因的列求和。
`gene_mapping.tsv` 仅保留 `source_gene`、`approved_symbol`、`hgnc_id`。
当前两个基因集合分别标准化为 20,867 和 16,903 个基因。

每张空间表保留 4,992 个位置，counts 是作者筛选后的子集。
只要求本切片 counts 与 metadata barcode 完全一致、且存在于坐标文件中，不要求 counts 等于全部 `in_tissue=1` 位置。
原始 counts 中有 10 个 spot 的 `in_tissue=0`，按组织外位置排除评估。
缺失 counts 的位置不补零。不同切片的 barcode 可重复，始终在切片内合并；任务 H5AD 的观测 ID 为 `切片名:barcode`。

## 病理标签与质控

原始 `Classification` 保存为 `annotation`，`patientid` 保存为 `patient_id`，保留 `subtype`、`nCount_RNA`、`nFeature_RNA`。

阳性 `1`：

```text
DCIS
Invasive cancer
Invasive cancer + lymphocytes
Invasive cancer + stroma
Invasive cancer + stroma + lymphocytes
Invasive cancer + adipose tissue + lymphocytes
Cancer trapped in lymphocyte aggregation
```

阴性 `0`：

```text
Adipose tissue
Lymphocytes
Stroma
Stroma + adipose tissue
Normal duct
Normal glands + lymphocytes
Normal + stroma + lymphocytes
TLS
```

`Artefact`、`Necrosis`、`Uncertain`、缺失及未知类别设为 `255`，不参与二分类。
`Artefact` 同时排除表达评估，`Necrosis`、`Uncertain`、缺失注释仍可参与表达评估。
`selected` 表示 `in_tissue=1` 且非 Artefact，表达额外要求 `has_expression`、`in_content`、共有基因总 counts 大于零，以及存在可聚合的有效像素。

| 切片 | 原始 filtered spots | 表达候选 | 二分类阳性 | 二分类阴性 |
| --- | ---: | ---: | ---: | ---: |
| 1142243F | 4784 | 4665 | 3627 | 470 |
| 1160920F | 4895 | 4847 | 3167 | 1679 |
| CID4290 | 2432 | 2419 | 2297 | 122 |
| CID4465 | 1211 | 1207 | 1131 | 76 |
| CID44971 | 1162 | 1159 | 590 | 569 |
| CID4535 | 1127 | 1102 | 782 | 246 |
| 合计 | 15611 | 15399 | 11594 | 3162 |

这些数量是几何及零 counts 质控前的候选数。
最终表达使用 tissue mask 内的有效圆盘像素；分类独立使用 canvas 支持的圆盘。

## 运行顺序

在 `seg2/prepare_dataset.py` 的 `main()` 中设置：

```python
dataset_name = "visium_wu_2021"
skip_features = False
```

阶段为 `counts → mpp → canvas → label_grid → pfm_features`。
`skip_features = True` 仅生成基础数据；所有阶段每次重新生成。
基础文件保持现有 Visium 布局，六张切片分别位于：

```text
data/visium_wu_2021/
├── gene_mapping.tsv
├── mpp.csv
└── 1mpp/
    ├── samples/<切片名>/gene_counts.tsv.gz
    ├── samples/<切片名>/spatial/...
    ├── label_mpp/2mpp/<切片名>/spatial/...
    └── feature_*/...
```

在 `seg2/prepare_task.py` 中设置：

```python
task_name = "her2st_to_visium_wu_2021"
```

完整共有基因集合由 HER2ST 全切片与 Wu 全部六张切片共同确定，当前为 11,797 个基因。
在完整集合上归一化，只用 HER2ST 全切片选择最多 512 个 SVG。
任务输出为：

```text
data/task/her2st_to_visium_wu_2021/
├── gene_spec.json
├── her2st.h5ad
└── visium_wu_2021.h5ad
```

先完成数据集及 PFM 准备，再生成任务 H5AD；重新运行 counts 会清理关联任务表达文件。
在 `configs/config.yaml` 中选择 `experiment@_global_: her2st_to_visium_wu_2021` 后训练。
任务使用 `transfer`，HER2ST 全切片训练，Wu 六张切片测试，macro 固定按切片平均。
表达平均线性预测值后做 log1p，计算 MSE/PCC；分类平均癌症概率后计算 AUC/AP/F1/Acc。

来源：[作者数据仓库](https://zenodo.org/records/4739739)、
[Wu et al. 2021](https://www.nature.com/articles/s41588-021-00911-1)。

当前默认配置已生成 HER2ST 训练 H5AD（13,618 × 512）和 Wu 测试 H5AD（15,399 × 512）。
区域质控后，1160920F 有 3 个候选 spot 在组织 mask 中没有可聚合像素，实际表达可评估 15,396 个区域；二分类可评估 14,756 个区域。

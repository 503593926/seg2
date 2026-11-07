# CytAssist FFPE human breast cancer

数据集名称为 `visium_cytassist_ffpe`，读取原始目录
`hbc_10x_cytassist_ffpe/raw/CytAssist_FFPE_Human_Breast_Cancer/`。

## 数据集准备

在 `seg2/prepare_dataset.py` 的 `main()` 中设置：

```python
dataset_name = "visium_cytassist_ffpe"
skip_features = False
```

阶段为 `counts → mpp → canvas → label_grid → pfm_features`。
`skip_features = True` 仅生成基础数据，每次运行会重新生成对应阶段的输出。

- counts：读取 `filtered_feature_bc_matrix.h5` 的基因级 Gene Expression。原矩阵为 4,992 个 spot、18,085 个基因。按 Ensembl ID 映射 HGNC，排除 35 个无法唯一映射的基因，当前保留 18,050 个。已映射的零表达列保留。
- spots：读取有表头的 `spatial/tissue_positions.csv` 和 `labels.xlsx` 的 `Visium` sheet，按 barcode 合并。`spot_id` 保留完整 barcode，`cluster_id`、`annotation` 保留原始注释。全部 4,992 个 spot 的 `in_tissue` 为 1。
- mpp：使用 OME 显微镜标定：X = 0.5475859960287621、Y = 0.5476093521857933 µm/px。按 100 µm 邻居间距拟合六角点阵，结果仅用于一致性检查（相对容差 2%）；`spot_diameter_fullres` 不用于估计物理 MPP。`mpp.csv` 的 `mpp_x`、`mpp_y` 是实际用于缩放的 OME 尺度；`axis_*`、`affine_*` 是点阵拟合结果。
- canvas：原始 TIFF 采用正常解码读取；共享 Visium 函数也支持 Block A 的只读 memmap。图像缩放到 1mpp，裁剪时保留选中 spot 的完整圆盘，并 padding 到 tile 大小。坐标使用 fullres H&E 像素坐标，不额外乘 `regist_target_img_scalef`。
- derive：复用 2mpp label_grid 和 PFM 特征提取。捕获直径为 55 µm，1mpp 半径为 27.5 像素，2mpp 半径为 13.75 像素。

六角点阵拟合和 canvas 处理位于 `seg2/ingest/utils/visium.py`，与 Block A 共用。
基础文件布局为：

```text
data/visium_cytassist_ffpe/
├── gene_mapping.tsv
├── mpp.csv
└── 1mpp/
    ├── samples/visium_cytassist_ffpe/
    │   ├── gene_counts.tsv.gz
    │   └── spatial/
    │       ├── he_1mpp.png
    │       ├── spatial_1mpp.tsv
    │       ├── tissue_mask.png
    │       ├── canvas_mask.png
    │       └── spot_qc_overlay.jpg
    ├── label_mpp/2mpp/visium_cytassist_ffpe/spatial/
    └── feature_*/...
```

## 标签约定

原始 Excel 包含 17 个 cluster 对应的 11 类生物学注释。按本项目的约定转换为二分类：

| Annotation | spot 数 | spot_label |
| --- | ---: | ---: |
| invasive | 599 | 1 |
| DCIS #1 | 344 | 1 |
| DCIS #2 | 210 | 1 |
| stromal/endothelial | 1005 | 0 |
| stromal/endothelial/immune | 594 | 0 |
| stromal | 555 | 0 |
| adipocytes | 505 | 0 |
| immune | 263 | 0 |
| myoepithelial/stromal/immune | 110 | 0 |
| mixed | 438 | 255 |
| mixed/invasive | 369 | 255 |

几何质控前，阳性 1,153 个、阴性 3,032 个、混合区域 807 个。
混合区域和未知类别排除二分类，仍可参与表达评估。
结果应注明标签来自发布的 cluster 注释。像素分割真值、超像素和伪标签不生成。

## 任务准备与训练

在 `seg2/prepare_task.py` 中设置：

```python
task_name = "her2st_to_visium_cytassist_ffpe"
```

在完整共有基因集合上归一化，只用 HER2ST 全切片选择最多 512 个 SVG。
当前 HGNC 参考下，共有基因为 11,448 个。任务结果位于：

```text
data/task/her2st_to_visium_cytassist_ffpe/
├── gene_spec.json
├── her2st.h5ad
└── visium_cytassist_ffpe.h5ad
```

先完成数据集及 PFM 特征准备，再生成任务 H5AD；重新运行 counts 会清理关联任务表达文件。
在 `configs/config.yaml` 中选择 `experiment@_global_: her2st_to_visium_cytassist_ffpe` 后训练。
任务采用 `transfer`，模型基因数与顺序从该任务 H5AD 获取。

测试复用圆形 region 聚合：表达按 tissue mask 内有效像素平均线性值，再做 log1p，计算 MSE/PCC；
癌症概率按 canvas 支持的圆盘平均，仅对 0/1 标签计算 AUC/AP/F1/Acc，macro 固定按切片平均。

来源：[10x 数据说明](https://cf.10xgenomics.com/samples/spatial-exp/2.0.0/CytAssist_FFPE_Human_Breast_Cancer/CytAssist_FFPE_Human_Breast_Cancer_web_summary.html)、
[作者数据与注释说明](https://github.com/10XGenomics/janesick_nature_comms_2023_companion)。

当前默认配置下，任务 H5AD 保存全部 4,992 个测试 spot。表达聚合有 20 个 spot 在组织 mask 内没有有效像素，实际可评估 4,972 个区域；二分类可评估 4,185 个。

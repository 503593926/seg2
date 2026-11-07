# Xenium FFPE breast cancer replicate 1

预处理入口 `seg2/prepare_dataset.py` 中选择 `dataset_name = "xenium_ffpe_rep1"`。
`skip_features = True` 仅生成基础数据；训练与测试前需要设为 `False` 提取配置中的 PFM 特征。
入口按 `counts → mpp → canvas → label_grid → regions → pfm_features` 顺序运行，每次重新生成。

- counts 从 10x H5 只读取 Gene Expression，使用 Ensembl ID 映射 HGNC。控制探针不参与表达，无法唯一映射的基因被排除。
- mpp 从 H&E OME 读取原图分辨率，从 `experiment.xenium` 读取 Xenium 像素大小，并核对配准矩阵的尺度。
- canvas 使用 H&E → Xenium 配准矩阵生成 Xenium 物理坐标系中的 1mpp 图像。细胞中心和多边形的原始单位为 µm，除以目标 MPP 后保存。canvas mask 排除未被 H&E 覆盖的区域及 padding。
- label_grid 复用现有 2mpp 图像和 mask 处理；regions 直接把细胞多边形栅格化到这个输出网格，生成 `region_map.npy`，0 为背景，正整数为原始 cell ID。边界像素重合时按 ID 顺序确定归属。没有有效输出像素的细胞不参与评估。
- 为复用已有 counts/H5AD 观测格式，spatial 表中的 `spot_id` 保存原始 cell ID，`spot_label` 保存细胞二分类标签。计算接口和结果文件统一使用 region 名称。

标签来自 `Cell_Barcode_Type_Matrices.xlsx` 的 `Xenium R1 Fig1-5 (supervised)`：
`Invasive_Tumor`、`Prolif_Invasive_Tumor`、`DCIS_1`、`DCIS_2` 为 1；明确非肿瘤类别为 0；
`Unlabeled`、`T_Cell_&_Tumor_Hybrid` 以及未知类别为 255，排除二分类评估。它们仍可参与表达评估。
这些是细胞类型注释导出的标签，不生成像素分割真值。

任务入口 `seg2/prepare_task.py` 中选择 `task_name = "her2st_to_xenium_ffpe_rep1"`。
任务在 HER2ST 和 Xenium 的 HGNC 共有基因上确定归一化分母，只用 HER2ST 选择 SVG。
`n_top_genes: 512` 为选择上限，实际数量受共有基因和 SVG 筛选结果限制。
任务结果位于 `data/task/her2st_to_xenium_ffpe_rep1/`。
应先完成数据集和 PFM 特征准备，再生成任务 H5AD；重新运行 counts 会清理关联的任务表达文件。

训练入口 `configs/config.yaml` 中选择 `experiment@_global_: her2st_to_xenium_ffpe_rep1`。
这个任务需要使用自身 H5AD 重新训练，模型基因顺序和 preprocessing_id 必须与任务一致。
测试复用同一条推理流程：先在 canvas 支持的细胞区域平均线性表达，再做 log1p，计算 MSE/PCC；
癌症概率独立按细胞区域平均，计算 AUC/AP/F1/Acc，macro 固定为切片平均。
HER2ST/STNet 的圆形 spot 表达仍使用已有 tissue mask 规则。
分类结果保存为 `region_predictions.tsv`，汇总键为 `region_classification`，表达预测 NPZ 使用 `region_ids` 和 `coordinates`。

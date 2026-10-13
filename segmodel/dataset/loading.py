import json
from pathlib import Path
from typing import cast

import anndata as ad
import cv2
import numpy as np
import scipy.sparse as sp

from segmodel.dataset.slide import Slide, TrainingSlide


def read_image(path, flag) -> np.ndarray:
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(path)
    return np.asarray(image)


def read_slide_inputs(image_path, feature_dirs, cls_scale):
    """读取训练和推理共用的图像、patch 特征与 CLS 特征。"""
    image = np.asarray(
        cv2.cvtColor(read_image(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB),
        dtype=np.uint8,
    )
    patch_maps = tuple(
        np.load(directory / "patch_map.npy", mmap_mode='r') for directory in feature_dirs
    )
    cls_maps = tuple(
        _pad_cls_map(np.load(directory / "cls_map.npy", mmap_mode='r'), image.shape, cls_scale)
        for directory in feature_dirs
    )
    return image, patch_maps, cls_maps


def fold_paths(project_root, dataset_cfg, test_names):
    if dataset_cfg.split_mode in {"loocv", "patient_loocv"}:
        data_root = project_root / str(dataset_cfg.data_root)
        names = dataset_cfg.expression_names if dataset_cfg.split_mode == "patient_loocv" else dataset_cfg.all_names
        roots = {name: data_root for name in names}
        directory = dataset_cfg.get("gene_expression_dir", "gene_expression/task1_loocv")
        gene_path = (
            data_root / str(directory) / f"test_{test_names[0]}.h5ad"
        )
    elif dataset_cfg.split_mode == "transfer":
        data_base = project_root / "data"
        ref_dataset = str(dataset_cfg.ref_dataset)
        tgt_dataset = str(dataset_cfg.tgt_dataset)
        ref_root = data_base / ref_dataset / "1mpp"
        tgt_root = data_base / tgt_dataset / "1mpp"
        roots = {name: ref_root for name in dataset_cfg.ref_names}
        roots.update({name: tgt_root for name in dataset_cfg.tgt_names})
        transfer_name = f"{ref_dataset}_to_{tgt_dataset}"
        gene_file = dataset_cfg.get("gene_expression_file", f"{transfer_name}.h5ad")
        gene_path = data_base / transfer_name / str(gene_file)
    else:
        raise ValueError(f"Unsupported split mode: {dataset_cfg.split_mode}")
    return roots, gene_path


def load_gene_adata(path, train_names):
    """读取指定切片的 spot 元数据和表达矩阵。"""
    adata = ad.read_h5ad(path, backed="r")
    try:
        available = set(adata.obs["name"].astype(str))
        missing = set(train_names) - available
        if missing:
            raise ValueError(f"{path}: expression data missing for slides {sorted(missing)}")
        mask = adata.obs["name"].astype(str).isin(train_names).to_numpy()
        gene_adata = adata[mask].to_memory()
    finally:
        adata.file.close()

    expression = gene_adata.X
    if sp.issparse(expression):
        sparse_expression = cast(sp.csr_matrix | sp.csc_matrix, expression)
        expression = sparse_expression.astype(np.float32, copy=False).toarray()
    else:
        expression = np.asarray(expression, dtype=np.float32)
    gene_adata.X = expression
    return gene_adata


def load_feature_scales(root, feature_maps, resolution_cfg):
    target_mpp = float(resolution_cfg.target_mpp)
    scales = set()
    for feature_map in feature_maps:
        metadata = json.loads((root / str(feature_map) / "metadata.json").read_text(encoding="utf-8"))
        patch_scale = round(float(metadata["patch_mpp"]) / target_mpp)
        cls_scale = round(float(metadata["cls_mpp"]) / target_mpp)
        scales.add((patch_scale, cls_scale))
    if len(scales) != 1:
        raise ValueError(f"PFM feature maps must share one grid, got {sorted(scales)}")
    return scales.pop()


def load_training_slides(roots, names, dataset_cfg, resolution_cfg, labeled_names=None) -> dict[str, TrainingSlide]:

    target_mpp = float(resolution_cfg.target_mpp)
    label_scale = round(float(resolution_cfg.label_mpp) / target_mpp)
    spot_radius_px = float(dataset_cfg.spot_diameter_um) / (2.0 * target_mpp)
    feature_maps = list(dataset_cfg.image_feature_maps)
    labeled_names = set(names if labeled_names is None else labeled_names)
    slides = {}
    for name in names:
        root = Path(roots[name])

        _, cls_scale = load_feature_scales(
            root, feature_maps, resolution_cfg
        )

        image, patch_maps, cls_maps = read_slide_inputs(
            root / "samples" / name / f"spatial/he_{target_mpp:g}mpp.png",
            [root / str(feature_map) / name for feature_map in feature_maps],
            cls_scale,
        )

        label_dir = root / "label_mpp" / name
        tissue_mask = read_image(
            label_dir / "spatial/tissue_mask.png",
            cv2.IMREAD_GRAYSCALE,
        )
        has_segmentation = name in labeled_names
        if has_segmentation:
            pseudo_dir = label_dir / str(dataset_cfg.pseudo_label_dir)
            pseudo_mask_uint8 = read_image(pseudo_dir / "pseudo_mask.png", cv2.IMREAD_GRAYSCALE)
            cancer_probability = read_image(pseudo_dir / "cancer_probability.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
            superpixels = np.load(label_dir / "snic/superpixels.npy").astype(np.int32)
        else:
            # 无分割监督: 255为未知, 0节点禁止标签传播; 表达另用tissue圆盘聚合。
            pseudo_mask_uint8 = np.full(tissue_mask.shape, 255, dtype=np.uint8)
            cancer_probability = np.zeros(tissue_mask.shape, dtype=np.float32)
            superpixels = np.zeros(tissue_mask.shape, dtype=np.int32)

        slides[name] = TrainingSlide(
            name=name,
            image=image,
            patch_maps=patch_maps,
            cls_maps=cls_maps,
            cls_scale=cls_scale,
            label_scale=label_scale,
            spot_radius_px=spot_radius_px,
            pseudo_mask_uint8=pseudo_mask_uint8,
            cancer_probability=cancer_probability,
            superpixels=superpixels,
            tissue_mask_bl=tissue_mask > 0,
            has_segmentation=has_segmentation,
        )
    return slides


def load_inference_slides(roots, names, dataset_cfg, resolution_cfg) -> dict[str, Slide]:
    """只读取 test/predict 所需的图像和离线特征。"""
    target_mpp = float(resolution_cfg.target_mpp)
    label_scale = round(float(resolution_cfg.label_mpp) / target_mpp)
    feature_maps = list(dataset_cfg.image_feature_maps)
    slides = {}
    for name in names:
        root = Path(roots[name])
        patch_scale, cls_scale = load_feature_scales(
            root, feature_maps, resolution_cfg
        )
        image, patch_maps, cls_maps = read_slide_inputs(
            root / "samples" / name / f"spatial/he_{target_mpp:g}mpp.png",
            [root / str(feature_map) / name for feature_map in feature_maps],
            cls_scale,
        )
        slide = Slide(name, image, patch_maps, cls_maps, cls_scale, label_scale)
        if slide.patch_scale != (patch_scale, patch_scale):
            raise ValueError(
                f"{name}: loaded patch scale {slide.patch_scale} does not match "
                f"metadata patch_mpp {patch_scale}px grid"
            )
        slides[name] = slide
    return slides


def _pad_cls_map(cls_map, image_shape, cls_scale):
    padding_rows = image_shape[0] // cls_scale - cls_map.shape[0]
    padding_columns = image_shape[1] // cls_scale - cls_map.shape[1]
    return np.pad(
        cls_map,
        ((0, padding_rows), (0, padding_columns), (0, 0)),
        mode="edge",
    )

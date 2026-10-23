import json
from pathlib import Path
from typing import cast

import anndata as ad
import cv2
import numpy as np
import scipy.sparse as sp

from segmodel.dataset.slide import Slide, TrainingSlide
from tools.label_grid import label_grid_root, label_scale as resolve_label_scale


def read_slide_inputs(image_path, feature_dirs):
    """读取训练和推理共用的图像、patch 特征与 CLS 特征。"""
    image_bgr = np.asarray(cv2.imread(str(image_path), cv2.IMREAD_COLOR))
    image = np.asarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    patch_maps = tuple(
        np.load(directory / "patch_map.npy", mmap_mode='r') for directory in feature_dirs
    )
    cls_maps = tuple(
        np.load(directory / "cls_map.npy") for directory in feature_dirs
    )
    return image, patch_maps, cls_maps


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


def _load_slide(root, name, dataset_cfg, resolution_cfg):
    """读取两种阶段共用的基础输入，返回切片和元数据中的 patch 步长。"""
    target_mpp = float(resolution_cfg.target_mpp)
    label_scale = resolve_label_scale(target_mpp, float(resolution_cfg.label_mpp))
    feature_maps = list(dataset_cfg.image_feature_maps)
    patch_scale, cls_scale = load_feature_scales(root, feature_maps, resolution_cfg)
    sample_dir = root / "samples" / name
    image, patch_maps, cls_maps = read_slide_inputs(
        sample_dir / f"spatial/he_{target_mpp:g}mpp.png",
        [root / str(feature_map) / name for feature_map in feature_maps],
    )
    return Slide(
        name, image, patch_maps, cls_maps, cls_scale, label_scale,
        has_segmentation=(sample_dir / "masks/training_mask.png").is_file(),
    ), patch_scale


def load_training_slides(roots, names, dataset_cfg, resolution_cfg) -> dict[str, TrainingSlide]:
    target_mpp = float(resolution_cfg.target_mpp)
    label_mpp = float(resolution_cfg.label_mpp)
    spot_radius_px = float(dataset_cfg.spot_diameter_um) / (2.0 * target_mpp)
    slides = {}
    for name in names:
        root = Path(roots[name])
        sample_dir = root / "samples" / name
        slide, _ = _load_slide(root, name, dataset_cfg, resolution_cfg)
        label_dir = label_grid_root(root, target_mpp, label_mpp) / name
        shape = (slide.height // slide.label_scale, slide.width // slide.label_scale)
        spatial_dir = sample_dir / "spatial" if slide.label_scale == 1 else label_dir / "spatial"
        tissue_mask = np.asarray(cv2.imread(
            str(spatial_dir / "tissue_mask.png"),
            cv2.IMREAD_GRAYSCALE,
        ))
        if slide.has_segmentation:
            pseudo_dir = label_dir / str(dataset_cfg.pseudo_label_dir)
            pseudo_mask_uint8 = np.asarray(cv2.imread(str(pseudo_dir / "pseudo_mask.png"), cv2.IMREAD_GRAYSCALE))
            cancer_probability = np.asarray(cv2.imread(
                str(pseudo_dir / "cancer_probability.png"), cv2.IMREAD_UNCHANGED,
            )).astype(np.float32) / 65535.0
            superpixels = np.load(label_dir / "snic/superpixels.npy").astype(np.int32)
        else:
            # 无分割监督: 255为未知, 0节点禁止标签传播; 表达另用tissue圆盘聚合。
            pseudo_mask_uint8 = np.broadcast_to(np.uint8(255), shape)
            cancer_probability = np.broadcast_to(np.float32(0), shape)
            superpixels = np.broadcast_to(np.int32(0), shape)

        if any(array.shape != shape for array in
               (tissue_mask, pseudo_mask_uint8, cancer_probability, superpixels)):
            raise ValueError(
                f"{name}: tissue, pseudo labels and SNIC must use prediction grid {shape}; "
                "run derive.get_label_mpp, derive.get_superpixel and derive.get_pseudo_label"
            )

        slides[name] = TrainingSlide(
            **vars(slide),
            spot_radius_px=spot_radius_px,
            pseudo_mask_uint8=pseudo_mask_uint8,
            cancer_probability=cancer_probability,
            superpixels=superpixels,
            tissue_mask_bl=tissue_mask > 0,
        )
    return slides


def load_inference_slides(roots, names, dataset_cfg, resolution_cfg) -> dict[str, Slide]:
    """只读取 test/predict 所需的图像和离线特征。"""
    slides = {}
    for name in names:
        slide, patch_scale = _load_slide(Path(roots[name]), name, dataset_cfg, resolution_cfg)
        if slide.patch_scale != (patch_scale, patch_scale):
            raise ValueError(
                f"{name}: loaded patch scale {slide.patch_scale} does not match "
                f"metadata patch_mpp {patch_scale}px grid"
            )
        slides[name] = slide
    return slides

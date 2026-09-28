import json
from pathlib import Path

import anndata as ad
import cv2
import numpy as np
import scipy.sparse as sp

from .slide import Slide, read_image


def fold_paths(project_root, dataset_cfg, test_names):
    if dataset_cfg.split_mode == "loocv":
        data_root = project_root / str(dataset_cfg.data_root)
        roots = {name: data_root for name in dataset_cfg.all_names}
        gene_path = (
            data_root / "gene_expression" / "task1_loocv" / f"test_{test_names[0]}.h5ad"
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
    mask = adata.obs["name"].astype(str).isin(train_names).to_numpy()
    gene_adata = adata[mask].to_memory()

    expression = gene_adata.X
    if sp.issparse(expression):
        expression = expression.astype(np.float32, copy=False).toarray()
    else:
        expression = np.asarray(expression, dtype=np.float32)
    gene_adata.X = expression
    return gene_adata


def feature_dirs(root, feature_maps, name):
    """一个切片在各 PFM 下的特征目录。"""
    return [root / str(feature_map) / name for feature_map in feature_maps]


def load_feature_scales(root, feature_maps, resolution_cfg):
    """各 PFM 的 patch / cls 步长（目标分辨率像素数）。

    各 PFM 的 patch 步长由各自 ViT 的 patch size 决定，可能不同；混用不同网格
    会破坏取窗对齐，所以在入口处要求它们落在同一张网格上。
    """
    target_mpp = float(resolution_cfg.target_mpp)
    scales = set()
    for feature_map in feature_maps:
        metadata = json.loads(
            (root / str(feature_map) / "metadata.json").read_text(encoding="utf-8")
        )
        scales.add(
            (
                round(float(metadata["patch_mpp"]) / target_mpp),
                round(float(metadata["cls_mpp"]) / target_mpp),
            )
        )
    if len(scales) != 1:
        raise ValueError(f"PFM feature maps must share one grid, got {sorted(scales)}")
    return scales.pop()


def _training_feature_maps(dataset_cfg):
    feature_maps = list(dataset_cfg.image_feature_maps)
    if not feature_maps:
        raise ValueError("image_feature_maps must not be empty")
    return feature_maps


def load_training_slides(roots, names, dataset_cfg, resolution_cfg):

    target_mpp = float(resolution_cfg.target_mpp)
    label_scale = round(float(resolution_cfg.label_mpp) / target_mpp)
    feature_maps = _training_feature_maps(dataset_cfg)
    slide_data = {}
    for name in names:
        root = Path(roots[name])

        patch_scale, cls_scale = load_feature_scales(
            root, feature_maps, resolution_cfg
        )

        slide = Slide.from_disk(
            name,
            root / "samples" / name / f"spatial/he_{target_mpp:g}mpp.png",
            feature_dirs(root, feature_maps, name),
            cls_scale,
        )

        label_dir = root / "label_mpp" / name
        pseudo_dir = label_dir / str(dataset_cfg.pseudo_label_dir)
        pseudo_mask = read_image(
            pseudo_dir / "pseudo_mask.png",
            cv2.IMREAD_GRAYSCALE,
        )
        cancer_probability = read_image(
            pseudo_dir / "cancer_probability.png",
            cv2.IMREAD_UNCHANGED,
        )
        tissue_mask = read_image(
            label_dir / "spatial/tissue_mask.png",
            cv2.IMREAD_GRAYSCALE,
        )
        superpixels = np.load(label_dir / "snic/superpixels.npy").astype(np.int32)

        for source, scale, arrays in (
            (
                "supervision",
                label_scale,
                (pseudo_mask, cancer_probability, tissue_mask, superpixels),
            ),
            ("patch map", patch_scale, slide.patch_maps),
            ("cls map", cls_scale, slide.cls_maps),
        ):
            expected_shape = (slide.height // scale, slide.width // scale)
            if any(value.shape[:2] != expected_shape for value in arrays):
                raise ValueError(
                    f"{name}: {source} does not match the "
                    f"{scale}px grid {expected_shape}"
                )

        slide_data[name] = {
            "image": slide.image,  # uint8
            "patch_map": slide.patch_maps,  # 各 PFM 的 float16 特征图
            "cls_map": slide.cls_maps,  # 各 PFM 的 float16 CLS 网格
            "patch_dims": slide.patch_dims,
            "cls_dims": slide.cls_dims,
            "patch_scale": patch_scale,
            "cls_scale": cls_scale,
            "pseudo_mask": pseudo_mask,  # uint8
            "cancer_probability": cancer_probability.astype(np.float32)
            / 65535.0,  # float32
            "superpixels": superpixels,  # int32
            "tissue_mask_bl": tissue_mask > 0,  # bool
        }
    return slide_data


def load_inference_slides(roots, names, dataset_cfg, resolution_cfg):
    """只读取 test/predict 所需的图像和离线特征。"""
    target_mpp = float(resolution_cfg.target_mpp)
    feature_maps = _training_feature_maps(dataset_cfg)
    slides = {}
    for name in names:
        root = Path(roots[name])
        patch_scale, cls_scale = load_feature_scales(
            root, feature_maps, resolution_cfg
        )
        slide = Slide.from_disk(
            name,
            root / "samples" / name / f"spatial/he_{target_mpp:g}mpp.png",
            feature_dirs(root, feature_maps, name),
            cls_scale,
        )
        if slide.patch_scale != (patch_scale, patch_scale):
            raise ValueError(
                f"{name}: loaded patch scale {slide.patch_scale} does not match "
                f"metadata patch_mpp {patch_scale}px grid"
            )
        slides[name] = slide
    return slides

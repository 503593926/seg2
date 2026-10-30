import json
from math import lcm
from pathlib import Path
from typing import Any

import anndata as ad
import cv2
import numpy as np
import scipy.sparse as sp


def read_slide_inputs(image_path, feature_dirs):
    image_bgr = np.asarray(cv2.imread(str(image_path), cv2.IMREAD_COLOR))
    image = np.asarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    patch_maps = tuple((np.load(directory / "patch_map.npy", mmap_mode="r") for directory in feature_dirs))
    cls_maps = tuple((np.load(directory / "cls_map.npy") for directory in feature_dirs))
    return (image, patch_maps, cls_maps)


def load_gene_adata(path, names):
    adata = ad.read_h5ad(path, backed="r")
    try:
        mask = adata.obs["name"].astype(str).isin(names).to_numpy()
        gene_adata = adata[mask].to_memory()
    finally:
        adata.file.close()
    expression = gene_adata.X
    if isinstance(expression, (sp.spmatrix, sp.sparray)):
        expression = sp.csr_matrix(expression, dtype=np.float32).toarray()
    else:
        expression = np.asarray(expression, dtype=np.float32)
    gene_adata.X = expression
    return gene_adata


def load_feature_scales(root, feature_maps, target_mpp):
    target_mpp = float(target_mpp)
    scales = set()
    for feature_map in feature_maps:
        metadata = json.loads((root / str(feature_map) / "metadata.json").read_text(encoding="utf-8"))
        patch_scale = round(float(metadata["patch_mpp"]) / target_mpp)
        cls_scale = round(float(metadata["cls_mpp"]) / target_mpp)
        scales.add((patch_scale, cls_scale))
    return scales.pop()


def read_sample(root, name, feature_maps, target_mpp, label_mpp) -> dict[str, Any]:
    root = Path(root)
    _, cls_step = load_feature_scales(root, feature_maps, target_mpp)
    sample_dir = root / "samples" / name
    image, patch_maps, cls_maps = read_slide_inputs(
        sample_dir / f"spatial/he_{float(target_mpp):g}mpp.png",
        [root / str(feature) / name for feature in feature_maps],
    )
    height, width = image.shape[:2]
    shapes = {values.shape[:2] for values in patch_maps}
    rows, _ = shapes.pop()
    patch_scale = height // rows
    scale = round(float(label_mpp) / float(target_mpp))
    return dict(
        name=name,
        image=image,
        patch_maps=patch_maps,
        cls_maps=cls_maps,
        cls_scale=cls_step,
        label_scale=scale,
        has_segmentation=(sample_dir / "masks/training_mask.png").is_file(),
        height=height,
        width=width,
        patch_scale=patch_scale,
        alignment=lcm(scale, patch_scale, cls_step),
        patch_dims=tuple((values.shape[-1] for values in patch_maps)),
        cls_dims=tuple((values.shape[-1] for values in cls_maps)),
    )

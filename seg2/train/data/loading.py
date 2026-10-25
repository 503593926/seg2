from math import lcm
from pathlib import Path
from typing import Any

import anndata as ad
import cv2
import numpy as np
import scipy.sparse as sp


def load_gene_adata(fold, cfg):
    data_dir = Path(fold["data_dir"])

    path = data_dir / str(cfg.dataset.gene_expression_dir) / f"{fold['name']}.h5ad"
    adata = ad.read_h5ad(path)

    mask = adata.obs["name"].astype(str).isin(fold['train_names']).to_numpy()
    gene_adata = adata[mask].copy()

    expression = gene_adata.X
    if isinstance(expression, (sp.spmatrix, sp.sparray)):
        expression = sp.csr_matrix(expression, dtype=np.float32).toarray()
    else:
        expression = np.asarray(expression, dtype=np.float32)
    gene_adata.X = expression
    return gene_adata

def read_slide_inputs(image_path, feature_dirs):
    image_bgr = np.asarray(cv2.imread(str(image_path), cv2.IMREAD_COLOR))
    image = np.asarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    patch_maps = tuple((np.load(directory / "patch_map.npy", mmap_mode="r") for directory in feature_dirs))
    cls_maps = tuple((np.load(directory / "cls_map.npy") for directory in feature_dirs))
    return (image, patch_maps, cls_maps)

def read_sample(data_dir, name, feature_maps, cfg):
    target_mpp = float(cfg.prep.resolution.target_mpp)
    label_mpp = float(cfg.prep.resolution.label_mpp)

    sample_dir = data_dir / "samples" / name
    label_dir = Path(data_dir) / "label_mpp" / f"{float(label_mpp):g}mpp" / name
    spatial_dir = label_dir / "spatial"


    image, patch_maps, cls_maps = read_slide_inputs(
        sample_dir / f"spatial/he_{float(target_mpp):g}mpp.png",
        [data_dir / str(feature) / name for feature in feature_maps],
    )

    patch_shape = patch_maps[0].shape[:2]
    patch_scale = image.shape[0] // patch_shape[0]

    cls_shape = cls_maps[0].shape[:2]
    cls_scale = image.shape[0] // cls_shape[0]

    label_scale = round(float(label_mpp) / float(target_mpp))
    alignment = lcm(label_scale, patch_scale, cls_scale)

    tissue_mask = np.asarray(cv2.imread(str(spatial_dir / "tissue_mask.png"), cv2.IMREAD_GRAYSCALE))
    has_segmentation = (sample_dir / "masks/training_mask.png").is_file()

    sample:dict[str, Any] = dict(
        name=name,

        image=image,
        patch_maps=patch_maps,
        cls_maps=cls_maps,

        patch_scale=patch_scale,
        cls_scale=cls_scale,
        label_scale=label_scale,

        has_segmentation=has_segmentation,
        alignment=alignment,
        tissue_mask_bl=tissue_mask > 0,
        spot_radius_px=float(cfg.dataset.spot_diameter_um) / (2 * target_mpp),
    )

    if has_segmentation:
        pseudo_dir = label_dir / str(cfg.dataset.pseudo_label_dir)
        sample.update(
            pseudo_mask=cv2.imread(str(pseudo_dir / "pseudo_mask.png"), cv2.IMREAD_GRAYSCALE),
            probability=np.asarray(
                cv2.imread(str(pseudo_dir / "cancer_probability.png"), cv2.IMREAD_UNCHANGED),
                dtype=np.float32,
            ) / 65535.0,
            superpixels=np.load(label_dir / "snic/superpixels.npy").astype(np.int32),
        )

    return sample



def load_training_slides(fold, cfg) -> dict:
    slides = {}
    data_dir = fold['data_dir']
    for name in fold['train_names']:
        slide = read_sample(data_dir, name, list(cfg.dataset.image_feature_maps), cfg)
        slides[name] = slide
    return slides

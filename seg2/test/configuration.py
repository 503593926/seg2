"""Build evaluation inputs from the composed experiment configuration."""

from pathlib import Path

import anndata as ad

from seg2.common.task import gene_expression_path


def evaluation_inputs(cfg, fold_name):
    external = str(cfg.task.split_mode) == "transfer"
    dataset = cfg.target_dataset if external else cfg.source_dataset
    gene_path = gene_expression_path(cfg, fold_name, target=external)
    data = ad.read_h5ad(gene_path, backed="r")
    try:
        names = list(map(str, data.uns["test_names"]))
    finally:
        data.file.close()
    return dataset, gene_path, names


def model_artifact(model_dir, weights):
    if str(weights) not in {"ema", "model"}:
        raise ValueError("Evaluation weights must be 'ema' or 'model'")
    filename = "final_ema_model.pt" if str(weights) == "ema" else "final_model.pt"
    return Path(model_dir) / filename


def evaluation_parameters(
    cfg,
    *,
    dataset,
    artifact_path,
    gene_path,
    output_dir,
    names,
    data_root=None,
    device=None,
    validation=False,
):
    options = cfg.evaluation.inference
    stage = cfg.evaluation.validation if validation else cfg.evaluation.test
    return {
        "artifact_path": str(artifact_path),
        "data_root": str(dataset.data_dir if data_root is None else data_root),
        "gene_path": str(gene_path),
        "output_dir": str(output_dir),
        "names": list(map(str, names)),
        "feature_maps": list(map(str, dataset.image_feature_maps)),
        "target_mpp": float(dataset.target_mpp),
        "spot_diameter_um": float(dataset.spot_diameter_um),
        "crop_size": int(options.crop_size),
        "stride": None if options.stride is None else int(options.stride),
        "inference_batch_size": int(options.batch_size),
        "device": str(options.device if device is None else device),
        "precision": str(options.precision),
        "expression_transform": str(options.expression_transform),
        "threshold": float(options.threshold),
        "visualization_top_n": int(stage.top_genes),
        "evaluate_spots": bool(cfg.evaluation.test.spot_classification),
    }

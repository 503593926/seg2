import json
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch

from seg2.models.model import JointModel
from seg2.test.data import (
    align_gene_adata,
    label_sample_dir,
    load_gene_adata,
    read_sample,
)
from seg2.test.evaluate import evaluate_spot_probabilities, validate_slide
from seg2.test.metrics import (
    spot_classification_metrics,
    summarize_metrics,
    summarize_spot_classification,
)
from seg2.test.visualization import (
    save_probability_overlay,
    save_spot_overlay,
    save_top_gene_predictions,
)


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _precision_context(device, precision):
    if precision == "32-true":
        return nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype={"bf16-mixed": torch.bfloat16, "16-mixed": torch.float16}[precision],
    )


def load_model(path, device: str | torch.device = "cpu"):
    artifact = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = JointModel(**artifact["model_config"])
    model.load_state_dict(artifact["state_dict"])
    model.model_config = artifact["model_config"]
    model.gene_names = list(map(str, artifact["gene_names"]))
    return (model.to(device).eval(), artifact)


def _save_slide_visualizations(
    directory,
    root,
    name,
    image_rgb,
    probability,
    gene_adata,
    target_mpp,
    prediction_mpp,
    spot_diameter_um,
    threshold,
):
    image = cv2.cvtColor(
        cv2.resize(
            image_rgb,
            (probability.shape[1], probability.shape[0]),
            interpolation=cv2.INTER_AREA,
        ),
        cv2.COLOR_RGB2BGR,
    )
    on_slide = gene_adata.obs["name"].astype(str).to_numpy() == name
    spots = (
        gene_adata.obs.loc[on_slide, ["pixel_x", "pixel_y"]].to_numpy(dtype=np.float32)
        * float(target_mpp)
        / float(prediction_mpp)
    )
    tissue = (
        np.asarray(
            cv2.imread(
                str(
                    label_sample_dir(root, name, prediction_mpp, target_mpp)
                    / "spatial/tissue_mask.png"
                ),
                cv2.IMREAD_GRAYSCALE,
            )
        )
        > 0
    )
    save_spot_overlay(
        directory,
        image,
        spots,
        float(spot_diameter_um) / (2.0 * float(prediction_mpp)),
    )
    save_probability_overlay(
        directory, image, tissue, probability, title=name, threshold=float(threshold)
    )


def run_test(
    artifact_path,
    data_root,
    gene_path,
    output_dir,
    *,
    names,
    feature_maps,
    target_mpp,
    spot_diameter_um,
    crop_size=896,
    stride=None,
    device="cpu",
    precision="32-true",
    inference_batch_size=1,
    expression_transform="log1p",
    threshold=0.5,
    visualization_top_n=0,
    evaluate_spots=False,
):
    names = list(map(str, names))
    device = torch.device(device)
    context = _precision_context(device, precision)
    model, artifact = load_model(artifact_path, device)
    output_dir = Path(output_dir)
    roots = (
        {name: Path(data_root[name]) for name in names}
        if isinstance(data_root, dict)
        else dict.fromkeys(names, Path(data_root))
    )
    raw_genes = load_gene_adata(gene_path, names)
    spec = artifact.get("expression_spec")
    if raw_genes.uns.get("preprocessing_id") or raw_genes.uns.get("external_task"):
        if not spec or spec.get("preprocessing_id") != raw_genes.uns.get(
            "preprocessing_id"
        ):
            raise ValueError(
                "The model and expression targets use different preprocessing specifications"
            )
        if spec["expression_transform"] != expression_transform:
            raise ValueError("Model and expression transforms differ")
    genes = align_gene_adata(raw_genes, model.gene_names)
    prediction_mpp = float(target_mpp) * model.output_scale
    output_dir.mkdir(parents=True, exist_ok=True)
    slide_metrics, expression_results, backgrounds = ({}, {}, {})
    spot_results = {}
    with context:
        for name in names:
            sample = read_sample(
                roots[name],
                name,
                list(feature_maps),
                float(target_mpp),
                prediction_mpp,
            )
            probability, metrics = validate_slide(
                model,
                sample,
                roots[name],
                device,
                crop_size=int(crop_size),
                stride=stride,
                batch_size=int(inference_batch_size),
                mpp=float(target_mpp),
                threshold=float(threshold),
                gene_adata=genes,
                spot_radius_px=float(spot_diameter_um) / (2 * float(target_mpp)),
                expression_transform=expression_transform,
                evaluate_segmentation=sample["has_segmentation"],
                expression_results=expression_results,
                save_dense_expression=visualization_top_n > 0,
            )
            directory = output_dir / name
            directory.mkdir(parents=True, exist_ok=True)
            if sample["has_segmentation"] or evaluate_spots:
                np.save(directory / "probability.npy", probability.astype(np.float32))
                cv2.imwrite(
                    str(directory / "mask.png"),
                    (probability >= threshold).astype(np.uint8) * 255,
                )
            if evaluate_spots:
                table = evaluate_spot_probabilities(
                    probability,
                    roots[name],
                    name,
                    float(target_mpp),
                    prediction_mpp,
                    float(spot_diameter_um),
                    threshold,
                )
                table.to_csv(directory / "spot_predictions.tsv", sep="\t", index=False)
                spot_results[name] = table
                usable = table.classification_valid
                metrics["spot_classification"] = spot_classification_metrics(
                    table.loc[usable, "probability"],
                    table.loc[usable, "spot_label"],
                    threshold,
                )
            if name in expression_results:
                np.savez_compressed(
                    directory / "gene_predictions.npz",
                    **{
                        key: value
                        for key, value in expression_results[name].items()
                        if not key.startswith("dense_")
                    },
                    gene_names=np.asarray(model.gene_names, dtype=str),
                    expression_transform=expression_transform,
                )
            _write_json(directory / "metrics.json", metrics)
            if visualization_top_n > 0:
                _save_slide_visualizations(
                    directory,
                    roots[name],
                    name,
                    sample["image"],
                    probability,
                    genes,
                    target_mpp,
                    prediction_mpp,
                    spot_diameter_um,
                    threshold,
                )
            slide_metrics[name] = metrics
            if visualization_top_n > 0:
                backgrounds[name] = {"image": sample["image"]}
            del sample
    summary = summarize_metrics(slide_metrics, expression_results)
    if evaluate_spots:
        summary["spot_classification"] = summarize_spot_classification(
            spot_results, threshold
        )
    _write_json(output_dir / "metrics.json", summary)
    if visualization_top_n > 0:
        save_top_gene_predictions(
            output_dir / "genes",
            expression_results,
            model.gene_names,
            backgrounds,
            top_n=int(visualization_top_n),
            spot_radius_px=float(spot_diameter_um) / (2 * float(target_mpp)),
            expression_transform=expression_transform,
            source_model="saved_model",
        )
    _write_json(
        output_dir / "metadata.json",
        {
            "artifact": str(Path(artifact_path).resolve()),
            "device": str(device),
            "precision": precision,
            "slides": names,
            "data_roots": {name: str(root.resolve()) for name, root in roots.items()},
            "gene_path": str(Path(gene_path).resolve()),
            "feature_maps": list(feature_maps),
            "target_mpp": float(target_mpp),
            "prediction_mpp": prediction_mpp,
            "crop_size": int(crop_size),
            "stride": stride,
            "expression_transform": expression_transform,
            "evaluate_spots": bool(evaluate_spots),
            "expression_spec": spec,
        },
    )
    return summary

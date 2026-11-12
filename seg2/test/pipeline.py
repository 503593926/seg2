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
from seg2.test.evaluate import evaluate_region_probabilities, validate_slide
from seg2.test.metrics import (
    region_classification_metrics,
    summarize_metrics,
    summarize_region_classification,
)
from seg2.test.visualization import (
    save_probability_overlay,
    save_region_overlay,
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
    region_map,
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
                    label_sample_dir(root=root, name=name, label_mpp=prediction_mpp, target_mpp=target_mpp)
                    / "spatial/tissue_mask.png"
                ),
                cv2.IMREAD_GRAYSCALE,
            )
        )
        > 0
    )
    save_region_overlay(
        directory=directory, image_bgr=image, coordinates=spots,
        radius_px=None if spot_diameter_um is None else float(spot_diameter_um) / (2.0 * float(prediction_mpp)),
        region_map=region_map, mpp=prediction_mpp,
    )
    save_probability_overlay(
        directory=directory, image_bgr=image, tissue=tissue, probability=probability, title=name, threshold=float(threshold)
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
    spot_diameter_um=None,
    region_type="circle",
    crop_size=896,
    stride=None,
    device="cpu",
    precision="32-true",
    inference_batch_size=1,
    expression_transform="log1p",
    threshold=0.5,
    visualization_top_n=0,
    evaluate_regions=False,
):
    if region_type not in {"circle", "polygon"}:
        raise ValueError(f"Unknown region geometry: {region_type}")
    if region_type == "circle" and (spot_diameter_um is None or float(spot_diameter_um) <= 0):
        raise ValueError("Circular regions require a positive spot_diameter_um")
    names = list(map(str, names))
    device = torch.device(device)
    context = _precision_context(device=device, precision=precision)
    model, _ = load_model(path=artifact_path, device=device)
    output_dir = Path(output_dir)
    roots = (
        {name: Path(data_root[name]) for name in names}
        if isinstance(data_root, dict)
        else dict.fromkeys(names, Path(data_root))
    )
    raw_genes = load_gene_adata(path=gene_path, names=names)
    genes = align_gene_adata(gene_adata=raw_genes, gene_names=model.gene_names)
    prediction_mpp = float(target_mpp) * model.output_scale
    output_dir.mkdir(parents=True, exist_ok=True)
    slide_metrics, expression_results, backgrounds = ({}, {}, {})
    region_results = {}
    with context:
        for name in names:
            sample = read_sample(
                root=roots[name], name=name, feature_maps=list(feature_maps),
                target_mpp=float(target_mpp), label_mpp=prediction_mpp, region_type=region_type,
            )
            probability, metrics = validate_slide(
                model=model, slide=sample, root=roots[name], device=device,
                crop_size=int(crop_size),
                stride=stride,
                batch_size=int(inference_batch_size),
                mpp=float(target_mpp),
                threshold=float(threshold),
                gene_adata=genes,
                spot_radius_px=None if spot_diameter_um is None else float(spot_diameter_um) / (2 * float(target_mpp)),
                expression_transform=expression_transform,
                evaluate_segmentation=sample["has_segmentation"],
                expression_results=expression_results,
                save_dense_expression=visualization_top_n > 0,
            )
            directory = output_dir / name
            directory.mkdir(parents=True, exist_ok=True)
            if sample["has_segmentation"] or evaluate_regions:
                np.save(directory / "probability.npy", probability.astype(np.float32))
                cv2.imwrite(
                    str(directory / "mask.png"),
                    (probability >= threshold).astype(np.uint8) * 255,
                )
            if evaluate_regions:
                table = evaluate_region_probabilities(
                    probability=probability, root=roots[name], name=name,
                    target_mpp=float(target_mpp), prediction_mpp=prediction_mpp,
                    spot_diameter_um=spot_diameter_um, threshold=threshold, region_map=sample["region_map"],
                )
                table.to_csv(directory / "region_predictions.tsv", sep="\t", index=False)
                region_results[name] = table
                usable = table.classification_valid
                metrics["region_classification"] = region_classification_metrics(
                    probability=table.loc[usable, "probability"],
                    labels=table.loc[usable, "spot_label"], threshold=threshold,
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
            _write_json(path=directory / "metrics.json", value=metrics)
            if visualization_top_n > 0:
                _save_slide_visualizations(
                    directory=directory, root=roots[name], name=name, image_rgb=sample["image"],
                    probability=probability, gene_adata=genes, target_mpp=target_mpp,
                    prediction_mpp=prediction_mpp, spot_diameter_um=spot_diameter_um,
                    threshold=threshold, region_map=sample["region_map"],
                )
            slide_metrics[name] = metrics
            if visualization_top_n > 0:
                backgrounds[name] = {"image": sample["image"], "region_map": sample["region_map"]}
            del sample
    summary = summarize_metrics(slides=slide_metrics, expression_results=expression_results)
    if evaluate_regions:
        summary["region_classification"] = summarize_region_classification(
            records=region_results, threshold=threshold
        )
    _write_json(path=output_dir / "metrics.json", value=summary)
    if visualization_top_n > 0:
        save_top_gene_predictions(
            directory=output_dir / "genes", expression_results=expression_results,
            gene_names=model.gene_names, slides=backgrounds,
            top_n=int(visualization_top_n),
            spot_radius_px=None if spot_diameter_um is None else float(spot_diameter_um) / (2 * float(target_mpp)),
            expression_transform=expression_transform,
            source_model="saved_model",
        )
    _write_json(
        path=output_dir / "metadata.json",
        value={
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
            "evaluate_regions": bool(evaluate_regions),
            "region_type": region_type,
        },
    )
    return summary

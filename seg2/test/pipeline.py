import json
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch

from seg2.infer.artifacts import load_model
from seg2.infer.memory import release_inference_cache
from seg2.test.data import load_gene_adata, read_sample
from seg2.test.evaluate import validate_slide
from seg2.test.metrics import summarize_metrics
from seg2.test.visualization import save_top_gene_predictions


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
):
    names = list(map(str, names))
    device = torch.device(device)
    context = _precision_context(device, precision)
    model, _ = load_model(artifact_path, device)
    output_dir = Path(output_dir)
    roots = (
        {name: Path(data_root[name]) for name in names}
        if isinstance(data_root, dict)
        else dict.fromkeys(names, Path(data_root))
    )
    genes = load_gene_adata(gene_path, names)
    genes = genes[:, model.gene_names].copy()
    prediction_mpp = float(target_mpp) * model.output_scale
    output_dir.mkdir(parents=True, exist_ok=True)
    slide_metrics, expression_results, backgrounds = ({}, {}, {})
    try:
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
                if sample["has_segmentation"]:
                    np.save(
                        directory / "probability.npy", probability.astype(np.float32)
                    )
                    cv2.imwrite(
                        str(directory / "mask.png"),
                        (probability >= threshold).astype(np.uint8) * 255,
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
                slide_metrics[name] = metrics
                if visualization_top_n > 0:
                    backgrounds[name] = {"image": sample["image"]}
                del sample
        summary = summarize_metrics(slide_metrics, expression_results)
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
                "data_roots": {
                    name: str(root.resolve()) for name, root in roots.items()
                },
                "gene_path": str(Path(gene_path).resolve()),
                "feature_maps": list(feature_maps),
                "target_mpp": float(target_mpp),
                "prediction_mpp": prediction_mpp,
                "crop_size": int(crop_size),
                "stride": stride,
                "expression_transform": expression_transform,
            },
        )
        return summary
    finally:
        del model
        release_inference_cache(device, "test end")

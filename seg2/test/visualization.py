import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from seg2.test.metrics import gene_pearson_correlations


def _write_image(path, image, parameters=()):
    cv2.imwrite(str(path), image, list(parameters))


def _uint16_probability(probability):
    values = np.nan_to_num(probability, nan=0.5, posinf=1.0, neginf=0.0)
    return np.rint(np.clip(values, 0.0, 1.0) * 65535).astype(np.uint16)


def _blend(image, selected, color, alpha=0.45):
    image[selected] = np.rint(
        image[selected].astype(np.float32) * (1.0 - alpha) + np.asarray(color, dtype=np.float32) * alpha
    ).astype(np.uint8)


def _mask_overlay(image_bgr, mask, region):
    overlay = image_bgr.copy()
    _blend(overlay, region & (mask == 0), (255, 0, 0))
    _blend(overlay, region & (mask == 1), (0, 0, 255))
    return overlay


def _overview_panel(image, title, size=512):
    canvas = np.full((size, size, 3), 245, dtype=np.uint8)
    available_height = size - 36
    height, width = image.shape[:2]
    scale = min(size / width, available_height / height)
    resized = cv2.resize(
        image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA
    )
    top = 36 + (available_height - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    cv2.putText(canvas, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def save_prediction_result(directory, image_rgb, probability, labels, valid, scale, threshold=0.5):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    size = (image_rgb.shape[1] // scale, image_rgb.shape[0] // scale)
    down_probability = cv2.resize(probability, size, interpolation=cv2.INTER_AREA)
    down_image = cv2.cvtColor(cv2.resize(image_rgb, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
    down_labels = cv2.resize(labels, size, interpolation=cv2.INTER_NEAREST)
    down_valid = cv2.resize(valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    prediction = ((down_probability >= threshold) & down_valid).astype(np.uint8)
    prediction_overlay = _mask_overlay(down_image, prediction, down_valid)
    expert_overlay = _mask_overlay(down_image, down_labels, down_valid)
    png_parameters = (cv2.IMWRITE_PNG_COMPRESSION, 3)
    jpg_parameters = (cv2.IMWRITE_JPEG_QUALITY, 90)
    _write_image(directory / "probability.png", _uint16_probability(down_probability), png_parameters)
    _write_image(directory / "mask_overlay.jpg", prediction_overlay, jpg_parameters)
    overview = np.hstack(
        (
            _overview_panel(down_image, "H&E"),
            _overview_panel(prediction_overlay, "Prediction"),
            _overview_panel(expert_overlay, "Expert label"),
        )
    )
    _write_image(directory / "overview.jpg", overview, jpg_parameters)


def _expression_background(image_rgb, spots, spot_radius_px, size=512):
    height, width = image_rgb.shape[:2]
    scale = min((size - 24) / width, (size - 100) / height)
    resized = cv2.resize(
        image_rgb, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA
    )
    resized = cv2.cvtColor(
        (resized.astype(np.float32) * 0.18 + 255 * 0.82).astype(np.uint8), cv2.COLOR_RGB2BGR
    )
    top = 44 + (size - 100 - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    background = np.full((size, size, 3), 255, dtype=np.uint8)
    background[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    centers = np.rint(np.asarray(spots) * scale + np.array([left, top])).astype(np.int32)
    return (background, centers, max(1, round(spot_radius_px * scale)))


def _expression_spot_panel(background, centers, radius, values, maximum, title):
    panel = background.copy()
    encoded = np.rint(np.clip(values / maximum, 0, 1) * 255).astype(np.uint8)
    if len(encoded):
        colors = cv2.applyColorMap(encoded[:, None], cv2.COLORMAP_VIRIDIS)[:, 0]
        for (x, y), color in zip(centers, colors):
            cv2.circle(panel, (int(x), int(y)), radius, tuple(map(int, color)), -1, cv2.LINE_AA)
    else:
        cv2.putText(
            panel,
            "No observed values",
            (140, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
    return _expression_colorbar(panel, maximum, f"{title} ({len(values)} spots)")


def _expression_colorbar(panel, maximum, title):
    size = panel.shape[0]
    cv2.putText(panel, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)
    gradient = np.tile(np.linspace(0, 255, size - 40).astype(np.uint8), (12, 1))
    panel[size - 42 : size - 30, 20 : size - 20] = cv2.applyColorMap(gradient, cv2.COLORMAP_VIRIDIS)
    for label, x in (("0", 20), (f"{maximum:.3g}", size - 80)):
        cv2.putText(panel, label, (x, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return panel


def _expression_dense_panel(heatmap, maximum, title, size=512):
    height, width = heatmap.shape[:2]
    scale = min((size - 24) / width, (size - 100) / height)
    resized = cv2.resize(
        heatmap,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    top = 44 + (size - 100 - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    panel = np.full((size, size, 3), 255, dtype=np.uint8)
    panel[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return _expression_colorbar(panel, maximum, title)


def _expression_records(expression_results):
    records = {}
    for name, source in expression_results.items():
        if not len(source["prediction"]):
            continue
        target = torch.as_tensor(source["target"]).detach().float().cpu()
        records[name] = {
            "prediction": torch.as_tensor(source["prediction"]).detach().float().cpu(),
            "target": target,
            "observed": torch.as_tensor(source["observed"]).bool().cpu(),
            "spots": np.asarray(source["spots"]),
        }
    return records


def _rank_expression_genes(records, gene_names, top_n):
    prediction = torch.cat([record["prediction"] for record in records.values()])
    target = torch.cat([record["target"] for record in records.values()])
    observed = torch.cat([record["observed"] for record in records.values()])
    correlations = gene_pearson_correlations(prediction, target, observed).numpy()
    eligible = np.flatnonzero(np.isfinite(correlations))
    indices = eligible[np.argsort(-correlations[eligible], kind="stable")[:top_n]]
    selected = []
    for position, index in enumerate(indices):
        mask = observed[:, index]
        normalization = {
            key: {"min": float(values[mask, index].min()), "max": float(values[mask, index].max())}
            for key, values in (("observed", target), ("predicted", prediction))
        }
        selected.append(
            {
                "rank": position + 1,
                "gene": str(gene_names[index]),
                "gene_index": int(index),
                "pcc": float(correlations[index]),
                "observed_spots": int(mask.sum()),
                "normalization": normalization,
                "slides": {},
            }
        )
    return (selected, len(eligible))


def _decode_selected_genes(records, expression_results, selected, expression_transform):
    indices = [gene["gene_index"] for gene in selected]
    for name, record in records.items():
        source = expression_results[name]
        dense = source["dense_expression"]
        if dense is None or not len(indices):
            continue
        values = np.asarray(dense["coefficients"]) @ np.asarray(dense["templates"])[:, indices]
        if expression_transform == "log1p":
            np.log1p(values, out=values)
        tissue = np.asarray(source["dense_tissue"], dtype=bool)
        record.update(dense_prediction=values, dense_tissue=tissue, dense_mpp=float(source["dense_mpp"]))
    for position, gene in enumerate(selected):
        dense_values = [
            record["dense_prediction"][..., position][record["dense_tissue"]]
            for record in records.values()
            if "dense_prediction" in record and record["dense_tissue"].any()
        ]
        if dense_values:
            minimum, maximum = (
                min((float(value.min()) for value in dense_values)),
                max((float(value.max()) for value in dense_values)),
            )
            gene["dense_normalization"] = {"min": minimum, "max": maximum}


def _save_gene_comparisons(
    directory, records, selected, slides, spot_radius_px, expression_transform, source_model
):
    indices = [gene["gene_index"] for gene in selected]
    for name, record in records.items():
        if not selected or "dense_prediction" not in record:
            continue
        slide_dir = directory / name
        slide_dir.mkdir(exist_ok=True)
        background, centers, radius = _expression_background(
            slides[name]["image"], record["spots"], spot_radius_px
        )
        local_pcc = gene_pearson_correlations(
            record["prediction"][:, indices], record["target"][:, indices], record["observed"][:, indices]
        ).numpy()
        for position, gene in enumerate(selected):
            index = gene["gene_index"]
            mask = record["observed"][:, index].numpy()
            values = [record[key][:, index].numpy()[mask] for key in ("target", "prediction")]
            bounds = [gene["normalization"][key] for key in ("observed", "predicted")]
            normalized_values = [
                (value - limits["min"]) / (limits["max"] - limits["min"])
                if limits["max"] > limits["min"]
                else np.zeros_like(value)
                for value, limits in zip(values, bounds)
            ]
            pcc = float(local_pcc[position])
            local_text = f"{pcc:.3f}" if pcc is not None else "n/a"
            title = f"{gene['rank']:02d}. {gene['gene']} | {name} | pooled PCC {gene['pcc']:.3f} | slide PCC {local_text}"
            safe_name = re.sub("[^A-Za-z0-9_.-]+", "_", gene["gene"]).strip("._") or "gene"
            stem = f"{gene['rank']:02d}_{safe_name}"
            dense_values = record["dense_prediction"][..., position]
            limits = gene["dense_normalization"]
            span = limits["max"] - limits["min"]
            normalized_dense = (
                (dense_values - limits["min"]) / span if span > 0 else np.zeros_like(dense_values)
            )
            encoded = np.rint(np.clip(normalized_dense, 0, 1) * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(encoded, cv2.COLORMAP_VIRIDIS)
            heatmap[~record["dense_tissue"]] = 255
            panels = [
                _expression_spot_panel(background, centers[mask], radius, value, 1.0, f"{label} normalized")
                for value, label in zip(normalized_values, ("Observed", "Predicted"))
            ]
            panels.append(
                _expression_dense_panel(heatmap, 1.0, f"Predicted {record['dense_mpp']:g}mpp normalized")
            )
            comparison = np.vstack((np.full((64, 1536, 3), 255, dtype=np.uint8), np.hstack(panels)))
            cv2.putText(
                comparison, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA
            )
            cv2.putText(
                comparison,
                f"Source: {source_model} | {expression_transform} | PCC from spots | independent min-max for observed/spot/dense predictions",
                (12, 49),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (50, 50, 50),
                1,
                cv2.LINE_AA,
            )
            filename = f"{stem}_dense_comparison_normalized.png"
            _write_image(slide_dir / filename, comparison, (cv2.IMWRITE_PNG_COMPRESSION, 3))
            gene["slides"][name] = {
                "pcc": pcc,
                "observed_spots": int(mask.sum()),
                "dense_comparison_normalized_file": f"{name}/{filename}",
                "dense_mpp": record["dense_mpp"],
                "dense_shape": list(dense_values.shape),
            }


def save_top_gene_predictions(
    directory,
    expression_results,
    gene_names,
    slides,
    *,
    top_n=10,
    spot_radius_px=50.0,
    expression_transform="log1p",
    source_model="ema",
):
    if top_n <= 0:
        return {}
    records = _expression_records(expression_results)
    if not records:
        return {}
    selected, eligible_count = _rank_expression_genes(records, gene_names, top_n)
    _decode_selected_genes(records, expression_results, selected, expression_transform)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _save_gene_comparisons(
        directory, records, selected, slides, spot_radius_px, expression_transform, source_model
    )
    columns = ("rank", "gene", "gene_index", "pcc", "observed_spots")
    with (directory / "top_genes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(({key: gene[key] for key in columns} for gene in selected))
    metadata = {
        "source_model": source_model,
        "ranking_scope": "pooled_validation_spots",
        "expression_transform": expression_transform,
        "requested_top_n": int(top_n),
        "eligible_genes": eligible_count,
        "selected_genes": selected,
        "normalized_color_scale": "independent min-max for observed and predicted per gene, pooled across slides; displayed range [0, 1]",
        "dense_normalized_color_scale": "independent dense min-max per gene over tissue pixels across slides; displayed range [0, 1]",
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata

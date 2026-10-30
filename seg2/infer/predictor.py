from collections.abc import Iterable
from math import ceil
from typing import TypedDict

import numpy as np
import torch

from seg2.infer.accumulator import PredictionAccumulator
from seg2.infer.prefetch import prefetch_batches


class ExpressionPrediction(TypedDict):
    coefficients: np.ndarray
    templates: np.ndarray


class SlidePrediction(TypedDict, total=False):
    probability: np.ndarray
    features: np.ndarray
    expression: ExpressionPrediction
    dense_expression: ExpressionPrediction


def tile_origins(length, crop_size, stride, alignment):
    if length <= crop_size:
        return [0]
    last = ceil((length - crop_size) / alignment) * alignment
    origins = list(range(0, last + 1, stride))
    if origins[-1] != last:
        origins.append(last)
    return origins


@torch.inference_mode()
def predict_slide(
    model,
    slide,
    crop_size,
    device,
    batch_size=1,
    stride=None,
    outputs: Iterable[str] = ("probability",),
    feature_pool=None,
    expression_pool=None,
) -> SlidePrediction:
    """Return only the requested NumPy predictions.

    outputs selects probability, features, expression (regional), or dense_expression.
    expression requires expression_pool. Both expression outputs contain
    coefficients and templates for decoding with @; dense_expression can be
    requested alongside regional expression.
    Pools provide output-grid pixels, rows and pixel_counts. feature_pool may also
    retain a reusable stitching cache in weights; expression results are never
    written into either pool.
    """
    outputs = frozenset((outputs,) if isinstance(outputs, str) else outputs)
    unknown = outputs - {"probability", "features", "expression", "dense_expression"}
    if unknown:
        raise ValueError(f"Unknown outputs: {', '.join(sorted(unknown))}")
    if not outputs:
        raise ValueError("outputs must contain at least one prediction name")
    if "expression" in outputs and expression_pool is None:
        raise ValueError("expression_pool is required for the 'expression' output")
    batch_size = int(batch_size)

    alignment = slide["alignment"]
    stride = (crop_size // 2 if stride is None else int(stride))
    assert stride % alignment == 0 and stride > 0

    height, width = slide["image"].shape[:2]
    rows = tile_origins(height, crop_size, stride, alignment)
    columns = tile_origins(width, crop_size, stride, alignment)

    origins = [(top, left) for top in rows for left in columns]

    accumulator = PredictionAccumulator(
        height,
        width,
        crop_size,
        origins,
        outputs=outputs,
        feature_pool=feature_pool,
        expression_pool=expression_pool,
    )
    was_training = model.training
    model.eval()

    batches = prefetch_batches(slide, origins, crop_size, batch_size, device)
    try:
        for batch_origins, inputs in batches:
            output = model(
                inputs["image"][0],
                inputs["patch"],
                inputs["cls"],
                compute_expression=accumulator.compute_expression,
            )
            accumulator.add_batch(batch_origins, output)
            del output, inputs
    finally:
        batches.close()
        model.train(was_training)
    return finalize_predictions(
        accumulator.finish(),
        outputs=outputs,
        feature_pool=accumulator.feature_pool,
        expression_pool=accumulator.expression_pool,
    )


def _pool_pixels(field, pixels, rows, pixel_counts):
    pixels = torch.as_tensor(pixels, device=field.device, dtype=torch.long)
    rows = torch.as_tensor(rows, device=field.device, dtype=torch.long)
    counts = torch.as_tensor(pixel_counts, device=field.device, dtype=torch.float64)
    sums = torch.zeros(
        (len(counts), field.shape[0]), device=field.device, dtype=torch.float64
    )
    values = field.flatten(1).T
    chunk = max(1,ceil(len(pixels) / 32))
    for offset in range(0, len(pixels), chunk):
        part = slice(offset, offset + chunk)
        selected = values.index_select(0, pixels[part]).to(torch.float64)
        sums.index_add_(0, rows[part], selected)
        del selected
    return sums / counts.clamp_min(1)[:, None]


def finalize_predictions(
    fields: dict[str, torch.Tensor],
    outputs: frozenset[str],
    feature_pool=None,
    expression_pool=None,
) -> SlidePrediction:
    result: SlidePrediction = {}
    if "probability" in outputs:
        result["probability"] = fields["probability"].cpu().numpy()
    if "features" in outputs:
        field = fields["features"]
        if feature_pool is not None:
            field = _pool_pixels(
                field,
                feature_pool["pixels"],
                feature_pool["rows"],
                feature_pool["pixel_counts"],
            )
        else:
            field = field.permute(1, 2, 0).contiguous()
        result["features"] = field.cpu().numpy()
    if outputs & {"expression", "dense_expression"}:
        templates = fields["templates"].cpu().numpy()
        if "expression" in outputs:
            assert expression_pool is not None
            coefficients = _pool_pixels(
                fields["coefficients"],
                expression_pool["pixels"],
                expression_pool["rows"],
                expression_pool["pixel_counts"],
            ).float()
            result["expression"] = {
                "coefficients": coefficients.cpu().numpy(), "templates": templates
            }
        if "dense_expression" in outputs:
            coefficients = fields["coefficients"].float().permute(1, 2, 0).contiguous()
            result["dense_expression"] = {
                "coefficients": coefficients.cpu().numpy(), "templates": templates
            }
    return result

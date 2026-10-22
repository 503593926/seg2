from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from math import ceil
from tempfile import TemporaryFile

import cv2
import numpy as np
import torch

from segmodel.inference.memory import inference_copy_stream


def _prediction_buffer(shape):
    """大幅状态/特征场用自动释放的临时文件映射，限制整片推理的常驻内存。"""
    if np.prod(shape) * np.dtype(np.float32).itemsize <= 256 * 1024**2:
        return np.zeros(shape, dtype=np.float32)
    with TemporaryFile(prefix="seg2_prediction_") as file:
        return np.memmap(file, dtype=np.float32, mode="w+", shape=shape)


def tile_origins(length, crop_size, stride, alignment):
    """生成覆盖整条边、且满足模型网格约束的窗口起点。

    对齐约束（crop_size/stride 必须是 alignment 的倍数）由 pipeline 解析几何量时校验，
    这里只挡住会让 range 步进失效的参数。
    """
    if crop_size <= 0 or stride <= 0 or stride > crop_size:
        raise ValueError(
            "crop_size and stride must be positive, with stride <= crop_size"
        )
    if length <= crop_size:
        return [0]

    last = ceil((length - crop_size) / alignment) * alignment
    origins = list(range(0, last + 1, stride))
    if origins[-1] != last:
        origins.append(last)
    return origins


def _prepare_batch(slide, origins, crop_size, device, copy_stream):
    """后台准备窗口并发起异步拷贝，不等待下一批准备完成才运行当前批。"""
    inputs = [slide.inputs(top, left, crop_size) for top, left in origins]
    batch = {"image": [torch.stack([item["image"] for item in inputs])]}
    for key in ("patch", "cls"):
        batch[key] = [
            torch.stack([item[key][index] for item in inputs])
            for index in range(len(inputs[0][key]))
        ]
    with torch.cuda.stream(copy_stream) if copy_stream is not None else nullcontext():
        if copy_stream is not None:
            batch = {key: [value.pin_memory() for value in values] for key, values in batch.items()}
        batch = {
            key: [value.to(device, non_blocking=copy_stream is not None) for value in values]
            for key, values in batch.items()
        }
        ready = torch.cuda.Event() if copy_stream is not None else None
        if ready is not None:
            ready.record(copy_stream)
    return batch, ready


def _prefetch_batches(slide, origins, crop_size, batch_size, device):
    """预取下一批输入；CUDA 拷贝与当前批前向重叠，保持批次顺序。"""
    device = torch.device(device)
    batches = [origins[start:start + batch_size] for start in range(0, len(origins), batch_size)]
    copy_stream = inference_copy_stream(device)
    current_stream = torch.cuda.current_stream(device) if copy_stream is not None else None
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(_prepare_batch, slide, batches[0], crop_size, device, copy_stream)
        try:
            for index, batch_origins in enumerate(batches):
                assert pending is not None
                inputs, ready = pending.result()
                pending = None
                if index + 1 < len(batches):
                    pending = executor.submit(
                        _prepare_batch, slide, batches[index + 1], crop_size, device, copy_stream,
                    )
                if current_stream is not None and ready is not None:
                    current_stream.wait_event(ready)
                    for values in inputs.values():
                        for value in values:
                            value.record_stream(current_stream)
                try:
                    yield batch_origins, inputs
                finally:
                    inputs = ready = None
        finally:
            if pending is not None:
                pending.cancel()


def _accumulate_pixels(field, pixels, rows, weights, sums):
    """按稀疏像素索引累加到节点或 spot，分块限制 FP64 临时显存。"""
    pixels = torch.as_tensor(pixels, device=sums.device)
    rows = torch.as_tensor(rows, device=sums.device, dtype=torch.long)
    weights = torch.as_tensor(weights, device=sums.device)
    values = field.flatten(1).T
    for offset in range(0, len(pixels), 262144):
        part = slice(offset, offset + 262144)
        selected = values.index_select(0, pixels[part]).to(torch.float64)
        selected.mul_(weights[part, None])
        sums.index_add_(0, rows[part], selected)
        del selected


@torch.inference_mode()
def predict_slide(
    model,
    slide,
    crop_size,
    stride,
    alignment,
    device,
    batch_size=1,
    return_features=True,
    feature_key="features",
    expression_factors=None,
    superpixel_pool=None,
    spot_pool=None,
):
    """在原生预测网格拼接概率和 dense 系数，并流式聚合节点或 spot。"""
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("inference batch_size must be positive")
    was_training = model.training
    model.eval()

    height, width = slide.height, slide.width
    window = np.maximum(
        np.outer(np.hanning(crop_size), np.hanning(crop_size)),
        0.01,
    ).astype(np.float32)

    probability_sum = None
    probability_weight = None
    grid_window = None
    grid_scale = None
    feature_sum = None
    feature_weight = None
    coefficient_sum = None
    templates = None
    node_sum = None
    spot_sum = None
    rows = tile_origins(height, crop_size, stride, alignment)
    columns = tile_origins(width, crop_size, stride, alignment)
    origins = [(top, left) for top in rows for left in columns]

    batches = _prefetch_batches(slide, origins, crop_size, batch_size, device)
    try:
        for batch_origins, inputs in batches:
            options = {}
            if getattr(model, "tissue_state_count", 0):
                options["dense_expression"] = return_features and feature_key == "gene_expression"
                options["compute_expression"] = (
                    options["dense_expression"] or expression_factors is not None or spot_pool is not None
                )
            output = model(
                inputs["image"][0], inputs["patch"], inputs["cls"],
                **options,
            )
            probability_grid = output["logits"].float().sigmoid()[:, 0].cpu().numpy()
            feature_batch = None
            if return_features and superpixel_pool is None:
                feature_batch = (
                    output[feature_key]
                    .float()
                    .cpu()
                    .numpy()
                    .transpose(0, 2, 3, 1)
                )
            coefficient_batch = None
            coefficients = None
            if (expression_factors is not None or spot_pool is not None) and "tissue_states" in output:
                coefficients = output["tissue_states"].float() * output["expression_amplitude"].float()
                if expression_factors is not None:
                    coefficient_batch = coefficients.cpu().numpy().transpose(0, 2, 3, 1)
                if templates is None:
                    templates = model.tissue_state_decoder.template_logits.float().softmax(dim=1).cpu().numpy()

            if probability_weight is None:
                tile = probability_grid.shape[-1]
                grid_scale = crop_size // tile
                if crop_size % tile or probability_grid.shape[-2] != tile:
                    raise ValueError("logit grid must evenly divide the crop")
                grid_shape = (height // grid_scale, width // grid_scale)
                probability_sum = np.zeros(grid_shape, dtype=np.float32)
                probability_weight = superpixel_pool.get("weights") if superpixel_pool is not None else None
                if probability_weight is None:
                    probability_weight = np.zeros(grid_shape, dtype=np.float32)
                grid_window = cv2.resize(
                    window,
                    (tile, tile),
                    interpolation=cv2.INTER_AREA,
                )
                # 流式聚合需要先知道每个像素的完整 Hann 权重和。
                if spot_pool is not None or (superpixel_pool is not None and "weights" not in superpixel_pool):
                    for top, left in origins:
                        grid_top, grid_left = top // grid_scale, left // grid_scale
                        valid_height = min(crop_size, height - top) // grid_scale
                        valid_width = min(crop_size, width - left) // grid_scale
                        probability_weight[
                            grid_top:grid_top + valid_height, grid_left:grid_left + valid_width
                        ] += grid_window[:valid_height, :valid_width]
                    if superpixel_pool is not None:
                        superpixel_pool["weights"] = probability_weight
                if spot_pool is not None:
                    assert coefficients is not None
                    spot_sum = torch.zeros(
                        (len(spot_pool["pixel_counts"]), coefficients.shape[1]),
                        dtype=torch.float64, device=coefficients.device,
                    )
                if coefficient_batch is not None:
                    coefficient_sum = _prediction_buffer((*grid_shape, coefficient_batch.shape[-1]))
                if return_features:
                    if superpixel_pool is None:
                        assert feature_batch is not None
                        feature_sum = _prediction_buffer((*grid_shape, feature_batch[0].shape[-1]))
                        feature_weight = np.zeros(grid_shape, dtype=np.float32)
                    else:
                        superpixels, pixel_counts = superpixel_pool["superpixels"], superpixel_pool["pixel_counts"]
                        if superpixels.shape != grid_shape or output[feature_key].shape[-2:] != (tile, tile):
                            raise ValueError("Superpixels, features and logits must use the same prediction grid")
                        node_sum = torch.zeros(
                            (len(pixel_counts), output[feature_key].shape[1]),
                            dtype=torch.float64, device=output[feature_key].device,
                        )

            assert grid_scale is not None and grid_window is not None
            assert probability_sum is not None
            for index, (top, left) in enumerate(batch_origins):
                grid_top, grid_left = top // grid_scale, left // grid_scale
                valid_height = min(crop_size, height - top) // grid_scale
                valid_width = min(crop_size, width - left) // grid_scale
                rows_slice = slice(grid_top, grid_top + valid_height)
                columns_slice = slice(grid_left, grid_left + valid_width)
                weight_window = grid_window[:valid_height, :valid_width]
                if superpixel_pool is None and spot_pool is None:
                    probability_weight[rows_slice, columns_slice] += weight_window
                probability_sum[rows_slice, columns_slice] += (
                    probability_grid[index][:valid_height, :valid_width]
                    * weight_window
                )
                if coefficient_batch is not None:
                    assert coefficient_sum is not None
                    coefficient_sum[rows_slice, columns_slice] += (
                        coefficient_batch[index, :valid_height, :valid_width] * weight_window[..., None]
                    )
                if spot_pool is not None:
                    assert coefficients is not None and spot_sum is not None
                    grid_width = probability_weight.shape[1]
                    first, last = np.searchsorted(
                        spot_pool["pixels"], [grid_top * grid_width, (grid_top + valid_height) * grid_width],
                    )
                    pixels = spot_pool["pixels"][first:last]
                    x = pixels % grid_width
                    inside = (x >= grid_left) & (x < grid_left + valid_width)
                    local_pixels = (pixels[inside] // grid_width - grid_top) * valid_width + x[inside] - grid_left
                    weights = (
                        weight_window.astype(np.float64)
                        / np.maximum(probability_weight[rows_slice, columns_slice], 1e-6)
                    ).ravel()[local_pixels]
                    _accumulate_pixels(
                        coefficients[index, :, :valid_height, :valid_width], local_pixels,
                        spot_pool["rows"][first:last][inside], weights, spot_sum,
                    )

                if return_features:
                    if superpixel_pool is None:
                        assert feature_batch is not None and feature_sum is not None and feature_weight is not None
                        features = feature_batch[index]
                        feature_sum[rows_slice, columns_slice] += (
                            features[:valid_height, :valid_width]
                            * weight_window[..., None]
                        )
                        feature_weight[rows_slice, columns_slice] += weight_window
                    else:
                        assert node_sum is not None
                        superpixels = superpixel_pool["superpixels"]
                        labels = superpixels[rows_slice, columns_slice].ravel()
                        foreground = np.flatnonzero(labels)
                        weights = (
                            weight_window.astype(np.float64)
                            / np.maximum(probability_weight[rows_slice, columns_slice], 1e-6)
                        ).ravel()[foreground]
                        _accumulate_pixels(
                            output[feature_key][index, :, :valid_height, :valid_width],
                            foreground, labels[foreground], weights, node_sum,
                        )
            del output, inputs, coefficients
    finally:
        batches.close()
        model.train(was_training)

    features = None
    if return_features:
        if superpixel_pool is None:
            assert feature_sum is not None and feature_weight is not None
            # 稠密基因图通道较多, 原地归一化避免再复制一份整片表达场。
            feature_sum /= np.maximum(feature_weight[..., None], 1e-6)
            features = feature_sum
        else:
            assert node_sum is not None
            pixel_counts = superpixel_pool["pixel_counts"]
            features = node_sum.cpu().numpy()
            features /= np.maximum(pixel_counts[:, None], 1)
            features[0] = 0
    assert probability_sum is not None and probability_weight is not None
    if spot_sum is not None:
        assert spot_pool is not None
        spot_pool["coefficients"] = (
            spot_sum.cpu().numpy() / np.maximum(spot_pool["pixel_counts"][:, None], 1)
        ).astype(np.float32)
        spot_pool["templates"] = templates
    if coefficient_sum is not None:
        assert expression_factors is not None
        # 仅缓存 K 通道系数；确定最高 PCC 基因后再解码所需通道。
        coefficient_sum /= np.maximum(probability_weight[..., None], 1e-6)
        expression_factors.update(coefficients=coefficient_sum, templates=templates)
    probability = probability_sum / np.maximum(probability_weight, 1e-6)
    return probability, features

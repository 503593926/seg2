from math import ceil

import cv2
import numpy as np
import torch


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


def _stack_features(batch_inputs, key, device):
    """把一批窗口的逐 PFM 特征列表堆叠成 K 个 (B, D_i, H_i, W_i)。"""
    return [
        torch.stack([inputs[key][index] for inputs in batch_inputs]).to(device)
        for index in range(len(batch_inputs[0][key]))
    ]


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
    feature_key="propagation_features",
    auxiliary_outputs=None,
    expression_factors=None,
):
    """重叠滑窗推理：分割头与 feature_key 指定的特征在 7mpp 网格加权拼接，
    概率最后上采样回 1mpp。返回 (probability, features)。"""
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
    auxiliary_sums = {}
    rows = tile_origins(height, crop_size, stride, alignment)
    columns = tile_origins(width, crop_size, stride, alignment)
    origins = [(top, left) for top in rows for left in columns]

    try:
        for start in range(0, len(origins), batch_size):
            batch_origins = origins[start : start + batch_size]
            batch_inputs = [
                slide.inputs(top, left, crop_size)
                for top, left in batch_origins
            ]
            output = model(
                torch.stack([inputs["image"] for inputs in batch_inputs]).to(device),
                _stack_features(batch_inputs, "patch", device),
                _stack_features(batch_inputs, "cls", device),
            )
            probability_grid = output["logits"].float().sigmoid()[:, 0].cpu().numpy()
            auxiliary_batch = {}
            if auxiliary_outputs is not None and "initial_logits" in output:
                auxiliary_batch = {
                    "initial_probability": output["initial_logits"].float().sigmoid()[:, 0].cpu().numpy(),
                    "refinement_gate": output["refinement_gate"].float()[:, 0].cpu().numpy(),
                }
            feature_batch = None
            if return_features:
                feature_batch = (
                    output[feature_key]
                    .float()
                    .cpu()
                    .numpy()
                    .transpose(0, 2, 3, 1)
                )
            coefficient_batch = None
            if expression_factors is not None and "tissue_states" in output:
                coefficient_batch = (
                    (output["tissue_states"].float() * output["expression_amplitude"].float())
                    .cpu().numpy().transpose(0, 2, 3, 1)
                )
                if templates is None:
                    templates = model.tissue_state_decoder.template_logits.float().softmax(dim=1).cpu().numpy()

            if probability_weight is None:
                tile = probability_grid.shape[-1]
                grid_scale = crop_size // tile
                if crop_size % tile or probability_grid.shape[-2] != tile:
                    raise ValueError("logit grid must evenly divide the crop")
                grid_shape = (height // grid_scale, width // grid_scale)
                probability_sum = np.zeros(grid_shape, dtype=np.float32)
                probability_weight = np.zeros(grid_shape, dtype=np.float32)
                auxiliary_sums = {key: np.zeros(grid_shape, dtype=np.float32) for key in auxiliary_batch}
                grid_window = cv2.resize(
                    window,
                    (tile, tile),
                    interpolation=cv2.INTER_AREA,
                )
                if coefficient_batch is not None:
                    coefficient_sum = np.zeros((*grid_shape, coefficient_batch.shape[-1]), dtype=np.float32)
                if return_features:
                    assert feature_batch is not None
                    feature_sum = np.zeros(
                        (*grid_shape, feature_batch[0].shape[-1]),
                        dtype=np.float32,
                    )
                    feature_weight = np.zeros(grid_shape, dtype=np.float32)

            assert grid_scale is not None and grid_window is not None
            assert probability_sum is not None
            for index, (top, left) in enumerate(batch_origins):
                grid_top, grid_left = top // grid_scale, left // grid_scale
                valid_height = min(crop_size, height - top) // grid_scale
                valid_width = min(crop_size, width - left) // grid_scale
                rows_slice = slice(grid_top, grid_top + valid_height)
                columns_slice = slice(grid_left, grid_left + valid_width)
                weight_window = grid_window[:valid_height, :valid_width]
                probability_weight[rows_slice, columns_slice] += weight_window
                probability_sum[rows_slice, columns_slice] += (
                    probability_grid[index][:valid_height, :valid_width]
                    * weight_window
                )
                for key, values in auxiliary_batch.items():
                    auxiliary_sums[key][rows_slice, columns_slice] += (
                        values[index, :valid_height, :valid_width] * weight_window
                    )
                if coefficient_batch is not None:
                    coefficient_sum[rows_slice, columns_slice] += (
                        coefficient_batch[index, :valid_height, :valid_width] * weight_window[..., None]
                    )

                if return_features:
                    assert feature_batch is not None and feature_sum is not None and feature_weight is not None
                    features = feature_batch[index]
                    feature_sum[rows_slice, columns_slice] += (
                        features[:valid_height, :valid_width]
                        * weight_window[..., None]
                    )
                    feature_weight[rows_slice, columns_slice] += weight_window
    finally:
        model.train(was_training)

    features = None
    if return_features:
        assert feature_sum is not None and feature_weight is not None
        # 稠密基因图通道较多, 原地归一化避免再复制一份整片表达场。
        feature_sum /= np.maximum(feature_weight[..., None], 1e-6)
        features = feature_sum
    assert probability_sum is not None and probability_weight is not None
    if coefficient_sum is not None:
        # 仅缓存 K 通道系数；确定最高 PCC 基因后再解码所需通道。
        coefficient_sum /= np.maximum(probability_weight[..., None], 1e-6)
        expression_factors.update(coefficients=coefficient_sum, templates=templates)
    probability = cv2.resize(
        probability_sum / np.maximum(probability_weight, 1e-6),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    if auxiliary_outputs is not None:
        for key, values in auxiliary_sums.items():
            values /= np.maximum(probability_weight, 1e-6)
            if key == "initial_probability":
                values = cv2.resize(values, (width, height), interpolation=cv2.INTER_LINEAR)
            auxiliary_outputs[key] = values
    return probability, features

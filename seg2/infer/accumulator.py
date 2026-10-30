from collections.abc import Iterable, Mapping

import cv2
import numpy as np
import torch


class _PredictionGrid:
    def __init__(
        self,
        height,
        width,
        crop_size,
        tile_shape,
        origins,
        device,
        cached_weights=None
    ):
        _, tile = tile_shape
        scale = crop_size // tile
        self.shape = (height // scale, width // scale)

        window = np.maximum(
            np.outer(np.hanning(crop_size), np.hanning(crop_size)), 0.01
        ).astype(np.float32)
        window = cv2.resize(window, (tile, tile), interpolation=cv2.INTER_AREA)
        window = torch.as_tensor(window, device=device)

        self.windows = {}
        for top, left in origins:
            y, x = (top // scale, left // scale)
            h, w = (
                min(crop_size, height - top) // scale,
                min(crop_size, width - left) // scale,
            )
            self.windows[top, left] = (slice(y, y + h), slice(x, x + w), window[:h, :w])
        if cached_weights is None:
            self.weights = torch.zeros(self.shape, device=device, dtype=torch.float32)
            for y, x, weight in self.windows.values():
                self.weights[y, x].add_(weight)
        else:
            self.weights = torch.as_tensor(cached_weights, device=device)
        self.denominator = self.weights.clamp_min(1e-6)


class PredictionAccumulator:
    def __init__(
        self,
        height,
        width,
        crop_size,
        origins,
        outputs: Iterable[str],
        feature_pool=None,
        expression_pool=None,
    ):
        self.height = height
        self.width = width
        self.crop_size = crop_size
        self.origins = origins
        self.outputs = frozenset(outputs)
        self.compute_expression = bool(self.outputs & {"expression", "dense_expression"})
        self.feature_pool = feature_pool if "features" in self.outputs else None
        self.expression_pool = expression_pool if "expression" in self.outputs else None

        self.grid: _PredictionGrid | None = None
        self.dense_sums: dict[str, torch.Tensor] = {}
        self.templates: torch.Tensor | None = None

    def _initialize(self, fields: Mapping[str, torch.Tensor]):
        reference = next(iter(fields.values()))
        self.grid = _PredictionGrid(
            self.height,
            self.width,
            self.crop_size,
            reference.shape[-2:],
            self.origins,
            reference.device,
            self.feature_pool.get("weights") if self.feature_pool is not None else None,
        )

        if self.feature_pool is not None:
            self.feature_pool["weights"] = self.grid.weights

        for name, field in fields.items():
            pooled = (name == "features" and self.feature_pool is not None) or (
                name == "coefficients" and self.expression_pool is not None
            )
            self.dense_sums[name] = torch.zeros(
                (field.shape[1], *self.grid.shape),
                device=field.device,
                dtype=torch.float64 if pooled else torch.float32,
            )

    def add_batch(
        self,
        origins: list[tuple[int, int]],
        output: Mapping[str, torch.Tensor]
    ):
        fields: dict[str, torch.Tensor] = {}
        if "probability" in self.outputs:
            fields["probability"] = output["logits"].float().sigmoid()
        if "features" in self.outputs:
            fields["features"] = output["features"].float()
        if self.compute_expression:
            fields["coefficients"] = (
                output["tissue_states"].float() * output["expression_amplitude"].float()
            )
            if self.templates is None:
                self.templates = output["expression_templates"].float()
        if self.grid is None:
            self._initialize(fields)
        assert self.grid is not None

        for index, origin in enumerate(origins):
            y, x, weight = self.grid.windows[origin]
            h, w = weight.shape
            for name, field in fields.items():
                self.dense_sums[name][:, y, x].addcmul_(field[index, :, :h, :w], weight)

    def finish(self) -> dict[str, torch.Tensor]:
        assert self.grid is not None
        for field in self.dense_sums.values():
            field.div_(self.grid.denominator)
        fields = self.dense_sums.copy()
        if "probability" in fields:
            fields["probability"] = fields["probability"][0]
        if self.compute_expression:
            assert self.templates is not None
            fields["templates"] = self.templates
        return fields

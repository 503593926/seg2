from typing import TypedDict

import numpy as np
import pandas as pd


class AffineFitResult(TypedDict):
    linear: np.ndarray
    translation: np.ndarray
    x_rmse_px: float
    y_rmse_px: float
    rank: int


def _fit_axis(array_coordinate: np.ndarray, pixel_coordinate: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(array_coordinate, pixel_coordinate, 1)
    prediction = slope * array_coordinate + intercept
    residual = pixel_coordinate - prediction
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "rmse_px": float(np.sqrt(np.mean(np.square(residual)))),
    }


def _fit_affine(
    array_x: np.ndarray, array_y: np.ndarray, pixel_x: np.ndarray, pixel_y: np.ndarray
) -> AffineFitResult:
    design = np.column_stack([array_x, array_y, np.ones(len(array_x))])
    pixel = np.column_stack([pixel_x, pixel_y])
    coefficients, _, rank, _ = np.linalg.lstsq(design, pixel, rcond=None)
    prediction = design @ coefficients
    residual = pixel - prediction
    linear = coefficients[:2, :].T
    translation = coefficients[2, :]
    return {
        "linear": linear,
        "translation": translation,
        "x_rmse_px": float(np.sqrt(np.mean(np.square(residual[:, 0])))),
        "y_rmse_px": float(np.sqrt(np.mean(np.square(residual[:, 1])))),
        "rank": int(rank),
    }


def estimate_mpp_from_spatial(spatial: pd.DataFrame, spot_pitch_um: float) -> tuple[pd.Series, pd.Series]:
    required_columns = {"array_x", "array_y", "pixel_x", "pixel_y"}
    valid = spatial.dropna(subset=list(required_columns))
    array_x = valid["array_x"].to_numpy(dtype=float)
    array_y = valid["array_y"].to_numpy(dtype=float)
    pixel_x = valid["pixel_x"].to_numpy(dtype=float)
    pixel_y = valid["pixel_y"].to_numpy(dtype=float)
    fit_x = _fit_axis(array_x, pixel_x)
    fit_y = _fit_axis(array_y, pixel_y)
    px_per_array_unit_x = abs(fit_x["slope"])
    px_per_array_unit_y = abs(fit_y["slope"])
    mpp_x = spot_pitch_um / px_per_array_unit_x
    mpp_y = spot_pitch_um / px_per_array_unit_y
    mpp_effective = (mpp_x + mpp_y) / 2.0
    anisotropy_percent = 100.0 * abs(mpp_x - mpp_y) / mpp_effective
    axis_fit = pd.Series(
        {
            "mpp_x": mpp_x,
            "mpp_y": mpp_y,
            "mpp_effective": mpp_effective,
            "anisotropy_percent": anisotropy_percent,
            "x_rmse_px": fit_x["rmse_px"],
            "y_rmse_px": fit_y["rmse_px"],
        },
        name="axis_fit",
    )
    affine = _fit_affine(array_x, array_y, pixel_x, pixel_y)
    linear = affine["linear"]
    px_per_array_unit_x_affine = np.linalg.norm(linear[:, 0])
    px_per_array_unit_y_affine = np.linalg.norm(linear[:, 1])
    mpp_x_affine = spot_pitch_um / float(px_per_array_unit_x_affine)
    mpp_y_affine = spot_pitch_um / float(px_per_array_unit_y_affine)
    mpp_effective_affine = (mpp_x_affine + mpp_y_affine) / 2.0
    anisotropy_percent_affine = 100.0 * abs(mpp_x_affine - mpp_y_affine) / mpp_effective_affine
    affine_fit = pd.Series(
        {
            "mpp_x": mpp_x_affine,
            "mpp_y": mpp_y_affine,
            "mpp_effective": mpp_effective_affine,
            "anisotropy_percent": anisotropy_percent_affine,
            "x_rmse_px": affine["x_rmse_px"],
            "y_rmse_px": affine["y_rmse_px"],
            "rank": affine["rank"],
        },
        name="affine_fit",
    )
    return (axis_fit, affine_fit)

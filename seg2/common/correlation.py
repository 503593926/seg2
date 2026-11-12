"""Per-gene Spearman correlations for region-level expression evaluation."""

import numpy as np
import torch


def gene_spearman_correlations(prediction, target, observed=None):
    """Rank observed finite pairs within each gene, assigning average tied ranks.

    Columns with fewer than two pairs or a constant input have undefined SCC.
    Chunking columns keeps ranking memory bounded for large cell datasets.
    """
    from scipy.stats import rankdata

    prediction_array = prediction.detach().float().cpu().numpy()
    target_array = target.detach().float().cpu().numpy()
    if prediction_array.ndim != 2 or prediction_array.shape != target_array.shape:
        raise ValueError("Prediction and target must share a regions-by-genes shape")
    mask = (
        np.ones(target_array.shape, dtype=bool)
        if observed is None
        else observed.detach().cpu().numpy().astype(bool)
    )
    if mask.shape != target_array.shape:
        raise ValueError("Observed mask must match prediction and target")
    mask = mask & np.isfinite(prediction_array) & np.isfinite(target_array)
    correlations = np.full(prediction_array.shape[1], np.nan, dtype=np.float64)
    if not len(prediction_array):
        return torch.as_tensor(correlations, device=prediction.device)
    for start in range(0, prediction_array.shape[1], 32):
        columns = slice(start, start + 32)
        valid = mask[:, columns]
        counts = valid.sum(0)
        ranked = []
        for values in (prediction_array[:, columns], target_array[:, columns]):
            ranks = rankdata(
                np.where(valid, values, np.nan), axis=0,
                method="average", nan_policy="omit",
            )
            means = np.where(valid, ranks, 0.0).sum(0) / np.maximum(counts, 1)
            ranked.append(np.where(valid, ranks - means, 0.0))
        p, t = ranked
        denominator = np.sqrt((p * p).sum(0) * (t * t).sum(0))
        usable = (counts >= 2) & (denominator > 0)
        values = correlations[columns]
        values[usable] = np.clip(
            (p[:, usable] * t[:, usable]).sum(0) / denominator[usable], -1, 1,
        )
    return torch.as_tensor(correlations, device=prediction.device)


def spearman_metrics(prediction, target, observed=None):
    correlations = gene_spearman_correlations(prediction, target, observed)
    usable = correlations[torch.isfinite(correlations)]
    return {
        "gene_scc_genes": usable.numel(),
        "gene_prediction_scc": float(usable.mean()) if usable.numel() else float("nan"),
        "gene_prediction_scc_max": float(usable.max()) if usable.numel() else float("nan"),
        "gene_prediction_scc_min": float(usable.min()) if usable.numel() else float("nan"),
    }

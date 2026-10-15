"""仅用当前 fold 的训练表达初始化共享模板，并缓存可复用的 CPU 拟合结果。"""

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy import sparse
import torch
from threadpoolctl import threadpool_limits


def _mix_state_counts(state_count):
    if state_count < 3:
        raise ValueError("mix initialization requires tissue_state_count >= 3")
    count, remainder = divmod(state_count, 3)
    return {"random": count + (remainder > 0), "nmf": count + (remainder > 1),
            "kmeans": count}


def _fit_patterns(expression, state_count, method, seed, max_iter, n_init):
    if method == "random":
        # 与解码器默认初始化使用相同的 logit 分布；局部 RNG 不改变训练随机状态。
        logits = np.random.default_rng(seed).normal(
            scale=0.5, size=(state_count, expression.shape[1]),
        ).astype(np.float32)
        return np.exp(logits - logits.max(axis=1, keepdims=True)), {"logit_std": 0.5}
    if method == "mix":
        patterns, diagnostics = [], {}
        # NMF/K-means 使用同一份训练表达；模板顺序为随机、NMF、K-means。
        for part, count in _mix_state_counts(state_count).items():
            values, details = _fit_patterns(expression, count, part, seed, max_iter, n_init)
            patterns.append(values)
            diagnostics[part] = details
        return np.concatenate(patterns, axis=0), diagnostics

    from sklearn.cluster import KMeans
    from sklearn.decomposition import NMF

    with threadpool_limits(limits=1):
        if method == "nmf":
            if state_count > min(expression.shape):
                raise ValueError("NMF state_count must not exceed the training spot count or gene count")
            estimator = NMF(
                n_components=state_count, init="nndsvda", solver="cd",
                max_iter=max_iter, random_state=seed,
            ).fit(expression)
            patterns = estimator.components_
            diagnostics = {"iterations": int(estimator.n_iter_),
                           "reconstruction_error": float(estimator.reconstruction_err_)}
        else:
            if state_count > len(expression):
                raise ValueError("K-means state_count must not exceed the training spot count")
            estimator = KMeans(
                n_clusters=state_count, n_init=n_init, max_iter=max_iter,
                random_state=seed,
            ).fit(expression)
            # 显式按最终标签取线性表达均值，避免未完全收敛时中心与分组不一致。
            groups = [expression[estimator.labels_ == index] for index in range(state_count)]
            if any(not len(group) for group in groups):
                raise ValueError("K-means produced empty groups; reduce tissue_state_count")
            patterns = np.stack([group.mean(axis=0) for group in groups])
            diagnostics = {"iterations": int(estimator.n_iter_), "inertia": float(estimator.inertia_)}
    return patterns, diagnostics


def initialize_expression_templates(
    model, gene_adata, train_names, config, *, expression_transform, project_root, output_dir,
):
    """覆盖模板 logits 后再创建 EMA；random 模式保持原初始化及随机状态。"""
    method = str(config.get("method", "random")).lower()
    if method not in {"random", "nmf", "kmeans", "mix"}:
        raise ValueError("template_init.method must be random, nmf, kmeans or mix")
    if method == "random":
        return None
    if not model.tissue_state_count:
        raise ValueError("NMF/K-means template initialization requires tissue_state_count >= 2")
    mix_counts = _mix_state_counts(model.tissue_state_count) if method == "mix" else None

    import sklearn

    seed = int(config.get("seed", 0))
    max_iter = int(config.get("max_iter", 1000))
    n_init = int(config.get("kmeans_n_init", 10))
    smoothing = float(config.get("smoothing", 0.01))
    min_counts = float(config.get("min_counts", 100))
    if max_iter < 1 or n_init < 1 or not 0 <= smoothing < 1:
        raise ValueError("Template initialization needs positive iteration counts and smoothing in [0, 1)")
    if not np.isfinite(min_counts) or min_counts < 0:
        raise ValueError("template_init.min_counts must be finite and nonnegative; 0 disables filtering")

    train_names = sorted(set(map(str, train_names)))
    names = gene_adata.obs["name"].astype(str)
    missing = set(train_names) - set(names)
    if not train_names or missing:
        raise ValueError(f"Template initialization is missing training slides: {sorted(missing)}")
    selected = names.isin(train_names).to_numpy()
    indices = np.flatnonzero(selected)
    input_spots = len(indices)
    excluded_low_count_spots = 0
    if min_counts > 0:
        if "normalization_total_counts" not in gene_adata.obs:
            raise ValueError("Template count filtering requires obs['normalization_total_counts']; set min_counts=0 to disable")
        counts = gene_adata.obs.iloc[indices]["normalization_total_counts"].to_numpy(dtype=np.float64)
        if not np.isfinite(counts).all() or np.any(counts < 0):
            raise ValueError("Training normalization_total_counts must be finite and nonnegative")
        keep = counts >= min_counts
        excluded_low_count_spots = int((~keep).sum())
        indices = indices[keep]
    expression = gene_adata.X[indices]
    expression = expression.toarray() if sparse.issparse(expression) else expression
    expression = np.asarray(expression, dtype=np.float32)
    if not np.isfinite(expression).all() or np.any(expression < 0):
        raise ValueError("Template initialization requires finite, nonnegative expression")
    if expression_transform == "log1p":
        with np.errstate(over="ignore", invalid="ignore"):
            expression = np.expm1(expression)
    elif expression_transform != "none":
        raise ValueError("Template initialization supports expression_transform log1p or none")
    if not np.isfinite(expression).all():
        raise ValueError("Expression is not finite after conversion to the linear scale")

    gene_names = list(map(str, gene_adata.var_names))
    decoder = model.tissue_state_decoder
    if expression.shape[1] != decoder.gene_count:
        raise ValueError("Template gene count does not match the decoder")
    # 所选基因全部为零的 spot 不提供表达组成信息，也不应占据一个模板。
    nonzero = expression.sum(axis=1) > 0
    excluded_zero_expression_spots = int((~nonzero).sum())
    expression = np.ascontiguousarray(expression[nonzero])
    if not len(expression):
        raise ValueError(f"No nonzero training expression remains after template_init.min_counts={min_counts:g} filtering")
    print(
        f"Expression template initialization: {method}, min_counts={min_counts:g}, "
        f"excluded {excluded_low_count_spots} low-count and {excluded_zero_expression_spots} zero-expression spots; "
        f"{len(expression)}/{input_spots} training spots retained",
        flush=True,
    )

    specification = {
        "version": 3 if method == "mix" else 2,
        "method": method, "state_count": model.tissue_state_count,
        "gene_names": gene_names, "train_names": train_names, "input_spots": input_spots,
        "fit_spots": len(expression), "expression_transform": expression_transform,
        "seed": seed, "max_iter": max_iter, "smoothing": smoothing,
        "min_counts": min_counts, "excluded_low_count_spots": excluded_low_count_spots,
        "excluded_zero_expression_spots": excluded_zero_expression_spots,
        "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
        "expression_sha256": hashlib.sha256(expression.tobytes()).hexdigest(),
    }
    if method in {"kmeans", "mix"}:
        specification["kmeans_n_init"] = n_init
    if mix_counts is not None:
        specification.update({f"{part}_state_count": count for part, count in mix_counts.items()})
    key = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    cache_dir = Path(project_root) / str(config.get("cache_dir", "outputs/template_initialization"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{method}_{key}.npz"
    cache_hit = cache_path.is_file()
    if cache_hit:
        with np.load(cache_path, allow_pickle=False) as cached:
            templates = cached["templates"]
            diagnostics = json.loads(str(cached["diagnostics"].item()))
        print(f"Expression template initialization: {method}, loading {cache_path}", flush=True)
    else:
        print(
            f"Expression template initialization: {method}, fitting {len(expression)} training spots "
            f"× {len(gene_names)} genes into {model.tissue_state_count} templates on CPU",
            flush=True,
        )
        patterns, diagnostics = _fit_patterns(
            expression, model.tissue_state_count, method, seed, max_iter, n_init,
        )
        patterns = np.asarray(patterns, dtype=np.float32)
        if not np.isfinite(patterns).all() or np.any(patterns.sum(axis=1) <= 0):
            raise ValueError("Template fitting produced an empty or nonfinite pattern; reduce tissue_state_count")
        patterns = np.maximum(patterns, 1e-8)
        templates = patterns / patterns.sum(axis=1, keepdims=True)
        templates = (1 - smoothing) * templates + smoothing / len(gene_names)
        # 原子替换，避免并行实验读取到只写了一部分的缓存。
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".npz", delete=False) as temporary:
            np.savez_compressed(
                temporary, templates=templates,
                specification=np.asarray(json.dumps(specification)),
                diagnostics=np.asarray(json.dumps(diagnostics)),
            )
        Path(temporary.name).replace(cache_path)

    if templates.shape != tuple(decoder.template_logits.shape) or not np.isfinite(templates).all():
        raise ValueError(f"Invalid expression template cache: {cache_path}")
    if np.any(templates <= 0) or not np.allclose(templates.sum(axis=1), 1, atol=1e-6):
        raise ValueError(f"Expression templates must be positive and sum to one: {cache_path}")
    with torch.no_grad():
        decoder.template_logits.copy_(torch.from_numpy(templates).log())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {**specification, **diagnostics, "cache_key": key,
              "cache_path": str(cache_path), "cache_hit": cache_hit}
    np.savez_compressed(
        output_dir / "template_initialization.npz", templates=templates,
        gene_names=np.asarray(gene_names), train_names=np.asarray(train_names),
    )
    (output_dir / "template_initialization.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return record

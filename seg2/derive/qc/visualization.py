from pathlib import Path

import cv2
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from numpy.typing import NDArray
from skimage.morphology import thin
from skimage.segmentation import find_boundaries
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

matplotlib.use("Agg")


def qc_visualize_superpixels(
    image_bgr: NDArray[np.uint8], superpixels: NDArray[np.int32], label_mpp: float, output_dir: Path
) -> None:
    boundary = find_boundaries(superpixels, connectivity=2, mode="thick")
    boundary &= superpixels > 0
    if label_mpp >= 4.0:
        boundary = thin(boundary)
    overlay = image_bgr.copy()
    overlay[boundary] = (0, 0, 255)
    _write_image(output_dir / "snic_boundary_overlay.png", overlay, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def qc_visualize_pseudo_label(
    output_dir: Path,
    image_bgr: NDArray[np.uint8],
    superpixels: NDArray[np.int32],
    pseudo_mask: NDArray[np.uint8],
    spot_x: NDArray[np.integer],
    spot_y: NDArray[np.integer],
    spot_labels: NDArray[np.uint8],
    label_mpp: float,
    qc_spot_diameter_um: float,
) -> None:
    boundary = find_boundaries(superpixels, connectivity=2, mode="thick")
    boundary &= superpixels > 0
    boundary_overlay = image_bgr.copy()
    boundary_overlay[boundary] = (0, 0, 255)
    _write_image(output_dir / "snic_boundary_overlay.png", boundary_overlay, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    height = int(image_bgr.shape[0])
    width = int(image_bgr.shape[1])
    scale = min(1.0, 1600.0 / max(height, width))
    output_width = max(1, round(width * scale))
    output_height = max(1, round(height * scale))
    thumbnail = cv2.resize(image_bgr, (output_width, output_height), interpolation=cv2.INTER_AREA)
    mask_overlay = thumbnail.copy()
    pseudo_small = cv2.resize(pseudo_mask, (output_width, output_height), interpolation=cv2.INTER_NEAREST)
    overlay_alpha = 0.35
    label_colors = {0: (0, 180, 0), 1: (0, 0, 255), 255: (128, 128, 128)}
    for label in (0, 1):
        selected = pseudo_small == label
        color = np.asarray(label_colors[label], dtype=np.float32)
        mask_overlay[selected] = np.rint(
            mask_overlay[selected].astype(np.float32) * (1.0 - overlay_alpha) + color * overlay_alpha
        ).astype(np.uint8)
    _write_image(output_dir / "qc_mask_overlay.jpg", mask_overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
    spot_overlay = thumbnail.copy()
    overlay = mask_overlay.copy()
    spot_radius = max(2, round(qc_spot_diameter_um / (2.0 * label_mpp) * scale))
    for x, y, label in zip(spot_x, spot_y, spot_labels):
        point = (round(float(x) * scale), round(float(y) * scale))
        for target in (spot_overlay, overlay):
            cv2.circle(target, point, spot_radius + 2, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(target, point, spot_radius, label_colors[int(label)], -1, cv2.LINE_AA)
    _write_image(output_dir / "qc_spot_overlay.jpg", spot_overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
    _write_image(output_dir / "qc_overlay.jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
    radius = qc_spot_diameter_um / (2.0 * label_mpp)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    for x, y, label in zip(spot_x, spot_y, spot_labels):
        blue, green, red = label_colors[int(label)]
        elements.append(
            f'  <circle cx="{int(x)}" cy="{int(y)}" r="{radius:g}" fill="rgb({red},{green},{blue})" />'
        )
    elements.append("</svg>")
    (output_dir / "qc_spots.svg").write_text("\n".join(elements) + "\n", encoding="utf-8")
    mask_elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" shape-rendering="crispEdges">'
    ]
    for label in (0, 1):
        selected = (pseudo_mask == label).astype(np.int8)
        changes = np.diff(np.pad(selected, ((0, 0), (1, 1))), axis=1)
        rows, starts = np.nonzero(changes == 1)
        _, ends = np.nonzero(changes == -1)
        if not len(starts):
            continue
        path = " ".join(
            (f"M{start} {y}h{end - start}v1H{start}z" for y, start, end in zip(rows, starts, ends))
        )
        blue, green, red = label_colors[label]
        mask_elements.append(f'  <path fill="rgb({red},{green},{blue})" d="{path}" />')
    mask_elements.append("</svg>")
    (output_dir / "qc_pseudo_mask.svg").write_text("\n".join(mask_elements) + "\n", encoding="utf-8")


def qc_visualize_feature_map(
    feature_root: Path,
    names: list[str],
    patch_mpp: float,
    cls_mpp: float,
    n_clusters: int = 8,
    samples_per_map: int = 1024,
    rows_per_chunk: int = 16,
    random_seed: int = 0,
) -> None:
    output_root = feature_root / "visualization"
    for feature_type in ("patch", "cls"):
        filename = f"{feature_type}_map.npy"
        paths = [feature_root / name / filename for name in names]
        print(f"[{feature_type}] fitting shared PCA and K-means", flush=True)
        sampled_features = _sample_features(paths, samples_per_map, random_seed)
        pca, kmeans, cluster_remap, pca_lower, pca_upper, norm_limits = _fit_visualization_models(
            sampled_features, n_clusters, random_seed
        )
        print(
            f"[{feature_type}] PCA explained variance={pca.explained_variance_ratio_.sum():.4f}", flush=True
        )
        del sampled_features
        for name, path in zip(names, paths):
            print(f"[{feature_type}] visualizing {name}", flush=True)
            pca_rgb, norm_map, cluster_map = _transform_feature_map(
                path, pca, kmeans, cluster_remap, pca_lower, pca_upper, rows_per_chunk
            )
            _save_visualizations(
                name,
                feature_type,
                output_root / name,
                pca_rgb,
                norm_map,
                cluster_map,
                norm_limits,
                n_clusters,
                patch_mpp,
                cls_mpp,
            )
            del pca_rgb, norm_map, cluster_map
    print(f"Wrote {output_root}")


def _write_image(path: Path, image: NDArray[np.uint8], parameters: list[int]) -> None:
    cv2.imwrite(str(path), image, parameters)


def _sample_features(paths: list[Path], samples_per_map: int, random_seed: int) -> NDArray[np.float32]:
    rng = np.random.default_rng(random_seed)
    sampled: list[NDArray[np.float32]] = []
    for path in paths:
        feature_map = np.load(path, mmap_mode="r")
        flattened = feature_map.reshape(-1, feature_map.shape[-1])
        sample_count = min(samples_per_map, flattened.shape[0])
        indices = np.sort(rng.choice(flattened.shape[0], sample_count, replace=False))
        sampled.append(np.asarray(flattened[indices], dtype=np.float32))
    return np.concatenate(sampled, axis=0)


def _fit_visualization_models(
    sampled_features: NDArray[np.float32], n_clusters: int, random_seed: int
) -> tuple[PCA, KMeans, NDArray[np.uint8], NDArray[np.float32], NDArray[np.float32], tuple[float, float]]:
    pca = PCA(n_components=3, svd_solver="randomized", random_state=random_seed)
    sampled_pca = pca.fit_transform(sampled_features)
    pca_lower: NDArray[np.float32] = np.percentile(sampled_pca, 1, axis=0).astype(np.float32)
    pca_upper: NDArray[np.float32] = np.percentile(sampled_pca, 99, axis=0).astype(np.float32)
    kmeans = KMeans(n_clusters=n_clusters, n_init=100, random_state=random_seed)
    kmeans.fit(sampled_pca)
    center_order = np.lexsort(
        (kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0])
    )
    cluster_remap: NDArray[np.uint8] = np.empty(n_clusters, dtype=np.uint8)
    cluster_remap[center_order] = np.arange(n_clusters, dtype=np.uint8)
    sampled_norm: NDArray[np.float32] = np.linalg.norm(sampled_features, axis=1)
    norm_limits = (float(np.percentile(sampled_norm, 1)), float(np.percentile(sampled_norm, 99)))
    return (pca, kmeans, cluster_remap, pca_lower, pca_upper, norm_limits)


def _transform_feature_map(
    path: Path,
    pca: PCA,
    kmeans: KMeans,
    cluster_remap: NDArray[np.uint8],
    pca_lower: NDArray[np.float32],
    pca_upper: NDArray[np.float32],
    rows_per_chunk: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.uint8]]:
    feature_map = np.load(path, mmap_mode="r")
    height, width, _ = feature_map.shape
    pca_rgb = np.empty((height, width, 3), dtype=np.float32)
    norm_map = np.empty((height, width), dtype=np.float32)
    cluster_map = np.empty((height, width), dtype=np.uint8)
    for row_start in range(0, height, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, height)
        block = np.asarray(feature_map[row_start:row_end], dtype=np.float32)
        flattened = block.reshape(-1, block.shape[-1])
        pca_scores = pca.transform(flattened).astype(np.float32)
        rgb = np.zeros_like(pca_scores)
        np.divide(pca_scores - pca_lower, pca_upper - pca_lower, out=rgb, where=pca_upper != pca_lower)
        pca_rgb[row_start:row_end] = np.clip(rgb, 0, 1).reshape(row_end - row_start, width, 3)
        norm_map[row_start:row_end] = np.linalg.norm(block, axis=-1)
        cluster_map[row_start:row_end] = cluster_remap[kmeans.predict(pca_scores)].reshape(
            row_end - row_start, width
        )
    return (pca_rgb, norm_map, cluster_map)


def _create_image_figure(
    image: NDArray[np.floating] | NDArray[np.uint8],
    title: str,
    cmap: str | mcolors.Colormap | None = None,
    norm: mcolors.Normalize | None = None,
) -> tuple[Figure, Axes, AxesImage]:
    height, width = image.shape[:2]
    figure_width = 7.2
    figure_height = max(2.4, min(7.2, figure_width * height / width))
    fig, ax = plt.subplots(figsize=(figure_width, figure_height), constrained_layout=True)
    artist = ax.imshow(image, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    return (fig, ax, artist)


def _save_figure(fig: Figure, output_path: Path) -> None:
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _save_visualizations(
    name: str,
    feature_type: str,
    output_dir: Path,
    pca_rgb: NDArray[np.float32],
    norm_map: NDArray[np.float32],
    cluster_map: NDArray[np.uint8],
    norm_limits: tuple[float, float],
    n_clusters: int,
    patch_mpp: float,
    cls_mpp: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_colors = ["#0F4D92", "#42949E", "#8BCF8B", "#E9A6A1", "#B64342", "#9A4D8E", "#EA84DD", "#767676"]
    display_name = (
        f"Patch map ({patch_mpp:g} mpp)" if feature_type == "patch" else f"CLS map ({cls_mpp:g} mpp)"
    )
    fig, _, _ = _create_image_figure(pca_rgb, f"{name} | {display_name} | PCA-RGB")
    _save_figure(fig, output_dir / f"{feature_type}_pca_rgb")
    norm = mcolors.Normalize(vmin=norm_limits[0], vmax=norm_limits[1])
    fig, ax, artist = _create_image_figure(
        norm_map, f"{name} | {display_name} | L2 norm", cmap="magma", norm=norm
    )
    colorbar = fig.colorbar(artist, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Feature-vector L2 norm")
    _save_figure(fig, output_dir / f"{feature_type}_l2_norm")
    cmap = mcolors.ListedColormap(cluster_colors[:n_clusters])
    cluster_norm = mcolors.BoundaryNorm(np.arange(n_clusters + 1) - 0.5, n_clusters)
    fig, ax, artist = _create_image_figure(
        cluster_map, f"{name} | {display_name} | K-means (PCA-3)", cmap=cmap, norm=cluster_norm
    )
    colorbar = fig.colorbar(artist, ax=ax, fraction=0.035, pad=0.02, ticks=np.arange(n_clusters))
    colorbar.set_label("Cluster")
    _save_figure(fig, output_dir / f"{feature_type}_kmeans")

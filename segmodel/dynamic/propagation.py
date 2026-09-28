import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from tools.propagate_random_walker import compute_seed_votes


def standardize(values):
    return (values - values.mean(axis=0)) / np.maximum(values.std(axis=0), 1e-6)


def weighted_features(lab, learned, tissue, learned_weight):
    """组合 Lab 与模型特征，供动态 SNIC 使用。"""
    if not tissue.any():
        raise ValueError("tissue mask contains no foreground pixels")
    features = np.zeros(
        (*tissue.shape, lab.shape[-1] + learned.shape[-1]),
        dtype=np.float32,
    )
    reference_scale = np.sqrt(np.mean(lab[tissue].std(axis=0) ** 2))
    features[tissue, : lab.shape[-1]] = lab[tissue]
    features[tissue, lab.shape[-1] :] = (
        standardize(learned[tissue])
        * np.sqrt(lab.shape[-1] * learned_weight / learned.shape[-1])
        * reference_scale
    )
    return features


def propagate_features(
    superpixels,
    lab,
    learned,
    spot_x,
    spot_y,
    spot_labels,
    beta,
    learned_weight,
    topology,
):
    """以 Lab 和模型特征构图，在超像素图上传播 spot 标签。"""
    labels = topology.foreground_labels
    means = np.zeros(
        (topology.max_label + 1, lab.shape[-1] + learned.shape[-1]),
        dtype=np.float64,
    )
    flat_superpixels = superpixels.ravel()
    channels = [*np.moveaxis(lab, -1, 0), *np.moveaxis(learned, -1, 0)]
    for channel, values in enumerate(channels):
        sums = np.bincount(
            flat_superpixels,
            weights=values.ravel(),
            minlength=len(means),
        )
        means[labels, channel] = sums[labels] / topology.pixel_counts[labels]

    lab_channels = lab.shape[-1]
    means[labels, :lab_channels] = standardize(means[labels, :lab_channels])
    means[labels, lab_channels:] = standardize(means[labels, lab_channels:]) * np.sqrt(
        lab_channels * learned_weight / learned.shape[-1]
    )
    delta = means[topology.first_labels] - means[topology.second_labels]
    conductance = np.maximum(np.exp(-beta * np.square(delta).sum(axis=1)), 1e-6)
    first, second = topology.first_nodes, topology.second_nodes
    adjacency = coo_matrix(
        (
            np.concatenate((conductance, conductance)),
            (
                np.concatenate((first, second)),
                np.concatenate((second, first)),
            ),
        ),
        shape=(len(labels), len(labels)),
    ).tocsr()
    laplacian = diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency

    votes = compute_seed_votes(
        superpixels,
        spot_x,
        spot_y,
        spot_labels,
        topology,
    )[labels]
    seeded = (votes == 1) | (votes == 2)
    probability = np.full(len(labels), 0.5, dtype=np.float64)
    probability[votes == 1] = 0.0
    probability[votes == 2] = 1.0
    seeded_components = np.unique(topology.component_labels[seeded])
    solvable = ~seeded & np.isin(
        topology.component_labels,
        seeded_components,
    )
    if solvable.any():
        right_hand_side = -laplacian[solvable][:, seeded] @ probability[seeded]
        probability[solvable] = spsolve(
            laplacian[solvable][:, solvable],
            right_hand_side,
        )

    lookup = np.zeros(topology.max_label + 1, dtype=np.float32)
    lookup[labels] = np.clip(probability, 0.0, 1.0)
    return lookup[superpixels]


def snap_spots(coordinates, labels, superpixels, label_scale, max_distance):
    """将组织边缘附近的已知 spot 吸附到最近的前景超像素。"""
    coordinates = np.rint(coordinates / label_scale).astype(np.int64)
    labels = np.asarray(labels, dtype=np.uint8)
    height, width = superpixels.shape
    inside = (
        (coordinates[:, 0] >= 0)
        & (coordinates[:, 0] < width)
        & (coordinates[:, 1] >= 0)
        & (coordinates[:, 1] < height)
    )
    coordinates, labels = coordinates[inside], labels[inside]
    if not len(coordinates):
        return np.empty(0, np.int64), np.empty(0, np.int64), labels

    distance, nearest = distance_transform_edt(
        superpixels == 0,
        return_indices=True,
    )
    x, y = coordinates[:, 0].copy(), coordinates[:, 1].copy()
    outside = (
        (superpixels[y, x] == 0) & (labels != 255) & (distance[y, x] <= max_distance)
    )
    original_x, original_y = x[outside], y[outside]
    x[outside] = nearest[1, original_y, original_x]
    y[outside] = nearest[0, original_y, original_x]
    return x, y, labels


def build_pseudo_mask(probability, tissue, non_cancer_threshold, cancer_threshold):
    mask = np.full(probability.shape, 255, dtype=np.uint8)
    mask[tissue & (probability <= non_cancer_threshold)] = 0
    mask[tissue & (probability >= cancer_threshold)] = 1
    return mask


def label_change_stats(old_mask, new_mask, tissue):
    """上一轮与新伪标签的覆盖率比值、以及重叠区翻转比例（只记录，不做接受判断）。"""
    old_valid = (old_mask != 255) & tissue
    new_valid = (new_mask != 255) & tissue
    overlap = old_valid & new_valid
    coverage_ratio = float(new_valid.sum() / max(int(old_valid.sum()), 1))
    flip_fraction = (
        float((old_mask[overlap] != new_mask[overlap]).mean()) if overlap.any() else 1.0
    )
    return coverage_ratio, flip_fraction

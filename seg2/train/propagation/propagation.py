import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve

from seg2.common.rw import build_random_walker_topology, compute_seed_votes


def standardize(values):
    return (values - values.mean(axis=0)) / np.maximum(values.std(axis=0), 1e-06)


def prepare_propagation(superpixels, lab, seeds):
    """Prepare the graph using saved, already snapped label-grid seed coordinates."""
    topology = build_random_walker_topology(superpixels)
    labels = topology.foreground_labels
    lab_means = np.zeros((len(labels), lab.shape[-1]), dtype=np.float64)
    flat_superpixels = superpixels.ravel()
    for channel, values in enumerate(np.moveaxis(lab, -1, 0)):
        sums = np.bincount(flat_superpixels, weights=values.ravel(), minlength=topology.max_label + 1)
        lab_means[:, channel] = sums[labels] / topology.pixel_counts[labels]
    x, y, spot_labels = seeds["x"], seeds["y"], seeds["labels"]
    votes = compute_seed_votes(superpixels, x, y, spot_labels, topology)[labels]
    seeded = (votes == 1) | (votes == 2)
    probability = np.full(len(labels), 0.5, dtype=np.float64)
    probability[votes == 1] = 0.0
    probability[votes == 2] = 1.0
    solvable = ~seeded & np.isin(topology.component_labels, np.unique(topology.component_labels[seeded]))
    pixels = np.flatnonzero(flat_superpixels)
    return {
        "topology": topology,
        "feature_pool": {
            "pixels": pixels,
            "rows": flat_superpixels[pixels],
            "pixel_counts": topology.pixel_counts,
            "weights": None,
        },
        "lab_means": standardize(lab_means),
        "spot_labels": spot_labels,
        "seeded": seeded,
        "solvable": solvable,
        "seed_probability": probability,
        "edge_rows": np.concatenate((topology.first_nodes, topology.second_nodes)),
        "edge_columns": np.concatenate((topology.second_nodes, topology.first_nodes)),
    }


def propagate_features(superpixels, learned, beta, learned_weight, cache):
    topology = cache["topology"]
    labels = topology.foreground_labels
    lab_means = cache["lab_means"]
    lab_channels = lab_means.shape[-1]
    means = np.zeros((topology.max_label + 1, lab_channels + learned.shape[-1]), dtype=np.float64)
    means[labels, :lab_channels] = lab_means
    means[labels, lab_channels:] = standardize(learned[labels]) * np.sqrt(
        lab_channels * learned_weight / learned.shape[-1]
    )
    delta = means[topology.first_labels] - means[topology.second_labels]
    conductance = np.maximum(np.exp(-beta * np.square(delta).sum(axis=1)), 1e-06)
    adjacency = coo_matrix(
        (np.concatenate((conductance, conductance)), (cache["edge_rows"], cache["edge_columns"])),
        shape=(len(labels), len(labels)),
    ).tocsr()
    laplacian = csr_matrix(diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency)
    seeded, solvable = (cache["seeded"], cache["solvable"])
    probability = cache["seed_probability"].copy()
    if solvable.any():
        right_hand_side = -laplacian[solvable][:, seeded] @ probability[seeded]
        probability[solvable] = spsolve(laplacian[solvable][:, solvable], right_hand_side)
    lookup = np.zeros(topology.max_label + 1, dtype=np.float32)
    lookup[labels] = np.clip(probability, 0.0, 1.0)
    return lookup[superpixels]


def build_pseudo_mask(probability, tissue, non_cancer_threshold, cancer_threshold):
    mask = np.full(probability.shape, 255, dtype=np.uint8)
    mask[tissue & (probability <= non_cancer_threshold)] = 0
    mask[tissue & (probability >= cancer_threshold)] = 1
    return mask


def label_change_stats(old_mask, new_mask, tissue):
    old_valid = (old_mask != 255) & tissue
    new_valid = (new_mask != 255) & tissue
    overlap = old_valid & new_valid
    coverage_ratio = float(new_valid.sum() / max(int(old_valid.sum()), 1))
    flip_fraction = float((old_mask[overlap] != new_mask[overlap]).mean()) if overlap.any() else 1.0
    return (coverage_ratio, flip_fraction)

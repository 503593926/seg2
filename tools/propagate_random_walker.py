import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

@dataclass(frozen=True)
class RandomWalkerTopology:
    shape: tuple[int, int]
    max_label: int
    foreground_labels: NDArray[np.int32]
    pixel_counts: NDArray[np.int64]
    edge_keys: NDArray[np.int64]
    first_labels: NDArray[np.int64]
    second_labels: NDArray[np.int64]
    first_nodes: NDArray[np.int64]
    second_nodes: NDArray[np.int64]
    component_labels: NDArray[np.int32]

def build_random_walker_topology(superpixels: NDArray[np.int32]) -> RandomWalkerTopology:
    # 拓扑只依赖超像素标签, 因此可以在多个传播轮次之间复用
    assert superpixels.ndim == 2, f'superpixels must be 2-D, got {superpixels.shape}'
    max_label = int(superpixels.max())
    flat_superpixels = superpixels.ravel()
    pixel_counts = np.bincount(
        flat_superpixels,
        minlength=max_label + 1,
    ).astype(np.int64, copy=False)
    foreground_labels = cast(
        NDArray[np.int32],
        (np.flatnonzero(pixel_counts[1:]) + 1).astype(np.int32, copy=False),
    )

    base = max_label + 1
    edge_parts: list[NDArray[np.int64]] = []
    for first, second in (
        (superpixels[:, :-1], superpixels[:, 1:]),
        (superpixels[:-1, :], superpixels[1:, :]),
    ):
        boundary = (first != second) & (first > 0) & (second > 0)
        smaller = np.minimum(first[boundary], second[boundary]).astype(np.int64)
        larger = np.maximum(first[boundary], second[boundary]).astype(np.int64)
        edge_parts.append(smaller * base + larger)
    edge_keys = np.unique(np.concatenate(edge_parts))
    first_labels = edge_keys // base
    second_labels = edge_keys % base

    node_index = np.full(max_label + 1, -1, dtype=np.int64)
    node_index[foreground_labels] = np.arange(foreground_labels.size)
    first_nodes = node_index[first_labels]
    second_nodes = node_index[second_labels]
    rows = np.concatenate((first_nodes, second_nodes))
    columns = np.concatenate((second_nodes, first_nodes))
    connectivity = coo_matrix(
        (np.ones(rows.size, dtype=np.uint8), (rows, columns)),
        shape=(foreground_labels.size, foreground_labels.size),
    ).tocsr()
    _, component_labels = connected_components(connectivity, directed=False)

    return RandomWalkerTopology(
        shape=(int(superpixels.shape[0]), int(superpixels.shape[1])),
        max_label=max_label,
        foreground_labels=foreground_labels,
        pixel_counts=pixel_counts,
        edge_keys=edge_keys,
        first_labels=first_labels,
        second_labels=second_labels,
        first_nodes=first_nodes,
        second_nodes=second_nodes,
        component_labels=component_labels.astype(np.int32, copy=False),
    )

def save_random_walker_topology(path: Path, topology: RandomWalkerTopology) -> None:
    # 先写临时文件再 replace, 避免异步任务留下半写入的缓存
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('wb') as file:
        np.savez(
            file,
            shape=np.asarray(topology.shape, dtype=np.int64),
            max_label=np.asarray(topology.max_label, dtype=np.int64),
            foreground_labels=topology.foreground_labels,
            pixel_counts=topology.pixel_counts,
            edge_keys=topology.edge_keys,
            first_labels=topology.first_labels,
            second_labels=topology.second_labels,
            first_nodes=topology.first_nodes,
            second_nodes=topology.second_nodes,
            component_labels=topology.component_labels,
        )
    os.replace(temporary, path)

def load_random_walker_topology(path: Path) -> RandomWalkerTopology:
    with np.load(path, allow_pickle=False) as data:
        shape_array = data['shape']

        return RandomWalkerTopology(
            shape=(int(shape_array[0]), int(shape_array[1])),
            max_label=int(data['max_label']),
            foreground_labels=data['foreground_labels'].astype(np.int32, copy=False),
            pixel_counts=data['pixel_counts'].astype(np.int64, copy=False),
            edge_keys=data['edge_keys'].astype(np.int64, copy=False),
            first_labels=data['first_labels'].astype(np.int64, copy=False),
            second_labels=data['second_labels'].astype(np.int64, copy=False),
            first_nodes=data['first_nodes'].astype(np.int64, copy=False),
            second_nodes=data['second_nodes'].astype(np.int64, copy=False),
            component_labels=data['component_labels'].astype(np.int32, copy=False),
        )

def compute_edge_conductance(
    superpixels: NDArray[np.int32],
    pixel_features: NDArray[np.floating],
    beta: float,
    topology: RandomWalkerTopology,
    learned_weight: float = 1.0,
) -> NDArray[np.float64]:
    assert topology.shape == superpixels.shape, (
        f'topology shape {topology.shape} does not match {superpixels.shape}'
    )
    assert pixel_features.shape[:2] == superpixels.shape, 'pixel features and superpixels are not aligned'
    assert learned_weight >= 0, f'learned_weight must be non-negative, got {learned_weight}'

    foreground_labels = topology.foreground_labels
    feature_means = np.zeros(
        (topology.max_label + 1, pixel_features.shape[2]),
        dtype=np.float64,
    )
    flat_superpixels = superpixels.ravel()
    for channel in range(pixel_features.shape[2]):
        feature_sums = np.bincount(
            flat_superpixels,
            weights=pixel_features[..., channel].ravel(),
            minlength=topology.max_label + 1,
        )
        feature_means[foreground_labels, channel] = (
            feature_sums[foreground_labels] / topology.pixel_counts[foreground_labels]
        )
    foreground_features = feature_means[foreground_labels]
    feature_std = foreground_features.std(axis=0)
    feature_std[feature_std == 0] = 1.0
    feature_means[foreground_labels] = (
        foreground_features - foreground_features.mean(axis=0)
    ) / feature_std
    if learned_weight != 1.0:
        feature_means[foreground_labels, -1] *= np.sqrt(learned_weight)

    feature_distance_squared = np.sum(
        (
            feature_means[topology.first_labels]
            - feature_means[topology.second_labels]
        )
        ** 2,
        axis=1,
    )

    return np.maximum(np.exp(-beta * feature_distance_squared), 1e-6)

def compute_seed_votes(
    superpixels: NDArray[np.int32],
    spot_x: NDArray[np.integer],
    spot_y: NDArray[np.integer],
    spot_labels: NDArray[np.uint8],
    topology: RandomWalkerTopology,
) -> NDArray[np.uint8]:
    # 每个超像素的种子投票位图: 1=阴性、2=阳性、3=冲突
    spot_superpixels = superpixels[spot_y, spot_x]
    known = (spot_labels != 255) & (spot_superpixels > 0)
    vote_values = np.where(spot_labels[known] == 0, 1, 2).astype(np.uint8)
    votes = np.zeros(topology.max_label + 1, dtype=np.uint8)
    np.bitwise_or.at(votes, spot_superpixels[known], vote_values)

    return votes

def propagate_random_walker(
    superpixels: NDArray[np.int32],
    pixel_features: NDArray[np.floating],
    spot_x: NDArray[np.integer],
    spot_y: NDArray[np.integer],
    spot_labels: NDArray[np.uint8],
    beta: float,
    topology: RandomWalkerTopology | None = None,
    learned_weight: float = 1.0,
) -> tuple[NDArray[np.float32], dict[str, int]]:
    if topology is None:
        topology = build_random_walker_topology(superpixels)
    assert topology.shape == superpixels.shape, (
        f'topology shape {topology.shape} does not match {superpixels.shape}'
    )
    assert pixel_features.shape[:2] == superpixels.shape, 'pixel features and superpixels are not aligned'
    assert learned_weight >= 0, f'learned_weight must be non-negative, got {learned_weight}'

    max_label = topology.max_label
    foreground_labels = topology.foreground_labels
    first_nodes = topology.first_nodes
    second_nodes = topology.second_nodes

    conductance = compute_edge_conductance(
        superpixels,
        pixel_features,
        beta,
        topology,
        learned_weight,
    )

    rows = np.concatenate((first_nodes, second_nodes))
    columns = np.concatenate((second_nodes, first_nodes))
    weights = np.concatenate((conductance, conductance))
    adjacency = coo_matrix(
        (weights, (rows, columns)),
        shape=(len(foreground_labels), len(foreground_labels)),
    ).tocsr()
    laplacian = (diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency).tocsr()

    # 同一超像素内标签一致时作为种子, 标签冲突时不作为种子
    votes = compute_seed_votes(superpixels, spot_x, spot_y, spot_labels, topology)

    node_votes = votes[foreground_labels]
    labeled = (node_votes == 1) | (node_votes == 2)
    probability = np.full(len(foreground_labels), 0.5, dtype=np.float64)
    probability[node_votes == 1] = 0.0
    probability[node_votes == 2] = 1.0

    # 每个含有种子的连通分量独立求解离散调和方程
    component_labels = topology.component_labels
    seeded_components = np.unique(component_labels[labeled])
    solvable = ~labeled & np.isin(component_labels, seeded_components)
    if np.any(solvable):
        right_hand_side = -laplacian[solvable][:, labeled] @ probability[labeled]
        probability[solvable] = cast(
            NDArray[np.float64],
            spsolve(laplacian[solvable][:, solvable], right_hand_side),
        )
    probability = np.clip(probability, 0.0, 1.0)

    label_probability = np.zeros(max_label + 1, dtype=np.float32)
    label_probability[foreground_labels] = probability.astype(np.float32)
    probability_map = label_probability[superpixels]

    stats = {
        'graph_nodes': len(foreground_labels),
        'graph_edges': len(topology.edge_keys),
        'seed_superpixels': int(np.count_nonzero(labeled)),
        'non_cancer_seed_superpixels': int(np.count_nonzero(node_votes == 1)),
        'cancer_seed_superpixels': int(np.count_nonzero(node_votes == 2)),
        'conflicting_seed_superpixels': int(np.count_nonzero(node_votes == 3)),
        'unseeded_components': len(
            np.setdiff1d(np.unique(component_labels), seeded_components)
        ),
    }

    return probability_map, stats

import numpy as np
from numpy.typing import NDArray


def get_interpolation_plan(
    lattice_coords: NDArray[np.float32],
    pixel_coords: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.float32]]:
    integer_coords = np.rint(lattice_coords).astype(np.int64)
    lookup = {(int(x), int(y)): index for index, (x, y) in enumerate(integer_coords)}
    assert np.array_equal(lattice_coords, integer_coords) and len(lookup) == len(integer_coords), (
        'Spot interpolation requires unique integer lattice coordinates'
    )
    groups: list[list[int]] = []
    # 只填相邻实测点的边中点和完整方格中心, 不跨缺失格点外推.
    for x, y in sorted(lookup):
        for offsets in (
            ((0, 0), (1, 0)),
            ((0, 0), (0, 1)),
            ((0, 0), (1, 0), (0, 1), (1, 1)),
        ):
            corners = [(x + dx, y + dy) for dx, dy in offsets]
            if all(corner in lookup for corner in corners):
                groups.append([lookup[corner] for corner in corners])

    neighbors = np.empty((len(groups), 4), dtype=np.int64)
    weights = np.zeros((len(groups), 4), dtype=np.float32)
    for index, group in enumerate(groups):
        neighbors[index] = group[0]
        neighbors[index, :len(group)] = group
        weights[index, :len(group)] = 1.0 / len(group)
    coordinates = (pixel_coords[neighbors] * weights[..., None]).sum(axis=1, dtype=np.float32)
    return coordinates, neighbors, weights

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from seg2.ingest.her2st.qc import qc_visualize_point_mask


def _write_image(path: Path, image: NDArray[np.uint8], parameters: list[int]) -> None:
    cv2.imwrite(str(path), image, parameters)


def get_point_mask(
    spot_x: NDArray[np.int64],
    spot_y: NDArray[np.int64],
    spot_labels: NDArray[np.uint8],
    shape: tuple[int, int],
) -> NDArray[np.uint8]:
    point_mask: NDArray[np.uint8] = np.full(shape, 255, dtype=np.uint8)
    point_mask[spot_y, spot_x] = spot_labels
    return point_mask


def get_supervised_flags(
    n_spots: int, ratio: float, rng: np.random.Generator
) -> NDArray[np.bool_]:
    supervised: NDArray[np.bool_] = np.zeros(n_spots, dtype=np.bool_)
    supervised[rng.choice(n_spots, size=round(ratio * n_spots), replace=False)] = True
    return supervised


def process_sample(
    samples_root: Path,
    name: str,
    name_index: int,
    target_mpp: float,
    ratios: list[float],
    seed: int,
) -> None:
    sample_dir = samples_root / name
    spatial_dir = sample_dir / "spatial"
    masks_dir = sample_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    image_mat = cv2.imread(
        str(spatial_dir / f"he_{target_mpp:g}mpp.png"), cv2.IMREAD_COLOR
    )
    image_bgr = np.asarray(image_mat, dtype=np.uint8)
    canvas_shape = (image_bgr.shape[0], image_bgr.shape[1])
    spatial = pd.read_csv(spatial_dir / f"spatial_{target_mpp:g}mpp.tsv", sep="\t")
    spots = spatial.loc[
        spatial["selected"].eq(1)
        & spatial["in_content"]
        & spatial["spot_label"].ne(255)
    ]
    spot_x = np.rint(spots["pixel_x"].to_numpy(dtype=np.float64)).astype(np.int64)
    spot_y = np.rint(spots["pixel_y"].to_numpy(dtype=np.float64)).astype(np.int64)
    spot_labels: NDArray[np.uint8] = spots["spot_label"].to_numpy(dtype=np.uint8)
    selection = pd.DataFrame({"spot_id": spots["spot_id"].to_numpy()})
    for ratio in ratios:
        percent = round(ratio * 100)
        flags = get_supervised_flags(
            len(spots), ratio, np.random.default_rng([seed, name_index])
        )
        _write_image(
            masks_dir / f"point_mask_{percent}.png",
            get_point_mask(
                spot_x[flags], spot_y[flags], spot_labels[flags], canvas_shape
            ),
            [cv2.IMWRITE_PNG_COMPRESSION, 9],
        )
        selection[f"ratio@{ratio:g}"] = flags
        qc_visualize_point_mask(sample_dir, percent, image_bgr, spots, flags)
        print(
            f"  ratio {ratio:g}: {int(flags.sum())}/{len(spots)} spots supervised",
            flush=True,
        )
    selection.to_csv(spatial_dir / "point_selection.tsv", sep="\t", index=False)


def run(cfg, force=False):
    samples_root = Path(str(cfg.dataset.data_dir)).expanduser().resolve() / "samples"
    names = list(cfg.dataset.labeled_names)
    ratios = list(map(float, cfg.prep.seeds.ratios))
    for index, name in enumerate(names):
        expected = [
            samples_root / name / "masks" / f"point_mask_{round(ratio * 100)}.png"
            for ratio in ratios
        ]
        expected.append(samples_root / name / "spatial/point_selection.tsv")
        if force or not all(p.is_file() for p in expected):
            process_sample(
                samples_root,
                name,
                index,
                float(cfg.prep.resolution.target_mpp),
                ratios,
                int(cfg.prep.seeds.seed),
            )

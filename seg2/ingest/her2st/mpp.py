from pathlib import Path

import pandas as pd

from seg2.common.imaging.estimate_mpp_from_spatial import estimate_mpp_from_spatial


def run(cfg, force=False):
    dataset = cfg.dataset
    raw_data_root = Path(str(dataset.raw_root))
    save_path = (
        Path(str(cfg.dataset.data_dir)).expanduser().resolve().parent / "mpp.csv"
    )
    if save_path.is_file() and (not force):
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    patient_names = sorted(dataset.patient_names)
    spot_pitch_um = float(dataset.array_pitch_um)
    records: list[dict[str, object]] = []
    for patient_name in patient_names:
        for selection_path in sorted(
            (raw_data_root / "ST-spotfiles").glob(f"{patient_name}*_selection.tsv")
        ):
            name = selection_path.stem.removesuffix("_selection")
            spatial = pd.read_csv(selection_path, sep="\t")
            selected = spatial.loc[spatial["selected"] == 1].rename(
                columns={"new_x": "array_x", "new_y": "array_y"}
            )
            axis_fit, affine_fit = estimate_mpp_from_spatial(selected, spot_pitch_um)
            records.append(
                {
                    "name": name,
                    **{f"axis_{key}": value for key, value in axis_fit.items()},
                    **{f"affine_{key}": value for key, value in affine_fit.items()},
                }
            )
    table = pd.DataFrame(records).set_index("name").sort_index()
    table.to_csv(save_path)
    print(f"\nWrote {save_path}")

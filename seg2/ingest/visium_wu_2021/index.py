from pathlib import Path

import pandas as pd


def build_index(raw_root):
    records = []
    for sample in sorted((Path(raw_root) / "raw").iterdir()):
        counts = sample / "filtered_count_matrix"
        if not sample.is_dir() or not counts.is_dir():
            continue
        paths = {
            "count_dir": counts,
            "position_path": sample / "spatial/tissue_positions_list.csv",
            "annotation_path": sample / "metadata.csv",
            "scale_path": sample / "spatial/scalefactors_json.json",
            "image_path": sample / "spatial/tissue_hires_image.png",
        }
        required = [
            counts / name
            for name in ("matrix.mtx.gz", "features.tsv.gz", "barcodes.tsv.gz")
        ]
        required += [path for key, path in paths.items() if key != "count_dir"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"[{sample.name}] Missing inputs: {missing}")
        records.append(
            {"name": sample.name, **{key: str(path) for key, path in paths.items()}}
        )
    if not records:
        raise ValueError("No Wu 2021 Visium sections found")
    return pd.DataFrame(data=records)

from pathlib import Path

import pandas as pd


def build_index(raw_root):
    raw_root = Path(raw_root).expanduser().resolve()
    metadata = pd.read_csv(raw_root / "raw/metadata.csv", dtype=str)
    records = []
    fields = {
        "image_path": ("he", "histology_image"),
        "count_path": ("stdata", "count_matrix"),
        "spot_path": ("spots", "spot_coordinates"),
        "annotation_path": ("coords", "tumor_annotation"),
    }
    for row in metadata.to_dict("records"):
        record = {
            "section_name": f"{row['patient']}_{row['replicate']}",
        }
        for key, (directory, column) in fields.items():
            path = (raw_root / "raw" / directory / row[column]).resolve()
            record[key] = str(path)
        records.append(record)

    table = pd.DataFrame(records).sort_values("section_name").reset_index(drop=True)

    return table

from pathlib import Path
import pandas as pd
from ingest.utils.estimate_mpp_from_spatial import estimate_mpp_from_spatial

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    raw_data_root = Path('/home/yfm/data/bioinfo/her2st-master/data')
    save_path = project_root / 'data/her2st/mpp.csv'
    save_path.parent.mkdir(parents=True, exist_ok=True)

    slice_names = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']
    spot_pitch_um = 200.0

    records: list[dict[str, object]] = []
    for name in slice_names:
        selection_path = raw_data_root / 'ST-spotfiles' / f'{name}_selection.tsv'
        spatial = pd.read_csv(selection_path, sep='\t')
        selected = spatial.loc[spatial['selected'] == 1].rename(
            columns={'new_x': 'array_x', 'new_y': 'array_y'},
        )
        axis_fit, affine_fit = estimate_mpp_from_spatial(selected, spot_pitch_um)
        records.append({
            'name': name,
            **axis_fit.add_prefix('axis_').to_dict(),
            **affine_fit.add_prefix('affine_').to_dict(),
        })

    table = pd.DataFrame(records).set_index('name').sort_index()
    table.to_csv(save_path)

    print(f'\nWrote {save_path}')


if __name__ == '__main__':
    main()

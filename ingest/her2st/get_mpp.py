from pathlib import Path
import pandas as pd
from ingest.utils.estimate_mpp_from_spatial import estimate_mpp_from_spatial

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    raw_data_root = Path('/home/yfm/data/bioinfo/her2st-master/data')
    save_path = project_root / 'data/her2st/mpp.csv'
    save_path.parent.mkdir(parents=True, exist_ok=True)

    patient_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    spot_pitch_um = 200.0

    records: list[dict[str, object]] = []
    # 按患者查找全部切片，逐张估算原始图像的 MPP。
    for patient_name in patient_names:
        for selection_path in sorted((raw_data_root / 'ST-spotfiles').glob(f'{patient_name}*_selection.tsv')):
            name = selection_path.stem.removesuffix('_selection')
            spatial = pd.read_csv(selection_path, sep='\t')
            selected = spatial.loc[spatial['selected'] == 1].rename(
                columns={'new_x': 'array_x', 'new_y': 'array_y'},
            )
            axis_fit, affine_fit = estimate_mpp_from_spatial(selected, spot_pitch_um)
            records.append({
                'name': name,
                **{f'axis_{key}': value for key, value in axis_fit.items()},
                **{f'affine_{key}': value for key, value in affine_fit.items()},
            })

    # 保留切片名索引，供下游逐张缩放图像与空间坐标。
    table = pd.DataFrame(records).set_index('name').sort_index()
    table.to_csv(save_path)

    print(f'\nWrote {save_path}')


if __name__ == '__main__':
    main()

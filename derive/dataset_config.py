from pathlib import Path

import yaml


def load_dataset_config() -> dict:
    project_root = Path(__file__).resolve().parents[1]
    with open(project_root / 'derive/dataset.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    with open(project_root / 'config/resolution.yaml', 'r', encoding='utf-8') as f:
        resolution = yaml.safe_load(f)
    dataset = str(config['dataset'])
    section = config['datasets'][dataset]
    samples_root = project_root / 'data' / dataset / f"{float(resolution['target_mpp']):g}mpp/samples"
    # 以实际生成的原始 counts 为准，按患者和切片排序。
    expression_names = [
        path.parent.name
        for patient in section['patient_names']
        for path in sorted(samples_root.glob(f'{patient}*/gene_counts.tsv.gz'))
    ]
    return {'dataset': dataset, **section, 'expression_names': expression_names}

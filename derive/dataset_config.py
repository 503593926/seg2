from pathlib import Path
import yaml



def load_dataset_config() -> dict:
    with open(str(Path(__file__).resolve().parent / 'dataset.yaml'), 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
    dataset = str(config["dataset"])
    section = config["datasets"][dataset]
    return {"dataset": dataset, **section}

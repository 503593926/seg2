import gc
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from seg2.derive import sample_names
from seg2.derive.pfm.common import FeatureMapExtractor
from seg2.derive.qc.visualization import qc_visualize_feature_map


def process_sample(
    extractor: FeatureMapExtractor,
    name: str,
    spatial_dir: Path,
    output_dir: Path,
    target_mpp: float,
) -> None:
    he_path = spatial_dir / f"he_{target_mpp:g}mpp.png"
    with Image.open(he_path) as source:
        image: NDArray[np.uint8] = np.asarray(source.convert("RGB"))
    patch_map, cls_map = extractor.extract(image)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "patch_map.npy", patch_map)
    np.save(output_dir / "cls_map.npy", cls_map)
    print(f"[{name}] patch_map={patch_map.shape} cls_map={cls_map.shape}", flush=True)


def run(cfg):
    from importlib import import_module

    dataset = cfg.dataset
    registry = {
        "h_optimus_0": "HO0Extractor",
        "h_optimus_1": "HO1Extractor",
        "uni": "UNIExtractor",
        "uni2": "UNI2Extractor",
        "virchow": "VirchowExtractor",
        "virchow2": "Virchow2Extractor",
        "conch": "CONCHExtractor",
        "genbio_pathfm": "GenBioPathFMExtractor",
    }
    options = cfg.prep.features
    weights_root = Path(str(cfg.paths.weights_root)).expanduser().resolve()
    root, target_mpp = (
        Path(str(dataset.data_dir)).expanduser().resolve(),
        float(cfg.prep.resolution.target_mpp),
    )
    names = sample_names(dataset=dataset)
    for key in options.extractors:
        extractor_type = getattr(import_module(f"seg2.derive.pfm.{key}"), registry[key])
        extractor = extractor_type(
            batch_size=int(options.batch_size),
            device=str(options.device),
            tile_size=int(cfg.prep.resolution.tile_size),
            resize=bool(options.resize),
            overlap_average=bool(options.overlap_average),
            overlap_cls=bool(options.overlap_cls),
            weights_root=weights_root,
        )
        rows, _ = extractor.patch_grid
        patch_mpp = extractor.tile_size * target_mpp / rows
        cls_mpp = extractor.cls_stride * target_mpp
        tag = f"tile_{extractor.tile_size}_mpp_{target_mpp:g}_resize_{int(extractor.resize)}"
        if extractor.overlap_average:
            tag += "_overlap_1"
        if extractor.overlap_cls:
            tag += "_cls_overlap_1"
        model_tag = extractor_type.__name__.removesuffix("Extractor").lower()
        output = root / f"feature_{model_tag}" / tag
        output.mkdir(parents=True, exist_ok=True)
        (output / "metadata.json").write_text(
            json.dumps({"patch_mpp": patch_mpp, "cls_mpp": cls_mpp}, indent=2) + "\n"
        )
        for name in names:
            process_sample(
                extractor=extractor,
                name=name,
                spatial_dir=root / "samples" / name / "spatial",
                output_dir=output / name,
                target_mpp=target_mpp,
            )
        del extractor
        gc.collect()
        qc_visualize_feature_map(
            feature_root=output, names=names, patch_mpp=patch_mpp, cls_mpp=cls_mpp
        )

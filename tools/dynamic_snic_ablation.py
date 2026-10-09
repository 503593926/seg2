"""动态 SNIC（整片重算）对可微传播的影响: 同一模型特征下只换拓扑。

流程:
1. 载入某个已完成 fold 的 final_model.pt，整片推理拿到 7mpp 传播特征；
2. 用模型特征在**整张切片**上重算 SNIC（与 DynamicPropagationCallback._update_slide 同口径）；
3. 在「prepare 拓扑」与「刷新拓扑」上各跑一次可微传播（同种子、同节点特征），
   按专家 mask 比较覆盖率与边界带 Dice。

用法:
  python tools/dynamic_snic_ablation.py --run outputs/her2st/2026-10-08/23-08-09/seed_0/fold_00_test_C1 --slide B1
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmodel.dataset.datamodule import FoldDataModule
from segmodel.dynamic.propagation import weighted_features
from segmodel.inference.predictor import predict_slide
from segmodel.models.segmentation import SegmentationModel
from tools.compare_pseudo_labels import expert_targets
from tools.propagate_pixel_vs_superpixel import seeds_on_grid, solve, superpixel_edges
from tools.snic_features import prepare_snic_lab, run_snic


def propagation_metrics(superpixels, tissue, lab, learned, spots, labels, fraction, valid_count, steps=400):
    """在给定拓扑上跑一次可微传播, 返回覆盖率与边界带 Dice。"""
    channels = [lab[..., c] for c in range(lab.shape[-1])] + [
        learned[..., c] for c in range(learned.shape[-1])
    ]
    nodes = np.unique(superpixels)
    nodes = nodes[nodes > 0]
    flat = superpixels.reshape(-1).astype(np.int64)
    nmax = int(superpixels.max()) + 1
    counts = np.bincount(flat, minlength=nmax).astype(np.float64)
    means = np.zeros((nmax, len(channels)))
    for index, values in enumerate(channels):
        means[:, index] = np.bincount(
            flat, weights=np.asarray(values, dtype=np.float64).reshape(-1), minlength=nmax
        ) / np.maximum(counts, 1)
    means = means[nodes]
    means = (means - means.mean(0)) / np.maximum(means.std(0), 1e-6)

    lookup = -np.ones(nmax, dtype=np.int64)
    lookup[nodes] = np.arange(nodes.size)
    first, second = superpixel_edges(superpixels, lookup, tissue)

    seed_grid = seeds_on_grid(spots[0], spots[1], labels, tissue, 0)
    seed_values = np.full(nodes.size, 255.0)
    known = seed_grid != 255
    positive = np.bincount(
        lookup[superpixels[known]], weights=(seed_grid[known] == 1), minlength=nodes.size
    )
    negative = np.bincount(
        lookup[superpixels[known]], weights=(seed_grid[known] == 0), minlength=nodes.size
    )
    seed_values[positive > negative] = 1.0
    seed_values[negative > positive] = 0.0

    values, _, used, residual = solve(
        first, second, means[first] - means[second], seed_values, nodes.size, 0.5, steps, 1e-8
    )
    field = np.full(tissue.shape, 0.5)
    field[tissue] = values[lookup[superpixels[tissue]]]
    hard = np.full(field.shape, 255, dtype=np.int64)
    confident = (field <= 0.2) | (field >= 0.8)
    hard[confident] = (field[confident] >= 0.5).astype(np.int64)

    cell_valid = valid_count >= 4
    truth = (fraction >= 0.5) & cell_valid
    covered = cell_valid & (hard != 255)
    kernel = np.ones((5, 5), dtype=np.uint8)
    band = (
        cv2.dilate(truth.astype(np.uint8), kernel)
        - cv2.erode(truth.astype(np.uint8), kernel)
    ).astype(bool)
    selected = covered & band
    prediction, target = hard[selected] == 1, truth[selected]
    return {
        "nodes": int(nodes.size),
        "pcg": int(used),
        "residual": float(residual),
        "coverage": float(covered.sum() / max(cell_valid.sum(), 1)),
        "band_dice": float(
            2 * (prediction & target).sum() / max(prediction.sum() + target.sum(), 1)
        ),
    }


def partition_overlap(old, new):
    """旧分区每个节点被新分区某个节点覆盖的最大比例（近似一致性）。"""
    total = int((old > 0).sum())
    covered = 0
    for value in np.unique(old):
        if value == 0:
            continue
        mask = old == value
        counts = np.bincount(new[mask].astype(np.int64))
        covered += int(counts.max()) if counts.size else 0
    return covered / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="fold 目录（含 final_model.pt）")
    parser.add_argument("--slide", default="B1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learned-weight", type=float, default=1.0)
    parser.add_argument("--superpixel-size-um", type=float, default=64.0)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=400)
    arguments = parser.parse_args()

    run_dir = Path(arguments.run)
    cfg = OmegaConf.load(run_dir / "config.yaml")
    slide_name = arguments.slide
    label_scale = int(round(float(cfg.resolution.label_mpp) / float(cfg.resolution.target_mpp)))
    fold_test = run_dir.name.split("_")[-1]
    train_names = [
        name
        for name in cfg.experiment.dataset.all_names
        if name != fold_test
    ]

    data_module = FoldDataModule(train_names, [fold_test], 0, cfg)
    data_module.setup("fit")
    slide = data_module.train_slides[slide_name]
    tissue = slide.tissue_mask_bl
    root = Path(str(cfg.experiment.dataset.data_root))

    print(f"[{slide_name}] 标签网格 {tissue.shape}, label_scale={label_scale}")
    model = SegmentationModel(
        pfm_patch_dims=slide.patch_dims,
        pfm_cls_dims=slide.cls_dims,
        **OmegaConf.to_container(cfg.experiment.model.image, resolve=True),
    )
    state = torch.load(run_dir / "final_model.pt", map_location="cpu")
    model.load_state_dict(state["state_dict"])
    device = torch.device(arguments.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    tiling = state["inference_config"]
    probability, learned = predict_slide(
        model,
        slide,
        crop_size=tiling["crop_size"],
        stride=tiling["stride"],
        alignment=slide.alignment,
        device=device,
        batch_size=arguments.batch_size,
    )
    print(f"整片推理完成: 概率 {probability.shape}, 模型特征 {learned.shape}")
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    image = cv2.resize(
        slide.image, (tissue.shape[1], tissue.shape[0]), interpolation=cv2.INTER_AREA
    )
    lab = prepare_snic_lab(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    features = weighted_features(lab, learned, tissue, arguments.learned_weight)
    label_mpp = float(cfg.resolution.label_mpp)
    segments = max(
        1,
        round(tissue.sum() / (arguments.superpixel_size_um / label_mpp) ** 2),
    )
    refreshed, _ = run_snic(
        features, tissue, segments, compactness=arguments.compactness
    )
    prepared = np.asarray(slide.superpixels, dtype=np.int64)
    print(
        f"拓扑: prepare {int(prepared.max())} 节点, 刷新 {int(refreshed.max())} 节点, "
        f"覆盖一致性 {partition_overlap(prepared, refreshed):.3f}"
    )

    fraction, valid_count, _ = expert_targets(str(root), slide_name, label_scale)
    genes = __import__("anndata").read_h5ad(
        root / "gene_expression" / "task1_loocv" / f"test_{fold_test}.h5ad"
    ).obs
    genes = genes[genes["name"].astype(str) == slide_name]
    spots = (
        np.rint(genes["pixel_x"].to_numpy() / label_scale).astype(np.int64),
        np.rint(genes["pixel_y"].to_numpy() / label_scale).astype(np.int64),
    )
    labels = genes["spot_label"].to_numpy().astype(np.int64)

    for tag, superpixels in (("prepare 拓扑", prepared), ("刷新拓扑", refreshed.astype(np.int64))):
        result = propagation_metrics(
            superpixels, tissue, lab, np.asarray(learned), spots, labels,
            fraction, valid_count, steps=arguments.steps,
        )
        print(
            f"{tag:<12} {result['nodes']:>5} 节点 pcg {result['pcg']:>4} 步 残差 {result['residual']:.1e} | "
            f"coverage={result['coverage']:.3f} band_dice={result['band_dice']:.4f}"
        )


if __name__ == "__main__":
    main()

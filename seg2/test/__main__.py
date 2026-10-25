import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--gene-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slides", nargs="+", required=True)
    parser.add_argument("--feature-maps", nargs="+", required=True)
    parser.add_argument("--target-mpp", type=float, required=True)
    parser.add_argument("--crop-size", type=int, default=896)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--inference-batch-size", type=int, default=1)
    parser.add_argument("--spot-diameter-um", type=float, required=True)
    parser.add_argument("--expression-transform", default="log1p")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--visualization-top-n", type=int, default=0)
    args = parser.parse_args()
    from seg2.test import run_test

    run_test(
        args.artifact,
        args.data_root,
        args.gene_path,
        args.output,
        names=args.slides,
        feature_maps=args.feature_maps,
        target_mpp=args.target_mpp,
        crop_size=args.crop_size,
        stride=args.stride,
        device=args.device,
        precision=args.precision,
        inference_batch_size=args.inference_batch_size,
        spot_diameter_um=args.spot_diameter_um,
        expression_transform=args.expression_transform,
        threshold=args.threshold,
        visualization_top_n=args.visualization_top_n,
    )


if __name__ == "__main__":
    main()

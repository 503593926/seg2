from pathlib import Path
from seg2.test import run_test

def main():
    project_root = Path(__file__).resolve().parents[2]
    model_dir = project_root / "outputs/her2st/2026-10-29/13-55-09/test_A/seed_0"
    data_root = project_root / "data/her2st/1mpp"

    run_test(
        artifact_path=model_dir / "final_ema_model.pt",
        data_root=data_root,
        gene_path=data_root / "gene_expression/patient_loocv/test_A.h5ad",
        output_dir=model_dir / "test",
        names=["A1", "A2", "A3", "A4", "A5", "A6"],
        feature_maps=[
            "feature_ho1/tile_224_mpp_1_resize_1_overlap_1",
            "feature_uni2/tile_224_mpp_1_resize_1_overlap_1",
            "feature_virchow2/tile_224_mpp_1_resize_1_overlap_1",
        ],
        target_mpp=1.0,
        crop_size=896,
        stride=None,
        device="cuda:0",
        precision="bf16-mixed",
        inference_batch_size=8,
        spot_diameter_um=100.0,
        expression_transform="log1p",
        threshold=0.5,
        visualization_top_n=0,
    )


if __name__ == "__main__":
    main()

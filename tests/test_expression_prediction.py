"""稠密表达的 spot 聚合、监督隔离、模型保存和独立评估。"""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from segmodel.modules.spot_pooling import pool_spot_expression
from segmodel.modules.propagation import DifferentiablePropagation
from segmodel.pipeline import _save_final_model
from segmodel.models.segmentation import SegmentationModel
from segmodel.inference.validation import gene_prediction_metrics
from segmodel.training.losses import gene_prediction_losses
from test_propagation_training import build_batch, build_module


def case(batch_size=2):
    batch = build_batch(batch=batch_size)
    spots = torch.tensor([[5., 5.], [26., 5.], [47., 47.], [68., 68.], [89., 89.], [26., 89.]])
    batch["spots"] = [spots.clone() for _ in range(batch_size)]
    batch["spot_labels"] = [torch.tensor([0, 0, 1, 0, 1, 1]) for _ in range(batch_size)]
    propagation = DifferentiablePropagation(
        label_scale=7, gate_range=0, feature_gradients=True, steps=100,
        non_cancer_threshold=0.2, cancer_threshold=0.8,
    )
    module = build_module(propagation, gene_count=8)
    module.image_cfg.loss_weights.segmentation = 0
    module.image_cfg.loss_weights.gene_prediction = 1
    module.log_dict = lambda *args, **kwargs: None
    module.log = lambda *args, **kwargs: None
    return module, batch


class SpotPoolingTest(unittest.TestCase):
    def test_dense_pool_preserves_within_node_expression_and_excludes_background(self):
        grid = torch.tensor([[0, 1, 1], [0, 1, 1], [0, 1, 1]])
        field = torch.tensor([[[100., 1., 9.], [100., 1., 9.], [100., 1., 9.]]], requires_grad=True)
        prediction, valid = pool_spot_expression(field, grid, torch.tensor([[1., 1.], [20., 20.]]), 1., 1)
        torch.testing.assert_close(prediction, torch.tensor([[3.], [0.]]))
        self.assertEqual(valid.tolist(), [True, False])
        prediction[valid].sum().backward()
        self.assertEqual(float(field.grad[:, :, 0].sum()), 0)
        self.assertGreater(float(field.grad.sum()), 0)


class GeneCorrelationLossTest(unittest.TestCase):
    @staticmethod
    def expression_case(predictions, targets):
        field = predictions.transpose(1, 2).unsqueeze(2).detach().requires_grad_()
        batch_size, spot_count, _ = predictions.shape
        spots = torch.stack((torch.arange(spot_count).float(), torch.zeros(spot_count)), dim=1)
        batch = {
            "superpixels": torch.ones(batch_size, 1, 1, spot_count, dtype=torch.long),
            "spots": [spots.clone() for _ in range(batch_size)],
            "gene_expression": list(targets.unbind()),
        }
        return {"gene_expression": field}, batch

    def losses(self, prediction, target, **kwargs):
        output, batch = self.expression_case(prediction, target)
        losses, count = gene_prediction_losses(
            output, batch, .1, 1, expression_transform="none",
            weights={"gene_prediction": 1, "gene_correlation": .1}, **kwargs,
        )
        return losses, count, output["gene_expression"]

    def test_correlation_is_gene_wise_and_preserves_mse_calibration(self):
        values = torch.arange(8).float() + 1
        target = torch.stack((values, values.flip(0) + 20), dim=1)[None]
        prediction = target * torch.tensor([3., .2]) + torch.tensor([100., 0.])
        losses, count, _ = self.losses(prediction, target)
        self.assertEqual(count, 8)
        self.assertAlmostEqual(float(losses["gene_correlation"].detach()), 0., places=6)
        torch.testing.assert_close(losses["gene_prediction"], (prediction - target).square().mean())

    def test_crops_are_not_pooled_into_one_correlation(self):
        target = torch.stack((torch.arange(8).float() + 1, torch.arange(8).float() + 100))[:, :, None]
        losses, _, _ = self.losses(target.flip(1), target)
        self.assertAlmostEqual(float(losses["gene_correlation"].detach()), 2., places=6)

    def test_matches_pearson_value_and_gradient(self):
        generator = torch.Generator().manual_seed(42)
        prediction = torch.rand(1, 8, 3, generator=generator)
        target = torch.rand(1, 8, 3, generator=generator)
        losses, _, field = self.losses(prediction, target)
        reference = torch.stack([
            1 - torch.corrcoef(torch.stack((field[0, gene, 0], target[0, :, gene])))[0, 1]
            for gene in range(3)
        ]).mean()
        torch.testing.assert_close(losses["gene_correlation"], reference)
        torch.testing.assert_close(
            torch.autograd.grad(losses["gene_correlation"], field, retain_graph=True)[0],
            torch.autograd.grad(reference, field)[0],
        )

    def test_constant_prediction_is_penalized_and_has_finite_nonzero_gradient(self):
        target = (torch.arange(8).float() + 1)[None, :, None].requires_grad_()
        losses, _, field = self.losses(torch.ones_like(target), target)
        self.assertEqual(float(losses["gene_correlation"].detach()), 1.)
        losses["gene_correlation"].backward()
        self.assertTrue(bool(torch.isfinite(field.grad).all()))
        self.assertGreater(float(field.grad.abs().sum()), 0)
        self.assertIsNone(target.grad)

    def test_constant_truth_is_excluded_without_changing_other_genes(self):
        values = torch.arange(8).float() + 1
        target = torch.stack((values, torch.ones(8)), dim=1)[None]
        prediction = torch.stack((values.flip(0), values), dim=1)[None]
        losses, _, field = self.losses(prediction, target)
        self.assertAlmostEqual(float(losses["gene_correlation"].detach()), 2., places=6)
        losses["gene_correlation"].backward()
        self.assertEqual(float(field.grad[:, 1].abs().sum()), 0)
        self.assertTrue(bool(torch.isfinite(field.grad).all()))

    def test_insufficient_spots_or_all_constant_truth_keep_differentiable_zero(self):
        for count, constant in ((0, False), (1, False), (7, False), (8, True)):
            with self.subTest(spots=count, constant_truth=constant):
                target = torch.ones(1, count, 2) if constant else torch.arange(count).float()[None, :, None].repeat(1, 1, 2)
                losses, actual_count, field = self.losses(torch.ones_like(target), target)
                self.assertEqual(actual_count, count)
                self.assertEqual(float(losses["gene_correlation"].detach()), 0)
                losses["gene_correlation"].backward()
                self.assertTrue(bool(torch.isfinite(field.grad).all()))

    def test_disabled_correlation_keeps_original_mse_and_skips_correlation_computation(self):
        prediction = (torch.arange(8).float() + 1)[None, :, None]
        output, batch = self.expression_case(prediction, prediction * 2)
        with patch("segmodel.training.losses._gene_correlation_terms", side_effect=AssertionError):
            losses, count = gene_prediction_losses(output, batch, .1, 1)
        self.assertEqual(set(losses), {"gene_prediction"})
        self.assertEqual(count, 8)
        torch.testing.assert_close(losses["gene_prediction"], (prediction.log1p() - (prediction * 2)).square().mean())

    def test_log_transform_precedes_correlation_and_cpu_autocast_is_safe(self):
        prediction = (torch.arange(8).float() + 1)[None, :, None]
        target = 3 * prediction.log1p() + 2
        output, batch = self.expression_case(prediction, target)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            losses, _ = gene_prediction_losses(
                output, batch, .1, 1, weights={"gene_prediction": 1, "gene_correlation": .1},
            )
        self.assertEqual(losses["gene_correlation"].dtype, torch.float32)
        self.assertAlmostEqual(float(losses["gene_correlation"].detach()), 0., places=6)
        (losses["gene_prediction"] + .1 * losses["gene_correlation"]).backward()
        self.assertTrue(bool(torch.isfinite(output["gene_expression"].grad).all()))

    def test_correlation_alone_updates_gene_head_and_is_logged_with_its_weight(self):
        module, batch = case(batch_size=1)
        module.eval()
        module.image_cfg.gene_correlation_min_spots = 3
        for name in module.image_cfg.loss_weights:
            module.image_cfg.loss_weights[name] = 0
        module.image_cfg.loss_weights.gene_correlation = .1
        logged = {}
        module.log_dict = lambda values, **kwargs: logged.update(values)
        loss = module.training_step(batch)
        self.assertIn("train/gene_correlation", logged)
        self.assertNotIn("train/gene_prediction", logged)
        torch.testing.assert_close(loss, .1 * logged["train/gene_correlation"])
        loss.backward()
        self.assertGreater(float(module.model.gene_head.weight.grad.abs().sum()), 0)



class ExpressionPredictionTest(unittest.TestCase):
    def test_validation_logs_independent_expression_metrics_and_handles_undefined_precision(self):
        module, batch = case(batch_size=1)
        module.validation_roots = {"heldout": Path("/unused")}
        module.tiling = {"crop_size": 112, "stride": 112, "alignment": 28}
        slide = SimpleNamespace(name="heldout", height=112, width=112, label_scale=7)
        slide.inputs = lambda *args: {
            "image": batch["image"][0],
            "patch": [value[0] for value in batch["patch"]],
            "cls": [value[0] for value in batch["cls"]],
        }
        module.ema_model.eval()
        with torch.no_grad():
            # 强制全阴性预测, precision 为 None 时验证日志仍应正常。
            module.ema_model.segmentation_head.weight.zero_()
            module.ema_model.segmentation_head.bias.fill_(-10)
        adata = SimpleNamespace(
            obs=pd.DataFrame({
                "name": ["heldout"] * 6 + ["other"],
                "pixel_x": [*batch["spots"][0][:, 0].tolist(), 0.],
                "pixel_y": [*batch["spots"][0][:, 1].tolist(), 0.],
            }),
            X=np.zeros((7, 8), dtype=np.float32),
        )
        module._trainer = SimpleNamespace(datamodule=SimpleNamespace(val_gene_adata=adata))
        logged = {}
        module.log_dict = lambda values, **kwargs: logged.update(values)
        with patch("segmodel.inference.validation.load_expert_masks", return_value=(
            np.ones((112, 112), dtype=np.uint8), np.ones((112, 112), dtype=bool),
        )), patch("segmodel.inference.validation.read_image", return_value=np.ones((16, 16), dtype=np.uint8)):
            metrics = module.validation_step(slide, 0)
        self.assertIsNone(metrics["precision"])
        self.assertNotIn("val/precision", logged)
        self.assertEqual(logged["val/gene_spots"], 6)
        self.assertGreater(logged["val/gene_prediction"], 0)
        self.assertIn("val/gene_prediction_pcc", logged)
        self.assertIn("val/dice", logged)


    def test_independent_gene_metrics_use_all_valid_spots_and_skip_constant_genes(self):
        tissue = torch.ones(3, 3, dtype=torch.bool)
        field = torch.stack((torch.arange(9).reshape(3, 3).float() + 1, torch.ones(3, 3)))
        spots = torch.tensor([[0., 0.], [1., 1.], [2., 2.]])
        prediction, valid = pool_spot_expression(field, tissue, spots, .1, 1)
        truth = prediction.log1p()
        metrics = gene_prediction_metrics(field, tissue, spots, truth, .1, 1)
        self.assertEqual(metrics["gene_spots"], 3)
        self.assertEqual(metrics["gene_pcc_genes"], 1)
        self.assertEqual(metrics["gene_prediction"], 0)
        self.assertAlmostEqual(metrics["gene_prediction_pcc"], 1., places=6)
        changed = gene_prediction_metrics(field, tissue, spots, truth + 2, .1, 1)
        self.assertAlmostEqual(changed["gene_prediction"], 4., places=6)
        empty = gene_prediction_metrics(field, torch.zeros_like(tissue), spots, truth, .1, 1)
        self.assertEqual(empty["gene_spots"], 0)
        self.assertTrue(torch.isnan(torch.tensor(empty["gene_prediction"])))


    def test_dense_head_preserves_backbone_rng_and_exports_for_image_only_inference(self):
        plain = build_module()
        plain_rng = torch.random.get_rng_state()
        guided = build_module(gene_count=8)
        self.assertTrue(torch.equal(plain_rng, torch.random.get_rng_state()))
        for key, value in plain.model.state_dict().items():
            torch.testing.assert_close(value, guided.model.state_dict()[key], rtol=0, atol=0)
        self.assertIsNotNone(guided.ema_model.gene_head)
        config_dir = str((Path(__file__).resolve().parents[1] / "segmodel/configs").resolve())
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config")
        cfg.experiment.model.image.joint_prediction = False
        cfg.experiment.model.image.tissue_state_count = 0
        cfg.experiment.model.image.encoder_chans = [8, 12, 16]
        cfg.experiment.model.image.decoder_chans = [12, 8]
        cfg.experiment.model.image.global_dim = 8
        cfg.experiment.model.image.spatial_dim = 4
        cfg.experiment.model.image.propagation_channels = 4
        cfg.experiment.model.image.spatial_pool = 4
        cfg.experiment.model.image.backbone_init = "pytorch"
        guided.model.gene_names = [f"g{i}" for i in range(8)]
        with tempfile.TemporaryDirectory() as directory:
            _save_final_model(cfg, guided.model, {}, Path(directory))
            artifact = torch.load(Path(directory) / "final_model.pt", weights_only=True)
        self.assertEqual(artifact["gene_names"], guided.model.gene_names)
        restored = SegmentationModel(**artifact["model_config"])
        restored.load_state_dict(artifact["state_dict"], strict=True)
        _, batch = case()
        guided.model.eval()
        restored.eval()
        inputs = (batch["image"], batch["patch"], batch["cls"], batch["pfm_mask"])
        expected = guided.model(*inputs)
        actual = restored(*inputs)
        torch.testing.assert_close(actual["gene_expression"], expected["gene_expression"], rtol=0, atol=0)
        torch.testing.assert_close(actual["logits"], expected["logits"], rtol=0, atol=0)


    def test_real_expression_changes_loss_but_not_predictions_or_rw(self):
        module, batch = case()
        module.eval()
        def predict(inputs):
            output = module(inputs["image"], inputs["patch"], inputs["cls"], inputs["pfm_mask"])
            propagated = module.propagation(inputs["superpixels"], output["features"], inputs["image"],
                                            inputs["spot_labels"], inputs["spots"])
            losses, count = gene_prediction_losses(output, inputs, module.spot_radius_px, module.label_scale)
            return output, propagated, losses["gene_prediction"], count
        output, propagated, loss, count = predict(batch)
        self.assertEqual(count, 12)
        changed = copy.deepcopy(batch)
        changed["gene_expression"] = [value + 2 for value in changed["gene_expression"]]
        output2, propagated2, loss2, _ = predict(changed)
        torch.testing.assert_close(output["gene_expression"], output2["gene_expression"], rtol=0, atol=0)
        torch.testing.assert_close(propagated[1], propagated2[1], rtol=0, atol=0)
        self.assertNotEqual(float(loss.detach()), float(loss2.detach()))

    def test_expression_supervision_needs_no_class_seeds_or_graph_edges(self):
        module, batch = case()
        batch["superpixels"].fill_(1)
        for label in batch["spot_labels"]:
            label.fill_(255)
        output = module(batch["image"], batch["patch"], batch["cls"], batch["pfm_mask"])
        losses = {}
        module._update_propagated_loss(losses, output, batch)
        self.assertEqual(set(losses), {"gene_prediction"})
        losses["gene_prediction"].backward()
        self.assertGreater(float(module.model.gene_head.weight.grad.abs().sum()), 0)

    def test_empty_spots_have_differentiable_zero_loss(self):
        module, batch = case()
        for key in ("spots", "spot_labels", "gene_expression"):
            batch[key] = [value[:0] for value in batch[key]]
        loss = module.training_step(batch)
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()

    def test_gene_head_occurs_once_in_optimizer_without_teacher_parameters(self):
        module, batch = case()
        module.training_step(batch).backward()
        self.assertGreater(float(module.model.gene_head.weight.grad.abs().sum()), 0)
        ids = [id(p) for group in module.configure_optimizers()["optimizer"].param_groups for p in group["params"]]
        expected = {id(p) for p in module.model.parameters() if p.requires_grad}
        expected.update(id(p) for p in module.propagation.parameters() if p.requires_grad)
        self.assertEqual(set(ids), expected)
        self.assertEqual(len(ids), len(set(ids)))


class PredictionConfigurationTest(unittest.TestCase):
    def test_single_her2st_entry_composes_patient_state_prediction(self):
        directory = Path(__file__).resolve().parents[1] / "segmodel/configs"
        self.assertEqual([path.name for path in (directory / "experiment").glob("*.yaml")], ["her2st.yaml"])
        with initialize_config_dir(config_dir=str(directory), version_base=None):
            cfg = compose(config_name="config")
        self.assertEqual(set(cfg.experiment), {"dataset", "model", "training", "augment", "propagation"})
        self.assertEqual(set(cfg.experiment.model), {"image"})
        self.assertEqual(set(cfg.experiment.training), {"image", "runtime"})
        self.assertEqual(cfg.experiment.dataset.split_mode, "patient_loocv")
        self.assertEqual(len(cfg.experiment.dataset.expression_names), 36)
        self.assertFalse(cfg.experiment.model.image.joint_prediction)
        self.assertEqual(cfg.experiment.model.image.tissue_state_count, 32)
        self.assertEqual(cfg.experiment.training.image.loss_weights.gene_correlation, .5)
        self.assertEqual(cfg.experiment.training.image.gene_correlation_min_spots, 8)


if __name__ == "__main__":
    unittest.main()

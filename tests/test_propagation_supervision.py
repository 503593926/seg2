"""数值梯度、隐藏真值隔离、批图独立性与训练接线的机制测试。"""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from segmodel.modules.graph_measurements import (
    measurement_operator, pool_spot_expression, reconstruct_expression, spot_measurements,
)
from segmodel.modules.propagation import DifferentiablePropagation
from segmodel.pipeline import _build_propagation_supervision, _save_final_model
from segmodel.models.segmentation import SegmentationModel
from segmodel.inference.validation import gene_prediction_metrics
from segmodel.training.propagation_supervision import PropagationSupervision
from test_propagation_training import build_batch, build_module, _to_cuda


def case(mode="propagation", batch_size=2):
    batch = build_batch(batch=batch_size)
    spots = torch.tensor([[5., 5.], [26., 5.], [47., 47.], [68., 68.], [89., 89.], [26., 89.]])
    batch["spots"] = [spots.clone() for _ in range(batch_size)]
    batch["spot_labels"] = [torch.tensor([0, 0, 1, 0, 1, 1]) for _ in range(batch_size)]
    batch["spot_query"] = [torch.tensor([False, False, True, True, False, False]) for _ in range(batch_size)]
    propagation = DifferentiablePropagation(
        label_scale=7, gate_range=0, feature_gradients=True, steps=100,
        non_cancer_threshold=0.2, cancer_threshold=0.8,
    )
    module = build_module(propagation, gene_count=8 if mode == "prediction" else 0)
    supervision = PropagationSupervision(
        module.model.feature_channels, 8, spot_radius_px=5, label_scale=7,
        gene_mode=mode, gene_steps=100, seed=0,
    )
    module.propagation_supervision = supervision
    weights = module.image_cfg.loss_weights
    weights.segmentation = 0.0
    weights.propagation_query = 1.0
    weights.gene_auxiliary = 0.1 if mode == "auxiliary" else 0.0
    weights.gene_propagation = 0.1 if mode == "propagation" else 0.0
    weights.gene_prediction = 0.1 if mode == "prediction" else 0.0
    return module, batch


def forward_graph(module, batch):
    output = module(batch["image"], batch["patch"], batch["cls"], batch["pfm_mask"])
    supervision = module.propagation_supervision
    masks = supervision.split_spots(batch)
    labels = [label.masked_fill(mask, 255) for label, mask in zip(batch["spot_labels"], masks)]
    propagated, graph = module.propagation(
        batch["superpixels"], output["features"], batch["image"], labels,
        batch["spots"], return_graph=True,
        **supervision.molecular_options(output),
    )
    return output, propagated, graph, supervision.observations(graph, batch, masks)


class MeasurementNumericsTest(unittest.TestCase):
    def test_solution_and_implicit_gradient_match_dense_solve(self):
        dtype = torch.float64
        edges = torch.stack((torch.arange(5), torch.arange(1, 6)), 1)
        weights = torch.tensor([.7, 1.1, .4, 1.5, .8], dtype=dtype, requires_grad=True)
        # 圆盘横跨多个节点, 不只测试点种子的特例; 一个基因完全为零。
        matrix = torch.tensor([[.6, .4, 0, 0, 0, 0], [0, 0, 0, 0, .3, .7]], dtype=dtype).to_sparse()
        expression = torch.tensor([[2., 5., 0.], [8., 1., 0.]], dtype=dtype)
        prior = expression.mean(0).expand(6, -1)
        value, residual = reconstruct_expression(
            weights, edges, matrix, expression, prior, steps=100, tolerance=1e-12,
        )
        operator = measurement_operator(torch.eye(6, dtype=dtype), weights, edges, matrix, 1., .001)
        expected = torch.linalg.solve(operator, matrix.T @ expression + .001 * prior).clamp_min(0)
        torch.testing.assert_close(value, expected, rtol=1e-10, atol=1e-10)
        actual_gradient = torch.autograd.grad(value[2].square().sum(), weights, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected[2].square().sum(), weights)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-9, atol=1e-9)
        self.assertLess(float(residual), 1e-10)
        self.assertGreater(float(actual_gradient.abs().sum()), 0)

    def test_disk_measurement_respects_tissue_and_linear_aggregation(self):
        grid = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        matrix = spot_measurements(grid, torch.tensor([[1., 1.], [20., 20.]]), torch.tensor([1, 2]), 1., 1)
        torch.testing.assert_close(matrix[0], torch.tensor([.75, .25]))
        self.assertEqual(float(matrix[1].sum()), 0)
        linear = torch.tensor([[1.], [9.]])
        self.assertAlmostEqual(float((matrix @ linear)[0].log1p()), float(torch.tensor(4.).log()), places=6)
        self.assertNotAlmostEqual(float((matrix @ linear.log1p())[0]), float(torch.tensor(4.).log()), places=3)

    def test_dense_pool_preserves_within_node_expression_and_excludes_background(self):
        grid = torch.tensor([[0, 1, 1], [0, 1, 1], [0, 1, 1]])
        field = torch.tensor([[[100., 1., 9.], [100., 1., 9.], [100., 1., 9.]]], requires_grad=True)
        prediction, valid = pool_spot_expression(field, grid, torch.tensor([[1., 1.], [20., 20.]]), 1., 1)
        torch.testing.assert_close(prediction, torch.tensor([[3.], [0.]]))
        self.assertEqual(valid.tolist(), [True, False])
        prediction[valid].sum().backward()
        self.assertEqual(float(field.grad[:, :, 0].sum()), 0)
        self.assertGreater(float(field.grad.sum()), 0)


class HiddenSpotSupervisionTest(unittest.TestCase):
    def test_query_labels_and_expression_do_not_change_predictions(self):
        module, batch = case()
        module.eval()
        _, result, graph, observed = forward_graph(module, batch)
        predicted, query, _ = module.propagation_supervision.predict_expression(graph, observed)
        changed = copy.deepcopy(batch)
        for index, mask in enumerate(batch["spot_query"]):
            changed["spot_labels"][index][mask] = 1 - changed["spot_labels"][index][mask]
            changed["gene_expression"][index][mask] += 10
        _, result2, graph2, observed2 = forward_graph(module, changed)
        predicted2, query2, _ = module.propagation_supervision.predict_expression(graph2, observed2)
        torch.testing.assert_close(result[1], result2[1], rtol=0, atol=0)
        torch.testing.assert_close(predicted[query], predicted2[query2], rtol=0, atol=0)
        changed["gene_expression"][0][0] += 1
        _, _, graph3, observed3 = forward_graph(module, changed)
        predicted3, _, _ = module.propagation_supervision.predict_expression(graph3, observed3)
        self.assertGreater(float((predicted3 - predicted).abs().sum().detach()), 0)

    def test_queries_are_grouped_and_split_rng_is_independent(self):
        module, batch = case()
        batch["spots"][0][1] = batch["spots"][0][2]
        masks = module.propagation_supervision.split_spots(batch)
        self.assertTrue(bool(masks[0][1] and masks[0][2]))
        del batch["spot_query"]
        before = torch.random.get_rng_state()
        first = module.propagation_supervision.split_spots(batch, step=3)
        second = module.propagation_supervision.split_spots(batch, step=3)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a, b))
            self.assertFalse(bool(a.all()))

    def test_gene_modes_use_identical_masks_with_unknown_class_labels(self):
        module, batch = case()
        del batch["spot_query"]
        batch["spot_labels"][0][2] = 255
        expected = None
        for mode in ("none", "auxiliary", "propagation", "prediction"):
            before = torch.random.get_rng_state()
            supervision = PropagationSupervision(8, 8, 5, label_scale=7, gene_mode=mode, seed=0)
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
            masks = supervision.split_spots(batch, step=2)
            if expected is not None:
                for a, b in zip(masks, expected):
                    self.assertTrue(torch.equal(a, b))
            expected = masks

    def test_auxiliary_and_propagation_skip_unanchored_components(self):
        module, batch = case()
        _, _, graph, observed = forward_graph(module, batch)
        # 将一个 query 独立成无 anchor 分量, 两种基因监督都应排除它。
        query_row = observed["query"].nonzero(as_tuple=True)[0][0]
        rows, nodes = observed["matrix"].indices()
        query_nodes = nodes[rows == query_row]
        graph["components"] = graph["components"].clone()
        graph["components"][query_nodes] = graph["components"].max() + 1
        usable = module.propagation_supervision.gene_query_mask(graph, observed)
        self.assertFalse(bool(usable[query_row]))

    def test_soft_pseudo_target_is_detached_from_features(self):
        module, batch = case("none")
        output, _, _, _ = forward_graph(module, batch)
        output["features"] = output["features"].detach().requires_grad_(True)
        output["logits"] = output["logits"].detach().requires_grad_(True)
        losses = {}
        module._update_propagated_loss(losses, output, batch)
        gradient = torch.autograd.grad(losses["propagation"], output["features"], allow_unused=True)[0]
        self.assertIsNone(gradient)
        self.assertTrue(losses["propagation_query"].requires_grad)

    def test_gene_graphs_do_not_mix_batch_samples(self):
        module, batch = case()
        module.eval()
        _, _, graph, observed = forward_graph(module, batch)
        first, _, _ = module.propagation_supervision.predict_expression(graph, observed)
        changed = copy.deepcopy(batch)
        changed["gene_expression"][1] += 5
        _, _, graph2, observed2 = forward_graph(module, changed)
        second, _, _ = module.propagation_supervision.predict_expression(graph2, observed2)
        torch.testing.assert_close(first[:6], second[:6], rtol=2e-5, atol=2e-5)

    def test_direct_query_and_gene_losses_each_reach_image_features(self):
        for name in ("propagation_query", "gene_propagation"):
            with self.subTest(loss=name):
                module, batch = case()
                for key in module.image_cfg.loss_weights:
                    module.image_cfg.loss_weights[key] = float(key == name)
                loss = module.training_step(batch)
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                gradient = sum(float(p.grad.abs().sum()) for p in module.model.unet.parameters() if p.grad is not None)
                self.assertGreater(gradient, 1e-8)

    def test_auxiliary_gene_head_is_optimized(self):
        module, batch = case("auxiliary")
        module.training_step(batch).backward()
        head = module.propagation_supervision.gene_head
        self.assertGreater(float(head.weight.grad.abs().sum()), 0)
        optim = module.configure_optimizers()["optimizer"]
        ids = [id(p) for group in optim.param_groups for p in group["params"]]
        self.assertIn(id(head.weight), ids)
        self.assertEqual(len(ids), len(set(ids)))

    def test_query_bce_exits_autocast_and_preserves_feature_gradients(self):
        module, batch = case("none")
        output, _, _, _ = forward_graph(module, batch)
        output["features"] = output["features"].detach().to(torch.bfloat16).requires_grad_(True)
        supervision = module.propagation_supervision
        _, expected, _ = supervision(module.propagation, output, batch)
        expected_gradient = torch.autograd.grad(expected["propagation_query"], output["features"])[0]
        binary_cross_entropy = torch.nn.functional.binary_cross_entropy

        def checked_bce(probability, target):
            # CPU 也检查 CUDA 的约束, 避免没有 GPU 时漏掉这个回归。
            self.assertFalse(torch.is_autocast_enabled(probability.device.type))
            self.assertEqual(probability.dtype, torch.float32)
            self.assertEqual(target.dtype, torch.float32)
            return binary_cross_entropy(probability, target)

        with patch("segmodel.training.propagation_supervision.F.binary_cross_entropy", side_effect=checked_bce) as bce:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                _, losses, _ = supervision(module.propagation, output, batch)
        bce.assert_called_once()
        torch.testing.assert_close(losses["propagation_query"], expected["propagation_query"])
        losses["propagation_query"].backward()
        self.assertTrue(bool(torch.isfinite(output["features"].grad).all()))
        self.assertGreater(float(output["features"].grad.abs().sum()), 0)
        torch.testing.assert_close(output["features"].grad, expected_gradient)

    def test_sparse_measurements_exit_autocast_in_all_gene_modes(self):
        sparse_mm = torch.sparse.mm

        def checked_mm(matrix, values):
            self.assertFalse(torch.is_autocast_enabled(values.device.type))
            self.assertEqual(matrix.dtype, torch.float32)
            self.assertEqual(values.dtype, torch.float32)
            return sparse_mm(matrix, values)

        for mode in ("none", "auxiliary", "propagation", "prediction"):
            with self.subTest(mode=mode):
                module, batch = case(mode)
                output, _, _, _ = forward_graph(module, batch)
                output["features"] = output["features"].detach().to(torch.bfloat16).requires_grad_(True)
                with patch("torch.sparse.mm", side_effect=checked_mm):
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        _, losses, _ = module.propagation_supervision(module.propagation, output, batch)
                    sum(losses.values()).backward()
                self.assertTrue(bool(torch.isfinite(output["features"].grad).all()))
                self.assertGreater(float(output["features"].grad.abs().sum()), 0)

    def test_empty_spots_and_no_anchors_have_differentiable_zero_loss(self):
        for empty in (False, True):
            module, batch = case()
            if empty:
                for key in ("spots", "spot_labels", "spot_query", "gene_expression", "spot_ids"):
                    batch[key] = [value[:0] for value in batch[key]]
            else:
                batch["spot_query"] = [torch.ones_like(value) for value in batch["spot_query"]]
            loss = module.training_step(batch)
            self.assertTrue(bool(torch.isfinite(loss)))
            self.assertEqual(float(loss.detach()), 0)
            loss.backward()

    def test_gene_supervision_works_without_class_seeds(self):
        module, batch = case()
        for label in batch["spot_labels"]:
            label.fill_(255)
        for key in module.image_cfg.loss_weights:
            module.image_cfg.loss_weights[key] = float(key == "gene_propagation")
        loss = module.training_step(batch)
        self.assertGreater(float(loss.detach()), 0)
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.model.unet.parameters() if p.grad is not None), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_cuda_mixed_precision_forward_and_backward(self):
        for mode in ("none", "auxiliary", "propagation", "prediction"):
            with self.subTest(mode=mode):
                module, batch = case(mode)
                module, batch = module.cuda(), _to_cuda(batch)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = module.training_step(batch)
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in module.parameters() if p.grad is not None))


class PredictedExpressionGuidanceTest(unittest.TestCase):
    def test_validation_logs_independent_expression_metrics_and_handles_undefined_precision(self):
        module, batch = case("prediction", batch_size=1)
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

    def test_real_expression_changes_loss_but_not_predictions_or_propagation(self):
        module, batch = case("prediction")
        module.eval()
        output, propagated, graph, _ = forward_graph(module, batch)
        loss, count = module.propagation_supervision.prediction_loss(output, batch)
        self.assertEqual(count, 12)  # 包括类别传播的 anchor, 不只计算 4 个 query。
        changed = copy.deepcopy(batch)
        changed["gene_expression"] = [value + 2 for value in changed["gene_expression"]]
        output2, propagated2, graph2, _ = forward_graph(module, changed)
        loss2, _ = module.propagation_supervision.prediction_loss(output2, changed)
        torch.testing.assert_close(output["gene_expression"], output2["gene_expression"], rtol=0, atol=0)
        torch.testing.assert_close(graph["conductance"], graph2["conductance"], rtol=0, atol=0)
        torch.testing.assert_close(propagated[1], propagated2[1], rtol=0, atol=0)
        self.assertNotEqual(float(loss.detach()), float(loss2.detach()))

    def test_molecular_field_changes_conductance_and_labels_with_bounded_strength(self):
        module, batch = case("prediction")
        output, _, _, _ = forward_graph(module, batch)
        # 同一视觉图上加入明确的分子落差, 不改变真实表达或类别种子。
        field = torch.ones_like(output["gene_expression"])
        field[:, :, :, :6] = 1000
        output["gene_expression"] = field
        supervision = module.propagation_supervision
        masks = supervision.split_spots(batch)
        labels = [value.masked_fill(mask, 255) for value, mask in zip(batch["spot_labels"], masks)]
        inputs = (batch["superpixels"], output["features"], batch["image"], labels, batch["spots"])
        plain, plain_graph = module.propagation(*inputs, return_graph=True)
        guided, graph = module.propagation(*inputs, return_graph=True, **supervision.molecular_options(output))
        modulation = graph["molecular_modulation"]
        self.assertGreaterEqual(float(modulation.min()), .7 - 1e-6)
        self.assertLessEqual(float(modulation.max()), 1.3 + 1e-6)
        torch.testing.assert_close(graph["conductance"], plain_graph["conductance"] * modulation)
        self.assertGreater(float((guided[1] - plain[1]).abs().sum().detach()), 1e-5)
        supervision.gene_edge_strength = 0
        disabled, _ = module.propagation(*inputs, return_graph=True, **supervision.molecular_options(output))
        torch.testing.assert_close(disabled[1], plain[1], rtol=0, atol=0)

    def test_gene_and_class_query_losses_each_train_dense_gene_head(self):
        for name in ("gene_prediction", "propagation_query"):
            with self.subTest(loss=name):
                module, batch = case("prediction")
                for key in module.image_cfg.loss_weights:
                    module.image_cfg.loss_weights[key] = float(key == name)
                loss = module.training_step(batch)
                loss.backward()
                self.assertTrue(bool(torch.isfinite(module.model.gene_head.weight.grad).all()))
                self.assertGreater(float(module.model.gene_head.weight.grad.abs().sum()), 0)
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.model.unet.parameters() if p.grad is not None), 0)
                ids = [id(p) for group in module.configure_optimizers()["optimizer"].param_groups for p in group["params"]]
                self.assertEqual(ids.count(id(module.model.gene_head.weight)), 1)

    def test_detaching_molecular_edges_blocks_class_query_gradient_to_gene_head(self):
        module, batch = case("prediction")
        module.propagation_supervision.detach_gene_edges = True
        for key in module.image_cfg.loss_weights:
            module.image_cfg.loss_weights[key] = float(key == "propagation_query")
        module.training_step(batch).backward()
        gradient = module.model.gene_head.weight.grad
        self.assertTrue(gradient is None or float(gradient.abs().sum()) == 0)

    def test_expression_supervision_needs_no_class_seeds_or_graph_edges(self):
        module, batch = case("prediction")
        batch["superpixels"].fill_(1)  # 单节点没有边, 类别传播返回 None。
        for label in batch["spot_labels"]:
            label.fill_(255)
        output = module(batch["image"], batch["patch"], batch["cls"], batch["pfm_mask"])
        propagated, losses, diagnostics = module.propagation_supervision(module.propagation, output, batch)
        self.assertIsNone(propagated)
        self.assertEqual(diagnostics["gene_spots"], 12)
        losses["gene_prediction"].backward()
        self.assertGreater(float(module.model.gene_head.weight.grad.abs().sum()), 0)

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
            cfg = compose(config_name="config", overrides=["experiment=her2st_gene_guided"])
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
        _, batch = case("prediction")
        guided.model.eval()
        restored.eval()
        inputs = (batch["image"], batch["patch"], batch["cls"], batch["pfm_mask"])
        expected = guided.model(*inputs)
        actual = restored(*inputs)
        torch.testing.assert_close(actual["gene_expression"], expected["gene_expression"], rtol=0, atol=0)
        torch.testing.assert_close(actual["logits"], expected["logits"], rtol=0, atol=0)


class SupervisionConfigurationTest(unittest.TestCase):
    def test_presets_build_and_leaking_pseudo_label_loss_is_rejected(self):
        config_dir = str((Path(__file__).resolve().parents[1] / "segmodel/configs").resolve())
        module, _ = case()
        data = SimpleNamespace(seed=0, train_gene_adata=SimpleNamespace(n_vars=8, uns={"log1p": {}}))
        slide = SimpleNamespace(spot_radius_px=50., label_scale=7)
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            for name in ("her2st", "her2st_masked", "her2st_query", "her2st_gene_auxiliary", "her2st_gene_propagation", "her2st_gene_shuffle", "her2st_gene_prediction", "her2st_gene_guided"):
                cfg = compose(config_name="config", overrides=[f"experiment={name}"])
                model = build_module(gene_count=8).model if name in {"her2st_gene_prediction", "her2st_gene_guided"} else module.model
                built = _build_propagation_supervision(cfg, data, model, module.propagation, slide)
                self.assertEqual(built is not None, name != "her2st")
            cfg = compose(config_name="config", overrides=["experiment=her2st_query"])
            cfg.experiment.training.image.loss_weights.segmentation = 1
            with self.assertRaisesRegex(ValueError, "reveal queries"):
                _build_propagation_supervision(cfg, data, module.model, module.propagation, slide)


if __name__ == "__main__":
    unittest.main()

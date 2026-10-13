"""批量表达监督、预计算 RW 拓扑和持久化 worker 的语义回归。"""

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader
from lightning_fabric.utilities.apply_func import move_data_to_device

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from segmodel.dataset.augment import _transform_geometry
from segmodel.dataset.datamodule import FoldDataModule
from segmodel.dataset.segmentation import EpochSampler, collate_samples
from segmodel.modules.propagation import DifferentiablePropagation, pcg_solve, prepare_graph
from segmodel.modules.spot_pooling import collate_spot_pools, pool_spot_batch, prepare_spot_pool
from segmodel.modules.tissue_states import TissueStateDecoder
from segmodel.training.losses import gene_prediction_losses
from test_patient_supervision import write_fixture
from test_propagation_training import build_batch
from test_tissue_states import state_module


def dense_reference(field, tissue, spots, radius, scale):
    yy, xx = torch.meshgrid(torch.arange(field.shape[-2]), torch.arange(field.shape[-1]), indexing='ij')
    rows = []
    for x, y in (spots / scale).tolist():
        keep = ((xx - x).square() + (yy - y).square() <= (radius / scale)**2) & (tissue > 0)
        rows.append(field[:, keep].float().mean(1) if bool(keep.any()) else field.sum((1, 2)) * 0)
    return torch.stack(rows) if rows else field.sum((1, 2))[None][:0]


class BatchedExpressionTest(unittest.TestCase):
    def test_fractional_disks_background_overlap_and_gradients_match_dense_reference(self):
        torch.manual_seed(7)
        fields = torch.randn(2, 5, 12, 13, requires_grad=True)
        tissue = torch.randint(0, 2, (2, 12, 13))
        spots = [torch.tensor([[0., 0.], [14.3, 16.8], [16.1, 18.2], [100., 100.]]), torch.empty(0, 2)]
        plans = [prepare_spot_pool(grid, positions, 9.1, 3) for grid, positions in zip(tissue, spots)]
        plan = collate_spot_pools(plans, fields.shape[-2:])
        actual = pool_spot_batch(fields, plan)
        expected = torch.cat([dense_reference(field, grid, positions, 9.1, 3)[item['indices']]
                              for field, grid, positions, item in zip(fields, tissue, spots, plans)])
        torch.testing.assert_close(actual, expected)
        coefficients = torch.randn_like(actual)
        torch.testing.assert_close(
            torch.autograd.grad((actual * coefficients).sum(), fields)[0],
            torch.autograd.grad((expected * coefficients).sum(), fields)[0],
        )

    def test_factorized_mse_pcc_and_parameter_gradients_match_dense_decoder(self):
        for mixed in (False, True):
            with self.subTest(mixed_precision=mixed):
                self.check_factorized_decoder(mixed)

    def check_factorized_decoder(self, mixed):
        torch.manual_seed(19)
        decoder = TissueStateDecoder(5, 11, state_count=4, hidden_dim=6)
        optimized = copy.deepcopy(decoder)
        features = torch.randn(3, 5, 12, 13, requires_grad=True)
        other_features = features.detach().clone().requires_grad_()
        tissue = torch.ones(3, 1, 12, 13, dtype=torch.bool)
        tissue[0, :, :3, :4] = False
        tissue[2] = False
        spots = [torch.tensor([[2.3, 3.8], [7.1, 5.4], [8.9, 8.2], [10., 10.]]),
                 torch.tensor([[4., 4.], [5., 5.], [6., 6.], [7., 7.]]), torch.tensor([[4., 4.]])]
        batch = {'tissue': tissue, 'superpixels': tissue, 'spots': spots,
                 'gene_expression': [torch.rand(len(positions), 11) for positions in spots]}
        batch['spot_pool'] = collate_spot_pools(
            [prepare_spot_pool(grid[0], positions, 2.1, 1) for grid, positions in zip(tissue, spots)],
            tissue.shape[-2:],
        )
        options = dict(weights={'gene_prediction': 1, 'gene_correlation': .5}, correlation_min_spots=3)
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=mixed):
            dense = decoder(features)
            fast = optimized(other_features, dense_expression=False)
            reference, count = gene_prediction_losses(dense, batch, 2.1, 1, **options)
            actual, actual_count = gene_prediction_losses(fast, batch, 2.1, 1, **options)
        self.assertNotIn('gene_expression', fast)
        self.assertEqual(count, actual_count)
        for name in reference:
            torch.testing.assert_close(actual[name], reference[name], rtol=3e-5, atol=3e-6)
        (reference['gene_prediction'] + .5 * reference['gene_correlation']).backward()
        (actual['gene_prediction'] + .5 * actual['gene_correlation']).backward()
        tolerances = dict(rtol=.03, atol=3e-5) if mixed else dict(rtol=3e-4, atol=1e-6)
        torch.testing.assert_close(other_features.grad, features.grad, **tolerances)
        for first, second in zip(decoder.parameters(), optimized.parameters()):
            if first.grad is None:
                self.assertIsNone(second.grad)
            else:
                torch.testing.assert_close(second.grad, first.grad, **tolerances)

    def test_empty_factorized_supervision_remains_differentiable(self):
        decoder = TissueStateDecoder(5, 11, state_count=4, hidden_dim=6)
        output = decoder(torch.randn(1, 5, 3, 3), dense_expression=False)
        batch = {'superpixels': torch.zeros(1, 1, 3, 3), 'spots': [torch.empty(0, 2)],
                 'gene_expression': [torch.empty(0, 11)]}
        losses, count = gene_prediction_losses(output, batch, 1, 1)
        self.assertEqual(count, 0)
        self.assertEqual(float(losses['gene_prediction'].detach()), 0)
        losses['gene_prediction'].backward()
        for parameter in (decoder.template_logits, decoder.state_head.weight, decoder.amplitude_head.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_bf16_training_uses_prepared_metadata_without_dense_gene_field(self):
        batch = build_batch()
        module = state_module().cuda()
        batch['spot_pool'] = collate_spot_pools([
            prepare_spot_pool(grid[0], spots, module.spot_radius_px, module.label_scale)
            for grid, spots in zip(batch['superpixels'], batch['spots'])
        ], batch['superpixels'].shape[-2:])
        batch['propagation_graphs'] = [
            prepare_graph(grid[0], labels, spots, 7)
            for grid, labels, spots in zip(batch['superpixels'], batch['spot_labels'], batch['spots'])
        ]
        observed = []
        hook = module.model.tissue_state_decoder.register_forward_hook(
            lambda model, inputs, output: observed.append(set(output))
        )
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = module.training_step(move_data_to_device(batch, 'cuda'))
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            self.assertNotIn('gene_expression', observed[0])
            for parameter in module.model.tissue_state_decoder.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
        finally:
            hook.remove()


class PreparedGraphTest(unittest.TestCase):
    def test_prepared_graphs_skip_cpu_graph_work_and_preserve_probabilities(self):
        batch = build_batch()
        module = DifferentiablePropagation(label_scale=7, gate_range=0)
        features = torch.randn(2, 5, 16, 16)
        args = (batch['superpixels'], features, batch['image'], batch['spot_labels'], batch['spots'])
        reference = module(*args)
        graphs = [prepare_graph(grid[0], labels, spots, 7)
                  for grid, labels, spots in zip(batch['superpixels'], batch['spot_labels'], batch['spots'])]
        with patch('segmodel.modules.propagation.prepare_graph', side_effect=AssertionError('already prepared')):
            actual = module(*args, graphs)
        self.assertEqual(actual[0], reference[0])
        torch.testing.assert_close(actual[1], reference[1], rtol=0, atol=0)
        torch.testing.assert_close(actual[2], reference[2])
        self.assertEqual(actual[3], reference[3])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_deferred_pcg_matches_cpu_including_convergence_and_breakdown(self):
        edges = torch.tensor([[0, 1], [1, 2], [2, 3]])
        seeded = torch.tensor([True, False, False, True])
        for weights in (torch.ones(3), torch.zeros(3), torch.tensor([1., .4, 2.])):
            for tolerance in (1e-8, 0.):
                # 双精度核对停机步数，排除 FP32 CPU/CUDA 融合算子的舍入差异。
                args = (weights.double(), edges, torch.tensor([0., 0., 0., 1.], dtype=torch.float64), seeded, ~seeded)
                expected, count = pcg_solve(*args, steps=100, tolerance=tolerance)
                actual, actual_count = pcg_solve(*(item.cuda() for item in args), steps=100, tolerance=tolerance)
                torch.testing.assert_close(actual.cpu(), expected)
                self.assertEqual(actual_count, count)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_pcg_preserves_conductance_gradients_after_early_convergence(self):
        edges = torch.tensor([[0, 1], [1, 2], [2, 3]])
        seeded = torch.tensor([True, False, False, True])
        weights = torch.tensor([.7, 1.2, .3], dtype=torch.float64, requires_grad=True)
        other_weights = weights.detach().cuda().requires_grad_()
        values = torch.tensor([0., 0., 0., 1.], dtype=torch.float64)
        expected, count = pcg_solve(weights, edges, values, seeded, ~seeded, 100, 1e-8)
        actual, actual_count = pcg_solve(other_weights, edges.cuda(), values.cuda(), seeded.cuda(), (~seeded).cuda(), 100, 1e-8)
        expected.square().sum().backward()
        actual.square().sum().backward()
        self.assertLess(count, 8)
        self.assertEqual(count, actual_count)
        torch.testing.assert_close(actual.cpu(), expected)
        torch.testing.assert_close(other_weights.grad.cpu(), weights.grad)


class PersistentWorkerTest(unittest.TestCase):
    def test_geometry_matches_torch_for_all_rotations_and_flips(self):
        for shape in ((3, 8, 9), (3, 1, 1), (3, 1, 9)):
            self.check_geometry(torch.arange(torch.tensor(shape).prod()).reshape(shape))

    def check_geometry(self, value):
        for turns in range(4):
            for flip_x in (False, True):
                for flip_y in (False, True):
                    expected = torch.rot90(value, turns, (-2, -1))
                    if flip_x:
                        expected = expected.flip(-1)
                    if flip_y:
                        expected = expected.flip(-2)
                    actual = _transform_geometry(value, turns, flip_x, flip_y)
                    torch.testing.assert_close(actual, expected)
                    self.assertTrue(actual.is_contiguous())

    def test_persistent_workers_read_epoch_and_shared_topology_with_fork_and_spawn(self):
        for context in ('fork', 'spawn'):
            with self.subTest(context=context), tempfile.TemporaryDirectory() as directory:
                cfg = write_fixture(directory)
                cfg.experiment.dataset.num_workers = 2
                cfg.experiment.dataset.samples_per_epoch = 4
                cfg.experiment.augment.enabled = True
                for name in ('stain', 'brightness', 'contrast', 'low_resolution'):
                    cfg.experiment.augment[name].probability = 0
                data = FoldDataModule(['A1', 'A2'], ['C1'], 0, cfg)
                data.setup('fit')
                dataset = data.train_dataset
                dataset.slide_names = ['A1']
                loader = DataLoader(dataset, batch_size=2, sampler=EpochSampler(dataset), num_workers=2,
                                    persistent_workers=True, collate_fn=collate_samples, multiprocessing_context=context)
                try:
                    first = list(loader)
                    pids = [worker.pid for worker in loader._iterator._workers]
                    dataset.slides['A1'].superpixels[...] += 1000
                    loader.sampler.set_epoch(1)
                    second = list(loader)
                    self.assertEqual(pids, [worker.pid for worker in loader._iterator._workers])
                    for batch_index, batch in enumerate(second):
                        expected = collate_samples([dataset[(1, 2 * batch_index + index)] for index in range(2)])
                        for key in ('image', 'superpixels'):
                            torch.testing.assert_close(batch[key], expected[key])
                        for actual, wanted in zip(batch['spots'], expected['spots']):
                            torch.testing.assert_close(actual, wanted)
                        for graph in batch['propagation_graphs']:
                            self.assertGreater(int(graph['nodes'].min()), 1000)
                    # 去掉拓扑偏移后仍不同，说明 worker 使用了新 epoch 的增强随机数。
                    self.assertFalse(torch.equal(first[0]['superpixels'], second[0]['superpixels'] - 1000))
                finally:
                    if loader._iterator is not None:
                        loader._iterator._shutdown_workers()
                    data.teardown('fit')


if __name__ == '__main__':
    unittest.main()

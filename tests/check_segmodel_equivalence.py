"""在独立进程中对照重构前后代码, 所有张量要求零容差相等。

从仓库根运行:
  python tests/check_segmodel_equivalence.py /path/to/before
其中 before/segmodel 是重构前的源码快照。测试使用小型合成数据, 不读取真实训练集。
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def capture(source_root, destination):
    # 仅在回归测试中切换源码根目录, 让两版各自在全新的进程中导入。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(source_root))
    import random
    import anndata as ad
    import numpy as np
    import pandas as pd
    import pytorch_lightning as pl
    import torch
    from hydra import compose, initialize_config_dir

    from segmodel.dataset.datamodule import FoldDataModule
    from segmodel.dataset.segmentation import SegmentationDataset, collate_samples
    from segmodel.dataset.slide import TrainingSlide
    from segmodel.inference.predictor import predict_slide
    from segmodel.models.distiller import Distiller
    from segmodel.models.gene_vae import GeneVAE
    from segmodel.models.segmentation import SegmentationModel
    from segmodel.training.callbacks import DatasetEpochCallback
    from segmodel.training.gene_module import GeneVAEModule
    from segmodel.training.segmentation_module import SegmentationModule

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with initialize_config_dir(version_base='1.3', config_dir=str(source_root / 'segmodel/configs')):
        cfg = compose(config_name='config')
    cfg.experiment.dataset.crop_size = 224
    cfg.experiment.dataset.samples_per_epoch = 6
    cfg.experiment.dataset.num_workers = 0
    cfg.experiment.training.image.batch_size = 3
    cfg.experiment.training.image.epochs = 3
    cfg.experiment.training.image.warmup_epochs = 1
    cfg.experiment.training.vae.pretrain_batch_size = 4
    cfg.experiment.training.image.loss_weights.relation = 0.25
    for name in ('stain', 'brightness', 'contrast', 'low_resolution'):
        cfg.experiment.augment[name].probability = 1.0
    for name in ('mixup', 'cutmix'):
        cfg.experiment.augment[name].probability = 1.0
    # 显式制造正负区域和完整 PFM, 确保两个混合增强都走实际计算路径。
    cfg.experiment.augment.pfm_mask.probability = 0.0

    def clone(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: clone(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [clone(item) for item in value]
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.copy())
        return value

    records = {}
    rng = np.random.default_rng(711)
    slides = {}
    for name in ('a', 'b'):
        probability = np.indices((64, 64)).sum(axis=0) % 2
        slides[name] = TrainingSlide(
            name=name,
            image=rng.integers(0, 256, (448, 448, 3), dtype=np.uint8),
            patch_maps=tuple(rng.normal(size=(32, 32, dim)).astype(np.float16) for dim in (6, 8)),
            cls_maps=tuple(rng.normal(size=(4, 4, dim)).astype(np.float16) for dim in (6, 8)),
            cls_scale=112,
            label_scale=7,
            spot_radius_px=7.0,
            pseudo_mask_uint8=probability.astype(np.uint8),
            cancer_probability=probability.astype(np.float32),
            superpixels=(np.indices((64, 64))[0] // 4 * 16 + np.indices((64, 64))[1] // 4 + 1).astype(np.int32),
            tissue_mask_bl=np.ones((64, 64), dtype=bool),
        )
    coordinates = [(x, y) for y in range(28, 420, 56) for x in range(28, 420, 56)]
    obs = pd.DataFrame([
        {'name': name, 'pixel_x': x, 'pixel_y': y, 'spot_label': i % 2}
        for name in slides for i, (x, y) in enumerate(coordinates)
    ])
    obs.index = obs.index.astype(str)
    adata = ad.AnnData(rng.normal(size=(len(obs), 9)).astype(np.float32), obs=obs)
    dataset = SegmentationDataset(slides, adata, list(slides), cfg, seed=37)
    for enabled, masking in ((False, 0.0), (True, 0.0), (True, 1.0)):
        cfg.experiment.augment.enabled = enabled
        cfg.experiment.augment.pfm_mask.probability = masking
        for epoch in (0, 1):
            dataset.epoch = epoch
            records[f'crops/{enabled}/{masking}/{epoch}'] = clone([dataset[i] for i in range(3)])
    cfg.experiment.augment.enabled = True
    cfg.experiment.augment.pfm_mask.probability = 0.0
    dataset.epoch = 0
    batch = collate_samples([dataset[i] for i in range(3)])

    model_kwargs = dict(
        pfm_patch_dims=(6, 8), pfm_cls_dims=(6, 8), encoder_chans=(8, 12, 16),
        decoder_chans=(12, 8), global_dim=8, spatial_dim=4, spatial_pool=2,
        propagation_channels=4, drop_path=(0.1, 0.1, 0.1),
    )
    # 当前 PyTorch 的 CPU antialias resize 不支持 BF16, 完整分割训练使用 FP32。
    # BF16 的 VAE/蒸馏路径另行验证, 不为测试改变原模型算子。
    for precision in ('32-true',):
        pl.seed_everything(37, workers=True)
        teacher = GeneVAE(9, latent_dim=4, hidden_dims=(8,), dropout=0.1)
        gene_module = GeneVAEModule(teacher, cfg.experiment.training.vae)
        records[f'{precision}/vae_initial'] = clone(gene_module.state_dict())
        data_module = FoldDataModule(['a', 'b'], ['a'], 37, cfg)
        data_module.train_gene_adata = adata
        data_module.val_gene_adata = adata[:8].copy()
        with tempfile.TemporaryDirectory(prefix='segmodel-vae-') as run_dir:
            trainer = pl.Trainer(
                accelerator='cpu', devices=1, precision=precision, max_epochs=2,
                logger=False, enable_checkpointing=False, enable_progress_bar=False,
                enable_model_summary=False, default_root_dir=run_dir,
                limit_train_batches=2, limit_val_batches=1,
            )
            trainer.fit(gene_module, data_module.vae_dataloader(), data_module.vae_val_dataloader())
        records[f'{precision}/vae_trained'] = clone(gene_module.state_dict())
        records[f'{precision}/vae_optimizer'] = clone(trainer.optimizers[0].state_dict())
        records[f'{precision}/vae_rng'] = torch.get_rng_state().clone()

        pl.seed_everything(37, workers=True)
        model = SegmentationModel(**model_kwargs)
        distiller = Distiller(model, teacher, spot_radius_px=7.0, label_scale=7)
        module = SegmentationModule(model, distiller, cfg.experiment.training.image,
                                    cfg.experiment.augment, label_scale=7)
        records[f'{precision}/initial'] = clone(module.state_dict())
        records[f'{precision}/parameters'] = [name for name, _ in module.named_parameters()]

        # Trainer 执行实际 backward、梯度裁剪、AdamW、scheduler 和 EMA hooks。
        class Recorder(pl.Callback):
            def on_before_backward(self, trainer, pl_module, loss):
                key = f'{precision}/step/{trainer.global_step}'
                records[key + '/loss'] = clone(loss)

            def on_after_backward(self, trainer, pl_module):
                key = f'{precision}/step/{trainer.global_step}'
                records[key + '/gradients'] = {
                    name: clone(parameter.grad) for name, parameter in pl_module.named_parameters()
                }

            def on_train_epoch_end(self, trainer, pl_module):
                key = f'{precision}/epoch/{trainer.current_epoch}'
                records[key + '/state'] = clone(pl_module.state_dict())
                records[key + '/optimizer'] = clone(trainer.optimizers[0].state_dict())
                records[key + '/scheduler'] = clone(trainer.lr_scheduler_configs[0].scheduler.state_dict())
                records[key + '/rng'] = torch.get_rng_state().clone()
                records[key + '/metrics'] = clone(trainer.callback_metrics)

        data_module.train_slides = slides
        data_module.train_dataset = dataset
        with tempfile.TemporaryDirectory(prefix='segmodel-image-') as run_dir:
            trainer = pl.Trainer(
                accelerator='cpu', devices=1, precision=precision, max_epochs=3,
                logger=False, enable_checkpointing=False, enable_progress_bar=False,
                enable_model_summary=False, default_root_dir=run_dir,
                callbacks=[DatasetEpochCallback(), Recorder()], limit_val_batches=0,
                num_sanity_val_steps=0, gradient_clip_val=1.0, gradient_clip_algorithm='norm',
            )
            trainer.fit(module, datamodule=data_module)
        assert trainer.callback_metrics['train/mixup_applied'] > 0
        assert trainer.callback_metrics['train/cutmix_applied'] > 0
        records[f'{precision}/final'] = clone(module.state_dict())
        records[f'{precision}/final_metrics'] = clone(trainer.callback_metrics)
        model.eval()
        for mask in (torch.ones(3, 2, dtype=torch.bool), torch.tensor([[True, False]] * 3)):
            with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16, enabled=precision == 'bf16-mixed'):
                records[f'{precision}/forward/{mask[0].tolist()}'] = clone(model(
                    batch['image'], batch['patch'], batch['cls'], mask,
                ))
        for return_features in (True, False):
            model.train()
            records[f'{precision}/prediction/{return_features}'] = clone(predict_slide(
                model, slides['a'], crop_size=224, stride=112, alignment=14,
                device='cpu', batch_size=2, return_features=return_features,
            ))
            assert model.training
        records[f'{precision}/rng'] = torch.get_rng_state().clone()
        records[f'{precision}/python_rng'] = random.getstate()
    pl.seed_everything(91, workers=True)
    teacher = GeneVAE(9, latent_dim=4, hidden_dims=(8,), dropout=0.1)
    expression = torch.from_numpy(adata.X[:4].copy())
    with torch.autocast('cpu', dtype=torch.bfloat16):
        records['bf16/vae_forward'] = clone(teacher(expression))
    distiller = Distiller(model, teacher, spot_radius_px=7.0, label_scale=7)
    features = torch.randn(3, model.feature_channels, 32, 32, requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        losses, diagnostics = distiller.losses({'features': features}, batch, relation_enabled=True)
    (losses['molecular'] + losses['relation']).backward()
    records['bf16/distillation'] = clone((losses, diagnostics, features.grad))
    records['bf16/projector_gradients'] = clone([p.grad for p in distiller.image_parameters])
    torch.save(records, destination)


def compare(before, after):
    import torch
    count = 0

    def check(left, right, path):
        nonlocal count
        if isinstance(left, torch.Tensor):
            assert left.dtype == right.dtype, path
            torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True, msg=path)
            count += 1
        elif isinstance(left, dict):
            assert list(left) == list(right), path
            for key in left:
                check(left[key], right[key], f'{path}/{key}')
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right), path
            for index, (a, b) in enumerate(zip(left, right)):
                check(a, b, f'{path}/{index}')
        else:
            assert left == right, f'{path}: {left!r} != {right!r}'

    check(torch.load(before, weights_only=False), torch.load(after, weights_only=False), 'records')
    print(f'PASS: {count} tensors exactly equal (rtol=0, atol=0); keys, order and scalar values equal.')


if __name__ == '__main__':
    if sys.argv[1] == '--capture':
        capture(Path(sys.argv[2]).resolve(), Path(sys.argv[3]))
    elif sys.argv[1] == '--compare':
        compare(sys.argv[2], sys.argv[3])
    else:
        baseline = Path(sys.argv[1]).resolve()
        current = Path(__file__).resolve().parents[1]
        environment = dict(os.environ, NO_ALBUMENTATIONS_UPDATE='1', MPLCONFIGDIR='/tmp/segmodel-mpl')
        with tempfile.TemporaryDirectory(prefix='segmodel-equivalence-') as directory:
            outputs = []
            for label, source in [('before', baseline), ('after', current)]:
                output = Path(directory) / f'{label}.pt'
                subprocess.run([sys.executable, __file__, '--capture', str(source), str(output)],
                               env=environment, check=True)
                outputs.append(output)
            compare(*outputs)

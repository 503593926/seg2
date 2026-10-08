"""Profile the current segmentation forward with cached PFM inputs (CPU only).

Run from the project root: python -m tools.profile_flops
Counts Conv2d, ConvTranspose2d and Linear multiply-adds, with 1 MAC = 2 FLOPs.
Normalization, activations, interpolation and other elementwise operations are
excluded. Independent module hooks cross-check PyTorch's operator counter.
No checkpoint, training state or existing model source is modified.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
import yaml

from segmodel.models.segmentation import SegmentationModel
from segmodel.models.gene_vae import GeneVAE
from segmodel.modules.blocks import ProjectionHead


def measure(model, args):
    macs = defaultdict(int)
    calls = defaultdict(int)
    handles = []

    def hook(name, layer, inputs, output):
        if isinstance(layer, nn.ConvTranspose2d):
            count = inputs[0].numel() * (layer.out_channels // layer.groups)
            count *= math.prod(layer.kernel_size)
        elif isinstance(layer, nn.Conv2d):
            count = output.numel() * (layer.in_channels // layer.groups)
            count *= math.prod(layer.kernel_size)
        else:
            count = output.numel() * layer.in_features
        macs[name] += int(count)
        calls[name] += 1

    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            handles.append(layer.register_forward_hook(
                lambda layer, inputs, output, name=name: hook(name, layer, inputs, output)
            ))
    try:
        with torch.inference_mode(), FlopCounterMode(display=False) as counter:
            output = model(*args)
    finally:
        for handle in handles:
            handle.remove()
    flops = int(counter.get_total_flops())
    assert flops == 2 * sum(macs.values()), (flops, 2 * sum(macs.values()))
    groups = defaultdict(int)
    for name, count in macs.items():
        groups[name.split('.')[0]] += count
    return {
        'flops': flops,
        'gflops': flops / 1e9,
        'macs': sum(macs.values()),
        'gmacs': sum(macs.values()) / 1e9,
        'parameters': sum(p.numel() for p in model.parameters()),
        'group_gflops': {name: 2 * count / 1e9 for name, count in groups.items()},
        'operator_flops': {str(op): int(n) for op, n in counter.flop_counts['Global'].items()},
        'layers': {name: {'macs': count, 'calls': calls[name]} for name, count in macs.items()},
    }, output


class TeacherEmbedding(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        self.encoder = teacher.encoder

    def forward(self, expression):
        mean, _ = self.encoder(expression)
        return mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', type=int, default=[224, 448, 896])
    parser.add_argument('--output', type=Path, default=Path('outputs/flops/segmodel_her2st.json'))
    opt = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    dataset = yaml.safe_load(Path('segmodel/configs/dataset/her2st.yaml').read_text(encoding='utf-8-sig'))
    model_cfg = yaml.safe_load(Path('segmodel/configs/model/base.yaml').read_text(encoding='utf-8-sig'))
    resolution = yaml.safe_load(Path('config/resolution.yaml').read_text())
    root = Path(dataset['data_root'])
    feature_info = []
    for directory in dataset['image_feature_maps']:
        folder = root / directory
        metadata = json.loads((folder / 'metadata.json').read_text())
        feature_info.append({
            'path': directory,
            'patch_dim': int(np.load(folder / 'C1/patch_map.npy', mmap_mode='r').shape[-1]),
            'cls_dim': int(np.load(folder / 'C1/cls_map.npy', mmap_mode='r').shape[-1]),
            'patch_scale': round(metadata['patch_mpp'] / resolution['target_mpp']),
            'cls_scale': round(metadata['cls_mpp'] / resolution['target_mpp']),
        })
    model = SegmentationModel(
        pfm_patch_dims=[f['patch_dim'] for f in feature_info],
        pfm_cls_dims=[f['cls_dim'] for f in feature_info],
        **model_cfg['image'],
    ).eval()
    report = {
        'torch_version': torch.__version__,
        'scope': 'SegmentationModel.forward; cached PFM features; batch size 1',
        'convention': '1 MAC = 2 FLOPs; Conv2d + ConvTranspose2d + Linear',
        'excluded': ['offline PFM feature extraction', 'normalization', 'activation',
                     'interpolation', 'pooling', 'elementwise operations', 'backward',
                     'optimizer updates', 'SNIC/random walker'],
        'model_config': model_cfg['image'],
        'features': feature_info,
        'default_crop_size': dataset['crop_size'],
        'measurements': [],
    }
    for size in opt.sizes:
        assert size % 28 == 0 and all(size % f['cls_scale'] == 0 for f in feature_info)
        image = torch.zeros(1, 3, size, size, dtype=torch.uint8)
        patch = [torch.randn(1, f['patch_dim'], size // f['patch_scale'], size // f['patch_scale']) for f in feature_info]
        cls = [torch.randn(1, f['cls_dim'], size // f['cls_scale'], size // f['cls_scale']) for f in feature_info]
        result, output = measure(model, (image, patch, cls))
        result['input_shape'] = list(image.shape)
        result['logits_shape'] = list(output['logits'].shape)
        report['measurements'].append(result)
        print(json.dumps({k: v for k, v in result.items() if k != 'layers'}, ensure_ascii=False))

    import h5py
    gene_path = root / 'gene_expression/task1_loocv/test_C1.h5ad'
    with h5py.File(gene_path, 'r') as handle:
        n_genes = handle['var'][handle['var'].attrs['_index']].shape[0]
    teacher = GeneVAE(n_genes, **model_cfg['gene']).eval().requires_grad_(False)
    teacher_result, _ = measure(TeacherEmbedding(teacher), (torch.randn(1, n_genes),))
    projector_result, _ = measure(ProjectionHead(model.feature_channels, teacher.latent_dim).eval(),
                                  (torch.randn(1, model.feature_channels),))
    report['training_only_per_spot'] = {
        'gene_count_for_test_C1_fold': n_genes,
        'frozen_teacher_encoder': teacher_result,
        'image_projector': projector_result,
        'note': 'Forward only; actual additional cost depends on spot count. Pooling/losses/backward excluded.',
    }
    opt.output.parent.mkdir(parents=True, exist_ok=True)
    opt.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('Saved', opt.output)


if __name__ == '__main__':
    main()

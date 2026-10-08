"""Profile the existing HER2ST nnU-Net plans using the same MAC/FLOP convention.

Run with /home/yfm/miniconda3/envs/seg_bls/bin/python -m tools.profile_nnunet_flops
The original network is reconstructed without changing weights or training state.
Large inputs are traced on meta tensors; a real CPU forward at 512x512 checks the
meta trace. Convolution/linear counts are independently checked by module hooks.
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch

from tools.profile_flops import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path('baselines/nnunet_v2/data/her2st/nnUNet_results')
    default = root / 'seed_0/Dataset000_her2st/nnUNetTrainer_100epochs__nnUNetPlans__2d/plans.json'
    parser.add_argument('--plans', type=Path, default=default)
    parser.add_argument('--output', type=Path, default=Path('outputs/flops/nnunet_v2_her2st.json'))
    opt = parser.parse_args()
    data_root = Path('baselines/nnunet_v2/data/her2st').resolve()
    for suffix in ('raw', 'preprocessed', 'results'):
        os.environ.setdefault('nnUNet_' + suffix, str(data_root / ('nnUNet_' + suffix)))

    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    plans = json.loads(opt.plans.read_text())
    dataset = json.loads(opt.plans.with_name('dataset.json').read_text())
    configuration = plans['configurations']['2d']
    architecture = configuration['architecture']
    manager = PlansManager(plans)
    output_channels = manager.get_label_manager(dataset).num_segmentation_heads
    input_channels = len(dataset['channel_names'])
    torch.set_num_threads(2)
    torch.manual_seed(0)

    def build(device):
        with torch.device(device):
            model = get_network_from_plans(
                architecture['network_class_name'], architecture['arch_kwargs'],
                architecture['_kw_requires_import'], input_channels, output_channels,
                allow_init=False, deep_supervision=False,
            ).eval()
        return model

    model = build('meta')
    cpu_model = build('cpu')
    meta_check, _ = measure(model, (torch.randn(1, input_channels, 512, 512, device='meta'),))
    cpu_check, _ = measure(cpu_model, (torch.randn(1, input_channels, 512, 512),))
    assert meta_check['flops'] == cpu_check['flops']
    assert meta_check['parameters'] == cpu_check['parameters']
    del cpu_model

    strides = architecture['arch_kwargs']['strides']
    divisor = [math.prod(stride[axis] for stride in strides) for axis in (0, 1)]
    requested = [896, 896]
    padded = [math.ceil(size / div) * div for size, div in zip(requested, divisor)]
    report = {
        'plans_path': str(opt.plans.resolve()),
        'torch_version': torch.__version__,
        'configuration': '2d',
        'architecture': architecture,
        'input_channels': input_channels,
        'output_channels': output_channels,
        'batch_size_for_measurement': 1,
        'deep_supervision': False,
        'convention': '1 MAC = 2 FLOPs; Conv2d + ConvTranspose2d + Linear',
        'excluded': ['normalization', 'activation', 'elementwise operations',
                     'softmax', 'tiling overlap', 'test-time mirroring', 'ensemble',
                     'backward', 'optimizer updates'],
        'required_spatial_divisibility': divisor,
        'verification': {'real_cpu_shape': [1, input_channels, 512, 512],
                         'real_cpu_flops': cpu_check['flops'],
                         'meta_flops': meta_check['flops'],
                         'module_hooks_match_operator_counter': True},
        'measurements': [],
    }
    for label, shape in [('native_patch', configuration['patch_size']),
                         ('896_square_padded_to_valid_shape', padded)]:
        result, output = measure(model, (torch.empty(1, input_channels, *shape, device='meta'),))
        result.update({'label': label, 'input_shape': [1, input_channels, *shape],
                       'logits_shape': list(output.shape)})
        if label != 'native_patch':
            result['requested_spatial_shape'] = requested
            result['note'] = 'Valid network input after divisibility padding; this is not the native sliding-window patch.'
        report['measurements'].append(result)
        print(json.dumps({k: v for k, v in result.items() if k != 'layers'}, ensure_ascii=False))
    report['seed_plans_architecture_consistency'] = {}
    for path in sorted(root.glob('seed_*/Dataset000_her2st/*/plans.json')):
        config = json.loads(path.read_text())['configurations']['2d']
        report['seed_plans_architecture_consistency'][path.parts[5]] = (
            config['architecture'] == architecture and config['patch_size'] == configuration['patch_size']
        )
    opt.output.parent.mkdir(parents=True, exist_ok=True)
    opt.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('Saved', opt.output)


if __name__ == '__main__':
    main()

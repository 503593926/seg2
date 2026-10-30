from concurrent.futures import ThreadPoolExecutor

import torch

from seg2.infer.window import window_inputs


def _prepare_batch(slide, origins, crop_size):
    '''
    batch: {
    "image": [ Tensor(N, 3, 896, 896)  uint8 ],

    "patch": [ Tensor(N, 1536, 64, 64),   # ho1        f16
               Tensor(N, 1536, 64, 64),   # uni2       f16
               Tensor(N, 1280, 64, 64) ], # virchow2   f16

    "cls":   [ Tensor(N, 1536, 4, 4),     # ho1        f16
               Tensor(N, 1536, 4, 4),     # uni2       f16
               Tensor(N, 1280, 4, 4) ],   # virchow2   f16
    }
    '''
    inputs = [window_inputs(slide, top, left, crop_size) for top, left in origins]
    batch = {"image": [torch.stack([item["image"] for item in inputs])]}
    for key in ("patch", "cls"):
        batch[key] = [
            torch.stack([item[key][index] for item in inputs]) for index in range(len(inputs[0][key]))
        ]
    return batch


def prefetch_batches(slide, origins, crop_size, batch_size, device):
    device = torch.device(device)
    batches = [origins[start : start + batch_size] for start in range(0, len(origins), batch_size)]
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(_prepare_batch, slide, batches[0], crop_size)
        for index, batch_origins in enumerate(batches):
            inputs = pending.result()
            if index + 1 < len(batches):
                pending = executor.submit(
                    _prepare_batch, slide, batches[index + 1], crop_size
                )
            inputs = {
                key: [value.to(device) for value in values]
                for key, values in inputs.items()
            }
            yield (batch_origins, inputs)

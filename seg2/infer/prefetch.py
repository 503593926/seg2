from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import torch

from seg2.infer.memory import inference_copy_stream
from seg2.infer.window import window_inputs


def _prepare_batch(slide, origins, crop_size, device, copy_stream):
    inputs = [window_inputs(slide, top, left, crop_size) for top, left in origins]
    batch = {"image": [torch.stack([item["image"] for item in inputs])]}
    for key in ("patch", "cls"):
        batch[key] = [
            torch.stack([item[key][index] for item in inputs]) for index in range(len(inputs[0][key]))
        ]
    with torch.cuda.stream(copy_stream) if copy_stream is not None else nullcontext():
        if copy_stream is not None:
            batch = {key: [value.pin_memory() for value in values] for key, values in batch.items()}
        batch = {
            key: [value.to(device, non_blocking=copy_stream is not None) for value in values]
            for key, values in batch.items()
        }
        ready = torch.cuda.Event() if copy_stream is not None else None
        if ready is not None:
            ready.record(copy_stream)
    return (batch, ready)


def _prefetch_batches(slide, origins, crop_size, batch_size, device):
    device = torch.device(device)
    batches = [origins[start : start + batch_size] for start in range(0, len(origins), batch_size)]
    copy_stream = inference_copy_stream(device)
    current_stream = torch.cuda.current_stream(device) if copy_stream is not None else None
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(_prepare_batch, slide, batches[0], crop_size, device, copy_stream)
        try:
            for index, batch_origins in enumerate(batches):
                inputs, ready = pending.result()
                pending = None
                if index + 1 < len(batches):
                    pending = executor.submit(
                        _prepare_batch, slide, batches[index + 1], crop_size, device, copy_stream
                    )
                if current_stream is not None and ready is not None:
                    current_stream.wait_event(ready)
                    for values in inputs.values():
                        for value in values:
                            value.record_stream(current_stream)
                try:
                    yield (batch_origins, inputs)
                finally:
                    inputs = ready = None
        finally:
            if pending is not None:
                pending.cancel()

import torch

_copy_streams: dict[int, torch.cuda.Stream] = {}


def inference_copy_stream(device):
    device = torch.device(device)
    if device.type != "cuda":
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _copy_streams:
        _copy_streams[index] = torch.cuda.Stream(device=index)
    return _copy_streams[index]


def release_inference_cache(device, stage):
    device = torch.device(device)
    if device.type != "cuda":
        return
    with torch.cuda.device(device):
        torch.cuda.synchronize(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        torch.cuda.empty_cache()
        remaining = torch.cuda.memory_reserved(device)
    print(
        f"{stage}: CUDA allocated={allocated / 1024**3:.2f} GiB, reserved={reserved / 1024**3:.2f}->{remaining / 1024**3:.2f} GiB",
        flush=True,
    )

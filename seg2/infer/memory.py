import torch


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

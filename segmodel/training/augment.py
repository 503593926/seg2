from math import lcm

import torch


def mixup_batch(batch, cfg):
    # 布尔 pfm_mask 无法表示插值后的可用程度，只选择所有 PFM 均可用的样本。
    eligible = batch["pfm_mask"].all(dim=1) & batch["valid"].flatten(1).any(dim=1)
    indices = eligible.nonzero(as_tuple=True)[0]
    if indices.numel() < 2:
        return None
    indices = indices[
        torch.randperm(indices.numel(), device=indices.device)[: int(cfg.max_samples)]
    ]
    if indices.numel() < 2:
        return None
    partners = indices.roll(1)
    ratio = torch.distributions.Beta(float(cfg.alpha), float(cfg.alpha)).sample().item()

    def mix(value):
        return ratio * value[indices].float() + (1 - ratio) * value[partners].float()

    image = batch["image"]
    if image.dtype == torch.uint8:
        image = image.float().mul(2.0 / 255.0).sub(1.0)
    mixed = {
        "image": mix(image),
        "patch": tuple(mix(value) for value in batch["patch"]),
        "cls": tuple(mix(value) for value in batch["cls"]),
        "pfm_mask": batch["pfm_mask"][indices],
    }
    return mixed, ratio, indices, partners


def cutmix_batch(batch, cfg):
    image = batch["image"]
    target = batch["cancer_probability"][:, 0]
    valid = batch["valid"][:, 0]
    available = batch["pfm_mask"].all(dim=1)
    positive = (target >= float(cfg.positive_threshold)) & valid & available[:, None, None]
    negative = (target <= float(cfg.negative_threshold)) & valid & available[:, None, None]
    donors = positive.flatten(1).any(dim=1).nonzero(as_tuple=True)[0]
    receivers = negative.flatten(1).any(dim=1).nonzero(as_tuple=True)[0]
    if not donors.numel() or not receivers.numel():
        return None

    height, width = image.shape[-2:]
    grids = (batch["cancer_probability"], *batch["patch"], *batch["cls"])
    if any(height % value.shape[-2] or width % value.shape[-1] for value in grids):
        raise ValueError("CutMix requires image, label and PFM grids to divide the crop")
    unit_y = lcm(*(height // value.shape[-2] for value in grids))
    unit_x = lcm(*(width // value.shape[-1] for value in grids))
    if height % unit_y or width % unit_x:
        raise ValueError("CutMix grids cannot share aligned crop boundaries")
    cells_y, cells_x = height // unit_y, width // unit_x
    min_cells = int(cfg.min_cells)
    if cells_y < min_cells or cells_x < min_cells:
        return None

    receivers = receivers[
        torch.randperm(receivers.numel(), device=receivers.device)[: int(cfg.max_samples)]
    ]
    # 每个目标 crop 选一个不同的病灶来源；来源和目标都必须有完整 PFM。
    entries = []
    for receiver in receivers.tolist():
        choices = donors[donors != receiver]
        if not choices.numel():
            continue
        donor = int(choices[torch.randint(choices.numel(), (1,), device=choices.device)])
        donor_points = positive[donor].nonzero()
        receiver_points = negative[receiver].nonzero()
        source_y, source_x = donor_points[
            torch.randint(donor_points.shape[0], (1,), device=donor_points.device)
        ][0].tolist()
        dest_y, dest_x = receiver_points[
            torch.randint(receiver_points.shape[0], (1,), device=receiver_points.device)
        ][0].tolist()

        cells_h = int(torch.randint(min_cells, min(int(cfg.max_cells), cells_y) + 1, (1,)))
        cells_w = int(torch.randint(min_cells, min(int(cfg.max_cells), cells_x) + 1, (1,)))
        source_y = source_y * (height // target.shape[-2]) // unit_y
        source_x = source_x * (width // target.shape[-1]) // unit_x
        dest_y = dest_y * (height // target.shape[-2]) // unit_y
        dest_x = dest_x * (width // target.shape[-1]) // unit_x
        source_y = min(max(source_y - cells_h // 2, 0), cells_y - cells_h) * unit_y
        source_x = min(max(source_x - cells_w // 2, 0), cells_x - cells_w) * unit_x
        dest_y = min(max(dest_y - cells_h // 2, 0), cells_y - cells_h) * unit_y
        dest_x = min(max(dest_x - cells_w // 2, 0), cells_x - cells_w) * unit_x
        entries.append((receiver, donor, source_y, source_x, dest_y, dest_x,
                        cells_h * unit_y, cells_w * unit_x))

    if not entries:
        return None
    receiver_indices = [entry[0] for entry in entries]
    mixed = {
        key: batch[key][receiver_indices].clone()
        for key in ("image", "cancer_probability", "valid")
    }
    mixed["patch"] = tuple(value[receiver_indices].clone() for value in batch["patch"])
    mixed["cls"] = tuple(value[receiver_indices].clone() for value in batch["cls"])
    mixed["pfm_mask"] = batch["pfm_mask"][receiver_indices]

    def paste(destination, source, index, entry):
        _, donor, sy, sx, dy, dx, h, w = entry
        scale_y = height // source.shape[-2]
        scale_x = width // source.shape[-1]
        destination[index, :, dy // scale_y:(dy + h) // scale_y,
                    dx // scale_x:(dx + w) // scale_x] = source[
            donor, :, sy // scale_y:(sy + h) // scale_y,
            sx // scale_x:(sx + w) // scale_x
        ]

    for index, entry in enumerate(entries):
        for key in ("image", "cancer_probability", "valid"):
            paste(mixed[key], batch[key], index, entry)
        for key in ("patch", "cls"):
            for destination, source in zip(mixed[key], batch[key]):
                paste(destination, source, index, entry)
    return mixed

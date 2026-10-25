from collections.abc import Callable

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def get_patch_grid(tile_size: int, resize: bool, patch_size: int) -> tuple[int, int]:
    input_size = 224 if resize else tile_size
    return (input_size // patch_size, input_size // patch_size)


def create_preprocess(
    resize: bool,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
) -> transforms.Compose:
    steps: list[Callable] = []
    if resize:
        steps.append(transforms.Resize((224, 224), interpolation=interpolation))
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(steps)


class TileDataset(Dataset):
    def __init__(
        self,
        image: NDArray[np.uint8],
        transform: Callable[[Image.Image], torch.Tensor],
        tile_size: int = 112,
        stride: int | None = None,
    ) -> None:
        self.image = image
        self.transform = transform
        self.tile_size = tile_size
        self.stride = stride or tile_size
        self.tile_rows = image.shape[0] // tile_size
        self.tile_columns = image.shape[1] // tile_size
        usable_height = self.tile_rows * tile_size
        usable_width = self.tile_columns * tile_size
        self.row_starts = list(range(0, usable_height - tile_size + 1, self.stride))
        self.column_starts = list(range(0, usable_width - tile_size + 1, self.stride))
        self.window_rows = len(self.row_starts)
        self.window_columns = len(self.column_starts)

    def __len__(self) -> int:
        return self.window_rows * self.window_columns

    def __getitem__(self, index: int) -> torch.Tensor:
        row, column = divmod(index, self.window_columns)
        y0, x0 = (self.row_starts[row], self.column_starts[column])
        tile = self.image[y0 : y0 + self.tile_size, x0 : x0 + self.tile_size]
        return self.transform(Image.fromarray(tile))


class FeatureMapExtractor:
    def __init__(
        self,
        model: torch.nn.Module,
        transform: Callable[[Image.Image], torch.Tensor],
        patch_grid: tuple[int, int],
        patch_dimension: int,
        cls_dimension: int,
        batch_size: int = 8,
        device: str | torch.device | None = None,
        tile_size: int = 112,
        resize: bool = True,
        overlap_average: bool = False,
        overlap_cls: bool = False,
    ) -> None:
        self.model = model
        self.transform = transform
        self.patch_grid = patch_grid
        self.patch_dimension = patch_dimension
        self.cls_dimension = cls_dimension
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tile_size = tile_size
        self.resize = resize
        self.input_size = 224 if resize else tile_size
        self.overlap_average = overlap_average
        self.overlap_cls = overlap_cls
        self.window_stride = tile_size // 2 if overlap_average else tile_size
        self.cls_stride = self.window_stride if overlap_cls else tile_size

    def _forward_tiles(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.model
        tokens = model.forward_features(images)
        patch_features = tokens[:, model.num_prefix_tokens :]
        cls_features = model.forward_head(tokens, pre_logits=True)
        return (patch_features, cls_features)

    def extract(self, image: NDArray[np.uint8]) -> tuple[NDArray[np.float16], NDArray[np.float16]]:
        self.model.to(self.device)
        self.model.eval()
        dataset = TileDataset(image, self.transform, self.tile_size, stride=self.window_stride)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        patch_rows, patch_columns = self.patch_grid
        patch_map_shape = (
            dataset.tile_rows * patch_rows,
            dataset.tile_columns * patch_columns,
            self.patch_dimension,
        )
        feature_map = (
            np.zeros(patch_map_shape, dtype=np.float16)
            if self.overlap_average
            else np.empty(patch_map_shape, dtype=np.float16)
        )
        overlap_counts = np.zeros(feature_map.shape[:2], dtype=np.uint8) if self.overlap_average else None
        cls_map = np.empty(
            (
                dataset.window_rows if self.overlap_cls else dataset.tile_rows,
                dataset.window_columns if self.overlap_cls else dataset.tile_columns,
                self.cls_dimension,
            ),
            dtype=np.float16,
        )
        tile_index = 0
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"
            ),
        ):
            for images in dataloader:
                images = images.to(self.device, non_blocking=True)
                patch_features, cls_features = self._forward_tiles(images)
                patch_features = patch_features.reshape(
                    images.shape[0], patch_rows, patch_columns, self.patch_dimension
                )
                patch_batch = patch_features.to(device="cpu", dtype=torch.float16).numpy()
                cls_batch = cls_features.to(device="cpu", dtype=torch.float16).numpy()
                for batch_index in range(images.shape[0]):
                    window_row, window_column = divmod(tile_index, dataset.window_columns)
                    source_y = dataset.row_starts[window_row]
                    source_x = dataset.column_starts[window_column]
                    y0 = source_y * patch_rows // self.tile_size
                    x0 = source_x * patch_columns // self.tile_size
                    y1, x1 = (y0 + patch_rows, x0 + patch_columns)
                    if overlap_counts is None:
                        feature_map[y0:y1, x0:x1] = patch_batch[batch_index]
                    else:
                        feature_map[y0:y1, x0:x1] += patch_batch[batch_index]
                        overlap_counts[y0:y1, x0:x1] += 1
                    if self.overlap_cls:
                        cls_map[window_row, window_column] = cls_batch[batch_index]
                    elif source_y % self.tile_size == 0 and source_x % self.tile_size == 0:
                        tile_row = source_y // self.tile_size
                        tile_column = source_x // self.tile_size
                        cls_map[tile_row, tile_column] = cls_batch[batch_index]
                    tile_index += 1
        if overlap_counts is not None and feature_map.size:
            np.divide(feature_map, overlap_counts[..., None], out=feature_map, casting="unsafe")
        return (feature_map, cls_map)

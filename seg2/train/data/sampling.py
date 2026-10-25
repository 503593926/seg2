from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler


class EpochSampler(Sampler):
    def __init__(self, dataset, batch_size=1, slides_per_batch=4):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.slides_per_batch = slides_per_batch

    def set_epoch(self, epoch):
        self.dataset.epoch = int(epoch)

    def __iter__(self) -> Iterator[tuple]:
        epoch = self.dataset.epoch
        for start in range(0, len(self.dataset), self.batch_size):
            count = min(self.batch_size, len(self.dataset) - start)
            rng = np.random.default_rng(np.random.SeedSequence([self.dataset.seed, epoch, start, 1]))
            names = self._sample_slides(rng, min(self.slides_per_batch, count))
            crops, extra = divmod(count, len(names))
            index = start
            for position, name in enumerate(names):
                for top, left in self.dataset._sample_slide_origins(
                    name, crops + (position < extra), rng
                ):
                    yield (epoch, index, name, top, left)
                    index += 1

    def _sample_slides(self, rng, count):
        dataset = self.dataset
        fraction = dataset.cfg.dataset.labeled_sampling_fraction
        if fraction is None:
            return rng.choice(dataset.slide_names, size=count, replace=False).tolist()
        expected = count * float(fraction)
        labeled_count = int(np.floor(expected)) + int(rng.random() < expected % 1)
        labeled_count = min(
            max(labeled_count, count - len(dataset.expression_only_names)),
            len(dataset.labeled_names),
        )
        names = (
            rng.choice(dataset.labeled_names, size=labeled_count, replace=False).tolist()
            + rng.choice(dataset.expression_only_names, size=count - labeled_count, replace=False).tolist()
        )
        rng.shuffle(names)
        return names

    def __len__(self):
        return len(self.dataset)

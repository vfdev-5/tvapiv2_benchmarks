from functools import partial

import numpy as np
import PIL

import torch
import torch.utils.benchmark as benchmark
from torchvision.prototype import features
from torchvision.prototype.transforms import functional as F_v2

from torchvision.transforms import functional as F_stable
from torchvision.transforms.functional import InterpolationMode


def get_pil_mask(size):
    target_data = np.zeros(size, dtype="int32")
    target_data[110:140, 120:160] = 1
    target_data[10:40, 120:160] = 2
    target_data[110:140, 20:60] = 3
    target_data[size[0] // 2 : size[0] // 2 + 50, size[1] // 2 : size[1] // 2 + 60] = 4
    target = PIL.Image.fromarray(target_data).convert("L")
    return target


def main():
    results = []
    min_run_time = 2

    for size in [256, 520, 720]:
        pil_mask = get_pil_mask((500, 600))
        mask = features.Mask(F_v2.pil_to_tensor(pil_mask).squeeze(0))

        transform_stable = partial(F_stable.resize, size=size, interpolation=InterpolationMode.NEAREST)
        transform_v2 = partial(F_v2.resize, size=[size], interpolation=InterpolationMode.NEAREST)

        # PIL resize
        results.append(
            benchmark.Timer(
                stmt=f"transform(data)",
                globals={
                    "data": pil_mask,
                    "transform": transform_stable,
                },
                num_threads=torch.get_num_threads(),
                label=f"Resize measurements: 500x600 -> {size}",
                sub_label="PIL mask data",
                description="stable",
            ).blocked_autorange(min_run_time=min_run_time)
        )
        # Mask resize
        results.append(
            benchmark.Timer(
                stmt=f"transform(data)",
                globals={
                    "data": mask,
                    "transform": transform_v2,
                },
                num_threads=torch.get_num_threads(),
                label=f"Resize measurements: 500x600 -> {size}",
                sub_label="Feature Mask data",
                description="v2",
            ).blocked_autorange(min_run_time=min_run_time)
        )

    compare = benchmark.Compare(results)
    compare.print()


if __name__ == "__main__":
    main()

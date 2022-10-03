import numpy as np
import PIL.Image

import torch
import torch.utils.benchmark as benchmark


def main():
    results = []
    min_run_time = 2

    for size in [256, 520, 720]:

        torch.manual_seed(1)
        tensor = torch.randint(0, 256, size=(3, size, size), dtype=torch.uint8)
        np_array = tensor.permute(1, 2, 0).contiguous().numpy()
        pil_img = PIL.Image.fromarray(np_array)

        torch.testing.assert_close(
            tensor.flip(-1),
            torch.from_numpy(np.asarray(pil_img.transpose(0))).clone().permute(2, 0, 1).contiguous()
        )

        # PIL hflip
        results.append(
            benchmark.Timer(
                stmt=f"data.transpose(0)",
                globals={
                    "data": pil_img,
                },
                num_threads=torch.get_num_threads(),
                label=f"HFlip measurements: {size}",
                description="Pillow",
            ).blocked_autorange(min_run_time=min_run_time)
        )
        # Tensor hflip
        results.append(
            benchmark.Timer(
                stmt=f"data.flip(-1)",
                globals={
                    "data": tensor,
                },
                num_threads=torch.get_num_threads(),
                label=f"HFlip measurements: {size}",
                description="torch tensor",
            ).blocked_autorange(min_run_time=min_run_time)
        )

    compare = benchmark.Compare(results)
    compare.print()


if __name__ == "__main__":
    main()

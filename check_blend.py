import torch
import torch.utils.benchmark as benchmark

from torch import Tensor


def _blend_original(img1: Tensor, img2: Tensor, ratio: float) -> Tensor:
    ratio = float(ratio)
    bound = 1.0 if img1.is_floating_point() else 255.0
    return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound).to(img1.dtype)


def _blend_new(img1: Tensor, img2: Tensor, ratio: float) -> Tensor:
    if ratio == 1.0:
        return img1
    ratio = float(ratio)
    bound = 1.0 if img1.is_floating_point() else 255.0

    if img2.is_floating_point():
        # Since img2 is float, we can do in-place ops on it. It's a throw-away tensor.
        # Our strategy is to convert img1 to float and copy it to avoid in-place modifications,
        # update img2 in-place and add it on the result with an in-place op.
        result = img1 * ratio
        # img2.mul_(1.0 - ratio)
        result.add_((1.0 - ratio) * img2)
    else:
        # Since img2 is not float, we can't do in-place ops on it.
        # To minimize copies/adds/muls we first convert img1 to float by multiplying it with ratio/(1-ratio).
        # This permits us to add img2 in-place to it, without further copies.
        # To ensure we have the correct result at the end, we multiply in-place with (1-ratio).
        result = img1 * (ratio / (1.0 - ratio))
        result.add_(img2).mul_(1.0 - ratio)

    return result.clamp_(0, bound).to(img1.dtype)


def bench_torch(fn, x, y, r, runtime=20):
    for _ in range(10):
        fn(x, y, r)

    results = benchmark.Timer(
                stmt=f"fn(x, y, {r})",
                globals={
                    "x": x,
                    "y": y,
                    "fn": fn,
                },
                num_threads=torch.get_num_threads(),
                label="Blend measurements",
                sub_label=f"{x.dtype}, {x.device}",
                description=fn.__name__,
            ).blocked_autorange(min_run_time=runtime)
    return results


shape = (3, 400, 500)

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")


torch.set_num_threads(1)

all_results = []
for fn in [_blend_new, _blend_original]:
    for device in ["cpu", "cuda"]:
        for dtype in [torch.uint8, torch.float32]:

            torch.manual_seed(12)
            x = torch.randint(0, 256, size=shape, dtype=dtype, device=device)
            y = torch.randint(0, 256, size=shape, dtype=dtype, device=device)

            if dtype == torch.float32:
                x.div_(255)
                y.div_(255)
            r = 0.23

            try:
                torch.testing.assert_close(_blend_original(x, y, r), fn(x, y, r), rtol=1e-5, atol=1e-5)
            except Exception as e:
                print(f"\nFunction does not match ref function for test case: {fn.__name__}, {dtype}, {device}")
                print(e)
            all_results.append(bench_torch(fn, x, y, r))


compare = benchmark.Compare(all_results)
compare.print()

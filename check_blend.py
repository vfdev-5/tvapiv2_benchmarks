import torch
from torch import Tensor
from torch.utils.benchmark import Timer, Compare
from itertools import product
from functools import partial


def blend_main(img1: Tensor, img2: Tensor, ratio: float) -> Tensor:
    ratio = float(ratio)
    bound = 1.0 if img1.is_floating_point() else 255.0
    return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound).to(img1.dtype)


def blend_mario(img1: Tensor, img2: Tensor, ratio: float) -> Tensor:
    ratio = float(ratio)
    bound = 1.0 if img1.is_floating_point() else 255.0
    return img1.mul(ratio).add_(img2, alpha=(1.0 - ratio)).clamp_(0, bound).to(img1.dtype)


def blend_super_mario(img1: Tensor, img2: Tensor, ratio: float) -> Tensor:
    dtype = img1.dtype
    fp = dtype.is_floating_point
    bound = 1.0 if fp else 255.0
    if not fp and img2.is_cuda:
        img2 = img2 * (1.0 - ratio)
    else:
        if not fp:
            img2 = img2.to(torch.float32)
        img2.mul_(1.0 - ratio)
    img2.add_(img1, alpha=ratio).clamp_(0, bound)
    return img2 if fp else img2.to(dtype)


def blend_datumbox(img1: Tensor, img2: Tensor, ratio: float) -> Tensor:
    if ratio == 1.0:
        return img1
    ratio = float(ratio)
    bound = 1.0 if img1.is_floating_point() else 255.0

    if img2.is_floating_point():
        # Since img2 is float, we can do in-place ops on it. It's a throw-away tensor.
        # Our strategy is to convert img1 to float and copy it to avoid in-place modifications,
        # update img2 in-place and add it on the result with an in-place op.
        result = img1 * ratio
        img2.mul_(1.0 - ratio)
        result.add_(img2)
    else:
        # Since img2 is not float, we can't do in-place ops on it.
        # To minimize copies/adds/muls we first convert img1 to float by multiplying it with ratio/(1-ratio).
        # This permits us to add img2 in-place to it, without further copies.
        # To ensure we have the correct result at the end, we multiply in-place with (1-ratio).
        result = img1 * (ratio / (1.0 - ratio))
        result.add_(img2).mul_(1.0 - ratio)

    return result.clamp_(0, bound).to(img1.dtype)


def gen_inputs():
    make_arg_int = partial(torch.randint, 0, 256, dtype=torch.uint8)
    makers = (make_arg_int, torch.randn)
    shapes = ((3, 400, 400),)
    devices = ("cpu", "cuda")
    fns = (blend_mario, blend_super_mario, blend_datumbox, blend_main)
    threads = (1, torch.get_num_threads())
    for make, shape, device, fn, threads in product(makers, shapes, devices, fns, threads):
        t1 = make(shape, device=device)
        t2 = make(shape, device=device)
        yield f"Equalize {device} {t1.dtype}", str(tuple(shape)), threads, fn, t1, t2, 0.5


def benchmark(label, sub_label, threads, f, t1, t2, *args, **kwargs):

    if f is not blend_main:
        out = f(t1, t2.clone(), *args, **kwargs)
        ref = blend_main(t1, t2.clone(), *args, **kwargs)
        try:
            torch.testing.assert_close(ref, out)
        except Exception:
            return None

    return Timer(
        "f(t1, t2, *args, **kwargs)",
        globals={
            "f": f,
            "t1": t1,
            "t2": t2.clone(),
            "args": args,
            "kwargs": kwargs,
        },
        label=label,
        description=f.__name__,
        sub_label=sub_label,
        num_threads=threads
    ).blocked_autorange(min_run_time=10)


results = []
for args in gen_inputs():
    res = benchmark(*args)
    if res is not None:
        results.append(res)

compare = Compare(results)
compare.trim_significant_figures()
compare.print()

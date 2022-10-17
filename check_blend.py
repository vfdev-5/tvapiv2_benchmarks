import torch
from torch import Tensor
from torch.utils.benchmark import Timer, Compare
from itertools import product
from functools import partial


def blend(img1: Tensor, img2: Tensor, ratio: float) -> Tensor:
    ratio = float(ratio)
    bound = 1.0 if img1.is_floating_point() else 255.0
    return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound).to(img1.dtype)


def blend_new(img1: Tensor, img2: Tensor, ratio: float) -> Tensor:
    ratio = float(ratio)
    bound = 1.0 if img1.is_floating_point() else 255.0
    return img1.mul(ratio).add_(img2, alpha=(1.0 - ratio)).clamp_(0, bound).to(img1.dtype)


def gen_inputs():
    make_arg_int = partial(torch.randint, 0, 256, dtype=torch.uint8)
    makers = (make_arg_int, torch.randn)
    shapes = ((3, 400, 400),)
    devices = ("cpu", "cuda")
    fns = (blend, blend_new)
    threads = (1, torch.get_num_threads())
    for make, shape, device, fn, threads in product(makers, shapes, devices, fns, threads):
        t1 = make(shape, device=device)
        t2 = make(shape, device=device)
        yield f"Equalize {device} {t1.dtype}", str(tuple(shape)), threads, fn, t1, t2, 0.5


def benchmark(label, sub_label, threads, f, *args, **kwargs):
    return Timer("f(*args, **kwargs)",
                 globals=locals(),
                 label=label,
                 description=f.__name__,
                 sub_label=sub_label,
                 num_threads=threads).blocked_autorange()


results = []
for args in gen_inputs():
    results.append(benchmark(*args))

compare = Compare(results)
compare.trim_significant_figures()
compare.print()

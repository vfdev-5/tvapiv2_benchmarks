import torch
from torch.utils.benchmark import Timer, Compare
from torchvision.transforms.functional import gaussian_blur as gaussian_blur_v1
from torchvision.prototype.transforms.functional import gaussian_blur as gaussian_blur_v2
from itertools import product
from functools import partial


def gen_inputs():
    make_arg_int = partial(torch.randint, 0, 256, dtype=torch.uint8)
    makers = (make_arg_int, torch.randn)
    shapes = ((3, 400, 400),)
    devices = ("cpu", "cuda")
    fns = (("v1", gaussian_blur_v1), ("v2", gaussian_blur_v2))
    threads = (1, torch.get_num_threads())
    for make, shape, device, tag_fn, threads in product(makers, shapes, devices, fns, threads):
        t1 = make(shape, device=device)
        yield f"Gaussian Blur {device} {t1.dtype}", str(tuple(shape)), threads, tag_fn[0], tag_fn[1], t1, 3, 0.7


def benchmark(label, sub_label, threads, tag, f, *args, **kwargs):
    return Timer("f(*args, **kwargs)",
                 globals=locals(),
                 label=label,
                 description=f.__name__ + f" {tag}",
                 sub_label=sub_label,
                 num_threads=threads).blocked_autorange(min_run_time=10)


results = []
for args in gen_inputs():
    results.append(benchmark(*args))

compare = Compare(results)
compare.trim_significant_figures()
compare.print()

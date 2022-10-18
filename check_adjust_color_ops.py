import torch
from torch.utils.benchmark import Timer, Compare
from torchvision.transforms import functional as F_stable
from torchvision.prototype.transforms import functional as F_v2
from itertools import product
from functools import partial


# min_run_time = 1
min_run_time = 10

def gen_inputs():
    make_arg_int = partial(torch.randint, 0, 256, dtype=torch.uint8)
    makers = (make_arg_int, torch.randn)
    shapes = ((3, 400, 400),)
    devices = ("cpu", "cuda")
    fns = ["adjust_brightness", "adjust_contrast", "adjust_saturation", "adjust_sharpness"]
    threads = (1, torch.get_num_threads())
    for make, shape, device, fn_name, threads in product(makers, shapes, devices, fns, threads):
        t1 = make(shape, device=device)

        fn = getattr(F_stable, fn_name)
        yield f"{fn_name.capitalize()} {device} {t1.dtype}", str(tuple(shape)), threads, "stable", fn, t1, 1.5

        fn = getattr(F_v2, fn_name)
        yield f"{fn_name.capitalize()} {device} {t1.dtype}", str(tuple(shape)), threads, "v2", fn, t1, 1.5


def benchmark(label, sub_label, threads, tag, f, *args, **kwargs):
    return Timer("f(*args, **kwargs)",
                 globals=locals(),
                 label=label,
                 description=f.__name__ + f" {tag}",
                 sub_label=sub_label,
                 num_threads=threads).blocked_autorange(min_run_time=min_run_time)


results = []
for args in gen_inputs():
    # print(args[:4])
    results.append(benchmark(*args))

compare = Compare(results)
compare.trim_significant_figures()
compare.print()

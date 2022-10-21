import torch
from torch.utils.benchmark import Timer, Compare
from torchvision.transforms import functional as F_stable
from torchvision.prototype.transforms import functional as F_v2
from itertools import product
from functools import partial


debug = False

if debug:
    min_run_time = 5
else:
    min_run_time = 10

def gen_inputs():
    make_arg_int = partial(torch.randint, 0, 256, dtype=torch.uint8)
    shapes = ((3, 400, 400),)

    if not debug:
        makers = (make_arg_int, torch.randn)
        devices = ("cpu", "cuda")
        fns = ["adjust_hue", ]
        threads = (1, torch.get_num_threads())
    else:
        makers = (make_arg_int, torch.randn)
        devices = ("cpu", )
        fns = ["adjust_hue", ]
        threads = (1, )

    for make, shape, device, fn_name, threads in product(makers, shapes, devices, fns, threads):
        t1 = make(shape, device=device)

        fn = getattr(F_stable, fn_name)
        yield f"{fn_name.capitalize()} {device} {t1.dtype}", str(tuple(shape)), threads, "stable", fn, t1, 0.5

        fn = getattr(F_v2, fn_name)
        yield f"{fn_name.capitalize()} {device} {t1.dtype}", str(tuple(shape)), threads, "v2", fn, t1, 0.5


def benchmark(label, sub_label, threads, tag, f, *args, **kwargs):
    if debug:
        f_ref = getattr(F_stable, f.__name__)
        if f is not f_ref:
            out = f(*args, **kwargs)
            ref = f_ref(*args, **kwargs)
            torch.testing.assert_close(ref, out)

    return Timer("f(*args, **kwargs)",
                 globals=locals(),
                 label=label,
                 description=f.__name__ + f" {tag}",
                 sub_label=sub_label,
                 num_threads=threads).blocked_autorange(min_run_time=min_run_time)


results = []
for args in gen_inputs():
    if debug:
        print(args[:4])
    results.append(benchmark(*args))

compare = Compare(results)
compare.trim_significant_figures()
compare.print()

import torch
from torch.utils.benchmark import Timer, Compare
from torchvision.prototype.transforms.functional import _geometry as F_v2
from itertools import product
from functools import partial

from prototype_common_utils import make_bounding_box_loaders


debug = False

if debug:
    min_run_time = 3
else:
    min_run_time = 10

def gen_inputs():

    make_bboxes = partial(make_bounding_box_loaders, extra_dims=[()])

    if not debug:
        makers = (make_bboxes, )
        devices = ("cpu", "cuda")
        fns = ["horizontal_flip_bounding_box", ]
        threads = (1, torch.get_num_threads())
    else:
        makers = (make_bboxes, )
        devices = ("cpu", )
        fns = ["horizontal_flip_bounding_box", ]
        threads = (1, )

    for make, device, fn_name, threads in product(makers, devices, fns, threads):
        for t1 in make():
            t1 = t1.load(device=device)

            fn_name_old = fn_name + "_old"
            fn = getattr(F_v2, fn_name_old)
            yield f"{fn_name} {device} {t1.format}", str(tuple(t1.shape)), threads, "v2", fn, t1, t1.format, t1.spatial_size

            fn = getattr(F_v2, fn_name)
            yield f"{fn_name} {device} {t1.format}", str(tuple(t1.shape)), threads, "v2", fn, t1, t1.format, t1.spatial_size

            # break


def benchmark(label, sub_label, threads, tag, f, *args, **kwargs):
    if debug:
        fn_name_old = "horizontal_flip_bounding_box_old"
        f_ref = getattr(F_v2, fn_name_old)

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

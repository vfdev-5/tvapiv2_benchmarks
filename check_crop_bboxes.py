import torch
from torch.utils.benchmark import Timer, Compare
from torchvision.prototype.transforms.functional import _geometry as F_v2
from itertools import product
from functools import partial

from prototype_common_utils import make_bounding_box_loaders


from typing import Tuple
from torchvision.prototype import features
from torchvision.prototype.transforms.functional._meta import (
    convert_format_bounding_box,
)


def crop_bounding_box_old(
    bounding_box: torch.Tensor,
    format: features.BoundingBoxFormat,
    top: int,
    left: int,
    height: int,
    width: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    # TODO: Investigate if it makes sense from a performance perspective to have an implementation for every
    #  BoundingBoxFormat instead of converting back and forth
    bounding_box = convert_format_bounding_box(
        bounding_box.clone(), old_format=format, new_format=features.BoundingBoxFormat.XYXY, inplace=True
    )

    # Crop or implicit pad if left and/or top have negative values:
    bounding_box[..., 0::2] -= left
    bounding_box[..., 1::2] -= top

    return (
        convert_format_bounding_box(
            bounding_box, old_format=features.BoundingBoxFormat.XYXY, new_format=format, inplace=True
        ),
        (height, width),
    )

F_v2.__dict__["crop_bounding_box_old"] = crop_bounding_box_old


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
        fns = ["crop_bounding_box", ]
        threads = (1, torch.get_num_threads())
    else:
        makers = (make_bboxes, )
        devices = ("cpu", )
        fns = ["crop_bounding_box", ]
        threads = (1, )

    for make, device, fn_name, threads in product(makers, devices, fns, threads):
        for t1 in make():
            t1 = t1.load(device=device)

            fn_name_old = fn_name + "_old"
            fn = getattr(F_v2, fn_name_old)
            yield f"{fn_name} {device} {t1.format}", str(tuple(t1.shape)), threads, "v2", fn, t1, t1.format, 1, 2, 3, 4

            fn = getattr(F_v2, fn_name)
            yield f"{fn_name} {device} {t1.format}", str(tuple(t1.shape)), threads, "v2", fn, t1, t1.format, 1, 2, 3, 4

            # break


def benchmark(label, sub_label, threads, tag, f, *args, **kwargs):
    # if debug:
    #     fn_name_old = "crop_bounding_box"
    #     f_ref = getattr(F_v2, fn_name_old)

    #     if f is not f_ref:
    #         out = f(*args, **kwargs)
    #         ref = f_ref(*args, **kwargs)
    #         torch.testing.assert_close(ref, out)

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

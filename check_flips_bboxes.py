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

def horizontal_flip_bounding_box_old(
    bounding_box: torch.Tensor, format: features.BoundingBoxFormat, spatial_size: Tuple[int, int]
) -> torch.Tensor:
    shape = bounding_box.shape

    # TODO: Investigate if it makes sense from a performance perspective to have an implementation for every
    #  BoundingBoxFormat instead of converting back and forth
    bounding_box = convert_format_bounding_box(
        bounding_box.clone(), old_format=format, new_format=features.BoundingBoxFormat.XYXY, inplace=True
    ).reshape(-1, 4)

    bounding_box[:, [0, 2]] = spatial_size[1] - bounding_box[:, [2, 0]]

    return convert_format_bounding_box(
        bounding_box, old_format=features.BoundingBoxFormat.XYXY, new_format=format, inplace=True
    ).reshape(shape)


def vertical_flip_bounding_box_old(
    bounding_box: torch.Tensor, format: features.BoundingBoxFormat, spatial_size: Tuple[int, int]
) -> torch.Tensor:
    shape = bounding_box.shape

    # TODO: Investigate if it makes sense from a performance perspective to have an implementation for every
    #  BoundingBoxFormat instead of converting back and forth
    bounding_box = convert_format_bounding_box(
        bounding_box.clone(), old_format=format, new_format=features.BoundingBoxFormat.XYXY, inplace=True
    ).reshape(-1, 4)

    bounding_box[:, [1, 3]] = spatial_size[0] - bounding_box[:, [3, 1]]

    return convert_format_bounding_box(
        bounding_box, old_format=features.BoundingBoxFormat.XYXY, new_format=format, inplace=True
    ).reshape(shape)


F_v2.__dict__["vertical_flip_bounding_box_old"] = vertical_flip_bounding_box_old
F_v2.__dict__["horizontal_flip_bounding_box_old"] = horizontal_flip_bounding_box_old


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
        fns = ["horizontal_flip_bounding_box", "vertical_flip_bounding_box"]
        threads = (1, torch.get_num_threads())
    else:
        makers = (make_bboxes, )
        devices = ("cpu", )
        fns = ["horizontal_flip_bounding_box", "vertical_flip_bounding_box"]
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

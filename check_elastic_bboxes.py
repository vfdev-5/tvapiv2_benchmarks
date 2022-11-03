import torch
from torch.utils.benchmark import Timer, Compare
from torchvision.prototype.transforms.functional import _geometry as F_v2
from itertools import product
from functools import partial

from prototype_common_utils import make_bounding_box_loaders


from typing import List
from torchvision.prototype import features
from torchvision.prototype.transforms.functional._meta import (
    convert_format_bounding_box,
)
from torchvision.transforms import functional_tensor as _FT


def elastic_bounding_box_old(
    bounding_box: torch.Tensor,
    format: features.BoundingBoxFormat,
    displacement: torch.Tensor,
) -> torch.Tensor:
    # TODO: add in docstring about approximation we are doing for grid inversion
    displacement = displacement.to(bounding_box.device)

    original_shape = bounding_box.shape
    bounding_box = (
        convert_format_bounding_box(bounding_box, old_format=format, new_format=features.BoundingBoxFormat.XYXY)
    ).reshape(-1, 4)

    # Question (vfdev-5): should we rely on good displacement shape and fetch image size from it
    # Or add spatial_size arg and check displacement shape
    spatial_size = displacement.shape[-3], displacement.shape[-2]

    id_grid = _FT._create_identity_grid(list(spatial_size)).to(bounding_box.device)
    # We construct an approximation of inverse grid as inv_grid = id_grid - displacement
    # This is not an exact inverse of the grid
    inv_grid = id_grid - displacement

    # Get points from bboxes
    points = bounding_box[:, [[0, 1], [2, 1], [2, 3], [0, 3]]].reshape(-1, 2)
    index_x = torch.floor(points[:, 0] + 0.5).to(dtype=torch.long)
    index_y = torch.floor(points[:, 1] + 0.5).to(dtype=torch.long)
    # Transform points:
    t_size = torch.tensor(spatial_size[::-1], device=displacement.device, dtype=displacement.dtype)
    transformed_points = (inv_grid[0, index_y, index_x, :] + 1) * 0.5 * t_size - 0.5

    transformed_points = transformed_points.reshape(-1, 4, 2)
    out_bbox_mins, _ = torch.min(transformed_points, dim=1)
    out_bbox_maxs, _ = torch.max(transformed_points, dim=1)
    out_bboxes = torch.cat([out_bbox_mins, out_bbox_maxs], dim=1).to(bounding_box.dtype)

    return convert_format_bounding_box(
        out_bboxes, old_format=features.BoundingBoxFormat.XYXY, new_format=format, inplace=True
    ).reshape(original_shape)


F_v2.__dict__["elastic_bounding_box_old"] = elastic_bounding_box_old


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
        fns = ["elastic_bounding_box", ]
        threads = (1, torch.get_num_threads())
    else:
        makers = (make_bboxes, )
        devices = ("cpu", )
        fns = ["elastic_bounding_box", ]
        threads = (1, )

    for make, device, fn_name, threads in product(makers, devices, fns, threads):
        for t1 in make():
            t1 = t1.load(device=device)

            displacement = torch.randn(1, 32, 32, 2)
            args = (t1.format, displacement)

            fn_name_old = fn_name + "_old"
            fn = getattr(F_v2, fn_name_old)
            yield f"{fn_name} {device} {t1.format}", str(tuple(t1.shape)), threads, "v2", fn, t1, *args

            fn = getattr(F_v2, fn_name)
            yield f"{fn_name} {device} {t1.format}", str(tuple(t1.shape)), threads, "v2", fn, t1, *args


def benchmark(label, sub_label, threads, tag, f, *args, **kwargs):
    if debug:
        fn_name_old = "elastic_bounding_box_old"
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

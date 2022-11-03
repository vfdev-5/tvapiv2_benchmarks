import torch
from torch.utils.benchmark import Timer, Compare
from torchvision.prototype.transforms.functional import _geometry as F_v2
from itertools import product
from functools import partial

from prototype_common_utils import make_bounding_box_loaders


from typing import Tuple, Union, List
from torchvision.prototype import features
from torchvision.transforms.functional_tensor import _parse_pad_padding


def pad_bounding_box_old(
    bounding_box: torch.Tensor,
    format: features.BoundingBoxFormat,
    spatial_size: Tuple[int, int],
    padding: Union[int, List[int]],
    padding_mode: str = "constant",
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    if padding_mode not in ["constant"]:
        # TODO: add support of other padding modes
        raise ValueError(f"Padding mode '{padding_mode}' is not supported with bounding boxes")

    left, right, top, bottom = _parse_pad_padding(padding)

    bounding_box = bounding_box.clone()

    # this works without conversion since padding only affects xy coordinates
    bounding_box[..., 0] += left
    bounding_box[..., 1] += top
    if format == features.BoundingBoxFormat.XYXY:
        bounding_box[..., 2] += left
        bounding_box[..., 3] += top

    height, width = spatial_size
    height += top + bottom
    width += left + right

    return bounding_box, (height, width)


F_v2.__dict__["pad_bounding_box_old"] = pad_bounding_box_old


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
        fns = ["pad_bounding_box", ]
        threads = (1, torch.get_num_threads())
    else:
        makers = (make_bboxes, )
        devices = ("cpu", )
        fns = ["pad_bounding_box", ]
        threads = (1, )

    for make, device, fn_name, threads in product(makers, devices, fns, threads):
        for t1 in make():
            t1 = t1.load(device=device)

            args = (t1.format, t1.spatial_size, (1, 2, 3, 4), "constant")

            fn_name_old = fn_name + "_old"
            fn = getattr(F_v2, fn_name_old)
            yield f"{fn_name} {device} {t1.format}", str(tuple(t1.shape)), threads, "v2", fn, t1, *args

            fn = getattr(F_v2, fn_name)
            yield f"{fn_name} {device} {t1.format}", str(tuple(t1.shape)), threads, "v2", fn, t1, *args


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

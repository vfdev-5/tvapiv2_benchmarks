import torch
from torch import Tensor
from torch.utils.benchmark import Timer, Compare
from itertools import product
from functools import partial


def fn_ref(img: Tensor) -> Tensor:
    r, g, b = img.unbind(dim=-3)
    # This implementation closely follows the TF one:
    # https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/image_ops_impl.py#L2105-L2138
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
    l_img = l_img.unsqueeze(dim=-3)
    return l_img


def fn_new_slow(img: Tensor) -> Tensor:
    l_img = (img * torch.tensor([[[0.2989]], [[0.587]], [[0.114]]], device=img.device)).sum(-3)
    l_img = l_img.to(img.dtype).unsqueeze(dim=-3)
    return l_img


def fn_new(img: Tensor) -> Tensor:
    r, g, b = img.unbind(dim=-3)
    l_img = (0.2989 * r).add_(g, alpha=0.587).add_(b, alpha=0.114)
    l_img = l_img.to(img.dtype).unsqueeze(dim=-3)
    return l_img


def gen_inputs():
    make_arg_int = partial(torch.randint, 0, 256, dtype=torch.uint8)
    makers = (make_arg_int, )
    shapes = ((3, 400, 400),)
    # devices = ("cpu", "cuda")
    devices = ("cpu", )
    fns = (fn_new, fn_ref)
    # threads = (1, torch.get_num_threads())
    threads = (1, )
    for make, shape, device, fn, threads in product(makers, shapes, devices, fns, threads):
        t1 = make(shape, device=device)
        yield f"RGB -> Gray {device} {t1.dtype}", str(tuple(shape)), threads, fn, t1


def benchmark(label, sub_label, threads, f, *args, **kwargs):

    if f is not fn_ref:
        out = f(*args, **kwargs)
        ref = fn_ref(*args, **kwargs)
        torch.testing.assert_close(ref, out)

    return Timer(
        "f(*args, **kwargs)",
        globals={
            "f": f,
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

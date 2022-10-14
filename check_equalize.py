import torch
import torch.utils.benchmark as benchmark
import random
from torch.utils.benchmark.utils import common, compare as benchmark_compare
from unittest.mock import patch


def compare_print(compare):
    # Hack benchmark.compare._Column to get more digits
    import itertools as it

    def _column__init__(
        self,
        grouped_results,
        time_scale: float,
        time_unit: str,
        trim_significant_figures: bool,
        highlight_warnings: bool,
    ):
        self._grouped_results = grouped_results
        self._flat_results = list(it.chain(*grouped_results))
        self._time_scale = time_scale
        self._time_unit = time_unit
        self._trim_significant_figures = trim_significant_figures
        self._highlight_warnings = highlight_warnings and any(r.has_warnings for r in self._flat_results if r)
        leading_digits = [
            int(torch.tensor(r.median / self._time_scale).log10().ceil()) if r else None for r in self._flat_results
        ]
        unit_digits = max(d for d in leading_digits if d is not None)
        decimal_digits = (
            min(
                max(m.significant_figures - digits, 0)
                for digits, m in zip(leading_digits, self._flat_results)
                if (m is not None) and (digits is not None)
            )
            if self._trim_significant_figures
            else 3
        )  # <---- 1 replaced by 3
        length = unit_digits + decimal_digits + (1 if decimal_digits else 0)
        self._template = f"{{:>{length}.{decimal_digits}f}}{{:>{7 if self._highlight_warnings else 0}}}"

    with patch.object(benchmark_compare._Column, "__init__", _column__init__):
        compare.print()


def run_bench_with_time(
    option,
    transform,
    tag,
    single_dtype=None,
    seed=22,
    target_types=None,
    size=None,
    num_runs=15,
    num_loops=1000,
    data=None,
):
    import time

    torch.set_num_threads(1)

    random.seed(seed)
    torch.manual_seed(seed)

    if data is not None:
        tested_dtypes = [(type(data), data)]
    else:
        if isinstance(single_dtype, dict):
            single_dtype_value = single_dtype[tag]
        else:
            single_dtype_value = single_dtype

        if single_dtype_value is not None:
            if not isinstance(single_dtype_value, (list, tuple)):
                single_dtype_value = [single_dtype_value, ]

            tested_dtypes = []
            for v in single_dtype_value:
                data = get_single_type_random_data(
                    option, single_dtype=v, target_types=target_types, size=size
                )
                tested_dtypes.append((v, data))
        else:
            tested_dtypes = [
                ("PIL", get_random_data_pil(option, target_types=target_types, size=size)),
                ("Tensor", get_random_data_tensor(option, target_types=target_types, size=size)),
                ("Feature", get_random_data_feature(option, target_types=target_types, size=size)),
            ]

    results = []
    for dtype_label, data in tested_dtypes:
        times = []

        label = f"{option} transforms measurements"
        sub_label = f"{dtype_label} Image data"
        description = tag
        task_spec = common.TaskSpec(
            stmt="",
            setup="",
            global_setup="",
            label=label,
            sub_label=sub_label,
            description=description,
            env=None,
            num_threads=torch.get_num_threads(),
        )

        for i in range(num_runs):
            started = time.time()
            for j in range(num_loops):

                random.seed(seed + i * num_loops + j)
                torch.manual_seed(seed + i * num_loops + j)
                transform(data)

            elapsed = time.time() - started
            times.append(elapsed)

        results.append(
            common.Measurement(number_per_run=num_loops, raw_times=times, task_spec=task_spec)
        )
    return results

def _equalize_image_tensor_vec(img: torch.Tensor) -> torch.Tensor:
    # input img shape should be [N, H, W]
    shape = img.shape
    # Compute image histogram:
    flat_img = img.flatten(start_dim=1).to(torch.long)  # -> [N, H * W]
    hist = flat_img.new_zeros(shape[0], 256)
    hist.scatter_add_(dim=1, index=flat_img, src=flat_img.new_ones(1).expand_as(flat_img))

    # Compute image cdf
    chist = hist.cumsum_(dim=1)
    # Compute steps, where step per channel is nonzero_hist[:-1].sum() // 255
    # Trick: nonzero_hist[:-1].sum() == chist[idx - 1], where idx = chist.argmax()
    idx = chist.argmax(dim=1).sub_(1)
    # If histogram is degenerate (hist of zero image), index is -1
    neg_idx_mask = idx < 0
    idx.clamp_(min=0)
    step = chist.gather(dim=1, index=idx.unsqueeze(1))
    step[neg_idx_mask] = 0
    step.div_(255, rounding_mode="floor")

    # Compute batched Look-up-table:
    # Necessary to avoid an integer division by zero, which raises
    clamped_step = step.clamp(min=1)
    chist.add_(torch.div(step, 2, rounding_mode="floor")).div_(clamped_step, rounding_mode="floor").clamp_(0, 255)
    lut = chist.to(torch.uint8)  # [N, 256]

    # Pad lut with zeros
    zeros = lut.new_zeros((1, 1)).expand(shape[0], 1)
    lut = torch.cat([zeros, lut[:, :-1]], dim=1)

    return torch.where((step == 0).unsqueeze(-1), img, lut.gather(dim=1, index=flat_img).view_as(img))


def equalize_image_tensor(image: torch.Tensor) -> torch.Tensor:
    if image.dtype != torch.uint8:
        raise TypeError(f"Only torch.uint8 image tensors are supported, but found {image.dtype}")

    height, width = image.shape[-2:]

    if image.numel() == 0:
        return image

    return _equalize_image_tensor_vec(image.view(-1, height, width)).view(image.shape)


def equalize_image_tensor_new(img):
    if img.dtype != torch.uint8:
        raise TypeError(f"Only torch.uint8 image tensors are supported, but found {img.dtype}")

    if img.numel() == 0:
        return img

    # input img shape should be [*, H, W]
    shape = img.shape
    # Compute image histogram:
    flat_img = img.flatten(start_dim=-2).to(torch.long)  # -> [*, H * W]
    hist = flat_img.new_zeros(shape[:-2] + (256,), dtype=torch.int32)
    hist.scatter_add_(dim=-1, index=flat_img, src=hist.new_ones(1).expand_as(flat_img))

    # Compute image cdf
    pixels = flat_img.size(-1)
    chist = hist.cumsum(dim=-1)
    # Compute steps, where step per channel is nonzero_hist[:-1].sum() // 255
    # Trick: nonzero_hist[-1] == hist[chist.argmax()]
    idx = chist.argmax(dim=-1).unsqueeze_(-1)
    step = pixels - hist.gather(dim=-1, index=idx)
    # We won't need these tensors anymore, so we can release the memory early
    idx = None
    hist = None
    step.div_(255, rounding_mode="floor")

    # Compute batched Look-up-table:
    non_zero = (step == 0)
    # Necessary to avoid an integer division by zero, which raises
    clamped_step = step.clamp_(min=1)
    chist = chist[..., :-1]
    chist.add_(torch.div(step, 2, rounding_mode="floor")) \
         .div_(clamped_step, rounding_mode="floor") \
         .clamp_(0, 255)
    lut = chist.to(torch.uint8)  # [*, 255]

    # Pad lut with zeros
    zeros = lut.new_zeros(1).expand(*lut.shape[:-1], 1)
    lut = torch.cat([zeros, lut], dim=-1)

    return torch.where(non_zero.unsqueeze_(-1), img, lut.gather(dim=-1, index=flat_img).view_as(img))


torch.manual_seed(12)
for device in ("cpu", "cuda"):
    data = torch.randint(0, 256, size=(3, 256, 256), dtype=torch.uint8, device=device)
    torch.testing.assert_close(equalize_image_tensor(data), equalize_image_tensor_new(data))

    all_results = []
    all_results.extend(run_bench_with_time(None, equalize_image_tensor, "main", data=data))
    all_results.extend(run_bench_with_time(None, equalize_image_tensor_new, "new", data=data))

    compare = benchmark.Compare(all_results)
    compare_print(compare)

from copy import deepcopy
from functools import partial

import det_transforms
import fire

import PIL.Image

import torch
import torch.utils.benchmark as benchmark
import torchvision
from torch.utils.benchmark.utils import common
from torchvision.prototype import features, transforms as transforms_v2
from torchvision.transforms import autoaugment as autoaugment_stable, transforms as transforms_stable
from torchvision.transforms.functional import InterpolationMode


def get_classification_transforms_stable_api(hflip_prob=1.0, auto_augment_policy=None, random_erase_prob=0.0):

    crop_size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    interpolation = InterpolationMode.BILINEAR

    trans = [transforms_stable.RandomResizedCrop(crop_size, interpolation=interpolation)]
    if hflip_prob > 0:
        trans.append(transforms_stable.RandomHorizontalFlip(hflip_prob))
    if auto_augment_policy is not None:
        if auto_augment_policy == "ra":
            trans.append(autoaugment_stable.RandAugment(interpolation=interpolation))
        elif auto_augment_policy == "ta_wide":
            trans.append(autoaugment_stable.TrivialAugmentWide(interpolation=interpolation))
        elif auto_augment_policy == "augmix":
            trans.append(autoaugment_stable.AugMix(interpolation=interpolation))
        else:
            aa_policy = autoaugment_stable.AutoAugmentPolicy(auto_augment_policy)
            trans.append(autoaugment_stable.AutoAugment(policy=aa_policy, interpolation=interpolation))

    ptt = transforms_stable.PILToTensor()

    def friendly_pil_to_tensor(image):
        if isinstance(image, PIL.Image.Image):
            return ptt(image)
        return image

    trans.extend(
        [
            friendly_pil_to_tensor,
            transforms_stable.ConvertImageDtype(torch.float),
            transforms_stable.Normalize(mean=mean, std=std),
        ]
    )
    if random_erase_prob > 0:
        trans.append(transforms_stable.RandomErasing(p=random_erase_prob))

    return transforms_stable.Compose(trans)


def get_classification_transforms_v2(hflip_prob=1.0, auto_augment_policy=None, random_erase_prob=0.0):
    crop_size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    interpolation = InterpolationMode.BILINEAR

    trans = [transforms_v2.RandomResizedCrop(crop_size, interpolation=interpolation)]
    if hflip_prob > 0:
        trans.append(transforms_v2.RandomHorizontalFlip(p=hflip_prob))
    if auto_augment_policy is not None:
        if auto_augment_policy == "ra":
            trans.append(transforms_v2.RandAugment(interpolation=interpolation))
        elif auto_augment_policy == "ta_wide":
            trans.append(transforms_v2.TrivialAugmentWide(interpolation=interpolation))
        elif auto_augment_policy == "augmix":
            trans.append(transforms_v2.AugMix(interpolation=interpolation))
        else:
            aa_policy = transforms_v2.AutoAugmentPolicy(auto_augment_policy)
            trans.append(transforms_v2.AutoAugment(policy=aa_policy, interpolation=interpolation))

    tit = transforms_v2.ToImageTensor()

    def friendly_to_image_tensor(image):
        if isinstance(image, PIL.Image.Image):
            return tit(image)
        return image

    trans.extend(
        [
            friendly_to_image_tensor,
            transforms_v2.ConvertImageDtype(torch.float),
            transforms_v2.Normalize(mean=mean, std=std),
        ]
    )
    if random_erase_prob > 0:
        trans.append(transforms_v2.RandomErasing(p=random_erase_prob))

    return transforms_v2.Compose(trans)


def get_classification_random_data_pil(size=(400, 500), **kwargs):
    return PIL.Image.new("RGB", size[::-1], 123)


def get_classification_random_data_tensor(size=(400, 500), **kwargs):
    return torch.randint(0, 256, size=(3, *size), dtype=torch.uint8)


def get_classification_random_data_feature(size=(400, 500), **kwargs):
    return features.Image(torch.randint(0, 256, size=(3, *size), dtype=torch.uint8))


class RefDetCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        image, target = sample
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


def copy_targets(*sample):
    image, target = sample if len(sample) == 2 else sample[0]
    target_copy = deepcopy(target)
    return image, target_copy


def get_detection_transforms_stable_api(data_augmentation, hflip_prob=1.0):
    mean = (123.0, 117.0, 104.0)

    ptt = det_transforms.PILToTensor()

    def friendly_pil_to_tensor(image, target):
        if isinstance(image, PIL.Image.Image):
            return ptt(image, target)
        return image, target

    if data_augmentation == "hflip":
        transforms = RefDetCompose(
            [
                copy_targets,
                det_transforms.RandomHorizontalFlip(p=hflip_prob),
                friendly_pil_to_tensor,
                det_transforms.ConvertImageDtype(torch.float),
            ]
        )
    elif data_augmentation == "lsj":
        transforms = RefDetCompose(
            [
                # As det_transforms.ScaleJitter does inplace transformations on mask
                # we perform a copy here and in v2
                copy_targets,
                det_transforms.ScaleJitter(target_size=(1024, 1024)),
                # Set fill as 0 to make it work on tensors
                det_transforms.FixedSizeCrop(size=(1024, 1024), fill=0),
                det_transforms.RandomHorizontalFlip(p=hflip_prob),
                friendly_pil_to_tensor,
                det_transforms.ConvertImageDtype(torch.float),
            ]
        )
    elif data_augmentation == "multiscale":
        transforms = RefDetCompose(
            [
                copy_targets,
                det_transforms.RandomShortestSize(
                    min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                    max_size=1333,
                ),
                det_transforms.RandomHorizontalFlip(p=hflip_prob),
                friendly_pil_to_tensor,
                det_transforms.ConvertImageDtype(torch.float),
            ]
        )
    elif data_augmentation == "ssd":
        transforms = RefDetCompose(
            [
                copy_targets,
                # det_transforms.RandomPhotometricDistort(p=1.0),
                det_transforms.RandomZoomOut(p=1.0, fill=[0, 0, 0]),
                det_transforms.RandomIoUCrop(),
                det_transforms.RandomHorizontalFlip(p=hflip_prob),
                friendly_pil_to_tensor,
                det_transforms.ConvertImageDtype(torch.float),
            ]
        )
    elif data_augmentation == "ssdlite":
        transforms = RefDetCompose(
            [
                copy_targets,
                det_transforms.RandomIoUCrop(),
                det_transforms.RandomHorizontalFlip(p=hflip_prob),
                friendly_pil_to_tensor,
                det_transforms.ConvertImageDtype(torch.float),
            ]
        )
    else:
        raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

    return transforms


class WrapIntoFeatures:
    def __call__(self, sample):
        image, target = sample

        if isinstance(image, PIL.Image.Image):
            image_size = (image.height, image.width)
        else:
            image_size = image.shape[-2:]

        wrapped_target = dict(
            boxes=features.BoundingBox(
                target["boxes"],
                format=features.BoundingBoxFormat.XYXY,
                image_size=image_size,
            ),
            labels=features.Label(target["labels"], categories=None),
        )
        if "masks" in target:
            wrapped_target["masks"] = features.SegmentationMask(target["masks"])

        return image, wrapped_target


def get_detection_transforms_v2(data_augmentation, hflip_prob=1.0):
    mean = (123.0, 117.0, 104.0)

    tit = transforms_v2.ToImageTensor()

    def friendly_to_image_tensor(sample):
        if isinstance(sample[0], PIL.Image.Image):
            return tit(sample)
        return sample

    if data_augmentation == "hflip":
        transforms = [
            copy_targets,
            WrapIntoFeatures(),
            transforms_v2.RandomHorizontalFlip(p=hflip_prob),
            friendly_to_image_tensor,
            transforms_v2.ConvertImageDtype(torch.float),
        ]
    elif data_augmentation == "lsj":
        transforms = [
            # As det_transforms.ScaleJitter does inplace transformations on mask
            # we perform a copy here and in stable
            copy_targets,
            WrapIntoFeatures(),
            transforms_v2.ScaleJitter(target_size=(1024, 1024)),
            # Set fill as 0 to make it work on tensors
            transforms_v2.FixedSizeCrop(size=(1024, 1024), fill=0),
            transforms_v2.RandomHorizontalFlip(p=hflip_prob),
            friendly_to_image_tensor,
            transforms_v2.ConvertImageDtype(torch.float),
        ]
    elif data_augmentation == "multiscale":
        transforms = [
            copy_targets,
            WrapIntoFeatures(),
            transforms_v2.RandomShortestSize(
                min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
            ),
            transforms_v2.RandomHorizontalFlip(p=hflip_prob),
            friendly_to_image_tensor,
            transforms_v2.ConvertImageDtype(torch.float),
        ]
    elif data_augmentation == "ssd":
        transforms = [
            copy_targets,
            WrapIntoFeatures(),
            # Can't check consistency vs stable API due to different random calls and implementation
            # transforms_v2.RandomPhotometricDistort(p=1.0),
            transforms_v2.RandomZoomOut(p=1.0, fill=[0, 0, 0]),
            transforms_v2.RandomIoUCrop(),
            transforms_v2.RandomHorizontalFlip(p=hflip_prob),
            friendly_to_image_tensor,
            transforms_v2.ConvertImageDtype(torch.float),
        ]
    elif data_augmentation == "ssdlite":
        transforms = [
            copy_targets,
            WrapIntoFeatures(),
            transforms_v2.RandomIoUCrop(),
            transforms_v2.RandomHorizontalFlip(p=hflip_prob),
            friendly_to_image_tensor,
            transforms_v2.ConvertImageDtype(torch.float),
        ]
    else:
        raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

    return transforms_v2.Compose(transforms)


def randint_with_tensor_bounds(arg1, arg2=None, **kwargs):
    low, high = torch.broadcast_tensors(
        *[torch.as_tensor(arg) for arg in ((0, arg1) if arg2 is None else (arg1, arg2))]
    )
    return torch.stack(
        [
            torch.randint(low_scalar, high_scalar, (), **kwargs)
            for low_scalar, high_scalar in zip(low.flatten().tolist(), high.flatten().tolist())
        ]
    ).reshape(low.shape)


def make_bounding_box(image_size, extra_dims, dtype=torch.long):
    height, width = image_size
    x1 = torch.randint(0, width // 2, extra_dims)
    y1 = torch.randint(0, height // 2, extra_dims)
    x2 = randint_with_tensor_bounds(x1 + 1, width - x1) + x1
    y2 = randint_with_tensor_bounds(y1 + 1, height - y1) + y1
    parts = (x1, y1, x2, y2)
    return torch.stack(parts, dim=-1).to(dtype)


def get_detection_random_data_pil(size=(600, 800), target_types=None, **kwargs):
    pil_image = PIL.Image.new("RGB", size[::-1], 123)
    target = {
        "boxes": make_bounding_box((600, 800), extra_dims=(22,), dtype=torch.float),
        "masks": torch.randint(0, 2, size=(22, *size), dtype=torch.long),
        "labels": torch.randint(0, 81, size=(22,)),
    }
    if target_types is not None:
        target = {k: target[k] for k in target_types}
    return pil_image, target


def get_detection_random_data_tensor(size=(600, 800), target_types=None, **kwargs):
    tensor_image = torch.randint(0, 256, size=(3, *size), dtype=torch.uint8)
    target = {
        "boxes": make_bounding_box((600, 800), extra_dims=(22,), dtype=torch.float),
        "masks": torch.randint(0, 2, size=(22, *size), dtype=torch.long),
        "labels": torch.randint(0, 81, size=(22,)),
    }
    if target_types is not None:
        target = {k: target[k] for k in target_types}
    return tensor_image, target


def get_detection_random_data_feature(size=(600, 800), target_types=None, **kwargs):
    feature_image = features.Image(torch.randint(0, 256, size=(3, *size), dtype=torch.uint8))
    target = {
        "boxes": make_bounding_box((600, 800), extra_dims=(22,), dtype=torch.float),
        "masks": torch.randint(0, 2, size=(22, *size), dtype=torch.long),
        "labels": torch.randint(0, 81, size=(22,)),
    }
    if target_types is not None:
        target = {k: target[k] for k in target_types}
    return feature_image, target


def _get_random_data_any(option, *, fn_classification, fn_detection, **kwargs):
    option = option.lower()
    if "classification" in option:
        return fn_classification(**kwargs)
    elif "detection" in option:
        return fn_detection(**kwargs)
    raise ValueError("Unsupported option '{option}'")


get_random_data_pil = partial(
    _get_random_data_any,
    fn_classification=get_classification_random_data_pil,
    fn_detection=get_detection_random_data_pil,
)


get_random_data_tensor = partial(
    _get_random_data_any,
    fn_classification=get_classification_random_data_tensor,
    fn_detection=get_detection_random_data_tensor,
)


get_random_data_feature = partial(
    _get_random_data_any,
    fn_classification=get_classification_random_data_feature,
    fn_detection=get_detection_random_data_feature,
)


def get_single_type_random_data(option, single_dtype="PIL", **kwargs):
    if single_dtype == "PIL":
        data = get_random_data_pil(option, **kwargs)
    elif single_dtype == "Tensor":
        data = get_random_data_tensor(option, **kwargs)
    elif single_dtype == "Feature":
        data = get_random_data_feature(option, **kwargs)
    else:
        raise ValueError(f"Unsupported single_dtype value: '{single_dtype}'")
    return data


def run_bench(option, transform, tag, single_dtype=None, seed=22, target_types=None):

    min_run_time = 10

    if single_dtype is not None:
        data = get_single_type_random_data(option, single_dtype=single_dtype, target_types=target_types)
        tested_dtypes = [(single_dtype, data)]
    else:
        tested_dtypes = [
            ("PIL", get_random_data_pil(option, target_types=target_types)),
            ("Tensor", get_random_data_tensor(option, target_types=target_types)),
            ("Feature", get_random_data_feature(option, target_types=target_types)),
        ]

    results = []
    for dtype_label, data in tested_dtypes:
        results.append(
            benchmark.Timer(
                stmt=f"torch.manual_seed({seed}); transform(data)",
                globals={
                    "data": data,
                    "transform": transform,
                },
                num_threads=torch.get_num_threads(),
                label=f"{option} transforms measurements",
                sub_label=f"{dtype_label} Image data",
                description=tag,
            ).blocked_autorange(min_run_time=min_run_time)
        )

    return results


def bench(option, t_stable, t_v2, quiet=True, single_dtype=None, seed=22, target_types=None):
    if not quiet:
        print("- Stable transforms:", t_stable)
        print("- Transforms v2:", t_v2)

    all_results = []
    for transform, tag in [(t_stable, "stable"), (t_v2, "v2")]:
        torch.manual_seed(seed)
        all_results += run_bench(
            option, transform, tag, single_dtype=single_dtype, seed=seed, target_types=target_types
        )
    compare = benchmark.Compare(all_results)
    compare.print()


def bench_with_time(
    option, t_stable, t_v2, quiet=True, single_dtype=None, seed=22, target_types=None, num_runs=10, num_loops=20
):
    if not quiet:
        print("- Stable transforms:", t_stable)
        print("- Transforms v2:", t_v2)

    import time

    torch.set_num_threads(1)

    all_results = []
    for transform, tag in [(t_stable, "stable"), (t_v2, "v2")]:
        torch.manual_seed(seed)

        if single_dtype is not None:
            data = get_single_type_random_data(option, single_dtype=single_dtype, target_types=target_types)
            tested_dtypes = [
                (single_dtype, data),
            ]
        else:
            tested_dtypes = [
                ("PIL", get_random_data_pil(option, target_types=target_types)),
                ("Tensor", get_random_data_tensor(option, target_types=target_types)),
                ("Feature", get_random_data_feature(option, target_types=target_types)),
            ]

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

                    torch.manual_seed(seed + i * num_loops + j)
                    transform(data)

                elapsed = time.time() - started
                times.append(elapsed)

            all_results.append(common.Measurement(number_per_run=num_loops, raw_times=times, task_spec=task_spec))

    compare = benchmark.Compare(all_results)
    compare.print()


def main_classification(
    hflip_prob=1.0,
    auto_augment_policy=None,
    random_erase_prob=0.0,
    quiet=True,
    single_dtype=None,
    seed=22,
    with_time=False,
    **kwargs,
):
    auto_augment_policies = [auto_augment_policy]
    random_erase_prob_list = [random_erase_prob]

    if auto_augment_policy == "all":
        auto_augment_policies = [None, "ra", "ta_wide", "augmix", "imagenet"]

    if random_erase_prob == "all":
        random_erase_prob_list = [0.0, 1.0]

    option = "Classification"

    bench_fn = bench if not with_time else bench_with_time

    for aa in auto_augment_policies:
        for re_prob in random_erase_prob_list:
            opt = option
            if aa is not None:
                opt += f" AA={aa}"
            if re_prob > 0.0:
                opt += f" RE={re_prob}"
            if not quiet:
                print(f"-- Benchmark: {opt}")
            t_stable = get_classification_transforms_stable_api(hflip_prob, aa, re_prob)
            t_v2 = get_classification_transforms_v2(hflip_prob, aa, re_prob)
            bench_fn(opt, t_stable, t_v2, quiet=quiet, single_dtype=single_dtype, seed=seed, num_runs=20, num_loops=50)

    if quiet:
        print("\n-----\n")
        for aa in auto_augment_policies:
            for re_prob in random_erase_prob_list:
                opt = option
                if aa is not None:
                    opt += f" AA={aa}"
                if re_prob > 0.0:
                    opt += f" RE={re_prob}"
                print(f"-- Benchmark: {opt}")
                t_stable = get_classification_transforms_stable_api(hflip_prob, aa, re_prob)
                t_v2 = get_classification_transforms_v2(hflip_prob, aa, re_prob)
                print("- Stable transforms:", t_stable)
                print("- Transforms v2:", t_v2)
                print("\n")


def main_detection(
    data_augmentation="hflip", hflip_prob=1.0, quiet=True, single_dtype=None, seed=22, with_time=False, **kwargs
):

    data_augmentation_list = [data_augmentation]

    if data_augmentation == "all":
        data_augmentation_list = ["hflip", "lsj", "multiscale", "ssd", "ssdlite"]

    option = "Detection"

    bench_fn = bench if not with_time else bench_with_time

    for a in data_augmentation_list:
        opt = option + f" da={a}"
        if not quiet:
            print(f"-- Benchmark: {opt}")
        t_stable = get_detection_transforms_stable_api(a, hflip_prob)
        t_v2 = get_detection_transforms_v2(a, hflip_prob)
        target_types = ["boxes", "labels"] if "ssd" in a else None
        bench_fn(opt, t_stable, t_v2, quiet=quiet, single_dtype=single_dtype, seed=seed, target_types=target_types)

    if quiet:
        print("\n-----\n")
        for a in data_augmentation_list:
            opt = option + f" da={a}"
            print(f"-- Benchmark: {opt}")
            t_stable = get_detection_transforms_stable_api(a, hflip_prob)
            t_v2 = get_detection_transforms_v2(a, hflip_prob)
            print("- Stable transforms:", t_stable)
            print("- Transforms v2:", t_v2)
            print("\n")


def main_debug(data_augmentation="hflip", hflip_prob=1.0, single_dtype="PIL", seed=22):

    t_stable = get_detection_transforms_stable_api(data_augmentation, hflip_prob)
    t_v2 = get_detection_transforms_v2(data_augmentation, hflip_prob)

    print("- Stable transforms:", t_stable)
    print("- Transforms v2:", t_v2)

    if "ssd" in data_augmentation:
        target_types = ["boxes", "labels"]
    else:
        target_types = None

    data = get_single_type_random_data("Detection", single_dtype=single_dtype, target_types=target_types)

    torch.manual_seed(seed)
    out_stable = t_stable(data)

    torch.manual_seed(seed)
    out_v2 = t_v2(data)

    print(data[0].shape if isinstance(data[0], torch.Tensor) else data[0].size, out_stable[0].shape, out_v2[0].shape)

    torch.testing.assert_close(out_stable[0], out_v2[0])
    target_stable, target_v2 = out_stable[1], out_v2[1]
    torch.testing.assert_close(target_stable["boxes"], target_v2["boxes"])
    torch.testing.assert_close(target_stable["labels"], target_v2["labels"])
    if "ssd" not in data_augmentation:
        torch.testing.assert_close(target_stable["masks"], target_v2["masks"])


def run_profiling(op, data, n=100):
    for _ in range(n):
        _ = op(data)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
        for _ in range(n):
            _ = op(data)

    print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))


def main_profile(data_augmentation="hflip", hflip_prob=1.0, single_dtype="PIL", seed=22):

    t_stable = get_detection_transforms_stable_api(data_augmentation, hflip_prob)
    t_v2 = get_detection_transforms_v2(data_augmentation, hflip_prob)
    data = get_single_type_random_data("Detection", single_dtype=single_dtype)

    print("\nProfile API v2")
    torch.manual_seed(seed)
    run_profiling(t_v2, data)

    print("\nProfile stable API")
    torch.manual_seed(seed)
    run_profiling(t_stable, data)


def main_profile_single_transform(t_name, t_args=(), t_kwargs={}, single_dtype="PIL", seed=22, n=100):

    print("Profile:", t_name, t_args, t_kwargs)

    if not hasattr(transforms_v2, t_name):
        raise ValueError("Unsupported transform name:", t_name)
    t_v2 = getattr(transforms_v2, t_name)(*t_args, **t_kwargs)

    option = "Classification"
    if hasattr(transforms_stable, t_name):
        t_stable = getattr(transforms_stable, t_name)(*t_args, **t_kwargs)
    elif hasattr(det_transforms, t_name):
        t_stable = getattr(det_transforms, t_name)(*t_args, **t_kwargs)
        option = "Detection"
    else:
        raise ValueError("Stable API does not have transform with name:", t_name)

    if option == "Detection":
        t_stable = RefDetCompose([copy_targets, t_stable])
        t_v2 = transforms_v2.Compose([copy_targets, WrapIntoFeatures(), t_v2])

    data = get_single_type_random_data(option, single_dtype=single_dtype)

    print("\nProfile stable API")
    print(t_stable)
    torch.manual_seed(seed)
    run_profiling(t_stable, data, n=n)

    print("\nProfile API v2")
    print(t_v2)
    torch.manual_seed(seed)
    run_profiling(t_v2, data, n=n)


def main_bench_with_time(data_augmentation="hflip", hflip_prob=1.0, quiet=True, single_dtype=None, seed=22, **kwargs):
    data_augmentation_list = [data_augmentation]

    if data_augmentation == "all":
        data_augmentation_list = ["hflip", "lsj", "multiscale", "ssd", "ssdlite"]

    option = "Detection"

    for a in data_augmentation_list:
        opt = option + f" da={a}"
        if not quiet:
            print(f"-- Benchmark: {opt}")
        t_stable = get_detection_transforms_stable_api(a, hflip_prob)
        t_v2 = get_detection_transforms_v2(a, hflip_prob)
        bench_with_time(opt, t_stable, t_v2, quiet=quiet, single_dtype=single_dtype, seed=seed)

    if quiet:
        print("\n-----\n")
        for a in data_augmentation_list:
            opt = option + f" da={a}"
            print(f"-- Benchmark: {opt}")
            t_stable = get_detection_transforms_stable_api(a, hflip_prob)
            t_v2 = get_detection_transforms_v2(a, hflip_prob)
            print("- Stable transforms:", t_stable)
            print("- Transforms v2:", t_v2)
            print("\n")


if __name__ == "__main__":
    from datetime import datetime

    print(f"Timestamp: {datetime.now().strftime('%Y%m%d-%H%M%S')}")
    print(f"Torch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"Num threads: {torch.get_num_threads()}")
    print("")

    fire.Fire(
        {
            "classification": main_classification,
            "detection": main_detection,
            "debug": main_debug,
            "profile": main_profile,
            "with_time": main_bench_with_time,
            "profile_transform": main_profile_single_transform,
        }
    )


# TODO:
#
# 10) add and measure anti-aliasing=True option

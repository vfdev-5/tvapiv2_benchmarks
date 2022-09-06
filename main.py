from copy import deepcopy

import cl_transforms
import det_transforms
import fire

import PIL.Image

import torch
import torch.utils.benchmark as benchmark
import torchvision
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


def get_classification_random_data(size=(400, 500), **kwargs):
    pil_image = PIL.Image.new("RGB", size[::-1], 123)
    tensor_image = torch.randint(0, 256, size=(3, *size), dtype=torch.uint8)
    feature_image = features.Image(torch.randint(0, 256, size=(3, *size), dtype=torch.uint8))

    return pil_image, tensor_image, feature_image


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
                det_transforms.RandomPhotometricDistort(p=1.0),
                det_transforms.RandomZoomOut(p=1.0, fill=list(mean)),
                det_transforms.RandomIoUCrop(),
                det_transforms.RandomHorizontalFlip(p=hflip_prob),
                friendly_pil_to_tensor,
                det_transforms.ConvertImageDtype(torch.float),
            ]
        )
    elif data_augmentation == "ssdlite":
        transforms = RefDetCompose(
            [
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
            masks=features.SegmentationMask(target["masks"]),
        )

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
            WrapIntoFeatures(),
            transforms_v2.RandomPhotometricDistort(p=1.0),
            transforms_v2.RandomZoomOut(p=1.0, fill=list(mean)),
            transforms_v2.RandomIoUCrop(),
            transforms_v2.RandomHorizontalFlip(p=hflip_prob),
            friendly_to_image_tensor,
            transforms_v2.ConvertImageDtype(torch.float),
        ]
    elif data_augmentation == "ssdlite":
        transforms = [
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


def get_detection_random_data(size=(600, 800), **kwargs):
    pil_image = PIL.Image.new("RGB", size[::-1], 123)
    tensor_image = torch.randint(0, 256, size=(3, *size), dtype=torch.uint8)
    feature_image = features.Image(torch.randint(0, 256, size=(3, *size), dtype=torch.uint8))

    target = {
        "boxes": make_bounding_box((600, 800), extra_dims=(22,), dtype=torch.float),
        "masks": torch.randint(0, 2, size=(22, *size), dtype=torch.long),
        "labels": torch.randint(0, 81, size=(22,)),
    }

    return [pil_image, target], [tensor_image, target], [feature_image, target]


def get_random_data(option, **kwargs):
    option = option.lower()
    if "classification" in option:
        return get_classification_random_data(**kwargs)
    elif "detection" in option:
        return get_detection_random_data(**kwargs)
    raise ValueError("Unsupported option '{option}'")


def run_bench(option, transform, tag):

    min_run_time = 7
    pil_image_data, tensor_data, feature_image_data = get_random_data(option)

    results = [
        benchmark.Timer(
            stmt=f"transform(data)",
            globals={
                "data": pil_image_data,
                "transform": transform,
            },
            num_threads=torch.get_num_threads(),
            label=f"{option} transforms measurements",
            sub_label="PIL Image data",
            description=tag,
        ).blocked_autorange(min_run_time=min_run_time),
        benchmark.Timer(
            stmt=f"transform(data)",
            globals={
                "data": tensor_data,
                "transform": transform,
            },
            num_threads=torch.get_num_threads(),
            label=f"{option} transforms measurements",
            sub_label="Tensor Image data",
            description=tag,
        ).blocked_autorange(min_run_time=min_run_time),
        benchmark.Timer(
            stmt=f"transform(data)",
            globals={
                "data": feature_image_data,
                "transform": transform,
            },
            num_threads=torch.get_num_threads(),
            label=f"{option} transforms measurements",
            sub_label="Feature Image data",
            description=tag,
        ).blocked_autorange(min_run_time=min_run_time),
    ]
    return results


def bench(option, transforms_stable, transforms_v2, quiet=True):
    if not quiet:
        print("- Stable transforms:", transforms_stable)
        print("- Transforms v2:", transforms_v2)

    all_results = []
    for transform, tag in [(transforms_stable, "stable"), (transforms_v2, "v2")]:
        torch.manual_seed(12)
        all_results += run_bench(option, transform, tag)
    compare = benchmark.Compare(all_results)
    compare.print()


def main_classification(hflip_prob=1.0, auto_augment_policy=None, random_erase_prob=0.0, quiet=True, **kwargs):
    auto_augment_policies = [
        auto_augment_policy,
    ]
    random_erase_prob_list = [
        random_erase_prob,
    ]

    if auto_augment_policy == "all":
        auto_augment_policies = [None, "ra", "ta_wide", "augmix", "imagenet"]

    if random_erase_prob == "all":
        random_erase_prob_list = [0.0, 1.0]

    option = "Classification"

    for aa in auto_augment_policies:
        for re_prob in random_erase_prob_list:
            opt = option
            if aa is not None:
                opt += f" AA={aa}"
            if re_prob > 0.0:
                opt += f" RE={re_prob}"
            if not quiet:
                print(f"-- Benchmark: {opt}")
            transforms_stable = get_classification_transforms_stable_api(hflip_prob, aa, re_prob)
            transforms_v2 = get_classification_transforms_v2(hflip_prob, aa, re_prob)
            bench(opt, transforms_stable, transforms_v2, quiet=quiet)

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
                transforms_stable = get_classification_transforms_stable_api(hflip_prob, aa, re_prob)
                transforms_v2 = get_classification_transforms_v2(hflip_prob, aa, re_prob)
                print("- Stable transforms:", transforms_stable)
                print("- Transforms v2:", transforms_v2)
                print("\n")


def main_detection(data_augmentation="hflip", hflip_prob=1.0, quiet=True, **kwargs):

    data_augmentation_list = [
        data_augmentation,
    ]

    if data_augmentation == "all":
        data_augmentation_list = ["hflip", "lsj", "multiscale", "ssd", "ssdlite"]

    option = "Detection"

    for a in data_augmentation_list:
        opt = option + f" da={a}"
        if not quiet:
            print(f"-- Benchmark: {opt}")
        transforms_stable = get_detection_transforms_stable_api(a, hflip_prob)
        transforms_v2 = get_detection_transforms_v2(a, hflip_prob)
        bench(opt, transforms_stable, transforms_v2, quiet=quiet)

    if quiet:
        print("\n-----\n")
        for a in data_augmentation_list:
            opt = option + f" da={a}"
            print(f"-- Benchmark: {opt}")
            transforms_stable = get_detection_transforms_stable_api(a, hflip_prob)
            transforms_v2 = get_detection_transforms_v2(a, hflip_prob)
            print("- Stable transforms:", transforms_stable)
            print("- Transforms v2:", transforms_v2)
            print("\n")


if __name__ == "__main__":
    print(f"Torch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"Num threads: {torch.get_num_threads()}")
    print("")

    fire.Fire(
        {
            "classification": main_classification,
            "detection": main_detection,
        }
    )

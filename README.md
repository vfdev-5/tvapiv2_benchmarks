# Utils to benchmark torchvision v2 vs stable on ref dataaugs

## Installation

- Install torchvision from a commit
- Install stable ref transforms:
```bash
curl https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py -o det_transforms.py
curl https://raw.githubusercontent.com/pytorch/vision/main/references/segmentation/transforms.py -o seg_transforms.py

curl https://raw.githubusercontent.com/pytorch/vision/main/test/prototype_common_utils.py -o prototype_common_utils.py
curl https://raw.githubusercontent.com/pytorch/vision/main/test/datasets_utils.py -o datasets_utils.py
curl https://raw.githubusercontent.com/pytorch/vision/main/test/common_utils.py -o common_utils.py
```
- Install `numpy`, `python-fire`:
```bash
pip install numpy
pip install fire
```

## Run benchmarks

```bash
bash bench.sh
```

### Outputs

```bash
ls output
```


## Transforms profiling

```bash
python -u main.py profile_transform --t_name=RandomResizedCrop --t_args="(224,)"

python -u main.py profile_transform --t_name=RandomErasing --t_args="(1.0, )" --single_dtype=Tensor

python -u main.py profile_tensor_vs_feature --n=10

python -u main.py classification --with_time --single_api=v2 --auto_augment_policy=None --random_erase_prob=0.0

python -u main.py cprofile_tensor_vs_feature --n=1000
```



### Feature vs Tensor

- Resize
```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     2000    0.607    0.000    0.607    0.000 {method 'to' of 'torch._C._TensorBase' objects}     # <----- cast to f32 ~ upsample
     1000    0.356    0.000    0.356    0.000 {built-in method torch._C._nn.upsample_bilinear2d}

     1000    0.010    0.000    0.024    0.000 /vision/torchvision/prototype/features/_feature.py:37(new_like)
    10000    0.009    0.000    0.014    0.000 /vision/torchvision/prototype/features/_feature.py:56(__torch_function__)

     1000    0.004    0.000    0.004    0.000 {built-in method _make_subclass}
     2000    0.003    0.000    0.003    0.000 {built-in method torch.as_tensor}

     1000    0.003    0.000    0.027    0.000 /vision/torchvision/prototype/features/_image.py:70(new_like)
     1000    0.002    0.000    0.012    0.000 /vision/torchvision/prototype/features/_image.py:39(__new__)
```

- Resized Crop

```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     1000    0.275    0.000    0.275    0.000 {built-in method torch._C._nn.upsample_bilinear2d}
     2000    0.130    0.000    0.130    0.000 {method 'to' of 'torch._C._TensorBase' objects}
2000/1000    0.102    0.000    0.104    0.000 {method 'flip' of 'torch._C._TensorBase' objects}
     1000    0.042    0.000    0.042    0.000 {built-in method torch.round}

    16000    0.024    0.000    0.123    0.000 /vision/torchvision/prototype/features/_feature.py:56(__torch_function__)
     2000    0.018    0.000    0.046    0.000 /vision/torchvision/prototype/features/_feature.py:37(new_like)
     2000    0.009    0.000    0.009    0.000 {built-in method _make_subclass}
     4000    0.006    0.000    0.006    0.000 {built-in method torch.as_tensor}
     2000    0.006    0.000    0.052    0.000 /vision/torchvision/prototype/features/_image.py:70(new_like)
     2000    0.005    0.000    0.023    0.000 /vision/torchvision/prototype/features/_image.py:39(__new__)
     1000    0.004    0.000    0.008    0.000 /vision/torchvision/prototype/features/_image.py:78(image_size)
     2000    0.003    0.000    0.013    0.000 /vision/torchvision/prototype/features/_feature.py:20(__new__)
     1000    0.002    0.000    0.137    0.000 /vision/torchvision/prototype/features/_image.py:127(horizontal_flip)
     2000    0.001    0.000    0.002    0.000 /vision/torchvision/prototype/features/_feature.py:13(is_simple_tensor)
     2000    0.001    0.000    0.001    0.000 /vision/torchvision/prototype/features/_feature.py:102(_F)
```
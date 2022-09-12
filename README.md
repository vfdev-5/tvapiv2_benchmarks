# Utils to benchmark torchvision v2 vs stable on ref dataaugs

## Installation

- Install torchvision from a commit
- Install stable ref transforms:
```bash
curl https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py -o det_transforms.py
```
- Install `python-fire`:
```bash
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
```



### Feature vs Tensor

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
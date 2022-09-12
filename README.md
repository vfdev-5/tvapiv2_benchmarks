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
```

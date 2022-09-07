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

```
bash bench.sh
```

### Outputs

```
ls output
```


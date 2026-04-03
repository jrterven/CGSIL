# CGSIL

Class-wise Gradient Surgery for Imbalanced Learning on CIFAR long-tailed benchmarks.

## Features

- ERM baseline
- Class-weighted cross entropy baseline
- Focal loss baseline
- CGSIL head-vs-tail gradient surgery
- CIFAR-10-LT and CIFAR-100-LT generation from standard torchvision CIFAR datasets
- CSV logging with accuracy, macro-F1, balanced accuracy, per-class recall, and per-class precision
- Best checkpoint selection by balanced accuracy

## Repository layout

- `train_cifar_lt.py` training and evaluation entrypoint
- `datasets/cifar_lt.py` long-tailed dataset builder
- `losses.py` focal loss
- `utils/metrics.py` macro-F1, balanced accuracy, per-class recall and precision
- `grad_surgery.py` CGSIL gradient operators

## Environment

PyTorch is expected to already be installed. Minimal extra packages are listed in `requirements.txt`.

Example setup:

```bash
conda activate cgsil
pip install -r requirements.txt
```

## Usage

### ERM

```bash
python train_cifar_lt.py \
  --dataset cifar10 \
  --method erm \
  --imbalance-ratio 100 \
  --epochs 200 \
  --batch-size 128 \
  --download
```

### Weighted cross entropy

```bash
python train_cifar_lt.py \
  --dataset cifar10 \
  --method weighted_ce \
  --imbalance-ratio 100 \
  --epochs 200 \
  --batch-size 128 \
  --download
```

### Focal loss

```bash
python train_cifar_lt.py \
  --dataset cifar10 \
  --method focal \
  --imbalance-ratio 100 \
  --focal-gamma 2.0 \
  --epochs 200 \
  --batch-size 128 \
  --download
```

### CGSIL

```bash
python train_cifar_lt.py \
  --dataset cifar10 \
  --method cgsil \
  --imbalance-ratio 100 \
  --tail-quantile 0.3 \
  --beta-start 0.8 \
  --beta-end 0.9 \
  --epochs 200 \
  --batch-size 128 \
  --download
```

## Outputs

Each run writes to `outputs/<experiment-name>/`:

- `metrics.csv`
- `config.json`
- `best_balanced_accuracy.pt`

## Notes on CGSIL

For each batch, the script:

1. Computes per-sample cross entropy losses.
2. Averages losses within each class present in the batch.
3. Builds class gradients with `torch.autograd.grad`.
4. Aggregates them into head and tail groups using the global class-frequency split.
5. Projects the head gradient away from the tail gradient when their dot product is negative.
6. Combines gradients using a linear `beta` schedule.

## Paper-oriented roadmap

Planned next steps:

- Full per-class surgery with class caps per batch for speed
- LDAM-DRW and Balanced Softmax baselines
- Tabular credit-card fraud experiments with AUROC and PR-AUC
- Plotting utilities and LaTeX-ready tables/sections

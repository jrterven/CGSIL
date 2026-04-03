from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

try:
    from tqdm import tqdm
except ImportError:
    class _TQDMFallback:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs):
            return None

    def tqdm(iterable, **kwargs):
        return _TQDMFallback(iterable, **kwargs)

from datasets.cifar_lt import build_cifar_lt_datasets
from grad_surgery import assign_gradient_vector, compute_group_cgsil_gradient
from losses import FocalLoss
from utils.metrics import compute_classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR long-tailed baselines and CGSIL")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--method", type=str, default="erm", choices=["erm", "weighted_ce", "focal", "cgsil"])
    parser.add_argument("--imbalance-ratio", type=float, default=100.0)
    parser.add_argument("--imbalance-type", type=str, default="exp", choices=["exp", "step", "none"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--lr-warmup-epochs", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--tail-quantile", type=float, default=0.3)
    parser.add_argument("--beta-start", type=float, default=0.8)
    parser.add_argument("--beta-end", type=float, default=0.9)
    parser.add_argument("--min-tail-samples", type=int, default=1)
    parser.add_argument("--min-tail-classes", type=int, default=1)
    parser.add_argument("--conflict-threshold", type=float, default=0.0)
    parser.add_argument("--cgsil-on-top-of", type=str, default="ce", choices=["ce", "weighted_ce", "focal"])
    parser.add_argument("--surgery-scope", type=str, default="all", choices=["all", "fc"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CSVLogger:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.fieldnames = None

    def log(self, row: dict) -> None:
        row = dict(row)
        if "per_class_recall" in row and not isinstance(row["per_class_recall"], str):
            row["per_class_recall"] = json.dumps(row["per_class_recall"])
        if "per_class_precision" in row and not isinstance(row["per_class_precision"], str):
            row["per_class_precision"] = json.dumps(row["per_class_precision"])
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
            write_header = True
        else:
            write_header = not self.file_path.exists() or self.file_path.stat().st_size == 0
        with self.file_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


class WarmupCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, total_epochs: int, warmup_epochs: int = 5, min_lr: float = 1e-5):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, epoch: int) -> None:
        if epoch < self.warmup_epochs:
            factor = float(epoch + 1) / float(max(1, self.warmup_epochs))
            lrs = [base_lr * factor for base_lr in self.base_lrs]
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs - 1)
            lrs = [self.min_lr + 0.5 * (base_lr - self.min_lr) * (1.0 + np.cos(np.pi * progress)) for base_lr in self.base_lrs]
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr


def build_model(num_classes: int) -> nn.Module:
    model = resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_transforms():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, test_transform


def build_tail_classes(class_counts: list[int], tail_quantile: float) -> list[int]:
    num_tail = max(1, int(np.ceil(len(class_counts) * tail_quantile)))
    sorted_classes = np.argsort(np.asarray(class_counts))
    return sorted_classes[:num_tail].astype(int).tolist()


def build_loss(method: str, class_weights: torch.Tensor | None, focal_gamma: float) -> nn.Module:
    if method == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    if method == "focal":
        return FocalLoss(gamma=focal_gamma, alpha=class_weights)
    return nn.CrossEntropyLoss()


def compute_per_sample_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_name: str,
    class_weights: torch.Tensor | None,
    focal_gamma: float,
) -> torch.Tensor:
    if loss_name == "weighted_ce":
        return nn.functional.cross_entropy(logits, targets, weight=class_weights, reduction="none")
    if loss_name == "focal":
        focal_loss = FocalLoss(gamma=focal_gamma, alpha=class_weights, reduction="none")
        return focal_loss(logits, targets)
    return nn.functional.cross_entropy(logits, targets, reduction="none")


def get_class_weights(class_counts: list[int], device: torch.device) -> torch.Tensor:
    counts = torch.tensor(class_counts, dtype=torch.float32, device=device)
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    weights = weights / weights.mean()
    return weights


def get_surgery_parameters(model: nn.Module, scope: str) -> list[torch.nn.Parameter]:
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if scope == "fc":
        selected = [
            parameter
            for name, parameter in named_parameters
            if name == "fc.weight" or name == "fc.bias" or name.endswith(".fc.weight") or name.endswith(".fc.bias")
        ]
        if selected:
            return selected
    return [parameter for _, parameter in named_parameters]


def move_to_device(model: nn.Module, device: torch.device) -> nn.Module:
    model = model.to(device)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)
    return model


def get_beta(epoch: int, total_epochs: int, beta_start: float, beta_end: float) -> float:
    if total_epochs <= 1:
        return beta_end
    progress = epoch / float(total_epochs - 1)
    return beta_start + (beta_end - beta_start) * progress


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    max_eval_batches: int | None = None,
) -> dict:
    model.eval()
    losses = []
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if loss.ndim > 0:
                loss = loss.mean()
            losses.append(float(loss.item()))
            preds = logits.argmax(dim=1)
            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            if max_eval_batches is not None and len(losses) >= max_eval_batches:
                break

    metrics = compute_classification_metrics(all_targets, all_preds, num_classes)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    class_weights: torch.Tensor,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    tail_classes: list[int],
    num_classes: int,
) -> dict:
    model.train()
    train_losses = []
    stats = defaultdict(list)
    surgery_params = get_surgery_parameters(model, args.surgery_scope)
    tail_class_set = set(int(class_id) for class_id in tail_classes)

    progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
    for step, (images, targets) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)

        if args.method == "cgsil":
            per_sample_losses = compute_per_sample_loss(
                logits,
                targets,
                args.cgsil_on_top_of,
                class_weights,
                args.focal_gamma,
            )
            base_loss = per_sample_losses.mean()
            per_class_losses = {}
            class_counts_in_batch = torch.bincount(targets, minlength=num_classes)
            unique_classes = torch.unique(targets)
            for class_id in unique_classes.tolist():
                mask = targets == class_id
                per_class_losses[int(class_id)] = per_sample_losses[mask].mean()

            tail_classes_present = [class_id for class_id in tail_classes if int(class_counts_in_batch[class_id].item()) > 0]
            tail_samples_in_batch = int(sum(int(class_counts_in_batch[class_id].item()) for class_id in tail_classes if class_id < num_classes))
            head_classes_present = [class_id for class_id in unique_classes.tolist() if class_id not in tail_class_set]
            cgsil_active = (
                epoch >= args.warmup_epochs
                and tail_samples_in_batch >= args.min_tail_samples
                and len(tail_classes_present) >= args.min_tail_classes
                and len(head_classes_present) > 0
            )
            beta = get_beta(epoch, args.epochs, args.beta_start, args.beta_end)

            base_loss.backward(retain_graph=cgsil_active)

            if cgsil_active:
                cgsil_output = compute_group_cgsil_gradient(
                    per_class_losses,
                    surgery_params,
                    tail_classes,
                    beta=beta,
                    conflict_threshold=args.conflict_threshold,
                )
                assign_gradient_vector(surgery_params, cgsil_output["gradient"])
            else:
                cgsil_output = {
                    "dot": 0.0,
                    "cosine": 0.0,
                    "tail_classes_present": tail_classes_present,
                    "head_classes_present": head_classes_present,
                    "surgery_applied": False,
                }

            optimizer.step()
            loss = base_loss.detach()
            stats["dot"].append(cgsil_output["dot"])
            stats["cosine"].append(cgsil_output["cosine"])
            stats["tail_classes_in_batch"].append(float(len(cgsil_output["tail_classes_present"])))
            stats["head_classes_in_batch"].append(float(len(cgsil_output["head_classes_present"])))
            stats["tail_samples_in_batch"].append(float(tail_samples_in_batch))
            stats["beta"].append(beta)
            stats["cgsil_active"].append(float(cgsil_active))
            stats["surgery_applied"].append(float(cgsil_output["surgery_applied"]))
        else:
            loss = criterion(logits, targets)
            if loss.ndim > 0:
                loss = loss.mean()
            loss.backward()
            optimizer.step()

        train_losses.append(float(loss.item()))
        if (step + 1) % args.print_freq == 0 or step == 0:
            progress.set_postfix(loss=f"{np.mean(train_losses):.4f}")
        if args.max_train_batches is not None and (step + 1) >= args.max_train_batches:
            break

    return {
        "loss": float(np.mean(train_losses)) if train_losses else 0.0,
        "dot": float(np.mean(stats["dot"])) if stats["dot"] else 0.0,
        "cosine": float(np.mean(stats["cosine"])) if stats["cosine"] else 0.0,
        "tail_classes_in_batch": float(np.mean(stats["tail_classes_in_batch"])) if stats["tail_classes_in_batch"] else 0.0,
        "head_classes_in_batch": float(np.mean(stats["head_classes_in_batch"])) if stats["head_classes_in_batch"] else 0.0,
        "tail_samples_in_batch": float(np.mean(stats["tail_samples_in_batch"])) if stats["tail_samples_in_batch"] else 0.0,
        "beta": float(np.mean(stats["beta"])) if stats["beta"] else 0.0,
        "cgsil_active": float(np.mean(stats["cgsil_active"])) if stats["cgsil_active"] else 0.0,
        "surgery_applied": float(np.mean(stats["surgery_applied"])) if stats["surgery_applied"] else 0.0,
    }


def save_checkpoint(checkpoint_path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict, args: argparse.Namespace, class_counts: list[int], tail_classes: list[int]) -> None:
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(
        {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
            "class_counts": class_counts,
            "tail_classes": tail_classes,
        },
        checkpoint_path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_transform, test_transform = build_transforms()
    train_dataset, test_dataset, info = build_cifar_lt_datasets(
        dataset_name=args.dataset,
        root=args.data_root,
        imbalance_ratio=args.imbalance_ratio,
        train_transform=train_transform,
        test_transform=test_transform,
        imbalance_type=args.imbalance_type,
        download=args.download,
        seed=args.seed,
    )

    experiment_name = args.experiment_name or f"{args.dataset}_{args.method}_ir{args.imbalance_ratio:g}_seed{args.seed}"
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = move_to_device(build_model(info.num_classes), device)
    class_weights = get_class_weights(info.class_counts, device)
    criterion = build_loss(args.method, class_weights if args.method in {"weighted_ce", "focal"} else None, args.focal_gamma)
    criterion = criterion.to(device)
    eval_criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    scheduler = WarmupCosineScheduler(optimizer, total_epochs=args.epochs, warmup_epochs=args.lr_warmup_epochs)
    tail_classes = build_tail_classes(info.class_counts, args.tail_quantile)
    logger = CSVLogger(output_dir / "metrics.csv")

    best_balanced_accuracy = -float("inf")
    best_checkpoint_path = output_dir / "best_balanced_accuracy.pt"

    config_path = output_dir / "config.json"
    with config_path.open("w") as handle:
        json.dump(
            {
                **vars(args),
                "num_classes": info.num_classes,
                "class_counts": info.class_counts,
                "class_names": info.class_names,
                "tail_classes": tail_classes,
            },
            handle,
            indent=2,
        )

    for epoch in range(args.epochs):
        scheduler.step(epoch)
        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, class_weights, device, epoch, args, tail_classes, info.num_classes)
        val_metrics = evaluate(model, test_loader, eval_criterion, device, info.num_classes, max_eval_batches=args.max_eval_batches)

        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_stats["loss"],
            "val_loss": val_metrics["loss"],
            "acc": val_metrics["accuracy"],
            "macro_f1": val_metrics["macro_f1"],
            "balanced_accuracy": val_metrics["balanced_accuracy"],
            "macro_precision": val_metrics["macro_precision"],
            "per_class_recall": val_metrics["per_class_recall"],
            "per_class_precision": val_metrics["per_class_precision"],
            "cgsil_dot": train_stats["dot"],
            "cgsil_cosine": train_stats["cosine"],
            "cgsil_beta": train_stats["beta"],
            "cgsil_active": train_stats["cgsil_active"],
            "cgsil_projection_applied": train_stats["surgery_applied"],
            "tail_classes_in_batch": train_stats["tail_classes_in_batch"],
            "head_classes_in_batch": train_stats["head_classes_in_batch"],
            "tail_samples_in_batch": train_stats["tail_samples_in_batch"],
        }
        logger.log(row)

        metric = val_metrics["balanced_accuracy"]
        if metric > best_balanced_accuracy:
            best_balanced_accuracy = metric
            save_checkpoint(best_checkpoint_path, model, optimizer, epoch + 1, val_metrics, args, info.class_counts, tail_classes)

        print(
            f"Epoch {epoch + 1:03d} | train_loss={train_stats['loss']:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"acc={val_metrics['accuracy']:.4f} | macro_f1={val_metrics['macro_f1']:.4f} | "
            f"balanced_acc={val_metrics['balanced_accuracy']:.4f}"
        )

    print(f"Best balanced accuracy: {best_balanced_accuracy:.4f}")
    print(f"Saved checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()

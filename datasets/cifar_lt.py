from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100


@dataclass
class CIFARLTInfo:
    num_classes: int
    class_counts: list[int]
    class_names: list[str]


def get_img_num_per_cls(num_classes: int, total_images: int, imbalance_type: str, imbalance_ratio: float) -> list[int]:
    if imbalance_ratio <= 0:
        raise ValueError("imbalance_ratio must be positive.")

    img_max = total_images / num_classes
    if imbalance_type == "exp":
        img_num_per_cls = [int(img_max * (imbalance_ratio ** (-class_idx / max(1, num_classes - 1)))) for class_idx in range(num_classes)]
    elif imbalance_type == "step":
        half = num_classes // 2
        img_num_per_cls = [int(img_max)] * half + [int(img_max / imbalance_ratio)] * (num_classes - half)
    else:
        img_num_per_cls = [int(img_max)] * num_classes
    return [max(1, count) for count in img_num_per_cls]


class CIFARLongTailDataset(Dataset):
    def __init__(self, data: np.ndarray, targets: list[int], transform: Callable | None = None):
        self.data = data
        self.targets = list(targets)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        image = Image.fromarray(self.data[index])
        target = self.targets[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def get_class_counts(self, num_classes: int) -> list[int]:
        counts = np.bincount(np.asarray(self.targets, dtype=np.int64), minlength=num_classes)
        return counts.astype(int).tolist()


_DATASET_FACTORY = {
    "cifar10": (CIFAR10, 10),
    "cifar100": (CIFAR100, 100),
}


def build_cifar_lt_datasets(
    dataset_name: str,
    root: str,
    imbalance_ratio: float,
    train_transform: Callable | None,
    test_transform: Callable | None,
    imbalance_type: str = "exp",
    download: bool = True,
    seed: int = 42,
):
    dataset_name = dataset_name.lower()
    if dataset_name not in _DATASET_FACTORY:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset_cls, num_classes = _DATASET_FACTORY[dataset_name]
    train_base = dataset_cls(root=root, train=True, download=download)
    test_base = dataset_cls(root=root, train=False, download=download)

    targets = np.asarray(train_base.targets, dtype=np.int64)
    img_num_per_cls = get_img_num_per_cls(num_classes, len(targets), imbalance_type, imbalance_ratio)

    rng = np.random.default_rng(seed)
    selected_indices = []
    for class_id in range(num_classes):
        class_indices = np.where(targets == class_id)[0]
        rng.shuffle(class_indices)
        selected_indices.extend(class_indices[: img_num_per_cls[class_id]].tolist())

    rng.shuffle(selected_indices)

    train_data = train_base.data[selected_indices]
    train_targets = targets[selected_indices].tolist()

    train_dataset = CIFARLongTailDataset(train_data, train_targets, transform=train_transform)
    test_dataset = CIFARLongTailDataset(test_base.data, list(test_base.targets), transform=test_transform)

    class_names = list(getattr(train_base, "classes", [str(index) for index in range(num_classes)]))
    info = CIFARLTInfo(num_classes=num_classes, class_counts=train_dataset.get_class_counts(num_classes), class_names=class_names)
    return train_dataset, test_dataset, info

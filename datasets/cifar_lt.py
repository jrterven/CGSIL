from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, SVHN


@dataclass
class LongTailDatasetInfo:
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


class LongTailDataset(Dataset):
    def __init__(self, data: np.ndarray, targets: list[int], transform: Callable | None = None):
        self.data = data
        self.targets = list(targets)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        image = self.data[index]
        if image.ndim == 3 and image.shape[0] in {1, 3}:
            image = np.transpose(image, (1, 2, 0))
        image = Image.fromarray(image)
        target = self.targets[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def get_class_counts(self, num_classes: int) -> list[int]:
        counts = np.bincount(np.asarray(self.targets, dtype=np.int64), minlength=num_classes)
        return counts.astype(int).tolist()


_NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "svhn": 10,
}


def load_base_dataset(dataset_name: str, root: str, train: bool, download: bool):
    dataset_name = dataset_name.lower()
    if dataset_name == "cifar10":
        dataset = CIFAR10(root=root, train=train, download=download)
        data = np.asarray(dataset.data)
        targets = np.asarray(dataset.targets, dtype=np.int64)
        class_names = list(dataset.classes)
        return data, targets, class_names
    if dataset_name == "cifar100":
        dataset = CIFAR100(root=root, train=train, download=download)
        data = np.asarray(dataset.data)
        targets = np.asarray(dataset.targets, dtype=np.int64)
        class_names = list(dataset.classes)
        return data, targets, class_names
    if dataset_name == "svhn":
        split = "train" if train else "test"
        dataset = SVHN(root=root, split=split, download=download)
        data = np.transpose(np.asarray(dataset.data), (0, 2, 3, 1))
        targets = np.asarray(dataset.labels, dtype=np.int64)
        class_names = [str(index) for index in range(10)]
        return data, targets, class_names
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_long_tail_datasets(
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
    if dataset_name not in _NUM_CLASSES:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    num_classes = _NUM_CLASSES[dataset_name]
    train_data_full, train_targets_full, class_names = load_base_dataset(dataset_name, root, train=True, download=download)
    test_data, test_targets, _ = load_base_dataset(dataset_name, root, train=False, download=download)

    targets = np.asarray(train_targets_full, dtype=np.int64)
    img_num_per_cls = get_img_num_per_cls(num_classes, len(targets), imbalance_type, imbalance_ratio)

    rng = np.random.default_rng(seed)
    selected_indices = []
    for class_id in range(num_classes):
        class_indices = np.where(targets == class_id)[0]
        rng.shuffle(class_indices)
        selected_indices.extend(class_indices[: img_num_per_cls[class_id]].tolist())

    rng.shuffle(selected_indices)

    train_data = train_data_full[selected_indices]
    train_targets = targets[selected_indices].tolist()

    train_dataset = LongTailDataset(train_data, train_targets, transform=train_transform)
    test_dataset = LongTailDataset(test_data, list(test_targets), transform=test_transform)

    info = LongTailDatasetInfo(num_classes=num_classes, class_counts=train_dataset.get_class_counts(num_classes), class_names=class_names)
    return train_dataset, test_dataset, info

from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class DatasetConfig:
    name: str = MISSING
    group: str = "dataset"
    classes: int = MISSING
    resolution: int = 224
    channels: int = 3
    patch_size: int = 16
    forget_fraction: float = 0.1
    forget_mode: str = "seed"  # seed or file
    cache_dir: str = "./cache"
    batch_size: int = 96
    batch_size_eval: int = 32
    height: int = "${dataset.resolution}"
    width: int = "${dataset.resolution}"
    forget_subset_mode: str = "random" # random, random_balanced (per class)
    data_path: str = "./datasets"


@dataclass
class CIFAR10(DatasetConfig):
    name: str = "cifar10"
    classes: int = 10

@dataclass
class CIFAR100(DatasetConfig):
    name: str = "cifar100"
    classes: int = 100
    forget_subset_mode: str = "random_balanced"

@dataclass
class PathMNIST(DatasetConfig):
    name: str = "pathmnist"
    classes: int = 9
    forget_subset_mode: str = "random_balanced"

@dataclass
class BloodMNIST(DatasetConfig):
    name: str = "bloodmnist"
    classes: int = 8
    forget_subset_mode: str = "random_balanced"

@dataclass
class DermaMNIST(DatasetConfig):
    name: str = "dermamnist"
    classes: int = 7
    forget_subset_mode: str = "random_balanced"

@dataclass
class TinyImageNet(DatasetConfig):
    name: str = "tinyimagenet"
    classes: int = 200
    forget_subset_mode: str = "random_balanced"

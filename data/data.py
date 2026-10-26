import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10, CIFAR100
from data.utils import Meta, MedMNISTWrapper, TinyImageNet
from medmnist import PathMNIST, BloodMNIST, DermaMNIST
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATASETS = {
    "cifar10": {
        "class": CIFAR10,
        "train_kwargs": {"train": True},
        "test_kwargs": {"train": False},
        "mean": IMAGENET_MEAN, # For consistency with the keys and pretrained models
        "std": IMAGENET_STD,
    },
    "cifar100": {
        "class": CIFAR100,
        "train_kwargs": {"train": True},
        "test_kwargs": {"train": False},
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
    "tinyimagenet": {
        "class": TinyImageNet,
        "train_kwargs": {"split": "train"},
        "test_kwargs":   {"split": "val"},
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
    "pathmnist": {
        "class": PathMNIST,
        "train_kwargs": {"split": "train", "size": 224},
        "val_kwargs":   {"split": "val",   "size": 224},
        "test_kwargs":  {"split": "test",  "size": 224},
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
    "bloodmnist": {
        "class": BloodMNIST,
        "train_kwargs": {"split": "train", "size": 224},
        "val_kwargs":   {"split": "val",   "size": 224},
        "test_kwargs":  {"split": "test",  "size": 224},
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
    "dermamnist": {
        "class": DermaMNIST,
        "train_kwargs": {"split": "train", "size": 224},
        "val_kwargs":   {"split": "val",   "size": 224},
        "test_kwargs":  {"split": "test",  "size": 224},
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },

}

def get_datasets(cfg, seed):
    return _get_datasets(cfg, seed)


def _get_datasets(cfg, seed):
    if cfg.name not in DATASETS:
        raise ValueError(f"Unsupported dataset '{cfg.name}'. Available: {list(DATASETS.keys())}")

    cfg_ds = DATASETS[cfg.name]
    DatasetClass = cfg_ds["class"]
    mean, std = cfg_ds["mean"], cfg_ds["std"]
    train_transform = get_transform(cfg, mean=mean, std=std, eval=False)
    test_transform = get_transform(cfg, mean=mean, std=std, eval=True)

    full_train = DatasetClass(
        root=cfg.data_path,
        transform=train_transform if not cfg.name.endswith("mnist") else test_transform,
        download=True,
        **cfg_ds["train_kwargs"],
    )

    # Deterministic preprocessing for key extraction and training-set evaluation.
    full_train_eval = DatasetClass(
        root=cfg.data_path,
        transform=test_transform,
        download=True,
        **cfg_ds["train_kwargs"],
    )

    test_set = DatasetClass(
        root=cfg.data_path,
        transform=test_transform,
        download=True,
        **cfg_ds["test_kwargs"],
    )
    if cfg.name.endswith("mnist"):
        val_set = DatasetClass(
            root=cfg.data_path,
            transform=test_transform,
            download=True,
            **cfg_ds["val_kwargs"],
        )
        if cfg.name.endswith("mnist"):
            full_train = MedMNISTWrapper(full_train)
            full_train_eval = MedMNISTWrapper(full_train_eval)
            val_set   = MedMNISTWrapper(val_set)
            test_set  = MedMNISTWrapper(test_set)
    else:
        test_set, val_set = random_split(test_set, [0.6, 0.4], generator=torch.Generator().manual_seed(seed))

    Ntrain = len(full_train)
    Nval   = len(val_set)
    return {
        "train": Meta(full_train, id_offset=0),
        "train_eval": Meta(full_train_eval, id_offset=0),
        "val":   Meta(val_set,   id_offset=Ntrain),
        "test":  Meta(test_set,  id_offset=Ntrain + Nval),
    }


def get_transform(cfg, mean, std, eval=False, normalize=True):
    if eval:
        resize = transforms.Resize(
            (cfg.height, cfg.width),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
    else:
        resize = transforms.RandomResizedCrop(
            (cfg.height, cfg.width),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
    transform_list = [resize]
    transform_list += [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
    if normalize:
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)

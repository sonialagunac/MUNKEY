from dataclasses import dataclass, field
from typing import Optional
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from .logging import LogConfig
from .data import DatasetConfig, CIFAR10, CIFAR100, PathMNIST, BloodMNIST, DermaMNIST, TinyImageNet
from .models import VitTiny, VitBase, ModelConfig
from .optimizer import AdamW, OptimizerConfig
from .modes import TokenConfig, RetrainConfig


@dataclass
class TrainConfigParam:
    num_workers: int = 4
    num_nodes: int = 1
    num_gpus: int = 1
    use_cpu: bool = False
    epochs: int = 100
    accum_batches: int = 1
    fp16: bool = True
    skip_sanity_checks: bool = True
    load_previous_checkpoint: bool = False
    resume_training_from: Optional[str] = None
    resume_eval_only: bool = False
    eval_every_n_epochs: int = 10
    enable_rmia: bool = False
    rmia_bank_seed: int = 42
    rmia_num_refs: int = 64
    rmia_ref_epochs: int = 100
    rmia_ref_learning_rate: float = 1.5e-4
    rmia_gamma: float = 1.0
    rmia_pairwise_chunk_size: int = 512


@dataclass
class TrainConfig:
    model: ModelConfig = MISSING
    mode: TokenConfig = MISSING
    dataset: DatasetConfig = MISSING
    training: TrainConfigParam = field(default_factory=TrainConfigParam)
    optimizer: OptimizerConfig = field(default_factory=AdamW)
    logging: LogConfig = field(default_factory=LogConfig)
    seed: int = 42
    debug: bool = False


def register_conf():
    OmegaConf.register_new_resolver(
        "compute_lr", lambda base_lr, batch_size: base_lr * (batch_size / 256), replace=True
    )
    cs = ConfigStore.instance()
    for node in (VitTiny, VitBase, CIFAR10, CIFAR100, PathMNIST, BloodMNIST,
                 DermaMNIST, TinyImageNet, AdamW, LogConfig,
                 TokenConfig, RetrainConfig):
        cs.store(group=node.group, name=node.name, node=node)
    cs.store(name="base", node=TrainConfig)

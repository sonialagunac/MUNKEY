from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class OptimizerConfig:
    name: str = MISSING
    group: str = "optimizer"
    learning_rate: float = MISSING


@dataclass
class AdamW(OptimizerConfig):
    name: str = "adamw"
    base_learning_rate: float = 1.5e-4
    betas: list = field(default_factory=lambda: [0.9, 0.99])
    weight_decay: float = 0.05
    learning_rate: float = (
        "${compute_lr:${model.optimizer.base_learning_rate},${dataset.batch_size}}"
    )

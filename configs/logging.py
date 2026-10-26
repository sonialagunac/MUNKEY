from dataclasses import dataclass
from typing import Optional


@dataclass
class LogConfig:
    name: str = "log"
    group: str = "logging"
    logdir: str = "./outputs"
    use_wandb: bool = False
    wandb_project_name: str = "MUNKEY"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_offline: bool = False
    keep_last_n_checkpoints: int = 1

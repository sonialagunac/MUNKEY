from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenConfig:
    name: str = "token"
    group: str = "mode"
    type: str = "mul_token"
    aggregation: str = "softmax"
    token_ensemble_topk: int = 16
    deep_prompt: bool = False
    cross_attn_layers: str = "set"
    use_key_as_token: bool = False
    use_token_ensemble: bool = False
    # Optional evaluation-only overrides; training settings stay unchanged.
    eval_use_token_ensemble: Optional[bool] = True
    eval_topk: Optional[int] = 4
    knn_train_prob: float = 0.2
    token_dropout_prob: float = 0.0
    image_dropout_prob: float = 0.0
    freeze_memory: bool = True


@dataclass
class RetrainConfig(TokenConfig):
    # Retrain the same memory architecture from scratch on retained examples.
    name: str = "scratchV"

from dataclasses import dataclass, field
from omegaconf import MISSING
from configs.optimizer import OptimizerConfig


@dataclass
class ModuleConfig:
    _target_: str = "transformers.ViTConfig"
    patch_size: int = "${dataset.patch_size}"
    image_size: int = "${dataset.resolution}"
    token_dim: int = 128
    num_hidden_layers: int = 12


@dataclass
class VitTinyConfig(ModuleConfig):
    hidden_size: int = 192
    num_attention_heads: int = 12
    intermediate_size: int = 768


@dataclass
class VitBaseConfig(ModuleConfig):
    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 3072


@dataclass
class ModelConfig:
    name: str = MISSING
    group: str = "model"
    models_path: str = MISSING # When working offline, make sure to have the models in ~/.cache/huggingface/hub
    config: ModuleConfig = MISSING
    optimizer: OptimizerConfig = "${optimizer}"


@dataclass
class VitTiny(ModelConfig):
    name: str = "vit-t"
    pretrained: bool = False
    models_path: str = "google/vit-base-patch16-224" # Actually using VIT base for key encoders
    config: VitTinyConfig = field(default_factory=VitTinyConfig)


@dataclass
class VitBase(ModelConfig):
    name: str = "vit-b"
    pretrained: bool = False
    models_path: str = "google/vit-base-patch16-224"
    config: VitBaseConfig = field(default_factory=VitBaseConfig)

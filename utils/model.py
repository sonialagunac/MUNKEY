import hashlib
import json
from pathlib import Path
from omegaconf import OmegaConf


def config_digest(value):
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]


def get_checkpoint_dir(cfg):
    # Exclude logging, resume, and evaluation-only options from training identity.
    mode = OmegaConf.to_container(cfg.mode, resolve=True)
    mode.pop("eval_use_token_ensemble", None)
    mode.pop("eval_topk", None)
    identity = {
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "mode": mode,
        "dataset": OmegaConf.to_container(cfg.dataset, resolve=True),
        "seed": cfg.seed,
        "epochs": cfg.training.epochs,
        "accum_batches": cfg.training.accum_batches,
        "fp16": cfg.training.fp16,
        "debug": cfg.debug,
    }
    name = f"{cfg.dataset.name}_{cfg.model.name}_{cfg.mode.name}_s{cfg.seed}_{config_digest(identity)}"
    return str(Path(cfg.logging.logdir) / "ckpts" / name)

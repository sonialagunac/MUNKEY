import os
from omegaconf import OmegaConf
import torch
from utils.model import config_digest


def get_rmia_merged_paths(cfg):
    identity = {
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "mode": OmegaConf.to_container(cfg.mode, resolve=True),
        "dataset": OmegaConf.to_container(cfg.dataset, resolve=True),
        "split_seed": cfg.seed,
        "bank_seed": cfg.training.rmia_bank_seed,
        "epochs": cfg.training.rmia_ref_epochs,
        "reference_lr": cfg.training.rmia_ref_learning_rate,
        "num_refs": cfg.training.rmia_num_refs,
        "debug": cfg.debug,
        "signal": "sm_taylor_softmax_t2_m0.6_n4",
    }
    base = os.path.join(cfg.dataset.cache_dir, "ref_models", cfg.dataset.name)
    stem = f"{cfg.model.name}_{cfg.mode.name}_{config_digest(identity)}"
    return (os.path.join(base, f"ref_likelihoods_{stem}.npy"),
            os.path.join(base, f"inclusion_mat_{stem}.npy"))


def _factorial(n: int) -> int:
    fact = 1
    for i in range(2, n + 1):
        fact *= i
    return fact


def _get_taylor(logit_signals: torch.Tensor, order: int) -> torch.Tensor:
    power = logit_signals
    taylor = power + 1.0
    for i in range(2, order):
        power = power * logit_signals
        taylor = taylor + (power / float(_factorial(i)))
    return taylor


def logits_to_rmia_signal(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    temp = 2.0
    margin = 0.6
    order = 4

    logit_signals = torch.div(logits, temp)
    taylor_logits = _get_taylor(logit_signals, order)
    taylor_logit_sum = taylor_logits.sum(dim=1, keepdim=True)
    true_logit = logit_signals.gather(1, labels.reshape(-1, 1))
    taylor_true_logit = taylor_logits.gather(1, labels.reshape(-1, 1))
    taylor_logit_sum = taylor_logit_sum - taylor_true_logit
    soft_taylor_true_logit = _get_taylor(true_logit - margin, order)
    taylor_logit_sum = taylor_logit_sum + soft_taylor_true_logit
    output_signals = torch.div(soft_taylor_true_logit, taylor_logit_sum)
    return output_signals.flatten()

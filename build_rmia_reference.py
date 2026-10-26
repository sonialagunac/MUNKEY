"""Train the optional RMIA reference bank using the same dataset/mode presets."""
from pathlib import Path
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader, Subset
from configs.config import TrainConfig, register_conf
from experiments.mul import MUL
from utils.rmia import get_rmia_merged_paths, logits_to_rmia_signal


def build_balanced_inclusion_train(n_train, num_refs, seed):
    if num_refs < 2 or num_refs % 2:
        raise ValueError("RMIA requires an even number of references, at least two")
    rng = np.random.default_rng(seed)
    inclusion = np.zeros((num_refs, n_train), dtype=bool)
    for idx in range(n_train):
        inclusion[rng.choice(num_refs, size=num_refs // 2, replace=False), idx] = True
    return inclusion


def make_loader(cfg, dataset, shuffle=False):
    return DataLoader(
        dataset, batch_size=cfg.dataset.batch_size if shuffle else cfg.dataset.batch_size_eval,
        shuffle=shuffle, num_workers=cfg.training.num_workers,
    )


def train_reference(model, loader, cfg, device):
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.training.rmia_ref_learning_rate,
        weight_decay=cfg.model.optimizer.weight_decay,
        betas=(0.9, 0.999),
    )
    epochs = cfg.training.rmia_ref_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(epochs):
        model.train()
        for batch_idx, batch in enumerate(loader):
            x, y, ids = (batch[k].to(device) for k in ("x", "y", "id"))
            optimizer.zero_grad()
            loss = F.cross_entropy(model.reference_forward(x, ids, batch_idx), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"Reference epoch {epoch + 1}/{epochs}: loss={loss.item():.4f}")


@torch.no_grad()
def compute_reference_signals(model, loaders, device, num_points):
    model.eval()
    signals = np.zeros(num_points, dtype=np.float32)
    for loader in loaders:
        for batch in loader:
            x, y, ids = (batch[k].to(device) for k in ("x", "y", "id"))
            logits = model.reference_forward(x, ids)
            signals[ids.cpu().numpy()] = logits_to_rmia_signal(logits, y).cpu().numpy()
    return signals


@hydra.main(version_base=None, config_path="configs", config_name="cifar10")
def main(cfg: TrainConfig):
    cfg.training.enable_rmia = False
    seed_everything(cfg.seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.training.use_cpu else "cpu")
    template = MUL(cfg)
    n_train = len(template.train_eval_dataset)
    n_total = n_train + len(template.validation_dataset) + len(template.test_dataset)
    refs = cfg.training.rmia_num_refs
    train_inclusion = build_balanced_inclusion_train(n_train, refs, cfg.training.rmia_bank_seed)
    inclusion = np.zeros((refs, n_total), dtype=bool)
    inclusion[:, :n_train] = train_inclusion
    signals = np.zeros((refs, n_total), dtype=np.float32)
    loaders = [make_loader(cfg, ds) for ds in
               (template.train_eval_dataset, template.validation_dataset, template.test_dataset)]
    del template
    for ref_idx in range(refs):
        seed_everything(cfg.training.rmia_bank_seed + ref_idx, workers=True)
        model = MUL(cfg).to(device)
        ids = np.flatnonzero(train_inclusion[ref_idx])
        if not len(ids):
            raise ValueError("Empty reference training subset; use a larger dataset")
        model.configure_rmia_reference_bank(torch.from_numpy(ids))
        loader = make_loader(cfg, Subset(model.full_train_dataset, ids.tolist()), shuffle=True)
        print(f"Training reference {ref_idx + 1}/{refs}")
        train_reference(model, loader, cfg, device)
        signals[ref_idx] = compute_reference_signals(model, loaders, device, n_total)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    ref_path, inc_path = get_rmia_merged_paths(cfg)
    Path(ref_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(ref_path, signals)
    np.save(inc_path, inclusion)
    print(f"Saved RMIA reference bank: {ref_path}")


if __name__ == "__main__":
    register_conf()
    main()

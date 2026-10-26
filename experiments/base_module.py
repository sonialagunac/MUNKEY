import random, torch
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import lightning.pytorch as pl
from hydra.utils import instantiate
from models.models import VIT
from data.data import get_datasets
from data.utils import Meta
import numpy as np
from torch.utils.data.dataloader import default_collate
from utils.forget import make_or_load_forget_ids
from sklearn.linear_model import LogisticRegression
import os
from utils.rmia import get_rmia_merged_paths

def seed_worker(worker_id):
    # Ensures each worker has a reproducible, unique seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def _empty_metrics_dict():
    # per-example metrics except acc (per-batch)
    return {
        "loss": [],
        "acc":  [],
        "entropy": [],
        "conf": [],
        "margin": [],
        "rmia": [],
        "ids": [],
        "rmia_signal": [],
    }

def to_numpy(x):
        # single tensor
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy().ravel()

        # list or tuple
        if isinstance(x, (list, tuple)):
            # list of tensors
            if len(x) > 0 and isinstance(x[0], torch.Tensor):
                x_cat = torch.cat([t.detach().float().cpu() for t in x], dim=0)
                return x_cat.numpy().ravel()
            # list of numbers
            return np.asarray(x, dtype=float).ravel()

        # already numpy or array like
        return np.asarray(x, dtype=float).ravel()

class BaseModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.loss_fn = F.cross_entropy
        self.model_type = self.cfg.model.name
        self.backbone = VIT(
            instantiate(self.cfg.model.config),
            models_path=self.cfg.model.models_path,
            pretrained=self.cfg.model.pretrained,
            cross_attn_layers=(self.cfg.mode.cross_attn_layers if self.cfg.mode.aggregation == "softmax_CA" else None),
        )
        self.repr_dim = self.cfg.model.config.hidden_size

        self.configure_data()
        self.classifier = nn.Linear(self.repr_dim, self.cfg.dataset.classes)
        self.num_instances = len(self.full_train_dataset)
        self.test_unlearn = False

        forget_size = (
            int(self.cfg.dataset.forget_fraction * self.num_instances)
            if not self.cfg.debug
            else int(self.num_instances / 2)
        )
        if self.cfg.dataset.forget_subset_mode == "random_balanced":
            train_labels = [
                self.full_train_dataset[i]["y"]
                for i in range(len(self.full_train_dataset))
            ]
        else:
            train_labels = None

        self.mask_forget_ids = make_or_load_forget_ids(
            train_len=self.num_instances,
            size=forget_size,
            mode=self.cfg.dataset.forget_mode,
            seed=self.cfg.seed,
            path=(
                f"{self.cfg.dataset.cache_dir}/forget_idxs/{self.cfg.dataset.name}/{self.cfg.seed}_{self.cfg.dataset.forget_fraction}_forget_ids.json"
                if self.cfg.dataset.cache_dir
                else None
            ),
            labels=train_labels
        )

        self.mask_retain_ids = make_or_load_forget_ids(
            train_len=self.num_instances,
            size=forget_size,
            mode=self.cfg.dataset.forget_mode,
            seed=self.cfg.seed,
            path=(
                f"{self.cfg.dataset.cache_dir}/forget_idxs/{self.cfg.dataset.name}/{self.cfg.seed}_{self.cfg.dataset.forget_fraction}_retain_ids.json"
                if self.cfg.dataset.cache_dir
                else None
            ),
            exclude=self.mask_forget_ids,
            labels=train_labels,
        )
        all_ids = set(range(len(self.full_train_dataset)))
        train_minus_forget_ids = sorted(all_ids - set(self.mask_forget_ids.tolist()))
        self.train_retain = Subset(
                self.train_eval_dataset,
                train_minus_forget_ids
            )
        self._configure_forget_retain_datasets()
        self._eval_buf = {
            "test": {"base": self._empty_split_dict("test")},
            "val":  {"base": self._empty_split_dict("val")},
        }
        self._suffix = ""
        self.rmia_enabled = False

        if getattr(self.cfg.training, "enable_rmia", False):
            self._load_rmia_tables()


    def _load_rmia_tables(self):
        ref_path, inc_path = get_rmia_merged_paths(self.cfg)
        if not (os.path.exists(ref_path) and os.path.exists(inc_path)):
            raise FileNotFoundError(
                "RMIA reference tables not found. Run build_rmia_reference.py with the same config first. "
                f"Expected: {ref_path} and {inc_path}"
            )
        ref_np = np.load(ref_path, mmap_mode="r")   # (R, N) float32
        inc_np = np.load(inc_path, mmap_mode="r")   # (R, N) bool
        expected = (self.cfg.training.rmia_num_refs,
                    self.num_instances + len(self.validation_dataset) + len(self.test_dataset))
        if ref_np.shape != expected or inc_np.shape != expected:
            raise ValueError(f"RMIA tables must have shape {expected}; rebuild for this config")

        # keep on CPU
        self.rmia_ref = torch.from_numpy(np.array(ref_np, copy=True)).contiguous()
        self.rmia_in  = torch.from_numpy(np.array(inc_np, copy=True)).contiguous()
        self.rmia_enabled = True
        print(
            f"[RMIA] Loaded {self.rmia_ref.shape[0]} references over "
            f"{self.rmia_ref.shape[1]} points from {ref_path}"
        )

    def _populate_rmia_buffers(self, buf):
        if not self.rmia_enabled:
            return
        for split_buf in buf.values():
            split_buf["rmia"] = []

        bg_name = self._current_metric_prefix
        if bg_name not in buf or len(buf[bg_name]["ids"]) == 0 or len(buf[bg_name]["rmia_signal"]) == 0:
            return

        background_ids = torch.cat(buf[bg_name]["ids"], dim=0).long()
        background_signals = torch.cat(buf[bg_name]["rmia_signal"], dim=0).float()

        for split_name in ("forget", "retain", bg_name):
            if split_name not in buf:
                continue
            split_buf = buf[split_name]
            if len(split_buf["ids"]) == 0 or len(split_buf["rmia_signal"]) == 0:
                continue
            target_ids = torch.cat(split_buf["ids"], dim=0).long()
            target_signals = torch.cat(split_buf["rmia_signal"], dim=0).float()
            split_buf["rmia"].append(
                self._compute_rmia_scores_against_background(
                    target_ids=target_ids,
                    target_signals=target_signals,
                    background_ids=background_ids,
                    background_signals=background_signals,
                    exclude_self=(split_name == bg_name),
                )
            )

    def _compute_rmia_scores_against_background(
        self,
        target_ids,
        target_signals,
        background_ids,
        background_signals,
        exclude_self=False,
    ):
        if not self.rmia_enabled or len(target_ids) == 0 or len(background_ids) == 0:
            return torch.empty(0, dtype=torch.float32)

        target_ids = target_ids.long()
        target_signals = target_signals.float()
        background_ids = background_ids.long()
        background_signals = background_signals.float()

        background_ids_np = background_ids.cpu().numpy()
        background_ref_signals = torch.from_numpy(
            np.asarray(self.rmia_ref[:, background_ids_np], dtype=np.float32)
        )
        eps = 1e-12
        gamma = float(getattr(self.cfg.training, "rmia_gamma", 2.0))
        scores = torch.zeros(len(target_ids), dtype=torch.float32)
        chunk_size = int(getattr(self.cfg.training, "rmia_pairwise_chunk_size", 512))

        for start in range(0, len(target_ids), chunk_size):
            end = min(start + chunk_size, len(target_ids))
            chunk_ids = target_ids[start:end]
            chunk_signals = target_signals[start:end]
            chunk_ids_np = chunk_ids.cpu().numpy()
            chunk_ref_signals = torch.from_numpy(
                np.asarray(self.rmia_ref[:, chunk_ids_np], dtype=np.float32)
            )
            chunk_inc = torch.from_numpy(
                np.asarray(self.rmia_in[:, chunk_ids_np], dtype=np.bool_)
            ).bool()

            for local_idx in range(len(chunk_ids)):
                ref_mask = ~chunk_inc[:, local_idx]
                if not ref_mask.any():
                    continue
                x_prior = chunk_ref_signals[ref_mask, local_idx].mean().clamp_min(eps)
                z_priors = background_ref_signals[ref_mask].mean(dim=0).clamp_min(eps)
                ratio_x = chunk_signals[local_idx].clamp_min(eps) / x_prior
                ratio_z = background_signals.clamp_min(eps) / z_priors
                cmp = (ratio_x / ratio_z) > gamma
                if exclude_self:
                    cmp = cmp.clone()
                    cmp[background_ids == chunk_ids[local_idx]] = False
                denom = int(cmp.numel()) - (1 if exclude_self else 0)
                scores[start + local_idx] = cmp.float().sum() / float(max(denom, 1))
        return scores

    def _buf_to_numpy(self, buf_split, keys=("loss", "entropy", "conf", "margin")):
        """
        buf_split: e.g. self._eval_buf["test"]["base"]["forget"]
        returns X: [N, len(keys)]
        """
        feat_arrays = []
        for k in keys:
            if k not in buf_split or len(buf_split[k]) == 0:
                raise ValueError(f"No buffered values for metric {k}")
            vals = torch.cat(buf_split[k], dim=0)          # [N]
            feat_arrays.append(to_numpy(vals))             # [N]
        X = np.stack(feat_arrays, axis=1)                  # [N, D]
        return X

    def _logreg_mia_auc_shadow(self, base_buf, prefix="", keys=("loss", "entropy", "conf", "margin")):
        """
        Train LR attack on retain vs test, evaluate on forget vs test.
        Returns: auc_forget_vs_test (float)
        """
        X_r = self._buf_to_numpy(base_buf["retain"], keys=keys)  # members used for attack training
        X_t = self._buf_to_numpy(base_buf[prefix], keys=keys)    # non members for attack training
        X_f = self._buf_to_numpy(base_buf["forget"], keys=keys)  # forgotten members (eval only)

        # Train set: retain vs test
        X_train = np.concatenate([X_r, X_t], axis=0)
        y_train = np.concatenate(
            [np.ones(len(X_r), dtype=int), np.zeros(len(X_t), dtype=int)], axis=0
        )

        # Eval set: forget vs test
        X_eval = np.concatenate([X_f, X_t], axis=0)
        y_eval = np.concatenate(
            [np.ones(len(X_f), dtype=int), np.zeros(len(X_t), dtype=int)], axis=0
        )

        attack = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=self.cfg.seed,
        )
        attack.fit(X_train, y_train)
        scores = attack.decision_function(X_eval)
        auc = roc_auc_score(y_eval, scores)
        return float(auc)


    def _empty_split_dict(self, split_name):
        # split_name = "test" or "val"
        return {
            split_name: _empty_metrics_dict(),
            "forget":  _empty_metrics_dict(),
            "retain":  _empty_metrics_dict(),
            "retain_full":  _empty_metrics_dict(),
        }


    def configure_data(self):
        datasets = get_datasets(self.cfg.dataset, self.cfg.seed)

        if self.cfg.debug:
            # Keep IDs contiguous when using the small development subset.
            n_train = min(100, len(datasets["train"]))
            n_val = min(100, len(datasets["val"]))
            for name, offset in (("train", 0), ("train_eval", 0),
                                 ("val", n_train), ("test", n_train + n_val)):
                ds = datasets[name]
                datasets[name] = Meta(Subset(ds, list(range(min(100, len(ds))))), id_offset=offset)
        self.full_train_dataset = datasets["train"]
        self.train_eval_dataset = datasets["train_eval"]
        self.validation_dataset = datasets["val"]
        self.test_dataset = datasets["test"]

    def _configure_forget_retain_datasets(self):
        self.forget_dataset = Subset(self.train_eval_dataset, self.mask_forget_ids)
        self.retain_dataset = Subset(self.train_eval_dataset, self.mask_retain_ids)

    def configure_optimizers(self):
        optim_cfg = self.cfg.model.optimizer
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        T_max = self.trainer.max_epochs
        lr = optim_cfg.learning_rate

        optimizer = optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=optim_cfg.weight_decay,
            betas=optim_cfg.betas,
        )

        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=T_max
        )
        return [optimizer], [lr_scheduler]

    def forward(self, x):
        features = self.extract_features(x)
        return self.classifier(features)

    def extract_features(self, x, token_embed=None, drop_image=False):
        return self.backbone(
            x,
            token_embed=token_embed,
            drop_image=drop_image,
            deep_prompt=self.cfg.mode.deep_prompt,
        )

    def _make_loader(
        self,
        dataset,
        batch_size,
        shuffle,
        drop_last,
        persistent_workers=False,
    ):
        """Internal helper to build a DataLoader with shared settings."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=default_collate,
            num_workers=self.cfg.training.num_workers,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=persistent_workers and self.cfg.training.num_workers > 0,
            drop_last=drop_last,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.cfg.seed),
        )

    def train_dataloader(self, drop_last=False):
        # persistent_workers=True only for training
        return self._make_loader(
            dataset=self.train_dataset,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=True,
            drop_last=drop_last,
            persistent_workers=True,
        )

    def _make_eval_loaders(self, main_dataset, shuffle: bool, test = False):
        dl_main = self._make_loader(
            dataset=main_dataset,
            batch_size=self.cfg.dataset.batch_size_eval,
            shuffle=shuffle,
            drop_last=False,
            persistent_workers=False,
        )

        dl_forget = self._make_loader(
            dataset=self.forget_dataset,
            batch_size=self.cfg.dataset.batch_size_eval,
            shuffle=shuffle,
            drop_last=False,
            persistent_workers=False,
        )

        dl_retain = self._make_loader(
            dataset=self.retain_dataset,
            batch_size=self.cfg.dataset.batch_size_eval,
            shuffle=shuffle,
            drop_last=False,
            persistent_workers=False,
        )

        if test:
            dl_retain_full = self._make_loader(
                dataset=self.train_retain,
                batch_size=self.cfg.dataset.batch_size_eval,
                shuffle=shuffle,
                drop_last=False,
                persistent_workers=False,
            )

            return [dl_main, dl_forget, dl_retain, dl_retain_full]
        else:
            return [dl_main, dl_forget, dl_retain]

    def val_dataloader(self, shuffle=False):
        return self._make_eval_loaders(self.validation_dataset, shuffle)

    def test_dataloader(self, shuffle=False):
        return self._make_eval_loaders(self.test_dataset, shuffle, test = True)

    # Metrics
    @torch.no_grad()
    def compute_accuracy(self, logits, targets):
        preds = torch.argmax(logits, dim=-1)
        return (preds == targets).float().mean()

    def simple_mia(self, member_scores, nonmember_scores, larger_scores_more_member=True):
        """
        member_scores: 1D np array for 'forget' (or 'retain')
        nonmember_scores: 1D np array for 'test' (reference non members)
        """
        member_scores = to_numpy(member_scores)
        nonmember_scores = to_numpy(nonmember_scores)

        # y: 1 = member (forget/retain), 0 = non-member (test)
        y = np.r_[
            np.ones(len(member_scores), dtype=int),
            np.zeros(len(nonmember_scores), dtype=int),
        ]
        scores = np.r_[member_scores, nonmember_scores]

        if not larger_scores_more_member:
            scores = -scores

        return roc_auc_score(y, scores)

    def brier_score_multiclass(self, logits, targets):
        """
        logits: [B, C]
        targets: [B] int class labels
        returns scalar Brier score (mean over batch)
        """
        probs = logits.softmax(dim=-1)                       # [B, C]
        y_onehot = F.one_hot(targets, num_classes=self.cfg.dataset.classes).float()
        # sum over classes, mean over batch
        brier = ((probs - y_onehot) ** 2).sum(dim=-1).mean()
        return brier

    def on_test_epoch_start(self):
        self._phase = "unlearn" if self.test_unlearn else "base"
        self._eval_buf["test"][self._phase] = self._empty_split_dict("test")
    def on_test_epoch_end(self):
        self.on_global_epoch_end(prefix = "test")

    def on_global_epoch_end(self, prefix = ""):
        self._current_metric_prefix = prefix
        metric_cfg = {
            "loss":    False,
            "entropy": False,
            "conf":    True,
            "margin":  True,
        }
        if self.rmia_enabled:
            metric_cfg["rmia"] = True

        base_buf = self._eval_buf[prefix]["base"]
        if self.rmia_enabled:
            self._populate_rmia_buffers(base_buf)
        for metric_name, larger_is_member in metric_cfg.items():
            if (
                len(base_buf["forget"][metric_name]) == 0
                or len(base_buf[prefix][metric_name]) == 0
                or len(base_buf["retain"][metric_name]) == 0
            ):
                continue
            f_vals = torch.cat(base_buf["forget"][metric_name], dim=0)
            t_vals = torch.cat(base_buf[prefix][metric_name], dim=0)
            r_vals = torch.cat(base_buf["retain"][metric_name], dim=0)

            if prefix == "val" or self._phase == "base":
                suffix = "" if prefix == "test" else "_base"
                mia_forget = self.simple_mia(f_vals, t_vals, larger_scores_more_member=larger_is_member)
                mia_retain = self.simple_mia(r_vals, t_vals, larger_scores_more_member=larger_is_member)
                suffix = "" if prefix == "test" else "_base"
                self.log(
                    f"{prefix}_forget/mia_{metric_name}{suffix}",
                    mia_forget,
                    add_dataloader_idx=False,
                )
                self.log(
                    f"{prefix}_retain/mia_{metric_name}{suffix}",
                    mia_retain,
                    add_dataloader_idx=False,
                )

        if prefix == "val" or self._phase == "base":
            mia_multi_shadow_base = self._logreg_mia_auc_shadow(base_buf, prefix)
            self.log(
                f"{prefix}_forget/mia_multi_lr{suffix}",
                mia_multi_shadow_base,
                add_dataloader_idx=False,
            )
        if prefix == "test" and self._phase != "unlearn":
            self.log("test_step", 0, add_dataloader_idx=False)

        if self._phase == "unlearn":
            un_buf = self._eval_buf[prefix]["unlearn"]
            if self.rmia_enabled:
                self._populate_rmia_buffers(un_buf)

            # tow still based on acc (unchanged)
            f_acc = torch.stack(base_buf["forget"]["acc"])
            t_acc = torch.stack(base_buf[prefix]["acc"])
            r_acc = torch.stack(base_buf["retain"]["acc"])
            un_f_acc = torch.stack(un_buf["forget"]["acc"])
            un_t_acc = torch.stack(un_buf[prefix]["acc"])
            un_r_acc = torch.stack(un_buf["retain"]["acc"])
            forget_tow = (un_f_acc - f_acc).mean()
            test_tow   = (un_t_acc - t_acc).mean()
            retain_tow   = (un_r_acc - r_acc).mean()
            self.log(f"{prefix}_forget/tow", forget_tow, add_dataloader_idx=False)
            self.log(f"{prefix}_retain/tow", retain_tow, add_dataloader_idx=False)
            self.log(f"{prefix}_{prefix}/tow",   test_tow,   add_dataloader_idx=False)
            suffix = "" if prefix == "test" else "_unlearn"
            # now per-metric MIA_unlearn
            for metric_name, larger_is_member in metric_cfg.items():
                if (
                    len(un_buf["forget"][metric_name]) == 0
                    or len(un_buf[prefix][metric_name]) == 0
                    or len(un_buf["retain"][metric_name]) == 0
                ):
                    continue
                un_f_vals = torch.cat(un_buf["forget"][metric_name], dim=0)
                un_t_vals = torch.cat(un_buf[prefix][metric_name], dim=0)
                un_r_vals = torch.cat(un_buf["retain"][metric_name], dim=0)

                # MIA after unlearning vs test after unlearning
                mia_forget_un = self.simple_mia(un_f_vals, un_t_vals, larger_scores_more_member=larger_is_member)
                mia_retain_un = self.simple_mia(un_r_vals, un_t_vals, larger_scores_more_member=larger_is_member)
                self.log(
                    f"{prefix}_forget/mia_{metric_name}{suffix}",
                    mia_forget_un,
                    add_dataloader_idx=False,
                )
                self.log(
                    f"{prefix}_retain/mia_{metric_name}{suffix}",
                    mia_retain_un,
                    add_dataloader_idx=False,
                )

            mia_multi_shadow = self._logreg_mia_auc_shadow(un_buf, prefix)
            mia_multi_shadow_base = self._logreg_mia_auc_shadow(base_buf, prefix)
            self.log(
                f"{prefix}_forget/mia_multi_lr{suffix}",
                mia_multi_shadow,
                add_dataloader_idx=False,
            )
            if "mia_multi_shadow_base" in locals():
                # Logging difference in mia to evaluate different token MUL methods
                self.log(
                        f"{prefix}_forget/delta_mia_multi_lr{suffix}",
                        mia_multi_shadow - mia_multi_shadow_base,
                        add_dataloader_idx=False,
                )
            if prefix == "test":
                r_acc_full = torch.stack(base_buf["retain_full"]["acc"])
                un_r_acc_full = torch.stack(un_buf["retain_full"]["acc"])
                retain_full_tow   = (un_r_acc_full - r_acc_full).mean()
                self.log(f"{prefix}_retain_full/tow", retain_full_tow, add_dataloader_idx=False)
                self.log("test_step", 1, add_dataloader_idx=False)

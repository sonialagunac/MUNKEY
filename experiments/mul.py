import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rmia import logits_to_rmia_signal
from torch.utils.data import DataLoader, Subset
from hydra.utils import instantiate
from models.models import TokenMemory
from experiments.base_module import BaseModule
from utils.encoders import ViTEncoder


class MUL(BaseModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.retrain = self.cfg.mode.name == "scratchV"
        self.train_dataset = (
            Subset(self.full_train_dataset, self.train_retain.indices)
            if self.retrain else self.full_train_dataset
        )

        self.token_dim = self.cfg.model.config.token_dim
        self.knn_train_prob = self.cfg.mode.knn_train_prob
        self.setup_key_encoder()
        self.keys_cache_path = self._build_keys_cache_path() # Load pre-computed keys
        self.token_memory = TokenMemory(
                self.num_instances, self.token_dim, self.repr_dim
            )
        if self.cfg.mode.use_key_as_token:
            self.proj = nn.Linear(self.key_hidden_size, self.repr_dim)

        if self.cfg.mode.aggregation not in ("softmax", "rank", "softmax_CA"):
            raise ValueError("mode.aggregation must be softmax, rank, or softmax_CA")
        if self.cfg.mode.aggregation == "softmax_CA" and (self.cfg.mode.use_token_ensemble or self.cfg.mode.eval_use_token_ensemble):
            raise ValueError("Cross-attention cannot be combined with token ensembling")
        if self.cfg.mode.use_key_as_token and (self.cfg.mode.use_token_ensemble or self.cfg.mode.eval_use_token_ensemble):
            raise ValueError("Direct keys cannot be combined with token ensembling")
        if self.cfg.mode.use_key_as_token and self.knn_train_prob:
            raise ValueError("Direct-key mode requires knn_train_prob=0")
        if not 0 <= self.cfg.mode.token_dropout_prob <= 1 or not 0 <= self.cfg.mode.image_dropout_prob <= 1:
            raise ValueError("Dropout probabilities must be in [0, 1]")
        if self.cfg.mode.token_dropout_prob + self.cfg.mode.image_dropout_prob > 1:
            raise ValueError("Image and token dropout probabilities must sum to at most 1")

    @property
    def use_token_ensemble(self):
        override = self.cfg.mode.eval_use_token_ensemble
        return self.cfg.mode.use_token_ensemble if self.training or override is None else override

    @property
    def retrieval_topk(self):
        override = self.cfg.mode.eval_topk
        return self.cfg.mode.token_ensemble_topk if self.training or override is None else override

    def _build_keys_cache_path(self) -> str:
        from utils.model import config_digest
        from pathlib import Path
        cache = Path(self.cfg.dataset.cache_dir) / "keys"
        cache.mkdir(parents=True, exist_ok=True)
        identity = {
            "dataset": self.cfg.dataset.name,
            "root": str(Path(self.cfg.dataset.data_path).resolve()),
            "resolution": self.cfg.dataset.resolution,
            "encoder": self.cfg.model.models_path,
            "instances": self.num_instances,
        }
        return str(cache / f"{self.cfg.dataset.name}_{config_digest(identity)}.pt")

    def get_seeded_p(self, batch_idx, offset=0, prob = True, device="cpu"):
        g = torch.Generator(device=device)
        # Adding 'offset' allows you to get different random numbers
        # for different decisions within the same batch
        g.manual_seed(self.cfg.seed + self.current_epoch * 10_000 + batch_idx + offset)
        if prob:
            return torch.rand((), generator=g).item()
        else:
            return g


    def _ensure_keys_ready(self):
        if hasattr(self, "keys") and self.keys is not None:
            return
        if os.path.exists(self.keys_cache_path):
            print(f"[MUL] Loading keys from cache: {self.keys_cache_path}")
            self.keys = torch.load(self.keys_cache_path, map_location="cpu", weights_only=True)
            if self.keys.shape != (self.num_instances, self.key_hidden_size):
                raise ValueError("Cached keys have an incompatible shape; remove this key cache and retry")
        else:
            print(f"[MUL] Computing keys and saving to: {self.keys_cache_path}")
            eval_loader = self._make_loader(
                dataset=self.train_eval_dataset,
                batch_size=self.cfg.dataset.batch_size_eval,
                shuffle=False,
                drop_last=False,
                persistent_workers=False,
            )
            self.keys = self.compute_keys(eval_loader)
            torch.save(self.keys.cpu(), self.keys_cache_path)
        self.keys_gpu_cache = None

    def configure_rmia_reference_bank(self, included_ids):
        self.rmia_reference_ids = included_ids.detach().cpu().long()
        self._ensure_keys_ready()

    def _extract_key_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.key_encoder(x)

    def reference_forward(self, x, ids, batch_idx=0):
        if not hasattr(self, "rmia_reference_ids"):
            raise RuntimeError("Configure the RMIA reference bank first")
        return self._forward_logits(x, ids, batch_idx, reference=True)

    def on_fit_start(self):
        self._ensure_keys_ready()

    def on_load_checkpoint(self, checkpoint):
        saved_cfg = checkpoint.get("hyper_parameters", {}).get("cfg", {})
        saved_mode = saved_cfg.get("mode", {}).get("name")
        if saved_mode != self.cfg.mode.name:
            raise ValueError(
                "Resume with the checkpoint's original mode. Retrain (M) must start "
                "fresh or resume its own scratchV checkpoint."
            )

    def on_test_start(self):
        self._ensure_keys_ready()

    def on_validation_start(self):
        self._ensure_keys_ready()

    def train(self, mode=True):
        super().train(mode)
        # A frozen encoder must also keep its inference behaviour during training.
        if hasattr(self, "key_encoder"):
            self.key_encoder.eval()
        return self

    def on_validation_epoch_start(self):
        for phase in ("unlearn", "base"):
            self._eval_buf["val"][phase] = self._empty_split_dict("val")

    def on_validation_epoch_end(self):
        self.on_global_epoch_end(prefix = "val")

    def _forward_logits(self, x, ids, batch_idx=0, reference=False):
        weights = None
        if self.training and not reference:
            keys = self.keys.to(x.device) if self.keys_gpu_cache is None else self.keys_gpu_cache.to(x.device)
            self.keys_gpu_cache = keys
            kq = keys[ids]
        else:
            kq = F.normalize(self._extract_key_features(x), dim=-1)
        if self.cfg.mode.use_key_as_token:
            token_embed = self.proj(F.normalize(kq, dim=-1)).unsqueeze(1)
        elif reference or not self.training or self.get_seeded_p(batch_idx) < self.knn_train_prob:
            excluded_ids = None
            if not reference:
                if self.training:
                    # Preserve batch exclusion; retraining must also exclude all
                    # forgotten entries, which were never part of its training set.
                    excluded_ids = ids
                    if self.retrain:
                        excluded_ids = torch.cat((ids, self.mask_forget_ids.to(ids.device)))
                elif self.retrain or self.test_unlearn:
                    excluded_ids = self.mask_forget_ids
            raw, weights = self.knn_mix_token(
                kq, topk=self.retrieval_topk,
                mask_forget_ids=excluded_ids,
                candidate_ids=self.rmia_reference_ids if reference else None,
                mask_self_ids=ids if reference and self.training else None,
                batch_idx=0 if reference else batch_idx,
            )
            if self.use_token_ensemble:
                raw = raw.reshape(-1, raw.shape[-1])
                token_embed = self.token_memory.adapter(raw).unsqueeze(1)
            else:
                weights = None
                token_embed = self.token_memory.adapter(raw).unsqueeze(1)
        else:
            token_embed = self.token_memory.get_tokens(ids)
        drop_image = False
        if self.training:
            p = random.random() if reference else self.get_seeded_p(batch_idx, offset=1)
            if p < self.cfg.mode.token_dropout_prob:
                token_embed = self.token_memory.get_null_tokens(x.size(0))
                weights = None
            elif p < self.cfg.mode.token_dropout_prob + self.cfg.mode.image_dropout_prob:
                drop_image = True
        logits = self.classifier(self.extract_features(x, token_embed=token_embed, drop_image=drop_image))
        if weights is not None:
            logits = logits.view(weights.shape[0], weights.shape[1], -1)
            logits = (weights.unsqueeze(-1) * logits).sum(dim=1)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self._forward_logits(batch["x"], batch["id"], batch_idx)
        loss = F.cross_entropy(logits, batch["y"])
        self.log("train/accuracy", self.compute_accuracy(logits, batch["y"]))
        self.log("train/loss", loss)
        return loss

    def evaluation_step(self, batch,  dataloader_idx=0):
        if self.trainer.testing:
            part = "test"
            self._phase = "unlearn" if self.test_unlearn else "base"
            name = self._phase
            self._suffix = ["test", "forget", "retain", "retain_full"][dataloader_idx]
        elif self.trainer.validating:
            part = "val"
            self._suffix = ["val", "forget", "retain"][dataloader_idx]
            self._phase = "unlearn" if self.test_unlearn else "base"
            name = self._phase

        x, y, ids = batch["x"], batch["y"], batch["id"]
        logits = self._forward_logits(x, ids)

        # --- per-example loss ---
        loss_vec = F.cross_entropy(logits, y, reduction="none")   # [B]
        loss = loss_vec.mean()

        # --- other scores from logits ---
        probs = logits.softmax(dim=-1)                            # [B, C]

        rmia_signal = logits_to_rmia_signal(logits, y) if self.rmia_enabled else None

        # entropy: lower -> more member-like
        entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1)  # [B]

        # confidence: max softmax prob, higher -> more member-like
        conf, _ = probs.max(dim=-1)                               # [B]

        # logit margin: top1 - top2, higher -> more member-like
        top2 = logits.topk(k=2, dim=-1).values                    # [B, 2]
        margin = top2[:, 0] - top2[:, 1]                          # [B]

        acc = self.compute_accuracy(logits, y)
        brier = self.brier_score_multiclass(logits, y)

        if self.trainer.testing or self.trainer.validating:
            self.log(
                f"{part}_{self._suffix}/{name}/loss",
                loss, add_dataloader_idx=False,
            )
            self.log(
                f"{part}_{self._suffix}/{name}/acc",
                acc, add_dataloader_idx=False,
            )
            self.log(
                f"{part}_{self._suffix}/{name}/brier",
                brier, add_dataloader_idx=False,
            )

            buf = self._eval_buf[part][self._phase][self._suffix]
            buf["loss"].append(loss_vec.detach().cpu())
            buf["entropy"].append(entropy.detach().cpu())
            buf["conf"].append(conf.detach().cpu())
            buf["margin"].append(margin.detach().cpu())
            buf["acc"].append(acc.detach().cpu())
            buf["ids"].append(ids.detach().cpu().long())
            if rmia_signal is not None:
                buf["rmia_signal"].append(rmia_signal.detach().cpu())

        return loss


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.test_unlearn = False
        self.evaluation_step(batch, dataloader_idx)
        if not self.retrain:
            self.test_unlearn = True
            self.evaluation_step(batch, dataloader_idx)
        return None

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.evaluation_step(batch, dataloader_idx)
        return None

    def setup_key_encoder(self) -> None:
        self.key_encoder = ViTEncoder(self.cfg.model.models_path, instantiate(self.cfg.model.config)).eval()
        for p in self.key_encoder.parameters():
            p.requires_grad_(False)
        self.key_hidden_size = self.key_encoder.hidden_size

    @torch.no_grad()
    def compute_keys(self, dataloader: DataLoader) -> torch.Tensor:
        """
        Compute and cache normalized feature keys for all instances.

        Args:
            dataloader: DataLoader containing all training samples

        Returns:
            keys: [num_instances, key_hidden_size] normalized feature keys
        """
        self.key_encoder.eval()
        keys = torch.zeros(
            self.num_instances, self.key_hidden_size
        )
        device = self.device

        for batch in dataloader:
            x = batch["x"].to(device)
            ids = batch["id"].long()

            features = self._extract_key_features(x)
            keys[ids] = F.normalize(features, dim=-1).cpu()
        return keys


    def knn_mix_token(
        self,
        query_feats: torch.Tensor,
        topk=16,
        tau=0.07,
        alpha=2.0,
        mask_forget_ids=None,
        batch_idx=0,
        candidate_ids=None,
        mask_self_ids=None,
    ):
        """
        query_feats: [B, D] L2-normalized key features from key_encoder
        returns: [B, d_t] raw token mixture (before adapter)
        """
        # Retrieval indices and weights do not require gradients.
        with torch.no_grad():
            # keys: [N, D] - use GPU cache if available
            if self.keys_gpu_cache is None or self.keys_gpu_cache.device != query_feats.device:
                self.keys_gpu_cache = self.keys.to(query_feats.device)
            if candidate_ids is not None:
                candidate_ids = candidate_ids.to(query_feats.device).long()
                K = self.keys_gpu_cache[candidate_ids]
            else:
                K = self.keys_gpu_cache
            sims = query_feats @ K.T  # [B, N] cosine if both L2-normalized
            if mask_forget_ids is not None and len(mask_forget_ids) > 0:
                if candidate_ids is None:
                    sims[:, mask_forget_ids] = float("-inf")  # hard exclude forgotten
                else:
                    assert not (candidate_ids is not None and mask_forget_ids is not None), (
            "knn_mix_token does not support candidate_ids and mask_forget_ids together.")
            if mask_self_ids is not None:
                ref_ids = candidate_ids
                same = ref_ids.unsqueeze(0) == mask_self_ids.unsqueeze(1)
                sims = sims.masked_fill(same, float("-inf"))

            available = int(torch.isfinite(sims).sum(dim=1).min().item())
            if available == 0:
                raise ValueError("No retrieval candidates remain after excluding forgotten/batch keys")
            topk = min(topk, available)
            if topk < 1:
                raise ValueError("Retrieval top-k must be positive")
            if self.training and topk > 4:
                # Variable K for regularization
                g_k = self.get_seeded_p(batch_idx, offset=3, prob=False)
                topk = torch.randint(4, topk + 1, (1,), generator=g_k).item()
            topv, local_topi = sims.topk(topk, dim=-1)  # [B, K]
            topi = candidate_ids[local_topi] if candidate_ids is not None else local_topi

        # gather tokens
        raw_tokens = self.token_memory.table.weight  # [N, d_t]
        neigh_tok = raw_tokens[topi]  # [B, K, d_t]

        if self.training:
            # Dont backprop through memory tokens in k-NN retrieval to avoid leaking
            if self.cfg.mode.freeze_memory:
                neigh_tok = neigh_tok.detach()
            if neigh_tok.shape[1] > 1:
                # Drop one of the closest three neighbors for regularization
                g_drop = self.get_seeded_p(batch_idx, offset=4, prob=False)
                drop_idx = torch.randint(0, min(3, topk), (1,), generator=g_drop).item()
                neigh_tok = torch.cat(
                    [neigh_tok[:, :drop_idx, :], neigh_tok[:, drop_idx + 1:, :]], dim=1
                )
                topv = torch.cat([topv[:, :drop_idx], topv[:, drop_idx + 1:]], dim=1)
        # Aggregation methods
        if self.cfg.mode.aggregation.startswith("softmax"):
            # Softmax based ranking
            w = F.softmax(topv / tau, dim=-1)  # [B, K]
        elif self.cfg.mode.aggregation == "rank":
            # RankVoting
            ranks = torch.arange(topv.shape[-1], device=sims.device).float()
            w = 1.0 / (alpha + ranks)
            w = w / w.sum()
            w = w.unsqueeze(0).expand(sims.size(0), -1)
        else:
            raise ValueError(f"Unknown aggregation method: {self.cfg.mode.aggregation}")

        if self.use_token_ensemble:
            return_tok = neigh_tok
        elif self.cfg.mode.aggregation.startswith(("softmax", "rank")):
            mix = (w.unsqueeze(-1) * neigh_tok).sum(dim=1)  # [B, d_t]
            return_tok = mix
        return return_tok, w

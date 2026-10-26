import os, json, numpy as np, torch
from typing import List, Optional
from collections import defaultdict

def _save_ids(ids: List[int], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".npy"):
        np.save(path, np.array(ids, dtype=np.int64))
    elif path.endswith(".json"):
        with open(path, "w") as f:
            json.dump(ids, f)
    else:
        raise ValueError("forget.path must end with .npy or .json")

def _load_ids(path: str) -> List[int]:
    if path.endswith(".npy"):
        return np.load(path).astype(np.int64).tolist()
    elif path.endswith(".json"):
        with open(path, "r") as f:
            return list(map(int, json.load(f)))
    else:
        raise ValueError("forget.path must end with .npy or .json")

def make_or_load_forget_ids(train_len: int, size, mode, seed, path, exclude=None, labels=None) -> List[int]:
    """Select reproducible forget IDs, optionally preserving class proportions."""
    ids: Optional[List[int]] = None

    exclude = {int(x.item()) if isinstance(x, torch.Tensor) else int(x) for x in exclude} if exclude is not None else set()
    pool = [i for i in range(train_len) if i not in exclude]
    size = min(size, len(pool))
    if size <= 0:
        return torch.empty(0, dtype=torch.long)

    if mode == "file":  # require path and load it.
        assert path, "dataset.forget.path must be set in 'file' mode"
        ids = _load_ids(path)
        if exclude.intersection(ids):
            raise RuntimeError(f"Loaded ids from {path} overlap with excluded ids.")

    elif mode == "seed": # always regenerate from seed (deterministic), and (optionally) save to path.
        rng = np.random.default_rng(seed)
        if labels is None:
            # Random sampling
            ids = rng.choice(pool, size=size, replace=False).astype(np.int64).tolist()
        else:
            # class-balanced sampling
            per_class = defaultdict(list)
            for idx in pool:
                per_class[int(labels[idx])].append(idx)

            ids = []
            total = sum(len(v) for v in per_class.values())

            for cls, cls_ids in per_class.items():
                if len(cls_ids) == 0:
                    continue
                k_cls = int(round(size * len(cls_ids) / total))
                k_cls = min(k_cls, len(cls_ids))
                if k_cls > 0:
                    ids.extend(rng.choice(cls_ids, size=k_cls, replace=False).tolist())

            # Fix rounding mismatch
            if len(ids) > size:
                ids = ids[:size]
            elif len(ids) < size:
                remaining = list(set(pool) - set(ids))
                extra = rng.choice(remaining, size=size - len(ids), replace=False).tolist()
                ids.extend(extra)

        if path and not os.path.exists(path): _save_ids(ids, path)

    else:
        raise ValueError(f"Unknown forget.mode: {mode}")

    # Sanity checks:
    if len(set(ids)) != len(ids):
        raise RuntimeError("Forget IDs contain duplicates.")
    if not all(0 <= i < train_len for i in ids):
        raise RuntimeError("Forget IDs out of bounds.")
    return torch.tensor(ids)

from torch.utils.data import Dataset
import os
from PIL import Image

class TinyImageNet(Dataset):
    def __init__(self, root, split="train", transform=None, download=False):
        # root should contain "tiny-imagenet-200"
        base_dir = os.path.join(root, "tiny-imagenet-200")
        self.split = split
        self.transform = transform

        with open(os.path.join(base_dir, "wnids.txt")) as f:
            self.classes = [l.strip() for l in f]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        if split == "train":
            split_dir = os.path.join(base_dir, "train")
            self.samples = self._load_train(split_dir)
        elif split == "val":
            self.samples = self._load_val(base_dir)
        elif split == "test":
            self.samples = self._load_test(base_dir)
            self.class_to_idx, self.classes = None, None
        else:
            raise ValueError(f"Invalid split: {split}")

    def _load_train(self, train_dir):
        samples = []
        for wnid, idx in self.class_to_idx.items():
            img_dir = os.path.join(train_dir, wnid, "images")
            for fname in sorted(os.listdir(img_dir)):
                if fname.endswith(".JPEG"):
                    samples.append((os.path.join(img_dir, fname), idx))
        return samples

    def _load_val(self, base_dir):
        ann_path = os.path.join(base_dir, "val", "val_annotations.txt")
        img_dir = os.path.join(base_dir, "val", "images")

        samples = []
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                img_name, wnid = parts[0], parts[1]

                # IMPORTANT: use the SAME mapping as train
                idx = self.class_to_idx[wnid]

                samples.append((os.path.join(img_dir, img_name), idx))

        return samples

    def _load_test(self, base_dir):
        img_dir = os.path.join(base_dir, "test", "images")
        samples = []
        for fname in sorted(os.listdir(img_dir)):
            if fname.lower().endswith(".jpeg"):
                path = os.path.join(img_dir, fname)
                samples.append((path, -1))  # no labels provided
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class Meta(Dataset):
    def __init__(self, dataset, id_offset=0):
        self.dataset = dataset
        self.id_offset = int(id_offset)

    def __getitem__(self, index):
        item = self.dataset[index]

        if isinstance(item, dict):
            x = item["x"]
            y = item["y"]
            extra = {k: v for k, v in item.items() if k not in ["x", "y", "id"]}
        else:
            x, y = item
            extra = {}

        return {"x": x, "y": y, "id": index + self.id_offset, **extra}

    def __len__(self):
        return len(self.dataset)

class MedMNISTWrapper(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __getitem__(self, index):
        x, y = self.base_dataset[index]

        # y is shape (1,) → scalar
        y = int(y[0])

        # ensure 3 channels (ViT expects this)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)

        return x, y

    def __len__(self):
        return len(self.base_dataset)

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torch



def aireadi_collate_fn(batch):

    glucose_window = torch.tensor(
        np.stack([b["glucose_window"] for b in batch]),
        dtype=torch.float32
    )

    # target = torch.tensor(
    #     [b["target"] for b in batch],
    #     dtype=torch.float32
    # )

    retinal_images = torch.tensor(
        np.stack([b["retinal_images"] for b in batch]),
        dtype=torch.float32
    )

    age = torch.tensor(
        [b["age"] for b in batch],
        dtype=torch.long
    )

    patient_id = torch.tensor(
        [b["patient_id"] for b in batch],
        dtype=torch.long
    )

    study_group = [b["study_group"] for b in batch]
    text_description = [b["text_description"] for b in batch]
    time_local = [b["time_local"] for b in batch]

    return {
        "glucose_window": glucose_window,
        # "target": target,
        "retinal_images": retinal_images,
        "age": age,
        "patient_id": patient_id,
        "study_group": study_group,
        "text_description": text_description,
        "time_local": time_local
    }


class AIREADIDataset(Dataset):

    def __init__(self,
                 split: str,
                 data_path: str,
                 window_size: int = 24):

        self.split_name = split
        self.data_path = data_path
        self.window_size = window_size
        # self.pred_horizon = pred_horizon

        self.glucose_min_ = 40.0
        self.glucose_max_ = 400.0

        self.load_glucose()
        self.load_meta_data()
        self.cache_retinal_images()
        self.build_windows()

    def load_glucose(self):

        parquet_path = f"{self.data_path}/glucose_{self.split_name}.parquet"
        retinal_root = f"{self.data_path}/retinal_photography/cfp/topcon_maestro2"

        df = pd.read_parquet(parquet_path)

        df["patient_id"] = df["patient_id"].apply(
            lambda x: x[0].replace("AIREADI-", "")
            if isinstance(x, np.ndarray) and len(x) > 0
            else None
        )

        df = df.dropna(subset=["patient_id"])

        valid_ids = set(os.listdir(retinal_root))
        df = df[df["patient_id"].isin(valid_ids)]

        df = df.sort_values(["patient_id"])

        print("rows:", len(df))
        print("patients:", df["patient_id"].nunique())

        self.glucose_df = df
        self.patient_groups = dict(tuple(df.groupby("patient_id")))

    def load_meta_data(self):
        self.meta_data = pd.read_csv("participants.tsv", sep="\t")

    def cache_retinal_images(self):

        print("Caching retinal images...")

        root = f"{self.data_path}/retinal_photography/cfp/topcon_maestro2"

        self.retinal_cache = {}

        order = [
            "macula_oct_cfp_l",
            "macula_oct_cfp_r",
            "wide_oct_cfp_l",
            "wide_oct_cfp_r",
        ]

        for pid in self.patient_groups.keys():

            patient_path = os.path.join(root, pid)

            if not os.path.exists(patient_path):
                continue

            files = os.listdir(patient_path)

            imgs = []

            for key in order:

                matches = sorted([f for f in files if key in f])

                if len(matches) == 0:
                    continue

                path = os.path.join(patient_path, matches[0])

                img = np.array(Image.open(path))

                imgs.append(img)

            if len(imgs) == 4:
                self.retinal_cache[pid] = np.stack(imgs)

        print("retinal cached:", len(self.retinal_cache))

    def build_windows(self):

        self.windows = []

        for pid, g in self.patient_groups.items():

            if pid not in self.retinal_cache:
                continue

            values = g["glucose"].values[0]

            max_start = len(values) - self.window_size

            if max_start <= 0:
                continue

            for i in range(max_start):
                self.windows.append((pid, i))

        print("total windows:", len(self.windows))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):

        sample = {}

        pid, start_idx = self.windows[idx]

        patient_df = self.patient_groups[pid]

        values = patient_df["glucose"].values[0]
        times = patient_df["time_local"].values[0]

        window = values[start_idx:start_idx + self.window_size]

        time_local = times[start_idx + self.window_size]

        window = (window - self.glucose_min_) / (self.glucose_max_ - self.glucose_min_)

        sample["glucose_window"] = window.astype(np.float32)
        sample["time_local"] = time_local
        sample["patient_id"] = int(pid)

        meta = self.meta_data.query(f"person_id == {pid}")

        age = meta["age"].item()
        study_group = meta["study_group"].item()

        sample["age"] = age
        sample["study_group"] = study_group
        sample["text_description"] = f"This patient is {age} years old, their status is {study_group}"

        sample["retinal_images"] = self.retinal_cache[pid]

        return sample


from torch.utils.data import DataLoader


def test_dataset(dataset, num_samples=20):

    print("\n===== DATASET SANITY CHECK =====")

    retinal_shapes = set()

    for i in range(num_samples):

        sample = dataset[i]

        g = sample["glucose_window"]
        r = sample["retinal_images"]

        print(f"\nSample {i}")
        print("glucose_window shape:", g.shape)
        print("retinal_images shape:", r.shape)

        retinal_shapes.add(r.shape)

        assert g.shape[0] == dataset.window_size, "glucose window size incorrect"
        assert r.shape[0] == 4, "retinal channel number incorrect"

    print("\nUnique retinal shapes found:", retinal_shapes)

    if len(retinal_shapes) == 1:
        print("✓ All retinal image sizes are consistent")
    else:
        print("⚠ WARNING: retinal sizes are inconsistent")


def test_dataloader(dataset, batch_size=4):

    print("\n===== DATALOADER TEST =====")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=aireadi_collate_fn
    )

    batch = next(iter(loader))

    print("\nBatch keys:", batch.keys())

    print("glucose_window:", batch["glucose_window"].shape)
    print("retinal_images:", batch["retinal_images"].shape)
    print("age:", batch["age"].shape)
    print("patient_id:", batch["patient_id"].shape)

    print("study_group example:", batch["study_group"][0])
    print("text_description example:", batch["text_description"][0])

    print("\n✓ DataLoader works correctly")


if __name__ == "__main__":

    train_set = AIREADIDataset(
        split="train",
        data_path="/Users/zhc/Documents/AI-READI",
        window_size=24
    )

    print("\nDataset size:", len(train_set))

    # 单 sample 测试
    sample = train_set[0]

    print("\nSingle sample check")
    print("glucose_window:", sample["glucose_window"].shape)
    print("retinal_images:", sample["retinal_images"].shape)

    # dataset sanity check
    test_dataset(train_set)

    # dataloader test
    test_dataloader(train_set)
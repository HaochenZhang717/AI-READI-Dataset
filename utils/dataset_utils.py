import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class MultiModalTSDataset(Dataset):
    def __init__(
        self,
        calorie_path,
        glucose_path,
        window_size=128,
        predict_length=64,
        stride=64,
    ):

        self.pred_length = predict_length

        self.calorie_df = pd.read_parquet(calorie_path)
        self.glucose_df = pd.read_parquet(glucose_path)


        def extract_pid(x):
            if x is None:
                return None

            if isinstance(x, (list, np.ndarray)):
                if len(x) == 0:
                    return None
                return x[0]

            return x

        self.calorie_df = self.calorie_df[~self.calorie_df["is_missing"]].reset_index(drop=True)


        self.calorie_df["pid"] = self.calorie_df["patient_id"].apply(extract_pid)
        self.glucose_df["pid"] = self.glucose_df["patient_id"].apply(extract_pid)

        # 分别找到各自有效 pid
        cal_ids = set(self.calorie_df["pid"].dropna())
        glu_ids = set(self.glucose_df["pid"].dropna())

        self.common_ids = list(cal_ids & glu_ids)

        # 过滤 dataframe
        self.calorie_df = self.calorie_df[
            self.calorie_df["pid"].apply(lambda x: x in self.common_ids)
        ].reset_index(drop=True)

        self.glucose_df = self.glucose_df[
            self.glucose_df["pid"].apply(lambda x: x in self.common_ids)
        ].reset_index(drop=True)

        # 建 index
        self.calorie_map = {
            row['pid']: row
            for _, row in self.calorie_df.iterrows()
        }

        self.glucose_map = {
            row['pid']: row
            for _, row in self.glucose_df.iterrows()
        }

        # ========= 新增：calorie 插值 =========
        self.aligned_data = {}

        for pid in self.common_ids:

            cal = self.calorie_map[pid]
            glu = self.glucose_map[pid]

            # ===== 1️⃣ 取数据 =====
            t_cal = np.array(cal["time_local"], dtype="datetime64[ns]")
            x_cal = np.array(cal["calorie"], dtype=np.float32)

            t_glu = np.array(glu["time_local"], dtype="datetime64[ns]")
            x_glu = np.array(glu["glucose"], dtype=np.float32)

            # ===== 2️⃣ 转时间为 float（秒）=====
            t_cal_float = t_cal.astype("int64") / 1e9
            t_glu_float = t_glu.astype("int64") / 1e9

            # ===== 3️⃣ 插值 =====
            if len(t_cal_float) > 1:
                interp_cal = np.interp(
                    t_glu_float,
                    t_cal_float,
                    x_cal,
                    left=0.0,
                    right=0.0
                )

                # mask：哪些是有效插值范围
                mask = (t_glu_float >= t_cal_float.min()) & (t_glu_float <= t_cal_float.max())

            else:
                interp_cal = np.zeros_like(x_glu, dtype=np.float32)
                mask = np.zeros_like(x_glu, dtype=bool)

            # ===== 4️⃣ 存起来 =====
            self.aligned_data[pid] = {
                "glucose": x_glu,
                "calorie_interp": interp_cal,
                "calorie_mask": mask.astype(np.float32),
                "time": t_glu
            }

        # ========= sliding window =========
        self.samples = []

        for pid in self.common_ids:

            data = self.aligned_data[pid]

            glu = data["glucose"]
            cal = data["calorie_interp"]
            mask = data["calorie_mask"]

            L = len(glu)

            # ❗防止长度不够
            if L < window_size:
                continue

            for start in range(0, L - window_size + 1, stride):
                end = start + window_size

                self.samples.append({
                    "glucose": glu[start:end],
                    "calorie": cal[start:end],
                    "mask": mask[start:end],
                    "pid": pid
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        glucose = torch.tensor(s["glucose"], dtype=torch.float32)
        calorie = torch.tensor(s["calorie"], dtype=torch.float32)

        all_modalities = torch.cat([glucose.unsqueeze(-1), calorie.unsqueeze(-1)], dim=-1)
        return all_modalities

if __name__ == "__main__":
    data = MultiModalTSDataset(
        calorie_path="../AI-READI-processed/calorie_train.parquet",
        glucose_path="../AI-READI-processed/glucose_train.parquet",
        window_size=128,
        stride=64,
    )
    a = data[0]
    print(a.shape)
import json
import pandas as pd
from tqdm import tqdm
import os
import numpy as np

BASE_PATH = "/playpen-shared/mshuang/morris/morris/d9ef6cf1-f6c3-4956-a91e-adf409e105f0/dataset/wearable_activity_monitor/heart_rate/garmin_vivosmart5"

TZ_MAP = {
    "pst": "US/Pacific", "pdt": "US/Pacific",
    "est": "US/Eastern", "edt": "US/Eastern",
    "cst": "US/Central", "cdt": "US/Central",
    "mst": "US/Mountain", "mdt": "US/Mountain",
}


def build_empty_sample(pid):
    return {
        "time_utc": np.array([], dtype="datetime64[ns]"),
        "heart_rate": np.array([], dtype=np.float32),
        "unit": np.array([], dtype="str"),
        "patient_id": np.array([pid]),
        "is_missing": True,
    }


def load_heart_rate_json(path: str, pid: str) -> dict:
    if not os.path.exists(path):
        return build_empty_sample(pid)

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return build_empty_sample(pid)

    header = data.get("header") or {}
    body = data.get("body") or {}

    patient_id = header.get("patient_id", header.get("uuid", pid))
    timezone = header.get("timezone", None)

    records = body.get("heart_rate") or []
    if not records:
        return build_empty_sample(pid)

    rows = []
    for r in records:
        etf = r.get("effective_time_frame") or {}
        t = etf.get("date_time", None)

        hr = r.get("heart_rate") or {}
        hr_val = hr.get("value", None)
        hr_unit = hr.get("unit", None)

        if t is None or hr_val is None:
            continue

        rows.append({
            "time": t,
            "heart_rate": hr_val,
            "unit": hr_unit,
            "patient_id": patient_id,
        })

    if len(rows) == 0:
        return build_empty_sample(pid)

    df = pd.DataFrame(rows)
    df["time_utc"] = pd.to_datetime(df["time"], format="ISO8601", utc=True, errors="coerce")
    df = df.drop(columns=["time"])
    df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")
    df = df.dropna(subset=["time_utc", "heart_rate"])

    if len(df) == 0:
        return build_empty_sample(pid)

    df = (
        df.sort_values("time_utc")
        .drop_duplicates(subset=["time_utc"], keep="last")
        .reset_index(drop=True)
    )

    if timezone is not None:
        tz = TZ_MAP.get(str(timezone).lower(), None)
        if tz is not None:
            df["time_local"] = df["time_utc"].dt.tz_convert(tz)

    for col in ["unit", "patient_id"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    result = {col: df[col].to_numpy() for col in df.columns}
    result["is_missing"] = False
    return result


def save_heart_rate_to_parquet(split_ids, save_path):
    result_list = []
    exist_count = 0
    missing_count = 0

    for split_id in tqdm(split_ids):
        path = f"{BASE_PATH}/{split_id}/{split_id}_heartrate.json"
        sample = load_heart_rate_json(path, split_id)

        if sample["is_missing"]:
            missing_count += 1
        else:
            exist_count += 1

        result_list.append(sample)

    print(f"Existing files: {exist_count}")
    print(f"Missing files: {missing_count}")

    def fix_sample(sample):
        sample["time_utc"] = sample["time_utc"].astype("datetime64[ns]").tolist()
        if "time_local" in sample:
            sample["time_local"] = sample["time_local"].astype("datetime64[ns]").tolist()
        sample["heart_rate"] = sample["heart_rate"].astype(float).tolist()
        sample["unit"] = sample["unit"].tolist()
        sample["patient_id"] = sample["patient_id"].tolist()
        return sample

    result_list = [fix_sample(x) for x in result_list]
    df = pd.DataFrame(result_list)
    df.to_parquet(save_path)


def save_heart_rate_all():
    participants_path = "participants.tsv"
    df = pd.read_csv(participants_path, sep="\t")
    df["person_id"] = df["person_id"].astype(str)

    train_ids = df.loc[df["recommended_split"] == "train", "person_id"].tolist()
    val_ids = df.loc[df["recommended_split"] == "val", "person_id"].tolist()
    test_ids = df.loc[df["recommended_split"] == "test", "person_id"].tolist()

    print("train:", len(train_ids))
    save_heart_rate_to_parquet(train_ids, "../AI-READI-processed/heart_rate_train.parquet")

    print("val:", len(val_ids))
    save_heart_rate_to_parquet(val_ids, "../AI-READI-processed/heart_rate_valid.parquet")

    print("test:", len(test_ids))
    save_heart_rate_to_parquet(test_ids, "../AI-READI-processed/heart_rate_test.parquet")


if __name__ == "__main__":
    save_heart_rate_all()

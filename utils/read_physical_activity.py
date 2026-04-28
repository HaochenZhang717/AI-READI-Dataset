import json
import pandas as pd
from tqdm import tqdm
import os
import numpy as np

BASE_PATH = "/playpen-shared/mshuang/morris/morris/d9ef6cf1-f6c3-4956-a91e-adf409e105f0/dataset/wearable_activity_monitor/physical_activity/garmin_vivosmart5"

TZ_MAP = {
    "pst": "US/Pacific", "pdt": "US/Pacific",
    "est": "US/Eastern", "edt": "US/Eastern",
    "cst": "US/Central", "cdt": "US/Central",
    "mst": "US/Mountain", "mdt": "US/Mountain",
}


def build_empty_sample(pid):
    return {
        "time_start_utc": np.array([], dtype="datetime64[ns]"),
        "time_end_utc": np.array([], dtype="datetime64[ns]"),
        "steps": np.array([], dtype=np.float32),
        "unit": np.array([], dtype="str"),
        "activity_name": np.array([], dtype="str"),
        "patient_id": np.array([pid]),
        "is_missing": True,
    }


def load_physical_activity_json(path: str, pid: str) -> dict:
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

    records = body.get("activity") or []
    if not records:
        return build_empty_sample(pid)

    rows = []
    for r in records:
        etf = r.get("effective_time_frame") or {}
        interval = etf.get("time_interval") or {}
        t_start = interval.get("start_date_time", None)
        t_end = interval.get("end_date_time", None)

        bmq = r.get("base_movement_quantity") or {}
        steps_val = bmq.get("value", None)
        steps_unit = bmq.get("unit", None)

        # skip records with no start time
        if t_start is None:
            continue

        rows.append({
            "time_start": t_start,
            "time_end": t_end,
            "steps": steps_val,
            "unit": steps_unit,
            "activity_name": r.get("activity_name", None),
            "patient_id": patient_id,
        })

    if len(rows) == 0:
        return build_empty_sample(pid)

    df = pd.DataFrame(rows)
    df["time_start_utc"] = pd.to_datetime(df["time_start"], format="ISO8601", utc=True, errors="coerce")
    df["time_end_utc"] = pd.to_datetime(df["time_end"], format="ISO8601", utc=True, errors="coerce")
    df = df.drop(columns=["time_start", "time_end"])

    # empty string → NaN, then coerce to float
    df["steps"] = pd.to_numeric(df["steps"], errors="coerce")
    df = df.dropna(subset=["time_start_utc"])

    if len(df) == 0:
        return build_empty_sample(pid)

    df = (
        df.sort_values("time_start_utc")
        .drop_duplicates(subset=["time_start_utc"], keep="last")
        .reset_index(drop=True)
    )

    if timezone is not None:
        tz = TZ_MAP.get(str(timezone).lower(), None)
        if tz is not None:
            df["time_start_local"] = df["time_start_utc"].dt.tz_convert(tz)
            df["time_end_local"] = df["time_end_utc"].dt.tz_convert(tz)

    for col in ["unit", "activity_name", "patient_id"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    result = {}
    for col in df.columns:
        s = df[col]
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            s = s.dt.tz_localize(None)
        result[col] = s.to_numpy()
    result["is_missing"] = False
    return result


def save_physical_activity_to_parquet(split_ids, save_path):
    result_list = []
    exist_count = 0
    missing_count = 0

    for split_id in tqdm(split_ids):
        path = f"{BASE_PATH}/{split_id}/{split_id}_activity.json"
        sample = load_physical_activity_json(path, split_id)

        if sample["is_missing"]:
            missing_count += 1
        else:
            exist_count += 1

        result_list.append(sample)

    print(f"Existing files: {exist_count}")
    print(f"Missing files: {missing_count}")

    def fix_sample(sample):
        sample["time_start_utc"] = pd.to_datetime(sample["time_start_utc"]).tolist()
        sample["time_end_utc"] = pd.to_datetime(sample["time_end_utc"]).tolist()
        if "time_start_local" in sample:
            sample["time_start_local"] = pd.to_datetime(sample["time_start_local"]).tolist()
            sample["time_end_local"] = pd.to_datetime(sample["time_end_local"]).tolist()
        sample["steps"] = sample["steps"].astype(float).tolist()
        sample["unit"] = sample["unit"].tolist()
        sample["activity_name"] = sample["activity_name"].tolist()
        sample["patient_id"] = sample["patient_id"].tolist()
        return sample

    result_list = [fix_sample(x) for x in result_list]
    df = pd.DataFrame(result_list)
    df.to_parquet(save_path)


def save_physical_activity_all():
    participants_path = "participants.tsv"
    df = pd.read_csv(participants_path, sep="\t")
    df["person_id"] = df["person_id"].astype(str)

    train_ids = df.loc[df["recommended_split"] == "train", "person_id"].tolist()
    val_ids = df.loc[df["recommended_split"] == "val", "person_id"].tolist()
    test_ids = df.loc[df["recommended_split"] == "test", "person_id"].tolist()

    print("train:", len(train_ids))
    save_physical_activity_to_parquet(train_ids, "../AI-READI-processed/physical_activity_train.parquet")

    print("val:", len(val_ids))
    save_physical_activity_to_parquet(val_ids, "../AI-READI-processed/physical_activity_valid.parquet")

    print("test:", len(test_ids))
    save_physical_activity_to_parquet(test_ids, "../AI-READI-processed/physical_activity_test.parquet")


if __name__ == "__main__":
    save_physical_activity_all()

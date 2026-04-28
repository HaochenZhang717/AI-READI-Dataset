import json
import pandas as pd
from tqdm import tqdm
import os
import numpy as np

BASE_PATH = "/playpen-shared/mshuang/morris/morris/d9ef6cf1-f6c3-4956-a91e-adf409e105f0/dataset/wearable_activity_monitor/oxygen_saturation/garmin_vivosmart5"

TZ_MAP = {
    "pst": "US/Pacific", "pdt": "US/Pacific",
    "est": "US/Eastern", "edt": "US/Eastern",
    "cst": "US/Central", "cdt": "US/Central",
    "mst": "US/Mountain", "mdt": "US/Mountain",
}


def build_empty_sample(pid):
    return {
        "time_utc": np.array([], dtype="datetime64[ns]"),
        "spo2": np.array([], dtype=np.float32),
        "unit": np.array([], dtype="str"),
        "measurement_method": np.array([], dtype="str"),
        "patient_id": np.array([pid]),
        "is_missing": True,
    }


def load_oxygen_saturation_json(path: str, pid: str) -> dict:
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

    records = body.get("breathing") or []
    if not records:
        return build_empty_sample(pid)

    rows = []
    for r in records:
        etf = r.get("effective_time_frame") or {}
        t = etf.get("date_time", None)

        spo2 = r.get("oxygen_saturation") or {}
        spo2_val = spo2.get("value", None)
        spo2_unit = spo2.get("unit", None)

        if t is None or spo2_val is None:
            continue

        rows.append({
            "time": t,
            "spo2": spo2_val,
            "unit": spo2_unit,
            "measurement_method": r.get("measurement_method", None),
            "patient_id": patient_id,
        })

    if len(rows) == 0:
        return build_empty_sample(pid)

    df = pd.DataFrame(rows)
    df["time_utc"] = pd.to_datetime(df["time"], format="ISO8601", utc=True, errors="coerce")
    df = df.drop(columns=["time"])
    df["spo2"] = pd.to_numeric(df["spo2"], errors="coerce")
    df = df.dropna(subset=["time_utc", "spo2"])

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

    for col in ["unit", "measurement_method", "patient_id"]:
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


def save_oxygen_saturation_to_parquet(split_ids, save_path):
    result_list = []
    exist_count = 0
    missing_count = 0

    for split_id in tqdm(split_ids):
        path = f"{BASE_PATH}/{split_id}/{split_id}_oxygensaturation.json"
        sample = load_oxygen_saturation_json(path, split_id)

        if sample["is_missing"]:
            missing_count += 1
        else:
            exist_count += 1

        result_list.append(sample)

    print(f"Existing files: {exist_count}")
    print(f"Missing files: {missing_count}")

    def fix_sample(sample):
        sample["time_utc"] = pd.to_datetime(sample["time_utc"]).tolist()
        if "time_local" in sample:
            sample["time_local"] = pd.to_datetime(sample["time_local"]).tolist()
        sample["spo2"] = sample["spo2"].astype(float).tolist()
        sample["unit"] = sample["unit"].tolist()
        sample["measurement_method"] = sample["measurement_method"].tolist()
        sample["patient_id"] = sample["patient_id"].tolist()
        return sample

    result_list = [fix_sample(x) for x in result_list]
    df = pd.DataFrame(result_list)
    df.to_parquet(save_path)


def save_oxygen_saturation_all():
    participants_path = "participants.tsv"
    df = pd.read_csv(participants_path, sep="\t")
    df["person_id"] = df["person_id"].astype(str)

    train_ids = df.loc[df["recommended_split"] == "train", "person_id"].tolist()
    val_ids = df.loc[df["recommended_split"] == "val", "person_id"].tolist()
    test_ids = df.loc[df["recommended_split"] == "test", "person_id"].tolist()

    print("train:", len(train_ids))
    save_oxygen_saturation_to_parquet(train_ids, "../AI-READI-processed/oxygen_saturation_train.parquet")

    print("val:", len(val_ids))
    save_oxygen_saturation_to_parquet(val_ids, "../AI-READI-processed/oxygen_saturation_valid.parquet")

    print("test:", len(test_ids))
    save_oxygen_saturation_to_parquet(test_ids, "../AI-READI-processed/oxygen_saturation_test.parquet")


if __name__ == "__main__":
    save_oxygen_saturation_all()

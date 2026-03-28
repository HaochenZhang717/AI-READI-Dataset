import json
import pandas as pd
from tqdm import tqdm
import os
import numpy as np


# =========================
# helper：构造空样本
# =========================
def build_empty_sample(pid):
    return {
        "time_utc": np.array([], dtype="datetime64[ns]"),
        "calorie": np.array([], dtype=np.float32),
        "unit": np.array([], dtype="str"),
        "event_type": np.array([], dtype="str"),
        "source_device_id": np.array([], dtype="str"),
        "patient_id": np.array([pid]),
        "is_missing": True,
    }


# =========================
# parser（核心修复版）
# =========================
def load_calorie_json(path: str, pid: str) -> dict:
    """
    永远返回一个 dict（即使是空）
    """

    # ❗文件不存在
    if not os.path.exists(path):
        return build_empty_sample(pid)

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return build_empty_sample(pid)

    # 🔥 防御式
    header = data.get("header") or {}
    body = data.get("body") or {}

    patient_id = header.get("patient_id", header.get("uuid", pid))
    timezone = header.get("timezone", None)

    # 🔥 兼容不同 key（你刚刚那个 activity 就是坑点）
    records = (
        body.get("physical_activity_calorie")
        or body.get("activity")
        or []
    )
    # ❗空记录
    if not records:
        return build_empty_sample(pid)

    rows = []

    for r in records:

        # ===== 时间 =====
        etf = r.get("effective_time_frame") or {}
        t = etf.get("date_time", None)

        # ===== calorie =====
        cal = r.get("calories_value") or {}
        calorie_val = cal.get("value", None)
        calorie_unit = cal.get("unit", None)

        # ===== 过滤非法 =====
        if t is None or calorie_val is None:
            continue
        rows.append({
            "time": t,
            "calorie": calorie_val,
            "unit": calorie_unit,
            "event_type": r.get("activity_name", None),  # 就用这个
            "source_device_id": r.get("source_device_id", None),
            "patient_id": patient_id,
        })



    # ❗过滤后为空
    if len(rows) == 0:
        return build_empty_sample(pid)

    df = pd.DataFrame(rows)

    # ===== 时间处理 =====
    df["time_utc"] = pd.to_datetime(df["time"], format="ISO8601", utc=True, errors="coerce")
    df = df.drop(columns=["time"])

    # ===== 数值处理 =====
    df["calorie"] = pd.to_numeric(df["calorie"], errors="coerce")

    # ===== 清洗 =====
    df = df.dropna(subset=["time_utc", "calorie"])

    if len(df) == 0:
        return build_empty_sample(pid)

    df = (
        df.sort_values("time_utc")
        .drop_duplicates(subset=["time_utc"], keep="last")
        .reset_index(drop=True)
    )

    # ===== timezone =====
    if timezone is not None:
        tz_map = {
            "pst": "US/Pacific",
            "pdt": "US/Pacific",
            "est": "US/Eastern",
            "edt": "US/Eastern",
            "cst": "US/Central",
            "cdt": "US/Central",
            "mst": "US/Mountain",
            "mdt": "US/Mountain",
        }

        tz = tz_map.get(str(timezone).lower(), None)

        if tz is not None:
            df["time_local"] = df["time_utc"].dt.tz_convert(tz)

    # ===== string columns =====
    for col in ["unit", "event_type", "source_device_id", "patient_id"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    result = {col: df[col].to_numpy() for col in df.columns}
    result["is_missing"] = False
    return result


# =========================
# parquet
# =========================
def save_calorie_to_parquet(split_ids, save_path):

    result_list = []
    exist_count = 0
    missing_count = 0

    for split_id in tqdm(split_ids):

        calorie_file_path = f"/playpen-shared/mshuang/morris/morris/d9ef6cf1-f6c3-4956-a91e-adf409e105f0/dataset/wearable_activity_monitor/physical_activity_calorie/garmin_vivosmart5/{split_id}/{split_id}_calorie.json"

        sample = load_calorie_json(calorie_file_path, split_id)

        if sample["is_missing"]:
            missing_count += 1
        else:
            exist_count += 1

        result_list.append(sample)

    print(f"Existing files: {exist_count}")
    print(f"Missing files: {missing_count}")

    df = pd.DataFrame(result_list)
    df.to_parquet(save_path)


# =========================
# main
# =========================
def save_calorie_all():

    participants_path = "participants.tsv"

    df = pd.read_csv(participants_path, sep="\t")
    df["person_id"] = df["person_id"].astype(str)

    train_ids = df.loc[df["recommended_split"] == "train", "person_id"].tolist()
    val_ids = df.loc[df["recommended_split"] == "val", "person_id"].tolist()
    test_ids = df.loc[df["recommended_split"] == "test", "person_id"].tolist()

    print("train:", len(train_ids))
    save_calorie_to_parquet(train_ids, "/playpen-shared/haochenz/AI-READI/calorie_train.parquet")

    print("val:", len(val_ids))
    save_calorie_to_parquet(val_ids, "/playpen-shared/haochenz/AI-READI/calorie_valid.parquet")

    print("test:", len(test_ids))
    save_calorie_to_parquet(test_ids, "/playpen-shared/haochenz/AI-READI/calorie_test.parquet")


if __name__ == "__main__":
    save_calorie_all()
    # scp -r haochenz@unites3.cs.unc.edu:/playpen-shared/haochenz/AI-READI/glucose_test.parquet ./
    # scp -r haochenz@unites3.cs.unc.edu:/playpen-shared/haochenz/AI-READI/glucose_train.parquet ./
    # scp -r haochenz@unites3.cs.unc.edu:/playpen-shared/haochenz/AI-READI/glucose_valid.parquet ./


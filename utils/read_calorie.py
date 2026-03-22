import json
import pandas as pd
from tqdm import tqdm
import os
import numpy as np

def load_calorie_json(path: str) -> dict:
    """
    Load calorie JSON and return a dictionary:

    {
        column_name : np.array(...)
    }
    """

    with open(path, "r") as f:
        data = json.load(f)

    header = data.get("header", {})
    body = data.get("body", {})

    patient_id = header.get("patient_id", header.get("uuid", None))
    timezone = header.get("timezone", None)

    records = body.get("physical_activity_calorie", [])
    if not records:
        return {}

    rows = []

    for r in records:
        time_interval = r.get("effective_time_frame", {}).get("time_interval", {})
        t = time_interval.get("start_date_time", None)

        cal = r.get("calorie_burned", {})
        calorie_val = cal.get("value", None)
        calorie_unit = cal.get("unit", None)

        rows.append(
            {
                "time": t,
                "calorie": calorie_val,
                "unit": calorie_unit,
                "event_type": r.get("event_type", None),
                "source_device_id": r.get("source_device_id", None),
                "patient_id": patient_id,
            }
        )

    df = pd.DataFrame(rows)

    # ===== 时间处理（完全一致）=====
    df["time_utc"] = pd.to_datetime(df["time"], format="ISO8601", utc=True, errors="coerce")
    df = df.drop(columns=["time"])

    breakpoint()
    # ===== 数值处理 =====
    df["calorie"] = pd.to_numeric(df["calorie"], errors="coerce")

    # ===== 清洗 =====
    df = df.dropna(subset=["time_utc", "calorie"])

    df = (
        df.sort_values("time_utc")
        .drop_duplicates(subset=["time_utc"], keep="last")
        .reset_index(drop=True)
    )

    # ===== timezone（完全对齐）=====
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

    # ===== string columns（对齐）=====
    for col in ["unit", "event_type", "source_device_id", "patient_id"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # ===== 转 dict =====
    result = {col: df[col].to_numpy() for col in df.columns}

    return result


def save_calorie_to_parquet(split_ids, save_path):

    result_list = []
    exist_count = 0
    missing_count = 0

    for split_id in tqdm(split_ids):

        calorie_file_path = f"/playpen-shared/mshuang/morris/morris/d9ef6cf1-f6c3-4956-a91e-adf409e105f0/dataset/wearable_activity_monitor/physical_activity_calorie/garmin_vivosmart5/{split_id}/{split_id}_calorie.json"

        if not os.path.exists(calorie_file_path):
            missing_count += 1

            # ✅ 构造空样本（关键！！）
            empty_sample = {
                "time_utc": np.array([], dtype="datetime64[ns]"),
                "calorie": np.array([], dtype=np.float32),
                "unit": np.array([], dtype="str"),
                "event_type": np.array([], dtype="str"),
                "source_device_id": np.array([], dtype="str"),
                "patient_id": np.array([split_id]),  # 保留 id
                "is_missing": True,  # 🔥 强烈建议加
            }

            result_list.append(empty_sample)
            continue

        calorie = load_calorie_json(calorie_file_path)

        if len(calorie) == 0:
            missing_count += 1

            empty_sample = {
                "time_utc": np.array([], dtype="datetime64[ns]"),
                "calorie": np.array([], dtype=np.float32),
                "unit": np.array([], dtype="str"),
                "event_type": np.array([], dtype="str"),
                "source_device_id": np.array([], dtype="str"),
                "patient_id": np.array([split_id]),
                "is_missing": True,
            }

            result_list.append(empty_sample)
            continue

        # ✅ 正常数据
        calorie["is_missing"] = False
        result_list.append(calorie)
        exist_count += 1

    print(f"Existing files: {exist_count}")
    print(f"Missing files: {missing_count}")

    df = pd.DataFrame(result_list)
    df.to_parquet(save_path)


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

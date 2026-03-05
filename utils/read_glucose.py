import pandas as pd
import json
import pandas as pd


import json
import pandas as pd
import numpy as np
from tqdm import tqdm


def load_blood_glucose_json(path: str) -> dict:
    """
    Load AI-READI CGM JSON (Dexcom-like) and return a dictionary:

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

    records = body.get("cgm", [])
    if not records:
        return {}

    rows = []

    for r in records:
        time_interval = r.get("effective_time_frame", {}).get("time_interval", {})
        t = time_interval.get("start_date_time", None)

        bg = r.get("blood_glucose", {})
        glucose_val = bg.get("value", None)
        glucose_unit = bg.get("unit", None)

        transmitter_time_val = r.get("transmitter_time", {}).get("value", None)

        rows.append(
            {
                "time": t,
                "glucose": glucose_val,
                "unit": glucose_unit,
                "event_type": r.get("event_type", None),
                "source_device_id": r.get("source_device_id", None),
                "transmitter_id": r.get("transmitter_id", None),
                "transmitter_time": transmitter_time_val,
                "patient_id": patient_id,
            }
        )

    df = pd.DataFrame(rows)

    # Parse datetime as UTC
    df["time_utc"] = pd.to_datetime(df["time"], format="ISO8601", utc=True, errors="coerce")
    df = df.drop(columns=["time"])

    # Ensure glucose numeric
    df["glucose"] = pd.to_numeric(df["glucose"], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=["time_utc", "glucose"])

    # Sort and drop duplicates
    df = (
        df.sort_values("time_utc")
        .drop_duplicates(subset=["time_utc"], keep="last")
        .reset_index(drop=True)
    )

    # Convert timezone if available
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

    # Fix string columns
    for col in ["unit", "event_type", "source_device_id", "transmitter_id", "patient_id"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # transmitter_time -> nullable integer
    if "transmitter_time" in df.columns:
        df["transmitter_time"] = (
            pd.to_numeric(df["transmitter_time"], errors="coerce").astype("Int64")
        )

    # ===== convert dataframe -> dict[column -> np.array] =====
    result = {col: df[col].to_numpy() for col in df.columns}

    return result


def save_to_parquet(split_ids, save_path):
    result_list = []
    for split_id in tqdm(split_ids):
        blood_glucose_file_path = f"/playpen-shared/mshuang/morris/morris/d9ef6cf1-f6c3-4956-a91e-adf409e105f0/dataset/wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6/{split_id}/{split_id}_DEX.json"
        glucose = load_blood_glucose_json(blood_glucose_file_path)
        result_list.append(glucose)
    df = pd.DataFrame(result_list)
    # df.to_parquet("/playpen/haochenz/AI-READI/glucose_train.parquet")
    df.to_parquet(save_path)

if __name__ == "__main__":
    participants_path = "participants.tsv"

    df = pd.read_csv(participants_path, sep="\t")

    # 确保类型统一
    df["person_id"] = df["person_id"].astype(str)

    train_ids = df.loc[df["recommended_split"] == "train", "person_id"].tolist()
    val_ids   = df.loc[df["recommended_split"] == "val", "person_id"].tolist()
    test_ids  = df.loc[df["recommended_split"] == "test", "person_id"].tolist()

    print("train:", len(train_ids))
    print("val:", len(val_ids))
    print("test:", len(test_ids))

    save_to_parquet(train_ids, "/playpen/haochenz/AI-READI/glucose_train.parquet")
    save_to_parquet(val_ids, "/playpen/haochenz/AI-READI/glucose_valid.parquet")
    save_to_parquet(test_ids, "/playpen/haochenz/AI-READI/glucose_test.parquet")





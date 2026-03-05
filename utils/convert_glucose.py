import os
import pandas as pd
from tqdm import tqdm
import json


def load_blood_glucose_json(path: str) -> pd.DataFrame:
    """
    Load AI-READI CGM JSON (Dexcom-like) and return a clean pandas DataFrame.

    Returns df with columns:
        - time_utc (datetime64[ns, UTC])
        - time_local (datetime64[ns, tz], optional)
        - glucose (float)
        - unit (str)
        - event_type (str)
        - source_device_id (str)
        - transmitter_id (str)
        - transmitter_time (int)
        - patient_id (str)
    """
    with open(path, "r") as f:
        data = json.load(f)

    header = data.get("header", {})
    body = data.get("body", {})

    patient_id = header.get("patient_id", header.get("uuid", None))
    timezone = header.get("timezone", None)

    records = body.get("cgm", [])
    if len(records) == 0:
        return pd.DataFrame()

    rows = []
    for r in records:
        time_interval = r.get("effective_time_frame", {}).get("time_interval", {})
        t = time_interval.get("start_date_time", None)

        bg = r.get("blood_glucose", {})
        glucose_val = bg.get("value", None)
        glucose_unit = bg.get("unit", None)

        transmitter_time_val = r.get("transmitter_time", {}).get("value", None)

        rows.append({
            "time": t,
            "glucose": glucose_val,
            "unit": glucose_unit,
            "event_type": r.get("event_type", None),
            "source_device_id": r.get("source_device_id", None),
            "transmitter_id": r.get("transmitter_id", None),
            "transmitter_time": transmitter_time_val,
            "patient_id": patient_id,
        })

    df = pd.DataFrame(rows)

    # parse datetime as UTC
    df["time_utc"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.drop(columns=["time"])

    # drop invalid rows
    df = df.dropna(subset=["time_utc", "glucose"])

    # ensure glucose numeric
    df["glucose"] = pd.to_numeric(df["glucose"], errors="coerce")
    df = df.dropna(subset=["glucose"])

    # sort and drop duplicates
    df = df.sort_values("time_utc").reset_index(drop=True)
    df = df.drop_duplicates(subset=["time_utc"], keep="last").reset_index(drop=True)

    # convert timezone if available
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
        tz = tz_map.get(timezone.lower(), None)
        if tz is not None:
            df["time_local"] = df["time_utc"].dt.tz_convert(tz)

    return df



input_dir = "/playpen-shared/mshuang/morris/morris/d9ef6cf1-f6c3-4956-a91e-adf409e105f0/dataset/wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6"
output_path = "/playpen/haochenz/AI-READI/all_glucose.parquet"

dfs = []

for root, dirs, files in os.walk(input_dir):
    for file in tqdm(files):
        if file.endswith(".json"):
            json_path = os.path.join(root, file)

            try:
                df = load_blood_glucose_json(json_path)

                if len(df) == 0:
                    continue

                dfs.append(df)

            except Exception as e:
                print(f"Error processing {json_path}: {e}")

# 合并
all_df = pd.concat(dfs, ignore_index=True)

# 保存 parquet
all_df.to_parquet(output_path, index=False)

print("Saved:", output_path)
print("Total rows:", len(all_df))
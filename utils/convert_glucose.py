import os
import json
import pandas as pd
from tqdm import tqdm


def load_blood_glucose_json(path: str) -> pd.DataFrame:
    """
    Load AI-READI CGM JSON (Dexcom-like) and return a clean pandas DataFrame.

    Returns df with columns:
        - time_utc (datetime64[ns, UTC])
        - time_local (datetime64[ns, tz], optional)
        - glucose (float)
        - unit (string)
        - event_type (string)
        - source_device_id (string)
        - transmitter_id (string)
        - transmitter_time (Int64)
        - patient_id (string)
    """
    with open(path, "r") as f:
        data = json.load(f)

    header = data.get("header", {})
    body = data.get("body", {})

    patient_id = header.get("patient_id", header.get("uuid", None))
    timezone = header.get("timezone", None)

    records = body.get("cgm", [])
    if not records:
        return pd.DataFrame()

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

    # Parse datetime as UTC (ISO8601 is typical for Open mHealth timestamps)
    df["time_utc"] = pd.to_datetime(df["time"], format="ISO8601", utc=True, errors="coerce")
    df = df.drop(columns=["time"])

    # Ensure glucose numeric
    df["glucose"] = pd.to_numeric(df["glucose"], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=["time_utc", "glucose"])

    # Sort and drop duplicates (keep last)
    df = df.sort_values("time_utc").drop_duplicates(subset=["time_utc"], keep="last").reset_index(drop=True)

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

    # ---- FIX FOR PARQUET (pyarrow) ----
    # Force id-like / categorical columns to pandas "string" dtype to avoid mixed object types
    for col in ["unit", "event_type", "source_device_id", "transmitter_id", "patient_id"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # transmitter_time: keep as nullable integer if possible
    if "transmitter_time" in df.columns:
        df["transmitter_time"] = pd.to_numeric(df["transmitter_time"], errors="coerce").astype("Int64")

    return df


input_dir = "/playpen-shared/mshuang/morris/morris/d9ef6cf1-f6c3-4956-a91e-adf409e105f0/dataset/wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6"
output_path = "/playpen/haochenz/AI-READI/all_glucose.parquet"

dfs = []
bad_files = 0

for root, _, files in os.walk(input_dir):
    for file in tqdm(files):
        if file.endswith(".json"):
            json_path = os.path.join(root, file)
            try:
                df = load_blood_glucose_json(json_path)
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                bad_files += 1
                print(f"[ERROR] {json_path}: {e}")

if len(dfs) == 0:
    raise RuntimeError("No valid CGM JSON files were loaded. Check input_dir.")

all_df = pd.concat(dfs, ignore_index=True)

# Final safety: enforce string dtype again (in case concat introduced object)
for col in ["unit", "event_type", "source_device_id", "transmitter_id", "patient_id"]:
    if col in all_df.columns:
        all_df[col] = all_df[col].astype("string")

# Save parquet (pyarrow engine is default if installed)
all_df.to_parquet(output_path, index=False)

print("Saved:", output_path)
print("Total rows:", len(all_df))
print("Bad files:", bad_files)


# ================================
# Load participants.tsv for split
# ================================
participants_path = "/playpen-shared/mshuang/morris/morris/d9ef6cf1-f6c3-4956-a91e-adf409e105f0/dataset/participants.tsv"
participants = pd.read_csv(participants_path, sep="\t")

# 确保类型一致
participants["person_id"] = participants["person_id"].astype("string")
all_df["patient_id"] = all_df["patient_id"].astype("string")

# 建立 patient → split 映射
split_map = dict(
    zip(participants["person_id"], participants["recommended_split"])
)

# 添加 split 列
all_df["split"] = all_df["patient_id"].map(split_map)

# 检查有没有 mapping 不到的
missing = all_df["split"].isna().sum()
print("Rows with missing split:", missing)


train_df = all_df[all_df["split"] == "train"]
val_df = all_df[all_df["split"] == "val"]
test_df = all_df[all_df["split"] == "test"]

print("Train rows:", len(train_df))
print("Val rows:", len(val_df))
print("Test rows:", len(test_df))


train_path = "/playpen/haochenz/AI-READI/glucose_train.parquet"
val_path = "/playpen/haochenz/AI-READI/glucose_val.parquet"
test_path = "/playpen/haochenz/AI-READI/glucose_test.parquet"

train_df.to_parquet(train_path, index=False)
val_df.to_parquet(val_path, index=False)
test_df.to_parquet(test_path, index=False)

print("Saved:")
print(train_path)
print(val_path)
print(test_path)

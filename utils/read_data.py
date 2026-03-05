import json
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import os
import numpy as np

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



def load_ecg_wfdb(record_path: str, try_ann_extensions=None):
    """
    Load ECG record from WFDB (.hea + .dat) and optionally load annotations (events).

    Parameters
    ----------
    record_path : str
        Path to record without extension, e.g.
        "/path/to/1001_ecg_25aafb4b" (no .hea/.dat)
    try_ann_extensions : list[str]
        Annotation file extensions to try, e.g. ["atr", "ann", "qrs"].
        If None, will try a default list.

    Returns
    -------
    result : dict
        {
            "signal": np.ndarray shape (N, C),
            "fs": float,
            "sig_names": list[str],
            "units": list[str],
            "events": pd.DataFrame (may be empty),
            "record": wfdb.Record
        }
    """

    if try_ann_extensions is None:
        try_ann_extensions = ["atr", "ann", "qrs", "ecg", "evt"]

    # ----------------------
    # Load ECG waveform
    # ----------------------
    record = wfdb.rdrecord(record_path)

    signal = record.p_signal  # float array (N, C)
    fs = record.fs
    sig_names = record.sig_name
    units = record.units

    # ----------------------
    # Try to load annotation
    # ----------------------
    events = []
    ann_loaded = False

    for ext in try_ann_extensions:
        ann_file = record_path + "." + ext
        if os.path.exists(ann_file):
            try:
                ann = wfdb.rdann(record_path, ext)
                ann_loaded = True

                for sample, symbol in zip(ann.sample, ann.symbol):
                    events.append({
                        "sample": int(sample),
                        "time_sec": float(sample) / fs,
                        "symbol": symbol,
                        "extension": ext
                    })

            except Exception as e:
                # ignore if cannot read
                continue

    events_df = pd.DataFrame(events)

    # sort events
    if len(events_df) > 0:
        events_df = events_df.sort_values("sample").reset_index(drop=True)

    return {
        "signal": signal,
        "fs": fs,
        "sig_names": sig_names,
        "units": units,
        "events": events_df,
        "record": record,
        "ann_loaded": ann_loaded,
    }



def plot_blood_glucose_timeseries(df, use_local_time=True):
    if use_local_time and "time_local" in df.columns:
        t = df["time_local"]
        xlabel = "Time (Local)"
    else:
        t = df["time_utc"]
        xlabel = "Time (UTC)"

    plt.figure(figsize=(14, 4))
    plt.plot(t, df["glucose"], linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel("Glucose (mg/dL)")
    plt.title("CGM Blood Glucose Time Series")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()




def plot_ecg_12lead_with_events(ecg_dict, t_start=0, t_end=10, sharex=True):
    """
    Plot multi-lead ECG (typically 12 leads) with event markers.

    Parameters
    ----------
    ecg_dict : dict
        output from load_ecg_wfdb()
    t_start : float
        start time in seconds
    t_end : float
        end time in seconds
    sharex : bool
        whether to share x-axis among subplots
    """

    signal = ecg_dict["signal"]
    fs = ecg_dict["fs"]
    events = ecg_dict["events"]
    sig_names = ecg_dict["sig_names"]

    n_channels = signal.shape[1]

    start_idx = int(t_start * fs)
    end_idx = int(t_end * fs)

    seg = signal[start_idx:end_idx, :]  # (T, C)
    t = np.arange(start_idx, end_idx) / fs

    # events in window
    if len(events) > 0:
        sub_events = events[(events["time_sec"] >= t_start) & (events["time_sec"] <= t_end)]
    else:
        sub_events = None

    # plot
    fig, axes = plt.subplots(
        n_channels, 1,
        figsize=(16, 2 * n_channels),
        sharex=sharex
    )

    if n_channels == 1:
        axes = [axes]

    for ch in range(n_channels):
        ax = axes[ch]
        ax.plot(t, seg[:, ch], linewidth=1)

        # event markers
        if sub_events is not None and len(sub_events) > 0:
            for _, row in sub_events.iterrows():
                ax.axvline(row["time_sec"], linestyle="--", linewidth=0.8, alpha=0.7)

                # annotate event symbol on top
                ax.text(
                    row["time_sec"],
                    np.max(seg[:, ch]),
                    row["symbol"],
                    rotation=90,
                    verticalalignment="bottom",
                    fontsize=8,
                    alpha=0.9
                )

        lead_name = sig_names[ch] if ch < len(sig_names) else f"ch{ch}"
        ax.set_ylabel(lead_name, rotation=0, labelpad=30, fontsize=10)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time (sec)")
    fig.suptitle(f"Multi-lead ECG ({t_start:.1f}s - {t_end:.1f}s)", fontsize=14)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    blood_glucose_file_path = "/Users/zhc/Downloads/AI-READI/wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6/1001/1001_DEX.json"
    df = load_blood_glucose_json(blood_glucose_file_path)
    plot_blood_glucose_timeseries(df, use_local_time=True)
    # # print(df.head())
    # # print(df.tail())
    #
    # ecg_file_path = "/Users/zhc/Downloads/AI-READI/cardiac_ecg/ecg_12lead/philips_tc30/1001/1001_ecg_25aafb4b"
    # ecg_dict = load_ecg_wfdb(ecg_file_path)
    # 
    # plot_ecg_12lead_with_events(ecg_dict, t_start=0, t_end=10, sharex=True)


    # df = pd.read_csv(
    #     "/Users/zhc/Downloads/AI-READI/participants.tsv",
    #     sep="\t")

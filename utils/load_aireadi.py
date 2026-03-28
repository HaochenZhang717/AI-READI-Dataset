#!/usr/bin/env python3
"""
Minimal example: Loading AI-READI dataset modalities.

Demonstrates how to load each data modality:
  1. Participants metadata  (participants.tsv)
  2. CGM glucose readings   (Dexcom G6 JSON)
  3. Clinical data           (OMOP CSVs: person, measurement, condition)
  4. ECG 12-lead waveforms   (WFDB .hea/.dat)
  5. Wearable activity       (Garmin JSON: heart rate, SpO2, stress, etc.)
  6. Retinal imaging         (DICOM: CFP, FAF, IR, OCT, OCTA, FLIO)
  7. Anomaly labels          (pre-computed CSV from ych/ pipeline)

Usage:
    python load_aireadi.py
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Root of the AI-READI dataset ─────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent

# =============================================================================
# 1. Participants metadata
# =============================================================================
def load_participants() -> pd.DataFrame:
    """Load participants.tsv with demographics and modality availability flags."""
    df = pd.read_csv(DATA_DIR / "participants.tsv", sep="\t")
    # Key columns: person_id, age, study_group, recommended_split,
    #   gender, race, ethnicity, cardiac_ecg, wearable_blood_glucose, ...
    return df


# =============================================================================
# 2. CGM (Continuous Glucose Monitoring)
# =============================================================================
def load_cgm(person_id: int) -> list[tuple[datetime, float]]:
    """Load CGM readings for one participant.

    Returns list of (timestamp, glucose_mg_dl) sorted by time.
    Data path: wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6/{pid}/{pid}_DEX.json
    """
    cgm_file = (DATA_DIR / "wearable_blood_glucose" / "continuous_glucose_monitoring"
                / "dexcom_g6" / str(person_id) / f"{person_id}_DEX.json")
    if not cgm_file.exists():
        return []

    with open(cgm_file) as f:
        data = json.load(f)

    readings = []
    for entry in data["body"]["cgm"]:
        val = entry.get("blood_glucose", {}).get("value")
        if val is None or val == "Low":
            continue
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue

        tf = entry.get("effective_time_frame", {})
        ti = tf.get("time_interval", {})
        ts_str = ti.get("start_date_time") or tf.get("date_time")
        if ts_str:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            readings.append((ts, val))

    readings.sort(key=lambda x: x[0])
    return readings


# =============================================================================
# 3. Clinical data (OMOP Common Data Model)
# =============================================================================
def load_person() -> pd.DataFrame:
    """Load person.csv — demographics (gender, race, ethnicity, year_of_birth)."""
    return pd.read_csv(DATA_DIR / "clinical_data" / "person.csv")


def load_measurements() -> pd.DataFrame:
    """Load measurement.csv — lab values, vitals, cognitive scores."""
    return pd.read_csv(DATA_DIR / "clinical_data" / "measurement.csv", low_memory=False)


def load_conditions() -> pd.DataFrame:
    """Load condition_occurrence.csv — diagnosed conditions."""
    return pd.read_csv(DATA_DIR / "clinical_data" / "condition_occurrence.csv", low_memory=False)


def get_hba1c(person_id: int, meas_df: pd.DataFrame = None) -> float | None:
    """Get HbA1c value for one participant."""
    if meas_df is None:
        meas_df = load_measurements()
    subset = meas_df[
        (meas_df["person_id"] == person_id)
        & (meas_df["measurement_source_value"].str.startswith("import_hba1c", na=False))
    ]
    if len(subset) == 0:
        return None
    return float(subset.iloc[-1]["value_as_number"])


# =============================================================================
# 4. ECG 12-lead waveforms (WFDB format)
# =============================================================================
def load_ecg(person_id: int, target_sr: int = 100, seq_len: int = 1000) -> np.ndarray | None:
    """Load 12-lead ECG, downsample to target_sr, return (seq_len, 12) array.

    Requires: pip install wfdb
    Data path: cardiac_ecg/ecg_12lead/philips_tc30/{pid}/{pid}_ecg_*.hea
    """
    import glob as _glob
    try:
        import wfdb
    except ImportError:
        raise ImportError("Install wfdb: pip install wfdb")

    from scipy.signal import decimate

    ecg_dir = DATA_DIR / "cardiac_ecg" / "ecg_12lead" / "philips_tc30" / str(person_id)
    if not ecg_dir.is_dir():
        return None

    hea_files = _glob.glob(str(ecg_dir / f"{person_id}_ecg_*.hea"))
    if not hea_files:
        return None

    rec_path = hea_files[0].replace(".hea", "")
    record = wfdb.rdrecord(rec_path)
    signal = record.p_signal  # (samples, 12) at 500 Hz
    if signal is None or signal.shape[1] != 12:
        return None

    orig_sr = 500
    factor = orig_sr // target_sr
    downsampled = decimate(signal, factor, axis=0)

    if downsampled.shape[0] >= seq_len:
        return downsampled[:seq_len].astype(np.float32)
    else:
        pad = np.zeros((seq_len - downsampled.shape[0], 12), dtype=np.float32)
        return np.concatenate([downsampled, pad], axis=0).astype(np.float32)


# =============================================================================
# 5. Wearable activity monitor (Garmin Vivosmart 5)
# =============================================================================
WEARABLE_SUBMODALITIES = {
    "heart_rate":        ("heart_rate",        "heart_rate",        "value"),
    "oxygen_saturation": ("oxygen_saturation", "oxygen_saturation", "value"),
    "stress":            ("stress",            "stress_level",      "value"),
    "respiratory_rate":  ("respiratory_rate",  "respiratory_rate",  "value"),
}


def load_wearable_submodality(person_id: int, submodality: str) -> np.ndarray | None:
    """Load one wearable sub-modality for a participant.

    Data path: wearable_activity_monitor/{submodality}/garmin_vivosmart5/{pid}/*.json
    """
    body_key, item_key, value_key = WEARABLE_SUBMODALITIES[submodality]
    sub_dir = (DATA_DIR / "wearable_activity_monitor" / submodality
               / "garmin_vivosmart5" / str(person_id))
    if not sub_dir.is_dir():
        return None

    values = []
    for jf in sub_dir.glob("*.json"):
        with open(jf) as f:
            data = json.load(f)
        for r in data.get("body", {}).get(body_key, []):
            val = r.get(item_key, {}).get(value_key)
            if val is not None:
                v = float(val)
                if v > 0:
                    values.append(v)

    return np.array(values, dtype=np.float32) if values else None


# =============================================================================
# 6. Retinal imaging (DICOM)
#    All retinal images are stored as DICOM (.dcm) files.
#    Requires: pip install pydicom
#    Modalities:
#      - retinal_photography/cfp/   — Color Fundus Photography (devices: icare_eidon, optomed_aurora, topcon_maestro2, topcon_triton)
#      - retinal_photography/faf/   — Fundus Autofluorescence (device: icare_eidon)
#      - retinal_photography/ir/    — Infrared Reflectance (device: heidelberg_spectralis)
#      - retinal_oct/structural_oct — OCT volume scans (devices: heidelberg_spectralis, topcon_maestro2, topcon_triton, zeiss_cirrus)
#      - retinal_octa/enface/       — OCTA en-face images
#      - retinal_octa/flow_cube/    — OCTA flow cube volumes
#      - retinal_octa/segmentation/ — OCTA segmentation maps
#      - retinal_flio/flio/         — Fluorescence Lifetime Imaging (device: heidelberg_flio)
# =============================================================================

def load_retinal_manifest(modality: str) -> pd.DataFrame:
    """Load the manifest.tsv for a retinal modality.

    Args:
        modality: One of 'retinal_photography', 'retinal_oct', 'retinal_octa', 'retinal_flio'

    Returns DataFrame with columns including person_id, manufacturer, laterality, filepath, etc.
    """
    return pd.read_csv(DATA_DIR / modality / "manifest.tsv", sep="\t")


def load_dicom_image(filepath: str) -> np.ndarray:
    """Load a single DICOM file and return the pixel array.

    Requires: pip install pydicom
    Args:
        filepath: Relative path from manifest (e.g., /retinal_photography/cfp/icare_eidon/1001/...)
                  or absolute path.
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError("Install pydicom: pip install pydicom")

    if not Path(filepath).is_absolute():
        filepath = str(DATA_DIR) + filepath

    dcm = pydicom.dcmread(filepath)
    return dcm.pixel_array


def find_retinal_files(person_id: int, modality: str, submodality: str = None,
                       device: str = None) -> list[Path]:
    """Find all retinal DICOM files for a participant.

    Args:
        person_id:   Participant ID
        modality:    'retinal_photography', 'retinal_oct', 'retinal_octa', 'retinal_flio'
        submodality: Sub-folder (e.g., 'cfp', 'faf', 'ir', 'structural_oct', 'enface', 'flio')
        device:      Device folder (e.g., 'icare_eidon', 'heidelberg_spectralis')

    Returns list of .dcm file paths.
    """
    base = DATA_DIR / modality
    if submodality:
        base = base / submodality
    if device:
        base = base / device

    pid_dir = base / str(person_id)
    if not pid_dir.is_dir():
        return []
    return sorted(pid_dir.glob("*.dcm"))


def load_cfp(person_id: int, device: str = "icare_eidon") -> list[np.ndarray]:
    """Load all Color Fundus Photography images for a participant.

    Returns list of pixel arrays (typically RGB, various resolutions).
    """
    files = find_retinal_files(person_id, "retinal_photography", "cfp", device)
    images = []
    for f in files:
        try:
            images.append(load_dicom_image(str(f)))
        except Exception:
            pass
    return images


def load_oct_volume(person_id: int, device: str = "heidelberg_spectralis") -> list[np.ndarray]:
    """Load structural OCT volume(s) for a participant.

    Each volume is a 3D array (num_frames, height, width).
    """
    files = find_retinal_files(person_id, "retinal_oct", "structural_oct", device)
    volumes = []
    for f in files:
        try:
            volumes.append(load_dicom_image(str(f)))
        except Exception:
            pass
    return volumes


def load_octa_enface(person_id: int, device: str = "topcon_maestro2") -> list[np.ndarray]:
    """Load OCTA en-face images for a participant."""
    files = find_retinal_files(person_id, "retinal_octa", "enface", device)
    images = []
    for f in files:
        try:
            images.append(load_dicom_image(str(f)))
        except Exception:
            pass
    return images


def load_flio(person_id: int) -> list[np.ndarray]:
    """Load FLIO (Fluorescence Lifetime Imaging) volumes for a participant.

    Each volume is (num_frames, 256, 256) — short or long wavelength.
    """
    files = find_retinal_files(person_id, "retinal_flio", "flio", "heidelberg_flio")
    volumes = []
    for f in files:
        try:
            volumes.append(load_dicom_image(str(f)))
        except Exception:
            pass
    return volumes


# =============================================================================
# 7. Anomaly labels (from ych/ signal anomaly detection pipeline)
# =============================================================================
def load_anomaly_window_labels() -> pd.DataFrame:
    """Load per-window CGM anomaly labels (189,290 rows).

    Columns: person_id, window_idx, start_ts, end_ts,
             anomaly_score, z_score, iso_score, anomaly_label (0/1)
    """
    path = DATA_DIR / "ych" / "anomaly_labels.csv"
    df = pd.read_csv(path)
    df["start_ts"] = pd.to_datetime(df["start_ts"])
    df["end_ts"] = pd.to_datetime(df["end_ts"])
    return df


def load_anomaly_participant_scores() -> pd.DataFrame:
    """Load per-participant anomaly summary scores (2,242 rows).

    Columns: person_id, study_group, group_short,
             mean/max/std_anomaly_score, n_windows,
             n_anomalous_windows, pct_anomalous_windows, hba1c, bmi
    """
    return pd.read_csv(DATA_DIR / "ych" / "anomaly_participant_scores.csv")


# =============================================================================
# Demo
# =============================================================================
if __name__ == "__main__":
    # --- 1. Participants ---
    participants = load_participants()
    print(f"Participants: {len(participants)}")
    print(f"  Study groups: {dict(participants['study_group'].value_counts())}")
    print(f"  Splits:       {dict(participants['recommended_split'].value_counts())}")

    # Pick one participant for demo
    pid = int(participants.iloc[0]["person_id"])
    print(f"\nDemo participant: {pid}")

    # --- 2. CGM ---
    cgm = load_cgm(pid)
    print(f"\nCGM readings: {len(cgm)}")
    if cgm:
        print(f"  First: {cgm[0][0]}  glucose={cgm[0][1]} mg/dL")
        print(f"  Last:  {cgm[-1][0]}  glucose={cgm[-1][1]} mg/dL")

    # --- 3. Clinical ---
    person_df = load_person()
    meas_df = load_measurements()
    cond_df = load_conditions()
    print(f"\nClinical data:")
    print(f"  Persons:      {len(person_df)}")
    print(f"  Measurements: {len(meas_df)}")
    print(f"  Conditions:   {len(cond_df)}")

    hba1c = get_hba1c(pid, meas_df)
    print(f"  HbA1c for {pid}: {hba1c}")

    # --- 4. ECG ---
    try:
        ecg = load_ecg(pid)
        if ecg is not None:
            print(f"\nECG shape: {ecg.shape}  (seq_len, 12 leads)")
        else:
            print(f"\nECG: not available for {pid}")
    except ImportError as e:
        print(f"\nECG: {e}")

    # --- 5. Wearable ---
    hr = load_wearable_submodality(pid, "heart_rate")
    if hr is not None:
        print(f"\nHeart rate: {len(hr)} readings, mean={hr.mean():.1f} bpm")
    else:
        print(f"\nHeart rate: not available for {pid}")

    # --- 6. Retinal imaging ---
    try:
        cfp_manifest = load_retinal_manifest("retinal_photography")
        oct_manifest = load_retinal_manifest("retinal_oct")
        print(f"\nRetinal imaging:")
        print(f"  Photography manifest: {len(cfp_manifest)} entries")
        print(f"  OCT manifest:         {len(oct_manifest)} entries")

        cfp_files = find_retinal_files(pid, "retinal_photography", "cfp", "icare_eidon")
        print(f"  CFP files for {pid}:  {len(cfp_files)}")

        cfp_images = load_cfp(pid)
        if cfp_images:
            print(f"  CFP image shape:     {cfp_images[0].shape}")

        oct_vols = load_oct_volume(pid)
        if oct_vols:
            print(f"  OCT volume shape:    {oct_vols[0].shape}  (frames, H, W)")

        flio_vols = load_flio(pid)
        if flio_vols:
            print(f"  FLIO volume shape:   {flio_vols[0].shape}")
    except ImportError as e:
        print(f"\nRetinal imaging: {e}")

    # --- 7. Anomaly labels ---
    window_labels = load_anomaly_window_labels()
    participant_scores = load_anomaly_participant_scores()
    print(f"\nAnomaly labels:")
    print(f"  Windows:      {len(window_labels)}  ({window_labels['anomaly_label'].sum()} anomalous)")
    print(f"  Participants: {len(participant_scores)}")

    pid_windows = window_labels[window_labels["person_id"] == pid]
    print(f"  Windows for {pid}: {len(pid_windows)} total, "
          f"{pid_windows['anomaly_label'].sum()} anomalous")
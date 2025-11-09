# data_preprocessing.py
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

# ----------------- CONFIG -----------------
INPUT_DIR = "output"            # parquet dataset root (partitioned by MMSI/Segment)
OUTPUT_FILE = "processed_data.npz"
RESAMPLE_INTERVAL = "1min"      # '30s', '1min', etc.
MAX_SEQ_LEN = 256               # time steps per sequence for the model
NUMERIC_COLS = ["Latitude", "Longitude", "SOG", "COG"]  # features we resample
MODEL_FEATURES = ["Latitude", "Longitude", "SOG"]       # features we feed the model
# ------------------------------------------

def load_all_parquet(root):
    """Read the whole partitioned parquet dataset into a single DataFrame."""
    df = pd.read_parquet(root)  # pyarrow will read all partitions and add columns
    # Basic hygiene
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"])
    # Make sure numeric columns are numeric
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Sort for safety
    df = df.sort_values(["MMSI", "Segment", "Timestamp"])
    print(f"✅ Loaded {len(df):,} rows from {df['MMSI'].nunique()} vessels, "
          f"{df.groupby(['MMSI','Segment'], observed=True).ngroups} segments.")
    return df

def resample_segment(seg: pd.DataFrame) -> pd.DataFrame:
    """
    Resample one (MMSI, Segment) to a fixed interval on numeric columns only.
    Adds back MMSI/Segment after resampling.
    """
    seg = seg.sort_values("Timestamp")
    seg = seg.set_index("Timestamp")

    # Work on numeric columns only to avoid the aggregation error
    num = seg[NUMERIC_COLS].copy()

    # Down/up-sample to fixed grid, then interpolate gaps inside the segment
    num = (
        num
        .resample(RESAMPLE_INTERVAL)
        .mean(numeric_only=True)  # <- IMPORTANT: only numeric columns
        .interpolate(method="time", limit_direction="both")
    )

    # Reattach IDs and Timestamp as column
    num = num.reset_index()  # brings Timestamp back as a column
    # Broadcast IDs
    mmsi = seg["MMSI"].iloc[0] if "MMSI" in seg.columns else seg.index.name
    seg_id = seg["Segment"].iloc[0] if "Segment" in seg.columns else None
    num["MMSI"] = mmsi
    num["Segment"] = seg_id

    return num

def segment_to_array(seg: pd.DataFrame) -> np.ndarray:
    """
    Convert a resampled segment to fixed-length array with selected features.
    Pads with zeros if shorter; truncates if longer.
    """
    arr = seg[MODEL_FEATURES].to_numpy()

    if len(arr) >= MAX_SEQ_LEN:
        arr = arr[:MAX_SEQ_LEN]
    else:
        pad = MAX_SEQ_LEN - len(arr)
        arr = np.pad(arr, ((0, pad), (0, 0)), mode="constant", constant_values=0.0)

    return arr

def main():
    df = load_all_parquet(INPUT_DIR)

    # Simple min-max normalization computed over the whole dataset (after cleaning)
    print("🔹 Normalizing features...")
    norm_df = df.copy()
    for c in MODEL_FEATURES:
        vmin, vmax = norm_df[c].min(), norm_df[c].max()
        # safeguard for constant columns
        if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
            norm_df[c] = 0.0
        else:
            norm_df[c] = (norm_df[c] - vmin) / (vmax - vmin)

    # Convert each (MMSI, Segment) to a fixed-length sequence
    groups = norm_df.groupby(["MMSI", "Segment"], observed=True)
    print("🔹 Converting segments to arrays...")
    X_list = []

    for (mmsi, seg_id), g in tqdm(groups, total=groups.ngroups):
        if g.empty:
            continue
        try:
            rs = resample_segment(g)
            arr = segment_to_array(rs)
            X_list.append(arr)
        except Exception as e:
            # Skip pathological segments but keep going
            # (you can log (mmsi, seg_id) here if you want)
            continue

    if not X_list:
        raise RuntimeError("No segments were converted. Check input data and config.")

    X = np.stack(X_list)  # shape: (num_segments, MAX_SEQ_LEN, len(MODEL_FEATURES))
    print(f"✅ Final dataset shape: {X.shape}  "
          f"(segments, timesteps={MAX_SEQ_LEN}, features={len(MODEL_FEATURES)})")

    np.savez_compressed(OUTPUT_FILE, X=X)
    print(f"💾 Saved preprocessed data to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

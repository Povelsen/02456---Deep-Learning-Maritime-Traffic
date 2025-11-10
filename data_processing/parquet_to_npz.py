"""
Parquet to NPZ Converter
=======================

Convert partitioned Parquet files to NPZ format for model training.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pyarrow.parquet as pq


def convert_parquet_to_npz(input_dir: str, output_file: str = "processed_data.npz",
                          resample_interval: str = "1min",
                          max_seq_len: int = 256,
                          numeric_cols: list = None,
                          model_features: list = None) -> str:
    """
    Convert Parquet dataset to NPZ format for model training.
    
    Args:
        input_dir: Input directory with partitioned Parquet files
        output_file: Output NPZ file path
        resample_interval: Resampling interval ('30s', '1min', etc.)
        max_seq_len: Maximum sequence length
        numeric_cols: Columns to resample
        model_features: Features to include in final dataset
        
    Returns:
        Path to output NPZ file
    """
    if numeric_cols is None:
        numeric_cols = ["Latitude", "Longitude", "SOG", "COG"]
    if model_features is None:
        model_features = ["Latitude", "Longitude", "SOG"]
    
    print(f"📊 Loading Parquet data from {input_dir}...")
    
    # Load all parquet files
    try:
        df = pq.read_table(input_dir).to_pandas()
        print(f"✅ Loaded {len(df):,} rows from {df['MMSI'].nunique()} vessels")
    except Exception as e:
        raise ValueError(f"Error reading parquet files: {e}")
    
    # Basic data cleaning
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"])
    
    # Ensure numeric columns are numeric
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Sort for safety
    df = df.sort_values(["MMSI", "Segment", "Timestamp"])
    
    # Store normalization stats
    norm_stats = {}
    print("🔹 Normalizing features...")
    
    for col in model_features:
        if col in df.columns:
            vmin, vmax = df[col].min(), df[col].max()
            norm_stats[f"{col}_min"] = vmin
            norm_stats[f"{col}_max"] = vmax
            
            # Safeguard for constant columns
            if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
                df[col] = 0.0
            else:
                df[col] = (df[col] - vmin) / (vmax - vmin)
    
    print(f"   Saved stats for: {list(norm_stats.keys())}")
    
    # Convert each (MMSI, Segment) to fixed-length sequences
    groups = df.groupby(["MMSI", "Segment"], observed=True)
    print("🔹 Converting segments to arrays...")
    
    X_list = []
    
    for (mmsi, seg_id), group in tqdm(groups, total=groups.ngroups):
        if group.empty:
            continue
            
        try:
            # Resample segment to fixed interval
            resampled = _resample_segment(group, resample_interval, numeric_cols)
            
            # Convert to fixed-length array
            arr = _segment_to_array(resampled, model_features, max_seq_len)
            X_list.append(arr)
            
        except Exception as e:
            # Skip problematic segments but continue
            print(f"   Warning: Skipping segment ({mmsi}, {seg_id}): {e}")
            continue
    
    if not X_list:
        raise RuntimeError("No segments were converted. Check input data and configuration.")
    
    # Stack all sequences
    X = np.stack(X_list)
    print(f"✅ Final dataset shape: {X.shape}")
    print(f"   (segments, timesteps={max_seq_len}, features={len(model_features)})")
    
    # Save to NPZ file
    np.savez_compressed(output_file, X=X, **norm_stats)
    print(f"💾 Saved preprocessed data to {output_file}")
    
    return output_file


def _resample_segment(seg: pd.DataFrame, resample_interval: str, numeric_cols: list) -> pd.DataFrame:
    """
    Resample a segment to fixed time intervals.
    
    Args:
        seg: Input segment DataFrame
        resample_interval: Resampling interval
        numeric_cols: Columns to resample
        
    Returns:
        Resampled DataFrame
    """
    seg = seg.sort_values("Timestamp").set_index("Timestamp")
    
    # Work on numeric columns only
    num = seg[numeric_cols].copy()
    
    # Resample and interpolate
    num = (
        num
        .resample(resample_interval)
        .mean(numeric_only=True)
        .interpolate(method="time", limit_direction="both")
    )
    
    # Reattach IDs
    num = num.reset_index()
    num["MMSI"] = seg["MMSI"].iloc[0]
    num["Segment"] = seg["Segment"].iloc[0]
    
    return num


def _segment_to_array(seg: pd.DataFrame, model_features: list, max_seq_len: int) -> np.ndarray:
    """
    Convert a resampled segment to fixed-length array.
    
    Args:
        seg: Resampled segment DataFrame
        model_features: Features to include
        max_seq_len: Maximum sequence length
        
    Returns:
        Fixed-length array
    """
    arr = seg[model_features].to_numpy()
    
    if len(arr) >= max_seq_len:
        arr = arr[:max_seq_len]
    else:
        pad = max_seq_len - len(arr)
        arr = np.pad(arr, ((0, pad), (0, 0)), mode="constant", constant_values=0.0)
    
    return arr


def batch_convert_parquet(parquet_dirs: list, output_base_dir: str) -> list:
    """
    Convert multiple Parquet directories to NPZ files.
    
    Args:
        parquet_dirs: List of Parquet directory paths
        output_base_dir: Base directory for NPZ output
        
    Returns:
        List of output NPZ file paths
    """
    npz_files = []
    
    for i, parquet_dir in enumerate(parquet_dirs):
        # Create output filename
        dir_name = os.path.basename(parquet_dir)
        output_file = os.path.join(output_base_dir, f"data_{i+1:03d}_{dir_name}.npz")
        
        try:
            convert_parquet_to_npz(parquet_dir, output_file)
            npz_files.append(output_file)
        except Exception as e:
            print(f"❌ Failed to process {parquet_dir}: {e}")
            continue
    
    return npz_files


if __name__ == "__main__":
    # Single directory processing
    input_dir = "output_parquet/"
    output_file = "processed_data.npz"
    
    convert_parquet_to_npz(input_dir, output_file)
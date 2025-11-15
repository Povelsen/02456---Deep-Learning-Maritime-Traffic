"""
CSV to Parquet Converter
=======================

Convert AIS CSV files to partitioned Parquet format for efficient storage and processing.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
import os


def convert_csv_to_parquet(file_path: str, out_path: str, bbox: list = None) -> str:
    """
    Convert AIS CSV file to partitioned Parquet format.
    
    Args:
        file_path: Path to input CSV file
        out_path: Output directory for parquet files
        bbox: Bounding box filter [north, west, south, east]
        
    Returns:
        Path to output directory
    """
    if bbox is None:
        bbox = [60, 0, 50, 20]  # Default: North Sea region
    
    print(f"🚢 Processing {file_path}...")
    
    # Define data types and columns
    dtypes = {
        "MMSI": "object",
        "SOG": float,
        "COG": float,
        "Longitude": float,
        "Latitude": float,
        "# Timestamp": "object",
        "Type of mobile": "object",
    }
    usecols = list(dtypes.keys())
    
    # Read CSV file
    try:
        df = pd.read_csv(file_path, usecols=usecols, dtype=dtypes)
        print(f"   Loaded {len(df):,} raw records")
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")
    
    # Apply bounding box filter
    north, west, south, east = bbox
    df = df[(df["Latitude"] <= north) & (df["Latitude"] >= south) & 
            (df["Longitude"] >= west) & (df["Longitude"] <= east)]
    print(f"   After bounding box filter: {len(df):,} records")
    
    # Filter by vessel type
    df = df[df["Type of mobile"].isin(["Class A", "Class B"])].drop(columns=["Type of mobile"])
    print(f"   After vessel type filter: {len(df):,} records")
    
    # Validate MMSI format
    df = df[df["MMSI"].str.len() == 9]  # Standard MMSI length
    df = df[df["MMSI"].str[:3].astype(int).between(200, 775)]  # Valid MID range
    print(f"   After MMSI validation: {len(df):,} records")
    
    # Convert timestamp
    df = df.rename(columns={"# Timestamp": "Timestamp"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    
    # Remove duplicates
    df = df.drop_duplicates(["Timestamp", "MMSI"], keep="first")
    print(f"   After deduplication: {len(df):,} records")
    
    def track_filter(g):
        """Filter vessel tracks based on quality criteria"""
        len_filt = len(g) > 256  # Minimum track length
        sog_filt = 1 <= g["SOG"].max() <= 50  # Speed range filter
        time_filt = (g["Timestamp"].max() - g["Timestamp"].min()).total_seconds() >= 60 * 60  # Minimum duration
        return len_filt and sog_filt and time_filt
    
    # Filter by track quality
    df = df.groupby("MMSI").filter(track_filter)
    df = df.sort_values(['MMSI', 'Timestamp'])
    print(f"   After track quality filter: {len(df):,} records")
    
    # Segment tracks by time gaps
    df['Segment'] = df.groupby('MMSI')['Timestamp'].transform(
        lambda x: (x.diff().dt.total_seconds().fillna(0) >= 15 * 60).cumsum())
    
    # Filter segments
    df = df.groupby(["MMSI", "Segment"]).filter(track_filter)
    df = df.reset_index(drop=True)
    print(f"   Final record count: {len(df):,}")
    
    # Convert SOG from knots to m/s
    knots_to_ms = 0.514444
    df["SOG"] = knots_to_ms * df["SOG"]
    
    # Create output directory
    os.makedirs(out_path, exist_ok=True)
    
    # Convert to PyArrow table and save as partitioned dataset
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=out_path,
        partition_cols=["MMSI", "Segment"]
    )
    
    print(f"✅ Done! Cleaned data saved to {out_path}")
    print(f"   Vessels: {df['MMSI'].nunique()}")
    print(f"   Segments: {df.groupby(['MMSI', 'Segment']).ngroups}")
    
    return out_path


def batch_convert_csv_files(csv_files: list, output_base_dir: str) -> list:
    """
    Convert multiple CSV files to parquet format.
    
    Args:
        csv_files: List of CSV file paths
        output_base_dir: Base directory for output
        
    Returns:
        List of output directories
    """
    output_dirs = []
    
    for i, csv_file in enumerate(csv_files):
        # Create subdirectory for each file
        file_name = os.path.basename(csv_file).replace('.csv', '')
        output_dir = os.path.join(output_base_dir, f"data_{i+1:03d}_{file_name}")
        
        try:
            convert_csv_to_parquet(csv_file, output_dir)
            output_dirs.append(output_dir)
        except Exception as e:
            print(f"❌ Failed to process {csv_file}: {e}")
            continue
    
    return output_dirs


if __name__ == "__main__":
    # Single file processing
    file_path = "aisdk-2025-02-12.csv"
    out_path = "output_parquet/"
    
    convert_csv_to_parquet(file_path, out_path)
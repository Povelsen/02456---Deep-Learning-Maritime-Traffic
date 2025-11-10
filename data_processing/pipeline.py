"""
Complete Data Processing Pipeline
=================================

End-to-end pipeline for processing AIS data from CSV to model-ready NPZ format.
"""

import os
import sys
import numpy as np
from typing import List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.csv_to_parquet import convert_csv_to_parquet, batch_convert_csv_files
from data_processing.parquet_to_npz import convert_parquet_to_npz, batch_convert_parquet


def run_complete_pipeline(csv_files: List[str], 
                         output_base_dir: str = "processed_data",
                         resample_interval: str = "1min",
                         max_seq_len: int = 256,
                         bbox: Optional[List[float]] = None) -> dict:
    """
    Run complete data processing pipeline from CSV to NPZ.
    
    Args:
        csv_files: List of CSV file paths to process
        output_base_dir: Base directory for all outputs
        resample_interval: Resampling interval for time series
        max_seq_len: Maximum sequence length for model input
        bbox: Bounding box filter [north, west, south, east]
        
    Returns:
        Dictionary with processing results
    """
    print("🚀 Starting Complete Data Processing Pipeline")
    print("=" * 60)
    
    # Create output directories
    parquet_dir = os.path.join(output_base_dir, "parquet")
    npz_dir = os.path.join(output_base_dir, "npz")
    
    os.makedirs(parquet_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)
    
    results = {
        'csv_files_processed': 0,
        'parquet_dirs_created': [],
        'npz_files_created': [],
        'errors': []
    }
    
    # Step 1: Convert CSV to Parquet
    print(f"\n📊 Step 1: Converting {len(csv_files)} CSV files to Parquet...")
    
    for i, csv_file in enumerate(csv_files):
        if not os.path.exists(csv_file):
            error_msg = f"CSV file not found: {csv_file}"
            print(f"❌ {error_msg}")
            results['errors'].append(error_msg)
            continue
        
        # Create subdirectory for this file
        file_name = os.path.basename(csv_file).replace('.csv', '')
        output_subdir = os.path.join(parquet_dir, f"{i+1:03d}_{file_name}")
        
        try:
            convert_csv_to_parquet(csv_file, output_subdir, bbox)
            results['parquet_dirs_created'].append(output_subdir)
            results['csv_files_processed'] += 1
            print(f"   ✅ Processed {csv_file} -> {output_subdir}")
        except Exception as e:
            error_msg = f"Failed to process {csv_file}: {str(e)}"
            print(f"❌ {error_msg}")
            results['errors'].append(error_msg)
            continue
    
    # Step 2: Convert Parquet to NPZ
    print(f"\n🔧 Step 2: Converting Parquet to NPZ format...")
    
    if results['parquet_dirs_created']:
        for parquet_dir in results['parquet_dirs_created']:
            dir_name = os.path.basename(parquet_dir)
            npz_file = os.path.join(npz_dir, f"{dir_name}.npz")
            
            try:
                convert_parquet_to_npz(
                    parquet_dir,
                    npz_file,
                    resample_interval=resample_interval,
                    max_seq_len=max_seq_len
                )
                results['npz_files_created'].append(npz_file)
                print(f"   ✅ Converted {parquet_dir} -> {npz_file}")
            except Exception as e:
                error_msg = f"Failed to convert {parquet_dir}: {str(e)}"
                print(f"❌ {error_msg}")
                results['errors'].append(error_msg)
                continue
    
    # Summary
    print(f"\n📋 Processing Summary:")
    print(f"   CSV files processed: {results['csv_files_processed']}/{len(csv_files)}")
    print(f"   Parquet directories: {len(results['parquet_dirs_created'])}")
    print(f"   NPZ files created: {len(results['npz_files_created'])}")
    print(f"   Errors encountered: {len(results['errors'])}")
    
    if results['errors']:
        print(f"\n⚠️  Errors:")
        for error in results['errors']:
            print(f"   - {error}")
    
    # Create consolidated dataset if multiple files
    if len(results['npz_files_created']) > 1:
        print(f"\n🔧 Step 3: Creating consolidated dataset...")
        consolidated_file = os.path.join(npz_dir, "consolidated_data.npz")
        
        try:
            _create_consolidated_dataset(results['npz_files_created'], consolidated_file)
            results['consolidated_file'] = consolidated_file
            print(f"   ✅ Consolidated dataset: {consolidated_file}")
        except Exception as e:
            error_msg = f"Failed to create consolidated dataset: {str(e)}"
            print(f"❌ {error_msg}")
            results['errors'].append(error_msg)
    
    print(f"\n🎉 Pipeline completed!")
    print(f"   Output directory: {output_base_dir}")
    
    return results


def _create_consolidated_dataset(npz_files: List[str], output_file: str) -> None:
    """
    Combine multiple NPZ files into a single consolidated dataset.
    
    Args:
        npz_files: List of NPZ file paths
        output_file: Output consolidated NPZ file path
    """
    all_data = []
    all_norm_stats = {}
    
    for i, npz_file in enumerate(npz_files):
        data = np.load(npz_file)
        X = data['X']
        all_data.append(X)
        
        # Store normalization stats with file index
        for key in data.files:
            if key != 'X':
                all_norm_stats[f"{key}_file{i+1}"] = data[key]
    
    # Combine all data
    consolidated_X = np.concatenate(all_data, axis=0)
    
    # Create new normalization stats (min/max across all files)
    consolidated_stats = {}
    for feature in ['Latitude', 'Longitude', 'SOG']:
        min_key = f"{feature}_min"
        max_key = f"{feature}_max"
        
        all_mins = [v for k, v in all_norm_stats.items() if k.endswith(f"{feature}_min")]
        all_maxs = [v for k, v in all_norm_stats.items() if k.endswith(f"{feature}_max")]
        
        if all_mins and all_maxs:
            consolidated_stats[min_key] = min(all_mins)
            consolidated_stats[max_key] = max(all_maxs)
    
    # Save consolidated dataset
    save_dict = {'X': consolidated_X, **consolidated_stats}
    np.savez_compressed(output_file, **save_dict)
    
    print(f"   Consolidated {len(npz_files)} files into {output_file}")
    print(f"   Total sequences: {len(consolidated_X):,}")


def main():
    """Example usage of the complete pipeline"""
    
    # Example: Process multiple CSV files
    csv_files = [
        "aisdk_2025-02-12.csv",
        "aisdk_2025-02-13.csv",
        "aisdk_2025-02-14.csv"
    ]
    
    # Danish AIS database region (approximate)
    denmark_bbox = [58.0, 7.5, 54.5, 15.5]
    
    results = run_complete_pipeline(
        csv_files=csv_files,
        output_base_dir="ais_data_processed",
        resample_interval="1min",
        max_seq_len=256,
        bbox=denmark_bbox
    )
    
    return results


if __name__ == "__main__":
    main()
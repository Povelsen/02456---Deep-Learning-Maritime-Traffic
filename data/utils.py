"""
Data Loading and Preparation Utilities
======================================

Functions for loading AIS trajectory data and preparing it for training.
"""

import numpy as np
import torch
from typing import Tuple, Dict

# Import Denmark coordinates
from utils.coordinates import get_denmark_coordinate_ranges


def load_and_prepare_data(data_path: str = "processed_data.npz") -> Tuple[np.ndarray, Dict]:
    """
    Load data and normalization stats.
    If data file not found, generates sample AIS trajectory data NEAR DENMARK.
    """
    try:
        data_file = np.load(data_path)
        X = data_file['X']
        # Load all other keys as norm_stats
        norm_stats = {k: data_file[k].item() for k in data_file.files if k != 'X'}
        print(f"✅ Loaded real data from {data_path}")
        print(f"   Data shape: {X.shape}")
        if norm_stats:
            print(f"   Loaded normalization stats: {list(norm_stats.keys())}")
        else:
            print("⚠️  Warning: No normalization stats found in .npz file. Maps may be inaccurate.")
        return X, norm_stats
    except FileNotFoundError:
        print("=" * 60)
        print("⚠️  WARNING: Data file not found!")
        print("   Generating SAMPLE data near DENMARK region...")
        print("   For real data, download from: https://web.ais.dk/aisdata/")
        print("=" * 60)
        
        # Generate sample AIS trajectory data in DENMARK region
        denmark = get_denmark_coordinate_ranges()
        np.random.seed(42)
        n_vessels = 100
        seq_length = 256
        features = 3  # [latitude, longitude, speed]
        
        sample_data = []
        
        for vessel_id in range(n_vessels):
            # Start position within Denmark region
            lat = np.random.uniform(denmark['lat_min'], denmark['lat_max'])
            lon = np.random.uniform(denmark['lon_min'], denmark['lon_max'])
            speed = np.random.uniform(5, 25)
            
            trajectory = []
            
            for t in range(seq_length):
                # Simulate realistic movement
                lat += np.random.normal(0, 0.0005) + 0.0001 * np.sin(t * 0.1)
                lon += np.random.normal(0, 0.0005) + 0.0001 * np.cos(t * 0.05)
                speed += np.random.normal(0, 0.2)
                speed = max(0, min(30, speed))
                
                # Keep within Denmark bounds (soft constraints)
                lat = np.clip(lat, denmark['lat_min'], denmark['lat_max'])
                lon = np.clip(lon, denmark['lon_min'], denmark['lon_max'])
                
                trajectory.append([lat, lon, speed])
            
            sample_data.append(trajectory)
        
        X_sample = np.array(sample_data)
        
        # Create normalization stats based on Denmark region
        norm_stats = {
            'Latitude_min': denmark['lat_min'],
            'Latitude_max': denmark['lat_max'],
            'Longitude_min': denmark['lon_min'],
            'Longitude_max': denmark['lon_max'],
            'SOG_min': 0,
            'SOG_max': 30
        }
        
        # Normalize sample data
        X_sample[:, :, 0] = (X_sample[:, :, 0] - denmark['lat_min']) / (denmark['lat_max'] - denmark['lat_min'])
        X_sample[:, :, 1] = (X_sample[:, :, 1] - denmark['lon_min']) / (denmark['lon_max'] - denmark['lon_min'])
        X_sample[:, :, 2] = (X_sample[:, :, 2] - 0) / (30 - 0)

        print(f"✅ Generated sample data shape: {X_sample.shape}")
        print(f"   Coordinate range: {denmark['lat_min']}°N - {denmark['lat_max']}°N, {denmark['lon_min']}°E - {denmark['lon_max']}°E")
        
        return X_sample, norm_stats
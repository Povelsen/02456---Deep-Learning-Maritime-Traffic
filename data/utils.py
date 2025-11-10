"""
Data Loading and Preparation Utilities
======================================

Functions for loading AIS trajectory data and preparing it for training.
"""

import numpy as np
import torch
from typing import Tuple, Dict


def load_and_prepare_data(data_path: str = "processed_data.npz") -> Tuple[np.ndarray, Dict]:
    """
    Load data and normalization stats.
    If data file not found, generates sample AIS trajectory data.
    """
    try:
        data_file = np.load(data_path)
        X = data_file['X']
        # Load all other keys as norm_stats
        norm_stats = {k: data_file[k].item() for k in data_file.files if k != 'X'}
        print(f"Loaded data shape: {X.shape}")
        if norm_stats:
            print(f"Loaded normalization stats: {list(norm_stats.keys())}")
        else:
            print("Warning: No normalization stats found in .npz file. Maps may be inaccurate.")
        return X, norm_stats
    except FileNotFoundError:
        print("Data file not found. Generating sample AIS trajectory data...")
        
        # Generate sample AIS trajectory data
        np.random.seed(42)
        n_vessels = 100
        seq_length = 256
        features = 3  # [latitude, longitude, speed]
        
        sample_data = []
        
        for vessel_id in range(n_vessels):
            lat = np.random.uniform(50, 60) # More realistic North Sea coords
            lon = np.random.uniform(0, 20)
            speed = np.random.uniform(5, 25)  
            
            trajectory = []
            
            for t in range(seq_length):
                lat += np.random.normal(0, 0.001) + 0.0001 * np.sin(t * 0.1)
                lon += np.random.normal(0, 0.001) + 0.0001 * np.cos(t * 0.05)
                speed += np.random.normal(0, 0.2)
                speed = max(0, min(30, speed))
                
                trajectory.append([lat, lon, speed])
            
            sample_data.append(trajectory)
        
        X_sample = np.array(sample_data)
        
        # Create dummy norm_stats for sample data
        norm_stats = {
            'Latitude_min': 50, 'Latitude_max': 60,
            'Longitude_min': 0, 'Longitude_max': 20,
            'SOG_min': 0, 'SOG_max': 30
        }
        
        # Normalize sample data
        X_sample[:, :, 0] = (X_sample[:, :, 0] - 50) / (60 - 50)
        X_sample[:, :, 1] = (X_sample[:, :, 1] - 0) / (20 - 0)
        X_sample[:, :, 2] = (X_sample[:, :, 2] - 0) / (30 - 0)

        print(f"Generated sample data shape: {X_sample.shape}")
        return X_sample, norm_stats
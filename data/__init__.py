"""
Data Processing and Dataset Module
==================================

This module handles data loading, preprocessing, and dataset creation
for vessel trajectory prediction.
"""

from .dataset import TrajectoryDataset
from .utils import load_and_prepare_data

__all__ = ['TrajectoryDataset', 'load_and_prepare_data']
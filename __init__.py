"""
Vessel Trajectory Prediction Package
====================================

A comprehensive PyTorch implementation for vessel trajectory prediction using
Transformer and LSTM neural networks with AIS data.

Modules:
--------
- models: Neural network architectures (Transformer, LSTM)
- data: Data processing and dataset handling
- training: Training utilities and model trainers
- visualization: Plotting and visualization tools
- utils: Utility functions and helpers
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = ""

from .models import *
from .data import *
from .training import *
from .visualization import *
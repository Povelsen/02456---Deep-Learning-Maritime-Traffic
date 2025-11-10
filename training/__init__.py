"""
Training Module
===============

Training utilities, model trainers, and comparison frameworks for
vessel trajectory prediction models.
"""

from .trainer import ModelTrainer
from .comparison import ModelComparisonFramework

__all__ = ['ModelTrainer', 'ModelComparisonFramework']
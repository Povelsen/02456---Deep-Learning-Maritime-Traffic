"""
Model Comparison Framework
==========================

Framework for comparing different trajectory prediction models and
generating performance summaries.
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import TrajectoryDataset
from training.trainer import ModelTrainer


class ModelComparisonFramework:
    """
    Framework for comparing different trajectory prediction models.
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.data_info = {}
        self.norm_stats = {}  # To store de-normalization stats
        
    def add_model(self, name: str, model: nn.Module):
        """Add a model to the comparison framework"""
        self.models[name] = model
        
    def prepare_data(self, data: np.ndarray, input_length: int = 200, 
                    output_length: int = 12) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray]:
        """Prepare data for training and evaluation"""
        
        X, y = [], []
        
        # Create sequences
        for trajectory in data:
            if len(trajectory) >= input_length + output_length:
                # Find non-zero start (if any padding)
                non_zero_idx = np.where(trajectory.any(axis=1))[0]
                if len(non_zero_idx) == 0:
                    continue  # Skip empty trajectories
                
                start_idx = non_zero_idx[0]
                end_idx = non_zero_idx[-1] + 1
                
                valid_len = end_idx - start_idx
                if valid_len < input_length + output_length:
                    continue  # Skip trajectories that are too short after padding removal

                # Create sequences from the valid part
                for i in range(start_idx, end_idx - (input_length + output_length) + 1):
                    X.append(trajectory[i : i + input_length])
                    y.append(trajectory[i + input_length : i + input_length + output_length])

        if not X:
            raise ValueError("No valid sequences created. Check input data and sequence lengths.")
            
        X = np.array(X)
        y = np.array(y)
        
        # Split data (Train/Val/Test)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        # Create datasets and data loaders
        train_dataset = TrajectoryDataset(X_train, y_train)
        val_dataset = TrajectoryDataset(X_val, y_val)
        test_dataset = TrajectoryDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        # Store data information
        self.data_info = {
            'total_sequences': len(X),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'input_length': input_length,
            'output_length': output_length,
            'features': data.shape[-1]
        }
        print("Data Info:", self.data_info)
        
        return train_loader, val_loader, test_loader, X_test, y_test
    
    def run_comparison(self, data: np.ndarray, norm_stats: Dict, output_dir: str, num_epochs: int = 30):
        """Run complete model comparison, with plotting"""
        
        self.norm_stats = norm_stats  # Save stats for visualization
        train_loader, val_loader, test_loader, X_test, y_test = self.prepare_data(data)
        
        # Train all models
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Training {name}...")
            print(f"{'='*60}")
            
            trainer = ModelTrainer(model)
            history = trainer.train(train_loader, val_loader, num_epochs=num_epochs)
            
            # Evaluate
            print(f"\nEvaluating {name}...")
            results = trainer.evaluate(X_test, y_test)
            
            self.results[name] = {
                'history': history,
                'metrics': results,
                'trainer': trainer,
                'model_info': {
                    'type': 'Transformer' if 'transformer' in name.lower() else 'LSTM',
                    'parameters': sum(p.numel() for p in model.parameters())
                }
            }
        
        return self.results
    
    def generate_performance_summary(self) -> pd.DataFrame:
        """Generate performance summary table"""
        summary_data = []
        
        for name, result in self.results.items():
            metrics = result['metrics']
            summary_data.append({
                'Model': name,
                'Type': result['model_info']['type'],
                'Parameters': f"{result['model_info']['parameters']:,}",
                'MSE': f"{metrics['mse']:.6f}",
                'MAE': f"{metrics['mae']:.6f}",
                'RMSE': f"{metrics['rmse']:.6f}"
            })
        
        return pd.DataFrame(summary_data)
    
    def save_results(self, output_dir: str = 'output'):
        """Save all results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Performance summary
        summary = self.generate_performance_summary()
        summary.to_csv(f'{output_dir}/pytorch_model_performance.csv', index=False)
        
        # Training history
        training_data = {}
        for name, result in self.results.items():
            training_data[name] = result['history']
        
        # Convert tensors to lists for JSON serialization
        for model_name in training_data:
            for key in training_data[model_name]:
                if isinstance(training_data[model_name][key], list):
                    training_data[model_name][key] = [float(x) for x in training_data[model_name][key]]
        
        with open(f'{output_dir}/pytorch_training_history.json', 'w') as f:
            json.dump(training_data, f, indent=2)
        
        # Data info
        with open(f'{output_dir}/pytorch_data_info.json', 'w') as f:
            json.dump(self.data_info, f, indent=2)
        
        # Model architectures
        model_architectures = {}
        for name, result in self.results.items():
            model_architectures[name] = {
                'architecture': str(result['trainer'].model),
                'parameters': result['model_info']['parameters']
            }
        
        with open(f'{output_dir}/pytorch_model_architectures.json', 'w') as f:
            json.dump(model_architectures, f, indent=2)
        
        print(f"Results saved to {output_dir}/")
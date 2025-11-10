#!/usr/bin/env python3
"""
Run Demo Script
===============

Comprehensive demonstration of the vessel trajectory prediction framework.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from models.transformer import TransformerTrajectoryPredictor
from models.lstm import LSTMTrajectoryPredictor
from data.dataset import TrajectoryDataset
from data.utils import load_and_prepare_data
from training.trainer import ModelTrainer


def create_demo_data():
    """Create demo trajectory data"""
    np.random.seed(42)
    
    # Create more realistic vessel trajectories
    n_vessels = 30
    seq_length = 256
    
    trajectories = []
    
    for vessel_id in range(n_vessels):
        # Different starting positions and patterns
        base_lat = np.random.uniform(50, 60)
        base_lon = np.random.uniform(0, 20)
        base_speed = np.random.uniform(8, 18)
        
        trajectory = []
        
        for t in range(seq_length):
            # Different movement patterns for each vessel
            if vessel_id % 3 == 0:
                # Linear movement
                lat = base_lat + 0.0005 * t + 0.001 * np.random.randn()
                lon = base_lon + 0.0003 * t + 0.001 * np.random.randn()
            elif vessel_id % 3 == 1:
                # Circular movement
                lat = base_lat + 0.05 * np.sin(t * 0.02) + 0.001 * np.random.randn()
                lon = base_lon + 0.05 * np.cos(t * 0.02) + 0.001 * np.random.randn()
            else:
                # Sinusoidal movement
                lat = base_lat + 0.03 * np.sin(t * 0.05 + vessel_id) + 0.001 * np.random.randn()
                lon = base_lon + 0.02 * np.cos(t * 0.03 + vessel_id) + 0.001 * np.random.randn()
            
            speed = base_speed + 3 * np.sin(t * 0.1) + 2 * np.random.randn()
            speed = max(0, min(25, speed))
            
            trajectory.append([lat, lon, speed])
        
        trajectories.append(trajectory)
    
    data = np.array(trajectories)
    
    # Normalize data
    norm_stats = {
        'Latitude_min': data[:, :, 0].min(),
        'Latitude_max': data[:, :, 0].max(),
        'Longitude_min': data[:, :, 1].min(),
        'Longitude_max': data[:, :, 1].max(),
        'SOG_min': data[:, :, 2].min(),
        'SOG_max': data[:, :, 2].max()
    }
    
    # Apply normalization
    for i, key in enumerate(['Latitude', 'Longitude', 'SOG']):
        min_val = norm_stats[f'{key}_min']
        max_val = norm_stats[f'{key}_max']
        if max_val > min_val:
            data[:, :, i] = (data[:, :, i] - min_val) / (max_val - min_val)
    
    return data, norm_stats


def run_comprehensive_demo():
    """Run comprehensive demonstration"""
    print("🚢 Comprehensive Vessel Trajectory Prediction Demo")
    print("=" * 70)
    
    # Create output directory
    output_dir = "demo_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create demo data
    print("\n📊 Creating demo data...")
    data, norm_stats = create_demo_data()
    print(f"   Created {data.shape[0]} trajectories with {data.shape[1]} time steps")
    print(f"   Features: Latitude, Longitude, Speed")
    
    # Prepare sequences
    X, y = [], []
    input_length = 200
    output_length = 12
    
    for trajectory in data:
        if len(trajectory) >= input_length + output_length:
            X.append(trajectory[:input_length])
            y.append(trajectory[input_length:input_length + output_length])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"   Prepared {len(X)} sequences: X={X.shape}, y={y.shape}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Create models
    print("\n🏗️  Creating advanced models...")
    
    transformer = TransformerTrajectoryPredictor(
        input_dim=3,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dff=128,
        prediction_horizon=12
    )
    
    lstm = LSTMTrajectoryPredictor(
        input_dim=3,
        hidden_dim=64,
        num_layers=2,
        prediction_horizon=12,
        dropout=0.1
    )
    
    transformer_params = sum(p.numel() for p in transformer.parameters())
    lstm_params = sum(p.numel() for p in lstm.parameters())
    
    print(f"   Transformer: {transformer_params:,} parameters")
    print(f"   LSTM: {lstm_params:,} parameters")
    
    # Create datasets and data loaders
    train_dataset = TrajectoryDataset(X_train, y_train)
    val_dataset = TrajectoryDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Train models
    print("\n🚀 Training models...")
    
    # Train Transformer
    print("   Training Transformer...")
    transformer_trainer = ModelTrainer(transformer)
    transformer_history = transformer_trainer.train(
        train_loader, val_loader, num_epochs=15, learning_rate=1e-3
    )
    
    # Train LSTM
    print("   Training LSTM...")
    lstm_trainer = ModelTrainer(lstm)
    lstm_history = lstm_trainer.train(
        train_loader, val_loader, num_epochs=15, learning_rate=1e-3
    )
    
    # Evaluate models
    print("\n📈 Evaluating models...")
    transformer_results = transformer_trainer.evaluate(X_test, y_test)
    lstm_results = lstm_trainer.evaluate(X_test, y_test)
    
    print(f"   Transformer - MSE: {transformer_results['mse']:.6f}, MAE: {transformer_results['mae']:.6f}, RMSE: {transformer_results['rmse']:.6f}")
    print(f"   LSTM - MSE: {lstm_results['mse']:.6f}, MAE: {lstm_results['mae']:.6f}, RMSE: {lstm_results['rmse']:.6f}")
    
    # Create comprehensive visualizations
    print("\n🎨 Creating comprehensive visualizations...")
    
    # Training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    epochs = range(1, len(transformer_history['train_loss']) + 1)
    
    axes[0, 0].plot(epochs, transformer_history['train_loss'], 'b-', label='Transformer Train', linewidth=2)
    axes[0, 0].plot(epochs, transformer_history['val_loss'], 'b--', label='Transformer Val', linewidth=2)
    axes[0, 0].plot(epochs, lstm_history['train_loss'], 'r-', label='LSTM Train', linewidth=2)
    axes[0, 0].plot(epochs, lstm_history['val_loss'], 'r--', label='LSTM Val', linewidth=2)
    axes[0, 0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE curves
    axes[0, 1].plot(epochs, transformer_history['train_mae'], 'b-', label='Transformer Train', linewidth=2)
    axes[0, 1].plot(epochs, transformer_history['val_mae'], 'b--', label='Transformer Val', linewidth=2)
    axes[0, 1].plot(epochs, lstm_history['train_mae'], 'r-', label='LSTM Train', linewidth=2)
    axes[0, 1].plot(epochs, lstm_history['val_mae'], 'r--', label='LSTM Val', linewidth=2)
    axes[0, 1].set_title('Training & Validation MAE', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Mean Absolute Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Model comparison
    models = ['Transformer', 'LSTM']
    mse_values = [transformer_results['mse'], lstm_results['mse']]
    mae_values = [transformer_results['mae'], lstm_results['mae']]
    rmse_values = [transformer_results['rmse'], lstm_results['rmse']]
    
    x_pos = np.arange(len(models))
    width = 0.25
    
    axes[1, 0].bar(x_pos - width, mse_values, width, label='MSE', alpha=0.8)
    axes[1, 0].bar(x_pos, mae_values, width, label='MAE', alpha=0.8)
    axes[1, 0].bar(x_pos + width, rmse_values, width, label='RMSE', alpha=0.8)
    axes[1, 0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('Error Values')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(models)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Test set error distribution
    transformer_errors = np.sqrt(np.mean((y_test - transformer_results['predictions'])**2, axis=(1, 2)))
    lstm_errors = np.sqrt(np.mean((y_test - lstm_results['predictions'])**2, axis=(1, 2)))
    
    axes[1, 1].hist(transformer_errors, bins=10, alpha=0.7, label='Transformer', color='blue')
    axes[1, 1].hist(lstm_errors, bins=10, alpha=0.7, label='LSTM', color='red')
    axes[1, 1].axvline(np.mean(transformer_errors), color='blue', linestyle='--', linewidth=2)
    axes[1, 1].axvline(np.mean(lstm_errors), color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_title('Test Set Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('RMSE per Sample')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create trajectory prediction visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i in range(min(6, len(X_test))):
        row = i // 3
        col = i % 3
        
        # De-normalize coordinates
        lat_min, lat_max = norm_stats['Latitude_min'], norm_stats['Latitude_max']
        lon_min, lon_max = norm_stats['Longitude_min'], norm_stats['Longitude_max']
        
        # Input trajectory
        input_lat = X_test[i, :, 0] * (lat_max - lat_min) + lat_min
        input_lon = X_test[i, :, 1] * (lon_max - lon_min) + lon_min
        
        # True future
        true_lat = y_test[i, :, 0] * (lat_max - lat_min) + lat_min
        true_lon = y_test[i, :, 1] * (lon_max - lon_min) + lon_min
        
        # Predicted future
        pred_lat = transformer_results['predictions'][i, :, 0] * (lat_max - lat_min) + lat_min
        pred_lon = transformer_results['predictions'][i, :, 1] * (lon_max - lon_min) + lon_min
        
        # Plot
        axes[row, col].plot(input_lon, input_lat, 'b-', label='Input', linewidth=2, alpha=0.7)
        axes[row, col].plot(true_lon, true_lat, 'g-', label='True Future', linewidth=3)
        axes[row, col].plot(pred_lon, pred_lat, 'r--', label='Predicted', linewidth=2)
        
        # Mark start and end points
        axes[row, col].plot(input_lon[0], input_lat[0], 'bo', markersize=8, label='Start')
        axes[row, col].plot(true_lon[-1], true_lat[-1], 'go', markersize=8, label='True End')
        axes[row, col].plot(pred_lon[-1], pred_lat[-1], 'ro', markersize=8, label='Pred End')
        
        axes[row, col].set_title(f'Trajectory {i+1}', fontsize=12, fontweight='bold')
        axes[row, col].set_xlabel('Longitude')
        axes[row, col].set_ylabel('Latitude')
        axes[row, col].grid(True, alpha=0.3)
        
        if i == 0:
            axes[row, col].legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/trajectory_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results summary
    results_summary = {
        'timestamp': str(np.datetime64('now')),
        'data_info': {
            'total_samples': len(X),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'sequence_length': input_length,
            'prediction_horizon': output_length,
            'features': 3
        },
        'model_info': {
            'transformer_params': transformer_params,
            'lstm_params': lstm_params
        },
        'performance': {
            'transformer': {
                'mse': float(transformer_results['mse']),
                'mae': float(transformer_results['mae']),
                'rmse': float(transformer_results['rmse'])
            },
            'lstm': {
                'mse': float(lstm_results['mse']),
                'mae': float(lstm_results['mae']),
                'rmse': float(lstm_results['rmse'])
            }
        },
        'training_epochs': len(transformer_history['train_loss'])
    }
    
    with open(f'{output_dir}/demo_results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"📁 Results saved to: {output_dir}/")
    print(f"   - comprehensive_analysis.png")
    print(f"   - trajectory_predictions.png") 
    print(f"   - demo_results_summary.json")
    
    # Print final summary
    print(f"\n📊 Final Performance Summary:")
    print(f"   Transformer: MSE={transformer_results['mse']:.6f}, MAE={transformer_results['mae']:.6f}, RMSE={transformer_results['rmse']:.6f}")
    print(f"   LSTM: MSE={lstm_results['mse']:.6f}, MAE={lstm_results['mae']:.6f}, RMSE={lstm_results['rmse']:.6f}")
    
    winner = "Transformer" if transformer_results['rmse'] < lstm_results['rmse'] else "LSTM"
    print(f"   🏆 Winner: {winner} (lower RMSE)")
    
    return results_summary


if __name__ == "__main__":
    run_comprehensive_demo()
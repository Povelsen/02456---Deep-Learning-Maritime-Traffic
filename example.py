#!/usr/bin/env python3
"""
Example Usage Script
====================

Demonstrates how to use the modular vessel trajectory prediction framework.
"""

import numpy as np
import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import TransformerTrajectoryPredictor, LSTMTrajectoryPredictor
from data import TrajectoryDataset, load_and_prepare_data
from training import ModelTrainer
from visualization import TrainingVisualizer, PredictionVisualizer


def create_sample_data(n_vessels=10, seq_length=256):
    """Create sample trajectory data for demonstration"""
    np.random.seed(42)
    
    sample_data = []
    for vessel_id in range(n_vessels):
        # Start position
        lat = np.random.uniform(50, 60)
        lon = np.random.uniform(0, 20)
        speed = np.random.uniform(5, 25)
        
        trajectory = []
        for t in range(seq_length):
            # Simulate realistic movement
            lat += np.random.normal(0, 0.001) + 0.0001 * np.sin(t * 0.1)
            lon += np.random.normal(0, 0.001) + 0.0001 * np.cos(t * 0.05)
            speed += np.random.normal(0, 0.2)
            speed = max(0, min(30, speed))
            
            trajectory.append([lat, lon, speed])
        
        sample_data.append(trajectory)
    
    data = np.array(sample_data)
    
    # Normalize
    norm_stats = {
        'Latitude_min': 50, 'Latitude_max': 60,
        'Longitude_min': 0, 'Longitude_max': 20,
        'SOG_min': 0, 'SOG_max': 30
    }
    
    data[:, :, 0] = (data[:, :, 0] - 50) / (60 - 50)
    data[:, :, 1] = (data[:, :, 1] - 0) / (20 - 0)
    data[:, :, 2] = (data[:, :, 2] - 0) / (30 - 0)
    
    return data, norm_stats


def demonstrate_framework():
    """Demonstrate the complete framework usage"""
    print("🚢 Vessel Trajectory Prediction Framework Demo")
    print("=" * 60)
    
    # Create output directory
    output_dir = "demo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample data
    print("\n📊 Creating sample data...")
    data, norm_stats = create_sample_data(n_vessels=20, seq_length=256)
    print(f"   Created {data.shape[0]} trajectories with {data.shape[1]} time steps")
    
    # Create models
    print("\n🏗️  Creating models...")
    transformer = TransformerTrajectoryPredictor(
        input_dim=3,
        d_model=64,
        num_heads=4,
        num_layers=2,
        prediction_horizon=12
    )
    
    lstm = LSTMTrajectoryPredictor(
        input_dim=3,
        hidden_dim=64,
        num_layers=2,
        prediction_horizon=12
    )
    
    print(f"   Transformer params: {sum(p.numel() for p in transformer.parameters()):,}")
    print(f"   LSTM params: {sum(p.numel() for p in lstm.parameters()):,}")
    
    # Prepare data for training
    print("\n🔧 Preparing training data...")
    from sklearn.model_selection import train_test_split
    
    X, y = [], []
    input_length = 200
    output_length = 12
    
    for trajectory in data:
        if len(trajectory) >= input_length + output_length:
            X.append(trajectory[:input_length])
            y.append(trajectory[input_length:input_length + output_length])
    
    X = np.array(X)
    y = np.array(y)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"   Training set: {X_train.shape}")
    print(f"   Validation set: {X_val.shape}")
    print(f"   Test set: {X_test.shape}")
    
    # Create data loaders
    from torch.utils.data import DataLoader
    
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
        train_loader, val_loader, num_epochs=10, learning_rate=1e-3
    )
    
    # Train LSTM
    print("   Training LSTM...")
    lstm_trainer = ModelTrainer(lstm)
    lstm_history = lstm_trainer.train(
        train_loader, val_loader, num_epochs=10, learning_rate=1e-3
    )
    
    # Evaluate models
    print("\n📈 Evaluating models...")
    transformer_results = transformer_trainer.evaluate(X_test, y_test)
    lstm_results = lstm_trainer.evaluate(X_test, y_test)
    
    print(f"   Transformer - MSE: {transformer_results['mse']:.6f}, MAE: {transformer_results['mae']:.6f}")
    print(f"   LSTM - MSE: {lstm_results['mse']:.6f}, MAE: {lstm_results['mae']:.6f}")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    
    training_viz = TrainingVisualizer(output_dir)
    prediction_viz = PredictionVisualizer(output_dir)
    
    # Training curves
    training_viz.plot_training_curves(transformer_history, "Transformer_Demo")
    training_viz.plot_training_curves(lstm_history, "LSTM_Demo")
    
    # Model comparison
    results_dict = {
        'Transformer': {
            'history': transformer_history,
            'metrics': transformer_results,
            'model_info': {'type': 'Transformer', 'parameters': sum(p.numel() for p in transformer.parameters())}
        },
        'LSTM': {
            'history': lstm_history,
            'metrics': lstm_results,
            'model_info': {'type': 'LSTM', 'parameters': sum(p.numel() for p in lstm.parameters())}
        }
    }
    
    training_viz.plot_model_comparison(results_dict, f"{output_dir}/demo_comparison.png")
    
    # Prediction visualizations
    prediction_viz.visualize_predictions(
        X_test, y_test, transformer_results['predictions'], 
        norm_stats, "Transformer_Demo", num_samples=3
    )
    
    prediction_viz.visualize_predictions(
        X_test, y_test, lstm_results['predictions'], 
        norm_stats, "LSTM_Demo", num_samples=3
    )
    
    # Test set analysis
    prediction_viz.create_test_set_analysis(
        X_test, y_test, transformer_results['predictions'], "Transformer_Demo"
    )
    
    prediction_viz.create_test_set_analysis(
        X_test, y_test, lstm_results['predictions'], "LSTM_Demo"
    )
    
    print(f"\n✅ Demo complete! Results saved to {output_dir}/")
    print(f"\n📁 Generated files:")
    print(f"   - Training curves: {output_dir}/*_training_curves.png")
    print(f"   - Model comparison: {output_dir}/demo_comparison.png")
    print(f"   - Prediction maps: {output_dir}/*_prediction_map_*.html")
    print(f"   - Error analysis: {output_dir}/*_error_analysis_*.png")
    print(f"   - Test analysis: {output_dir}/*_test_analysis.png")


if __name__ == "__main__":
    demonstrate_framework()
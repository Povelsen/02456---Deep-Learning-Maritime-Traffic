"""
Visualization Utilities
=======================

Plotting functions for training curves, model performance, and trajectory predictions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import folium
from typing import Dict, Optional
import seaborn as sns

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


class TrainingVisualizer:
    """Visualization utilities for training progress and performance"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_training_curves(self, history: Dict, model_name: str, save_path: Optional[str] = None):
        """Plot training and validation loss curves"""
        if save_path is None:
            save_path = f"{self.output_dir}/{model_name}_training_curves.png"
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title(f'{model_name} - Loss Curves', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE curves
        ax2.plot(epochs, history['train_mae'], 'b-', label='Training MAE', linewidth=2)
        ax2.plot(epochs, history['val_mae'], 'r-', label='Validation MAE', linewidth=2)
        ax2.set_title(f'{model_name} - MAE Curves', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning curve comparison
        ax3.plot(epochs, history['train_loss'], 'b-', label='Training', linewidth=2)
        ax3.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
        ax3.set_title(f'{model_name} - Learning Curves', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Convergence analysis
        if len(history['val_loss']) > 10:
            recent_loss = history['val_loss'][-10:]
            improvement = (recent_loss[0] - recent_loss[-1]) / recent_loss[0] * 100
            ax4.text(0.5, 0.7, f'Recent Improvement:\n{improvement:.2f}%', 
                    ha='center', va='center', fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        ax4.text(0.5, 0.3, f'Total Epochs: {len(history["train_loss"])}', 
                ha='center', va='center', fontsize=12)
        ax4.set_title(f'{model_name} - Training Summary', fontsize=14, fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_model_comparison(self, results: Dict, save_path: Optional[str] = None):
        """Compare multiple models' performance"""
        if save_path is None:
            save_path = f"{self.output_dir}/model_comparison.png"
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        model_names = list(results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        
        # Loss comparison
        for i, (name, result) in enumerate(results.items()):
            history = result['history']
            epochs = range(1, len(history['val_loss']) + 1)
            ax1.plot(epochs, history['val_loss'], color=colors[i], 
                    label=f'{name}', linewidth=2, marker='o', markersize=3)
        
        ax1.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Validation Loss (MSE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE comparison
        for i, (name, result) in enumerate(results.items()):
            history = result['history']
            epochs = range(1, len(history['val_mae']) + 1)
            ax2.plot(epochs, history['val_mae'], color=colors[i], 
                    label=f'{name}', linewidth=2, marker='s', markersize=3)
        
        ax2.set_title('Validation MAE Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Final performance metrics
        metrics = ['MSE', 'MAE', 'RMSE']
        x_pos = np.arange(len(metrics))
        width = 0.8 / len(model_names)
        
        for i, (name, result) in enumerate(results.items()):
            values = [
                result['metrics']['mse'],
                result['metrics']['mae'],
                result['metrics']['rmse']
            ]
            ax3.bar(x_pos + i * width, values, width, label=name, 
                   color=colors[i], alpha=0.7)
        
        ax3.set_title('Final Performance Metrics', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Error Values')
        ax3.set_xticks(x_pos + width * (len(model_names) - 1) / 2)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Model complexity vs performance
        complexities = []
        performances = []
        
        for name, result in results.items():
            complexities.append(result['model_info']['parameters'])
            performances.append(1 / result['metrics']['rmse'])  # Higher is better
        
        for i, name in enumerate(model_names):
            ax4.scatter(complexities[i], performances[i], s=100, 
                       color=colors[i], alpha=0.7, label=name)
        
        ax4.set_title('Model Complexity vs Performance', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Number of Parameters')
        ax4.set_ylabel('Performance (1/RMSE)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path


class PredictionVisualizer:
    """Visualization utilities for trajectory predictions"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize_predictions(self, X_test: np.ndarray, y_test: np.ndarray, 
                            predictions: np.ndarray, norm_stats: Dict, 
                            model_name: str, num_samples: int = 5):
        """Create interactive maps comparing true vs predicted trajectories"""
        
        # Get normalization stats
        lat_min = norm_stats.get('Latitude_min', 0)
        lat_max = norm_stats.get('Latitude_max', 1)
        lon_min = norm_stats.get('Longitude_min', 0)
        lon_max = norm_stats.get('Longitude_max', 1)
        
        # De-normalization function
        def denorm(x, vmin, vmax):
            return (x * (vmax - vmin)) + vmin
        
        maps_created = []
        
        for i in range(min(num_samples, len(X_test))):
            # De-normalize paths
            input_lat = denorm(X_test[i, :, 0], lat_min, lat_max)
            input_lon = denorm(X_test[i, :, 1], lon_min, lon_max)
            
            true_lat = denorm(y_test[i, :, 0], lat_min, lat_max)
            true_lon = denorm(y_test[i, :, 1], lon_min, lon_max)
            
            pred_lat = denorm(predictions[i, :, 0], lat_min, lat_max)
            pred_lon = denorm(predictions[i, :, 1], lon_min, lon_max)
            
            # Combine for paths
            path_input = list(zip(input_lat, input_lon))
            path_true = list(zip(true_lat, true_lon))
            path_pred = list(zip(pred_lat, pred_lon))
            
            # Create map centered on the start of the prediction
            m = folium.Map(location=path_true[0], zoom_start=12)
            
            # Add paths
            folium.PolyLine(path_input, color='blue', weight=2.5, opacity=1, 
                           popup='Input Path', tooltip='Historical Trajectory').add_to(m)
            folium.PolyLine(path_true, color='green', weight=4, opacity=0.8, 
                           popup='True Future Path', tooltip='Actual Future Path').add_to(m)
            folium.PolyLine(path_pred, color='red', weight=3, opacity=0.8, 
                           popup='Predicted Path', tooltip='Model Prediction').add_to(m)
            
            # Add start/end markers
            folium.Marker(path_input[0], popup='Start Input', 
                         icon=folium.Icon(color='blue', icon='ship')).add_to(m)
            folium.Marker(path_true[0], popup='Start Prediction', 
                         icon=folium.Icon(color='green', icon='play')).add_to(m)
            folium.Marker(path_true[-1], popup='End True', 
                         icon=folium.Icon(color='green', icon='stop')).add_to(m)
            folium.Marker(path_pred[-1], popup='End Predicted', 
                         icon=folium.Icon(color='red', icon='flag')).add_to(m)
            
            # Add error analysis
            errors = np.sqrt((np.array(true_lat) - np.array(pred_lat))**2 + 
                           (np.array(true_lon) - np.array(pred_lon))**2)
            
            # Save map
            save_path = f'{self.output_dir}/{model_name}_prediction_map_{i}.html'
            m.save(save_path)
            maps_created.append(save_path)
            
            # Create error plot
            self._create_error_plot(errors, i, model_name)
        
        return maps_created
    
    def _create_error_plot(self, errors: np.ndarray, sample_idx: int, model_name: str):
        """Create error analysis plot for a prediction"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Error over time
        time_steps = range(1, len(errors) + 1)
        ax1.plot(time_steps, errors, 'r-', linewidth=2, marker='o')
        ax1.set_title(f'{model_name} - Prediction Error (Sample {sample_idx})', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Prediction Time Step')
        ax1.set_ylabel('Position Error (degrees)')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative error
        cumulative_error = np.cumsum(errors)
        ax2.plot(time_steps, cumulative_error, 'b-', linewidth=2, marker='s')
        ax2.set_title(f'{model_name} - Cumulative Error (Sample {sample_idx})', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('Prediction Time Step')
        ax2.set_ylabel('Cumulative Error')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f'Mean Error: {np.mean(errors):.4f}\nMax Error: {np.max(errors):.4f}\nFinal Error: {errors[-1]:.4f}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        save_path = f'{self.output_dir}/{model_name}_error_analysis_{sample_idx}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_test_set_analysis(self, X_test: np.ndarray, y_test: np.ndarray, 
                               predictions: np.ndarray, model_name: str):
        """Create comprehensive test set analysis"""
        
        # Calculate errors for all test samples
        errors = np.sqrt(np.mean((y_test - predictions)**2, axis=(1, 2)))
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Error distribution
        ax1.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
        ax1.set_title(f'{model_name} - Test Set Error Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('RMSE')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Error vs trajectory length (if available)
        trajectory_lengths = np.sum(np.any(X_test != 0, axis=2), axis=1)
        ax2.scatter(trajectory_lengths, errors, alpha=0.6, color='green')
        ax2.set_title(f'{model_name} - Error vs Trajectory Length', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Trajectory Length (non-zero points)')
        ax2.set_ylabel('RMSE')
        ax2.grid(True, alpha=0.3)
        
        # Prediction confidence (error over time steps)
        step_errors = np.sqrt(np.mean((y_test - predictions)**2, axis=0))
        time_steps = range(1, len(step_errors) + 1)
        ax3.plot(time_steps, step_errors, 'm-', linewidth=2, marker='o')
        ax3.set_title(f'{model_name} - Error by Prediction Step', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Prediction Step')
        ax3.set_ylabel('Mean RMSE')
        ax3.grid(True, alpha=0.3)
        
        # Performance summary
        ax4.axis('off')
        summary_text = f"""
Test Set Performance Summary

Total Samples: {len(errors):,}
Mean RMSE: {np.mean(errors):.6f}
Median RMSE: {np.median(errors):.6f}
Std RMSE: {np.std(errors):.6f}
Min RMSE: {np.min(errors):.6f}
Max RMSE: {np.max(errors):.6f}

Best 25%: {np.percentile(errors, 25):.6f}
Worst 25%: {np.percentile(errors, 75):.6f}
        """
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
        
        plt.tight_layout()
        save_path = f'{self.output_dir}/{model_name}_test_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
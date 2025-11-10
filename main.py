#!/usr/bin/env python3
"""
Main Entry Point for Vessel Trajectory Prediction
===============================================

Complete PyTorch implementation for vessel trajectory prediction using
Transformer and LSTM neural networks with comprehensive visualization.

Usage:
------
    python main.py
    python main.py --data_path processed_data.npz --epochs 50 --output_dir results/
"""

import argparse
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import TransformerTrajectoryPredictor, LSTMTrajectoryPredictor
from data import load_and_prepare_data
from training.comparison import ModelComparisonFramework
from visualization.plots import TrainingVisualizer, PredictionVisualizer
from data_processing import run_complete_pipeline


def main():
    """Main execution function"""
    print("🚢 Enhanced PyTorch Vessel Trajectory Prediction")
    print("=" * 70)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Vessel Trajectory Prediction with PyTorch')
    parser.add_argument('--data_path', type=str, default='processed_data.npz',
                       help='Path to data file (default: processed_data.npz)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory for results (default: output)')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Generate comprehensive visualizations (default: True)')
    
    # Data processing arguments
    parser.add_argument('--csv_files', nargs='+', type=str,
                       help='List of CSV files to process (alternative to --data_path)')
    parser.add_argument('--process_data', action='store_true',
                       help='Process CSV files before training')
    parser.add_argument('--resample_interval', type=str, default='1min',
                       help='Resampling interval for data processing (default: 1min)')
    parser.add_argument('--max_seq_len', type=int, default=256,
                       help='Maximum sequence length for data processing (default: 256)')
    parser.add_argument('--bbox', nargs=4, type=float,
                       help='Bounding box filter [north, west, south, east]')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle data processing if requested
    if args.process_data and args.csv_files:
        print(f"\n🔧 Processing CSV files before training...")
        
        # Set default bounding box (Denmark region)
        bbox = args.bbox if args.bbox else [58.0, 7.5, 54.5, 15.5]
        
        # Run data processing pipeline
        processing_results = run_complete_pipeline(
            csv_files=args.csv_files,
            output_base_dir="processed_ais_data",
            resample_interval=args.resample_interval,
            max_seq_len=args.max_seq_len,
            bbox=bbox
        )
        
        # Use the consolidated dataset if available, otherwise use first NPZ file
        if 'consolidated_file' in processing_results:
            args.data_path = processing_results['consolidated_file']
        elif processing_results['npz_files_created']:
            args.data_path = processing_results['npz_files_created'][0]
        else:
            print("❌ No data files were created. Using default data generation.")
    
    # Load data
    print(f"\n📊 Loading data from {args.data_path}...")
    data, norm_stats = load_and_prepare_data(args.data_path)
    
    if data is not None:
        print(f"✅ Data loaded successfully!")
        print(f"   - Shape: {data.shape}")
        print(f"   - Features: {data.shape[-1]}")
        print(f"   - Normalization stats: {list(norm_stats.keys())}")
        
        # Create models
        print(f"\n🏗️  Creating models...")
        transformer = TransformerTrajectoryPredictor(
            input_dim=data.shape[-1],
            d_model=128,
            num_heads=8,
            num_layers=4,
            prediction_horizon=12
        )
        
        lstm = LSTMTrajectoryPredictor(
            input_dim=data.shape[-1],
            hidden_dim=128,
            num_layers=2,
            prediction_horizon=12
        )
        
        # Print model information
        print("\n📋 Model Information:")
        transformer_params = sum(p.numel() for p in transformer.parameters())
        lstm_params = sum(p.numel() for p in lstm.parameters())
        print(f"   Transformer Parameters: {transformer_params:,}")
        print(f"   LSTM Parameters: {lstm_params:,}")
        
        # Initialize comparison framework
        comparison = ModelComparisonFramework()
        comparison.add_model('Transformer', transformer)
        comparison.add_model('LSTM', lstm)
        
        # Initialize visualizers
        training_viz = TrainingVisualizer(args.output_dir)
        prediction_viz = PredictionVisualizer(args.output_dir)
        
        # Run comparison
        print(f"\n🚀 Running model comparison for {args.epochs} epochs...")
        results = comparison.run_comparison(data, norm_stats, args.output_dir, num_epochs=args.epochs)
        
        # Generate comprehensive visualizations
        if args.visualize:
            print(f"\n📈 Generating comprehensive visualizations...")
            
            # Training curves for each model
            for name, result in results.items():
                print(f"   - Training curves for {name}...")
                training_viz.plot_training_curves(result['history'], name)
            
            # Model comparison
            print(f"   - Model comparison plots...")
            training_viz.plot_model_comparison(results)
            
            # Prediction visualizations
            print(f"   - Prediction maps and analysis...")
            for name, result in results.items():
                print(f"     * {name} predictions...")
                
                # Get test data from comparison framework
                _, _, _, X_test, y_test = comparison.prepare_data(data)
                predictions = result['metrics']['predictions']
                
                # Create prediction maps
                prediction_viz.visualize_predictions(
                    X_test, y_test, predictions, norm_stats, name
                )
                
                # Create test set analysis
                prediction_viz.create_test_set_analysis(
                    X_test, y_test, predictions, name
                )
        
        # Generate and display results
        print(f"\n📊 Performance Summary:")
        print("=" * 80)
        summary = comparison.generate_performance_summary()
        print(summary.to_string(index=False))
        
        # Save results
        print(f"\n💾 Saving results...")
        comparison.save_results(args.output_dir)
        
        print(f"\n🎉 Comparison complete! Results saved to {args.output_dir}/")
        
        # Print file locations
        print(f"\n📁 Generated files:")
        print(f"   - Performance summary: {args.output_dir}/pytorch_model_performance.csv")
        print(f"   - Training history: {args.output_dir}/pytorch_training_history.json")
        print(f"   - Training curves: {args.output_dir}/*_training_curves.png")
        print(f"   - Model comparison: {args.output_dir}/model_comparison.png")
        print(f"   - Prediction maps: {args.output_dir}/*_prediction_map_*.html")
        print(f"   - Error analysis: {args.output_dir}/*_error_analysis_*.png")
        print(f"   - Test analysis: {args.output_dir}/*_test_analysis.png")
        
        return comparison, results
    
    else:
        print("❌ Failed to load data. Please check your data file.")
        return None, None


if __name__ == "__main__":
    comparison, results = main()
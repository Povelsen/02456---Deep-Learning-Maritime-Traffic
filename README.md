# Vessel Trajectory Prediction with PyTorch

This repository contains a complete implementation of Transformer and LSTM neural networks for predicting vessel trajectories using AIS (Automatic Identification System) data.

## 🚢 Overview

The project compares two advanced neural network architectures for maritime trajectory prediction:

- **Transformer Model**: Uses multi-head attention mechanisms for parallel sequence processing
- **LSTM Model**: Uses recurrent connections for sequential pattern learning

## 📁 Project Structure

```
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── output/                    # Generated results and plots
├── src/
│   ├── models/                # Neural network architectures
│   │   ├── transformer.py     # Transformer components
│   │   ├── lstm.py            # LSTM model
│   │   └── __init__.py
│   ├── data/                  # Data processing
│   │   ├── dataset.py         # Dataset classes
│   │   ├── utils.py           # Data loading utilities
│   │   └── __init__.py
│   ├── training/              # Training utilities
│   │   ├── trainer.py         # Model trainer
│   │   ├── comparison.py      # Model comparison framework
│   │   └── __init__.py
│   ├── visualization/         # Plotting utilities
│   │   ├── plots.py           # Visualization functions
│   │   └── __init__.py
│   └── utils/                 # General utilities
└── scripts/                   # Utility scripts
```

## 🛠 Installation

1. Install PyTorch and dependencies:
```bash
pip install -r requirements.txt
```

2. Run the main script:
```bash
python main.py
```

## 🚀 Usage

### Basic Usage
```bash
python main.py
```

### Advanced Options
```bash
python main.py --data_path your_data.npz --epochs 50 --output_dir results/
```

### Command Line Arguments
- `--data_path`: Path to AIS data file (default: processed_data.npz)
- `--epochs`: Number of training epochs (default: 30)
- `--output_dir`: Output directory for results (default: output)
- `--visualize`: Generate comprehensive visualizations (default: True)

## 📊 Model Architecture

### Transformer Model
- **Input Embedding**: Linear projection to 128-dimensional space
- **Positional Encoding**: Sinusoidal encoding for temporal information
- **Transformer Blocks**: 4 encoder layers with multi-head attention
- **Output**: Dense layers predicting 12 future trajectory points

### LSTM Model
- **LSTM Layers**: 2 layers with 128 hidden units each
- **Dropout**: 20% dropout for regularization
- **Output**: Dense layers predicting 12 future trajectory points

## 📈 Performance Metrics

The framework evaluates models using:
- **MSE (Mean Squared Error)**: Average squared difference between predictions and targets
- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and targets
- **RMSE (Root Mean Squared Error)**: Square root of MSE for interpretability

## 🎯 Features

### Data Processing
- Automatic data normalization
- Sequence generation for time series prediction
- Train/validation/test splitting

### Training Features
- Early stopping to prevent overfitting
- Learning rate scheduling
- Gradient clipping for stability
- Progress bars for monitoring

### Visualization Features
- Training and validation loss curves
- Model performance comparison
- Interactive trajectory prediction maps
- Error analysis and distribution plots
- Test set performance analysis

### Model Features
- Multi-head attention (Transformer)
- Positional encoding for temporal patterns
- Dropout regularization
- Residual connections

## 📊 Expected Results

Typical performance on AIS trajectory data:
- **Transformer**: RMSE ~0.40-0.45
- **LSTM**: RMSE ~0.42-0.47

*Note: Results may vary depending on data quality and quantity*

## 🔧 Customization

### Model Configuration
Modify hyperparameters in the model classes:
```python
transformer = TransformerTrajectoryPredictor(
    input_dim=3,      # Number of features
    d_model=128,      # Model dimension
    num_heads=8,      # Attention heads
    num_layers=4,     # Transformer layers
    prediction_horizon=12  # Future steps to predict
)
```

### Data Format
Input data should be numpy arrays with shape:
- `(n_samples, sequence_length, n_features)`
- Features: [latitude, longitude, speed_over_ground]

## 📁 Output Files

The framework generates comprehensive output:

### Performance Files
- `pytorch_model_performance.csv` - Model performance metrics
- `pytorch_training_history.json` - Training loss curves and metrics
- `pytorch_data_info.json` - Dataset characteristics
- `pytorch_model_architectures.json` - Model architecture details

### Visualization Files
- `*_training_curves.png` - Training progress plots
- `model_comparison.png` - Comparative performance analysis
- `*_prediction_map_*.html` - Interactive trajectory maps
- `*_error_analysis_*.png` - Error analysis plots
- `*_test_analysis.png` - Test set performance summary

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🔗 References

- "Attention Is All You Need" - Transformer paper
- "Long Short-Term Memory" - LSTM paper
- AIS data format specifications

## 📞 Support

For questions or issues:
1. Check the documentation above
2. Review the code comments
3. Create an issue in the repository

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Maritime research community for AIS data standards
- Open source contributors for supporting libraries
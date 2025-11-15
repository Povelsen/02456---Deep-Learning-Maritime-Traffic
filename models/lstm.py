"""
LSTM Model for Vessel Trajectory Prediction
==========================================

Implementation of LSTM (Long Short-Term Memory) neural network for
predicting vessel trajectories from AIS data.
"""

import torch
import torch.nn as nn


class LSTMTrajectoryPredictor(nn.Module):
    """
    LSTM-based trajectory prediction model.
    """
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 128, num_layers: int = 2, 
                 prediction_horizon: int = 12, dropout: float = 0.2):
        super(LSTMTrajectoryPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        self.input_dim = input_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, prediction_horizon * input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        lstm_output, (hidden, cell) = self.lstm(x)
        
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        
        predictions = self.output_layers(last_hidden)
        predictions = predictions.view(batch_size, self.prediction_horizon, self.input_dim)
        
        return predictions
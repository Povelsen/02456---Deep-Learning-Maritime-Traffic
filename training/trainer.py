"""
Model Training Utilities
========================

Training classes and utilities for vessel trajectory prediction models.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict


class ModelTrainer:
    """
    Training class for trajectory prediction models.
    """
    
    def __init__(self, model: nn.Module, device: str = None, output_dir: str = "output"):
        if device is None:
            # Auto-detect device with priority: CUDA -> MPS -> CPU
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # Check for Apple MPS (MacBook M-series)
                self.device = 'mps'
            else:
                self.device = 'cpu'
            print(f"No device specified. Auto-detecting: {self.device}")
        else:
            # Use user-specified device
            self.device = device
            print(f"User specified device: {self.device}")
            
        self.model = model.to(self.device)
        self.output_dir = output_dir
        self.history = {
            'train_loss': [], 'val_loss': [], 
            'train_mae': [], 'val_mae': []
        }
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 50, learning_rate: float = 1e-4, 
              model_name: str = "model") -> Dict:
        """Train the model with early stopping and learning rate scheduling"""
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_mae = 0.0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            for batch_X, batch_y in train_pbar:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_mae += torch.mean(torch.abs(predictions - batch_y)).item()
                
                train_pbar.set_postfix({'loss': loss.item()})
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_mae = 0.0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
                for batch_X, batch_y in val_pbar:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    predictions = self.model(batch_X)
                    loss = criterion(predictions, batch_y)
                    
                    val_loss += loss.item()
                    val_mae += torch.mean(torch.abs(predictions - batch_y)).item()
                    
                    val_pbar.set_postfix({'loss': loss.item()})
            
            # Calculate averages
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_train_mae = train_mae / len(train_loader)
            avg_val_mae = val_mae / len(val_loader)
            
            # Update history
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['train_mae'].append(avg_train_mae)
            self.history['val_mae'].append(avg_val_mae)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model as .pt file in output directory
                model_path = os.path.join(self.output_dir, f'{model_name}_best.pt')
                torch.save(self.model.state_dict(), model_path)
                print(f"   💾 Saved best model: {model_path}")
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Print progress every epoch
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}, "
                  f"Train MAE: {avg_train_mae:.6f}, "
                  f"Val MAE: {avg_val_mae:.6f}")
        
        # Save final model as .pt file in output directory
        final_model_path = os.path.join(self.output_dir, f'{model_name}_final.pt')
        torch.save(self.model.state_dict(), final_model_path)
        print(f"💾 Saved final model: {final_model_path}")
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model performance"""
        predictions = self.predict(X)
        
        y_true = y.reshape(-1, y.shape[-1])
        y_pred = predictions.reshape(-1, predictions.shape[-1])
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse, 
            'mae': mae, 
            'rmse': rmse, 
            'predictions': predictions
        }
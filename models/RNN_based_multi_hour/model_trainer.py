"""
Training and Forecasting Script for RNN Multivariate Forecasting Model

Loads processed data from data_adapter and trains/forecasts using the model.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from typing import Dict, Optional

# Import model components
from .model import (
    RNN_fourier,
    RNN_train_fourier,
    training_config,
    fourier_config,
    build_model_dp,
    save_state,
    load_state
)


class ModelTrainer:
    """
    Handles training and forecasting for the RNN model.
    """
    
    def __init__(
        self,
        data_dir: str,
        model_save_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            data_dir: Directory containing model_input folder (from data_adapter)
            model_save_dir: Where to save trained model (default: data_dir/trained_model)
            device: 'cuda', 'mps', or 'cpu' (auto-detect if None)
        """
        self.data_dir = data_dir
        self.input_dir = os.path.join(data_dir, "features", "model_input")
        self.model_save_dir = model_save_dir or os.path.join(data_dir, "trained_model")
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Will be loaded
        self.model_config: Dict = {}
        self.train_df: pd.DataFrame = None
        self.forecast_input: pd.DataFrame = None
        self.model = None
        self.trainer = None
        
    def load_data(self):
        """Load processed data and config from data_adapter output."""
        print("=" * 60)
        print("Loading data")
        print("=" * 60)
        
        # Load model config
        config_path = os.path.join(self.input_dir, "model_config.json")
        with open(config_path, 'r') as f:
            self.model_config = json.load(f)
        print(f"Loaded model config: {config_path}")
        
        # Load training data
        train_path = os.path.join(self.input_dir, "train_data.parquet")
        self.train_df = pd.read_parquet(train_path)
        print(f"Loaded training data: {self.train_df.shape}")
        
        # Load forecast input
        forecast_path = os.path.join(self.input_dir, "forecast_input.parquet")
        self.forecast_input = pd.read_parquet(forecast_path)
        print(f"Loaded forecast input: {self.forecast_input.shape}")
        
        # Print config summary
        print(f"\nModel config:")
        print(f"  n_series: {self.model_config['n_series']}")
        print(f"  n_shared: {self.model_config['n_shared']}")
        print(f"  n_indep: {self.model_config['n_indep']}")
        print(f"  H: {self.model_config['H']}")
        print(f"  text_embed_dims: {self.model_config['text_embed_dims']}")
        
    def build_model(
        self,
        # Fourier config
        K_weekly: int = 3,
        K_monthly: int = 6,
        K_yearly: int = 10,
        fourier_mode: str = "matrix",
        # Model architecture
        d_model: int = 128,
        latent_dim: int = 32,
        nhead: int = 4,
        dropout: float = 0.0,
        embed_dim: int = 32,
        embed_hidden: int = 64,
        text_embed_hidden: int = 256,
    ):
        """Build the RNN model with specified configuration."""
        print("\n" + "=" * 60)
        print("Building model")
        print("=" * 60)
        
        # Fourier configuration
        self.fourier_conf = fourier_config(
            mode=fourier_mode,
            K_weekly=K_weekly,
            K_monthly=K_monthly,
            K_yearly=K_yearly,
            P_WEEK=7.0,
            P_MONTH=365.25 / 12.0,
            P_yearly=365.25,
        )
        
        # Calculate fourier dimension
        K_total = K_weekly + K_monthly + K_yearly
        fourier_dim = 2 * K_total
        print(f"Fourier dim: {fourier_dim} (K_total={K_total})")
        
        # Model configuration from data
        n_series = self.model_config['n_series']
        n_shared = self.model_config['n_shared']
        n_indep = self.model_config['n_indep']
        H = self.model_config['H']
        text_embed_dims = self.model_config['text_embed_dims']
        
        # Build model config dict
        model_params = dict(
            fourier_dim=fourier_dim,
            xf_mode=fourier_mode,
            d_model=d_model,
            latent_dim=latent_dim,
            nhead=nhead,
            n_series=n_series,
            n_shared=n_shared,
            n_indep=n_indep,
            H=H,
            dropout=dropout,
            embed_dim=embed_dim,
            embed_hidden=embed_hidden,
        )
        
        # Add text embedding params if we have embeddings
        if text_embed_dims:
            model_params['text_embed_dims'] = text_embed_dims
            model_params['text_embed_hidden'] = text_embed_hidden
        
        print(f"Model parameters:")
        for k, v in model_params.items():
            print(f"  {k}: {v}")
        
        # Build model based on device
        if self.device.type == "cuda":
            # Use DataParallel for CUDA
            self.model = build_model_dp(RNN_fourier, **model_params)
        else:
            # Direct instantiation for MPS/CPU
            self.model = RNN_fourier(**model_params)
            self.model = self.model.to(self.device)
        
        print(f"\nModel built successfully on {self.device}")
        
        return self.model

    def train(
            self,
            n_epochs: int = 32,
            T_hist: int = 32,
            lr: float = 5e-4,
            lambda0: float = 1e-5,
            lambdaf: float = 5e-4,
            deterministic: bool = True,
    ):
        """Train the model on all training data."""
        print("\n" + "=" * 60)
        print("Training model")
        print("=" * 60)

        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Training configuration
        train_conf = training_config(
            n_epochs=n_epochs,
            device=self.device,
            T_hist=T_hist,
            lr=lr,
            lambda0=lambda0,
            lambdaf=lambdaf,
        )

        print(f"Training config:")
        print(f"  n_epochs: {n_epochs}")
        print(f"  T_hist: {T_hist}")
        print(f"  lr: {lr}")
        print(f"  device: {self.device}")
        print(f"  Training data: {len(self.train_df)} days")

        # Create trainer
        self.trainer = RNN_train_fourier(
            self.model,
            train_conf,
            self.fourier_conf,
            deterministic=deterministic
        )

        # Train on all data
        print(f"\nStarting training...")
        self.trainer(self.train_df)

        print("\nTraining complete!")

        return self.trainer

    
    def forecast(self) -> Dict:
        """
        Run forecast on the forecast_input data.
        
        Returns:
            Dictionary with predictions per series
        """
        print("\n" + "=" * 60)
        print("Running forecast")
        print("=" * 60)
        
        if self.trainer is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Forecast
        results = self.trainer.forecast(self.forecast_input)
        
        # Format results
        n_series = self.model_config['n_series']
        tickers = self.model_config['tickers']
        H = self.model_config['H']
        
        forecast_day = self.forecast_input['day'].iloc[0]
        
        print(f"\nForecast for: {forecast_day}")
        print("-" * 40)
        
        formatted_results = {
            'forecast_day': str(forecast_day),
            'predictions': {}
        }
        
        for s in range(n_series):
            ticker = tickers[s]
            pred = results['test_pred'][s]
            
            # pred shape is (H,) for single day forecast
            if len(pred) == H:
                hourly_pred = pred
            else:
                # Take last H values if multiple days
                hourly_pred = pred[-H:]
            
            formatted_results['predictions'][ticker] = hourly_pred.tolist()
            
            print(f"\n{ticker}:")
            print(f"  Predictions (H={H}): {hourly_pred[:5]}... (showing first 5)")
            print(f"  Mean: {np.mean(hourly_pred):.6f}")
            print(f"  Std: {np.std(hourly_pred):.6f}")
        
        return formatted_results
    
    def save_model(self, filename: str = "model.pt"):
        """Save trained model to disk."""
        os.makedirs(self.model_save_dir, exist_ok=True)
        path = os.path.join(self.model_save_dir, filename)
        save_state(self.model, path)
        print(f"Model saved: {path}")
        
        # Also save training config for later loading
        config_path = os.path.join(self.model_save_dir, "training_info.json")
        info = {
            'model_config': self.model_config,
            'fourier_conf': {
                'mode': self.fourier_conf.mode,
                'K_weekly': self.fourier_conf.K_weekly,
                'K_monthly': self.fourier_conf.K_monthly,
                'K_yearly': self.fourier_conf.K_yearly,
            }
        }
        with open(config_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"Training info saved: {config_path}")
    
    def load_model(self, filename: str = "model.pt"):
        """Load trained model from disk."""
        path = os.path.join(self.model_save_dir, filename)
        load_state(self.model, path, map_location=str(self.device))
        print(f"Model loaded: {path}")


def main():
    """Example usage."""
    
    # Initialize trainer
    trainer = ModelTrainer(
        data_dir="../../test_data",  # Adjust path as needed
    )
    
    # Load data
    trainer.load_data()
    
    # Build model
    trainer.build_model(
        K_weekly=3,
        K_monthly=6,
        K_yearly=10,
        d_model=128,
        latent_dim=32,
        nhead=4,
    )
    
    # Train
    trainer.train(
        n_epochs=32,
        T_hist=32,
        lr=5e-4,
        test_ratio=0.2,
    )
    
    # Save model
    trainer.save_model()
    
    # Forecast
    results = trainer.forecast()
    
    # Save forecast results
    results_path = os.path.join(trainer.model_save_dir, "forecast_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nForecast results saved: {results_path}")


if __name__ == "__main__":
    main()

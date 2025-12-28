"""
Data Adapter for RNN Multivariate Forecasting Model

Transforms feature data from FeatureAgent format to the model's expected format.
Following the exact step-by-step approach.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple


def simple_dst_fix(df: pd.DataFrame, start_at_midnight: bool = True) -> pd.DataFrame:
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds")

    df = df[~df["ds"].duplicated(keep="first")]

    start = df["ds"].iloc[0]
    if start_at_midnight:
        start = start.normalize()
    end = df["ds"].iloc[-1]
    full_idx = pd.date_range(start, end, freq="h")

    out = df.set_index("ds").reindex(full_idx)

    num_cols = out.select_dtypes(include="number").columns
    out[num_cols] = out[num_cols].ffill()

    if out[num_cols].isna().any().any():
        out[num_cols] = out[num_cols].bfill()

    out = out.rename_axis("ds").reset_index()

    return out


class DataAdapter:
    
    def __init__(
        self,
        data_dir: str,
        y_column: str = "compute_log_return_y",
        prediction_horizon: int = 1,
        output_dir: Optional[str] = None
    ):
        self.data_dir = data_dir
        self.y_column = y_column
        self.prediction_horizon = prediction_horizon
        self.output_dir = output_dir or os.path.join(data_dir, "features", "model_input")
        
        # Will be populated from feature dictionary
        self.feature_dict: Dict = {}
        self.has_y: bool = False
        self.H: int = 0
        self.embedding_dim: int = 0
        
        # Mappings (Step 1)
        self.ticker_to_series: Dict[str, int] = {}
        self.feature_to_idx: Dict[str, int] = {}
        self.share_to_idx: Dict[str, int] = {}
        
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Run full transformation pipeline."""
        
        # ============================================================
        # STEP 1: Read feature dictionary
        # ============================================================
        print("=" * 60)
        print("STEP 1: Read feature dictionary")
        print("=" * 60)
        
        dict_path = os.path.join(self.data_dir, "features", "feature_dictionary.json")
        with open(dict_path, 'r') as f:
            self.feature_dict = json.load(f)
        
        # Get tickers and label each with number
        tickers = self.feature_dict.get("tickers", [])
        self.ticker_to_series = {ticker: idx for idx, ticker in enumerate(tickers)}
        print(f"ticker_to_series: {self.ticker_to_series}")
        
        # Get independent features and check if Y exists
        indep_info = self.feature_dict.get("datasets", {}).get("independent_factors", {})
        all_factors = list(indep_info.get("factors", {}).keys())
        
        self.has_y = self.y_column in all_factors
        print(f"has_y ('{self.y_column}'): {self.has_y}")
        
        # Label each independent feature with number (excluding Y)
        indep_features = [f for f in all_factors if f != self.y_column]
        self.feature_to_idx = {feat: idx for idx, feat in enumerate(indep_features)}
        print(f"feature_to_idx: {self.feature_to_idx}")
        
        # Get shared features and label with number
        shared_info = self.feature_dict.get("datasets", {}).get("shared_factors", {})
        shared_features = list(shared_info.get("factors", {}).keys())
        self.share_to_idx = {feat: idx for idx, feat in enumerate(shared_features)}
        print(f"share_to_idx: {self.share_to_idx}")
        
        # Get embedding dimension
        emb_info = self.feature_dict.get("datasets", {}).get("embeddings", {})
        self.embedding_dim = emb_info.get("dimension", 0)
        print(f"embedding_dim: {self.embedding_dim}")
        
        # ============================================================
        # STEP 2-6: Process each independent factor file
        # ============================================================
        print("\n" + "=" * 60)
        print("STEP 2-6: Process each series")
        print("=" * 60)
        
        series_dfs = []
        
        for ticker, series_idx in self.ticker_to_series.items():
            print(f"\n--- Processing {ticker} (series {series_idx}) ---")
            
            # Read independent factor file
            path = os.path.join(self.data_dir, "features", "independent_factors", f"{ticker}.parquet")
            df = pd.read_parquet(path)
            print(f"  Loaded: {df.shape}")
            
            # Rename columns: y_column -> 'y', features -> 'x_{idx}'
            rename_map = {'Datetime': 'ds'}
            if self.has_y and self.y_column in df.columns:
                rename_map[self.y_column] = 'y'
            for feat, idx in self.feature_to_idx.items():
                if feat in df.columns:
                    rename_map[feat] = f'x_{idx}'
            
            df = df.reset_index().rename(columns=rename_map)
            print(f"  Renamed: {list(df.columns)}")
            
            # --- Process Y: pivot to hourly columns ---
            if 'y' in df.columns:
                y_df = df[['ds', 'y']].copy()
                y_df['ds'] = pd.to_datetime(y_df['ds'])
                y_df['day'] = y_df['ds'].dt.date
                y_df['hour'] = y_df['ds'].dt.hour
                
                # Pivot: df.pivot(index="day", columns="hour", values="y").add_prefix("y_").rename_axis(None, axis=1).reset_index()
                y_pivot = (
                    y_df.pivot(index="day", columns="hour", values="y")
                    .add_prefix("y_")
                    .rename_axis(None, axis=1)
                    .reset_index()
                )

                # Detect H from number of hour columns
                self.H = len([c for c in y_pivot.columns if c.startswith('y_')])
                print(f"  Y pivot: {y_pivot.shape}, H={self.H}")
            else:
                y_pivot = None
            
            # --- Process X: aggregate hourly to daily ---
            x_cols = [c for c in df.columns if c.startswith('x_')]
            if x_cols:
                x_df = df[['ds'] + x_cols].copy()
                x_df['ds'] = pd.to_datetime(x_df['ds'])
                x_df['day'] = x_df['ds'].dt.date
                
                # Aggregate: use last value of day (or you could expand to columns)
                x_daily = x_df.groupby('day')[x_cols].last().reset_index()
                print(f"  X daily: {x_daily.shape}")
            else:
                x_daily = None
            
            # --- Combine Y and X ---
            if y_pivot is not None and x_daily is not None:
                series_df = y_pivot.merge(x_daily, on='day', how='inner')
            elif y_pivot is not None:
                series_df = y_pivot
            elif x_daily is not None:
                series_df = x_daily
            else:
                raise ValueError(f"No Y or X data for {ticker}")
            
            print(f"  Combined: {series_df.shape}")
            
            # --- Append series label to Y and X columns ---
            rename_suffix = {}
            for col in series_df.columns:
                if col.startswith('y_'):
                    rename_suffix[col] = f"{col}_s{series_idx}"
                elif col.startswith('x_'):
                    rename_suffix[col] = f"{col}_s{series_idx}"
            
            series_df = series_df.rename(columns=rename_suffix)
            print(f"  With series suffix: {list(series_df.columns)[:5]}...")
            
            series_dfs.append(series_df)
        
        # --- Inner join all series together ---
        print("\n--- Inner join all series ---")
        big_df = series_dfs[0]
        for sdf in series_dfs[1:]:
            big_df = big_df.merge(sdf, on='day', how='inner')
        print(f"Combined all series: {big_df.shape}")


        
        # ============================================================
        # STEP 7: Shared data
        # ============================================================
        print("\n" + "=" * 60)
        print("STEP 7: Process shared data")
        print("=" * 60)
        
        shared_path = os.path.join(self.data_dir, "features", "shared_factors.parquet")
        if os.path.exists(shared_path) and self.share_to_idx:
            shared_df = pd.read_parquet(shared_path)
            print(f"Loaded shared: {shared_df.shape}")

            shared_df = shared_df.reset_index()
            
            # Rename columns to share_{idx}
            rename_map = {}
            for feat, idx in self.share_to_idx.items():
                if feat in shared_df.columns:
                    rename_map[feat] = f'share_{idx}'
            shared_df = shared_df.rename(columns=rename_map)
            
            # Convert time and extract day
            shared_df['Datetime'] = pd.to_datetime(shared_df['Datetime'])
            shared_df['day'] = shared_df['Datetime'].dt.date
            
            # Aggregate to daily
            share_cols = [c for c in shared_df.columns if c.startswith('share_')]
            shared_daily = shared_df.groupby('day')[share_cols].last().reset_index()
            print(f"Shared daily: {shared_daily.shape}")
            
            # Join to big dataframe
            big_df = big_df.merge(shared_daily, on='day', how='inner')
            print(f"After joining shared: {big_df.shape}")
        else:
            print("No shared factors found")
        
        # ============================================================
        # STEP 8: Embedding
        # ============================================================
        print("\n" + "=" * 60)
        print("STEP 8: Process embeddings")
        print("=" * 60)
        
        for ticker, series_idx in self.ticker_to_series.items():
            emb_path = os.path.join(self.data_dir, "features", "embeddings", f"{ticker}.npy")
            
            if not os.path.exists(emb_path):
                print(f"No embedding for {ticker}")
                continue
            
            # Embedding is 1D array without time index
            emb = np.load(emb_path)
            if emb.ndim > 1:
                emb = emb.flatten()
            print(f"{ticker} embedding: {emb.shape}")
            

            # Create all embedding columns at once
            emb_cols_dict = {f"emb_0_s{series_idx}_d{d}": emb[d] for d in range(len(emb))}
            emb_df = pd.DataFrame(emb_cols_dict, index=big_df.index)
            big_df = pd.concat([big_df, emb_df], axis=1)
            
            print(f"  Added {len(emb)} embedding columns for {ticker}")
        
        print(f"\nAfter embeddings: {big_df.shape}")
        
        # ============================================================
        # STEP 9: Time shift - use past data to predict future
        # ============================================================
        print("\n" + "=" * 60)
        print("STEP 9: Time shift")
        print("=" * 60)
        
        # Sort by day
        big_df = big_df.dropna().sort_values('day').reset_index(drop=True)
        
        # Identify X columns (features to shift)
        x_cols = [c for c in big_df.columns if c.startswith('x_') or c.startswith('share_') or c.startswith('emb_')]
        y_cols = [c for c in big_df.columns if c.startswith('y_')]
        
        # Take most recent row as forecast input (X only, no Y yet)
        forecast_input = big_df[big_df['day'] == max(big_df['day'])][['day'] + x_cols].copy()
        print(f"Forecast input (most recent {self.prediction_horizon} days): {forecast_input.shape}")
        
        # Move remaining feature data forward: feature at t-1 goes to t
        # This builds relationship: use past data to predict future
        x_shifted = big_df[x_cols].shift(self.prediction_horizon)
        train_df = pd.concat([big_df[['day'] + y_cols], x_shifted], axis=1)
        
        # Drop rows where X is NaN (first prediction_horizon rows)
        train_df = train_df.dropna().reset_index(drop=True)
        print(f"Training data (shifted): {train_df.shape}")
        
        # Convert day to datetime
        train_df['day'] = pd.to_datetime(train_df['day'])
        forecast_input['day'] = pd.to_datetime(forecast_input['day']) + pd.Timedelta(days=1)

        print(forecast_input)
        
        # ============================================================
        # STEP 10: Save to local folder + model parameters
        # ============================================================
        print("\n" + "=" * 60)
        print("STEP 10: Save outputs")
        print("=" * 60)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save training data
        train_path = os.path.join(self.output_dir, "train_data.parquet")
        train_df.to_parquet(train_path, index=False)
        print(f"Saved: {train_path}")
        
        # Save forecast input
        forecast_path = os.path.join(self.output_dir, "forecast_input.parquet")
        forecast_input.to_parquet(forecast_path, index=False)
        print(f"Saved: {forecast_path}")
        
        # Model parameters based on data
        model_config = {
            # Model parameters
            "n_series": len(self.ticker_to_series),
            "n_shared": len(self.share_to_idx),
            "n_indep": [len(self.feature_to_idx)] * len(self.ticker_to_series),
            "H": self.H,
            "text_embed_dims": [self.embedding_dim] if self.embedding_dim > 0 else None,
            
            # Metadata
            "tickers": list(self.ticker_to_series.keys()),
            "ticker_to_series": self.ticker_to_series,
            "feature_to_idx": self.feature_to_idx,
            "share_to_idx": self.share_to_idx,
            "y_column": self.y_column,
            "prediction_horizon": self.prediction_horizon,
            "embedding_dim": self.embedding_dim,
        }
        
        config_path = os.path.join(self.output_dir, "model_config.json")
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        print(f"Saved: {config_path}")
        
        # ============================================================
        # Summary
        # ============================================================
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Training data: {train_df.shape}")
        print(f"Forecast input: {forecast_input.shape}")
        print(f"\nModel config:")
        for k, v in model_config.items():
            if k not in ['tickers', 'ticker_to_series', 'feature_to_idx', 'share_to_idx']:
                print(f"  {k}: {v}")
        
        return train_df, forecast_input, model_config


def main():
    """Test the data adapter."""
    adapter = DataAdapter(
        data_dir="./test_data",
        y_column="compute_log_return_y",
        prediction_horizon=1
    )
    
    train_df, forecast_input, model_config = adapter.run()


if __name__ == "__main__":
    main()

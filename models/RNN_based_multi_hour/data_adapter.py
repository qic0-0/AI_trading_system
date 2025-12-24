import os
import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAdapter:
    # Configuration embedded directly in code
    BARS_PER_DAY = 7
    OUTPUT_HOURS = 7  # H for model
    Y_COLUMN = "compute_log_return_y"
    PREDICTION_HORIZON = 1
    
    # Aggregation methods for each factor
    AGGREGATIONS = {
        "compute_return_1h": "mean", 
        "compute_return_5h": "mean", 
        "compute_rsi": "last", 
        "compute_volatility": "mean", 
        "compute_macd": "last", 
        "compute_macd_signal": "last", 
        "compute_macd_hist": "last", 
        "compute_volume_ratio": "mean", 
        "compute_price_range": "mean", 
        "compute_sector_return": "mean",
        "compute_spy_return": "mean",
        "compute_vix_level": "last",
        "compute_qqq_return": "mean",
        "compute_iwm_return": "mean",
        "compute_fed_rate": "last",
        "compute_cpi": "last"
    }
    
    def __init__(self, data_dir, tickers, y_column=None, prediction_horizon=None):
        # IMPORTANT: Use the data_dir PARAMETER, not a hardcoded string!
        self.data_dir = data_dir  # This is passed in, e.g., "./test_data"
        self.tickers = tickers
        self.y_column = y_column or self.Y_COLUMN
        self.prediction_horizon = prediction_horizon or self.PREDICTION_HORIZON
        
    def load_features(self):
        logger.info("Loading features...")
        independent_factors = []
        for ticker in self.tickers:
            path = os.path.join(self.data_dir, "features", "independent_factors", f"{ticker}.parquet")
            independent_factors.append(pd.read_parquet(path))
        
        shared_factors_path = os.path.join(self.data_dir, "features", "shared_factors.parquet")
        shared_factors = pd.read_parquet(shared_factors_path)
        
        embeddings = []
        for ticker in self.tickers:
            path = os.path.join(self.data_dir, "features", "embeddings", f"{ticker}.npy")
            embeddings.append(np.load(path))
        
        logger.info("Features loaded.")
        return independent_factors, shared_factors, embeddings
    
    def transform(self, independent_factors, shared_factors, embeddings):
        logger.info("Applying transformations...")
        
        # Apply aggregation methods for resolution conversion
        aggregated_independent_factors = []
        for factor in independent_factors:
            aggregated_factor = factor.resample('D').agg(self.AGGREGATIONS)
            aggregated_independent_factors.append(aggregated_factor)
        
        aggregated_shared_factors = shared_factors.resample('D').agg(self.AGGREGATIONS)
        
        # Shift X forward by prediction_horizon (Y keeps true time index)
        shifted_independent_factors = []
        for factor in aggregated_independent_factors:
            shifted_factor = factor.shift(+self.prediction_horizon)
            shifted_independent_factors.append(shifted_factor)
        
        shifted_shared_factors = aggregated_shared_factors.shift(+self.prediction_horizon)
        
        # Broadcast embeddings to all timestamps
        broadcasted_embeddings = []
        for embedding in embeddings:
            broadcasted_embedding = np.broadcast_to(embedding, (len(aggregated_independent_factors[0]), embedding.shape[0]))
            broadcasted_embeddings.append(broadcasted_embedding)
        
        # Rename columns to match model's expected format
        transformed_independent_factors = []
        for i, factor in enumerate(shifted_independent_factors):
            factor.columns = [f"x_{j}_s{i}" for j in range(len(factor.columns))]
            transformed_independent_factors.append(factor)
        
        transformed_shared_factors.columns = [f"share_{j}" for j in range(len(transformed_shared_factors.columns))]
        
        transformed_embeddings = []
        for i, embedding in enumerate(broadcasted_embeddings):
            embedding_df = pd.DataFrame(embedding)
            embedding_df.columns = [f"emb_0_s{i}_d{j}" for j in range(len(embedding_df.columns))]
            transformed_embeddings.append(embedding_df)
        
        # Create Y values
        y_values = []
        for i, factor in enumerate(aggregated_independent_factors):
            y_value = factor[self.y_column]
            y_value.columns = [f"y_{j}_s{i}" for j in range(self.OUTPUT_HOURS)]
            y_values.append(y_value)
        
        logger.info("Transformations applied.")
        return transformed_independent_factors, transformed_shared_factors, transformed_embeddings, y_values
    
    def get_train_test_split(self, data, test_ratio=0.2):
        logger.info("Splitting data into train and test sets...")
        train_size = int((1 - test_ratio) * len(data))
        train_data, test_data = data[:train_size], data[train_size:]
        logger.info("Data split into train and test sets.")
        return train_data, test_data
    
    def get_model_config(self):
        return {
            "H": self.OUTPUT_HOURS,
            "bars_per_day": self.BARS_PER_DAY,
            "prediction_horizon": self.prediction_horizon
        }
    
    def run(self):
        independent_factors, shared_factors, embeddings = self.load_features()
        transformed_independent_factors, transformed_shared_factors, transformed_embeddings, y_values = self.transform(independent_factors, shared_factors, embeddings)
        
        # Combine data into a single DataFrame
        data = pd.concat(transformed_independent_factors + [transformed_shared_factors] + transformed_embeddings + y_values, axis=1)
        
        # Drop NaN values
        data = data.dropna()
        
        # Split data into train and test sets
        train_data, test_data = self.get_train_test_split(data)
        
        return train_data, test_data

# Usage
adapter = DataAdapter("./test_data", ["AAPL", "JNJ"])
train_data, test_data = adapter.run()
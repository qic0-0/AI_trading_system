"""
Configuration settings for the Quant Trading Agent System.

This module contains all configuration parameters including:
- LLM API settings (Argonne/OpenAI compatible)
- Data source API keys
- Portfolio constraints
- Model parameters
"""

from dataclasses import dataclass
from typing import Optional
import os

os.environ.setdefault('FINNHUB_API_KEY', 'd4o83k9r01quuso86nr0d4o83k9r01quuso86nrg')
os.environ.setdefault('FRED_API_KEY', '58fa5225eb36955a2517348c39979576')

@dataclass
class LLMConfig:
    """Configuration for LLM API access."""

    # Argonne API endpoint
    api_base_url: str = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
    api_key: str = ""  # Will be fetched via inference_auth_token.get_access_token()
    model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    temperature: float = 0.7
    max_tokens: int = 2000
    max_retries: int = 3
    timeout: int = 60


@dataclass
class DataSourceConfig:
    """Configuration for data source APIs."""
    
    # Yahoo Finance - no API key needed
    yfinance_enabled: bool = True
    
    # Finnhub for news
    finnhub_api_key: str = os.getenv("FINNHUB_API_KEY", "")
    
    # Alpha Vantage for news (alternative)
    alpha_vantage_api_key: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    
    # FRED for economic data
    fred_api_key: str = os.getenv("FRED_API_KEY", "")


@dataclass
class PortfolioConfig:
    """Configuration for portfolio constraints."""
    
    initial_cash: float = 100000.0
    max_position_pct: float = 0.80  # Max 30% in single position
    min_cash_reserve_pct: float = 0.05  # Keep 10% in cash
    allow_shorting: bool = True  # Long-only strategy

@dataclass
class ModelConfig:
    """Configuration for quant model."""

    model_type: str = "hmm"  # "hmm" or "xgboost"
    model_path: str = "models/"  # Directory for model files
    design_doc_path: str = "config/model_design.md"  # HMM design
    xgboost_design_doc_path: str = "config/model_design_xgboost.md"  # XGBoost design
    retrain_frequency_days: int = 30
    n_states: int = 3  # Bull, Bear, Sideways (HMM only)
    prediction_horizon: int = 5  # 5-day returns


@dataclass
class RAGConfig:
    """Configuration for RAG knowledge base."""

    vector_db_path: str = "knowledge/vector_store"
    embedding_model: str = "all-MiniLM-L6-v2"  # sentence-transformers model
    chunk_size: int = 500
    top_k_results: int = 5
    news_top_k: int = 5
    knowledge_top_k: int = 3
    historical_events_top_k: int = 2


@dataclass
class SystemConfig:
    """Main system configuration combining all configs."""

    llm: LLMConfig = None
    data_sources: DataSourceConfig = None
    portfolio: PortfolioConfig = None
    model: ModelConfig = None
    rag: RAGConfig = None

    # Default tickers to trade
    tickers: list = None

    # Trading frequency
    trading_frequency_days: int = 5

    def __post_init__(self):
        self.llm = self.llm or LLMConfig()
        self.data_sources = self.data_sources or DataSourceConfig()
        self.portfolio = self.portfolio or PortfolioConfig()
        self.model = self.model or ModelConfig()
        self.rag = self.rag or RAGConfig()
        self.tickers = self.tickers or ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"]


# Global config instance
config = SystemConfig()

"""
Feature Engineering Agent - Agent 2 in the Quant Trading System.

Responsibilities:
1. Register factor computation tools from factor_formulas.py
2. Parse user requests for specific factors
3. Check tool requirements against data dictionary
4. Load required data from Data Agent outputs
5. Compute factors using registered tools
6. Save outputs: independent_factors.parquet, shared_factors.parquet, embeddings/

Flow:
    User Request → Get Tool Requirements → Check Data Dictionary → Load Data → Compute Factors → Save

Usage:
    from feature_agent import FeatureAgent

    agent = FeatureAgent(config, data_dir="./data")
    result = agent.run({
        "tickers": ["AAPL", "MSFT"],
        "factors": ["compute_rsi", "compute_macd", "compute_spy_return"],
        "compute_y": True
    })
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Set
from datetime import datetime

# Import tool registry from factor_formulas
from .factor_formulas import (
    TOOLS,
    get_tool,
    call_tool,
    list_tools,
    get_tool_requirements
)

logger = logging.getLogger(__name__)


class FeatureAgent:
    """
    Agent responsible for computing factors from raw data.

    Output Structure:
        data/
        ├── features/
        │   ├── independent_factors/    # Per ticker factors
        │   │   ├── AAPL.parquet
        │   │   └── MSFT.parquet
        │   ├── shared_factors.parquet  # Shared across all tickers
        │   └── embeddings/             # Text embeddings per ticker
        │       ├── AAPL.npy
        │       └── MSFT.npy
    """

    def __init__(self, config, data_dir: str = "./data"):
        """
        Initialize Feature Agent.

        Args:
            config: SystemConfig object or dict
            data_dir: Base directory for data (same as Data Agent)
        """
        self.config = config
        self.data_dir = data_dir
        self.data_dictionary = None

        # Get embedding config from config
        if hasattr(config, 'rag'):
            # SystemConfig object
            self.embedding_api_base_url = getattr(config.rag, 'embedding_api_base_url',
                "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1")
            self.embedding_model = getattr(config.rag, 'embedding_model',
                "mistralai/Mistral-7B-Instruct-v0.3-embed")
        elif isinstance(config, dict):
            # Dict config
            self.embedding_api_base_url = config.get("embedding_api_base_url",
                "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1")
            self.embedding_model = config.get("embedding_model",
                "mistralai/Mistral-7B-Instruct-v0.3-embed")
        else:
            self.embedding_api_base_url = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
            self.embedding_model = "mistralai/Mistral-7B-Instruct-v0.3-embed"

        # Create output directories
        self._create_directories()

        # Load data dictionary
        self._load_data_dictionary()

        # All registered tools are available from factor_formulas
        logger.info(f"Feature Agent initialized with {len(TOOLS)} registered tools")

    def _create_directories(self):
        """Create output directories."""
        subdirs = [
            "features/independent_factors",
            "features/embeddings"
        ]
        for subdir in subdirs:
            path = os.path.join(self.data_dir, subdir)
            os.makedirs(path, exist_ok=True)

    def _load_data_dictionary(self):
        """Load data dictionary from Data Agent."""
        dict_path = os.path.join(self.data_dir, "data_dictionary.json")
        if os.path.exists(dict_path):
            with open(dict_path, 'r') as f:
                self.data_dictionary = json.load(f)
            logger.info("Loaded data dictionary")
        else:
            logger.warning("Data dictionary not found. Run Data Agent first.")
            self.data_dictionary = {}

    def _check_data_availability(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if required data is available in data dictionary.

        Args:
            requirements: Output from get_tool_requirements()

        Returns:
            Dict with availability status and missing data
        """
        available = {"status": True, "missing": [], "available": []}

        if not self.data_dictionary:
            available["status"] = False
            available["missing"].append("Data dictionary not loaded")
            return available

        datasets = self.data_dictionary.get("datasets", {})

        # Check price_data
        if requirements["price_data"]["columns"]:
            if "price_data" in datasets:
                available["available"].append(f"price_data: {requirements['price_data']['columns']}")
            else:
                available["status"] = False
                available["missing"].append("price_data not found")

        # Check market_data
        if requirements["market_data"]["tickers"]:
            if "market_data" in datasets:
                available["available"].append(f"market_data: {requirements['market_data']['tickers']}")
            else:
                available["status"] = False
                available["missing"].append("market_data not found")

        # Check economic
        if requirements["economic"]["columns"]:
            if "economic_indicators" in datasets:
                available["available"].append(f"economic: {requirements['economic']['columns']}")
            else:
                available["status"] = False
                available["missing"].append("economic_indicators not found")

        # Check company_profiles
        if requirements["company_profiles"]["columns"]:
            if "company_profiles" in datasets:
                available["available"].append(f"company_profiles: {requirements['company_profiles']['columns']}")
            else:
                available["status"] = False
                available["missing"].append("company_profiles not found")

        return available

    def _load_price_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load price data for a ticker."""
        path = os.path.join(self.data_dir, "price_data", f"{ticker}.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)
        logger.warning(f"Price data not found for {ticker}")
        return None

    def _load_market_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load market data (SPY, VIX, QQQ, IWM, sector ETFs)."""
        path = os.path.join(self.data_dir, "market_data", f"{ticker}.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)
        logger.warning(f"Market data not found for {ticker}")
        return None

    def _load_economic_data(self) -> Optional[pd.DataFrame]:
        """Load economic indicators."""
        path = os.path.join(self.data_dir, "economic", "indicators.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)
        logger.warning("Economic data not found")
        return None

    def _load_company_profile(self, ticker: str) -> Optional[Dict]:
        """Load company profile for a ticker."""
        path = os.path.join(self.data_dir, "company_profiles", f"{ticker}.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        logger.warning(f"Company profile not found for {ticker}")
        return None

    def _load_fundamentals(self, ticker: str) -> Optional[Dict]:
        """Load fundamentals to get sector info."""
        path = os.path.join(self.data_dir, "fundamentals", f"{ticker}.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None

    def _get_sector_etf(self, ticker: str) -> Optional[str]:
        """Get sector ETF for a ticker."""
        fundamentals = self._load_fundamentals(ticker)
        if fundamentals:
            sector = fundamentals.get("sector")
            # Sector ETF mapping (same as data_agent)
            sector_etf_map = {
                "Technology": "XLK",
                "Financial Services": "XLF",
                "Energy": "XLE",
                "Healthcare": "XLV",
                "Industrials": "XLI",
                "Consumer Defensive": "XLP",
                "Consumer Cyclical": "XLY",
                "Utilities": "XLU",
                "Basic Materials": "XLB",
                "Real Estate": "XLRE",
                "Communication Services": "XLC",
            }
            return sector_etf_map.get(sector)
        return None

    def _compute_independent_factors(
        self,
        ticker: str,
        factor_tools: List[str],
        price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute independent factors for a single ticker.

        Args:
            ticker: Stock ticker
            factor_tools: List of tool names to compute
            price_df: Price DataFrame for the ticker

        Returns:
            DataFrame with computed factors
        """
        factors = pd.DataFrame(index=price_df.index)

        for tool_name in factor_tools:
            tool = get_tool(tool_name)
            req = tool.get("requires", {})

            # Skip if not independent
            if tool["output_type"] != "independent":
                continue

            # Check if this tool needs price_data
            if req.get("data") != "price_data":
                continue

            try:
                # Handle special case: sector_return needs extra data
                if tool_name == "compute_sector_return":
                    sector_etf = self._get_sector_etf(ticker)
                    if sector_etf:
                        sector_df = self._load_market_data(sector_etf)
                        if sector_df is not None:
                            factors[tool_name] = call_tool(
                                tool_name,
                                ticker_df=price_df,
                                sector_etf_df=sector_df
                            )
                else:
                    # Standard call with price_df
                    factors[tool_name] = call_tool(tool_name, df=price_df)

                logger.info(f"  Computed {tool_name} for {ticker}")

            except Exception as e:
                logger.error(f"  Error computing {tool_name} for {ticker}: {e}")

        return factors

    def _compute_shared_factors(
        self,
        factor_tools: List[str],
        target_index: pd.Index
    ) -> pd.DataFrame:
        """
        Compute shared factors (same for all tickers).

        Args:
            factor_tools: List of tool names to compute
            target_index: Target datetime index for alignment

        Returns:
            DataFrame with computed shared factors
        """
        factors = pd.DataFrame(index=target_index)

        # Load market data once
        spy_df = self._load_market_data("SPY")
        vix_df = self._load_market_data("VIX")
        qqq_df = self._load_market_data("QQQ")
        iwm_df = self._load_market_data("IWM")
        econ_df = self._load_economic_data()

        for tool_name in factor_tools:
            tool = get_tool(tool_name)

            # Skip if not shared
            if tool["output_type"] != "shared":
                continue

            try:
                req = tool.get("requires", {})

                if tool_name == "compute_spy_return" and spy_df is not None:
                    result = call_tool(tool_name, spy_df=spy_df)
                    factors[tool_name] = result.reindex(target_index, method='ffill')

                elif tool_name == "compute_vix_level" and vix_df is not None:
                    result = call_tool(tool_name, vix_df=vix_df)
                    factors[tool_name] = result.reindex(target_index, method='ffill')

                elif tool_name == "compute_qqq_return" and qqq_df is not None:
                    result = call_tool(tool_name, qqq_df=qqq_df)
                    factors[tool_name] = result.reindex(target_index, method='ffill')

                elif tool_name == "compute_iwm_return" and iwm_df is not None:
                    result = call_tool(tool_name, iwm_df=iwm_df)
                    factors[tool_name] = result.reindex(target_index, method='ffill')

                elif tool_name == "compute_fed_rate" and econ_df is not None:
                    factors[tool_name] = call_tool(
                        tool_name,
                        econ_df=econ_df,
                        target_index=target_index
                    )

                elif tool_name == "compute_cpi" and econ_df is not None:
                    factors[tool_name] = call_tool(
                        tool_name,
                        econ_df=econ_df,
                        target_index=target_index
                    )

                logger.info(f"  Computed {tool_name}")

            except Exception as e:
                logger.error(f"  Error computing {tool_name}: {e}")

        return factors

    def _embedding_is_fresh(self, ticker: str, max_age_days: int = 30) -> bool:
        """Check if embedding file exists and is less than max_age_days old."""
        import time
        path = os.path.join(self.data_dir, "features", "embeddings", f"{ticker}.npy")
        if not os.path.exists(path):
            return False
        file_age_days = (time.time() - os.path.getmtime(path)) / (60 * 60 * 24)
        return file_age_days < max_age_days

    def _compute_embeddings(
        self,
        tickers: List[str],
        factor_tools: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Compute text embeddings for company profiles using Argonne API.

        Args:
            tickers: List of tickers
            factor_tools: List of tool names

        Returns:
            Dict mapping ticker to embedding vector
        """
        embeddings = {}

        # Check if embedding tools are requested
        embedding_tools = [t for t in factor_tools if get_tool(t)["output_type"] == "embedding"]
        if not embedding_tools:
            return embeddings

        if "compute_company_embedding" in embedding_tools:
            for ticker in tickers:
                # Skip if embedding is fresh (< 30 days old)
                if self._embedding_is_fresh(ticker):
                    # Load existing embedding
                    path = os.path.join(self.data_dir, "features", "embeddings", f"{ticker}.npy")
                    embeddings[ticker] = np.load(path)
                    logger.info(f"  Skipping {ticker} embedding (fresh, < 30 days old)")
                    continue

                profile = self._load_company_profile(ticker)
                if profile:
                    try:
                        embedding = call_tool(
                            "compute_company_embedding",
                            profile=profile,
                            api_base_url=self.embedding_api_base_url,
                            model=self.embedding_model,
                            api_key=None  # Will use inference_auth_token
                        )
                        if len(embedding) > 0:
                            embeddings[ticker] = embedding
                            logger.info(f"  Computed embedding for {ticker}")
                    except Exception as e:
                        logger.error(f"  Error computing embedding for {ticker}: {e}")

        return embeddings

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run feature engineering for specified tickers and factors.

        Args:
            input_data: {
                "tickers": List[str],           # Stock basket
                "factors": List[str],           # Tool names to compute (or "all")
                "compute_embeddings": bool      # Include embeddings (default False)
            }

        Returns:
            Dict with computed factor paths and logs
            On error: {"success": False, "failed_step": str, "error": str, "logs": list}
        """
        tickers = input_data.get("tickers", [])
        requested_factors = input_data.get("factors", "all")
        compute_embeddings = input_data.get("compute_embeddings", False)

        logs = []
        result = {
            "independent_factors": {},
            "shared_factors_path": None,
            "embeddings": {}
        }

        # -----------------------------------------------------------------
        # 1. Determine which tools to use
        # -----------------------------------------------------------------
        try:
            logs.append("=== Determining Tools to Use ===")

            if requested_factors == "all":
                factor_tools = list_tools()  # All tools
            else:
                factor_tools = list(requested_factors)

            # Add embedding tool if requested
            if compute_embeddings and "compute_company_embedding" not in factor_tools:
                factor_tools = factor_tools + ["compute_company_embedding"]

            logs.append(f"Selected {len(factor_tools)} tools: {factor_tools}")

        except Exception as e:
            return {
                "success": False,
                "failed_step": "Determining Tools",
                "error": str(e),
                "logs": logs
            }

        # -----------------------------------------------------------------
        # 2. Get tool requirements
        # -----------------------------------------------------------------
        try:
            logs.append("=== Checking Tool Requirements ===")
            requirements = get_tool_requirements(factor_tools)
            logs.append(f"Requirements: {requirements}")

        except Exception as e:
            return {
                "success": False,
                "failed_step": "Getting Requirements",
                "error": str(e),
                "logs": logs
            }

        # -----------------------------------------------------------------
        # 3. Check data availability
        # -----------------------------------------------------------------
        try:
            logs.append("=== Checking Data Availability ===")
            availability = self._check_data_availability(requirements)

            if not availability["status"]:
                logs.append(f"Missing data: {availability['missing']}")
                raise ValueError(f"Missing required data: {availability['missing']}")

            logs.append(f"Available: {availability['available']}")

        except Exception as e:
            return {
                "success": False,
                "failed_step": "Checking Data Availability",
                "error": str(e),
                "logs": logs
            }

        # -----------------------------------------------------------------
        # 4. Compute independent factors (per ticker)
        # -----------------------------------------------------------------
        try:
            logs.append("=== Computing Independent Factors ===")

            for ticker in tickers:
                logs.append(f"Processing {ticker}...")

                # Load price data
                price_df = self._load_price_data(ticker)
                if price_df is None:
                    raise ValueError(f"Price data not found for {ticker}")

                # Compute factors
                factors_df = self._compute_independent_factors(ticker, factor_tools, price_df)

                # Save
                path = os.path.join(self.data_dir, "features", "independent_factors", f"{ticker}.parquet")
                factors_df.to_parquet(path)
                result["independent_factors"][ticker] = path
                logs.append(f"  Saved {len(factors_df.columns)} factors for {ticker}")

        except Exception as e:
            return {
                "success": False,
                "failed_step": "Computing Independent Factors",
                "error": str(e),
                "logs": logs
            }

        # -----------------------------------------------------------------
        # 5. Compute shared factors (once for all tickers)
        # -----------------------------------------------------------------
        try:
            logs.append("=== Computing Shared Factors ===")

            # Use first ticker's index as reference
            first_ticker = tickers[0]
            price_df = self._load_price_data(first_ticker)
            target_index = price_df.index

            # Compute shared factors
            shared_df = self._compute_shared_factors(factor_tools, target_index)

            if not shared_df.empty:
                path = os.path.join(self.data_dir, "features", "shared_factors.parquet")
                shared_df.to_parquet(path)
                result["shared_factors_path"] = path
                logs.append(f"  Saved {len(shared_df.columns)} shared factors")
            else:
                logs.append("  No shared factors requested")

        except Exception as e:
            return {
                "success": False,
                "failed_step": "Computing Shared Factors",
                "error": str(e),
                "logs": logs
            }

        # -----------------------------------------------------------------
        # 6. Compute embeddings (if requested)
        # -----------------------------------------------------------------
        if compute_embeddings:
            try:
                logs.append("=== Computing Embeddings ===")

                embeddings = self._compute_embeddings(tickers, factor_tools)

                for ticker, embedding in embeddings.items():
                    path = os.path.join(self.data_dir, "features", "embeddings", f"{ticker}.npy")
                    np.save(path, embedding)
                    result["embeddings"][ticker] = path

                logs.append(f"  Saved embeddings for {len(embeddings)} tickers")

            except Exception as e:
                return {
                    "success": False,
                    "failed_step": "Computing Embeddings",
                    "error": str(e),
                    "logs": logs
                }

        # -----------------------------------------------------------------
        # 7. Generate feature dictionary
        # -----------------------------------------------------------------

        try:
            logs.append("=== Generating Feature Dictionary ===")

            independent_factors = [t for t in factor_tools if get_tool(t)["output_type"] == "independent"]
            shared_factors = [t for t in factor_tools if get_tool(t)["output_type"] == "shared"]

            feature_dictionary = {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tickers": tickers,
                "datasets": {
                    "independent_factors": {
                        "description": "Per-ticker factors computed from price data",
                        "scope": "Per ticker",
                        "frequency": "Hourly",
                        "source": "price_data",
                        "path_pattern": "features/independent_factors/{ticker}.parquet",
                        "factors": {
                            tool_name: get_tool(tool_name)["description"]
                            for tool_name in independent_factors
                        },
                        "tickers": list(result["independent_factors"].keys())
                    },
                    "shared_factors": {
                        "description": "Market and economic factors shared across all tickers",
                        "scope": "Shared (all tickers)",
                        "frequency": "Hourly",
                        "source": "market_data, economic",
                        "path": "features/shared_factors.parquet",
                        "factors": {
                            tool_name: get_tool(tool_name)["description"]
                            for tool_name in shared_factors
                        }
                    },
                    "embeddings": {
                        "description": "Text embeddings from company profiles",
                        "scope": "Per ticker",
                        "frequency": "Monthly",
                        "source": "company_profiles",
                        "path_pattern": "features/embeddings/{ticker}.npy",
                        "dimension": 384,
                        "model": self.embedding_model if hasattr(self, 'embedding_model') else "all-MiniLM-L6-v2",
                        "tickers": list(result["embeddings"].keys())
                    }
                },
                "scope_summary": {
                    "per_ticker": [
                        "independent_factors",
                        "embeddings"
                    ],
                    "shared_all": [
                        "shared_factors (SPY, VIX, QQQ, IWM returns, Fed rate, CPI)"
                    ]
                },
                "tools_used": factor_tools
            }

            dict_path = os.path.join(self.data_dir, "features", "feature_dictionary.json")
            with open(dict_path, 'w') as f:
                json.dump(feature_dictionary, f, indent=2)

            logs.append(f"Feature dictionary saved to {dict_path}")

        except Exception as e:
            return {
                "success": False,
                "failed_step": "Generating Feature Dictionary",
                "error": str(e),
                "logs": logs
            }

        logs.append("=== Feature Engineering Complete ===")
        return {
            "success": True,
            "feature_paths": result,
            "feature_dictionary": feature_dictionary,
            "logs": logs
        }

    def get_available_tools(self) -> Dict[str, List[str]]:
        """Get all available tools grouped by type."""
        return {
            "independent": list_tools("independent"),
            "shared": list_tools("shared"),
            "embedding": list_tools("embedding")
        }

    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get detailed info about a specific tool."""
        return get_tool(tool_name)




# =============================================================================
# Note: Run tests using test/test_feature_agent.py
# Direct execution won't work due to relative imports
# =============================================================================
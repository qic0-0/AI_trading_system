"""
Data Agent - Agent 1 in the Quant Trading System.

Responsibilities:
1. General Data:
   - Stock prices (OHLCV) - Hourly, per ticker
   - Market indices (SPY, VIX, QQQ, IWM) - Hourly, shared
   - Sector ETFs - Hourly, shared by sector
   - Fundamentals (P/E, P/B, etc.) - Daily, per ticker
   - Economic indicators (FRED) - Daily, shared

2. Text Data (for embedding):
   - Company profile (description, sector, industry) - Monthly, per ticker

Data Sources:
- yfinance: Prices, fundamentals, company profiles
- FRED: Economic indicators

Features:
- Chunked fetching: Fetches data in 1-year intervals (for future API with more data)
- Smart caching: Checks local data first, only fetches missing data
- Modular design: Easy to swap API in future

Note: Yahoo Finance only stores ~2 years of hourly data.
      Chunking/caching code is ready for future stronger APIs.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import os
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Data Source Abstraction (Easy to swap API)
# =============================================================================

class BaseDataSource(ABC):
    """Abstract base class for data sources. Implement this to add new data providers."""

    @abstractmethod
    def fetch_hourly_prices(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch hourly OHLCV data."""
        pass

    @abstractmethod
    def fetch_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Fetch fundamental data."""
        pass

    @abstractmethod
    def fetch_company_profile(self, ticker: str) -> Dict[str, Any]:
        """Fetch company text profile for embedding."""
        pass


class YFinanceDataSource(BaseDataSource):
    """Yahoo Finance data source implementation."""

    # Chunk size for fetching (365 days, safe under 730 limit)
    CHUNK_DAYS = 365

    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

    def fetch_hourly_prices(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch hourly OHLCV data using yfinance.

        Automatically splits into 1-year chunks to avoid 730-day limit.

        Note: yfinance 1h interval limit is ~730 days (~2 years)
        Market hours: 9:30, 10:30, 11:30, 12:30, 13:30, 14:30, 15:30

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data (hourly)
        """
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            # Split into chunks
            chunks = self._generate_date_chunks(start_dt, end_dt)
            logger.info(f"Fetching {ticker} in {len(chunks)} chunk(s)")

            all_data = []
            stock = self.yf.Ticker(ticker)

            for chunk_start, chunk_end in chunks:
                chunk_start_str = chunk_start.strftime("%Y-%m-%d")
                chunk_end_str = chunk_end.strftime("%Y-%m-%d")

                logger.info(f"  Fetching chunk: {chunk_start_str} to {chunk_end_str}")

                df = stock.history(start=chunk_start_str, end=chunk_end_str, interval="1h")

                if not df.empty:
                    all_data.append(df)

            if not all_data:
                logger.warning(f"No hourly price data found for {ticker}")
                return pd.DataFrame()

            # Combine all chunks
            combined_df = pd.concat(all_data)

            # Remove duplicates (in case of overlapping chunks)
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

            # Sort by index
            combined_df = combined_df.sort_index()

            # Keep standard columns
            combined_df = combined_df[['Open', 'High', 'Low', 'Close', 'Volume']]

            # Filter to market hours only (9:30 AM - 4:00 PM ET)
            combined_df = combined_df.between_time('09:30', '15:30')

            logger.info(f"Fetched {len(combined_df)} hourly bars for {ticker}")
            return combined_df

        except Exception as e:
            logger.error(f"Error fetching hourly prices for {ticker}: {e}")
            return pd.DataFrame()

    def _generate_date_chunks(self, start_dt: datetime, end_dt: datetime) -> List[Tuple[datetime, datetime]]:
        """
        Generate list of date chunks (1 year each).

        Args:
            start_dt: Start datetime
            end_dt: End datetime

        Returns:
            List of (chunk_start, chunk_end) tuples
        """
        chunks = []
        current_start = start_dt

        while current_start < end_dt:
            chunk_end = current_start + timedelta(days=self.CHUNK_DAYS)
            if chunk_end > end_dt:
                chunk_end = end_dt
            chunks.append((current_start, chunk_end))
            current_start = chunk_end

        return chunks

    def fetch_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch fundamental data using yfinance.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with P/E, P/B, market cap, etc.
        """
        try:
            stock = self.yf.Ticker(ticker)
            info = stock.info

            fundamentals = {
                "ticker": ticker,
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "pb_ratio": info.get("priceToBook"),
                "market_cap": info.get("marketCap"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "avg_volume": info.get("averageVolume"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            logger.info(f"Fetched fundamentals for {ticker}")
            return fundamentals

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}

    def fetch_company_profile(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch company text profile for embedding.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with business description, sector, industry
        """
        try:
            stock = self.yf.Ticker(ticker)
            info = stock.info

            profile = {
                "ticker": ticker,
                "company_name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "business_summary": info.get("longBusinessSummary", ""),
                "website": info.get("website", ""),
                "country": info.get("country", ""),
                "employees": info.get("fullTimeEmployees"),
                "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            logger.info(f"Fetched company profile for {ticker}")
            return profile

        except Exception as e:
            logger.error(f"Error fetching company profile for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}


# =============================================================================
# FRED Data Source
# =============================================================================

class FREDDataSource:
    """FRED economic data source."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_economic_indicators(
        self,
        indicators: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, pd.Series]:
        """
        Fetch economic indicators from FRED.

        Args:
            indicators: List of FRED codes (e.g., "FEDFUNDS", "CPIAUCSL")
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            Dict mapping indicator code to time series
        """
        if not self.api_key:
            logger.warning("FRED API key not configured, skipping economic indicators")
            return {}

        try:
            from fredapi import Fred
            fred = Fred(api_key=self.api_key)

            result = {}
            for indicator in indicators:
                try:
                    series = fred.get_series(indicator, start_date, end_date)
                    result[indicator] = series
                    logger.info(f"Fetched {len(series)} data points for {indicator}")
                except Exception as e:
                    logger.error(f"Error fetching {indicator}: {e}")
                    result[indicator] = pd.Series()

            return result

        except ImportError:
            logger.error("fredapi not installed. Run: pip install fredapi")
            return {}
        except Exception as e:
            logger.error(f"Error initializing FRED client: {e}")
            return {}


# =============================================================================
# Sector ETF Mapping
# =============================================================================

SECTOR_ETF_MAP = {
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

# Market indices (always fetched)
MARKET_INDICES = ["SPY", "QQQ", "IWM"]
VOLATILITY_INDEX = "^VIX"


# =============================================================================
# Data Agent
# =============================================================================

class DataAgent:
    """
    Agent responsible for collecting raw data from various sources.

    Output Structure:
        data/
        ├── price_data/          # Hourly OHLCV (per ticker)
        ├── market_data/         # Hourly indices & ETFs (shared)
        ├── company_profiles/    # Text for embedding (per ticker, monthly)
        ├── fundamentals/        # Daily (per ticker)
        └── economic/            # Daily (shared)

    Features:
        - Chunked fetching: 1-year intervals to avoid API limits
        - Smart caching: Only fetches missing data, preserves existing
    """

    def __init__(self, config, data_dir: str = "./data"):
        """
        Initialize Data Agent.

        Args:
            config: SystemConfig object (from config.py) or dict with:
                - data_sources.fred_api_key: FRED API key
            data_dir: Base directory for data storage
        """
        self.config = config
        self.data_dir = data_dir

        # Get FRED API key from config
        # Support both SystemConfig object and plain dict
        if hasattr(config, 'data_sources'):
            # SystemConfig object
            fred_api_key = config.data_sources.fred_api_key
        elif isinstance(config, dict):
            # Plain dict
            fred_api_key = config.get("fred_api_key", "")
        else:
            fred_api_key = ""

        # Initialize data sources
        self.price_source = YFinanceDataSource()
        self.fred_source = FREDDataSource(fred_api_key)

        # Create data directories
        self._create_directories()

    def _create_directories(self):
        """Create data storage directories."""
        subdirs = [
            "price_data",
            "market_data",
            "company_profiles",
            "fundamentals",
            "economic"
        ]
        for subdir in subdirs:
            path = os.path.join(self.data_dir, subdir)
            os.makedirs(path, exist_ok=True)

    def _get_local_price_data(self, ticker: str, folder: str = "price_data") -> Optional[pd.DataFrame]:
        """
        Load existing local price data if available.

        Args:
            ticker: Stock ticker symbol
            folder: Subfolder name (price_data or market_data)

        Returns:
            DataFrame if exists, None otherwise
        """
        path = os.path.join(self.data_dir, folder, f"{ticker}.parquet")
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                logger.info(f"Found local data for {ticker}: {len(df)} rows")
                return df
            except Exception as e:
                logger.warning(f"Error reading local data for {ticker}: {e}")
                return None
        return None

    def _merge_price_data(
        self,
        local_df: Optional[pd.DataFrame],
        new_df: pd.DataFrame,
        request_start: datetime,
        request_end: datetime
    ) -> pd.DataFrame:
        """
        Merge local and new price data intelligently.

        Logic:
        - Keep local data that's earlier than request start
        - Keep local data that's within request range (avoid re-fetch)
        - Add new data for gaps
        - Remove duplicates

        Args:
            local_df: Existing local DataFrame (or None)
            new_df: Newly fetched DataFrame
            request_start: Requested start date
            request_end: Requested end date

        Returns:
            Merged DataFrame
        """
        if local_df is None or local_df.empty:
            return new_df

        if new_df.empty:
            return local_df

        # Combine both DataFrames
        combined = pd.concat([local_df, new_df])

        # Remove duplicates, keep first (prefer existing data)
        combined = combined[~combined.index.duplicated(keep='first')]

        # Sort by index
        combined = combined.sort_index()

        return combined

    def _smart_fetch_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        folder: str = "price_data"
    ) -> pd.DataFrame:
        """
        Smart fetch with local caching.

        1. Check if local file exists
        2. Determine what data is missing
        3. Fetch only missing data
        4. Merge and return

        Args:
            ticker: Stock ticker symbol
            start_date: Requested start date (YYYY-MM-DD)
            end_date: Requested end date (YYYY-MM-DD)
            folder: Subfolder name

        Returns:
            DataFrame with complete data
        """
        request_start = datetime.strptime(start_date, "%Y-%m-%d")
        request_end = datetime.strptime(end_date, "%Y-%m-%d")

        # Load local data
        local_df = self._get_local_price_data(ticker, folder)

        if local_df is not None and not local_df.empty:
            # Determine local data range
            local_start = local_df.index.min().to_pydatetime().replace(tzinfo=None)
            local_end = local_df.index.max().to_pydatetime().replace(tzinfo=None)

            logger.info(f"Local data range for {ticker}: {local_start.date()} to {local_end.date()}")
            logger.info(f"Requested range: {request_start.date()} to {request_end.date()}")

            # Determine what needs to be fetched
            fetch_ranges = []

            # Need earlier data?
            if request_start < local_start:
                fetch_ranges.append((request_start, local_start - timedelta(days=1)))
                logger.info(f"  Need to fetch earlier: {request_start.date()} to {(local_start - timedelta(days=1)).date()}")

            # Need later data?
            if request_end > local_end:
                fetch_ranges.append((local_end + timedelta(days=1), request_end))
                logger.info(f"  Need to fetch later: {(local_end + timedelta(days=1)).date()} to {request_end.date()}")

            if not fetch_ranges:
                logger.info(f"Local data is complete for {ticker}, no fetch needed")
                return local_df

            # Fetch missing ranges
            new_dfs = []
            for fetch_start, fetch_end in fetch_ranges:
                fetch_start_str = fetch_start.strftime("%Y-%m-%d")
                fetch_end_str = fetch_end.strftime("%Y-%m-%d")
                logger.info(f"Fetching missing data: {fetch_start_str} to {fetch_end_str}")

                new_df = self.price_source.fetch_hourly_prices(ticker, fetch_start_str, fetch_end_str)
                if not new_df.empty:
                    new_dfs.append(new_df)

            # Merge all data
            if new_dfs:
                all_new = pd.concat(new_dfs)
                merged = self._merge_price_data(local_df, all_new, request_start, request_end)
            else:
                merged = local_df

            return merged
        else:
            # No local data, fetch everything
            logger.info(f"No local data for {ticker}, fetching full range")
            return self.price_source.fetch_hourly_prices(ticker, start_date, end_date)

    def _smart_fetch_economic(
        self,
        indicators: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Smart fetch economic indicators with local caching.

        Args:
            indicators: List of FRED codes
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with economic indicators
        """
        path = os.path.join(self.data_dir, "economic", "indicators.parquet")
        request_start = datetime.strptime(start_date, "%Y-%m-%d")
        request_end = datetime.strptime(end_date, "%Y-%m-%d")

        # Load local data if exists
        local_df = None
        if os.path.exists(path):
            try:
                local_df = pd.read_parquet(path)
                logger.info(f"Found local economic data: {len(local_df)} rows")
            except Exception as e:
                logger.warning(f"Error reading local economic data: {e}")

        if local_df is not None and not local_df.empty:
            local_start = local_df.index.min().to_pydatetime().replace(tzinfo=None)
            local_end = local_df.index.max().to_pydatetime().replace(tzinfo=None)

            logger.info(f"Local economic data range: {local_start.date()} to {local_end.date()}")
            logger.info(f"Requested range: {request_start.date()} to {request_end.date()}")

            # Check if we need to fetch more data
            fetch_ranges = []

            # Need earlier data?
            if request_start < local_start:
                fetch_ranges.append((request_start, local_start - timedelta(days=1)))

            # Need later data?
            if request_end > local_end:
                fetch_ranges.append((local_end + timedelta(days=1), request_end))

            if not fetch_ranges:
                logger.info("Local economic data is complete, no fetch needed")
                return local_df

            # Fetch missing ranges
            new_dfs = [local_df]
            for fetch_start, fetch_end in fetch_ranges:
                fetch_start_str = fetch_start.strftime("%Y-%m-%d")
                fetch_end_str = fetch_end.strftime("%Y-%m-%d")
                logger.info(f"Fetching missing economic data: {fetch_start_str} to {fetch_end_str}")

                econ_data = self.fred_source.fetch_economic_indicators(
                    indicators, fetch_start_str, fetch_end_str
                )
                if econ_data:
                    new_df = pd.DataFrame(econ_data)
                    new_dfs.append(new_df)

            # Combine all
            combined = pd.concat(new_dfs)
            combined = combined[~combined.index.duplicated(keep='first')]
            combined = combined.sort_index()
            return combined
        else:
            # No local data, fetch everything
            logger.info("No local economic data, fetching full range")
            econ_data = self.fred_source.fetch_economic_indicators(
                indicators, start_date, end_date
            )
            if econ_data:
                return pd.DataFrame(econ_data)
            return pd.DataFrame()

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run data collection for specified tickers and date range.

        Args:
            input_data: {
                "tickers": List[str],           # Stock basket
                "start_date": str,              # YYYY-MM-DD
                "end_date": str,                # YYYY-MM-DD
                "fetch_fundamentals": bool,     # Default True
                "fetch_company_profiles": bool, # Default True
                "economic_indicators": List[str] # FRED codes
            }

        Returns:
            Dict with collected data paths and data dictionary
            On error: {"success": False, "failed_step": str, "error": str, "logs": list}
        """
        tickers = input_data.get("tickers", [])
        start_date = input_data.get("start_date")
        end_date = input_data.get("end_date")
        fetch_fundamentals = input_data.get("fetch_fundamentals", True)
        fetch_company_profiles = input_data.get("fetch_company_profiles", True)
        economic_indicators = input_data.get("economic_indicators", [])

        logs = []
        result = {
            "price_data": {},
            "market_data": {},
            "company_profiles": {},
            "fundamentals": {},
            "economic_indicators": {},
            "sector_etfs": {}
        }

        # -----------------------------------------------------------------
        # Check and adjust for yfinance 730-day limit
        # -----------------------------------------------------------------
        MAX_DAYS = 730
        today = datetime.now()
        earliest_allowed = today - timedelta(days=MAX_DAYS)

        request_start = datetime.strptime(start_date, "%Y-%m-%d")
        request_end = datetime.strptime(end_date, "%Y-%m-%d")

        if request_start < earliest_allowed:
            old_start = start_date
            start_date = earliest_allowed.strftime("%Y-%m-%d")
            logs.append(f"WARNING: yfinance hourly limit is {MAX_DAYS} days")
            logs.append(f"  Requested start: {old_start}")
            logs.append(f"  Adjusted start:  {start_date}")
            logger.warning(f"Adjusted start_date from {old_start} to {start_date} (yfinance {MAX_DAYS}-day limit)")

        logs.append(f"Date range: {start_date} to {end_date}")

        # -----------------------------------------------------------------
        # 1. Fetch stock prices (hourly, per ticker)
        # -----------------------------------------------------------------
        try:
            logs.append("=== Fetching Stock Prices (Hourly) ===")
            for ticker in tickers:
                logs.append(f"Processing {ticker}...")
                df = self._smart_fetch_prices(ticker, start_date, end_date, folder="price_data")
                if df.empty:
                    raise ValueError(f"No price data returned for {ticker}")
                path = os.path.join(self.data_dir, "price_data", f"{ticker}.parquet")
                df.to_parquet(path)
                result["price_data"][ticker] = path
                logs.append(f"  Saved {len(df)} hourly bars for {ticker}")
        except Exception as e:
            return {
                "success": False,
                "failed_step": "Fetching Stock Prices",
                "error": str(e),
                "logs": logs
            }

        # -----------------------------------------------------------------
        # 2. Fetch market indices (hourly, shared)
        # -----------------------------------------------------------------
        try:
            logs.append("=== Fetching Market Indices (Hourly) ===")
            for index in MARKET_INDICES:
                logs.append(f"Processing {index}...")
                df = self._smart_fetch_prices(index, start_date, end_date, folder="market_data")
                if df.empty:
                    raise ValueError(f"No price data returned for {index}")
                path = os.path.join(self.data_dir, "market_data", f"{index}.parquet")
                df.to_parquet(path)
                result["market_data"][index] = path
                logs.append(f"  Saved {len(df)} hourly bars for {index}")

            # VIX (special ticker symbol)
            logs.append(f"Processing VIX...")
            df = self._smart_fetch_prices(VOLATILITY_INDEX, start_date, end_date, folder="market_data")
            if df.empty:
                raise ValueError(f"No price data returned for VIX")
            path = os.path.join(self.data_dir, "market_data", "VIX.parquet")
            df.to_parquet(path)
            result["market_data"]["VIX"] = path
            logs.append(f"  Saved {len(df)} hourly bars for VIX")
        except Exception as e:
            return {
                "success": False,
                "failed_step": "Fetching Market Indices",
                "error": str(e),
                "logs": logs
            }

        # -----------------------------------------------------------------
        # 3. Fetch sector ETFs (hourly, shared by sector)
        # -----------------------------------------------------------------
        try:
            logs.append("=== Fetching Sector ETFs (Hourly) ===")
            required_sectors = self._get_required_sectors(tickers)
            for sector, etf in required_sectors.items():
                logs.append(f"Processing {etf} ({sector})...")
                df = self._smart_fetch_prices(etf, start_date, end_date, folder="market_data")
                if df.empty:
                    raise ValueError(f"No price data returned for {etf} ({sector})")
                path = os.path.join(self.data_dir, "market_data", f"{etf}.parquet")
                df.to_parquet(path)
                result["sector_etfs"][etf] = path
                result["market_data"][etf] = path
                logs.append(f"  Saved {len(df)} hourly bars for {etf}")
        except Exception as e:
            return {
                "success": False,
                "failed_step": "Fetching Sector ETFs",
                "error": str(e),
                "logs": logs
            }

        # -----------------------------------------------------------------
        # 4. Fetch fundamentals (daily, per ticker)
        # -----------------------------------------------------------------
        if fetch_fundamentals:
            try:
                logs.append("=== Fetching Fundamentals (Daily) ===")
                for ticker in tickers:
                    logs.append(f"Fetching fundamentals for {ticker}...")
                    data = self.price_source.fetch_fundamentals(ticker)
                    if "error" in data:
                        raise ValueError(f"Failed to fetch fundamentals for {ticker}: {data['error']}")
                    path = os.path.join(self.data_dir, "fundamentals", f"{ticker}.json")
                    with open(path, 'w') as f:
                        json.dump(data, f, indent=2)
                    result["fundamentals"][ticker] = path
            except Exception as e:
                return {
                    "success": False,
                    "failed_step": "Fetching Fundamentals",
                    "error": str(e),
                    "logs": logs
                }

        # -----------------------------------------------------------------
        # 5. Fetch company profiles (monthly, per ticker)
        # -----------------------------------------------------------------
        if fetch_company_profiles:
            try:
                logs.append("=== Fetching Company Profiles (Monthly) ===")
                for ticker in tickers:
                    if self._should_update_profile(ticker):
                        logs.append(f"Fetching company profile for {ticker}...")
                        data = self.price_source.fetch_company_profile(ticker)
                        if "error" in data:
                            raise ValueError(f"Failed to fetch company profile for {ticker}: {data['error']}")
                        path = os.path.join(self.data_dir, "company_profiles", f"{ticker}.json")
                        with open(path, 'w') as f:
                            json.dump(data, f, indent=2)
                        result["company_profiles"][ticker] = path
                    else:
                        logs.append(f"Company profile for {ticker} is up to date (monthly)")
                        path = os.path.join(self.data_dir, "company_profiles", f"{ticker}.json")
                        result["company_profiles"][ticker] = path
            except Exception as e:
                return {
                    "success": False,
                    "failed_step": "Fetching Company Profiles",
                    "error": str(e),
                    "logs": logs
                }

        # -----------------------------------------------------------------
        # 6. Fetch economic indicators (daily, shared)
        # -----------------------------------------------------------------
        if economic_indicators:
            try:
                logs.append("=== Fetching Economic Indicators ===")
                logs.append(f"Indicators: {economic_indicators}")
                econ_df = self._smart_fetch_economic(economic_indicators, start_date, end_date)
                if econ_df.empty:
                    raise ValueError("No economic indicators returned. Check FRED API key.")

                # Resample to hourly using first ticker's index
                if tickers and result["price_data"]:
                    first_ticker = tickers[0]
                    price_path = os.path.join(self.data_dir, "price_data", f"{first_ticker}.parquet")
                    if os.path.exists(price_path):
                        price_df = pd.read_parquet(price_path)
                        hourly_index = price_df.index
                        econ_df.index = econ_df.index.tz_localize('America/New_York')
                        econ_df = econ_df.reindex(hourly_index, method='ffill')
                        logs.append(f"  Resampled economic data to hourly ({len(econ_df)} rows)")

                path = os.path.join(self.data_dir, "economic", "indicators.parquet")
                econ_df.to_parquet(path)
                result["economic_indicators"]["path"] = path
                result["economic_indicators"]["indicators"] = list(econ_df.columns)
                logs.append(f"  Saved {len(econ_df)} rows of economic data")
            except Exception as e:
                return {
                    "success": False,
                    "failed_step": "Fetching Economic Indicators",
                    "error": str(e),
                    "logs": logs
                }


        # -----------------------------------------------------------------
        # 7. Generate data dictionary
        # -----------------------------------------------------------------
        try:
            logs.append("=== Generating Data Dictionary ===")
            data_dictionary = self._generate_data_dictionary(result, tickers, required_sectors)
            dict_path = os.path.join(self.data_dir, "data_dictionary.json")
            with open(dict_path, 'w') as f:
                json.dump(data_dictionary, f, indent=2)
            logs.append(f"Data dictionary saved to {dict_path}")
        except Exception as e:
            return {
                "success": False,
                "failed_step": "Generating Data Dictionary",
                "error": str(e),
                "logs": logs
            }

        logs.append("=== Data Collection Complete ===")
        return {
            "success": True,
            "data_paths": result,
            "data_dictionary": data_dictionary,
            "logs": logs
        }

    def _get_required_sectors(self, tickers: List[str]) -> Dict[str, str]:
        """
        Determine which sector ETFs are needed based on basket stocks.

        Args:
            tickers: List of stock tickers

        Returns:
            Dict mapping sector name to ETF symbol
        """
        required_sectors = {}

        for ticker in tickers:
            try:
                fundamentals = self.price_source.fetch_fundamentals(ticker)
                sector = fundamentals.get("sector")
                if sector and sector in SECTOR_ETF_MAP:
                    required_sectors[sector] = SECTOR_ETF_MAP[sector]
            except Exception as e:
                logger.warning(f"Could not determine sector for {ticker}: {e}")

        return required_sectors

    def _should_update_profile(self, ticker: str) -> bool:
        """
        Check if company profile needs update (monthly).

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if profile should be updated
        """
        path = os.path.join(self.data_dir, "company_profiles", f"{ticker}.json")

        if not os.path.exists(path):
            return True

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            fetched_at = data.get("fetched_at")
            if not fetched_at:
                return True

            fetched_date = datetime.strptime(fetched_at, "%Y-%m-%d %H:%M:%S")
            days_since_fetch = (datetime.now() - fetched_date).days

            # Update if more than 30 days old
            return days_since_fetch > 30

        except Exception:
            return True

    def _generate_data_dictionary(
        self,
        result: Dict[str, Any],
        tickers: List[str],
        required_sectors: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Generate data dictionary describing all collected data.

        Args:
            result: Collection result with paths
            tickers: List of stock tickers
            required_sectors: Dict of sector -> ETF

        Returns:
            Data dictionary
        """
        dictionary = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tickers": tickers,
            "datasets": {
                "price_data": {
                    "description": "Hourly OHLCV price data for each stock",
                    "scope": "Per ticker",
                    "frequency": "Hourly",
                    "source": "yfinance",
                    "columns": ["Open", "High", "Low", "Close", "Volume"],
                    "market_hours": "9:30 AM - 3:30 PM ET",
                    "tickers": list(result["price_data"].keys())
                },
                "market_data": {
                    "description": "Hourly data for market indices and sector ETFs",
                    "scope": "Shared (all tickers)",
                    "frequency": "Hourly",
                    "source": "yfinance",
                    "indices": {
                        "SPY": "S&P 500 ETF - Market benchmark",
                        "QQQ": "Nasdaq 100 ETF - Tech benchmark",
                        "IWM": "Russell 2000 ETF - Small cap",
                        "VIX": "Volatility Index - Fear gauge"
                    },
                    "sector_etfs": {
                        etf: f"{sector} sector ETF"
                        for sector, etf in required_sectors.items()
                    }
                },
                "fundamentals": {
                    "description": "Company fundamental data",
                    "scope": "Per ticker",
                    "frequency": "Daily",
                    "source": "yfinance",
                    "fields": [
                        "pe_ratio", "forward_pe", "pb_ratio", "market_cap",
                        "dividend_yield", "beta", "52_week_high", "52_week_low",
                        "avg_volume", "sector", "industry"
                    ],
                    "tickers": list(result["fundamentals"].keys())
                },
                "company_profiles": {
                    "description": "Company text profiles for embedding",
                    "scope": "Per ticker",
                    "frequency": "Monthly",
                    "source": "yfinance",
                    "fields": [
                        "company_name", "sector", "industry",
                        "business_summary", "website", "country", "employees"
                    ],
                    "tickers": list(result["company_profiles"].keys())
                },
                "economic_indicators": {
                    "description": "Macroeconomic indicators from FRED",
                    "scope": "Shared (all tickers)",
                    "frequency": "Daily",
                    "source": "FRED",
                    "indicators": result["economic_indicators"].get("indicators", [])
                }
            },
            "scope_summary": {
                "per_ticker": [
                    "price_data",
                    "fundamentals",
                    "company_profiles"
                ],
                "shared_all": [
                    "market_data (indices: SPY, QQQ, IWM, VIX)",
                    "economic_indicators"
                ],
                "shared_by_sector": [
                    f"market_data (sector ETFs: {', '.join(required_sectors.values())})"
                ]
            }
        }

        return dictionary


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Option 1: Use SystemConfig from config.py
    # from config import config
    # agent = DataAgent(config, data_dir="./data")

    # Option 2: Use plain dict (for testing)
    config = {
        "fred_api_key": ""  # Add your FRED API key
    }
    agent = DataAgent(config, data_dir="./data")

    # Run data collection
    result = agent.run({
        "tickers": ["AAPL", "MSFT", "JPM"],
        "start_date": "2023-01-01",
        "end_date": "2024-12-01",
        "fetch_fundamentals": True,
        "fetch_company_profiles": True,
        "economic_indicators": ["FEDFUNDS", "CPIAUCSL"]
    })

    # Print results
    if result["success"]:
        print("\n=== Data Collection Complete ===")
        print(f"Logs: {len(result['logs'])} entries")
        for log in result['logs']:
            print(f"  {log}")
    else:
        print("\n=== Data Collection Failed ===")
        print(f"Failed step: {result['failed_step']}")
        print(f"Error: {result['error']}")
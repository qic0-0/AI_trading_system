"""
Data Agent - Agent 1 in the Quant Trading System.

Responsibilities:
- Collect price data (via yfinance)
- Collect news data (via Finnhub/Alpha Vantage)
- Collect fundamental data (via yfinance)
- Collect economic indicators (via FRED)
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import yfinance as yf
import requests
import logging

logger = logging.getLogger(__name__)

from .base_agent import BaseAgent, AgentOutput


class DataAgent(BaseAgent):
    """
    Agent responsible for collecting raw data from various sources.
    
    Data Sources:
    - Yahoo Finance: Price data, fundamentals
    - Finnhub/Alpha Vantage: News articles
    - FRED: Economic indicators
    """
    
    def __init__(self, llm_client, config):
        super().__init__("DataAgent", llm_client, config)
        
    @property
    def system_prompt(self) -> str:
        return """You are a Data Collection Agent for a quantitative trading system.

Your role is to:
1. Collect price data (OHLCV) for requested tickers
2. Fetch relevant news articles
3. Retrieve fundamental data (P/E, P/B, market cap, etc.)
4. Get economic indicators (Fed rates, CPI, etc.)

You have access to the following tools:
- fetch_price_data: Get historical price data
- fetch_fundamentals: Get company fundamentals
- fetch_news: Get news articles
- fetch_economic_indicators: Get economic data from FRED

Always ensure data quality and report any issues with data collection."""

    def _register_tools(self):
        """Register data collection tools."""
        
        self.register_tool(
            name="fetch_price_data",
            func=self.fetch_price_data,
            description="Fetch historical OHLCV price data for a ticker",
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"}
                },
                "required": ["ticker", "start_date", "end_date"]
            }
        )
        
        self.register_tool(
            name="fetch_fundamentals",
            func=self.fetch_fundamentals,
            description="Fetch fundamental data for a ticker (P/E, P/B, market cap)",
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["ticker"]
            }
        )
        
        self.register_tool(
            name="fetch_news",
            func=self.fetch_news,
            description="Fetch news articles for a ticker",
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"}
                },
                "required": ["ticker"]
            }
        )
        
        self.register_tool(
            name="fetch_economic_indicators",
            func=self.fetch_economic_indicators,
            description="Fetch economic indicators from FRED",
            parameters={
                "type": "object",
                "properties": {
                    "indicators": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of FRED indicator codes (e.g., 'FEDFUNDS', 'CPIAUCSL')"
                    },
                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"}
                },
                "required": ["indicators"]
            }
        )

    def run(self, input_data: Dict[str, Any]) -> AgentOutput:
        """
        Run data collection for specified tickers and date range.

        Args:
            input_data: {
                "tickers": List[str],
                "start_date": str,
                "end_date": str,
                "collect_news": bool,
                "collect_fundamentals": bool,
                "economic_indicators": List[str]
            }

        Returns:
            AgentOutput with collected data and data dictionary
        """
        tickers = input_data.get("tickers", [])
        start_date = input_data.get("start_date")
        end_date = input_data.get("end_date")
        collect_news = input_data.get("collect_news", False)
        collect_fundamentals = input_data.get("collect_fundamentals", False)
        economic_indicators = input_data.get("economic_indicators", [])

        logs = []
        collected_data = {
            "price_data": {},
            "fundamentals": {},
            "news": {},
            "economic_indicators": {}
        }

        # Collect price data for each ticker
        for ticker in tickers:
            logs.append(f"Fetching price data for {ticker}...")
            collected_data["price_data"][ticker] = self.fetch_price_data(
                ticker, start_date, end_date
            )

            # Collect fundamentals if requested
            if collect_fundamentals:
                logs.append(f"Fetching fundamentals for {ticker}...")
                collected_data["fundamentals"][ticker] = self.fetch_fundamentals(ticker)

            # Collect news if requested
            if collect_news:
                logs.append(f"Fetching news for {ticker}...")
                collected_data["news"][ticker] = self.fetch_news(
                    ticker, start_date, end_date
                )

        # Collect economic indicators if requested
        if economic_indicators:
            logs.append(f"Fetching economic indicators: {economic_indicators}...")
            collected_data["economic_indicators"] = self.fetch_economic_indicators(
                economic_indicators, start_date, end_date
            )

        # Generate data dictionary
        data_dictionary = self.generate_data_dictionary(collected_data)

        return AgentOutput(
            success=True,
            data={
                "collected_data": collected_data,
                "data_dictionary": data_dictionary
            },
            message=f"Collected data for {len(tickers)} tickers",
            logs=logs
        )

    # ==================== Tool Implementations ====================
    def fetch_price_data(
            self,
            ticker: str,
            start_date: str,
            end_date: str
    ) -> pd.DataFrame:
        """
        Fetch OHLCV price data using yfinance.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if df.empty:
                logger.warning(f"No price data found for {ticker}")
                return pd.DataFrame()

            # Keep standard columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            logger.info(f"Fetched {len(df)} days of price data for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error fetching price data for {ticker}: {e}")
            return pd.DataFrame()

    def fetch_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch fundamental data using yfinance.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with P/E, P/B, market cap, etc.
        """
        try:
            stock = yf.Ticker(ticker)
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
            }

            logger.info(f"Fetched fundamentals for {ticker}")
            return fundamentals

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}

    def fetch_news(
            self,
            ticker: str,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch news articles using Finnhub.

        Args:
            ticker: Stock ticker symbol
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)

        Returns:
            List of news articles with headline, summary, date, source
        """
        api_key = self.config.data_sources.finnhub_api_key

        if not api_key:
            logger.warning("Finnhub API key not configured, skipping news fetch")
            return []

        try:
            # Finnhub API endpoint
            url = "https://finnhub.io/api/v1/company-news"

            # Default to last 30 days if no dates provided
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                start_date = (datetime.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")

            params = {
                "symbol": ticker,
                "from": start_date,
                "to": end_date,
                "token": api_key
            }

            response = requests.get(url, params=params)
            response.raise_for_status()

            news_data = response.json()

            # Parse and format news articles
            articles = []
            for item in news_data:
                articles.append({
                    "headline": item.get("headline", ""),
                    "summary": item.get("summary", ""),
                    "datetime": datetime.fromtimestamp(item.get("datetime", 0)).strftime("%Y-%m-%d %H:%M:%S"),
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                    "ticker": ticker
                })

            logger.info(f"Fetched {len(articles)} news articles for {ticker}")
            return articles

        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return []

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
        api_key = self.config.data_sources.fred_api_key

        if not api_key:
            logger.warning("FRED API key not configured, skipping economic indicators")
            return {}

        try:
            from fredapi import Fred
            fred = Fred(api_key=api_key)

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

    def generate_data_dictionary(self, collected_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a data dictionary describing all collected data.

        Args:
            collected_data: The collected data from all sources

        Returns:
            Data dictionary describing available data
        """
        dictionary = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "datasets": {}
        }

        # Document price data
        if collected_data.get("price_data"):
            tickers = list(collected_data["price_data"].keys())
            sample_df = next(
                (df for df in collected_data["price_data"].values() if not df.empty),
                pd.DataFrame()
            )
            dictionary["datasets"]["price_data"] = {
                "description": "Daily OHLCV price data",
                "source": "yfinance",
                "tickers": tickers,
                "columns": list(sample_df.columns) if not sample_df.empty else [],
                "date_range": {
                    "start": str(sample_df.index.min()) if not sample_df.empty else None,
                    "end": str(sample_df.index.max()) if not sample_df.empty else None
                }
            }

        # Document fundamentals
        if collected_data.get("fundamentals"):
            sample = next(
                (f for f in collected_data["fundamentals"].values() if "error" not in f),
                {}
            )
            dictionary["datasets"]["fundamentals"] = {
                "description": "Company fundamental data",
                "source": "yfinance",
                "tickers": list(collected_data["fundamentals"].keys()),
                "fields": [k for k in sample.keys() if k != "ticker"]
            }

        # Document news
        if collected_data.get("news"):
            total_articles = sum(len(articles) for articles in collected_data["news"].values())
            dictionary["datasets"]["news"] = {
                "description": "Company news articles",
                "source": "Finnhub",
                "tickers": list(collected_data["news"].keys()),
                "total_articles": total_articles,
                "fields": ["headline", "summary", "datetime", "source", "url"]
            }

        # Document economic indicators
        if collected_data.get("economic_indicators"):
            dictionary["datasets"]["economic_indicators"] = {
                "description": "Macroeconomic indicators",
                "source": "FRED",
                "indicators": list(collected_data["economic_indicators"].keys())
            }

        return dictionary
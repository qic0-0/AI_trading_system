"""
Factor Formulas - Calculation functions for feature engineering.

Each factor is an independent tool that can be registered and called by agent.
Easy to add more formulas later.

Usage:
    from factor_formulas import TOOLS, get_tool, call_tool, list_tools

    # Get all tools
    print(TOOLS.keys())

    # Call specific tool
    result = call_tool("compute_rsi", df=price_df, period=14)

    # Agent can select any subset
    selected = ["compute_rsi", "compute_spy_return"]
    for tool_name in selected:
        result = call_tool(tool_name, **params)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable, List
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL REGISTRY
# =============================================================================

TOOLS: Dict[str, Dict[str, Any]] = {}


def register_tool(
    name: str,
    description: str,
    output_type: str,
    requires: Dict[str, Any] = None
):
    """
    Decorator to register a function as a tool.

    Args:
        name: Tool name (unique identifier)
        description: What this tool computes
        output_type: "independent" | "shared" | "y_value" | "embedding"
        requires: Data requirements {
            "data": str,           # Data source: "price_data" | "market_data" | "economic" | "company_profiles"
            "columns": List[str],  # Required columns from data
            "extra": str           # Extra data needed (e.g., "sector_etf")
        }
    """
    def decorator(func: Callable) -> Callable:
        TOOLS[name] = {
            "func": func,
            "description": description,
            "output_type": output_type,
            "requires": requires or {},
            "params": func.__code__.co_varnames[:func.__code__.co_argcount]
        }
        return func
    return decorator


def get_tool(name: str) -> Dict[str, Any]:
    """Get tool info by name."""
    if name not in TOOLS:
        raise ValueError(f"Tool '{name}' not found. Available: {list(TOOLS.keys())}")
    return TOOLS[name]


def call_tool(name: str, **kwargs) -> Any:
    """Call a tool by name with given parameters."""
    tool = get_tool(name)
    return tool["func"](**kwargs)


def list_tools(output_type: Optional[str] = None) -> List[str]:
    """
    List available tools.

    Args:
        output_type: Filter by type (None = all)

    Returns:
        List of tool names
    """
    if output_type is None:
        return list(TOOLS.keys())
    return [name for name, info in TOOLS.items() if info["output_type"] == output_type]


# =============================================================================
# Y VALUE TOOLS (Just another factor - Quant Agent decides what is Y)
# =============================================================================

@register_tool(
    name="compute_log_return_y",
    description="Compute log return within same hourly bar. Formula: ln(Close/Open)",
    output_type="independent",
    requires={
        "data": "price_data",
        "columns": ["Open", "Close"]
    }
)
def compute_log_return_y(df: pd.DataFrame) -> pd.Series:
    """
    Compute Y value: log return within same hourly bar.

    Formula: y = ln(Close / Open)

    Args:
        df: DataFrame with 'Open' and 'Close' columns

    Returns:
        Series of log returns
    """
    return np.log(df['Close'] / df['Open'])


# =============================================================================
# INDEPENDENT FACTOR TOOLS (Per Ticker)
# =============================================================================

@register_tool(
    name="compute_return_1h",
    description="Compute 1-hour log return. Formula: ln(Close_t / Close_t-1)",
    output_type="independent",
    requires={
        "data": "price_data",
        "columns": ["Close"]
    }
)
def compute_return_1h(df: pd.DataFrame) -> pd.Series:
    """
    Compute 1-hour log return.

    Args:
        df: DataFrame with 'Close' column

    Returns:
        Series of 1-hour log returns
    """
    return np.log(df['Close'] / df['Close'].shift(1))


@register_tool(
    name="compute_return_5h",
    description="Compute 5-hour log return. Formula: ln(Close_t / Close_t-5)",
    output_type="independent",
    requires={
        "data": "price_data",
        "columns": ["Close"]
    }
)
def compute_return_5h(df: pd.DataFrame) -> pd.Series:
    """
    Compute 5-hour log return.

    Args:
        df: DataFrame with 'Close' column

    Returns:
        Series of 5-hour log returns
    """
    return np.log(df['Close'] / df['Close'].shift(5))


@register_tool(
    name="compute_return",
    description="Compute N-period log return. Formula: ln(Close_t / Close_t-period)",
    output_type="independent",
    requires={
        "data": "price_data",
        "columns": ["Close"]
    }
)
def compute_return(df: pd.DataFrame, period: int = 1) -> pd.Series:
    """
    Compute log return over N periods.

    Args:
        df: DataFrame with 'Close' column
        period: Number of periods to look back

    Returns:
        Series of log returns
    """
    return np.log(df['Close'] / df['Close'].shift(period))


@register_tool(
    name="compute_rsi",
    description="Compute Relative Strength Index (RSI). Range 0-100.",
    output_type="independent",
    requires={
        "data": "price_data",
        "columns": ["Close"]
    }
)
def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).

    Formula: 100 - (100 / (1 + RS)) where RS = avg_gain / avg_loss

    Args:
        df: DataFrame with 'Close' column
        period: RSI period (default 14)

    Returns:
        Series of RSI values (0-100)
    """
    delta = df['Close'].diff()

    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


@register_tool(
    name="compute_volatility",
    description="Compute rolling volatility (std of returns).",
    output_type="independent",
    requires={
        "data": "price_data",
        "columns": ["Close"]
    }
)
def compute_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Compute rolling volatility (standard deviation of returns).

    Args:
        df: DataFrame with 'Close' column
        period: Rolling window size

    Returns:
        Series of volatility values
    """
    returns = np.log(df['Close'] / df['Close'].shift(1))
    return returns.rolling(window=period).std()


@register_tool(
    name="compute_macd",
    description="Compute MACD line. Formula: EMA(12) - EMA(26)",
    output_type="independent",
    requires={
        "data": "price_data",
        "columns": ["Close"]
    }
)
def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    """
    Compute MACD line.

    Args:
        df: DataFrame with 'Close' column
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)

    Returns:
        Series of MACD values
    """
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


@register_tool(
    name="compute_macd_signal",
    description="Compute MACD signal line. Formula: EMA(9) of MACD",
    output_type="independent",
    requires={
        "data": "price_data",
        "columns": ["Close"]
    }
)
def compute_macd_signal(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """
    Compute MACD signal line.

    Args:
        df: DataFrame with 'Close' column
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)

    Returns:
        Series of MACD signal values
    """
    macd_line = compute_macd(df, fast, slow)
    return macd_line.ewm(span=signal, adjust=False).mean()


@register_tool(
    name="compute_macd_hist",
    description="Compute MACD histogram. Formula: MACD - Signal",
    output_type="independent",
    requires={
        "data": "price_data",
        "columns": ["Close"]
    }
)
def compute_macd_hist(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """
    Compute MACD histogram.

    Args:
        df: DataFrame with 'Close' column
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)

    Returns:
        Series of MACD histogram values
    """
    macd_line = compute_macd(df, fast, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


@register_tool(
    name="compute_volume_ratio",
    description="Compute volume ratio (current / average).",
    output_type="independent",
    requires={
        "data": "price_data",
        "columns": ["Volume"]
    }
)
def compute_volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Compute volume ratio (current volume / average volume).

    Args:
        df: DataFrame with 'Volume' column
        period: Rolling window for average

    Returns:
        Series of volume ratios
    """
    avg_volume = df['Volume'].rolling(window=period).mean()
    return df['Volume'] / avg_volume


@register_tool(
    name="compute_price_range",
    description="Compute price range within bar. Formula: (High-Low)/Open",
    output_type="independent",
    requires={
        "data": "price_data",
        "columns": ["High", "Low", "Open"]
    }
)
def compute_price_range(df: pd.DataFrame) -> pd.Series:
    """
    Compute price range within bar.

    Formula: (High - Low) / Open

    Args:
        df: DataFrame with 'High', 'Low', 'Open' columns

    Returns:
        Series of price range ratios
    """
    return (df['High'] - df['Low']) / df['Open']


@register_tool(
    name="compute_sector_return",
    description="Compute sector ETF return for a ticker.",
    output_type="independent",
    requires={
        "data": "price_data",
        "columns": ["Close"],
        "extra": "sector_etf"
    }
)
def compute_sector_return(ticker_df: pd.DataFrame, sector_etf_df: pd.DataFrame, period: int = 1) -> pd.Series:
    """
    Compute sector ETF return (aligned to ticker's timestamps).

    Args:
        ticker_df: Ticker DataFrame (for index alignment)
        sector_etf_df: Sector ETF DataFrame with 'Close' column
        period: Number of periods for return calculation

    Returns:
        Series of sector returns (aligned to ticker index)
    """
    sector_return = np.log(sector_etf_df['Close'] / sector_etf_df['Close'].shift(period))
    return sector_return.reindex(ticker_df.index, method='ffill')


# =============================================================================
# SHARED FACTOR TOOLS (All Tickers)
# =============================================================================

@register_tool(
    name="compute_spy_return",
    description="Compute SPY (S&P 500) return.",
    output_type="shared",
    requires={
        "data": "market_data",
        "ticker": "SPY",
        "columns": ["Close"]
    }
)
def compute_spy_return(spy_df: pd.DataFrame, period: int = 1) -> pd.Series:
    """
    Compute SPY (market) return.

    Args:
        spy_df: SPY DataFrame with 'Close' column
        period: Number of periods for return calculation

    Returns:
        Series of SPY returns
    """
    return np.log(spy_df['Close'] / spy_df['Close'].shift(period))


@register_tool(
    name="compute_vix_level",
    description="Get VIX level (volatility index).",
    output_type="shared",
    requires={
        "data": "market_data",
        "ticker": "VIX",
        "columns": ["Close"]
    }
)
def compute_vix_level(vix_df: pd.DataFrame) -> pd.Series:
    """
    Get VIX level (raw close value).

    Args:
        vix_df: VIX DataFrame with 'Close' column

    Returns:
        Series of VIX levels
    """
    return vix_df['Close']


@register_tool(
    name="compute_qqq_return",
    description="Compute QQQ (Nasdaq 100) return.",
    output_type="shared",
    requires={
        "data": "market_data",
        "ticker": "QQQ",
        "columns": ["Close"]
    }
)
def compute_qqq_return(qqq_df: pd.DataFrame, period: int = 1) -> pd.Series:
    """
    Compute QQQ (Nasdaq) return.

    Args:
        qqq_df: QQQ DataFrame with 'Close' column
        period: Number of periods for return calculation

    Returns:
        Series of QQQ returns
    """
    return np.log(qqq_df['Close'] / qqq_df['Close'].shift(period))


@register_tool(
    name="compute_iwm_return",
    description="Compute IWM (Russell 2000) return.",
    output_type="shared",
    requires={
        "data": "market_data",
        "ticker": "IWM",
        "columns": ["Close"]
    }
)
def compute_iwm_return(iwm_df: pd.DataFrame, period: int = 1) -> pd.Series:
    """
    Compute IWM (Russell 2000) return.

    Args:
        iwm_df: IWM DataFrame with 'Close' column
        period: Number of periods for return calculation

    Returns:
        Series of IWM returns
    """
    return np.log(iwm_df['Close'] / iwm_df['Close'].shift(period))


@register_tool(
    name="compute_fed_rate",
    description="Get Federal Funds Rate (forward-filled to hourly).",
    output_type="shared",
    requires={
        "data": "economic",
        "columns": ["FEDFUNDS"]
    }
)
def compute_fed_rate(econ_df: pd.DataFrame, target_index: pd.Index) -> pd.Series:
    """
    Get Federal Funds Rate (forward-filled to hourly).

    Args:
        econ_df: Economic DataFrame with 'FEDFUNDS' column
        target_index: Target datetime index for alignment

    Returns:
        Series of Fed rate values (forward-filled)
    """
    if 'FEDFUNDS' not in econ_df.columns:
        return pd.Series(index=target_index, dtype=float)

    fed_rate = econ_df['FEDFUNDS']
    return fed_rate.reindex(target_index, method='ffill')


@register_tool(
    name="compute_cpi",
    description="Get CPI value (forward-filled to hourly).",
    output_type="shared",
    requires={
        "data": "economic",
        "columns": ["CPIAUCSL"]
    }
)
def compute_cpi(econ_df: pd.DataFrame, target_index: pd.Index) -> pd.Series:
    """
    Get CPI value (forward-filled to hourly).

    Args:
        econ_df: Economic DataFrame with 'CPIAUCSL' column
        target_index: Target datetime index for alignment

    Returns:
        Series of CPI values (forward-filled)
    """
    if 'CPIAUCSL' not in econ_df.columns:
        return pd.Series(index=target_index, dtype=float)

    cpi = econ_df['CPIAUCSL']
    return cpi.reindex(target_index, method='ffill')


# =============================================================================
# EMBEDDING TOOLS (Text → Vector)
# =============================================================================

@register_tool(
    name="compute_text_embedding",
    description="Compute embedding for text. Tries Argonne API first, falls back to local model.",
    output_type="embedding",
    requires={
        "data": "text_input",
        "columns": []
    }
)
def compute_text_embedding(
    text: str,
    api_base_url: str = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
    model: str = "mistralai/Mistral-7B-Instruct-v0.3-embed",
    api_key: Optional[str] = None,
    use_local_fallback: bool = True,
    local_model: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Compute embedding for text.

    Tries Argonne API first, falls back to local sentence-transformers if API fails.

    Args:
        text: Text to embed
        api_base_url: Argonne API base URL
        model: Embedding model for Argonne
        api_key: API key/access token (if None, tries inference_auth_token)
        use_local_fallback: If True, use local model when API fails
        local_model: Local sentence-transformers model name

    Returns:
        Embedding vector (np.ndarray)
    """
    # Try Argonne API first
    try:
        from openai import OpenAI

        # Get access token if not provided
        if api_key is None:
            try:
                from inference_auth_token import get_access_token
                api_key = get_access_token()
            except ImportError:
                logger.warning("inference_auth_token not available, trying local fallback")
                if use_local_fallback:
                    return _compute_local_embedding(text, local_model)
                return np.array([])

        client = OpenAI(
            api_key=api_key,
            base_url=api_base_url
        )

        response = client.embeddings.create(
            model=model,
            input=text,
            encoding_format="float"
        )

        embedding = np.array(response.data[0].embedding)
        logger.info(f"Generated Argonne embedding: {embedding.shape[0]} dimensions")
        return embedding

    except Exception as e:
        logger.error(f"Error generating embedding: {e}")

        # Fall back to local model
        if use_local_fallback:
            logger.info("Falling back to local embedding model...")
            return _compute_local_embedding(text, local_model)

        return np.array([])


def _compute_local_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Compute embedding using local sentence-transformers model.

    Args:
        text: Text to embed
        model_name: Model name from sentence-transformers

    Returns:
        Embedding vector (384-dim for MiniLM)
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        embedding = model.encode(text, convert_to_numpy=True)
        logger.info(f"Generated local embedding: {embedding.shape[0]} dimensions")
        return embedding

    except ImportError:
        logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        return np.array([])
    except Exception as e:
        logger.error(f"Error generating local embedding: {e}")
        return np.array([])


@register_tool(
    name="compute_company_embedding",
    description="Compute embedding from company profile. Tries Argonne API, falls back to local.",
    output_type="embedding",
    requires={
        "data": "company_profiles",
        "columns": ["company_name", "sector", "industry", "business_summary"]
    }
)
def compute_company_embedding(
    profile: Dict,
    api_base_url: str = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
    model: str = "mistralai/Mistral-7B-Instruct-v0.3-embed",
    api_key: Optional[str] = None,
    use_local_fallback: bool = True,
    local_model: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Compute embedding from company profile dict.

    Combines multiple text fields for richer embedding.

    Args:
        profile: Company profile dict with keys:
            - company_name, sector, industry, business_summary
        api_base_url: Argonne API base URL
        model: Embedding model
        api_key: API key/access token
        use_local_fallback: If True, use local model when API fails
        local_model: Local sentence-transformers model name

    Returns:
        Embedding vector
    """
    text_parts = []

    if profile.get("company_name"):
        text_parts.append(f"Company: {profile['company_name']}")

    if profile.get("sector"):
        text_parts.append(f"Sector: {profile['sector']}")

    if profile.get("industry"):
        text_parts.append(f"Industry: {profile['industry']}")

    if profile.get("business_summary"):
        text_parts.append(f"Description: {profile['business_summary']}")

    combined_text = " | ".join(text_parts)

    if not combined_text.strip():
        logger.warning("Empty profile text, returning zero embedding")
        return np.zeros(384)  # Local model dimension

    return compute_text_embedding(
        combined_text,
        api_base_url,
        model,
        api_key,
        use_local_fallback,
        local_model
    )


# =============================================================================
# TOOL INFO & UTILITIES
# =============================================================================

def get_tool_info(name: str) -> str:
    """Get formatted info about a tool."""
    tool = get_tool(name)
    return f"""
Tool: {name}
Type: {tool['output_type']}
Description: {tool['description']}
Parameters: {tool['params']}
"""


def print_all_tools():
    """Print all registered tools with requirements."""
    print("=" * 60)
    print("REGISTERED FACTOR TOOLS")
    print("=" * 60)

    for output_type in ["independent", "shared", "embedding"]:
        tools = list_tools(output_type)
        if tools:
            print(f"\n{output_type.upper()} ({len(tools)} tools):")
            print("-" * 40)
            for name in tools:
                info = TOOLS[name]
                print(f"  {name}")
                print(f"    └─ {info['description']}")
                if info.get('requires'):
                    req = info['requires']
                    print(f"    └─ requires: data={req.get('data')}, columns={req.get('columns')}")


def get_tool_requirements(tool_names: List[str]) -> Dict[str, set]:
    """
    Get all data requirements for a list of tools.

    Args:
        tool_names: List of tool names

    Returns:
        Dict mapping data source -> set of required columns/tickers
    """
    requirements = {
        "price_data": {"columns": set()},
        "market_data": {"tickers": set(), "columns": set()},
        "economic": {"columns": set()},
        "company_profiles": {"columns": set()}
    }

    for name in tool_names:
        tool = get_tool(name)
        req = tool.get("requires", {})
        data_source = req.get("data")

        if data_source == "price_data":
            requirements["price_data"]["columns"].update(req.get("columns", []))
        elif data_source == "market_data":
            if req.get("ticker"):
                requirements["market_data"]["tickers"].add(req.get("ticker"))
            requirements["market_data"]["columns"].update(req.get("columns", []))
        elif data_source == "economic":
            requirements["economic"]["columns"].update(req.get("columns", []))
        elif data_source == "company_profiles":
            requirements["company_profiles"]["columns"].update(req.get("columns", []))

    return requirements


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Print all registered tools
    print_all_tools()

    # Example: List tools by type
    print("\n\nIndependent tools:", list_tools("independent"))
    print("Shared tools:", list_tools("shared"))

    # Example: Get requirements for selected tools
    print("\n\n" + "=" * 60)
    print("EXAMPLE: Get requirements for selected tools")
    print("=" * 60)
    selected = ["compute_rsi", "compute_macd", "compute_spy_return", "compute_fed_rate"]
    print(f"Selected tools: {selected}")
    requirements = get_tool_requirements(selected)
    print(f"Requirements: {requirements}")
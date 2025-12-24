"""
Test script for Feature Agent - Single comprehensive test
Tests all factors and embeddings for multiple tickers
"""

import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)

from config.config import config
from agents import FeatureAgent


def test_feature_agent():
    """Test Feature Agent with all factors and embeddings."""

    print("=" * 60)
    print("Testing Feature Agent - Full Test")
    print("=" * 60)

    # Initialize agent
    agent = FeatureAgent(config, data_dir="./test_data")
    print("✓ Agent initialized")

    # Print available tools
    print("\n=== Available Tools ===")
    tools = agent.get_available_tools()
    for tool_type, tool_list in tools.items():
        print(f"  {tool_type}: {tool_list}")

    # Define test tickers
    tickers = ["AAPL", "JNJ"]

    # Run with ALL factors and embeddings
    print("\n=== Running Feature Engineering ===")
    print(f"Tickers: {tickers}")

    result = agent.run({
        "tickers": tickers,
        "factors": [
            # Y value (can be used as target)
            "compute_log_return_y",
            # Price-based factors (independent per ticker)
            "compute_return_1h",
            "compute_return_5h",
            "compute_rsi",
            "compute_volatility",
            "compute_macd",
            "compute_macd_signal",
            "compute_macd_hist",
            "compute_volume_ratio",
            "compute_price_range",
            "compute_sector_return",
            # Market factors (shared across tickers)
            "compute_spy_return",
            "compute_vix_level",
            "compute_qqq_return",
            "compute_iwm_return",
            # Economic factors (shared across tickers)
            "compute_fed_rate",
            "compute_cpi"
        ],
        "compute_embeddings": True  # Include text embeddings
    })

    # Print results
    print("\n" + "=" * 60)
    if result["success"]:
        print("✓ FEATURE ENGINEERING SUCCESS")
        print("=" * 60)

        # Print logs
        print("\nExecution Logs:")
        for log in result['logs']:
            print(f"  {log}")

        # Print feature paths
        print("\nOutput Paths:")
        for key, value in result['feature_paths'].items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

        # Verify files exist and show stats
        print("\n=== Verifying Output Files ===")
        import pandas as pd
        import numpy as np

        # Check independent factors
        print("\nIndependent Factors (per ticker):")
        for ticker in tickers:
            path = f"./test_data/features/independent_factors/{ticker}.parquet"
            if os.path.exists(path):
                df = pd.read_parquet(path)
                print(f"  ✓ {ticker}: {len(df)} rows, {len(df.columns)} factors")
                print(f"      Columns: {list(df.columns)}")
            else:
                print(f"  ✗ {ticker}: file not found")

        # Check shared factors
        print("\nShared Factors:")
        shared_path = "./test_data/features/shared_factors.parquet"
        if os.path.exists(shared_path):
            df = pd.read_parquet(shared_path)
            print(f"  ✓ shared_factors: {len(df)} rows")
            print(f"      Columns: {list(df.columns)}")
        else:
            print(f"  ✗ shared_factors: file not found")

        # Check embeddings
        print("\nEmbeddings:")
        for ticker in tickers:
            path = f"./test_data/features/embeddings/{ticker}.npy"
            if os.path.exists(path):
                emb = np.load(path)
                print(f"  ✓ {ticker} embedding: shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")
            else:
                print(f"  ✗ {ticker} embedding: file not found")

        # Check feature dictionary
        print("\nFeature Dictionary:")
        dict_path = "./test_data/features/feature_dictionary.json"
        if os.path.exists(dict_path):
            import json
            with open(dict_path, 'r') as f:
                feature_dict = json.load(f)
            print(f"  ✓ feature_dictionary.json")
            print(f"      Tickers: {feature_dict.get('tickers', [])}")
            print(f"      Tools used: {len(feature_dict.get('tools_used', []))}")
            print(f"      Embeddings: {feature_dict.get('embeddings_computed', False)}")
        else:
            print(f"  ✗ feature_dictionary.json: file not found")

    else:
        print("✗ FEATURE ENGINEERING FAILED")
        print("=" * 60)
        print(f"\nFailed step: {result.get('failed_step', 'Unknown')}")
        print(f"Error: {result.get('error', 'Unknown error')}")
        print("\nLogs before failure:")
        for log in result.get('logs', []):
            print(f"  {log}")
        return False

    return True


if __name__ == "__main__":
    success = test_feature_agent()

    print("\n" + "=" * 60)
    print("FINAL RESULT:", "✓ PASS" if success else "✗ FAIL")
    print("=" * 60)

    sys.exit(0 if success else 1)
"""
Test script for Data Agent
Run this on your local machine to test the data agent.
"""

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import config and agent
from config import config
from agents.data_agent import DataAgent

def test_data_agent():
    """Test Data Agent with a small example."""

    print("=" * 60)
    print("Testing Data Agent")
    print("=" * 60)

    # Initialize agent
    agent = DataAgent(config, data_dir="./test_data")
    print("✓ Agent initialized")

    # Test with 3 years of data (will be fetched in chunks)
    # This tests: chunked fetching + smart caching
    start_str = "2024-01-01"
    end_str = "2024-12-19"

    print(f"\nDate range: {start_str} to {end_str} (~3 years)")
    print("This will be fetched in 1-year chunks")

    # Run with multiple tickers
    print("\nRunning data collection...")
    result = agent.run({
        "tickers": ["AAPL", "MSFT", "JPM", "XOM", "JNJ"],
        "start_date": start_str,
        "end_date": end_str,
        "fetch_fundamentals": True,
        "fetch_company_profiles": True,
        "economic_indicators": ["FEDFUNDS", "CPIAUCSL"]
    })

    # Print results
    print("\n" + "=" * 60)
    if result["success"]:
        print("✓ SUCCESS")
        print("=" * 60)
        print("\nLogs:")
        for log in result['logs']:
            print(f"  {log}")
        print("\nData paths:")
        for key, value in result['data_paths'].items():
            print(f"  {key}: {value}")
    else:
        print("✗ FAILED")
        print("=" * 60)
        print(f"\nFailed step: {result['failed_step']}")
        print(f"Error: {result['error']}")
        print("\nLogs before failure:")
        for log in result['logs']:
            print(f"  {log}")
        return False

    return True


if __name__ == "__main__":
    success = test_data_agent()
    sys.exit(0 if success else 1)
"""
Test script for Quant MCP Agent
Run this on your local machine to test the quant modeling agent.

Prerequisites:
1. Run test_data_agent.py first to generate test_data/
2. Run test_feature_agent.py to generate test_data/features/
3. Have model code in models/RNN_based_multi_hour/
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
from llm.llm_client import LLMClient
from agents.quant_mcp import MCPBrain, MCPConfig


def verify_data_adapter(model_name):
    """Verify the generated data adapter works."""

    model_dir = f"./models/{model_name}"
    adapter_path = os.path.join(model_dir, "data_adapter.py")

    if not os.path.exists(adapter_path):
        print("  ✗ data_adapter.py not found")
        return False

    try:
        import importlib.util
        import json

        # Import adapter
        spec = importlib.util.spec_from_file_location("data_adapter", adapter_path)
        adapter_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter_module)
        print("  ✓ Imported data_adapter.py")

        # Load tickers from feature dictionary
        with open("./test_data/features/feature_dictionary.json", 'r') as f:
            feature_dict = json.load(f)
        tickers = feature_dict.get("tickers", ["AAPL", "JNJ"])

        # Create adapter
        adapter = adapter_module.DataAdapter(
            data_dir="./test_data",
            tickers=tickers,
            y_column="compute_log_return_y",
            prediction_horizon=1
        )
        print("  ✓ Created DataAdapter instance")

        # Load and transform
        adapter.load_features()
        print("  ✓ Loaded features")

        df = adapter.transform()
        print(f"  ✓ Transformed data: {df.shape}")

        # Column summary
        x_cols = [c for c in df.columns if c.startswith('x_')]
        y_cols = [c for c in df.columns if c.startswith('y_')]
        share_cols = [c for c in df.columns if c.startswith('share_')]
        emb_cols = [c for c in df.columns if c.startswith('emb_')]

        print(f"\n  Column Summary:")
        print(f"    X features: {len(x_cols)}")
        print(f"    Y targets: {len(y_cols)}")
        print(f"    Shared: {len(share_cols)}")
        print(f"    Embeddings: {len(emb_cols)}")
        print(f"    Total: {len(df.columns)} columns, {len(df)} rows")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quant_mcp_agent():
    """Test Quant MCP Agent with RNN_based_multi_hour model."""

    print("=" * 60)
    print("Testing Quant MCP Agent")
    print("=" * 60)

    # Check prerequisites
    print("\n=== Checking Prerequisites ===")

    if not os.path.exists("./test_data/features/feature_dictionary.json"):
        print("✗ Feature dictionary not found!")
        print("  Please run test_data_agent.py and test_feature_agent.py first")
        return False
    print("✓ Feature dictionary found")

    if not os.path.exists("./models/RNN_based_multi_hour/model.py"):
        print("✗ Model code not found!")
        print("  Expected: ./models/RNN_based_multi_hour/model.py")
        return False
    print("✓ Model code found")

    if not os.path.exists("./models/RNN_based_multi_hour/MODEL_USAGE.md"):
        print("✗ Usage guide not found!")
        print("  Expected: ./models/RNN_based_multi_hour/MODEL_USAGE.md")
        return False
    print("✓ Usage guide found")

    # Initialize MCP
    llm_client = LLMClient(config.llm)
    mcp_config = MCPConfig(
        data_dir="./test_data",
        models_dir="./models"
    )

    mcp = MCPBrain(llm_client, mcp_config)
    print("✓ MCP Brain initialized")

    # Print available tools
    print("\n=== MCP Configuration ===")
    print(f"  Data dir: {mcp_config.data_dir}")
    print(f"  Models dir: {mcp_config.models_dir}")
    print(f"  Max retries: {mcp_config.max_retries}")

    # Run MCP
    print("\n=== Running MCP ===")
    print(f"Model: RNN_based_multi_hour")
    print(f"Y column: compute_log_return_y")
    print(f"Prediction horizon: 1")

    result = mcp.run(
        model_name="RNN_based_multi_hour",
        data_dir="./test_data",
        y_column="compute_log_return_y",
        prediction_horizon=1
    )

    # Print results
    print("\n" + "=" * 60)

    if result.status == "SUCCESS":
        print("✓ MCP SUCCESS")
        print("=" * 60)

        print("\nExecution Logs:")
        for log in result.logs:
            print(f"  {log}")

        print("\nGenerated Files:")
        for name, path in result.generated_files.items():
            exists = os.path.exists(path) if path else False
            status = "✓" if exists else "✗"
            print(f"  {status} {name}: {path}")

        if result.test_results:
            print("\nTest Results:")
            for test_name, test_result in result.test_results.items():
                if isinstance(test_result, dict):
                    status = "✓" if test_result.get("passed") else "✗"
                    error = test_result.get("error", "")
                    print(f"  {status} {test_name}" + (f": {error}" if error and not test_result.get("passed") else ""))

        # Verify data adapter works
        print("\n=== Verifying Data Adapter ===")
        verify_data_adapter("RNN_based_multi_hour")

        return True

    elif result.status == "PAUSED":
        print("⚠ MCP PAUSED - Blocking Issues Found")
        print("=" * 60)

        print("\nExecution Logs:")
        for log in result.logs:
            print(f"  {log}")

        print("\nBlocking Issues:")
        for issue in result.issues:
            if isinstance(issue, dict):
                print(f"  - Type: {issue.get('type')}")
                print(f"    Message: {issue.get('message')}")
                if issue.get('options'):
                    print(f"    Options: {issue.get('options')}")
            else:
                print(f"  - {issue}")

        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                if isinstance(warning, dict):
                    print(f"  - {warning.get('message')}")
                else:
                    print(f"  - {warning}")

        print(f"\nSession ID: {result.session_id}")
        print(f"Config path: {result.config_path}")
        print(f"Reasoning path: {result.reasoning_path}")

        print("\nTo resume, edit adapter_config.json and call:")
        print(f"  result = mcp.resume('{result.session_id}', user_overrides={{...}})")

        # Still verify partial work
        if result.config_path and os.path.exists(result.config_path):
            print("\n=== Generated Config Preview ===")
            import json
            with open(result.config_path, 'r') as f:
                config_data = json.load(f)
            print(f"  Source format: {config_data.get('source_format', {}).get('resolution')}")
            print(f"  Bars per day: {config_data.get('source_format', {}).get('bars_per_day')}")
            print(f"  Y column: {config_data.get('y_config', {}).get('column')}")

        return True  # Paused is acceptable

    else:
        print("✗ MCP FAILED")
        print("=" * 60)
        print(f"\nError: {result.message}")

        print("\nLogs:")
        for log in result.logs:
            print(f"  {log}")

        if result.issues:
            print("\nIssues:")
            for issue in result.issues:
                print(f"  - {issue}")

        return False


if __name__ == "__main__":
    success = test_quant_mcp_agent()

    print("\n" + "=" * 60)
    print("FINAL RESULT:", "✓ PASS" if success else "✗ FAIL")
    print("=" * 60)

    sys.exit(0 if success else 1)
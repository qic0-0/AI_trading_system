"""
Quant MCP - Model-Code-Protocol based Quant Agent System.

This package provides an LLM-powered system for:
1. Writing model code from descriptions (Model Code Agent)
2. Adapting data to model format (Data Adapt Agent)
3. Testing the complete pipeline (Test Agent)
4. Coordinating the workflow (MCP Brain)

Usage:
    from quant_mcp import MCPBrain, MCPConfig
    from llm.llm_client import LLMClient
    from config import config
    
    # Initialize
    llm_client = LLMClient(config.llm)
    mcp_config = MCPConfig(data_dir="./data", models_dir="./models")
    mcp = MCPBrain(llm_client, mcp_config)
    
    # Run (may pause if issues found)
    result = mcp.run(
        model_name="my_model",
        data_dir="./data",
        y_column="compute_log_return_y",
        prediction_horizon=1
    )
    
    # If paused, provide overrides and resume
    if result.status == "PAUSED":
        result = mcp.resume(
            session_id=result.session_id,
            user_overrides={"y_config": {"output_columns": 7}}
        )

Package Structure:
    quant_mcp/
    ├── __init__.py              # This file
    ├── mcp_config.py            # Configuration classes
    ├── base_mcp_agent.py        # Base agent with common tools
    ├── mcp_brain.py             # Coordinator
    ├── model_code_agent.py      # Writes model.py and usage_guide.md
    ├── data_adapt_agent.py      # Writes data_adapter.py
    └── test_agent.py            # Tests the pipeline
"""

from .mcp_config import (
    MCPConfig,
    MCPResult,
    AgentTask,
    AgentResponse,
    get_model_paths,
    get_features_dir,
    get_feature_dictionary_path,
    find_provided_files
)

from .mcp_brain import MCPBrain

from .base_mcp_agent import BaseMCPAgent

from .model_code_agent import ModelCodeAgent

from .data_adapt_agent import DataAdaptAgent

from .test_agent import TestAgent


__all__ = [
    # Config
    "MCPConfig",
    "MCPResult",
    "AgentTask",
    "AgentResponse",
    
    # Path helpers
    "get_model_paths",
    "get_features_dir",
    "get_feature_dictionary_path",
    "find_provided_files",
    
    # Agents
    "MCPBrain",
    "BaseMCPAgent",
    "ModelCodeAgent",
    "DataAdaptAgent",
    "TestAgent",
]

__version__ = "0.1.0"
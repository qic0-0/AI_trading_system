"""Quant MCP orchestration package.

This package implements a model-centric, LLM-driven workflow to:
- read model_description in models/{model_name}/
- generate missing components (model.py, data_adapter.py, pipeline/run scripts)
- run tests and route failures

All model artifacts live inside models/{model_name}/.
"""

from .brain import run_mcp

__all__ = ["run_mcp"]

"""Simple registry for MCP agents.

In this first iteration we dispatch by name (see executor.py). This file exists
so you can later expand to a richer capabilities registry (MCP-style) without
changing call sites.
"""

AGENTS = {
    "model_agent": "agents.quant_mcp.agents.model_agent",
    "data_adapter_agent": "agents.quant_mcp.agents.data_adapter_agent",
    "runner_agent": "agents.quant_mcp.agents.runner_agent",
    "test_agent": "agents.quant_mcp.agents.test_agent",
}

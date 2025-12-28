from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class MCPError(Exception):
    """Base class for MCP errors."""


class UserSetupError(MCPError):
    """Missing data / missing files / configuration problems."""


class ProvidedModelCodeError(MCPError):
    """Errors that appear to originate from user-provided model code."""


class AgentGenerationError(MCPError):
    """LLM-generated code or file generation failed."""


@dataclass
class ClassifiedError:
    kind: str  # 'user', 'provided_model', 'model_agent', 'adapter_agent', 'runner_agent', 'test_agent'
    message: str
    detail: Optional[str] = None


def classify_exception(exc: Exception, *, provided_model: bool) -> ClassifiedError:
    """Best-effort routing. Keep simple and explicit.

    Rules:
    - Missing files or config -> user
    - If provided_model and error looks like inside model.py import/call -> provided_model
    - Otherwise -> generic agent generation error
    """
    msg = str(exc)
    name = exc.__class__.__name__

    if isinstance(exc, FileNotFoundError):
        return ClassifiedError(kind="user", message=f"Missing required file: {msg}")
    if isinstance(exc, UserSetupError):
        return ClassifiedError(kind="user", message=msg)

    # Heuristic: if it mentions model.py and model is provided, we stop and ask user
    if provided_model and ("model.py" in msg or "models/" in msg) and ("ImportError" in name or "ModuleNotFoundError" in name or "AttributeError" in name):
        return ClassifiedError(kind="provided_model", message=msg)

    # Default
    return ClassifiedError(kind="agent", message=f"{name}: {msg}")

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .context import MCPContext
from .errors import ClassifiedError, classify_exception
from .planner import PlanStep


@dataclass
class StepResult:
    step: str
    success: bool
    written_files: List[str] = field(default_factory=list)
    message: str = ""
    error: Optional[ClassifiedError] = None


def _dispatch(step_name: str):
    if step_name == "model_agent":
        from .agents.model_agent import run as _run
        return _run
    if step_name == "data_adapter_agent":
        from .agents.data_adapter_agent import run as _run
        return _run
    if step_name == "runner_agent":
        from .agents.runner_agent import run as _run
        return _run
    if step_name == "test_agent":
        from .agents.test_agent import run as _run
        return _run
    raise ValueError(f"Unknown step: {step_name}")


def execute_plan(ctx: MCPContext, plan: List[PlanStep]) -> List[StepResult]:
    model_dir = ctx.paths.model_dir
    model_py = model_dir / "model.py"
    provided_model = model_py.exists()

    results: List[StepResult] = []

    for step in plan:
        fn = _dispatch(step.name)
        try:
            out = fn(ctx, payload=step.payload, provided_model=provided_model)
            results.append(StepResult(step=step.name, success=True, written_files=out.get("written_files", []), message=out.get("message", "")))
        except Exception as exc:
            tb = traceback.format_exc()
            classified = classify_exception(exc, provided_model=provided_model)
            classified.detail = tb
            results.append(StepResult(step=step.name, success=False, message=str(exc), error=classified))
            if ctx.policy.stop_on_failure:
                break
    return results

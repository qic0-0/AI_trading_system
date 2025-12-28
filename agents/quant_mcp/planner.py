from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .context import MCPContext


@dataclass
class PlanStep:
    name: str
    payload: Dict


def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False


def build_plan(ctx: MCPContext) -> List[PlanStep]:
    """Build an execution plan based on what's missing in models/{model_name}.

    model_description.md is mandatory.
    model.py may be provided; otherwise generate.
    data_adapter.py, pipeline.py, run.py, tests are generated/maintained.
    """
    model_dir = ctx.paths.model_dir
    model_py = model_dir / "model.py"
    adapter_py = model_dir / "data_adapter.py"
    pipeline_py = model_dir / "pipeline.py"
    run_py = model_dir / "run.py"
    spec_yaml = model_dir / "model_spec.yaml"
    tests_dir = model_dir / "tests"

    steps: List[PlanStep] = []

    if not _exists(model_py):
        steps.append(PlanStep("model_agent", {}))

    # data adapter should always exist/updated
    if not _exists(adapter_py) or not _exists(spec_yaml):
        steps.append(PlanStep("data_adapter_agent", {}))

    # runner/pipeline
    if not _exists(pipeline_py) or not _exists(run_py):
        steps.append(PlanStep("runner_agent", {}))

    # tests
    if not _exists(tests_dir) or not any(tests_dir.glob("test_*.py")):
        steps.append(PlanStep("test_agent", {}))

    return steps

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from .context import build_context
from .errors import UserSetupError
from .planner import build_plan
from .executor import execute_plan


def _ensure_required_inputs(ctx):
    # model folder and model_description
    ctx.paths.model_dir.mkdir(parents=True, exist_ok=True)
    if not ctx.paths.model_description_path.exists():
        raise UserSetupError(
            f"Missing model_description.md. Expected at: {ctx.paths.model_description_path}"
        )
    if not ctx.target_factor:
        raise UserSetupError(
            "Missing target_factor in config.model.target_factor. "
            "Set it to a key in {data_dir}/features/feature_dictionary.json."
        )

    # Ensure target exists in dictionary
    fd = ctx.feature_dictionary
    # We accept either top-level key or nested under independent/shared
    exists = False
    if ctx.target_factor in fd:
        exists = True
    else:
        for k in ("independent_factors", "shared_factors", "targets"):
            if isinstance(fd.get(k), dict) and ctx.target_factor in fd[k]:
                exists = True
                break
        # sometimes factors list
        for k in ("independent", "shared"):
            if isinstance(fd.get(k), list) and ctx.target_factor in fd[k]:
                exists = True
                break
    if not exists:
        raise UserSetupError(
            f"target_factor '{ctx.target_factor}' not found in feature_dictionary.json."
        )


def run_mcp(config: Any, llm_client: Any) -> Dict[str, Any]:
    """Entry point for Quant MCP.

    Args:
        config: SystemConfig
        llm_client: instance of llm.llm_client.LLMClient

    Returns:
        dict summary with plan + step results.
    """
    ctx = build_context(config, llm_client=llm_client)
    _ensure_required_inputs(ctx)

    plan = build_plan(ctx)
    results = execute_plan(ctx, plan)

    ok = all(r.success for r in results) and (len(results) == len(plan))
    return {
        "success": ok,
        "model_name": ctx.model_name,
        "data_dir": str(ctx.paths.data_dir),
        "model_dir": str(ctx.paths.model_dir),
        "target_factor": ctx.target_factor,
        "plan": [s.name for s in plan],
        "results": [
            {
                "step": r.step,
                "success": r.success,
                "message": r.message,
                "written_files": r.written_files,
                "error": asdict(r.error) if r.error else None,
            }
            for r in results
        ],
    }

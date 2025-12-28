from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict

from ..context import MCPContext
from ..codegen import llm_generate_files, write_files


SYSTEM_PROMPT = """You are a test engineer for a quant trading codebase.

Goal: write minimal smoke tests for a model folder.

You will be given model_description.md and model_spec.yaml.

Tests must:
- import model, data_adapter, pipeline modules
- build a tiny dataset (may use a small subset of tickers)
- run train/predict functions
- check that X timestamps precede y timestamps (no leakage) if meta provides them

Constraints:
- Keep tests fast and robust.
- Tests should not require network.
- Tests should read from data_dir/features; if missing, raise a clear message.

Return ONLY JSON:
{"files": [{"path": "tests/test_smoke.py", "content": "..."}, {"path": "tests/test_shapes.py", "content": "..."}]}
"""


def run(ctx: MCPContext, payload: Dict[str, Any], provided_model: bool) -> Dict[str, Any]:
    model_dir = ctx.paths.model_dir
    desc = ctx.paths.model_description_path.read_text(encoding="utf-8")
    spec_path = model_dir / "model_spec.yaml"
    spec = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""

    user_prompt = f"""Model folder: {model_dir}
Data dir: {ctx.paths.data_dir}
Target factor: {ctx.target_factor}

model_description.md:
{desc}

model_spec.yaml:
{spec}

Write tests under tests/.
Assume the runner is in pipeline.py and CLI in run.py.
"""

    files = llm_generate_files(ctx.llm_client, SYSTEM_PROMPT, user_prompt)
    written = write_files(model_dir, files)

    # Optional: run smoke test immediately if requested
    if payload.get("run_tests"):
        test_file = model_dir / "tests" / "test_smoke.py"
        if test_file.exists():
            subprocess.run(["python", str(test_file)], check=False)

    return {"message": "Generated tests.", "written_files": written}

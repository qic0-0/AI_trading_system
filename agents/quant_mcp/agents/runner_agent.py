from __future__ import annotations

from typing import Any, Dict

from ..context import MCPContext
from ..codegen import llm_generate_files, write_files


SYSTEM_PROMPT = """You are a software engineer implementing a runnable pipeline for a quant model.

You will be given model_description.md, model_spec.yaml (machine spec), and the existence of model.py and data_adapter.py.

Goal:
- Write models/{model_name}/pipeline.py and models/{model_name}/run.py.

Hard constraints:
- Do not modify user-provided model.py.
- All reads of features must happen through data_adapter.build_dataset.
- Save artifacts under models/{model_name}/artifacts and predictions under models/{model_name}/outputs.

Required behavior:
- pipeline.py defines functions train_and_save(...) and predict_and_save(...).
- run.py provides CLI: python run.py train ... / python run.py predict ...

Return ONLY JSON:
{"files": [{"path": "pipeline.py", "content": "..."}, {"path": "run.py", "content": "..."}]}
"""


def run(ctx: MCPContext, payload: Dict[str, Any], provided_model: bool) -> Dict[str, Any]:
    model_dir = ctx.paths.model_dir
    desc = ctx.paths.model_description_path.read_text(encoding="utf-8")
    spec_path = model_dir / "model_spec.yaml"
    spec = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""

    user_prompt = f"""Model folder: {model_dir}

model_description.md:
{desc}

model_spec.yaml:
{spec}

Implement pipeline.py and run.py. Use relative imports:
from .model import Model
from .data_adapter import build_dataset

Assume build_dataset returns a dict with keys: X, y, meta.
Assume Model has fit/predict and optional save/load.
"""

    files = llm_generate_files(ctx.llm_client, SYSTEM_PROMPT, user_prompt)
    written = write_files(model_dir, files)
    return {"message": "Generated runner pipeline and CLI.", "written_files": written}

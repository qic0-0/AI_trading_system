from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..context import MCPContext
from ..codegen import llm_generate_files, write_files


SYSTEM_PROMPT = """You are a senior ML engineer building a quantitative trading system.

Task: generate model code for a quant model based on a model_description.md.

Constraints:
- Write ALL artifacts within the provided model folder.
- Do NOT reference external project files except standard Python libs and common ML libs (numpy, pandas, pytorch, sklearn).
- Provide a clean, minimal API. Prefer:
    class Model:
        def fit(self, X, y=None, **params): ...
        def predict(self, X, **params): ...
        def save(self, path): ...
        @classmethod
        def load(cls, path): ...
- Also produce model_spec.yaml capturing expected input/output formats and parameters.
- Also produce MODEL_USAGE.md describing how the model is used as a quant model (e.g., daily inputs -> next-day slots).

Return ONLY JSON of the form:
{"files": [{"path": "model.py", "content": "..."}, ...]}
"""


def run(ctx: MCPContext, payload: Dict[str, Any], provided_model: bool) -> Dict[str, Any]:
    """Generate model.py and usage/spec if model.py is missing."""
    model_dir = ctx.paths.model_dir
    desc_path = ctx.paths.model_description_path
    description = desc_path.read_text(encoding="utf-8")

    user_prompt = f"""Model folder: {model_dir}

Here is model_description.md:

{description}

Generate files model.py, model_spec.yaml, MODEL_USAGE.md.
- model_spec.yaml must include: training_mode (pooled/per_ticker), input_frequency, forecast_horizon, forecast_slots (if applicable), and required_features (names) and target_name.
- If unsure, make reasonable defaults and document them in MODEL_USAGE.md.
"""

    files = llm_generate_files(ctx.llm_client, SYSTEM_PROMPT, user_prompt)
    written = write_files(model_dir, files)
    return {
        "message": "Generated model code and usage/spec.",
        "written_files": written,
    }

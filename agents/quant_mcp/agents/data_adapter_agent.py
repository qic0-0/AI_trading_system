from __future__ import annotations

import json
from typing import Any, Dict

from ..context import MCPContext
from ..codegen import llm_generate_files, write_files


SYSTEM_PROMPT = """You are a data engineering expert writing a data adapter for a quant model.

You will be given:
- model_description.md (how the model is used)
- optional model_spec.yaml content if present
- feature_dictionary.json describing available features under {data_dir}/features/
- a user chosen target_factor name (y)

Goal:
- Create models/{model_name}/data_adapter.py and models/{model_name}/model_spec.yaml (update/overwrite if needed).

Hard constraints:
- ALL data (including y) must come from the features folder produced by FeatureAgent.
- Embeddings have no time index; broadcast/forward-fill them across time as needed.
- Enforce forecasting causality: inputs X must use ONLY information available before y is realized. Use shifting/windows.
- The adapter must NOT fetch raw market data.
- Keep code readable and deterministic.

Required API:
- def build_dataset(data_dir: str, tickers: list[str], start_date: str, end_date: str, target_factor: str, mode: str = "train", **kwargs) -> dict
  returning {'X': X, 'y': y_or_None, 'meta': {...}}

Return ONLY JSON:
{"files": [{"path": "data_adapter.py", "content": "..."}, {"path": "model_spec.yaml", "content": "..."}]}
"""


def run(ctx: MCPContext, payload: Dict[str, Any], provided_model: bool) -> Dict[str, Any]:
    model_dir = ctx.paths.model_dir
    desc = ctx.paths.model_description_path.read_text(encoding="utf-8")

    spec_path = model_dir / "model_spec.yaml"
    existing_spec = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""

    fd = json.dumps(ctx.feature_dictionary, indent=2)

    user_prompt = f"""Model folder: {model_dir}
Data dir: {ctx.paths.data_dir}
Features dir: {ctx.paths.features_dir}
Target factor (y): {ctx.target_factor}

model_description.md:
{desc}

Existing model_spec.yaml (may be empty):
{existing_spec}

feature_dictionary.json:
{fd}

Write/Update data_adapter.py and model_spec.yaml.
- model_spec.yaml must explicitly document aggregation rules and time-shift rules.
- build_dataset must read from {ctx.paths.features_dir} paths described in the feature dictionary.
"""

    files = llm_generate_files(ctx.llm_client, SYSTEM_PROMPT, user_prompt)
    written = write_files(model_dir, files)
    return {"message": "Generated data adapter and model spec.", "written_files": written}

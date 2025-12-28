from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class MCPPaths:
    data_dir: Path
    features_dir: Path
    models_dir: Path
    model_dir: Path
    feature_dictionary_path: Path
    model_description_path: Path


@dataclass
class MCPPolicy:
    stop_on_failure: bool = True
    provided_model_code_is_readonly: bool = True
    missing_data_is_user_error: bool = True


@dataclass
class MCPContext:
    """Runtime context passed to MCP agents."""

    config: Any
    model_name: str
    target_factor: str
    paths: MCPPaths
    policy: MCPPolicy
    feature_dictionary: Dict[str, Any]
    feature_inventory: Dict[str, Any]   #  <-- ADD THIS
    llm_client: Any

def extract_factor_inventory(feature_dict: dict) -> dict:
    datasets = feature_dict.get("datasets", {})

    indep = datasets.get("independent_factors", {}).get("factors", {}) or {}
    shared = datasets.get("shared_factors", {}).get("factors", {}) or {}
    emb = datasets.get("embeddings", {}) or {}

    return {
        "independent": list(indep.keys()),
        "shared": list(shared.keys()),
        "embeddings": {
            "enabled": bool(emb),
            "dimension": emb.get("dimension"),
            "path_pattern": emb.get("path_pattern"),
            "tickers": emb.get("tickers", []),
        },
    }



def _getattr_path(obj: Any, name: str, default: str) -> str:
    v = getattr(obj, name, None)
    return v if isinstance(v, str) and v.strip() else default


def build_context(config: Any, llm_client: Any) -> MCPContext:
    """Build MCPContext from SystemConfig.

    We prefer MCP fields living in config.model (per your design). If they don't
    exist (older config), we fall back to sane defaults.
    """

    model_cfg = getattr(config, "model", None)
    if model_cfg is None:
        raise ValueError("config.model is required")

    # New MCP-style fields (preferred)
    data_dir_s = _getattr_path(model_cfg, "data_dir", "data")
    models_dir_s = _getattr_path(model_cfg, "models_dir", "models")
    model_name = _getattr_path(model_cfg, "model_name", "ExampleModel")
    target_factor = _getattr_path(model_cfg, "target_factor", "")

    # Backward-compat: model_path may exist in old config
    if hasattr(model_cfg, "model_path") and not hasattr(model_cfg, "models_dir"):
        models_dir_s = _getattr_path(model_cfg, "model_path", models_dir_s)

    data_dir = Path(data_dir_s)
    features_dir = data_dir / "features"
    models_dir = Path(models_dir_s)
    model_dir = models_dir / model_name

    feature_dictionary_path = features_dir / "feature_dictionary.json"
    model_description_path = model_dir / "model_description.md"

    paths = MCPPaths(
        data_dir=data_dir,
        features_dir=features_dir,
        models_dir=models_dir,
        model_dir=model_dir,
        feature_dictionary_path=feature_dictionary_path,
        model_description_path=model_description_path,
    )

    policy = MCPPolicy(
        stop_on_failure=bool(getattr(model_cfg, "stop_on_failure", True)),
        provided_model_code_is_readonly=bool(getattr(model_cfg, "provided_model_code_is_readonly", True)),
        missing_data_is_user_error=bool(getattr(model_cfg, "missing_data_is_user_error", True)),
    )

    # Load feature dictionary (required for data-adapt agent)
    if not feature_dictionary_path.exists():
        raise FileNotFoundError(f"Missing feature_dictionary.json at {feature_dictionary_path}")
    feature_dictionary = json.loads(feature_dictionary_path.read_text(encoding="utf-8"))

    feature_inventory = extract_factor_inventory(feature_dictionary)
    all_factors = set(feature_inventory["independent"]) | set(feature_inventory["shared"])

    # Validate target_factor if provided
    if target_factor and target_factor not in all_factors:
        raise ValueError(
            f"target_factor='{target_factor}' not found in feature_dictionary.json. "
            f"Available factors: {sorted(all_factors)}"
        )

    if not target_factor:
        # Allow empty target in config, but keep in context (brain will raise)
        target_factor = ""

    return MCPContext(
        config=config,
        model_name=model_name,
        target_factor=target_factor,
        paths=paths,
        policy=policy,
        feature_dictionary=feature_dictionary,
        feature_inventory=feature_inventory,  # <-- ADD THIS
        llm_client=llm_client,
    )

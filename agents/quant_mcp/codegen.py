from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from llm.llm_client import Message


class CodegenFormatError(RuntimeError):
    pass


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON object from LLM output.

    We expect the model to return a single JSON object. If it returns extra text,
    we try to locate the first '{' and last '}' and parse that.
    """
    text = text.strip()
    if not text:
        raise CodegenFormatError("Empty LLM response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise CodegenFormatError("LLM response did not contain JSON")
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as e:
            raise CodegenFormatError(f"Failed to parse JSON: {e}")


def write_files(base_dir: Path, files: List[Dict[str, str]]) -> List[str]:
    written: List[str] = []
    for f in files:
        rel_path = f.get("path")
        content = f.get("content", "")
        if not rel_path:
            continue
        path = (base_dir / rel_path).resolve()
        # Safety: ensure within model folder
        if base_dir.resolve() not in path.parents and path != base_dir.resolve():
            raise ValueError(f"Refusing to write outside base_dir: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(str(path))
    return written


def llm_generate_files(
    llm_client: Any,
    system_prompt: str,
    user_prompt: str,
) -> List[Dict[str, str]]:
    """Call LLM to generate files.

    Expected response JSON schema:
      {"files": [{"path": "relative/path.py", "content": "..."}, ...]}
    """
    msgs = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ]
    resp = llm_client.chat(msgs)
    obj = _extract_json(resp.content)
    files = obj.get("files")
    if not isinstance(files, list):
        raise CodegenFormatError("Response JSON missing 'files' list")
    out: List[Dict[str, str]] = []
    for it in files:
        if not isinstance(it, dict) or "path" not in it:
            continue
        out.append({"path": str(it["path"]), "content": str(it.get("content", ""))})
    if not out:
        raise CodegenFormatError("No files generated")
    return out

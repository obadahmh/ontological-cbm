import json
import os
from pathlib import Path

from src.paths import add_repo_root_to_sys_path

add_repo_root_to_sys_path()
from src.config import apply_env_overrides, load_config, merge_dict


def test_load_merge_json_config(tmp_path: Path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"a": 1, "nested": {"x": "y"}}), encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg["a"] == 1
    merged = merge_dict(cfg, {"nested": {"x": "z"}, "b": 2})
    assert merged["nested"]["x"] == "z"
    assert merged["b"] == 2


def test_env_override():
    os.environ["TESTPREFIX_VALUE"] = "new"
    cfg = apply_env_overrides({"value": "old"}, prefix="TESTPREFIX_")
    assert cfg["value"] == "new"
    os.environ.pop("TESTPREFIX_VALUE", None)

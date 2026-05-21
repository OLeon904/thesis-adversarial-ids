from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"
ENV_CONFIG = "THESIS_CONFIG"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_config_path(path: Path | str | None = None) -> Path:
    """Resolve config file: explicit path, THESIS_CONFIG env, or default.yaml."""
    if path is not None:
        p = Path(path)
    elif env := os.environ.get(ENV_CONFIG):
        p = Path(env)
    else:
        return DEFAULT_CONFIG

    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    cfg_path = resolve_config_path(path)
    base = _load_yaml(DEFAULT_CONFIG)
    if cfg_path.resolve() != DEFAULT_CONFIG.resolve():
        overlay = _load_yaml(cfg_path)
        cfg = _deep_merge(base, overlay)
    else:
        cfg = base

    for key in ("raw_dir", "processed_dir", "results_dir"):
        cfg["paths"][key] = str(ROOT / cfg["paths"][key])
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for key in ("raw_dir", "processed_dir", "results_dir"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def save_config_snapshot(
    out_dir: Path | str,
    config_path: Path | str | None = None,
    *,
    dest_name: str = "config_snapshot.yaml",
) -> Path:
    """Copy resolved config YAML into a run directory; return destination path."""
    src = resolve_config_path(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / dest_name
    shutil.copy2(src, dest)
    return dest.resolve()

"""Config and path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    # .../projects/papers/<this>/ecg_scf -> repo is parents[3]
    return Path(__file__).resolve().parents[3]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else project_root() / "configs" / "default.yaml"
    if not cfg_path.is_absolute():
        cand = project_root() / cfg_path
        cfg_path = cand if cand.exists() else Path(cfg_path)
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(rel: str | Path, base: Path | None = None) -> Path:
    """Resolve relative to project dir, then repo root."""
    p = Path(rel)
    if p.is_absolute():
        return p
    base = base or project_root()
    c1 = (base / p).resolve()
    if c1.exists():
        return c1
    c2 = (repo_root() / "projects" / "papers" / base.name / p).resolve()
    if c2.exists():
        return c2
    # datasets often live under repo/projects/datasets
    c3 = (repo_root() / p).resolve()
    if c3.exists():
        return c3
    # common: ../../datasets from project
    c4 = (base / p).resolve()
    return c4

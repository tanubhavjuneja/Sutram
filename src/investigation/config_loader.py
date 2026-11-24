"""Small config loader used by the CLI/web entry points.

Kept separate from :mod:`investigation.service` so the CLI can import the
config without importing the analytics stack.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .service import InvestigationConfig


def load_config(path: Optional[str] = None, case_root: Optional[str] = None,
                lang: Optional[str] = None) -> InvestigationConfig:
    cfg = InvestigationConfig()
    if path and Path(path).exists():
        cfg = InvestigationConfig.from_yaml(path)
    if case_root:
        cfg.case_root = case_root
    if lang:
        cfg.language = lang
    return cfg

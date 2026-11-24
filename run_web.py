"""Launch the Sutram Investigator web workspace.

    python run_web.py --open
    python run_web.py --case-root data/cases --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from web.server import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

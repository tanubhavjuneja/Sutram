"""Interactive conversational forensic assistant.

    python ask.py <case_id>
    python ask.py <case_id> --lang hi
    python ask.py <case_id> --question "Who did Viktor call before the transfer?"

This is a thin convenience wrapper around ``run_case.py chat`` / ``ask`` so the
assistant can be launched without remembering the full CLI syntax.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from aml.i18n import SUPPORTED_LANGUAGES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case", help="case id or path to a case directory")
    ap.add_argument("--root", default="data/cases", help="case root directory")
    ap.add_argument("--config", default=str(ROOT / "config" / "investigation.yaml"))
    ap.add_argument("--lang", default=None, choices=SUPPORTED_LANGUAGES,
                    help="answer language (default: case language)")
    ap.add_argument("--question", default="", help="ask one question and exit")
    ap.add_argument("--image", action="append", default=[],
                    help="attach an image for vision analysis (repeatable)")
    args = ap.parse_args()

    from run_case import main as run_main
    common = ["--root", args.root, "--config", args.config]
    if args.lang:
        common += ["--lang", args.lang]
    if args.question:
        command = common + ["ask", args.case, args.question]
        for image in args.image:
            command += ["--image", image]
        return run_main(command)
    return run_main(common + ["chat", args.case])


if __name__ == "__main__":
    raise SystemExit(main())

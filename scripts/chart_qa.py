"""Generate every Sutram chart and (optionally) read each one back with the
Cloudflare vision model.

Usage::

    python scripts/chart_qa.py --case operation-nightfall-20261123T195835
    python scripts/chart_qa.py --case <id> --no-vision
    python scripts/chart_qa.py --case <id> --vision-model @cf/meta/llama-4-scout-17b-16e-instruct

The script writes ``reports/chart_qa/chart_qa.md`` and ``chart_qa.json`` inside
the case directory.  Vision observations are model observations, not forensic
conclusions; the deterministic chart files are the source of truth.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import os

os.environ.setdefault("AML_SKIP_TORCH", "1")

from aml.config import load_investigation_config  # noqa: E402
from aml.service import InvestigationService  # noqa: E402
from aml import visuals  # noqa: E402

QA_PROMPT = (
    "This image is a chart from a forensic investigation report. Do a visual "
    "QA in 2-4 sentences. State: (1) the chart title, (2) the main message, "
    "(3) whether any labels are cut off, overlapping, or unreadable. Be "
    "specific about any visual defect you see. Do not invent data."
)
ILLUSTRATION_PROMPT = (
    "This is a generated illustration, not evidence. In 2-3 sentences describe "
    "what it shows and whether it contains readable text or obvious artifacts."
)


def _case_id(service: InvestigationService, requested: str) -> str:
    if requested:
        return requested
    cases = service.list_cases()
    if not cases:
        raise SystemExit("No cases found. Run `python run_case.py demo` first.")
    return cases[0]["case_id"]


def _render_all(service: InvestigationService, case_id: str, out_dir: Path,
                themes=("light", "dark")) -> List[dict]:
    case = service.load_case(case_id)
    index = service.build_index(case)
    res, error = service._result_for_charts(case, index)
    if res is None:
        raise SystemExit(f"Analytics could not run: {error}")

    charts: List[dict] = []
    for theme in themes:
        suffix = "" if theme == "light" else "_dark"
        charts.append({
            "key": f"timeline{suffix}",
            "title": "Daily traffic timeline",
            "path": visuals.render_timeline(
                res, out_dir, name=f"timeline{suffix}.png", theme=theme),
        })
        charts.append({
            "key": f"people{suffix}",
            "title": "People who stand out",
            "path": visuals.render_person_attention(
                res, out_dir, name=f"people{suffix}.png", theme=theme),
        })
        charts.append({
            "key": f"sources{suffix}",
            "title": "Ingestion volumes",
            "path": visuals.render_sources(
                res, out_dir, name=f"sources{suffix}.png", theme=theme),
        })
        charts.append({
            "key": f"network{suffix}",
            "title": "Suspect correlation graph",
            "path": visuals.render_correlation_graph(
                res, out_dir, name=f"network{suffix}.png", theme=theme,
                tol_minutes=service.config.tol_minutes),
        })
        charts.append({
            "key": f"map{suffix}",
            "title": "Activity map",
            "path": visuals.render_location_map(
                res, out_dir, name=f"map{suffix}.png", theme=theme),
        })
        if getattr(res, "model_scores", None):
            charts.append({
                "key": f"models{suffix}",
                "title": "Detection engine scores",
                "path": visuals.render_model_scores(
                    res, out_dir, name=f"models{suffix}.png", theme=theme),
            })
        for eid, _ent in sorted(
                case.entities.items(), key=lambda kv: -kv[1].risk_score)[:3]:
            eid = str(eid)
            ent = case.entities.get(eid)
            name = ent.display_name if ent else eid
            charts.append({
                "key": f"entity_{eid}{suffix}",
                "title": f"Activity timeline — {name} ({eid})",
                "path": visuals.render_entity_timeline(
                    res, out_dir, eid,
                    name=f"entity_{eid}{suffix}.png", theme=theme),
            })
    return charts


def _read_charts(service: InvestigationService, charts: List[dict],
                 use_vision: bool) -> None:
    if not use_vision:
        return
    llm = service.llm
    if not llm.available:
        print("Vision skipped: Cloudflare is not configured.")
        return
    for chart in charts:
        try:
            chart["vision"] = llm.describe_image(
                chart["path"].read_bytes(), QA_PROMPT, lang="en", max_tokens=320)
        except Exception as exc:
            chart["vision"] = f"vision failed: {type(exc).__name__}: {exc}"
        print(f"  read {chart['key']}: {(chart.get('vision') or '—')[:120]}")


def _write_report(out_dir: Path, case_id: str, charts: List[dict]) -> None:
    generated = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Chart QA — generated renders and vision read-back",
        "",
        f"- case: `{case_id}`",
        f"- generated: {generated}",
        "- vision answers are model observations, not evidence.",
        "",
    ]
    for chart in charts:
        lines += [
            f"## {chart['title']}",
            "",
            f"![{chart['title']}]({chart['path'].name})",
            "",
        ]
        if chart.get("vision"):
            lines += ["**Vision read-back:**", "", chart["vision"], ""]
    (out_dir / "chart_qa.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "case_id": case_id,
        "generated_at": generated,
        "charts": [
            {
                "key": c["key"],
                "title": c["title"],
                "file": c["path"].name,
                "vision": c.get("vision"),
            }
            for c in charts
        ],
    }
    (out_dir / "chart_qa.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="")
    ap.add_argument("--case-root", default="data/cases")
    ap.add_argument("--config", default=str(ROOT / "config" / "investigation.yaml"))
    ap.add_argument("--out", default="")
    ap.add_argument("--no-vision", action="store_true")
    args = ap.parse_args()

    cfg = load_investigation_config(args.config, case_root=args.case_root)
    service = InvestigationService(case_root=cfg.case_root, config=cfg)
    case_id = _case_id(service, args.case)
    case = service.load_case(case_id)
    out_dir = Path(args.out) if args.out else Path(case.root) / "reports" / "chart_qa"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating charts for {case_id} -> {out_dir}")
    charts = _render_all(service, case_id, out_dir)
    _read_charts(service, charts, use_vision=not args.no_vision)
    _write_report(out_dir, case_id, charts)
    print(f"Wrote {out_dir / 'chart_qa.md'}")
    print(f"Wrote {out_dir / 'chart_qa.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

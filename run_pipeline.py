"""One-command analysis: auto-detect schema, auto-run models, write a report.

Usage:
    python run_pipeline.py --config config/config.yaml [data/...] [file.csv ...]

Pass zero or more files/directories. Any CSV / JSON / JSONL / TXT /LOG file is
loaded, columns are auto-detected, the models the data can support are run
automatically, and a markdown + JSON report is written to the configured dir.

Investigation mode (the same pipeline, case workspace enabled):
    python run_pipeline.py --investigation --title "Operation Nightfall" evidence/
    python run_pipeline.py --investigation --case-id <id> --ask "What looks suspicious?"
    python run_pipeline.py --investigation --case-id <id> --lang hi
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from aml.config import (  # noqa: E402
    PipelineConfig,
    load_investigation_config,
)
from aml.loaders import collect_files  # noqa: E402
from aml.pipeline import CasePipeline, Pipeline  # noqa: E402
from aml.report import generate_report  # noqa: E402


def _collect_case_paths(inputs) -> list:
    """Collect every file (structured or document) for the case workspace."""
    out: list[str] = []
    seen = set()
    for raw in inputs:
        path = Path(raw)
        candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            key = str(candidate.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(str(candidate))
    return out


def _run_investigation(args, cfg: PipelineConfig) -> int:
    inv_cfg = load_investigation_config(
        args.investigation_config,
        case_root=args.case_root,
        lang=args.lang,
    )
    # The investigation workspace inherits the analytics thresholds unless the
    # investigation config overrides them.
    pipe = CasePipeline(
        case_root=args.case_root,
        investigation_config=inv_cfg,
        min_events_per_entity=inv_cfg.min_events_per_entity,
        tol_minutes=inv_cfg.tol_minutes,
        structuring_threshold=inv_cfg.structuring_threshold,
        probe_cfg=cfg.probe_config(),
    )

    paths = _collect_case_paths(args.inputs) if args.inputs else []
    case_id = args.case_id
    if not case_id:
        if not paths:
            print("Investigation mode needs evidence files or --case-id.", file=sys.stderr)
            return 2
        title = args.title or Path(paths[0]).parent.name or "Untitled case"
        case = pipe.create_case(title, description="Created by run_pipeline.py",
                                language=args.lang)
        case_id = case.case_id
        print(f"Created case {case_id}")

    if paths:
        ingested = pipe.ingest_case(case_id, paths, notes="run_pipeline.py")
        print(f"Ingested {len(ingested.get('added', []))} evidence item(s): "
              f"{ingested.get('events', 0)} events, {ingested.get('chunks', 0)} chunks, "
              f"{ingested.get('entities', 0)} entities")
        for warning in ingested.get("warnings", []):
            print(f"  warning: {warning}")

    if args.ask:
        answer = pipe.ask(case_id, args.ask, lang=args.lang)
        print(answer.answer)
        print(f"\n[intent: {answer.intent} · confidence: {answer.confidence:.2f}]")
        if answer.followups:
            print("Follow-ups:")
            for question in answer.followups:
                print(f"  - {question}")
        return 0

    result = pipe.run_case_result(case_id, analyze=True, report=True, lang=args.lang)
    case = result.case
    print(f"\nCase {case.case_id}: {case.title}")
    print(f"  evidence: {len(result.evidence)}  entities: {len(case.entities)}  "
          f"findings: {len(result.findings)}  leads: {len(result.recommendations)}")
    if result.summary:
        print(f"\n{result.summary.get('text', '')}")
    if result.findings:
        print("\nTop findings:")
        for finding in result.findings[:5]:
            print(f"  [{finding.severity}] {finding.title}")
    if result.recommendations:
        print("\nTop recommendations:")
        for rec in result.recommendations[:5]:
            print(f"  [{rec.priority}] {rec.action}")
    print(f"\nCase workspace: {case.root}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--investigation", action="store_true",
                    help="use the case workspace (evidence + assistant + report)")
    ap.add_argument("--investigation-config",
                    default=str(ROOT / "config" / "investigation.yaml"))
    ap.add_argument("--case-root", default="data/cases")
    ap.add_argument("--case-id", default="")
    ap.add_argument("--title", default="")
    ap.add_argument("--ask", default="")
    ap.add_argument("--lang", default="en")
    ap.add_argument("inputs", nargs="*", help="files or directories to analyze")
    args = ap.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    cfg.report_dir = cfg.report_dir or "data/report"

    if args.investigation or args.case_id or args.ask:
        return _run_investigation(args, cfg)

    if args.inputs:
        paths = collect_files(args.inputs)
    else:
        src_dir = ROOT / (cfg.sources_dir or "data/sources")
        paths = collect_files([str(src_dir)])

    if not paths:
        print("No supported input files found.", file=sys.stderr)
        return 2

    print(f"Loaded {len(paths)} file(s):")
    for p in paths:
        print(f"   - {Path(p).name}")

    pipe = Pipeline(
        min_events_per_entity=cfg.min_events_per_entity,
        tol_minutes=cfg.tol_minutes,
        structuring_threshold=cfg.structuring_threshold,
        probe_cfg=cfg.probe_config(),
    )
    result = pipe.run(paths, with_models=cfg.with_models)

    print("\nSources detected:")
    for rs in result.per_source:
        print(f"   {rs.source:8s} {Path(rs.file).name:24s} rows={rs.n_rows:<6,} "
              f"entities={rs.n_entities}")
    print("Model verdicts:")
    for v in result.sufficiency.values():
        if v.model == "schema_detection":
            continue
        print(f"   {v.model:20s} {v.status:<9s} {v.reason[:80]}")

    written = generate_report(result, cfg, cfg.report_dir)

    print("\nRanked insights (top 10):")
    for i, r in enumerate(result.rankings[:10], 1):
        print(f"   {i:2d}. {r['entity_id']}  score={r['score']:.3f}  fired={r['models_fired']}")
    print("\nReport written:")
    for kind, path in written.items():
        print(f"   {kind:8s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

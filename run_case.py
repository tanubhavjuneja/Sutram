"""Sutram Investigator — case workspace CLI.

Examples
--------
    python run_case.py demo
    python run_case.py list
    python run_case.py create --title "Operation Nightfall" --lang en
    python run_case.py ingest <case_id> evidence/*.csv evidence/*.eml
    python run_case.py analyze <case_id>
    python run_case.py ask <case_id> "Who did Viktor call right before the transfer?"
    python run_case.py chat <case_id> --lang hi
    python run_case.py summary <case_id>
    python run_case.py recommend <case_id>
    python run_case.py diary <case_id>
    python run_case.py report <case_id> --lang en --formats md,html,json,pdf

Case data lives under ``data/cases/<case_id>/`` (configurable with ``--root``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# Avoid the eager torch probe in aml/__init__.py for the investigation CLI;
# the analytics models that need torch are optional and warn-skip anyway.
import os
os.environ.setdefault("AML_SKIP_TORCH", "1")

from investigation import InvestigationService  # noqa: E402
from investigation.config_loader import load_config  # noqa: E402
from investigation.i18n import SUPPORTED_LANGUAGES, language_name  # noqa: E402


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _print_json(value) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def _service(args) -> InvestigationService:
    cfg = load_config(args.config, case_root=args.root, lang=args.lang)
    return InvestigationService(case_root=cfg.case_root, config=cfg, lang=args.lang)


def _resolve_case(svc: InvestigationService, case_id: str):
    return svc.load_case(case_id)


def cmd_list(args) -> int:
    svc = _service(args)
    cases = svc.list_cases()
    if args.json:
        _print_json(cases)
        return 0
    if not cases:
        print("No cases yet. Run: python run_case.py demo")
        return 0
    print(f"{'CASE ID':48s} {'TITLE':32s} {'EVID':>5s} {'ENT':>4s} {'FIND':>5s} {'REC':>4s}")
    for c in cases:
        s = c["stats"]
        print(f"{c['case_id']:48s} {c['title'][:32]:32s} {s['evidence']:5d} "
              f"{s['entities']:4d} {s['findings']:5d} {s['recommendations']:4d}")
    return 0


def cmd_create(args) -> int:
    svc = _service(args)
    case = svc.create_case(args.title, description=args.description or "", language=args.lang)
    print(f"Created case {case.case_id}")
    print(f"  root: {case.root}")
    if args.json:
        _print_json(case.to_dict())
    return 0


def cmd_ingest(args) -> int:
    svc = _service(args)
    result = svc.ingest(args.case, args.paths, custodian=args.custodian or "",
                        notes=args.notes or "")
    if args.json:
        _print_json(result)
        return 0
    print(f"Case {result['case_id']}: ingested {len(result['added'])} item(s)")
    print(f"  events: {result['events']}  chunks: {result['chunks']}  entities: {result['entities']}")
    for item in result["per_evidence"]:
        print(f"  {item['evidence_id']}  {item['filename']:32s} {item['source_type']:10s} "
              f"rows={item['rows']:<6} chunks={item['chunks']}")
        for warning in item["warnings"]:
            print(f"      warning: {warning}")
    for warning in result["warnings"]:
        print(f"  warning: {warning}")
    return 0


def cmd_analyze(args) -> int:
    svc = _service(args)
    result = svc.analyze(args.case, run_report=not args.no_report)
    if args.json:
        _print_json(result)
        return 0
    if not result.get("ok"):
        print(f"Analysis not run: {result.get('error')}")
        return 1
    print(f"Case {result['case_id']}: analysis complete")
    print(f"  models: {', '.join(result['models']) or '—'}")
    print(f"  findings: {result['findings']}  recommendations: {result['recommendations']}")
    print("  sufficiency:")
    for model, status in result["sufficiency"].items():
        print(f"    {model:20s} {status}")
    if result["top_rankings"]:
        print("  top rankings:")
        for i, r in enumerate(result["top_rankings"][:10], 1):
            print(f"    {i:2d}. {r['entity_id']}  score={r['score']:.3f}  "
                  f"fired={','.join(r.get('models_fired', []))}")
    for warning in result["warnings"]:
        print(f"  warning: {warning}")
    return 0


def cmd_ask(args) -> int:
    svc = _service(args)
    answer = svc.ask(args.case, args.question, lang=args.lang, session_id=args.session)
    if args.json:
        _print_json(answer.to_dict())
        return 0
    print(answer.answer)
    print(f"\n[intent: {answer.intent} · confidence: {answer.confidence:.2f} · "
          f"language: {answer.language}]")
    if answer.followups:
        print("\nFollow-ups:")
        for q in answer.followups:
            print(f"  - {q}")
    return 0


def cmd_chat(args) -> int:
    svc = _service(args)
    case = _resolve_case(svc, args.case)
    lang = args.lang or case.language
    print(f"Sutram Investigator — {case.title} ({case.case_id})")
    print("Type a question, or :lang <code>, :summary, :diary, :recommend, :report, :help, :quit")
    print(f"Languages: {', '.join(SUPPORTED_LANGUAGES)}")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in (":quit", ":q", "exit", "quit"):
            break
        if line.startswith(":lang"):
            parts = line.split()
            if len(parts) > 1 and parts[1] in SUPPORTED_LANGUAGES:
                lang = parts[1]
                print(f"Language set to {language_name(lang)} ({lang})")
            else:
                print(f"Supported: {', '.join(SUPPORTED_LANGUAGES)}")
            continue
        if line == ":summary":
            result = svc.summarize(case.case_id, lang=lang)
            print(result["text"])
            continue
        if line == ":diary":
            result = svc.diary(case.case_id, lang=lang)
            for entry in result["entries"]:
                print(f"- {entry['timestamp'][:19]} · {entry['entry_type']} — {entry['text']}")
            print(f"Chain: {'verified' if result['chain_ok'] else 'TAMPERED'}")
            continue
        if line == ":recommend":
            result = svc.recommend(case.case_id, lang=lang)
            for rec in result["recommendations"][:12]:
                print(f"- [{rec['priority']}] {rec['action']}")
                print(f"    {rec['rationale']}")
            continue
        if line == ":report":
            result = svc.report(case.case_id, lang=lang)
            print("Report files:")
            for kind, path in result["files"].items():
                print(f"  {kind}: {path}")
            continue
        if line in (":help", "help"):
            print("Ask about evidence, entities, timelines, relationships, suspicious patterns,")
            print("recommendations, the diary or the report. Try:")
            for q in [
                "What evidence do we have?",
                "Who is the main suspect?",
                "Show transfers above 10,000 in the last 30 days.",
                "What is the connection between the top two entities?",
                "What looks suspicious?",
                "What should I investigate next?",
                "Draft a case report.",
            ]:
                print(f"  - {q}")
            continue
        answer = svc.ask(case.case_id, line, lang=lang)
        print()
        print(answer.answer)
        print(f"\n[intent: {answer.intent} · confidence: {answer.confidence:.2f}]")
    return 0


def cmd_summary(args) -> int:
    svc = _service(args)
    result = svc.summarize(args.case, lang=args.lang)
    if args.json:
        _print_json(result)
    else:
        print(result["text"])
        if result.get("limitations"):
            from investigation.i18n import t
            case = svc.load_case(args.case)
            print("\n" + t("report.limitations", args.lang or case.language) + ":")
            for item in result["limitations"]:
                print(f"- {item}")
    return 0


def cmd_entities(args) -> int:
    svc = _service(args)
    case = _resolve_case(svc, args.case)
    entities = sorted(case.entities.values(), key=lambda e: (-e.risk_score, -e.n_events, e.display_name))
    if args.json:
        _print_json([e.to_dict() for e in entities])
        return 0
    print(f"{'ID':8s} {'NAME':28s} {'RISK':>5s} {'EVENTS':>6s} {'SOURCES':24s} IDENTIFIERS")
    for e in entities:
        print(f"{e.entity_id:8s} {e.display_name[:28]:28s} {e.risk_score:5.2f} "
              f"{e.n_events:6d} {','.join(e.sources)[:24]:24s} {','.join(e.identifiers[:4])}")
    return 0


def cmd_profile(args) -> int:
    svc = _service(args)
    result = svc.entity_profile(args.case, args.entity, lang=args.lang)
    if not result:
        print(f"Entity not found: {args.entity}")
        return 1
    if args.json:
        _print_json(result)
    else:
        print(result["text"])
        if result.get("findings"):
            print("\nFindings:")
            for f in result["findings"]:
                print(f"- [{f['severity']}] {f['title']}")
        if result.get("citations"):
            print("\nEvidence:")
            for c in result["citations"][:6]:
                print(f"- {c['file']} · {c['timestamp']} — {c['snippet']}")
    return 0


def cmd_timeline(args) -> int:
    svc = _service(args)
    result = svc.timeline(args.case, query=args.query or "", lang=args.lang, limit=args.limit)
    if args.json:
        _print_json(result)
        return 0
    print(result["text"])
    for item in result["items"][: args.limit]:
        print(f"- {item.get('timestamp', '')[:19]} · {item.get('source', '')} — "
              f"{item.get('title', '')}")
        print(f"    {item.get('snippet', '')[:240]}")
    return 0


def cmd_findings(args) -> int:
    svc = _service(args)
    findings = svc.findings(args.case)
    if args.json:
        _print_json(findings)
        return 0
    if not findings:
        print("No findings yet. Run: python run_case.py analyze <case>")
        return 0
    for f in findings:
        print(f"[{f['severity']}] {f['title']}  ({f['kind']}, confidence {f['confidence']:.2f})")
        print(f"    {f['description']}")
        print(f"    entities={f['entity_ids']} evidence={f['evidence_ids']}")
    return 0


def cmd_recommend(args) -> int:
    svc = _service(args)
    result = svc.recommend(args.case, lang=args.lang)
    if args.json:
        _print_json(result)
        return 0
    for rec in result["recommendations"]:
        print(f"[{rec['priority']}] {rec['action']}")
        print(f"    {rec['rationale']}")
        print(f"    category={rec['category']} id={rec['recommendation_id']}")
    return 0


def cmd_diary(args) -> int:
    svc = _service(args)
    result = svc.diary(args.case, lang=args.lang)
    if args.json:
        _print_json(result)
        return 0
    print(f"{result['title']} — chain {'verified' if result['chain_ok'] else 'TAMPERED'}")
    for entry in result["entries"]:
        print(f"- {entry['timestamp'][:19]} · {entry['entry_type']:14s} · "
              f"{entry['actor']:8s} — {entry['text']}")
    return 0


def cmd_report(args) -> int:
    svc = _service(args)
    formats = tuple(x.strip() for x in args.formats.split(",") if x.strip())
    result = svc.report(args.case, lang=args.lang, formats=formats)
    if args.json:
        _print_json(result)
        return 0
    print("Report files:")
    for kind, path in result["files"].items():
        print(f"  {kind:6s} {path}")
    return 0


def cmd_note(args) -> int:
    svc = _service(args)
    entry = svc.add_note(args.case, args.text, actor=args.actor or "analyst")
    print(f"Diary entry {entry['entry_id']}: {entry['text']}")
    return 0


def cmd_verify(args) -> int:
    svc = _service(args)
    result = svc.verify_diary(args.case)
    if args.json:
        _print_json(result)
    else:
        print(f"Diary chain: {'verified' if result['ok'] else 'TAMPERED'}")
        for error in result["errors"]:
            print(f"- {error}")
    return 0 if result["ok"] else 1


def cmd_overview(args) -> int:
    svc = _service(args)
    result = svc.overview(args.case, lang=args.lang)
    if args.json:
        _print_json(result)
        return 0
    print(result["summary"]["text"])
    print("\nStats:", json.dumps(result["stats"], ensure_ascii=False))
    print("Gaps:")
    for gap in result["gaps"]:
        print(f"- {gap['key']} ({gap['severity']})")
    return 0


def cmd_demo(args) -> int:
    from investigation.demo_case import build_demo_case
    result = build_demo_case(
        root=Path(args.demo_dir) if args.demo_dir else ROOT / "data" / "demo_case",
        case_root=Path(args.root),
        lang=args.lang or "en",
    )
    if args.json:
        _print_json(result)
        return 0
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/cases", help="case root directory")
    ap.add_argument("--config", default=str(ROOT / "config" / "investigation.yaml"))
    ap.add_argument("--lang", default=None, choices=SUPPORTED_LANGUAGES,
                    help="UI/answer language (default: case language)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    sub = ap.add_subparsers(dest="command", required=True)

    # Allow --lang both before and after the subcommand.  The subparser default
    # is SUPPRESS so it does not overwrite a global --lang supplied before it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--lang", default=argparse.SUPPRESS,
                        choices=SUPPORTED_LANGUAGES,
                        help="UI/answer language (default: case language)")
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="machine-readable output")
    _SUB_PARSER = sub.add_parser

    def _sub(name, **kwargs):
        return _SUB_PARSER(name, parents=[common], **kwargs)

    p = _sub("list", help="list cases")
    p.set_defaults(func=cmd_list)

    p = _sub("create", help="create a case")
    p.add_argument("--title", required=True)
    p.add_argument("--description", default="")
    p.set_defaults(func=cmd_create)

    p = _sub("ingest", help="add evidence files to a case")
    p.add_argument("case")
    p.add_argument("paths", nargs="+")
    p.add_argument("--custodian", default="")
    p.add_argument("--notes", default="")
    p.set_defaults(func=cmd_ingest)

    p = _sub("analyze", help="run the anomaly analytics on structured evidence")
    p.add_argument("case")
    p.add_argument("--no-report", action="store_true", help="skip the analytics HTML/PDF report")
    p.set_defaults(func=cmd_analyze)

    p = _sub("ask", help="ask one natural-language question")
    p.add_argument("case")
    p.add_argument("question")
    p.add_argument("--session", default="default")
    p.set_defaults(func=cmd_ask)

    p = _sub("chat", help="interactive conversational assistant")
    p.add_argument("case")
    p.set_defaults(func=cmd_chat)

    p = _sub("summary", help="case summary")
    p.add_argument("case")
    p.set_defaults(func=cmd_summary)

    p = _sub("entities", help="list entities")
    p.add_argument("case")
    p.set_defaults(func=cmd_entities)

    p = _sub("profile", help="entity dossier")
    p.add_argument("case")
    p.add_argument("entity")
    p.set_defaults(func=cmd_profile)

    p = _sub("timeline", help="case timeline")
    p.add_argument("case")
    p.add_argument("--query", default="")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_timeline)

    p = _sub("findings", help="list findings")
    p.add_argument("case")
    p.set_defaults(func=cmd_findings)

    p = _sub("recommend", help="generate/list investigative recommendations")
    p.add_argument("case")
    p.set_defaults(func=cmd_recommend)

    p = _sub("diary", help="show the case diary")
    p.add_argument("case")
    p.set_defaults(func=cmd_diary)

    p = _sub("report", help="generate the draft case report")
    p.add_argument("case")
    p.add_argument("--formats", default="md,html,json,pdf")
    p.set_defaults(func=cmd_report)

    p = _sub("note", help="append a note to the case diary")
    p.add_argument("case")
    p.add_argument("text")
    p.add_argument("--actor", default="analyst")
    p.set_defaults(func=cmd_note)

    p = _sub("verify", help="verify the diary hash chain")
    p.add_argument("case")
    p.set_defaults(func=cmd_verify)

    p = _sub("overview", help="case overview JSON")
    p.add_argument("case")
    p.set_defaults(func=cmd_overview)

    p = _sub("demo", help="generate and run the demo case")
    p.add_argument("--demo-dir", default="")
    p.set_defaults(func=cmd_demo)
    return ap


def main(argv=None) -> int:
    _utf8_stdout()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

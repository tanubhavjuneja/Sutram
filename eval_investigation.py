"""Evaluation harness for the investigation assistant.

Measures the new problem-statement deliverables on a demo/ground-truth case:

* intent accuracy (multilingual query understanding)
* entity-resolution hit rate
* expected-keyword containment
* citation coverage
* evidence-file recall@5 / MRR (retrieval proxy)
* recommendation category coverage
* per-language parity

Usage:
    python eval_investigation.py --case <case_id>
    python eval_investigation.py --case <case_id> --ground-truth data/demo_case/ground_truth.json
    python eval_investigation.py --case <case_id> --out reports/eval.json --strict
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import os
os.environ.setdefault("AML_SKIP_TORCH", "1")

from aml.service import InvestigationService  # noqa: E402
from aml.config import load_config  # noqa: E402


def _json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str).lower()


def _contains_any(text: str, needles) -> bool:
    low = (text or "").lower()
    return any(str(n).lower() in low for n in needles if str(n).strip())


def _citation_files(answer) -> list:
    return [c.file or "" for c in answer.citations]


def _entity_hit(answer, expected_names, case) -> bool:
    if not expected_names:
        return True
    hay = (answer.answer or "").lower()
    if _contains_any(hay, expected_names):
        return True
    if _contains_any(" ".join(c.snippet or "" for c in answer.citations), expected_names):
        return True
    if _contains_any(_json_text(answer.data), expected_names):
        return True
    # fall back to entity ids whose names match
    for name in expected_names:
        for ent in case.entities.values():
            if name.lower() in [n.lower() for n in ent.names]:
                if ent.entity_id in _json_text(answer.data):
                    return True
    return False


def _evidence_rank(answer, expected_files) -> int:
    """1-based rank of the first citation matching any expected file, else 0."""
    if not expected_files:
        return 0
    for i, citation in enumerate(answer.citations, 1):
        file = (citation.file or "").lower()
        for pattern in expected_files:
            if fnmatch.fnmatch(file, str(pattern).lower()) or str(pattern).lower() in file:
                return i
    return 0


def _recommendation_coverage(svc, case_id, expected_categories) -> dict:
    result = svc.recommend(case_id)
    actual = {r["category"] for r in result["recommendations"]}
    expected = set(expected_categories or [])
    hit = expected & actual
    return {
        "expected": sorted(expected),
        "actual": sorted(actual),
        "hit": sorted(hit),
        "missing": sorted(expected - actual),
        "coverage": round(len(hit) / max(len(expected), 1), 3),
    }


def _run_query(svc, case, query: dict) -> dict:
    start = time.perf_counter()
    answer = svc.ask(case.case_id, query["question"], lang=query.get("lang", "en"),
                     session_id=f"eval-{query.get('id', 'q')}")
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    expected_intent = query.get("intent", "")
    intent_ok = (answer.intent == expected_intent) if expected_intent else True
    entity_ok = _entity_hit(answer, query.get("entity_names", []), case)
    keyword_ok = _contains_any(answer.answer, query.get("keywords", [])) if query.get("keywords") else True
    rank = _evidence_rank(answer, query.get("evidence_files", []))
    evidence_expected = bool(query.get("evidence_files"))
    citation_required = answer.intent in {
        "search", "entity_profile", "timeline", "relationship", "correlation",
        "aggregate", "anomaly", "summarize", "recommend",
    }
    return {
        "id": query.get("id", ""),
        "lang": query.get("lang", "en"),
        "question": query["question"],
        "expected_intent": expected_intent,
        "actual_intent": answer.intent,
        "intent_ok": intent_ok,
        "entity_ok": entity_ok,
        "keyword_ok": keyword_ok,
        "citation_count": len(answer.citations),
        "citation_required": citation_required,
        "citation_ok": len(answer.citations) > 0,
        "evidence_expected": evidence_expected,
        "evidence_rank": rank,
        "evidence_ok": (rank > 0) if evidence_expected else True,
        "latency_ms": latency_ms,
        "confidence": answer.confidence,
        "answer_preview": answer.answer[:400],
        "citation_files": _citation_files(answer),
    }


def _aggregate(results: list, rec_cov: dict) -> dict:
    n = max(len(results), 1)
    evidence_queries = [r for r in results if r["evidence_expected"]]
    citation_queries = [r for r in results if r["citation_required"]]
    by_lang = defaultdict(list)
    for r in results:
        by_lang[r["lang"]].append(r)
    lang_metrics = {}
    for lang, rows in by_lang.items():
        lang_citation = [r for r in rows if r["citation_required"]]
        lang_metrics[lang] = {
            "n": len(rows),
            "intent_accuracy": round(sum(r["intent_ok"] for r in rows) / len(rows), 3),
            "entity_hit_rate": round(sum(r["entity_ok"] for r in rows) / len(rows), 3),
            "keyword_hit_rate": round(sum(r["keyword_ok"] for r in rows) / len(rows), 3),
            "citation_coverage": round(
                sum(r["citation_ok"] for r in lang_citation) / max(len(lang_citation), 1), 3),
            "evidence_recall_at_5": round(
                sum(1 for r in rows if r["evidence_expected"] and 0 < r["evidence_rank"] <= 5)
                / max(sum(1 for r in rows if r["evidence_expected"]), 1), 3),
        }
    mrr = 0.0
    if evidence_queries:
        mrr = sum((1.0 / r["evidence_rank"]) if r["evidence_rank"] else 0.0
                  for r in evidence_queries) / len(evidence_queries)
    return {
        "n_queries": len(results),
        "intent_accuracy": round(sum(r["intent_ok"] for r in results) / n, 3),
        "entity_hit_rate": round(sum(r["entity_ok"] for r in results) / n, 3),
        "keyword_hit_rate": round(sum(r["keyword_ok"] for r in results) / n, 3),
        "citation_coverage": round(
            sum(r["citation_ok"] for r in citation_queries) / max(len(citation_queries), 1), 3),
        "citation_queries": len(citation_queries),
        "evidence_queries": len(evidence_queries),
        "evidence_recall_at_5": round(
            sum(1 for r in evidence_queries if 0 < r["evidence_rank"] <= 5)
            / max(len(evidence_queries), 1), 3),
        "mrr": round(mrr, 3),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / n, 1),
        "recommendation_coverage": rec_cov,
        "by_language": lang_metrics,
    }


def _write_markdown(path: Path, results: list, metrics: dict) -> None:
    lines = [
        "# Investigation assistant evaluation",
        "",
        f"- queries: {metrics['n_queries']}",
        f"- intent accuracy: {metrics['intent_accuracy']:.3f}",
        f"- entity hit rate: {metrics['entity_hit_rate']:.3f}",
        f"- keyword hit rate: {metrics['keyword_hit_rate']:.3f}",
        f"- citation coverage: {metrics['citation_coverage']:.3f}",
        f"- evidence recall@5: {metrics['evidence_recall_at_5']:.3f}",
        f"- MRR: {metrics['mrr']:.3f}",
        f"- recommendation coverage: {metrics['recommendation_coverage']['coverage']:.3f}",
        f"- avg latency: {metrics['avg_latency_ms']:.1f} ms",
        "",
        "## Per query",
        "",
        "| id | lang | expected | actual | intent | entity | keyword | cites | evidence rank |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['id']} | {r['lang']} | {r['expected_intent']} | {r['actual_intent']} | "
            f"{'✓' if r['intent_ok'] else '✗'} | {'✓' if r['entity_ok'] else '✗'} | "
            f"{'✓' if r['keyword_ok'] else '✗'} | {r['citation_count']} | "
            f"{r['evidence_rank'] or '—'} |"
        )
    lines += ["", "## Per language", "",
              "| lang | n | intent | entity | keyword | citations | recall@5 |",
              "|---|---|---|---|---|---|---|"]
    for lang, m in sorted(metrics["by_language"].items()):
        lines.append(
            f"| {lang} | {m['n']} | {m['intent_accuracy']:.3f} | {m['entity_hit_rate']:.3f} | "
            f"{m['keyword_hit_rate']:.3f} | {m['citation_coverage']:.3f} | "
            f"{m['evidence_recall_at_5']:.3f} |"
        )
    lines += ["", "## Recommendation coverage", "",
              f"- expected: {', '.join(metrics['recommendation_coverage']['expected'])}",
              f"- actual: {', '.join(metrics['recommendation_coverage']['actual'])}",
              f"- missing: {', '.join(metrics['recommendation_coverage']['missing']) or 'none'}",
              ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True, help="case id or case directory")
    ap.add_argument("--root", default="data/cases")
    ap.add_argument("--config", default=str(ROOT / "config" / "investigation.yaml"))
    ap.add_argument("--ground-truth", default="", help="ground_truth.json path")
    ap.add_argument("--out", default="", help="output JSON path")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if intent accuracy < 0.8 or entity hit rate < 0.7")
    args = ap.parse_args()

    cfg = load_config(args.config, case_root=args.root)
    svc = InvestigationService(case_root=cfg.case_root, config=cfg)
    case = svc.load_case(args.case)

    truth_path = args.ground_truth or (case.metadata or {}).get("ground_truth", "")
    if not truth_path or not Path(truth_path).exists():
        print("No ground truth found. Pass --ground-truth or use the demo case.",
              file=sys.stderr)
        return 2
    truth = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    queries = truth.get("queries", [])
    if not queries:
        print("Ground truth has no queries.", file=sys.stderr)
        return 2

    results = []
    for query in queries:
        result = _run_query(svc, case, query)
        results.append(result)
        print(f"[{result['id']:20s}] intent={result['actual_intent']:16s} "
              f"cites={result['citation_count']:<2d} evidence_rank={result['evidence_rank']} "
              f"{'OK' if result['intent_ok'] else 'INTENT-MISS'}")

    rec_cov = _recommendation_coverage(
        svc, case.case_id, truth.get("expected_recommendation_categories", []))
    metrics = _aggregate(results, rec_cov)

    out_path = Path(args.out) if args.out else Path(case.root) / "reports" / "eval_investigation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"metrics": metrics, "results": results},
                                   indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
    _write_markdown(out_path.with_suffix(".md"), results, metrics)

    print("\n" + "=" * 64)
    print("INVESTIGATION EVALUATION")
    for key in ("n_queries", "intent_accuracy", "entity_hit_rate", "keyword_hit_rate",
                "citation_coverage", "evidence_recall_at_5", "mrr", "avg_latency_ms"):
        print(f"  {key:24s} {metrics[key]}")
    print(f"  recommendation_coverage  {metrics['recommendation_coverage']['coverage']}")
    print("  by language:")
    for lang, m in sorted(metrics["by_language"].items()):
        print(f"    {lang}: n={m['n']} intent={m['intent_accuracy']} "
              f"entity={m['entity_hit_rate']} keyword={m['keyword_hit_rate']} "
              f"cites={m['citation_coverage']} recall@5={m['evidence_recall_at_5']}")
    print(f"\nWrote {out_path}")
    print(f"Wrote {out_path.with_suffix('.md')}")
    print("=" * 64)

    if args.strict and (metrics["intent_accuracy"] < 0.8 or metrics["entity_hit_rate"] < 0.7):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

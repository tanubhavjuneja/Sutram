"""Investigative lead generation.

Recommendations are derived from two sources:

1. **Evidence gaps** — what the case cannot currently prove (missing bank/CDR/
   social/email evidence, undated records, thin volume, unresolved identifiers).
2. **Findings** — what the analytics layer flagged (structuring, colocation,
   layering, fan-in/out, dormancy, generic anomalies).

Each recommendation carries a priority, a category, a rationale and links to
the findings/evidence that justify it.  The rules are deterministic and
auditable; an LLM can rephrase them later but never invents the lead.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from .i18n import t
from .index import EvidenceIndex, _safe_ts
from .models import Case, Finding, Recommendation
from .summarize import evidence_gaps

_PRIORITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def _entity_name(case: Case, entity_id: str) -> str:
    ent = case.entities.get(str(entity_id))
    return ent.display_name if ent else str(entity_id)


def _entity_identifiers(case: Case, entity_id: str) -> str:
    ent = case.entities.get(str(entity_id))
    return ", ".join(ent.identifiers[:6]) if ent else "—"


def _finding_location(finding: Finding) -> str:
    loc = (finding.metadata or {}).get("location")
    if loc:
        return str(loc)
    for citation in finding.citations:
        snippet = citation.snippet or ""
        for token in snippet.split():
            if token.upper().startswith(("T", "BRANCH", "LAT", "LOC")) and any(ch.isdigit() for ch in token):
                return token
    return "the relevant location"


def _finding_timestamp(finding: Finding) -> str:
    for citation in finding.citations:
        if citation.timestamp:
            return str(citation.timestamp)[:19].replace("T", " ")
    return "the relevant time"


def _counterparty_count(index: EvidenceIndex, entity_id: str) -> int:
    if index.events.empty:
        return 0
    sub = index.events[
        (index.events["entity_id"] == str(entity_id))
        & (index.events["counterparty_id"].astype(str) != "")
    ]
    return int(sub["counterparty_id"].nunique())


def _timeline_gap(index: EvidenceIndex, min_days: int = 30) -> Optional[tuple]:
    stamps: List[pd.Timestamp] = []
    if not index.events.empty:
        stamps.extend([s for s in index.events["event_ts"].dropna().tolist()])
    for chunk in index.chunks:
        ts = _safe_ts(chunk.timestamp)
        if ts is not None:
            stamps.append(ts)
    if len(stamps) < 2:
        return None
    stamps = sorted(stamps)
    best = None
    for a, b in zip(stamps, stamps[1:]):
        gap = (b - a).days
        if gap >= min_days and (best is None or gap > best[0]):
            best = (gap, a, b)
    if not best:
        return None
    return (best[0], best[1].isoformat(), best[2].isoformat())


def generate_recommendations(case: Case, index: EvidenceIndex,
                             lang: str = "en", min_events: int = 30) -> List[Recommendation]:
    """Regenerate the automatic recommendation set (manual ones are kept)."""
    existing_manual = [
        r for r in case.recommendations
        if not (r.metadata or {}).get("auto")
    ]
    recs: List[Recommendation] = []
    seen: set = set()

    def add(key: str, priority: str, category: str,
            action_key: str, rationale_key: str, params: dict,
            entity_ids: Optional[List[str]] = None,
            finding_ids: Optional[List[str]] = None,
            evidence_ids: Optional[List[str]] = None) -> None:
        dedupe = f"{key}|{','.join(entity_ids or [])}|{','.join(finding_ids or [])}"
        if dedupe in seen:
            return
        seen.add(dedupe)
        action = t(action_key, lang, **params)
        rationale = t(rationale_key, lang, **params)
        recs.append(Recommendation(
            recommendation_id="",  # assigned after sorting
            priority=priority,
            action=action,
            rationale=rationale,
            category=category,
            entity_ids=list(entity_ids or []),
            finding_ids=list(finding_ids or []),
            evidence_ids=list(evidence_ids or []),
            metadata={"auto": True, "key": key, "lang": lang},
        ))

    # 1) evidence gaps ------------------------------------------------------
    for gap in evidence_gaps(case, index, min_events=min_events):
        add(
            key=f"gap|{gap['key']}",
            priority=gap["severity"],
            category=gap["category"],
            action_key=gap["action_key"],
            rationale_key=gap["rationale_key"],
            params=gap.get("params", {}),
        )

    # 2) finding-driven leads ----------------------------------------------
    threshold = float(((case.metadata or {}).get("analytics", {}) or {}).get("structuring_threshold", 10000.0))
    for finding in case.findings:
        entity_ids = list(finding.entity_ids)
        name = _entity_name(case, entity_ids[0]) if entity_ids else "the flagged entity"
        identifiers = _entity_identifiers(case, entity_ids[0]) if entity_ids else "—"
        kind = finding.kind
        common = {
            "name": name,
            "identifiers": identifiers,
            "kind": kind,
            "threshold": f"{threshold:,.0f}",
            "location": _finding_location(finding),
            "timestamp": _finding_timestamp(finding),
            "n": _counterparty_count(index, entity_ids[0]) if entity_ids else 0,
        }
        if kind == "structuring":
            add("structuring", "HIGH", "records",
                "rec.structuring.action", "rec.structuring.rationale", common,
                entity_ids, [finding.finding_id], finding.evidence_ids)
        elif kind == "colocation":
            add("colocation", "HIGH", "forensics",
                "rec.colocation.action", "rec.colocation.rationale", common,
                entity_ids, [finding.finding_id], finding.evidence_ids)
        elif kind == "chain":
            add("chain", "HIGH", "records",
                "rec.chain.action", "rec.chain.rationale", common,
                entity_ids, [finding.finding_id], finding.evidence_ids)
        elif kind == "fan_io":
            add("fan_io", "HIGH", "analysis",
                "rec.fan.action", "rec.fan.rationale", common,
                entity_ids, [finding.finding_id], finding.evidence_ids)
        elif kind in ("dormancy", "silence"):
            add("dormancy", "MEDIUM", "analysis",
                "rec.dormancy.action", "rec.dormancy.rationale", common,
                entity_ids, [finding.finding_id], finding.evidence_ids)
        else:
            add(f"corroborate|{kind}", "MEDIUM", "corroboration",
                "rec.corroborate.action", "rec.corroborate.rationale", common,
                entity_ids, [finding.finding_id], finding.evidence_ids)

    # Ensure at least one corroboration lead when findings exist but the
    # finding kinds did not already produce one.  This is a genuine next step:
    # every automated signal should be checked against an independent source.
    if case.findings and not any(r.category == "corroboration" for r in recs):
        top = case.findings[0]
        top_eid = top.entity_ids[0] if top.entity_ids else ""
        add(
            f"corroborate|top|{top.finding_id}", "MEDIUM", "corroboration",
            "rec.corroborate.action", "rec.corroborate.rationale",
            {
                "name": _entity_name(case, top_eid) if top_eid else "the flagged entity",
                "kind": top.kind,
            },
            top.entity_ids, [top.finding_id], top.evidence_ids,
        )

    # 3) high-risk entity actions ------------------------------------------
    high_risk = [
        ent for ent in case.entities.values()
        if ent.risk_score >= 0.6 or any(
            f.severity == "HIGH" and ent.entity_id in f.entity_ids
            for f in case.findings
        )
    ]
    high_risk.sort(key=lambda e: (-e.risk_score, e.entity_id))
    for ent in high_risk[:8]:
        linked = [f.finding_id for f in case.findings if ent.entity_id in f.entity_ids]
        add(f"interview|{ent.entity_id}", "HIGH", "interview",
            "rec.interview.action", "rec.interview.rationale",
            {"name": ent.display_name, "entity_id": ent.entity_id,
             "risk": "HIGH" if ent.risk_score >= 0.7 else "MEDIUM",
             "n_findings": len(linked)},
            [ent.entity_id], linked)
        if ent.identifiers:
            add(f"records|{ent.entity_id}", "MEDIUM", "records",
                "rec.records.action", "rec.records.rationale",
                {"name": ent.display_name, "identifiers": _entity_identifiers(case, ent.entity_id)},
                [ent.entity_id], linked)

    # 4) timeline gap -------------------------------------------------------
    gap = _timeline_gap(index)
    if gap:
        days, start, end = gap
        add("timeline_gap", "MEDIUM", "evidence_gap",
            "rec.timeline_gap.action", "rec.timeline_gap.rationale",
            {"start": start[:10], "end": end[:10], "days": days})

    # 5) integrity ----------------------------------------------------------
    for item in case.evidence:
        if not item.sha256:
            add(f"integrity|{item.evidence_id}", "MEDIUM", "forensics",
                "rec.integrity.action", "rec.integrity.rationale", {},
                evidence_ids=[item.evidence_id])
            break

    # 6) report review ------------------------------------------------------
    if case.findings:
        add("report_review", "LOW", "analysis",
            "rec.report.action", "rec.report.rationale", {})

    # sort + assign stable ids
    recs.sort(key=lambda r: (_PRIORITY_ORDER.get(r.priority, 3),
                             r.category, r.action))
    for i, rec in enumerate(recs, 1):
        rec.recommendation_id = f"REC{i:04d}"
    case.recommendations = existing_manual + recs
    case.touch()
    return recs


def recommendation_summary(case: Case) -> dict:
    counts = Counter(r.priority for r in case.recommendations)
    return {
        "total": len(case.recommendations),
        "by_priority": dict(counts),
        "by_category": dict(Counter(r.category for r in case.recommendations)),
    }

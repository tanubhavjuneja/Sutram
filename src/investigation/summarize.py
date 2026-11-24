"""Case, entity and timeline summarization.

Summaries are assembled from the index and the case record, not from raw model
scores.  Every numeric claim is traceable to the evidence inventory or the
event index; every finding/recommendation links back to its source objects.

The text is template-based so it can be produced offline and translated into
the supported locales without an LLM.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

import pandas as pd

from .i18n import t
from .index import EvidenceIndex
from .models import Case, QueryPlan

_SOURCE_ORDER = ["bank", "cdr", "social", "email", "chat", "document", "image", "note", "other"]


def _sources_label(sources) -> str:
    if not sources:
        return "—"
    return ", ".join(str(s) for s in sources)


def _fmt_date(value: str) -> str:
    if not value:
        return "—"
    return str(value)[:19].replace("T", " ")


def _fmt_money(value: float) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "0.00"


def evidence_gaps(case: Case, index: EvidenceIndex, min_events: int = 30) -> List[dict]:
    """Return evidence gaps as structured, recommendation-ready objects."""
    gaps: List[dict] = []
    present = {item.source_type for item in case.evidence}
    # structured sources
    for source, action_key, rationale_key in (
        ("bank", "rec.gap.bank.action", "rec.gap.bank.rationale"),
        ("cdr", "rec.gap.cdr.action", "rec.gap.cdr.rationale"),
        ("social", "rec.gap.social.action", "rec.gap.social.rationale"),
    ):
        if source not in present:
            gaps.append({
                "key": f"missing_{source}",
                "severity": "HIGH" if source in ("bank", "cdr") else "MEDIUM",
                "category": "evidence_gap",
                "action_key": action_key,
                "rationale_key": rationale_key,
                "params": {},
            })
    if not ({"email", "chat"} & present):
        gaps.append({
            "key": "missing_email_chat",
            "severity": "MEDIUM",
            "category": "evidence_gap",
            "action_key": "rec.gap.email.action",
            "rationale_key": "rec.gap.email.rationale",
            "params": {},
        })

    # undated evidence
    if not index.events.empty:
        dated = index.events["event_ts"].notna().mean()
        if dated < 0.95:
            gaps.append({
                "key": "undated_evidence",
                "severity": "MEDIUM" if dated < 0.5 else "LOW",
                "category": "evidence_gap",
                "action_key": "rec.gap.timestamps.action",
                "rationale_key": "rec.gap.timestamps.rationale",
                "params": {"dated_ratio": round(float(dated), 3)},
            })

    # thin per-entity volume
    n_entities = max(len(case.entities), 1)
    if not index.events.empty and len(index.events) / n_entities < min_events:
        gaps.append({
            "key": "thin_volume",
            "severity": "MEDIUM",
            "category": "evidence_gap",
            "action_key": "rec.gap.volume.action",
            "rationale_key": "rec.gap.volume.rationale",
            "params": {
                "events_per_entity": round(len(index.events) / n_entities, 1),
                "threshold": min_events,
            },
        })

    # unresolved counterparties
    if not index.events.empty:
        cp = index.events["counterparty_id"].replace("", pd.NA).dropna().astype(str)
        if not cp.empty:
            unresolved = [c for c in cp.unique() if c not in case.entities]
            ratio = len(unresolved) / max(len(cp.unique()), 1)
            if ratio >= 0.25:
                gaps.append({
                    "key": "unresolved_identifiers",
                    "severity": "MEDIUM" if ratio < 0.6 else "HIGH",
                    "category": "evidence_gap",
                    "action_key": "rec.gap.unresolved.action",
                    "rationale_key": "rec.gap.unresolved.rationale",
                    "params": {"unresolved_ratio": round(float(ratio), 3),
                               "unresolved": unresolved[:10]},
                })

    # duplicate / integrity note
    duplicates = [
        item for item in case.evidence
        if (item.metadata or {}).get("duplicates")
    ]
    if duplicates:
        gaps.append({
            "key": "duplicate_evidence",
            "severity": "LOW",
            "category": "evidence_gap",
            "action_key": "rec.integrity.action",
            "rationale_key": "rec.integrity.rationale",
            "params": {"duplicates": [d.evidence_id for d in duplicates]},
        })
    return gaps


def limitations(case: Case, index: EvidenceIndex, lang: str = "en") -> List[str]:
    """Human-readable limitations for the case report (localized)."""
    out: List[str] = []
    analytics = (case.metadata or {}).get("analytics", {}) or {}
    sufficiency = analytics.get("sufficiency", {}) or {}
    for model, verdict in sorted(sufficiency.items()):
        if isinstance(verdict, dict):
            status = verdict.get("status", "")
            reason = verdict.get("reason", "")
        else:
            status = str(verdict)
            reason = ""
        if status == "BLOCKED":
            out.append(t("limitations.blocked", lang, model=model,
                         reason=reason or "required evidence missing"))
        elif status == "DEGRADED":
            out.append(t("limitations.degraded", lang, model=model,
                         reason=reason or "reduced evidence coverage"))
    if not analytics:
        out.append(t("limitations.no_analytics", lang))
    if index.events.empty:
        out.append(t("limitations.no_structured", lang))
    if index.chunks:
        out.append(t("limitations.ocr", lang))
    if not out:
        out.append(t("limitations.none", lang))
    out.append(t("limitations.review", lang))
    return out


def case_summary(case: Case, index: EvidenceIndex, lang: str = "en",
                 min_events: int = 30) -> dict:
    stats = index.stats()
    sources = [s for s in _SOURCE_ORDER if s in stats.get("sources", {})]
    start = _fmt_date(stats.get("start", ""))
    end = _fmt_date(stats.get("end", ""))
    top = index.top_entities(5)
    findings = case.findings
    kinds = Counter(f.kind for f in findings)
    gaps = evidence_gaps(case, index, min_events=min_events)
    recommendations = sorted(
        case.recommendations,
        key=lambda r: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(r.priority, 3),
    )

    if stats.get("events", 0) or stats.get("chunks", 0):
        text = t(
            "summary.overview", lang,
            title=case.title,
            n_evidence=len(case.evidence),
            sources=_sources_label(sources),
            start=start,
            end=end,
            n_entities=len(case.entities),
            n_findings=len(findings),
        )
    else:
        text = t("summary.no_dates", lang,
                 n_evidence=len(case.evidence),
                 sources=_sources_label(sources))
    if top:
        text += " " + t("summary.top_entities", lang,
                        names=", ".join(f"{e['name']} ({e['risk_score']:.2f})" for e in top))
    if findings:
        kind_label = ", ".join(f"{k} ×{v}" for k, v in kinds.most_common())
        text += " " + t("summary.findings", lang, kinds=kind_label)
    if gaps:
        gap_labels = ", ".join(g["key"].replace("_", " ") for g in gaps)
        text += " " + t("summary.gaps", lang, gaps=gap_labels)
    if recommendations:
        text += " " + t("summary.next_steps", lang, action=recommendations[0].action)

    return {
        "case_id": case.case_id,
        "title": case.title,
        "text": text,
        "stats": stats,
        "sources": sources,
        "start": stats.get("start", ""),
        "end": stats.get("end", ""),
        "top_entities": top,
        "findings": {
            "total": len(findings),
            "by_kind": dict(kinds),
            "by_severity": dict(Counter(f.severity for f in findings)),
        },
        "gaps": gaps,
        "recommendations": [r.to_dict() for r in recommendations[:10]],
        "limitations": limitations(case, index, lang=lang),
    }


def entity_dossier(case: Case, index: EvidenceIndex, entity_id: str,
                   lang: str = "en") -> Optional[dict]:
    profile = index.entity_profile(entity_id)
    if not profile or (not profile.get("n_events") and not profile.get("n_counterparty_events")
                       and entity_id not in case.entities):
        return None
    risk = "HIGH" if profile["risk_score"] >= 0.7 else ("MEDIUM" if profile["risk_score"] >= 0.45 else "LOW")
    text = t(
        "summary.entity", lang,
        name=profile["name"],
        entity_id=profile["entity_id"],
        risk=risk,
        n_events=profile["n_events"],
        sources=_sources_label(profile["sources"]),
        first_seen=_fmt_date(profile["first_seen"]),
        last_seen=_fmt_date(profile["last_seen"]),
        identifiers=", ".join(profile["identifiers"]) or "—",
    )
    if profile["risk_factors"]:
        text += " " + t("answer.entity.risk_factors", lang,
                        factors="; ".join(profile["risk_factors"]))
    if profile["findings"]:
        text += " " + t("summary.entity.findings", lang,
                        titles="; ".join(f["title"] for f in profile["findings"][:5]))
    if profile["top_counterparts"]:
        text += " " + t("summary.entity.counterparts", lang,
                        names=", ".join(c["name"] for c in profile["top_counterparts"][:5]))
    return {
        **profile,
        "risk": risk,
        "text": text,
        "total_amount_fmt": _fmt_money(profile["total_amount"]),
        "avg_amount_fmt": _fmt_money(profile["avg_amount"]),
        "max_amount_fmt": _fmt_money(profile["max_amount"]),
    }


def timeline_summary(case: Case, index: EvidenceIndex, plan: QueryPlan,
                     lang: str = "en", limit: int = 50) -> dict:
    items = index.timeline(plan, limit=limit)
    if items:
        start = items[0].get("timestamp", "")
        end = items[-1].get("timestamp", "")
        text = t("summary.timeline", lang, n=len(items),
                 start=_fmt_date(start), end=_fmt_date(end))
    else:
        text = t("answer.timeline.none", lang)
    return {
        "text": text,
        "items": items,
        "n": len(items),
    }

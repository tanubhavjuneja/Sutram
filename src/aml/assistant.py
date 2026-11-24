"""Conversational forensic assistant.

The assistant is an intent router over the evidence index.  It never invents
facts: every answer is assembled from indexed events/chunks, case findings,
recommendations or the diary, and factual claims carry citations.

Session context resolves follow-ups ("who did *he* call?") by remembering the
last resolved entities and intent.  The optional LLM only polishes the draft;
the deterministic answer is always available.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .i18n import t
from .index import EvidenceIndex
from .llm import LLMClient
from .case import Answer, Case, Citation, QueryPlan
from .query import QueryParser
from .recommend import recommendation_summary
from .retrieval import HybridRetriever
from .summarize import case_summary, entity_dossier, timeline_summary

_ENTITY_INTENTS = {"entity_profile", "relationship", "correlation", "aggregate"}
_PRONOUN_RE = re.compile(
    r"\b(they|them|their|theirs|he|him|his|she|her|hers|that person|this person|"
    r"those people|these people|the suspect|the subject|उनका|उनके|उन्हें|वह|उस|"
    r"él|ella|ellos|ellas|su|sus|il|elle|ils|elles|son|sa|ses|er|sie|sein|seine|"
    r"ele|ela|eles|elas|seu|sua|seus|suas)\b",
    re.IGNORECASE,
)


@dataclass
class SessionContext:
    language: str = "en"
    last_entity_ids: List[str] = field(default_factory=list)
    last_intent: str = ""
    last_plan: Dict = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)

    def remember(self, plan: QueryPlan, answer: Answer) -> None:
        self.language = plan.language or self.language
        self.last_intent = plan.intent
        self.last_plan = plan.to_dict()
        if plan.entity_ids:
            self.last_entity_ids = list(plan.entity_ids)
        elif answer.data.get("entity_ids"):
            self.last_entity_ids = list(answer.data["entity_ids"])
        self.history.append({"question": plan.raw, "intent": plan.intent,
                             "answer": answer.answer[:500]})
        self.history = self.history[-20:]


def _clean_hits(hits: List[dict]) -> List[dict]:
    out = []
    for hit in hits:
        item = {k: v for k, v in hit.items() if not k.startswith("_")}
        out.append(item)
    return out


def _dedupe_citations(citations: List[Citation], limit: int = 8) -> List[Citation]:
    seen = set()
    out: List[Citation] = []
    for c in citations:
        key = (c.evidence_id, c.event_id, c.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= limit:
            break
    return out


def _fmt_ts(value: str) -> str:
    return (value or "").replace("T", " ")[:19] or "undated"


class ForensicAssistant:
    def __init__(self, case: Case, index: EvidenceIndex, lang: str = "en",
                 llm: Optional[LLMClient] = None, min_events: int = 30):
        self.case = case
        self.index = index
        self.lang = lang or case.language or "en"
        self.retriever = HybridRetriever(index)
        self.llm = llm or LLMClient()
        self.min_events = min_events
        self.parser = QueryParser(
            alias_to_entity=index.aliases,
            entity_names={eid: ent.names for eid, ent in index.entities.items()},
            case_start=self._case_start(),
            case_end=self._case_end(),
            default_lang=self.lang,
            known_locations=self._known_locations(),
        )

    # -- case anchors ------------------------------------------------------
    def _case_start(self):
        stats = self.index.stats()
        return stats.get("start") or None

    def _case_end(self):
        stats = self.index.stats()
        return stats.get("end") or None

    def _known_locations(self) -> List[str]:
        if self.index.events.empty:
            return []
        locs = self.index.events["location"].replace("", None).dropna().astype(str)
        return sorted(set(locs))[:200]

    # -- public ------------------------------------------------------------
    def ask(self, question: str, lang: Optional[str] = None,
            context: Optional[SessionContext] = None) -> Answer:
        context = context or SessionContext(language=self.lang)
        plan = self.parser.parse(question, lang=lang or context.language or self.lang)

        # follow-up entity resolution
        if plan.intent in _ENTITY_INTENTS and not plan.entity_ids:
            if context.last_entity_ids and (
                _PRONOUN_RE.search(question)
                or plan.intent in ("entity_profile", "correlation", "aggregate")
            ):
                plan.entity_ids = list(context.last_entity_ids)
                plan.notes.append("resolved follow-up pronoun to previous entity")
        if plan.intent == "relationship" and len(plan.entity_ids) < 2:
            if len(context.last_entity_ids) >= 2:
                plan.entity_ids = list(context.last_entity_ids[:2])
                plan.notes.append("resolved relationship to previous entity pair")

        handler = getattr(self, f"_answer_{plan.intent}", None)
        if handler is None:
            handler = self._answer_search
        answer = handler(plan, context)
        answer.query_plan = plan.to_dict()
        answer.language = plan.language
        answer.intent = plan.intent

        # optional grounded polish
        if self.llm.available and answer.citations:
            polished = self.llm.polish_answer(question, answer.answer,
                                              answer.citations, lang=plan.language)
            if polished and polished != answer.answer:
                answer.answer = polished
                answer.data["llm_polished"] = True

        context.remember(plan, answer)
        return answer

    # -- search ------------------------------------------------------------
    def _answer_search(self, plan: QueryPlan, context: SessionContext) -> Answer:
        result = self.retriever.search(plan)
        if not result.hits:
            text = t("answer.search.none", plan.language, query=plan.raw)
            return Answer(answer=text, citations=[], hits=[], confidence=0.2,
                          followups=self._followups(plan, context),
                          data={"total": 0})
        lines = [t("answer.search.header", plan.language, n=len(result.hits))]
        for i, hit in enumerate(result.hits[:10], 1):
            ts = _fmt_ts(hit.get("timestamp", ""))
            lines.append(
                f"{i}. **{hit.get('title', '')}** — {ts} · {hit.get('source', '')}  \n"
                f"   {hit.get('snippet', '')}"
            )
        if len(result.hits) > 10:
            lines.append(t("answer.search.footer", plan.language,
                           n=10, total=len(result.hits)))
        body = "\n".join(lines)
        body += self._citation_block(result.citations, plan.language)
        return Answer(
            answer=body,
            citations=result.citations,
            hits=_clean_hits(result.hits),
            confidence=self._confidence(result.hits, result.citations),
            followups=self._followups(plan, context),
            data={"total": len(result.hits), "intent": "search"},
        )

    # -- entity profile ----------------------------------------------------
    def _answer_entity_profile(self, plan: QueryPlan, context: SessionContext) -> Answer:
        if not plan.entity_ids:
            names = ", ".join(e.display_name for e in list(self.case.entities.values())[:12])
            text = t("answer.entity.not_found", plan.language,
                     term=plan.raw, names=names or "—")
            return Answer(answer=text, confidence=0.2,
                          followups=self._followups(plan, context),
                          data={"entity_ids": []})
        eid = plan.entity_ids[0]
        dossier = entity_dossier(self.case, self.index, eid, lang=plan.language)
        if not dossier:
            text = t("answer.entity.not_found", plan.language,
                     term=plan.raw, names=", ".join(e.display_name for e in self.case.entities.values()))
            return Answer(answer=text, confidence=0.2,
                          followups=self._followups(plan, context),
                          data={"entity_ids": [eid]})
        citations = [Citation.from_dict(c) for c in dossier.get("citations", [])]
        lines = [dossier["text"]]
        if dossier.get("findings"):
            lines.append(t("summary.entity.findings", plan.language,
                           titles="; ".join(f["title"] for f in dossier["findings"][:5])))
        body = "\n\n".join(lines) + self._citation_block(citations, plan.language)
        return Answer(
            answer=body,
            citations=citations,
            hits=[],
            confidence=self._confidence([], citations),
            followups=self._followups(plan, context),
            data={"entity_ids": [eid], "profile": dossier},
        )

    # -- timeline ----------------------------------------------------------
    def _answer_timeline(self, plan: QueryPlan, context: SessionContext) -> Answer:
        summary = timeline_summary(self.case, self.index, plan, lang=plan.language)
        items = summary["items"]
        if not items:
            return Answer(answer=t("answer.timeline.none", plan.language),
                          confidence=0.2, followups=self._followups(plan, context),
                          data={"items": []})
        lines = [t("answer.timeline.header", plan.language, n=len(items))]
        for i, hit in enumerate(items[:20], 1):
            lines.append(
                f"{i}. **{_fmt_ts(hit.get('timestamp', ''))}** · {hit.get('source', '')} — "
                f"{hit.get('title', '')}  \n   {hit.get('snippet', '')}"
            )
        citations = _dedupe_citations([
            self.retriever.citation_for_hit(h) for h in items
            if self.retriever.citation_for_hit(h) is not None
        ])
        body = "\n".join(lines) + self._citation_block(citations, plan.language)
        return Answer(answer=body, citations=citations, hits=_clean_hits(items),
                      confidence=self._confidence(items, citations),
                      followups=self._followups(plan, context),
                      data={"items": _clean_hits(items)})

    # -- relationship ------------------------------------------------------
    def _answer_relationship(self, plan: QueryPlan, context: SessionContext) -> Answer:
        if len(plan.entity_ids) < 2:
            known = ", ".join(e.display_name for e in list(self.case.entities.values())[:15])
            text = (t("answer.entity.not_found", plan.language, term=plan.raw, names=known)
                    + " " + "Please name two entities, e.g. 'connection between A and B'.")
            return Answer(answer=text, confidence=0.2,
                          followups=self._followups(plan, context),
                          data={"entity_ids": plan.entity_ids})
        a, b = plan.entity_ids[0], plan.entity_ids[1]
        rel = self.index.relationship(a, b)
        lines = [t("answer.relationship.header", plan.language,
                   a=rel["a_name"], b=rel["b_name"])]
        citations: List[Citation] = []
        if rel["direct"]:
            d = rel["direct"]
            lines.append(t("answer.relationship.direct", plan.language, n=d["count"]))
            lines.append(
                f"- sources: {', '.join(d['sources']) or '—'}; "
                f"amount: {d['amount']:,.2f}; first: {_fmt_ts(d['first'])}; last: {_fmt_ts(d['last'])}"
            )
            # direct edge events as citations
            if not self.index.events.empty:
                sub = self.index.events[
                    ((self.index.events["entity_id"] == a) & (self.index.events["counterparty_id"] == b))
                    | ((self.index.events["entity_id"] == b) & (self.index.events["counterparty_id"] == a))
                ].sort_values("event_ts", ascending=False).head(4)
                citations.extend([self.index.citation_for_event(row) for _, row in sub.iterrows()])
        elif rel["shared_chunks"]:
            lines.append(t("answer.relationship.shared", plan.language,
                           n=len(rel["shared_chunks"])))
        else:
            lines.append(t("answer.relationship.none", plan.language))
        if rel["path"] and len(rel["path"]) > 2:
            lines.append(t("answer.relationship.path", plan.language,
                           path=" → ".join(p["name"] for p in rel["path"])))
        if rel["common"]:
            lines.append(t("answer.relationship.common", plan.language,
                           names=", ".join(c["name"] for c in rel["common"][:8])))
        if rel["shared_chunks"]:
            for chunk in rel["shared_chunks"][:3]:
                lines.append(
                    f"- {chunk.get('file') or chunk['evidence_id']} · "
                    f"{_fmt_ts(chunk.get('timestamp', ''))} — {chunk['snippet']}"
                )
                citation = chunk.get("citation")
                if citation:
                    citations.append(Citation.from_dict(citation))
        citations = _dedupe_citations(citations)
        body = "\n".join(lines) + self._citation_block(citations, plan.language)
        return Answer(answer=body, citations=citations, hits=[],
                      confidence=self._confidence([], citations),
                      followups=self._followups(plan, context),
                      data={"relationship": rel, "entity_ids": [a, b]})

    # -- correlation -------------------------------------------------------
    def _answer_correlation(self, plan: QueryPlan, context: SessionContext) -> Answer:
        result = self.retriever.correlate_calls(plan)
        if not result["pairs"]:
            text = t("answer.correlation.none", plan.language,
                     tol=result["tol_minutes"])
            return Answer(answer=text, confidence=0.25,
                          followups=self._followups(plan, context),
                          data={"pairs": [], "n": 0})
        lines = [t("answer.correlation.header", plan.language,
                   n=len(result["pairs"]), tol=result["tol_minutes"])]
        for i, pair in enumerate(result["pairs"][:10], 1):
            comm = pair["communication"]
            transfer = pair["transfer"]
            lines.append(
                f"{i}. **{pair['entity_name']}** → {pair['counterparty_name']}: "
                f"{comm['source']} at {_fmt_ts(comm['timestamp'])} then transfer "
                f"{transfer['amount']:,.2f} at {_fmt_ts(transfer['timestamp'])} "
                f"({pair['delta_minutes']:.1f} min apart)"
            )
        body = "\n".join(lines) + self._citation_block(result["citations"], plan.language)
        return Answer(answer=body, citations=result["citations"], hits=[],
                      confidence=self._confidence([], result["citations"]),
                      followups=self._followups(plan, context),
                      data={"pairs": result["pairs"], "n": result["n"],
                            "tol_minutes": result["tol_minutes"],
                            "entity_ids": plan.entity_ids})

    # -- aggregate ---------------------------------------------------------
    def _answer_aggregate(self, plan: QueryPlan, context: SessionContext) -> Answer:
        agg = self.index.aggregate(plan)
        if agg["count"] == 0:
            text = t("answer.search.none", plan.language, query=plan.raw)
            return Answer(answer=text, confidence=0.2,
                          followups=self._followups(plan, context),
                          data={"aggregate": agg})
        lines = [t("answer.aggregate.header", plan.language, n=agg["count"])]
        lines.append(t("answer.aggregate.total", plan.language, total=f"{agg['total']:,.2f}"))
        lines.append(t("answer.aggregate.count", plan.language, count=agg["count"]))
        lines.append(t("answer.aggregate.avg", plan.language, avg=f"{agg['avg']:,.2f}"))
        lines.append(t("answer.aggregate.largest", plan.language, largest=f"{agg['max']:,.2f}"))
        if agg["by_source"]:
            lines.append(f"- by source: {agg['by_source']}")
        if agg["by_direction"]:
            lines.append(f"- by direction: {agg['by_direction']}")
        citations = _dedupe_citations([
            self.retriever.citation_for_hit(h) for h in agg.get("hits", [])
            if self.retriever.citation_for_hit(h) is not None
        ])
        body = "\n".join(lines) + self._citation_block(citations, plan.language)
        return Answer(answer=body, citations=citations, hits=_clean_hits(agg.get("hits", [])),
                      confidence=self._confidence(agg.get("hits", []), citations),
                      followups=self._followups(plan, context),
                      data={"aggregate": agg, "entity_ids": plan.entity_ids})

    # -- anomaly -----------------------------------------------------------
    def _answer_anomaly(self, plan: QueryPlan, context: SessionContext) -> Answer:
        findings = sorted(
            self.case.findings,
            key=lambda f: ({"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(f.severity, 3),
                           -f.confidence),
        )
        if not findings:
            return Answer(answer=t("answer.anomaly.none", plan.language),
                          confidence=0.2, followups=self._followups(plan, context),
                          data={"findings": []})
        lines = [t("answer.anomaly.header", plan.language, n=len(findings))]
        citations: List[Citation] = []
        for i, finding in enumerate(findings[:10], 1):
            lines.append(
                f"{i}. **{finding.title}** — {finding.severity} · {finding.kind} · "
                f"confidence {finding.confidence:.2f}  \n   {finding.description}"
            )
            citations.extend(finding.citations)
        citations = _dedupe_citations(citations)
        body = "\n".join(lines) + self._citation_block(citations, plan.language)
        return Answer(answer=body, citations=citations, hits=[],
                      confidence=self._confidence([], citations),
                      followups=self._followups(plan, context),
                      data={"findings": [f.to_dict() for f in findings[:20]],
                            "entity_ids": plan.entity_ids})

    # -- summary -----------------------------------------------------------
    def _answer_summarize(self, plan: QueryPlan, context: SessionContext) -> Answer:
        summary = case_summary(self.case, self.index, lang=plan.language,
                               min_events=self.min_events)
        citations: List[Citation] = []
        for entity in summary.get("top_entities", [])[:3]:
            citations.extend(self.retriever.entity_citations(entity["entity_id"], limit=2))
        for finding in self.case.findings[:3]:
            citations.extend(finding.citations)
        citations = _dedupe_citations(citations)
        body = (t("answer.summarize.header", plan.language, title=self.case.title)
                + "\n\n" + summary["text"])
        if summary.get("limitations"):
            body += "\n\n" + t("summary.limitations", plan.language,
                               items="; ".join(summary["limitations"][:4]))
        body += self._citation_block(citations, plan.language)
        return Answer(answer=body, citations=citations, hits=[],
                      confidence=self._confidence([], citations),
                      followups=self._followups(plan, context),
                      data={"summary": summary, "entity_ids": plan.entity_ids})

    # -- recommendations ---------------------------------------------------
    def _answer_recommend(self, plan: QueryPlan, context: SessionContext) -> Answer:
        recs = sorted(
            self.case.recommendations,
            key=lambda r: ({"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(r.priority, 3),
                           r.recommendation_id),
        )
        if not recs:
            return Answer(answer=t("answer.recommend.none", plan.language),
                          confidence=0.2, followups=self._followups(plan, context),
                          data={"recommendations": []})
        lines = [t("answer.recommend.header", plan.language)]
        for i, rec in enumerate(recs[:12], 1):
            lines.append(
                f"{i}. **[{rec.priority}] {rec.action}**  \n"
                f"   {rec.rationale}  \n"
                f"   _category: {rec.category} · id: {rec.recommendation_id}_"
            )
        citations = _dedupe_citations([
            c for rec in recs[:6] for c in self._citations_for_recommendation(rec)
        ])
        body = "\n".join(lines) + self._citation_block(citations, plan.language)
        return Answer(answer=body, citations=citations, hits=[],
                      confidence=0.85 if citations else 0.65,
                      followups=self._followups(plan, context),
                      data={"recommendations": [r.to_dict() for r in recs],
                            "summary": recommendation_summary(self.case),
                            "entity_ids": plan.entity_ids})

    def _citations_for_recommendation(self, rec) -> List[Citation]:
        out: List[Citation] = []
        for finding in self.case.findings:
            if finding.finding_id in rec.finding_ids:
                out.extend(finding.citations)
        for eid in rec.entity_ids[:1]:
            out.extend(self.retriever.entity_citations(eid, limit=2))
        return out

    # -- diary -------------------------------------------------------------
    def _answer_diary(self, plan: QueryPlan, context: SessionContext) -> Answer:
        from .diary import verify_chain
        ok, errors = verify_chain(self.case)
        status = t("diary.verified", plan.language) if ok else t("diary.tampered", plan.language)
        lines = [t("answer.diary.header", plan.language,
                   n=len(self.case.diary), status=status)]
        for entry in self.case.diary[-20:]:
            lines.append(
                f"- **{_fmt_ts(entry.timestamp)}** · `{entry.entry_type}` · "
                f"{entry.actor} — {entry.text}"
            )
        return Answer(answer="\n".join(lines), citations=[], hits=[],
                      confidence=0.95 if ok else 0.4,
                      followups=self._followups(plan, context),
                      data={"diary": [d.to_dict() for d in self.case.diary],
                            "chain_ok": ok, "chain_errors": errors})

    # -- evidence inventory ------------------------------------------------
    def _answer_evidence_inventory(self, plan: QueryPlan, context: SessionContext) -> Answer:
        items = self.index.evidence_summary()
        if not items:
            return Answer(answer=t("answer.search.none", plan.language, query=plan.raw),
                          confidence=0.2, followups=self._followups(plan, context),
                          data={"evidence": []})
        lines = [t("answer.evidence_inventory.header", plan.language, n=len(items))]
        for item in items:
            lines.append(
                f"- **{item['evidence_id']}** · {item['filename']} · {item['source_type']} · "
                f"{item['n_rows']} row(s) / {item['n_chunks']} chunk(s) · "
                f"`{item['sha256'][:16]}…`"
            )
        return Answer(answer="\n".join(lines), citations=[], hits=[],
                      confidence=0.95,
                      followups=self._followups(plan, context),
                      data={"evidence": items})

    # -- entity list -------------------------------------------------------
    def _answer_entity_list(self, plan: QueryPlan, context: SessionContext) -> Answer:
        entities = sorted(
            self.case.entities.values(),
            key=lambda e: (-e.risk_score, -e.n_events, e.display_name),
        )
        if not entities:
            return Answer(answer=t("answer.search.none", plan.language, query=plan.raw),
                          confidence=0.2, followups=self._followups(plan, context),
                          data={"entities": []})
        lines = [t("answer.entity_list.header", plan.language, n=len(entities))]
        for ent in entities[:30]:
            risk = "HIGH" if ent.risk_score >= 0.7 else ("MEDIUM" if ent.risk_score >= 0.45 else "LOW")
            lines.append(
                f"- **{ent.display_name}** ({ent.entity_id}) — {risk} risk · "
                f"{ent.n_events} event(s) · {', '.join(ent.sources) or '—'} · "
                f"ids: {', '.join(ent.identifiers[:4]) or '—'}"
            )
        return Answer(answer="\n".join(lines), citations=[], hits=[],
                      confidence=0.95,
                      followups=self._followups(plan, context),
                      data={"entities": [e.to_dict() for e in entities]} )

    # -- report ------------------------------------------------------------
    def _answer_report(self, plan: QueryPlan, context: SessionContext) -> Answer:
        from .case_report import generate_case_report
        out_dir = Path(self.case.root or ".") / "reports"
        try:
            files = generate_case_report(self.case, self.index, lang=plan.language,
                                         out_dir=out_dir, formats=("md", "html", "json"))
        except Exception as exc:
            return Answer(answer=f"Report generation failed: {exc}",
                          confidence=0.1, followups=self._followups(plan, context),
                          data={"error": str(exc)})
        path = files.get("md") or files.get("html") or ""
        return Answer(answer=t("answer.report.header", plan.language, path=path),
                      citations=[], hits=[], confidence=0.9,
                      followups=self._followups(plan, context),
                      data={"report_files": files})

    # -- help --------------------------------------------------------------
    def _answer_help(self, plan: QueryPlan, context: SessionContext) -> Answer:
        examples = t("query.examples", plan.language)
        if not isinstance(examples, list):
            examples = []
        lines = [t("answer.help", plan.language)]
        if examples:
            lines.append("")
            lines.append("**" + t("chat.suggestions", plan.language) + "**")
            lines.extend(f"- {q}" for q in examples[:6])
        return Answer(answer="\n".join(lines), citations=[], hits=[],
                      confidence=0.9,
                      followups=examples[:3],
                      data={"examples": examples})

    # -- formatting helpers ------------------------------------------------
    def _citation_block(self, citations: List[Citation], lang: str) -> str:
        citations = _dedupe_citations(citations, limit=8)
        if not citations:
            return ""
        lines = ["", "**" + t("common.citations", lang) + "**"]
        for i, c in enumerate(citations, 1):
            lines.append(t("answer.citation_line", lang,
                           label=f"C{i}",
                           file=c.file or c.evidence_id,
                           timestamp=_fmt_ts(c.timestamp),
                           snippet=c.snippet))
        return "\n".join(lines)

    def _confidence(self, hits: List[dict], citations: List[Citation]) -> float:
        if not citations:
            return 0.25
        avg = sum(float(h.get("score", 0.0)) for h in hits) / max(len(hits), 1) if hits else 0.5
        return round(min(0.95, 0.45 + 0.08 * len(citations) + 0.25 * avg), 2)

    def _followups(self, plan: QueryPlan, context: SessionContext) -> List[str]:
        examples = t("query.examples", plan.language)
        if not isinstance(examples, list):
            examples = []
        # For non-English sessions, prefer the localized example questions
        # rather than English follow-up templates.
        if plan.language != "en" and examples:
            return examples[:3]
        if plan.intent == "entity_profile" and plan.entity_ids:
            name = self.index.entity_label(plan.entity_ids[0])
            return [
                f"Show transfers involving {name}.",
                f"Who are {name}'s top counterparts?",
                "What should I investigate next?",
            ]
        if plan.intent == "search" and plan.entity_ids:
            name = self.index.entity_label(plan.entity_ids[0])
            return [
                f"Who did {name} call right before a transfer?",
                f"What is {name}'s risk profile?",
                "Summarize the case.",
            ]
        if plan.intent == "anomaly":
            return ["What should I investigate next?",
                    "Show the evidence behind the top finding.",
                    "Draft a case report."]
        if plan.intent == "summarize":
            return ["What should I investigate next?",
                    "Show the case diary.",
                    "Draft a case report."]
        if plan.intent == "recommend":
            return ["Show the case diary.", "Draft a case report.",
                    "What looks suspicious?"]
        if plan.intent == "diary":
            return ["Draft a case report.", "What should I investigate next?",
                    "Summarize the case."]
        return examples[:3]

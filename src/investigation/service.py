"""Investigation service — the orchestrator behind the CLI, web API and UI.

Responsibilities:

* create/load/save cases and evidence
* ingest structured + unstructured evidence and rebuild the index
* run the verified ``aml`` analytics core and convert its output into case
  findings / entity risk scores / recommendations
* answer questions through the conversational assistant
* generate summaries, diaries and draft case reports
* keep the case diary and audit log current

The service is the only place that mutates a case; the assistant and report
generators are read-only over the objects they receive.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yaml

from .assistant import ForensicAssistant, SessionContext
from .case_report import generate_case_report
from .case_store import CaseStore
from .diary import append_entry, verify_chain
from .evidence import IngestResult, ingest_paths
from .i18n import SUPPORTED_LANGUAGES, language_name, normalize_language, t
from .index import EvidenceIndex
from .llm import LLMClient, LLMConfig
from .models import Answer, Case, Citation, Entity, Finding, QueryPlan, Recommendation, utc_now
from .recommend import generate_recommendations, recommendation_summary
from .summarize import case_summary, entity_dossier, evidence_gaps, timeline_summary


@dataclass
class InvestigationConfig:
    case_root: str = "data/cases"
    language: str = "en"
    min_events_per_entity: int = 30
    tol_minutes: int = 25
    structuring_threshold: float = 10000.0
    top_k: int = 10
    max_findings: int = 25
    llm_provider: str = "none"
    llm_model: str = ""
    llm_base_url: str = ""
    llm_timeout: int = 30

    @classmethod
    def from_yaml(cls, path: str | Path) -> "InvestigationConfig":
        p = Path(path)
        cfg = cls()
        if not p.exists():
            return cfg
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            return cfg
        section = raw.get("investigation", raw) or {}
        for key, value in section.items():
            if key in cfg.__dataclass_fields__:
                setattr(cfg, key, value)
        return cfg

    def llm_config(self) -> LLMConfig:
        import os
        provider = (self.llm_provider or os.getenv("INVESTIGATION_LLM_PROVIDER") or "none").lower()
        if provider == "none":
            return LLMConfig.from_env()
        return LLMConfig(
            provider=provider,
            model=self.llm_model or os.getenv("INVESTIGATION_LLM_MODEL", ""),
            base_url=self.llm_base_url or os.getenv("INVESTIGATION_LLM_BASE_URL", ""),
            api_key=os.getenv("INVESTIGATION_LLM_API_KEY", "") or os.getenv("OPENAI_API_KEY", ""),
            timeout=int(self.llm_timeout or 30),
            enabled=True,
        )


def _infer_finding_kind(text: str) -> str:
    """Infer a typology from the headline (empty string when no match)."""
    low = (text or "").lower()
    if "reporting limit" in low or "just under" in low or "structur" in low:
        return "structuring"
    if "same person" in low or ("within" in low and ("call" in low or "post" in low)):
        return "colocation"
    if "layering" in low or "multi-hop" in low or "chain" in low:
        return "chain"
    if "funnel" in low or "collect-then-disperse" in low or "different people" in low:
        return "fan_io"
    if "large single amount" in low and ("quiet" in low or "silence" in low):
        return "silence"
    if "went quiet" in low or "dormant" in low or "re-activated" in low:
        return "dormancy"
    return ""


def _risk_from_score(score: float) -> str:
    if score >= 0.7:
        return "HIGH"
    if score >= 0.45:
        return "MEDIUM"
    return "LOW"


class InvestigationService:
    def __init__(self, case_root: str = "data/cases",
                 config: Optional[InvestigationConfig] = None,
                 lang: Optional[str] = None):
        self.config = config or InvestigationConfig(case_root=case_root)
        if case_root and case_root != self.config.case_root:
            self.config.case_root = case_root
        self.store = CaseStore(self.config.case_root)
        self.lang = normalize_language(lang or self.config.language)
        self.llm = LLMClient(self.config.llm_config())
        self._index_cache: Dict[str, Tuple[str, EvidenceIndex]] = {}
        self._sessions: Dict[Tuple[str, str], SessionContext] = {}

    # -- case lifecycle ----------------------------------------------------
    def create_case(self, title: str, description: str = "",
                    language: Optional[str] = None) -> Case:
        case = self.store.create_case(
            title, description=description,
            language=normalize_language(language or self.lang),
        )
        return case

    def load_case(self, case_id: str) -> Case:
        return self.store.load_case(case_id)

    def list_cases(self) -> List[dict]:
        return self.store.list_cases()

    def save_case(self, case: Case) -> None:
        self.store.save_case(case)

    # -- ingestion ---------------------------------------------------------
    def ingest(self, case_id: str, paths: Sequence[str], custodian: str = "",
               notes: str = "") -> dict:
        case = self.load_case(case_id)
        result = ingest_paths(
            case, self.store, paths, custodian=custodian, notes=notes,
            min_events_per_entity=self.config.min_events_per_entity,
        )
        files = ", ".join(item.filename for item in case.evidence[-len(result.added_evidence_ids):]) \
            if result.added_evidence_ids else "—"
        text = t("diary.ingest", case.language,
                 n=len(result.added_evidence_ids), files=files)
        self.store.append_diary(case, "ingest", text, actor="system",
                                evidence_ids=result.added_evidence_ids)
        self._index_cache.pop(case.case_id, None)
        return {
            "case_id": case.case_id,
            "added": result.added_evidence_ids,
            "events": int(len(result.events)),
            "chunks": len(result.chunks),
            "entities": len(result.entities),
            "warnings": result.warnings,
            "per_evidence": result.per_evidence,
            "stats": case.stats(),
        }

    # -- index -------------------------------------------------------------
    def build_index(self, case: Case, force: bool = False) -> EvidenceIndex:
        key = case.case_id
        cached = self._index_cache.get(key)
        # diary updates change updated_at but not the evidence index; use an
        # explicit index_version so a chat turn does not rebuild the index
        version = (case.metadata or {}).get("index_version") or case.updated_at
        if cached and not force and cached[0] == version:
            return cached[1]
        events = self.store.load_events(case)
        chunks = self.store.load_chunks(case)
        entities = self.store.load_entities(case)
        aliases = self.store.load_aliases(case)
        # prefer the in-memory case entities (they include newly ingested items)
        if case.entities:
            entities = {**entities, **case.entities}
        index = EvidenceIndex(case, events=events, chunks=chunks,
                              entities=entities, aliases=aliases)
        self._index_cache[key] = (version, index)
        return index

    # -- analytics ---------------------------------------------------------
    def analyze(self, case_id: str, run_report: bool = True) -> dict:
        case = self.load_case(case_id)
        structured = [
            item for item in case.evidence
            if item.source_type in ("bank", "cdr", "social")
        ]
        if not structured:
            return {
                "case_id": case.case_id,
                "ok": False,
                "error": "No structured bank/CDR/social evidence is indexed. "
                         "Add a CSV/JSON/JSONL export to run the anomaly models.",
                "findings": len(case.findings),
            }
        paths = [str(self.store.resolve_evidence_path(case, item)) for item in structured]

        # lazy import keeps the CLI startup fast and avoids torch when unused
        from aml.config import PipelineConfig
        from aml.pipeline import Pipeline
        from aml.insights import build_insights
        from aml.report import generate_report

        aml_cfg = PipelineConfig()
        aml_cfg.top_k = self.config.top_k
        aml_cfg.min_events_per_entity = self.config.min_events_per_entity
        aml_cfg.tol_minutes = self.config.tol_minutes
        aml_cfg.structuring_threshold = self.config.structuring_threshold
        pipe = Pipeline(
            min_events_per_entity=self.config.min_events_per_entity,
            tol_minutes=self.config.tol_minutes,
            structuring_threshold=self.config.structuring_threshold,
        )
        res = pipe.run(paths, with_models=True)
        index = self.build_index(case, force=True)

        # map pipeline entity ids onto case entity ids
        id_map = self._map_pipeline_entities(case, res)
        ranking_score = {
            id_map.get(r["entity_id"], r["entity_id"]): float(r["score"])
            for r in res.rankings
        }
        models_by_entity: Dict[str, List[str]] = {}
        for r in res.rankings:
            eid = id_map.get(r["entity_id"], r["entity_id"])
            models_by_entity[eid] = list(r.get("models_fired", []))

        # convert insights into findings
        insights = build_insights(
            res, top_k=self.config.top_k,
            tol_minutes=self.config.tol_minutes,
            structuring_threshold=self.config.structuring_threshold,
        )
        auto_findings: List[Finding] = []
        for insight in insights:
            peid = insight["entity_id"]
            eid = id_map.get(peid, peid)
            headline = insight.get("headline", "").replace("**", "")
            description = " · ".join(e.replace("**", "") for e in insight.get("evidence", []))
            kind = (_infer_finding_kind(headline)
                    or _infer_finding_kind(description)
                    or "anomaly")
            score = ranking_score.get(eid, 0.0)
            citations = self._citations_for_finding(index, eid, kind)
            auto_findings.append(Finding(
                finding_id="",
                kind=kind,
                title=headline or f"Anomalous activity for {index.entity_label(eid)}",
                description=description or "Activity profile differs from the peer population.",
                severity=insight.get("risk", _risk_from_score(score)),
                confidence=round(min(0.99, max(score, 0.35)), 3),
                entity_ids=[eid],
                evidence_ids=sorted({c.evidence_id for c in citations if c.evidence_id}),
                citations=citations,
                detected_by=["insights"] + models_by_entity.get(eid, []),
                metadata={"auto": True, "pipeline_entity_id": peid,
                          "headline": headline, "insight_rank": insight.get("rank")},
            ))

        # model-signal findings for top ranked entities without an insight
        seen_entities = {f.entity_ids[0] for f in auto_findings if f.entity_ids}
        for r in res.rankings[:10]:
            eid = id_map.get(r["entity_id"], r["entity_id"])
            if eid in seen_entities:
                continue
            explanation = res.explanations.get(r["entity_id"], "")
            if not explanation:
                continue
            citations = self._citations_for_finding(index, eid, "anomaly")
            auto_findings.append(Finding(
                finding_id="",
                kind="model_signal",
                title=f"Multi-model anomaly signal for {index.entity_label(eid)}",
                description=explanation,
                severity=_risk_from_score(float(r["score"])),
                confidence=round(float(r["score"]), 3),
                entity_ids=[eid],
                evidence_ids=sorted({c.evidence_id for c in citations if c.evidence_id}),
                citations=citations,
                detected_by=list(r.get("models_fired", [])),
                metadata={"auto": True, "pipeline_entity_id": r["entity_id"]},
            ))

        auto_findings.sort(key=lambda f: (-f.confidence, f.finding_id))
        for i, finding in enumerate(auto_findings[: self.config.max_findings], 1):
            finding.finding_id = f"F{i:04d}"
        auto_findings = auto_findings[: self.config.max_findings]

        # replace previous automatic findings, keep manual ones
        case.findings = [f for f in case.findings if not (f.metadata or {}).get("auto")] + auto_findings

        # update entity risk scores / factors
        risk_scores: Dict[str, float] = {}
        for eid, ent in case.entities.items():
            linked = [f for f in case.findings if eid in f.entity_ids]
            score = max([ranking_score.get(eid, 0.0)] + [f.confidence for f in linked] + [0.0])
            ent.risk_score = round(float(score), 3)
            ent.risk_factors = sorted({f"{f.kind}: {f.title}" for f in linked})[:10]
            risk_scores[eid] = ent.risk_score
        case.metadata["risk_scores"] = risk_scores

        # analytics report (existing polished anomaly brief)
        report_files: Dict[str, str] = {}
        if run_report:
            try:
                report_files = generate_report(res, aml_cfg, self.store.analytics_dir(case))
            except Exception as exc:
                case.metadata.setdefault("warnings", []).append(f"analytics report failed: {exc}")

        case.metadata["analytics"] = {
            "generated_at": utc_now(),
            "n_models": len(res.model_scores),
            "n_events": int(len(res.unified)),
            "n_entities": int(res.unified["entity_id"].nunique()) if not res.unified.empty else 0,
            "sufficiency": {
                k: {"status": v.status, "reason": v.reason}
                for k, v in res.sufficiency.items()
            },
            "rankings": res.rankings[:20],
            "warnings": list(res.warnings),
            "report_files": report_files,
            "structuring_threshold": self.config.structuring_threshold,
            "tol_minutes": self.config.tol_minutes,
            "min_events_per_entity": self.config.min_events_per_entity,
            "insights": insights,
        }
        case.touch()

        # recommendations depend on findings + gaps
        generate_recommendations(case, index, lang=case.language,
                                 min_events=self.config.min_events_per_entity)

        case.metadata["index_version"] = utc_now()
        top = case.findings[0].title if case.findings else "—"
        self.store.append_diary(
            case, "analysis",
            t("diary.analysis", case.language,
              models=len(res.model_scores), findings=len(case.findings), top=top),
            actor="system",
        )
        self.store.save_case(case)
        self._index_cache.pop(case.case_id, None)
        return {
            "case_id": case.case_id,
            "ok": True,
            "models": list(res.model_scores.keys()),
            "findings": len(case.findings),
            "recommendations": len(case.recommendations),
            "entities": len(case.entities),
            "report_files": report_files,
            "sufficiency": {k: v.status for k, v in res.sufficiency.items()},
            "top_rankings": res.rankings[:10],
            "warnings": list(res.warnings),
        }

    def _map_pipeline_entities(self, case: Case, res) -> Dict[str, str]:
        """Map pipeline entity ids to case entity ids by identifier/name."""
        mapping: Dict[str, str] = {}
        em = res.entity_map
        if em is None:
            return mapping
        reverse: Dict[str, List[str]] = {}
        for ident, eid in (getattr(em, "entity_id_by_identifier", {}) or {}).items():
            reverse.setdefault(eid, []).append(str(ident))
        by_ident = {ident: eid for eid, ent in case.entities.items() for ident in ent.identifiers}
        by_name = {name.strip().lower(): eid for eid, ent in case.entities.items() for name in ent.names}
        next_num = max([int(re.sub(r"\D", "", eid) or 0) for eid in case.entities] or [0]) + 1
        for peid in reverse:
            target = None
            for ident in reverse[peid]:
                if ident in by_ident:
                    target = by_ident[ident]
                    break
            if target is None:
                for name in (getattr(em, "entity_names", {}) or {}).get(peid, set()):
                    target = by_name.get(str(name).strip().lower())
                    if target:
                        break
            if target is None:
                target = peid
                if target in case.entities:
                    target = f"E{next_num:04d}"
                    next_num += 1
                case.entities[target] = Entity(
                    entity_id=target,
                    names=sorted((getattr(em, "entity_names", {}) or {}).get(peid, set())),
                    identifiers=sorted(set(reverse[peid])),
                    sources=[],
                    n_events=0,
                    metadata={"from_analytics": True},
                )
                for ident in reverse[peid]:
                    by_ident[ident] = target
            mapping[peid] = target
        return mapping

    def _citations_for_finding(self, index: EvidenceIndex, entity_id: str,
                               kind: str, limit: int = 4) -> List[Citation]:
        if index.events.empty:
            return []
        sub = index.events[
            (index.events["entity_id"] == entity_id)
            | (index.events["counterparty_id"] == entity_id)
        ].copy()
        if sub.empty:
            return []
        if kind == "structuring":
            amt = sub["event_amount"]
            band = sub[(amt >= 0.85 * self.config.structuring_threshold)
                       & (amt < self.config.structuring_threshold)]
            if not band.empty:
                sub = band
        elif kind == "colocation":
            comm = sub[sub["source"].isin(("cdr", "social"))]
            if not comm.empty:
                sub = comm
        elif kind == "chain":
            bank = sub[sub["source"] == "bank"]
            if not bank.empty:
                sub = bank
        sub = sub.assign(sort_amount=sub["event_amount"].fillna(0)).sort_values(
            ["sort_amount", "event_ts"], ascending=False
        ).head(limit)
        return [index.citation_for_event(row) for _, row in sub.iterrows()]

    # -- conversation ------------------------------------------------------
    def _maybe_analyze(self, case: Case, plan: QueryPlan) -> bool:
        if plan.intent not in ("anomaly", "summarize", "recommend", "report"):
            return False
        if case.findings:
            return False
        if not any(item.source_type in ("bank", "cdr", "social") for item in case.evidence):
            return False
        result = self.analyze(case.case_id)
        return bool(result.get("ok"))

    def _ensure_recommendations(self, case: Case, index: EvidenceIndex,
                                lang: Optional[str] = None) -> Case:
        target = normalize_language(lang or case.language)
        auto = [r for r in case.recommendations if (r.metadata or {}).get("auto")]
        auto_langs = {(r.metadata or {}).get("lang") for r in auto}
        if auto and target in auto_langs:
            return case
        generate_recommendations(
            case, index, lang=target,
            min_events=self.config.min_events_per_entity,
        )
        self.store.save_case(case)
        return case

    def ask(self, case_id: str, question: str, lang: Optional[str] = None,
            session_id: str = "default") -> Answer:
        case = self.load_case(case_id)
        index = self.build_index(case)
        assistant = ForensicAssistant(case, index, lang=lang or case.language,
                                      llm=self.llm,
                                      min_events=self.config.min_events_per_entity)
        plan = assistant.parser.parse(question, lang=lang or case.language)
        if self._maybe_analyze(case, plan):
            case = self.load_case(case_id)
            index = self.build_index(case, force=True)
            assistant = ForensicAssistant(case, index, lang=lang or case.language,
                                          llm=self.llm,
                                          min_events=self.config.min_events_per_entity)
        if plan.intent in ("recommend", "report", "summarize"):
            target_lang = normalize_language(lang or case.language)
            auto_langs = {
                (r.metadata or {}).get("lang")
                for r in case.recommendations
                if (r.metadata or {}).get("auto")
            }
            if not case.recommendations or target_lang not in auto_langs:
                case = self._ensure_recommendations(case, index, lang=target_lang)
                index = self.build_index(case, force=True)
                assistant = ForensicAssistant(case, index, lang=lang or case.language,
                                              llm=self.llm,
                                              min_events=self.config.min_events_per_entity)
        key = (case.case_id, session_id or "default")
        context = self._sessions.get(key) or SessionContext(language=case.language)
        answer = assistant.ask(question, lang=lang or case.language, context=context)
        self._sessions[key] = context
        # diary: query + result count
        self.store.append_diary(
            case, "query",
            t("diary.query", case.language, language=answer.language, intent=answer.intent,
              query=question, n=len(answer.citations) or len(answer.hits)),
            actor="analyst",
        )
        if answer.intent == "report" and answer.data.get("report_files"):
            files = answer.data["report_files"]
            self.store.append_diary(
                case, "export",
                t("diary.report", case.language,
                  path=files.get("md") or files.get("html") or ""),
                actor="analyst",
            )
        return answer

    # -- summaries / views -------------------------------------------------
    def summarize(self, case_id: str, lang: Optional[str] = None) -> dict:
        case = self.load_case(case_id)
        index = self.build_index(case)
        if not case.findings and any(i.source_type in ("bank", "cdr", "social") for i in case.evidence):
            result = self.analyze(case_id)
            if result.get("ok"):
                case = self.load_case(case_id)
                index = self.build_index(case, force=True)
        case = self._ensure_recommendations(case, index, lang=lang or case.language)
        return case_summary(case, index, lang=normalize_language(lang or case.language),
                            min_events=self.config.min_events_per_entity)

    def entity_profile(self, case_id: str, entity_id: str,
                       lang: Optional[str] = None) -> Optional[dict]:
        case = self.load_case(case_id)
        index = self.build_index(case)
        return entity_dossier(case, index, entity_id,
                              lang=normalize_language(lang or case.language))

    def timeline(self, case_id: str, query: str = "", lang: Optional[str] = None,
                 limit: int = 100) -> dict:
        case = self.load_case(case_id)
        index = self.build_index(case)
        parser = self._parser_for(case, index, lang)
        plan = parser.parse(query or "", lang=lang or case.language)
        plan.limit = limit
        return timeline_summary(case, index, plan,
                                lang=normalize_language(lang or case.language),
                                limit=limit)

    def findings(self, case_id: str) -> List[dict]:
        case = self.load_case(case_id)
        return [f.to_dict() for f in sorted(
            case.findings,
            key=lambda f: ({"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(f.severity, 3), -f.confidence),
        )]

    def recommend(self, case_id: str, lang: Optional[str] = None) -> dict:
        case = self.load_case(case_id)
        index = self.build_index(case)
        if not case.findings and any(i.source_type in ("bank", "cdr", "social") for i in case.evidence):
            result = self.analyze(case_id)
            if result.get("ok"):
                case = self.load_case(case_id)
                index = self.build_index(case, force=True)
        recs = generate_recommendations(case, index, lang=normalize_language(lang or case.language),
                                        min_events=self.config.min_events_per_entity)
        self.store.save_case(case)
        if recs:
            self.store.append_diary(
                case, "recommendation",
                t("diary.recommendation", case.language,
                  id=recs[0].recommendation_id, action=recs[0].action,
                  priority=recs[0].priority),
                actor="system",
            )
        return {
            "case_id": case.case_id,
            "recommendations": [r.to_dict() for r in recs],
            "summary": recommendation_summary(case),
        }

    def diary(self, case_id: str, lang: Optional[str] = None) -> dict:
        case = self.load_case(case_id)
        ok, errors = verify_chain(case)
        return {
            "case_id": case.case_id,
            "entries": [d.to_dict() for d in case.diary],
            "chain_ok": ok,
            "chain_errors": errors,
            "title": t("diary.title", normalize_language(lang or case.language)),
        }

    def report(self, case_id: str, lang: Optional[str] = None,
               formats: Sequence[str] = ("md", "html", "json", "pdf")) -> dict:
        case = self.load_case(case_id)
        index = self.build_index(case)
        if not case.findings and any(i.source_type in ("bank", "cdr", "social") for i in case.evidence):
            result = self.analyze(case_id)
            if result.get("ok"):
                case = self.load_case(case_id)
                index = self.build_index(case, force=True)
        case = self._ensure_recommendations(case, index, lang=lang or case.language)
        files = generate_case_report(
            case, index, lang=normalize_language(lang or case.language),
            out_dir=self.store.reports_dir(case), formats=tuple(formats),
            min_events=self.config.min_events_per_entity,
        )
        self.store.append_diary(
            case, "export",
            t("diary.report", case.language, path=files.get("md") or files.get("html") or ""),
            actor="analyst",
        )
        return {"case_id": case.case_id, "files": files}

    def add_note(self, case_id: str, text: str, actor: str = "analyst") -> dict:
        case = self.load_case(case_id)
        entry = self.store.append_diary(case, "note", text, actor=actor)
        return entry.to_dict()

    def verify_diary(self, case_id: str) -> dict:
        case = self.load_case(case_id)
        ok, errors = verify_chain(case)
        return {"case_id": case.case_id, "ok": ok, "errors": errors}

    def overview(self, case_id: str, lang: Optional[str] = None) -> dict:
        case = self.load_case(case_id)
        index = self.build_index(case)
        summary = case_summary(case, index, lang=normalize_language(lang or case.language),
                               min_events=self.config.min_events_per_entity)
        return {
            "case": case.to_dict(),
            "stats": index.stats(),
            "summary": summary,
            "gaps": evidence_gaps(case, index, min_events=self.config.min_events_per_entity),
            "languages": [{"code": code, "name": language_name(code)}
                          for code in SUPPORTED_LANGUAGES],
            "llm": self.llm.config.to_dict(),
        }

    def _parser_for(self, case: Case, index: EvidenceIndex, lang: Optional[str]):
        from .query import QueryParser
        return QueryParser(
            alias_to_entity=index.aliases,
            entity_names={eid: ent.names for eid, ent in index.entities.items()},
            case_start=index.stats().get("start") or None,
            case_end=index.stats().get("end") or None,
            default_lang=normalize_language(lang or case.language),
        )

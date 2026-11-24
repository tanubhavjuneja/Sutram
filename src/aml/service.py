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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .assistant import ForensicAssistant, SessionContext
from .case_report import generate_case_report
from .case_store import CaseStore
from .config import InvestigationConfig
from .diary import append_entry, verify_chain
from .evidence import IngestResult, ingest_paths
from .i18n import SUPPORTED_LANGUAGES, language_name, normalize_language, t
from .index import EvidenceIndex
from .llm import LLMClient
from .case import Answer, Case, Citation, Entity, Finding, QueryPlan, Recommendation, utc_now
from .recommend import generate_recommendations, recommendation_summary
from .summarize import case_summary, entity_dossier, evidence_gaps, timeline_summary


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
        # Latest analytics PipelineResult per case, reused by the chart/deep-dive
        # tools without re-running the models on every chat turn.
        self._analytics_results: Dict[str, Any] = {}

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
        self._analytics_results.pop(case.case_id, None)
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
        self._analytics_results[case.case_id] = res
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

    def _finish_answer(self, case: Case, question: str, answer: Answer) -> Answer:
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

    # -- agentic tools -----------------------------------------------------
    def _run_tool(self, case: Case, index: EvidenceIndex, plan: QueryPlan,
                  question: str, lang: str) -> Answer:
        if plan.intent == "run_analysis":
            return self._tool_run_analysis(case, index, plan, lang)
        if plan.intent == "recheck":
            return self._tool_recheck(case, index, plan, lang)
        if plan.intent == "plot":
            return self._tool_plot(case, index, plan, lang)
        if plan.intent == "deep_dive":
            return self._tool_deep_dive(case, index, plan, lang)
        if plan.intent == "generate_image":
            return self._tool_generate_image(case, index, plan, lang)
        return Answer(answer="Unsupported tool request.", intent=plan.intent,
                      language=lang, confidence=0.2)

    @staticmethod
    def _decode_image(data_url: str) -> Optional[bytes]:
        import base64
        try:
            if data_url.startswith("data:"):
                _, encoded = data_url.split(",", 1)
                return base64.b64decode(encoded)
            return base64.b64decode(data_url)
        except Exception:
            return None

    def _tool_vision(self, case: Case, question: str, images: List[str],
                     lang: str) -> Answer:
        if not self.llm.available:
            return Answer(
                answer=("**Image analysis unavailable**\n\nThe vision model is not "
                        "configured. Add the Cloudflare Workers AI credentials to `.env`."),
                intent="vision", language=lang, confidence=0.2,
            )
        out_dir = self.store.reports_dir(case) / "chat"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        image_records = []
        descriptions = []
        for idx, data_url in enumerate(images[:3], 1):
            raw = self._decode_image(data_url)
            if raw:
                ext = "jpg" if raw[:2] == b"\xff\xd8" else "png"
                path = out_dir / f"upload_{stamp}_{idx}.{ext}"
                path.write_bytes(raw)
                rel = path.relative_to(Path(case.root)).as_posix()
                image_records.append({
                    "kind": "upload",
                    "path": str(path),
                    "url": f"/files/{case.case_id}/{rel}",
                    "caption": f"Uploaded image {idx}",
                })
            description = self.llm.describe_image(data_url, question, lang=lang)
            if description:
                descriptions.append((idx, description))
        if not descriptions:
            return Answer(
                answer=("**Image analysis failed**\n\nThe vision model could not read "
                        "the uploaded image. Check the file format and try again."),
                intent="vision", language=lang, confidence=0.2,
                images=image_records,
            )
        lines = [
            "**Image analysis (vision model)**",
            "",
            "*Model observation — verify against the original evidence. "
            "This is not a forensic conclusion.*",
            "",
        ]
        for idx, description in descriptions:
            if len(descriptions) > 1:
                lines.append(f"**Image {idx}**")
            lines.append(description)
            lines.append("")
        return Answer(
            answer="\n".join(lines), intent="vision", language=lang,
            images=image_records, confidence=0.7,
            data={"vision_model": self.llm.config.model},
        )

    def _tool_generate_image(self, case: Case, index: EvidenceIndex,
                             plan: QueryPlan, lang: str) -> Answer:
        if not self.llm.available:
            return Answer(
                answer=("**Image generation unavailable**\n\nThe image model is not "
                        "configured. Add the Cloudflare Workers AI credentials to `.env`."),
                intent="generate_image", language=lang, confidence=0.2,
            )
        prompt = (plan.image_prompt or plan.raw or "").strip()
        if not prompt:
            return Answer(answer="**Image generation needs a prompt.**",
                          intent="generate_image", language=lang, confidence=0.2)
        image_bytes = self.llm.generate_image(prompt)
        if not image_bytes:
            return Answer(
                answer=("**Image generation failed**\n\nThe model did not return an "
                        "image. Try a shorter or more specific prompt."),
                intent="generate_image", language=lang, confidence=0.2,
            )
        out_dir = self.store.reports_dir(case) / "chat"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        ext = "jpg" if image_bytes[:2] == b"\xff\xd8" else "png"
        path = out_dir / f"generated_{stamp}.{ext}"
        path.write_bytes(image_bytes)
        rel = path.relative_to(Path(case.root)).as_posix()
        image = {
            "kind": "generated",
            "path": str(path),
            "url": f"/files/{case.case_id}/{rel}",
            "caption": f"Generated illustration — {prompt[:90]}",
        }
        text = (
            "**Generated illustration**\n\n"
            f"**Prompt:** {prompt}\n\n"
            f"Generated with `{self.llm.config.image_model}`. "
            "This is a synthetic illustration, not evidence."
        )
        return Answer(answer=text, intent="generate_image", language=lang,
                      images=[image], confidence=0.8,
                      data={"prompt": prompt, "model": self.llm.config.image_model})

    def _synthetic_result(self, case: Case, index: EvidenceIndex):
        """Build a chart-compatible PipelineResult from the case index."""
        from types import SimpleNamespace
        from .pipeline import PipelineResult
        from .schema import SourceResolved

        rankings = [
            {"entity_id": eid, "score": float(ent.risk_score),
             "n_events": int(ent.n_events), "models_fired": []}
            for eid, ent in case.entities.items()
        ]
        entity_map = SimpleNamespace(
            entity_names={eid: set(ent.names) for eid, ent in case.entities.items()},
            entity_id_by_identifier={},
            name_to_entity={},
        )
        per_source = [
            SourceResolved(source=item.source_type, file=item.path,
                           resolved_fields=set(), n_rows=item.n_rows,
                           n_entities=0)
            for item in case.evidence
        ]
        return PipelineResult(unified=index.events, per_source=per_source,
                              sufficiency={}, entity_map=entity_map,
                              rankings=rankings)

    def _result_for_charts(self, case: Case, index: EvidenceIndex):
        res = self._analytics_results.get(case.case_id)
        if res is not None:
            return res, ""
        structured = [i for i in case.evidence
                      if i.source_type in ("bank", "cdr", "social")]
        if structured:
            result = self.analyze(case.case_id, run_report=False)
            res = self._analytics_results.get(case.case_id)
            if res is not None:
                return res, ""
            return None, result.get("error", "Analytics could not run.")
        return self._synthetic_result(case, index), ""

    def _render_chart(self, case: Case, res, kind: str, entity_id: str = "") -> Optional[dict]:
        from . import visuals

        out_dir = self.store.reports_dir(case) / "chat"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        if kind == "map":
            path = visuals.render_location_map(res, out_dir, name=f"chat_map_{stamp}.png")
            caption = "Activity map — locations & tower telemetry"
        elif kind == "timeline":
            path = visuals.render_timeline(res, out_dir, name=f"chat_timeline_{stamp}.png")
            caption = "Daily traffic timeline"
        elif kind == "entity_timeline":
            eid = entity_id or next(iter(case.entities), "")
            if not eid:
                return None
            ent = case.entities.get(eid)
            name = ent.display_name if ent else eid
            path = visuals.render_entity_timeline(
                res, out_dir, eid, name=f"chat_entity_{eid}_{stamp}.png")
            caption = f"Activity timeline — {name} ({eid})"
        elif kind == "people":
            path = visuals.render_person_attention(
                res, out_dir, name=f"chat_people_{stamp}.png")
            caption = "People who stand out"
        elif kind == "sources":
            path = visuals.render_sources(res, out_dir, name=f"chat_sources_{stamp}.png")
            caption = "Evidence sources"
        elif kind == "models":
            if not getattr(res, "model_scores", None):
                return None
            path = visuals.render_model_scores(
                res, out_dir, name=f"chat_models_{stamp}.png")
            caption = "Detection engine scores"
        else:
            path = visuals.render_correlation_graph(
                res, out_dir, name=f"chat_network_{stamp}.png",
                tol_minutes=self.config.tol_minutes)
            caption = "Suspect correlation graph"
        rel = path.relative_to(Path(case.root)).as_posix()
        return {"kind": kind, "path": str(path),
                "url": f"/files/{case.case_id}/{rel}", "caption": caption}

    def _citations_from_hits(self, index: EvidenceIndex, hits: List[dict],
                             limit: int = 6) -> List[Citation]:
        from .retrieval import HybridRetriever
        retriever = HybridRetriever(index)
        out: List[Citation] = []
        seen = set()
        for hit in hits:
            citation = retriever.citation_for_hit(hit)
            if citation is None:
                continue
            key = (citation.evidence_id, citation.event_id, citation.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            out.append(citation)
            if len(out) >= limit:
                break
        return out

    def _tool_plot(self, case: Case, index: EvidenceIndex, plan: QueryPlan,
                   lang: str) -> Answer:
        kind = plan.chart_kind or "network"
        res, error = self._result_for_charts(case, index)
        if res is None:
            return Answer(answer=f"**Chart unavailable**\n\n{error}", intent="plot",
                          language=lang, confidence=0.3,
                          data={"chart_kind": kind})
        entity_id = plan.entity_ids[0] if plan.entity_ids else ""
        image = self._render_chart(case, res, kind, entity_id=entity_id)
        if image is None:
            return Answer(
                answer=("**Chart unavailable**\n\nI could not render that chart "
                        "from the current evidence. Run the analysis first, then try again."),
                intent="plot", language=lang, confidence=0.3,
                data={"chart_kind": kind},
            )
        text = (
            f"**Generated chart — {image['caption']}**\n\n"
            f"Rendered from the current case evidence at {utc_now()}. "
            "The image is shown below; the same chart is available in the case report."
        )
        return Answer(answer=text, intent="plot", language=lang, confidence=0.92,
                      images=[image], data={"chart_kind": kind, "image": image})

    def _tool_run_analysis(self, case: Case, index: EvidenceIndex, plan: QueryPlan,
                           lang: str) -> Answer:
        result = self.analyze(case.case_id)
        if not result.get("ok"):
            return Answer(answer=f"**Analysis could not run**\n\n{result.get('error')}",
                          intent="run_analysis", language=lang, confidence=0.3,
                          data={"analysis": result})
        case = self.load_case(case.case_id)
        index = self.build_index(case, force=True)
        lines = [
            "**Analysis rerun complete.**",
            "",
            f"- models run: {', '.join(result.get('models', [])) or '—'}",
            f"- findings: **{result.get('findings', 0)}**",
            f"- recommendations: **{result.get('recommendations', 0)}**",
            "",
            "| model | verdict |",
            "|---|---|",
        ]
        for model, status in sorted(result.get("sufficiency", {}).items()):
            lines.append(f"| {model} | {status} |")
        rankings = result.get("top_rankings", [])[:8]
        if rankings:
            lines += ["", "**Top ranked entities**", "",
                      "| rank | entity | score | fired |",
                      "|---|---|---|---|"]
            for i, row in enumerate(rankings, 1):
                lines.append(
                    f"| {i} | {row.get('entity_id', '')} | {row.get('score', 0):.3f} | "
                    f"{', '.join(row.get('models_fired', [])) or '—'} |"
                )
        if case.findings:
            lines += ["", "**Top findings**", ""]
            for finding in case.findings[:5]:
                lines.append(f"- [{finding.severity}] {finding.title}")
        citations = []
        for finding in case.findings[:3]:
            citations.extend(finding.citations)
        seen = set()
        unique_citations = []
        for citation in citations:
            key = (citation.evidence_id, citation.event_id, citation.chunk_id)
            if key not in seen:
                seen.add(key)
                unique_citations.append(citation)
        analytics_res = self._analytics_results.get(case.case_id)
        image = self._render_chart(case, analytics_res, "network") if analytics_res is not None else None
        artifacts = []
        report_files = (case.metadata.get("analytics") or {}).get("report_files", {}) or {}
        for label, key in (("Analytics report (HTML)", "html"),
                           ("Analytics report (PDF)", "pdf")):
            path = report_files.get(key)
            if path and Path(path).exists():
                rel = Path(path).resolve().relative_to(Path(case.root).resolve()).as_posix()
                artifacts.append({"label": label, "path": str(path),
                                  "url": f"/files/{case.case_id}/{rel}"})
        return Answer(
            answer="\n".join(lines), intent="run_analysis", language=lang,
            citations=unique_citations[:8],
            images=[image] if image else [],
            artifacts=artifacts,
            confidence=0.95,
            data={"analysis": result},
        )

    def _tool_recheck(self, case: Case, index: EvidenceIndex, plan: QueryPlan,
                      lang: str) -> Answer:
        agg = index.aggregate(plan)
        if agg["count"] == 0:
            return Answer(
                answer="**Recomputed numbers**\n\nNo indexed events match that request.",
                intent="recheck", language=lang, confidence=0.25,
                data={"aggregate": agg},
            )
        lines = [
            "**Recomputed numbers — from the current evidence index.**",
            "",
            f"*Recomputed at {utc_now()} from {agg['count']:,} matching event(s).*",
            "",
            "| metric | value |",
            "|---|---|",
            f"| matching events | {agg['count']:,} |",
            f"| total amount | {agg['total']:,.2f} |",
            f"| average amount | {agg['avg']:,.2f} |",
            f"| smallest amount | {agg['min']:,.2f} |",
            f"| largest amount | {agg['max']:,.2f} |",
        ]
        if agg.get("by_source"):
            lines.append(f"| by source | {', '.join(f'{k}: {v:,}' for k, v in agg['by_source'].items())} |")
        if agg.get("by_direction"):
            lines.append(f"| by direction | {', '.join(f'{k}: {v:,}' for k, v in agg['by_direction'].items())} |")
        citations = self._citations_from_hits(index, agg.get("hits", []))
        lines += ["", "The numbers above are recomputed from the indexed events, "
                      "not copied from the previous report."]
        return Answer(answer="\n".join(lines), intent="recheck", language=lang,
                      citations=citations, hits=agg.get("hits", []),
                      confidence=0.9, data={"aggregate": agg})

    def _tool_deep_dive(self, case: Case, index: EvidenceIndex, plan: QueryPlan,
                        lang: str) -> Answer:
        text = (plan.raw or "").lower()
        if "structur" in text:
            model = "structuring"
        elif "chain" in text or "layer" in text:
            model = "chain"
        elif "network" in text or "hub" in text or "fan" in text:
            model = "network"
        elif "time" in text or "call" in text or "transfer" in text:
            model = "time_correlation"
        elif "stat" in text or "outlier" in text or "anomaly" in text:
            model = "statml"
        else:
            model = "network"
        from .models import run_model
        scores = run_model(
            model, index.events,
            tolerance_minutes=self.config.tol_minutes,
            structuring_threshold=self.config.structuring_threshold,
        )
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:10]
        lines = [
            f"**Deep dive — {model} model**",
            "",
            f"Scored **{len(scores)}** entities from the current evidence index. "
            "This is a fresh model run, not a copy of the previous report.",
        ]
        citations: List[Citation] = []
        if top:
            lines += ["", "| entity | name | score |", "|---|---|---|"]
            for eid, score in top:
                ent = case.entities.get(str(eid))
                name = ent.display_name if ent else str(eid)
                lines.append(f"| {eid} | {name} | {score:.3f} |")
            focus = plan.entity_ids[0] if plan.entity_ids else str(top[0][0])
            if focus in case.entities:
                from .summarize import entity_dossier
                dossier = entity_dossier(case, index, focus, lang=lang)
                if dossier:
                    lines += ["", dossier["text"]]
                    citations = [Citation.from_dict(c) for c in dossier.get("citations", [])]
            if not citations:
                citations = self._citations_from_hits(
                    index, index.search_events(plan, top_k=5))
        else:
            lines += ["", "The model returned no scores for the current evidence."]
        if plan.entity_ids:
            chart_res, _chart_error = self._result_for_charts(case, index)
            image = (self._render_chart(case, chart_res, "entity_timeline",
                                        entity_id=plan.entity_ids[0])
                     if chart_res is not None else None)
        else:
            image = None
        return Answer(answer="\n".join(lines), intent="deep_dive", language=lang,
                      citations=citations[:8], images=[image] if image else [],
                      confidence=0.85, data={"model": model, "scores": dict(top)})

    def ask(self, case_id: str, question: str, lang: Optional[str] = None,
            session_id: str = "default", images: Optional[List[str]] = None) -> Answer:
        case = self.load_case(case_id)
        index = self.build_index(case)
        target_lang = normalize_language(lang or case.language)
        if images:
            answer = self._tool_vision(case, question, images, target_lang)
            return self._finish_answer(case, question, answer)
        assistant = ForensicAssistant(case, index, lang=lang or case.language,
                                      llm=self.llm,
                                      min_events=self.config.min_events_per_entity)
        plan = assistant.parser.parse(question, lang=lang or case.language)
        if plan.intent in ("run_analysis", "recheck", "plot", "deep_dive",
                           "generate_image"):
            answer = self._run_tool(case, index, plan, question, target_lang)
            return self._finish_answer(case, question, answer)
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
        return self._finish_answer(case, question, answer)

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

"""Unified pipeline: sources -> schema detect -> entity resolve -> long df -> models.

Assembles ONLY the model set the data can support (sufficiency engine) and
produces ranked insights with explanations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .schema import SourceResolved
from .schema_detect import detect_dataframe
from .sufficiency import SufficiencyEngine, SufficiencyVerdict

UNIFIED_COLUMNS = [
    "entity_id", "event_type", "timestamp", "source",
    "amount", "counterparty_id", "location", "direction", "duration",
]


@dataclass
class PipelineResult:
    unified: pd.DataFrame
    per_source: List[SourceResolved]
    sufficiency: Dict[str, SufficiencyVerdict]
    entity_map: object
    rankings: List[dict] = field(default_factory=list)
    explanations: Dict[str, str] = field(default_factory=dict)
    model_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    # Optional investigation-workspace fields.  The analytics-only pipeline
    # leaves these empty; CasePipeline populates them so callers can consume
    # one result object for both anomaly analytics and case investigation.
    case: Any = None
    evidence: List[Any] = field(default_factory=list)
    chunks: List[Any] = field(default_factory=list)
    index: Any = None
    findings: List[Any] = field(default_factory=list)
    recommendations: List[Any] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    diary: List[Any] = field(default_factory=list)


class Pipeline:
    def __init__(self, min_events_per_entity: int = 30, tol_minutes: int = 30,
                 structuring_threshold: float = 10000.0,
                 fusion: str = "borda", probe_cfg=None):
        self.min_events_per_entity = min_events_per_entity
        self.tol_minutes = tol_minutes
        self.structuring_threshold = structuring_threshold
        self.fusion = fusion
        self.probe_cfg = probe_cfg

    def ingest(self, paths):
        from .loaders import load_any
        frames = []
        resolved_sources = []
        for path in paths:
            loaded = load_any(path)
            det = detect_dataframe(loaded.df, file=loaded.file, probe_cfg=self.probe_cfg)
            if loaded.warning:
                det.warnings.insert(0, loaded.warning)
            norm = det.normalized
            rs = SourceResolved(
                source=det.source_type, file=det.file,
                resolved_fields=det.resolved_slots,
                warnings=list(det.warnings),
                n_rows=det.n_rows, n_entities=0,
            )
            frames.append(norm)
            resolved_sources.append(rs)
        non_empty = [f for f in frames if f.shape[0] > 0]
        if non_empty:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                unified = pd.concat(non_empty, ignore_index=True, sort=False)
        else:
            unified = pd.DataFrame()
        # the same logical fact sheet may arrive in several formats (csv +
        # json of the same export); normalise remaining 'missing' spellings,
        # then drop exact duplicate rows so one event is never counted twice
        # by merge_asof / counts / feature vectors
        if not unified.empty:
            from .schema_detect import _clean_ids
            for c in unified.columns:
                if unified[c].dtype == object:
                    unified[c] = _clean_ids(unified[c])
            unified = unified.drop_duplicates()
        for f in UNIFIED_COLUMNS + ["actor_raw", "counterparty_id_raw", "actor_name"]:
            if f not in unified.columns:
                unified[f] = np.nan
        if "timestamp" in unified.columns:
            unified["timestamp"] = pd.to_datetime(unified["timestamp"], errors="coerce")
        return unified, resolved_sources

    def resolve(self, unified, resolved_sources):
        from .entity_resolve import resolve_entities
        em = resolve_entities(unified)
        unified = unified.copy()
        unified["entity_id"] = unified["actor_raw"].map(em.entity_id_by_identifier)
        unified["counterparty_id"] = unified["counterparty_id_raw"].map(em.entity_id_by_identifier)
        for rs in resolved_sources:
            sub = unified[unified["source"] == rs.source]
            rs.n_entities = int(sub["entity_id"].nunique()) if "entity_id" in sub else 0
        return unified, em

    def run(self, paths, with_models=True) -> PipelineResult:
        unified, resolved_sources = self.ingest(paths)
        unified, em = self.resolve(unified, resolved_sources)
        engine = SufficiencyEngine(
            resolved_sources, self.min_events_per_entity,
            unified_n_rows=len(unified),
            unified_n_entities=int(unified["entity_id"].nunique()) if not unified.empty else 0,
        )
        sufficiency = engine.evaluate()
        result = PipelineResult(
            unified=unified, per_source=resolved_sources,
            sufficiency=sufficiency, entity_map=em,
        )
        if with_models and not unified.empty:
            self._run_models(result)
        return result

    def _run_models(self, result):
        from . import models as _m
        sufficiency = result.sufficiency
        df = result.unified
        scores = {}
        for model in ("time_correlation", "network", "statml", "benford", "structuring", "behavioral", "chain"):
            verdict = sufficiency[model].status
            if verdict != "BLOCKED":
                sc = _m.run_model(model, df, tolerance_minutes=self.tol_minutes,
                                  structuring_threshold=self.structuring_threshold)
                if sc:
                    scores[model] = sc
                else:
                    # honest empty result: the verdict let this model run but
                    # no events qualified — surface it instead of disappearing
                    result.warnings.append(
                        f"{model}: no events qualified for scoring "
                        f"(verdict {verdict} but empty score set; likely no eligible "
                        f"cross-source/edge/amount events in the unified frame)"
                    )
        fusion = self._fuse(scores, df)
        result.model_scores = scores
        result.rankings = fusion.rank
        result.explanations = fusion.explanation
        result.warnings = [v.reason for v in sufficiency.values() if v.status == "BLOCKED"] + result.warnings
        return result

    def _fuse(self, scores, df):
        """Build the fused ranking; unknown fusion names fall back to Borda."""
        from . import models as _m
        if self.fusion == "meta":
            try:
                from .fusion_meta import fuse_rank_meta
                return fuse_rank_meta(scores, df)
            except Exception:
                pass
        if self.fusion in ("score_mean", "rank_avg", "top2"):
            fn = {"score_mean": _m.fuse_score_mean, "rank_avg": _m.fuse_rank_average,
                  "top2": _m.fuse_model_scores}[self.fusion]
            return fn(scores, df)
        return _m.fuse_rank_borda(scores, df)


class CasePipeline(Pipeline):
    """Pipeline facade for the case workspace.

    The analytics core stays exactly as it was; this class adds the case
    lifecycle, evidence ingestion, retrieval, assistant, recommendations,
    diary and report methods as part of the same ``aml`` package and the same
    public pipeline entry point.
    """

    def __init__(self, case_root: str = "data/cases", investigation_config=None,
                 **kwargs):
        super().__init__(**kwargs)
        from .config import InvestigationConfig
        from .service import InvestigationService
        cfg = investigation_config or InvestigationConfig(case_root=case_root)
        if case_root and not cfg.case_root:
            cfg.case_root = case_root
        self.service = InvestigationService(case_root=cfg.case_root, config=cfg)

    # -- case lifecycle ----------------------------------------------------
    def create_case(self, title: str, description: str = "", language: str = "en"):
        return self.service.create_case(title, description=description, language=language)

    def load_case(self, case_id: str):
        return self.service.load_case(case_id)

    def list_cases(self):
        return self.service.list_cases()

    def ingest_case(self, case_id: str, paths: Sequence[str], custodian: str = "",
                    notes: str = ""):
        return self.service.ingest(case_id, paths, custodian=custodian, notes=notes)

    def analyze_case(self, case_id: str, run_report: bool = True):
        return self.service.analyze(case_id, run_report=run_report)

    # -- conversation / retrieval -----------------------------------------
    def ask(self, case_id: str, question: str, lang: Optional[str] = None,
            session_id: str = "default", images: Optional[List[str]] = None):
        return self.service.ask(case_id, question, lang=lang,
                                session_id=session_id, images=images)

    def search(self, case_id: str, question: str, lang: Optional[str] = None):
        case = self.service.load_case(case_id)
        index = self.service.build_index(case)
        parser = self.service._parser_for(case, index, lang)
        plan = parser.parse(question, lang=lang or case.language)
        from .retrieval import HybridRetriever
        return HybridRetriever(index).search(plan)

    def summarize(self, case_id: str, lang: Optional[str] = None):
        return self.service.summarize(case_id, lang=lang)

    def recommend(self, case_id: str, lang: Optional[str] = None):
        return self.service.recommend(case_id, lang=lang)

    def report(self, case_id: str, lang: Optional[str] = None,
               formats: Sequence[str] = ("md", "html", "json", "pdf")):
        return self.service.report(case_id, lang=lang, formats=formats)

    def diary(self, case_id: str, lang: Optional[str] = None):
        return self.service.diary(case_id, lang=lang)

    # -- convenience -------------------------------------------------------
    def run_case(self, case_id: str, paths: Optional[Sequence[str]] = None,
                 analyze: bool = True, report: bool = True,
                 lang: Optional[str] = None):
        """Ingest (optional), analyze and report; return the updated case."""
        self.service.load_case(case_id)
        if paths:
            self.service.ingest(case_id, paths)
        if analyze:
            self.service.analyze(case_id, run_report=report)
        if report:
            self.service.report(case_id, lang=lang)
        return self.service.load_case(case_id)

    def run_case_result(self, case_id: str, paths: Optional[Sequence[str]] = None,
                        analyze: bool = True, report: bool = True,
                        lang: Optional[str] = None) -> PipelineResult:
        """Return a unified PipelineResult containing analytics + case data."""
        case = self.run_case(case_id, paths=paths, analyze=analyze,
                             report=report, lang=lang)
        index = self.service.build_index(case)
        events = self.service.store.load_events(case)
        summary = self.service.summarize(case_id, lang=lang)
        return PipelineResult(
            unified=events,
            per_source=[],
            sufficiency={},
            entity_map=None,
            case=case,
            evidence=list(case.evidence),
            chunks=list(index.chunks),
            index=index,
            findings=list(case.findings),
            recommendations=list(case.recommendations),
            summary=summary,
            diary=list(case.diary),
        )
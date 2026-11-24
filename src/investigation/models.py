"""Dataclasses shared by the investigation layer.

Everything persisted in ``case.json`` / index files is defined here.  The
objects are deliberately plain (dataclasses + JSON-safe primitives) so a case
can be inspected, diffed and archived without a database.

Timestamps are ISO-8601 strings in UTC.  A case is the single aggregate root:
evidence, entities, findings, recommendations and the diary all belong to it.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def utc_now() -> str:
    """Current UTC time as an ISO-8601 string (seconds precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str, fallback: str = "case") -> str:
    """Filesystem-safe slug used in case ids."""
    text = re.sub(r"[^a-zA-Z0-9]+", "-", value or "").strip("-").lower()
    return text[:48] or fallback


def as_list(value) -> List[str]:
    """Coerce a scalar/iterable into a de-duplicated string list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    out: List[str] = []
    seen = set()
    for item in value:
        text = str(item) if item is not None else ""
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _from_dict(cls, data: Optional[dict]):
    if not data:
        return cls()
    known = {f for f in cls.__dataclass_fields__}
    return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# query plan
# ---------------------------------------------------------------------------
@dataclass
class TimeRange:
    start: Optional[str] = None          # ISO-8601
    end: Optional[str] = None            # ISO-8601
    label: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "TimeRange":
        return _from_dict(cls, data)


@dataclass
class AmountRange:
    min: Optional[float] = None
    max: Optional[float] = None
    label: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "AmountRange":
        return _from_dict(cls, data)


@dataclass
class QueryPlan:
    raw: str
    language: str = "en"
    intent: str = "search"
    entity_terms: List[str] = field(default_factory=list)
    entity_ids: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    event_types: List[str] = field(default_factory=list)
    time_range: TimeRange = field(default_factory=TimeRange)
    amount_range: AmountRange = field(default_factory=AmountRange)
    direction: Optional[str] = None
    location_terms: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    limit: int = 20
    wants_citations: bool = True
    ambiguous: List[dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["time_range"] = self.time_range.to_dict()
        data["amount_range"] = self.amount_range.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "QueryPlan":
        data = dict(data or {})
        data["time_range"] = TimeRange.from_dict(data.get("time_range"))
        data["amount_range"] = AmountRange.from_dict(data.get("amount_range"))
        return _from_dict(cls, data)


# ---------------------------------------------------------------------------
# evidence + citations
# ---------------------------------------------------------------------------
@dataclass
class EvidenceItem:
    evidence_id: str
    path: str                         # relative to the case root
    filename: str
    sha256: str = ""
    size_bytes: int = 0
    mime: str = ""
    source_type: str = "other"        # bank|cdr|social|email|chat|document|note|image|other
    acquired_at: str = field(default_factory=utc_now)
    custodian: str = ""
    notes: str = ""
    original_path: str = ""
    n_rows: int = 0
    n_chunks: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "EvidenceItem":
        return _from_dict(cls, data)


@dataclass
class Citation:
    evidence_id: str
    chunk_id: str = ""
    event_id: str = ""
    file: str = ""
    timestamp: str = ""
    snippet: str = ""
    source_type: str = ""
    entity_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Citation":
        return _from_dict(cls, data)

    def label(self, index: int) -> str:
        return f"C{index}"


# ---------------------------------------------------------------------------
# entities
# ---------------------------------------------------------------------------
@dataclass
class Entity:
    entity_id: str
    names: List[str] = field(default_factory=list)
    identifiers: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)       # suspect|victim|witness|unknown
    sources: List[str] = field(default_factory=list)
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    n_events: int = 0
    risk_score: float = 0.0
    risk_factors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return self.names[0] if self.names else self.entity_id

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Entity":
        return _from_dict(cls, data)


@dataclass
class Chunk:
    chunk_id: str
    evidence_id: str
    text: str
    timestamp: Optional[str] = None
    source_type: str = "document"
    entity_ids: List[str] = field(default_factory=list)
    location: Optional[str] = None
    line_start: int = 1
    line_end: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Chunk":
        return _from_dict(cls, data)


# ---------------------------------------------------------------------------
# findings + recommendations
# ---------------------------------------------------------------------------
@dataclass
class Finding:
    finding_id: str
    kind: str                        # structuring|colocation|chain|fan_io|dormancy|silence|anomaly|model_signal|insight
    title: str
    description: str
    severity: str = "LOW"            # HIGH|MEDIUM|LOW
    confidence: float = 0.0
    entity_ids: List[str] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    detected_by: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["citations"] = [c.to_dict() for c in self.citations]
        return data

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Finding":
        data = dict(data or {})
        data["citations"] = [Citation.from_dict(c) for c in data.get("citations", [])]
        return _from_dict(cls, data)


@dataclass
class Recommendation:
    recommendation_id: str
    priority: str                    # HIGH|MEDIUM|LOW
    action: str
    rationale: str
    category: str = "analysis"       # evidence_gap|interview|records|forensics|corroboration|analysis
    entity_ids: List[str] = field(default_factory=list)
    finding_ids: List[str] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    status: str = "open"             # open|accepted|done|dismissed
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Recommendation":
        return _from_dict(cls, data)


# ---------------------------------------------------------------------------
# diary
# ---------------------------------------------------------------------------
@dataclass
class DiaryEntry:
    entry_id: str
    seq: int
    timestamp: str
    actor: str
    entry_type: str                  # ingest|analysis|query|finding|recommendation|action|export|note
    text: str
    evidence_ids: List[str] = field(default_factory=list)
    prev_hash: str = ""
    hash: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "DiaryEntry":
        return _from_dict(cls, data)


# ---------------------------------------------------------------------------
# answers
# ---------------------------------------------------------------------------
@dataclass
class Answer:
    answer: str
    intent: str = "search"
    language: str = "en"
    citations: List[Citation] = field(default_factory=list)
    hits: List[dict] = field(default_factory=list)
    confidence: float = 0.0
    followups: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    query_plan: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "intent": self.intent,
            "language": self.language,
            "citations": [c.to_dict() for c in self.citations],
            "hits": self.hits,
            "confidence": round(float(self.confidence), 3),
            "followups": self.followups,
            "data": self.data,
            "query_plan": self.query_plan,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Answer":
        data = dict(data or {})
        data["citations"] = [Citation.from_dict(c) for c in data.get("citations", [])]
        return _from_dict(cls, data)


# ---------------------------------------------------------------------------
# case aggregate
# ---------------------------------------------------------------------------
@dataclass
class Case:
    case_id: str
    title: str
    description: str = ""
    language: str = "en"
    status: str = "open"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    evidence: List[EvidenceItem] = field(default_factory=list)
    entities: Dict[str, Entity] = field(default_factory=dict)
    findings: List[Finding] = field(default_factory=list)
    recommendations: List[Recommendation] = field(default_factory=list)
    diary: List[DiaryEntry] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    root: str = ""

    # -- convenience -------------------------------------------------------
    def evidence_by_id(self, evidence_id: str) -> Optional[EvidenceItem]:
        for item in self.evidence:
            if item.evidence_id == evidence_id:
                return item
        return None

    def entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    def finding(self, finding_id: str) -> Optional[Finding]:
        for item in self.findings:
            if item.finding_id == finding_id:
                return item
        return None

    def touch(self) -> None:
        self.updated_at = utc_now()

    def stats(self) -> dict:
        sources: Dict[str, int] = {}
        for item in self.evidence:
            sources[item.source_type] = sources.get(item.source_type, 0) + 1
        return {
            "evidence": len(self.evidence),
            "sources": sources,
            "entities": len(self.entities),
            "findings": len(self.findings),
            "recommendations": len(self.recommendations),
            "diary_entries": len(self.diary),
        }

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "title": self.title,
            "description": self.description,
            "language": self.language,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "evidence": [e.to_dict() for e in self.evidence],
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "findings": [f.to_dict() for f in self.findings],
            "recommendations": [r.to_dict() for r in self.recommendations],
            "diary": [d.to_dict() for d in self.diary],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict, root: str = "") -> "Case":
        data = dict(data or {})
        return cls(
            case_id=str(data.get("case_id", "")),
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            language=str(data.get("language", "en")),
            status=str(data.get("status", "open")),
            created_at=str(data.get("created_at") or utc_now()),
            updated_at=str(data.get("updated_at") or utc_now()),
            evidence=[EvidenceItem.from_dict(e) for e in data.get("evidence", [])],
            entities={
                str(k): Entity.from_dict(v)
                for k, v in (data.get("entities") or {}).items()
            },
            findings=[Finding.from_dict(f) for f in data.get("findings", [])],
            recommendations=[Recommendation.from_dict(r) for r in data.get("recommendations", [])],
            diary=[DiaryEntry.from_dict(d) for d in data.get("diary", [])],
            metadata=dict(data.get("metadata") or {}),
            root=root,
        )

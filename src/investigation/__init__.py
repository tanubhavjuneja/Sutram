"""Sutram Investigator — case-centric digital investigation assistant.

This package sits on top of the verified ``aml`` analytics core and adds:

* case + evidence + chain-of-custody management
* text/event indexing and hybrid evidence retrieval
* multilingual natural-language querying and answer synthesis
* case/entity/timeline summarization
* investigative recommendation generation
* a hash-chained case diary
* draft case reports
* a conversational assistant (CLI + web)

The analytics core remains unchanged; it is invoked by
:class:`investigation.service.InvestigationService` when suspicious-pattern
analysis is required.
"""

from .models import (
    AmountRange,
    Answer,
    Case,
    Chunk,
    Citation,
    DiaryEntry,
    Entity,
    EvidenceItem,
    Finding,
    QueryPlan,
    Recommendation,
    TimeRange,
)
from .i18n import SUPPORTED_LANGUAGES, detect_language, t
from .service import InvestigationService

__all__ = [
    "AmountRange",
    "Answer",
    "Case",
    "Chunk",
    "Citation",
    "DiaryEntry",
    "Entity",
    "EvidenceItem",
    "Finding",
    "InvestigationService",
    "QueryPlan",
    "Recommendation",
    "SUPPORTED_LANGUAGES",
    "TimeRange",
    "detect_language",
    "t",
]

__version__ = "0.1.0"

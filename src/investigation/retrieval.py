"""Hybrid evidence retrieval with citation grounding.

``HybridRetriever`` is the read path used by the assistant.  It combines:

* structured event filters (entity / source / time / amount / direction / place)
* BM25 over event text and document chunks
* one-hop entity-graph expansion
* recency, amount and risk boosts
* citation construction (evidence id + file + timestamp + snippet)

The result is a ranked list of hits plus a de-duplicated citation list.  Every
answer the assistant produces is built from these two objects, which is what
keeps the conversation grounded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .index import EvidenceIndex, _safe_ts
from .models import Citation, QueryPlan


@dataclass
class RetrievalResult:
    hits: List[dict] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    total: int = 0
    query_plan: Optional[QueryPlan] = None

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "citations": [c.to_dict() for c in self.citations],
            "total": self.total,
            "query_plan": self.query_plan.to_dict() if self.query_plan else None,
        }


class HybridRetriever:
    def __init__(self, index: EvidenceIndex):
        self.index = index

    # -- search ------------------------------------------------------------
    def search(self, plan: QueryPlan, top_k: Optional[int] = None) -> RetrievalResult:
        top_k = max(1, min(int(top_k or plan.limit or 20), 100))
        hits = self.index.search(plan, top_k=top_k)
        citations: List[Citation] = []
        seen = set()
        for hit in hits:
            citation = self.citation_for_hit(hit)
            if citation is None:
                continue
            key = (citation.evidence_id, citation.event_id, citation.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            citations.append(citation)
        return RetrievalResult(hits=hits, citations=citations,
                               total=len(hits), query_plan=plan)

    def citation_for_hit(self, hit: dict) -> Optional[Citation]:
        kind = hit.get("kind")
        if kind == "event":
            idx = hit.get("_row_index")
            if idx is None or self.index.events.empty or idx not in self.index.events.index:
                return None
            return self.index.citation_for_event(self.index.events.loc[idx])
        if kind == "chunk":
            idx = hit.get("_chunk_index")
            if idx is None or idx < 0 or idx >= len(self.index.chunks):
                return None
            return self.index.citation_for_chunk(self.index.chunks[idx])
        return None

    # -- call/transfer correlation ----------------------------------------
    def correlate_calls(self, plan: QueryPlan, tol_minutes: int = 25,
                        top_k: int = 10) -> dict:
        """Find calls/posts shortly before a transfer to the same party.

        This is the conversational counterpart of the AML time-correlation
        model, but it returns the actual event pairs and citations so the
        investigator can inspect them.
        """
        df = self.index.events
        if df.empty:
            return {"pairs": [], "citations": [], "n": 0, "tol_minutes": tol_minutes}

        # restrict to the queried entity when present; otherwise all entities
        sub = df
        if plan.entity_ids:
            sub = df[df["entity_id"].isin(plan.entity_ids) | df["counterparty_id"].isin(plan.entity_ids)]
        bank = sub[sub["source"] == "bank"].dropna(subset=["event_ts", "entity_id", "counterparty_id"])
        comm = sub[sub["source"].isin(("cdr", "social"))].dropna(subset=["event_ts", "entity_id", "counterparty_id"])
        if bank.empty or comm.empty:
            return {"pairs": [], "citations": [], "n": 0, "tol_minutes": tol_minutes}

        # apply the query's time/amount filters to the bank side
        if plan.time_range.start:
            start = _safe_ts(plan.time_range.start)
            if start is not None:
                bank = bank[bank["event_ts"] >= start]
        if plan.time_range.end:
            end = _safe_ts(plan.time_range.end)
            if end is not None:
                bank = bank[bank["event_ts"] <= end]
        if plan.amount_range.min is not None:
            bank = bank[bank["event_amount"] >= float(plan.amount_range.min)]
        if plan.amount_range.max is not None:
            bank = bank[bank["event_amount"] <= float(plan.amount_range.max)]

        pairs: List[dict] = []
        citations: List[Citation] = []
        tol = pd.Timedelta(minutes=max(1, int(tol_minutes)))
        for _, brow in bank.iterrows():
            bts = brow["event_ts"]
            if pd.isna(bts):
                continue
            same_party = comm[comm["counterparty_id"] == brow["counterparty_id"]]
            if same_party.empty:
                continue
            # communications before the transfer, within tolerance
            near = same_party[
                (same_party["event_ts"] <= bts) & (same_party["event_ts"] >= bts - tol)
            ]
            if near.empty:
                continue
            c_row = near.sort_values("event_ts").iloc[-1]
            delta = (bts - c_row["event_ts"]).total_seconds() / 60.0
            c_cit = self.index.citation_for_event(c_row)
            b_cit = self.index.citation_for_event(brow)
            citations.extend([c_cit, b_cit])
            pairs.append({
                "entity_id": str(brow["entity_id"]),
                "entity_name": self.index.entity_label(str(brow["entity_id"])),
                "counterparty_id": str(brow["counterparty_id"]),
                "counterparty_name": self.index.entity_label(str(brow["counterparty_id"])),
                "communication": {
                    "event_id": str(c_row["event_id"]),
                    "source": str(c_row["source"]),
                    "timestamp": c_cit.timestamp,
                    "snippet": c_cit.snippet,
                },
                "transfer": {
                    "event_id": str(brow["event_id"]),
                    "amount": float(brow["event_amount"]) if not pd.isna(brow["event_amount"]) else None,
                    "timestamp": b_cit.timestamp,
                    "snippet": b_cit.snippet,
                },
                "delta_minutes": round(float(delta), 1),
            })
        pairs.sort(key=lambda p: p["delta_minutes"])
        # dedupe citations
        seen = set()
        unique: List[Citation] = []
        for c in citations:
            key = (c.evidence_id, c.event_id, c.chunk_id)
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return {
            "pairs": pairs[:top_k],
            "citations": unique,
            "n": len(pairs),
            "tol_minutes": tol_minutes,
        }

    # -- convenience -------------------------------------------------------
    def entity_citations(self, entity_id: str, limit: int = 5) -> List[Citation]:
        if self.index.events.empty:
            return []
        sub = self.index.events[
            (self.index.events["entity_id"] == entity_id)
            | (self.index.events["counterparty_id"] == entity_id)
        ]
        if sub.empty:
            return []
        sub = sub.sort_values("event_ts", ascending=False).head(limit)
        return [self.index.citation_for_event(row) for _, row in sub.iterrows()]

    def finding_citations(self, entity_ids: List[str], limit: int = 5) -> List[Citation]:
        if not entity_ids:
            return []
        ids = set(entity_ids)
        sub = self.index.events[
            self.index.events["entity_id"].isin(ids)
            | self.index.events["counterparty_id"].isin(ids)
        ]
        if sub.empty:
            return []
        # prefer large / recent events
        sub = sub.assign(sort_amount=sub["event_amount"].fillna(0)).sort_values(
            ["sort_amount", "event_ts"], ascending=False
        ).head(limit)
        return [self.index.citation_for_event(row) for _, row in sub.iterrows()]

"""Evidence index: BM25 text search + structured event filters + entity graph.

The index is built from two streams:

* **events** — normalized structured records (bank/CDR/social) produced by the
  existing ``aml`` schema pipeline, carrying ``evidence_id`` provenance;
* **chunks** — text spans from emails, PDFs, DOCX, notes and chat logs.

Both are searchable.  Structured questions use event filters + BM25 over the
event text; document questions use BM25 over chunks.  The entity graph links
people/accounts/phones across both streams and supports relationship queries.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .i18n import STOPWORDS, normalize_text
from .models import Case, Chunk, Citation, Entity, QueryPlan

# universal stopword set for search (English is the common denominator; other
# languages' content words are deliberately kept so multilingual queries work)
_SEARCH_STOP = {normalize_text(w) for w in STOPWORDS.get("en", set())}
_SEARCH_STOP |= {"the", "a", "an", "of", "to", "in", "on", "for", "and", "or",
                 "is", "are", "was", "were", "be", "been", "with", "from",
                 "that", "this", "these", "those", "it", "its", "as", "at",
                 "by", "we", "you", "they", "he", "she", "his", "her", "their"}

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Lowercase, accent-fold, tokenize, drop stopwords and 1-char tokens."""
    low = normalize_text(str(text or ""))
    out: List[str] = []
    for tok in _TOKEN_RE.findall(low):
        if len(tok) < 2 or tok in _SEARCH_STOP:
            continue
        out.append(tok)
    return out


class BM25:
    """Okapi BM25 over a list of tokenized documents."""

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = [list(d) for d in docs]
        self.n = len(self.docs)
        self.k1 = k1
        self.b = b
        self.avgdl = (sum(len(d) for d in self.docs) / self.n) if self.n else 0.0
        self.doc_len = [len(d) for d in self.docs]
        self.tf: List[Counter] = [Counter(d) for d in self.docs]
        df: Counter = Counter()
        for doc in self.docs:
            df.update(set(doc))
        self.idf = {
            term: math.log(1.0 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query_tokens: Sequence[str], top_k: int = 50) -> List[Tuple[int, float]]:
        if not self.n or not query_tokens:
            return []
        scores: List[Tuple[int, float]] = []
        for i, tf in enumerate(self.tf):
            score = 0.0
            norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1.0))
            for term in query_tokens:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                score += self.idf.get(term, 0.0) * (freq * (self.k1 + 1.0)) / (freq + norm)
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda kv: (-kv[1], kv[0]))
        return scores[:top_k]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _safe_ts(value) -> Optional[pd.Timestamp]:
    if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("UTC")
    except Exception:
        return None


def _iso(value) -> str:
    ts = _safe_ts(value)
    return ts.isoformat() if ts is not None else ""


def _money(value) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return ""


def _short(text: str, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class EvidenceIndex:
    def __init__(self, case: Case, events: Optional[pd.DataFrame] = None,
                 chunks: Optional[List[Chunk]] = None,
                 entities: Optional[Dict[str, Entity]] = None,
                 aliases: Optional[Dict[str, str]] = None):
        self.case = case
        self.events = events.copy() if events is not None else pd.DataFrame()
        self.chunks = list(chunks or [])
        self.entities = dict(entities or case.entities)
        self.aliases = dict(aliases or {})
        self._prepare_events()
        self._prepare_chunks()
        self._build_graph()

    # -- preparation -------------------------------------------------------
    def _prepare_events(self) -> None:
        if self.events is None or self.events.empty:
            self.events = pd.DataFrame(columns=[
                "event_id", "evidence_id", "source", "event_type", "timestamp",
                "entity_id", "counterparty_id", "amount", "direction", "location",
                "duration", "search_text", "event_ts", "event_amount",
            ])
            self.event_bm25 = BM25([])
            return
        df = self.events
        if "event_id" not in df.columns:
            df["event_id"] = [f"EV{i:06d}" for i in range(1, len(df) + 1)]
        df["event_id"] = df["event_id"].astype(str)
        for col in ("evidence_id", "source", "event_type", "entity_id",
                    "counterparty_id", "direction", "location"):
            if col not in df.columns:
                df[col] = ""
            # avoid pandas 4's float64 + str fill warning: normalize to object
            df[col] = df[col].astype("object").where(df[col].notna(), "").astype(str)
        if "timestamp" in df.columns:
            df["event_ts"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        else:
            df["event_ts"] = pd.NaT
        if "amount" in df.columns:
            df["event_amount"] = pd.to_numeric(df["amount"], errors="coerce")
        else:
            df["event_amount"] = np.nan
        labels = {eid: ent.display_name for eid, ent in self.entities.items()}
        texts = []
        for row in df.itertuples(index=False):
            actor = labels.get(getattr(row, "entity_id", ""), getattr(row, "entity_id", ""))
            cp = labels.get(getattr(row, "counterparty_id", ""), getattr(row, "counterparty_id", ""))
            parts = [
                getattr(row, "source", ""), getattr(row, "event_type", ""),
                getattr(row, "direction", ""), getattr(row, "location", ""),
                actor, cp, getattr(row, "entity_id", ""), getattr(row, "counterparty_id", ""),
                _money(getattr(row, "event_amount", None)), _iso(getattr(row, "event_ts", None)),
                getattr(row, "evidence_id", ""),
            ]
            texts.append(" ".join(str(p) for p in parts if p not in (None, "", "nan")))
        df["search_text"] = texts
        self.events = df
        self.event_bm25 = BM25([tokenize(t) for t in texts])

    def _prepare_chunks(self) -> None:
        self.chunk_tokens = [
            tokenize(" ".join([
                c.text or "",
                self._evidence_filename(c.evidence_id),
                c.source_type or "",
                " ".join(self.entity_label(e) for e in c.entity_ids),
                c.location or "",
            ]))
            for c in self.chunks
        ]
        self.chunk_bm25 = BM25(self.chunk_tokens)
        self._chunks_by_id = {c.chunk_id: c for c in self.chunks}

    def _build_graph(self) -> None:
        self.adj: Dict[str, Dict[str, dict]] = defaultdict(dict)
        if self.events.empty:
            return
        for row in self.events.itertuples(index=False):
            a = getattr(row, "entity_id", "")
            b = getattr(row, "counterparty_id", "")
            if not a or not b or a == "nan" or b == "nan":
                continue
            for src, dst in ((a, b), (b, a)):
                edge = self.adj[src].setdefault(dst, {
                    "count": 0, "sources": set(), "amount": 0.0,
                    "first": None, "last": None,
                })
                edge["count"] += 1
                edge["sources"].add(getattr(row, "source", ""))
                amt = getattr(row, "event_amount", None)
                if amt is not None and not (isinstance(amt, float) and math.isnan(amt)):
                    edge["amount"] += float(amt)
                ts = getattr(row, "event_ts", None)
                if ts is not None and not pd.isna(ts):
                    if edge["first"] is None or ts < edge["first"]:
                        edge["first"] = ts
                    if edge["last"] is None or ts > edge["last"]:
                        edge["last"] = ts

    # -- labels / citations ------------------------------------------------
    def entity_label(self, entity_id: str) -> str:
        ent = self.entities.get(str(entity_id))
        return ent.display_name if ent else str(entity_id)

    def _evidence_filename(self, evidence_id: str) -> str:
        item = self.case.evidence_by_id(str(evidence_id))
        return item.filename if item else ""

    def citation_for_event(self, row) -> Citation:
        evidence_id = str(getattr(row, "evidence_id", "") or "")
        entity_ids = [x for x in (str(getattr(row, "entity_id", "") or ""),
                                  str(getattr(row, "counterparty_id", "") or "")) if x and x != "nan"]
        actor = self.entity_label(getattr(row, "entity_id", ""))
        cp = self.entity_label(getattr(row, "counterparty_id", ""))
        amount = _money(getattr(row, "event_amount", None))
        direction = str(getattr(row, "direction", "") or "")
        source = str(getattr(row, "source", "") or "")
        location = str(getattr(row, "location", "") or "")
        bits = [source, direction, amount, f"→ {cp}" if cp else "", f"@ {location}" if location else ""]
        snippet = f"{actor}: " + " ".join(b for b in bits if b)
        return Citation(
            evidence_id=evidence_id,
            event_id=str(getattr(row, "event_id", "") or ""),
            file=self._evidence_filename(evidence_id),
            timestamp=_iso(getattr(row, "event_ts", None)),
            snippet=_short(snippet, 240),
            source_type=source,
            entity_ids=entity_ids,
        )

    def citation_for_chunk(self, chunk: Chunk) -> Citation:
        item = self.case.evidence_by_id(chunk.evidence_id)
        return Citation(
            evidence_id=chunk.evidence_id,
            chunk_id=chunk.chunk_id,
            file=item.filename if item else "",
            timestamp=chunk.timestamp or "",
            snippet=_short(chunk.text, 260),
            source_type=chunk.source_type,
            entity_ids=list(chunk.entity_ids),
        )

    # -- stats -------------------------------------------------------------
    def stats(self) -> dict:
        timestamps: List[pd.Timestamp] = []
        if not self.events.empty:
            timestamps.extend([t for t in self.events["event_ts"].dropna().tolist()])
        for chunk in self.chunks:
            ts = _safe_ts(chunk.timestamp)
            if ts is not None:
                timestamps.append(ts)
        start = min(timestamps).isoformat() if timestamps else ""
        end = max(timestamps).isoformat() if timestamps else ""
        sources: Counter = Counter()
        for _, row in self.events.iterrows():
            sources[str(row.get("source", ""))] += 1
        for chunk in self.chunks:
            sources[str(chunk.source_type)] += 1
        return {
            "events": int(len(self.events)),
            "chunks": len(self.chunks),
            "entities": len(self.entities),
            "evidence": len(self.case.evidence),
            "sources": dict(sources),
            "start": start,
            "end": end,
        }

    # -- filters -----------------------------------------------------------
    def _event_mask(self, plan: QueryPlan) -> pd.Series:
        df = self.events
        mask = pd.Series(True, index=df.index)
        if df.empty:
            return mask
        if plan.entity_ids:
            ids = set(plan.entity_ids)
            mask &= df["entity_id"].isin(ids) | df["counterparty_id"].isin(ids)
        if plan.sources:
            # structured sources only; email/chat/document chunks are handled
            # by search_chunks, but a query can ask for both.
            structured = [s for s in plan.sources if s in ("bank", "cdr", "social")]
            if structured:
                mask &= df["source"].isin(structured)
        if plan.event_types:
            mask &= df["event_type"].isin(plan.event_types)
        if plan.time_range.start:
            start = _safe_ts(plan.time_range.start)
            if start is not None:
                mask &= df["event_ts"].notna() & (df["event_ts"] >= start)
        if plan.time_range.end:
            end = _safe_ts(plan.time_range.end)
            if end is not None:
                mask &= df["event_ts"].notna() & (df["event_ts"] <= end)
        if plan.amount_range.min is not None:
            mask &= df["event_amount"].notna() & (df["event_amount"] >= float(plan.amount_range.min))
        if plan.amount_range.max is not None:
            mask &= df["event_amount"].notna() & (df["event_amount"] <= float(plan.amount_range.max))
        if plan.direction:
            mask &= df["direction"].str.lower().str.contains(plan.direction, na=False)
        if plan.location_terms:
            loc = df["location"].fillna("").astype(str).str.lower()
            loc_mask = pd.Series(False, index=df.index)
            for term in plan.location_terms:
                loc_mask |= loc.str.contains(str(term).lower(), regex=False, na=False)
            mask &= loc_mask
        return mask

    def _chunk_mask(self, plan: QueryPlan) -> List[bool]:
        mask: List[bool] = []
        for chunk in self.chunks:
            ok = True
            if plan.entity_ids and not (set(plan.entity_ids) & set(chunk.entity_ids)):
                ok = False
            if ok and plan.sources and chunk.source_type not in plan.sources:
                # document/email/chat filters are exact; a bank query should
                # not return a PDF unless the user asked for documents too
                ok = False
            if ok and plan.time_range.start:
                ts = _safe_ts(chunk.timestamp)
                start = _safe_ts(plan.time_range.start)
                if ts is None or start is None or ts < start:
                    ok = False
            if ok and plan.time_range.end:
                ts = _safe_ts(chunk.timestamp)
                end = _safe_ts(plan.time_range.end)
                if ts is None or end is None or ts > end:
                    ok = False
            if ok and plan.location_terms:
                hay = f"{chunk.location or ''} {chunk.text or ''}".lower()
                if not any(str(t).lower() in hay for t in plan.location_terms):
                    ok = False
            mask.append(ok)
        return mask

    # -- search ------------------------------------------------------------
    def search_events(self, plan: QueryPlan, top_k: int = 20) -> List[dict]:
        if self.events.empty:
            return []
        mask = self._event_mask(plan)
        candidates = list(self.events.index[mask])
        if not candidates:
            return []
        bm25_scores: Dict[int, float] = {}
        if plan.keywords:
            bm25_scores = dict(self.event_bm25.search(plan.keywords, top_k=len(self.events)))
            candidates = [i for i in candidates if bm25_scores.get(i, 0.0) > 0]
        if not candidates:
            return []
        max_bm25 = max((bm25_scores.get(i, 0.0) for i in candidates), default=0.0) or 1.0
        ts_values = self.events.loc[candidates, "event_ts"].dropna()
        ts_min = ts_values.min() if not ts_values.empty else None
        ts_max = ts_values.max() if not ts_values.empty else None
        max_amount = float(self.events.loc[candidates, "event_amount"].max() or 0.0) or 1.0
        risk = (self.case.metadata or {}).get("risk_scores", {}) or {}
        ids = set(plan.entity_ids)

        hits: List[dict] = []
        for i in candidates:
            row = self.events.loc[i]
            score = 0.55 * (bm25_scores.get(i, 0.0) / max_bm25)
            actor = str(row.get("entity_id", ""))
            cp = str(row.get("counterparty_id", ""))
            if ids:
                if actor in ids:
                    score += 0.22
                if cp in ids:
                    score += 0.12
            ts = row.get("event_ts")
            if ts is not None and not pd.isna(ts) and ts_min is not None and ts_max is not None and ts_max > ts_min:
                score += 0.12 * float((ts - ts_min) / (ts_max - ts_min))
            amt = row.get("event_amount")
            if amt is not None and not pd.isna(amt) and amt > 0:
                score += 0.08 * float(math.log1p(amt) / math.log1p(max_amount))
            score += 0.15 * float(risk.get(actor, 0.0))
            if plan.sources and str(row.get("source", "")) in plan.sources:
                score += 0.05
            hits.append({
                "kind": "event",
                "id": str(row.get("event_id", "")),
                "score": round(min(score, 1.0), 4),
                "title": (f"{self.entity_label(actor)} → {self.entity_label(cp)}"
                          if cp and cp != "nan" else self.entity_label(actor)),
                "snippet": self.citation_for_event(row).snippet,
                "timestamp": _iso(ts),
                "source": str(row.get("source", "")),
                "entity_ids": [x for x in (actor, cp) if x and x != "nan"],
                "evidence_id": str(row.get("evidence_id", "")),
                "metadata": {
                    "event_type": str(row.get("event_type", "")),
                    "direction": str(row.get("direction", "")),
                    "amount": float(amt) if amt is not None and not pd.isna(amt) else None,
                    "location": str(row.get("location", "")),
                },
                "_row_index": int(i),
            })
        hits.sort(key=lambda h: (-h["score"], h["timestamp"]))
        return hits[:top_k]

    def search_chunks(self, plan: QueryPlan, top_k: int = 20) -> List[dict]:
        if not self.chunks:
            return []
        mask = self._chunk_mask(plan)
        candidates = [i for i, ok in enumerate(mask) if ok]
        if not candidates:
            return []
        bm25_scores: Dict[int, float] = {}
        if plan.keywords:
            bm25_scores = dict(self.chunk_bm25.search(plan.keywords, top_k=len(self.chunks)))
            candidates = [i for i in candidates if bm25_scores.get(i, 0.0) > 0]
        if not candidates:
            return []
        max_bm25 = max((bm25_scores.get(i, 0.0) for i in candidates), default=0.0) or 1.0
        ids = set(plan.entity_ids)
        hits: List[dict] = []
        for i in candidates:
            chunk = self.chunks[i]
            score = 0.70 * (bm25_scores.get(i, 0.0) / max_bm25)
            if ids and set(ids) & set(chunk.entity_ids):
                score += 0.20
            if plan.sources and chunk.source_type in plan.sources:
                score += 0.05
            hits.append({
                "kind": "chunk",
                "id": chunk.chunk_id,
                "score": round(min(score, 1.0), 4),
                "title": f"{chunk.source_type} · {self._evidence_filename(chunk.evidence_id)}",
                "snippet": _short(chunk.text, 300),
                "timestamp": chunk.timestamp or "",
                "source": chunk.source_type,
                "entity_ids": list(chunk.entity_ids),
                "evidence_id": chunk.evidence_id,
                "metadata": {
                    "line_start": chunk.line_start,
                    "line_end": chunk.line_end,
                    "location": chunk.location,
                },
                "_chunk_index": i,
            })
        hits.sort(key=lambda h: (-h["score"], h["timestamp"]))
        return hits[:top_k]

    # -- graph -------------------------------------------------------------
    def neighbors(self, entity_ids: Iterable[str], hops: int = 1) -> Set[str]:
        seen: Set[str] = set(entity_ids)
        frontier = set(entity_ids)
        for _ in range(max(1, hops)):
            nxt: Set[str] = set()
            for eid in frontier:
                nxt.update(self.adj.get(eid, {}).keys())
            nxt -= seen
            seen |= nxt
            frontier = nxt
        return seen

    def relationship(self, a: str, b: str) -> dict:
        a, b = str(a), str(b)
        direct = self.adj.get(a, {}).get(b)
        common = sorted(set(self.adj.get(a, {})) & set(self.adj.get(b, {})))
        path = self._shortest_path(a, b)
        shared_chunks = [
            c for c in self.chunks if a in c.entity_ids and b in c.entity_ids
        ]
        return {
            "a": a,
            "b": b,
            "a_name": self.entity_label(a),
            "b_name": self.entity_label(b),
            "direct": ({
                "count": direct["count"],
                "sources": sorted(x for x in direct["sources"] if x),
                "amount": round(float(direct["amount"]), 2),
                "first": direct["first"].isoformat() if direct["first"] is not None else "",
                "last": direct["last"].isoformat() if direct["last"] is not None else "",
            } if direct else None),
            "common": [{"entity_id": e, "name": self.entity_label(e)} for e in common[:10]],
            "path": [{"entity_id": e, "name": self.entity_label(e)} for e in path],
            "shared_chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "evidence_id": c.evidence_id,
                    "file": self._evidence_filename(c.evidence_id),
                    "timestamp": c.timestamp or "",
                    "snippet": _short(c.text, 220),
                    "citation": self.citation_for_chunk(c).to_dict(),
                }
                for c in shared_chunks[:5]
            ],
        }

    def _shortest_path(self, a: str, b: str, max_depth: int = 4) -> List[str]:
        if a == b:
            return [a]
        seen = {a}
        queue = deque([(a, [a])])
        while queue:
            node, path = queue.popleft()
            if len(path) > max_depth:
                continue
            for nxt in self.adj.get(node, {}):
                if nxt in seen:
                    continue
                if nxt == b:
                    return path + [nxt]
                seen.add(nxt)
                queue.append((nxt, path + [nxt]))
        return []

    # -- aggregate / timeline / profile -----------------------------------
    def aggregate(self, plan: QueryPlan) -> dict:
        if self.events.empty:
            return {"count": 0, "total": 0.0, "avg": 0.0, "min": 0.0, "max": 0.0,
                    "by_source": {}, "by_direction": {}, "by_entity": {}, "hits": []}
        mask = self._event_mask(plan)
        sub = self.events[mask]
        if plan.keywords:
            scores = dict(self.event_bm25.search(plan.keywords, top_k=len(self.events)))
            sub = sub[[i for i in sub.index if scores.get(i, 0.0) > 0]]
        amounts = pd.to_numeric(sub.get("event_amount"), errors="coerce").dropna()
        by_source = sub["source"].value_counts().to_dict() if not sub.empty else {}
        by_direction = sub["direction"].value_counts().to_dict() if not sub.empty else {}
        by_entity = (
            sub["entity_id"].value_counts().head(10).to_dict() if not sub.empty else {}
        )
        hits = self.search_events(plan, top_k=min(plan.limit, 10))
        return {
            "count": int(len(sub)),
            "total": float(amounts.sum()) if not amounts.empty else 0.0,
            "avg": float(amounts.mean()) if not amounts.empty else 0.0,
            "min": float(amounts.min()) if not amounts.empty else 0.0,
            "max": float(amounts.max()) if not amounts.empty else 0.0,
            "by_source": {str(k): int(v) for k, v in by_source.items()},
            "by_direction": {str(k): int(v) for k, v in by_direction.items()},
            "by_entity": {
                self.entity_label(str(k)): int(v) for k, v in by_entity.items()
            },
            "hits": hits,
        }

    def timeline(self, plan: QueryPlan, limit: int = 50) -> List[dict]:
        event_hits = self.search_events(plan, top_k=max(limit, 20))
        chunk_hits = self.search_chunks(plan, top_k=max(limit, 20))
        combined = [h for h in event_hits + chunk_hits if h.get("timestamp")]
        combined.sort(key=lambda h: h.get("timestamp", ""))
        return combined[:limit]

    def entity_profile(self, entity_id: str) -> dict:
        eid = str(entity_id)
        ent = self.entities.get(eid)
        actor_events = self.events[self.events["entity_id"] == eid] if not self.events.empty else pd.DataFrame()
        cp_events = self.events[self.events["counterparty_id"] == eid] if not self.events.empty else pd.DataFrame()
        chunk_mentions = [c for c in self.chunks if eid in c.entity_ids]
        timestamps = []
        if not actor_events.empty:
            timestamps.extend(actor_events["event_ts"].dropna().tolist())
        if not cp_events.empty:
            timestamps.extend(cp_events["event_ts"].dropna().tolist())
        for chunk in chunk_mentions:
            ts = _safe_ts(chunk.timestamp)
            if ts is not None:
                timestamps.append(ts)
        sources = set()
        if not actor_events.empty:
            sources |= set(actor_events["source"].astype(str))
        if not cp_events.empty:
            sources |= set(cp_events["source"].astype(str))
        sources |= {c.source_type for c in chunk_mentions}

        amounts = (
            pd.to_numeric(actor_events.get("event_amount"), errors="coerce").dropna()
            if not actor_events.empty else pd.Series(dtype=float)
        )
        if amounts.empty and chunk_mentions:
            text_amounts = []
            for chunk in chunk_mentions:
                text_amounts.extend((chunk.metadata or {}).get("amounts", []) or [])
            amounts = pd.Series(text_amounts, dtype=float)

        # counterparts: structured edges first, then co-mentions in text
        cp_counter: Counter = Counter()
        if not actor_events.empty:
            for value, count in actor_events["counterparty_id"].replace("", np.nan).dropna().value_counts().items():
                cp_counter[str(value)] += int(count)
        for chunk in chunk_mentions:
            for other in chunk.entity_ids:
                if other != eid:
                    cp_counter[str(other)] += 1
        top_cp = cp_counter.most_common(10)

        loc_counter: Counter = Counter()
        if not actor_events.empty:
            for value, count in actor_events["location"].replace("", np.nan).dropna().value_counts().items():
                loc_counter[str(value)] += int(count)
        for chunk in chunk_mentions:
            if chunk.location:
                loc_counter[str(chunk.location)] += 1

        findings = [f for f in self.case.findings if eid in f.entity_ids]
        citations: List[Citation] = []
        if not actor_events.empty:
            recent = actor_events.sort_values("event_ts", ascending=False).head(5)
            citations.extend(self.citation_for_event(row) for _, row in recent.iterrows())
        for chunk in sorted(chunk_mentions, key=lambda c: c.timestamp or "", reverse=True)[:5]:
            citations.append(self.citation_for_chunk(chunk))
        seen = set()
        unique_citations: List[Citation] = []
        for c in citations:
            key = (c.evidence_id, c.event_id, c.chunk_id)
            if key not in seen:
                seen.add(key)
                unique_citations.append(c)

        n_structured = int(len(actor_events))
        n_text = int(len(chunk_mentions))
        return {
            "entity_id": eid,
            "name": ent.display_name if ent else eid,
            "names": ent.names if ent else [],
            "identifiers": ent.identifiers if ent else [],
            "roles": ent.roles if ent else [],
            "risk_score": ent.risk_score if ent else 0.0,
            "risk_factors": ent.risk_factors if ent else [],
            "n_events": n_structured + n_text,
            "n_structured_events": n_structured,
            "n_text_mentions": n_text,
            "n_counterparty_events": int(len(cp_events)),
            "sources": sorted(s for s in sources if s),
            "first_seen": min(timestamps).isoformat() if timestamps else "",
            "last_seen": max(timestamps).isoformat() if timestamps else "",
            "total_amount": float(amounts.sum()) if not amounts.empty else 0.0,
            "avg_amount": float(amounts.mean()) if not amounts.empty else 0.0,
            "max_amount": float(amounts.max()) if not amounts.empty else 0.0,
            "top_counterparts": [
                {"entity_id": str(k), "name": self.entity_label(str(k)), "count": int(v)}
                for k, v in top_cp
            ],
            "locations": {str(k): int(v) for k, v in loc_counter.most_common(10)},
            "findings": [f.to_dict() for f in findings],
            "citations": [c.to_dict() for c in unique_citations],
            "chunks": [
                {"chunk_id": c.chunk_id, "evidence_id": c.evidence_id,
                 "snippet": _short(c.text, 220)}
                for c in chunk_mentions[:5]
            ],
        }

    def top_entities(self, n: int = 10) -> List[dict]:
        rows = []
        for eid, ent in self.entities.items():
            rows.append({
                "entity_id": eid,
                "name": ent.display_name,
                "risk_score": float(ent.risk_score),
                "n_events": int(ent.n_events),
                "sources": list(ent.sources),
            })
        rows.sort(key=lambda r: (-r["risk_score"], -r["n_events"], r["name"]))
        return rows[:n]

    def evidence_summary(self) -> List[dict]:
        out = []
        for item in self.case.evidence:
            out.append({
                "evidence_id": item.evidence_id,
                "filename": item.filename,
                "source_type": item.source_type,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
                "n_rows": item.n_rows,
                "n_chunks": item.n_chunks,
                "acquired_at": item.acquired_at,
                "custodian": item.custodian,
                "warnings": list(item.warnings),
            })
        return out

    # -- generic -----------------------------------------------------------
    def search(self, plan: QueryPlan, top_k: int = 20) -> List[dict]:
        """Hybrid search: events + chunks + graph expansion, deduped."""
        event_hits = self.search_events(plan, top_k=top_k)
        chunk_hits = self.search_chunks(plan, top_k=top_k)
        hits = event_hits + chunk_hits

        # graph expansion: if the query names entities, include events involving
        # their one-hop neighbours at a small score so relationship context is
        # not lost when the direct entity has little activity.
        if plan.entity_ids and self.adj:
            neighbours = self.neighbors(plan.entity_ids, hops=1) - set(plan.entity_ids)
            if neighbours:
                expanded = QueryPlan.from_dict(plan.to_dict())
                expanded.entity_ids = list(neighbours)
                extra = self.search_events(expanded, top_k=max(5, top_k // 3))
                for hit in extra:
                    hit = dict(hit)
                    hit["score"] = round(hit["score"] * 0.55, 4)
                    hit["metadata"] = dict(hit.get("metadata") or {})
                    hit["metadata"]["graph_expanded"] = True
                    hits.append(hit)

        # dedupe by (kind, id)
        seen = set()
        deduped = []
        for hit in sorted(hits, key=lambda h: -h["score"]):
            key = (hit["kind"], hit["id"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hit)
        return deduped[:top_k]

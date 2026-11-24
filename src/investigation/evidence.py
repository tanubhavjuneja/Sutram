"""Evidence ingestion: structured records + documents + communications.

Structured files (bank/CDR/social CSV/JSON/JSONL/TXT) are parsed by the
existing ``aml`` schema pipeline so the investigation layer inherits its
runtime schema detection, normalization and entity resolution.

Unstructured files are parsed here:

* ``.eml`` / ``.mbox`` — headers, body, attachments metadata
* ``.pdf`` — text per page (pypdf)
* ``.docx`` — document.xml text (stdlib zip/xml)
* ``.md`` / ``.rst`` / ``.txt`` / ``.log`` — text, with chat-log detection
* images — EXIF/metadata when Pillow is available, otherwise file metadata

Every text span becomes a :class:`Chunk` with line/page provenance and entity
mentions.  Entity mentions are linked to the case's resolved entities by
alias/identifier; repeated unknown identifiers (emails, phones, handles,
account numbers) become text-only entities so they are still searchable.
"""
from __future__ import annotations

import email
import json
import mimetypes
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .case_store import CaseStore
from .i18n import normalize_text
from .models import Case, Chunk, Entity, EvidenceItem, utc_now

# ---------------------------------------------------------------------------
# file classification
# ---------------------------------------------------------------------------
STRUCTURED_EXT = {".csv", ".csv.gz", ".tsv", ".json", ".jsonl", ".ndjson"}
TEXT_EXT = {".txt", ".log", ".md", ".rst", ".text"}
EMAIL_EXT = {".eml", ".mbox"}
PDF_EXT = {".pdf"}
DOCX_EXT = {".docx"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".heic"}

_ISO_RE = re.compile(
    r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})"
    r"(?:[T ](\d{1,2}):(\d{2})(?::(\d{2}))?)?\b"
)
_DMY_RE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](20\d{2})\b")
_CHAT_LINE_RE = re.compile(
    r"^\s*(?:\[)?(?P<ts>20\d{2}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}(?::\d{2})?)?(?:\])?\s*"
    r"(?:[-–]\s*)?(?P<speaker>[A-Z][\w .'’-]{1,40}):\s*(?P<text>.+)$"
)
_IDENTIFIER_RE = re.compile(
    r"(?P<email>\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b)"
    r"|(?P<phone>(?<!\d)\+?\d{10,15}(?!\d))"
    r"|(?P<handle>@[\w.]+)"
    r"|(?P<account>\bACCT[-_]?[A-Za-z0-9]{3,}\b)",
    re.IGNORECASE,
)
_AMOUNT_RE = re.compile(
    r"(?P<currency>[$€£₹])\s?(?P<num>[\d][\d,]*(?:\.\d{1,2})?)"
    r"|(?P<num2>[\d][\d,]*(?:\.\d{1,2})?)\s?(?P<code>USD|EUR|GBP|INR|CAD|AUD|CHF|JPY)"
)


def classify_source_type(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in EMAIL_EXT:
        return "email"
    if suffix in PDF_EXT:
        return "document"
    if suffix in DOCX_EXT:
        return "document"
    if suffix in IMAGE_EXT:
        return "image"
    if suffix in (".md", ".rst"):
        return "note"
    if suffix in STRUCTURED_EXT or suffix in TEXT_EXT:
        return "other"  # resolved after schema detection / text inspection
    return "other"


# ---------------------------------------------------------------------------
# parsing result
# ---------------------------------------------------------------------------
@dataclass
class ParsedEvidence:
    item: EvidenceItem
    events: Optional[pd.DataFrame] = None
    chunks: List[Chunk] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    detected_source: str = "other"


@dataclass
class IngestResult:
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    chunks: List[Chunk] = field(default_factory=list)
    entities: Dict[str, Entity] = field(default_factory=dict)
    aliases: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    per_evidence: List[dict] = field(default_factory=list)
    added_evidence_ids: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------
def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="replace")
    except OSError:
        return ""


def extract_timestamp(text: str) -> Optional[str]:
    """First plausible timestamp in free text, as ISO-8601 UTC."""
    if not text:
        return None
    m = _ISO_RE.search(text)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        hour = int(m.group(4) or 0)
        minute = int(m.group(5) or 0)
        second = int(m.group(6) or 0)
        try:
            return datetime(year, month, day, hour, minute, second,
                            tzinfo=timezone.utc).isoformat()
        except ValueError:
            pass
    m = _DMY_RE.search(text)
    if m:
        a, b, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        day, month = (a, b) if a > 12 else (b, a) if b > 12 else (a, b)
        try:
            return datetime(year, month, day, tzinfo=timezone.utc).isoformat()
        except ValueError:
            return None
    return None


def extract_amounts(text: str) -> List[float]:
    out: List[float] = []
    for m in _AMOUNT_RE.finditer(text or ""):
        raw = m.group("num") or m.group("num2") or ""
        raw = raw.replace(",", "")
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150,
               line_offset: int = 1) -> List[Tuple[str, int, int]]:
    """Split text into overlapping chunks with line ranges.

    Paragraph boundaries are preferred; very long paragraphs are split at
    sentence boundaries.  The overlap keeps a fact that straddles a boundary
    retrievable from both chunks.
    """
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return []
    lines = text.split("\n")
    chunks: List[Tuple[str, int, int]] = []
    buf: List[str] = []
    buf_start = line_offset
    buf_len = 0

    def flush(end_line: int) -> None:
        nonlocal buf, buf_start, buf_len
        if not buf:
            return
        chunk = "\n".join(buf).strip()
        if chunk:
            chunks.append((chunk, buf_start, end_line))
        if overlap > 0 and chunk:
            tail = chunk[-overlap:]
            # start the next chunk with a clean tail at a word boundary
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            buf = [tail] if tail.strip() else []
            buf_len = len(tail)
            buf_start = max(buf_start, end_line - 1)
        else:
            buf = []
            buf_len = 0
            buf_start = end_line + 1

    for i, line in enumerate(lines, start=line_offset):
        if not buf:
            buf_start = i
        if len(line) > max_chars:
            # split long line into sentence-ish pieces
            for piece in re.split(r"(?<=[.!?])\s+", line):
                if buf_len + len(piece) > max_chars and buf:
                    flush(i)
                buf.append(piece)
                buf_len += len(piece) + 1
            continue
        if buf_len + len(line) + 1 > max_chars and buf:
            flush(i - 1)
        buf.append(line)
        buf_len += len(line) + 1
    if buf:
        chunk = "\n".join(buf).strip()
        if chunk:
            chunks.append((chunk, buf_start, len(lines) + line_offset - 1))
    return chunks


def _parse_email_date(value: str) -> Optional[str]:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return extract_timestamp(value)


# ---------------------------------------------------------------------------
# parsers
# ---------------------------------------------------------------------------
def parse_email(path: Path) -> Tuple[str, Dict, List[str]]:
    """Return (text, metadata, warnings) for an .eml file."""
    warnings: List[str] = []
    try:
        with open(path, "rb") as fh:
            msg = BytesParser(policy=policy.default).parse(fh)
    except Exception as exc:
        return "", {}, [f"email parse failed: {exc}"]
    subject = str(msg.get("subject", "") or "")
    from_ = str(msg.get("from", "") or "")
    to = str(msg.get("to", "") or "")
    cc = str(msg.get("cc", "") or "")
    date = str(msg.get("date", "") or "")
    message_id = str(msg.get("message-id", "") or "")
    body = ""
    try:
        part = msg.get_body(preferencelist=("plain", "html"))
        if part is not None:
            body = part.get_content()
            if part.get_content_type() == "text/html":
                body = re.sub(r"<[^>]+>", " ", body)
    except Exception as exc:
        warnings.append(f"email body extraction failed: {exc}")
    attachments = []
    try:
        for part in msg.iter_attachments():
            attachments.append({
                "filename": part.get_filename() or "",
                "content_type": part.get_content_type(),
                "size": len(part.get_payload(decode=True) or b""),
            })
    except Exception:
        pass
    text = "\n".join([
        f"Subject: {subject}",
        f"From: {from_}",
        f"To: {to}",
        f"Cc: {cc}" if cc else "",
        f"Date: {date}",
        "",
        body,
    ]).strip()
    metadata = {
        "subject": subject,
        "from": from_,
        "to": to,
        "cc": cc,
        "date": date,
        "message_id": message_id,
        "attachments": attachments,
        "timestamp": _parse_email_date(date),
    }
    return text, metadata, warnings


def parse_pdf(path: Path) -> Tuple[str, Dict, List[str]]:
    warnings: List[str] = []
    try:
        from pypdf import PdfReader
    except Exception:
        return "", {}, ["pypdf is not installed; PDF text extraction skipped"]
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        return "", {}, [f"PDF open failed: {exc}"]
    pages: List[str] = []
    for i, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            warnings.append(f"page {i} extraction failed: {exc}")
            text = ""
        if text.strip():
            pages.append(f"--- page {i} ---\n{text.strip()}")
    meta = {}
    try:
        info = reader.metadata or {}
        for key in ("/Title", "/Author", "/Subject", "/CreationDate", "/ModDate"):
            if key in info:
                meta[key.lstrip("/").lower()] = str(info[key])
    except Exception:
        pass
    text = "\n\n".join(pages)
    meta["pages"] = len(reader.pages)
    meta["timestamp"] = extract_timestamp(str(meta.get("creationdate", "")))
    return text, meta, warnings


def parse_docx(path: Path) -> Tuple[str, Dict, List[str]]:
    warnings: List[str] = []
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml").decode("utf-8", errors="replace")
    except Exception as exc:
        return "", {}, [f"DOCX open failed: {exc}"]
    # paragraph boundaries, then strip tags
    xml = re.sub(r"</w:p>", "\n", xml)
    text = re.sub(r"<[^>]+>", "", xml)
    text = (text.replace("&amp;", "&").replace("&lt;", "<")
                .replace("&gt;", ">").replace("&quot;", '"')
                .replace("&apos;", "'"))
    meta = {"timestamp": extract_timestamp(text)}
    return text, meta, warnings


def parse_image(path: Path) -> Tuple[str, Dict, List[str]]:
    warnings: List[str] = []
    meta: Dict = {"mime": mimetypes.guess_type(str(path))[0] or ""}
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        with Image.open(path) as img:
            meta["width"], meta["height"] = img.size
            exif = img.getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, str(tag_id))
                    if isinstance(value, (str, int, float)):
                        meta[str(tag)] = value
                    elif isinstance(value, bytes):
                        meta[str(tag)] = value.decode("utf-8", errors="replace")[:200]
    except Exception:
        warnings.append("Pillow not installed or EXIF unavailable; image metadata limited to file facts")
    return "", meta, warnings


def _parse_structured(item: EvidenceItem, path: Path, probe_cfg=None):
    """Try the aml structured path; return (df, detection, warnings) or None."""
    try:
        from aml.loaders import load_any
        from aml.schema_detect import detect_dataframe
    except Exception as exc:  # pragma: no cover - dependency issue
        return None, None, [f"aml import failed: {exc}"]
    try:
        loaded = load_any(str(path))
    except Exception as exc:
        return None, None, [f"load failed: {exc}"]
    if loaded.df is None or loaded.df.empty:
        return None, None, []
    try:
        det = detect_dataframe(loaded.df, file=str(path), probe_cfg=probe_cfg)
    except Exception as exc:
        return None, None, [f"schema detection failed: {exc}"]
    if det.source_type not in ("bank", "cdr", "social"):
        return None, det, []
    norm = det.normalized
    if norm is None or norm.empty:
        return None, det, list(det.warnings)
    norm = norm.copy()
    norm["evidence_id"] = item.evidence_id
    norm["row_index"] = range(len(norm))
    norm["source_file"] = item.filename
    return norm, det, list(det.warnings)


def _chat_chunks(text: str, evidence_id: str, start_id: int) -> Tuple[List[Chunk], Dict]:
    """Parse a chat-log-like text into timestamped chunks."""
    lines = text.replace("\r\n", "\n").split("\n")
    messages: List[Tuple[Optional[str], str, str]] = []
    for line in lines:
        m = _CHAT_LINE_RE.match(line)
        if not m:
            continue
        ts = extract_timestamp(m.group("ts") or "") or None
        speaker = (m.group("speaker") or "").strip()
        body = (m.group("text") or "").strip()
        if body:
            messages.append((ts, speaker, body))
    if len(messages) < 2:
        return [], {}
    chunks: List[Chunk] = []
    window = 8
    for i in range(0, len(messages), window):
        group = messages[i:i + window]
        ts = next((g[0] for g in group if g[0]), None)
        body = "\n".join(f"{speaker}: {msg}" for _ts, speaker, msg in group)
        chunks.append(Chunk(
            chunk_id=f"CH{start_id + len(chunks):06d}",
            evidence_id=evidence_id,
            text=body,
            timestamp=ts,
            source_type="chat",
            line_start=i + 1,
            line_end=min(i + window, len(messages)),
            metadata={"speakers": sorted({g[1] for g in group if g[1]})},
        ))
    return chunks, {"messages": len(messages), "format": "chat"}


def parse_evidence(item: EvidenceItem, path: Path, probe_cfg=None,
                   chunk_start: int = 1) -> ParsedEvidence:
    suffix = Path(path).suffix.lower()
    warnings: List[str] = []
    metadata: Dict = {}
    chunks: List[Chunk] = []
    events = None
    detected = item.source_type

    # Chat logs are a distinct evidence type: check before generic schema
    # detection so a "[timestamp] Name: message" file is not misread as a
    # social/post table.
    if suffix in (".txt", ".log"):
        raw_text = read_text(path)
        chat_chunks, chat_meta = _chat_chunks(raw_text, item.evidence_id, chunk_start)
        if chat_chunks:
            metadata.update(chat_meta)
            metadata["detected_source"] = "chat"
            return ParsedEvidence(item=item, events=None, chunks=chat_chunks,
                                  warnings=warnings, metadata=metadata,
                                  detected_source="chat")

    if suffix in STRUCTURED_EXT or suffix in (".txt", ".log"):
        events, det, warns = _parse_structured(item, path, probe_cfg=probe_cfg)
        warnings.extend(warns)
        if events is not None:
            detected = det.source_type if det else "other"
            metadata["detected_source"] = detected
            metadata["n_columns"] = int(det.n_rows) if det else 0
            return ParsedEvidence(item=item, events=events, chunks=[], warnings=warnings,
                                  metadata=metadata, detected_source=detected)
        # not structured: fall through to text
        if det is not None:
            warnings.append(
                f"structured detection produced no usable events "
                f"(detected {det.source_type}); treating as text"
            )

    if suffix in EMAIL_EXT:
        text, metadata, warns = parse_email(path)
        warnings.extend(warns)
        detected = "email"
        ts = metadata.get("timestamp")
        for chunk_text_value, start, end in chunk_text(text):
            chunks.append(Chunk(
                chunk_id=f"CH{chunk_start + len(chunks):06d}",
                evidence_id=item.evidence_id,
                text=chunk_text_value,
                timestamp=ts or extract_timestamp(chunk_text_value),
                source_type="email",
                line_start=start,
                line_end=end,
                metadata={
                    "subject": metadata.get("subject", ""),
                    "from": metadata.get("from", ""),
                    "to": metadata.get("to", ""),
                },
            ))
    elif suffix in PDF_EXT:
        text, metadata, warns = parse_pdf(path)
        warnings.extend(warns)
        detected = "document"
        ts = metadata.get("timestamp")
        for chunk_text_value, start, end in chunk_text(text):
            page_match = re.search(r"--- page (\d+) ---", chunk_text_value)
            chunks.append(Chunk(
                chunk_id=f"CH{chunk_start + len(chunks):06d}",
                evidence_id=item.evidence_id,
                text=chunk_text_value,
                timestamp=ts or extract_timestamp(chunk_text_value),
                source_type="document",
                line_start=start,
                line_end=end,
                metadata={"page": int(page_match.group(1)) if page_match else None},
            ))
    elif suffix in DOCX_EXT:
        text, metadata, warns = parse_docx(path)
        warnings.extend(warns)
        detected = "document"
        for chunk_text_value, start, end in chunk_text(text):
            chunks.append(Chunk(
                chunk_id=f"CH{chunk_start + len(chunks):06d}",
                evidence_id=item.evidence_id,
                text=chunk_text_value,
                timestamp=metadata.get("timestamp") or extract_timestamp(chunk_text_value),
                source_type="document",
                line_start=start,
                line_end=end,
            ))
    elif suffix in IMAGE_EXT:
        text, metadata, warns = parse_image(path)
        warnings.extend(warns)
        detected = "image"
        if text.strip():
            for chunk_text_value, start, end in chunk_text(text):
                chunks.append(Chunk(
                    chunk_id=f"CH{chunk_start + len(chunks):06d}",
                    evidence_id=item.evidence_id,
                    text=chunk_text_value,
                    timestamp=extract_timestamp(chunk_text_value),
                    source_type="image",
                    line_start=start,
                    line_end=end,
                ))
    else:
        # generic text / note / log
        text = read_text(path)
        if not text.strip():
            warnings.append("no extractable text; binary evidence retained by hash only")
            detected = "other"
        else:
            detected = "note" if suffix in (".md", ".rst") else "document"
            chat_chunks, chat_meta = _chat_chunks(text, item.evidence_id, chunk_start)
            if chat_chunks:
                chunks = chat_chunks
                metadata.update(chat_meta)
                detected = "chat"
            else:
                for chunk_text_value, start, end in chunk_text(text):
                    chunks.append(Chunk(
                        chunk_id=f"CH{chunk_start + len(chunks):06d}",
                        evidence_id=item.evidence_id,
                        text=chunk_text_value,
                        timestamp=extract_timestamp(chunk_text_value),
                        source_type=detected,
                        line_start=start,
                        line_end=end,
                    ))
    return ParsedEvidence(item=item, events=events, chunks=chunks,
                          warnings=warnings, metadata=metadata,
                          detected_source=detected)


# ---------------------------------------------------------------------------
# entity linking
# ---------------------------------------------------------------------------
def build_alias_index(entities: Dict[str, Entity]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for eid, ent in entities.items():
        for name in ent.names:
            if name:
                aliases[name] = eid
        for ident in ent.identifiers:
            if ident:
                aliases[ident] = eid
    return aliases


def _alias_regex(aliases: Iterable[str]):
    items = sorted({a for a in aliases if a}, key=len, reverse=True)
    if not items:
        return None
    return re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(a) for a in items) + r")(?!\w)",
        re.IGNORECASE,
    )


def extract_mentions(text: str, alias_to_entity: Dict[str, str]) -> Set[str]:
    """Resolve entity mentions in free text by alias and identifier shape."""
    found: Set[str] = set()
    if not text:
        return found
    normalized = normalize_text(text)
    # alias lookup: build a normalized alias map once per call (small cases)
    norm_map = {normalize_text(k): v for k, v in alias_to_entity.items() if k}
    for alias, eid in norm_map.items():
        if not alias:
            continue
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized):
            found.add(eid)
    # identifier-shaped tokens
    for match in _IDENTIFIER_RE.finditer(text):
        token = match.group(0)
        eid = alias_to_entity.get(token) or alias_to_entity.get(token.lower())
        if eid:
            found.add(eid)
    return found


def _identifier_tokens(text: str) -> List[str]:
    return [m.group(0) for m in _IDENTIFIER_RE.finditer(text or "")]


def _merge_entity_maps(existing: Dict[str, Entity],
                       new: Dict[str, Entity]) -> Tuple[Dict[str, Entity], Dict[str, str]]:
    """Merge a freshly built entity map into the case's existing entities.

    Returns ``(merged, id_mapping)`` where ``id_mapping`` maps new entity ids
    to the ids retained in the merged registry.
    """
    merged = {k: Entity.from_dict(v.to_dict()) for k, v in existing.items()}
    mapping: Dict[str, str] = {}
    by_identifier = {ident: eid for eid, ent in merged.items() for ident in ent.identifiers}
    by_name = {normalize_text(name): eid for eid, ent in merged.items() for name in ent.names}
    next_num = max([int(re.sub(r"\D", "", eid) or 0) for eid in merged] or [0]) + 1

    for new_id, ent in new.items():
        target = None
        for ident in ent.identifiers:
            if ident in by_identifier:
                target = by_identifier[ident]
                break
        if target is None:
            for name in ent.names:
                key = normalize_text(name)
                if key in by_name:
                    target = by_name[key]
                    break
        if target is None and new_id in merged:
            target = f"E{next_num:04d}"
            next_num += 1
        if target is None:
            target = new_id
        mapping[new_id] = target
        if target in merged:
            dst = merged[target]
            dst.names = sorted(set(dst.names) | set(ent.names))
            dst.identifiers = sorted(set(dst.identifiers) | set(ent.identifiers))
            dst.sources = sorted(set(dst.sources) | set(ent.sources))
            dst.roles = sorted(set(dst.roles) | set(ent.roles))
            dst.n_events = max(dst.n_events, ent.n_events)
            dst.first_seen = min(filter(None, [dst.first_seen, ent.first_seen])) if (dst.first_seen or ent.first_seen) else None
            dst.last_seen = max(filter(None, [dst.last_seen, ent.last_seen])) if (dst.last_seen or ent.last_seen) else None
            dst.metadata = {**dst.metadata, **ent.metadata}
        else:
            merged[target] = ent
        for ident in ent.identifiers:
            by_identifier[ident] = target
        for name in ent.names:
            by_name[normalize_text(name)] = target
    return merged, mapping


def _entities_from_events(events: pd.DataFrame, entity_map) -> Dict[str, Entity]:
    entities: Dict[str, Entity] = {}
    if events is None or events.empty:
        return entities
    id_to_entity = getattr(entity_map, "entity_id_by_identifier", {}) or {}
    names = getattr(entity_map, "entity_names", {}) or {}
    reverse: Dict[str, List[str]] = {}
    for ident, eid in id_to_entity.items():
        reverse.setdefault(eid, []).append(str(ident))
    for eid, identifiers in reverse.items():
        sub = events[events["entity_id"] == eid]
        ts = pd.to_datetime(sub.get("timestamp"), errors="coerce", utc=True).dropna()
        entities[eid] = Entity(
            entity_id=eid,
            names=sorted(names.get(eid, set())),
            identifiers=sorted(set(identifiers)),
            roles=["unknown"],
            sources=sorted(set(sub.get("source", pd.Series(dtype=str)).dropna().astype(str))),
            first_seen=ts.min().isoformat() if not ts.empty else None,
            last_seen=ts.max().isoformat() if not ts.empty else None,
            n_events=int(len(sub)),
            metadata={"from_structured": True},
        )
    return entities


_NAME_STOP = {
    "subject", "from", "to", "cc", "bcc", "date", "time", "account", "holder",
    "opening", "form", "case", "note", "notes", "report", "reports", "reported",
    "source", "sources", "meeting", "tower", "branch", "location", "offshore",
    "instructions", "instruction", "message", "messages", "delete", "good",
    "meet", "yes", "no", "dear", "regards", "best", "thanks", "email", "phone",
    "bank", "transfer", "transfers", "payment", "payments", "transaction",
    "transactions", "amount", "amounts", "total", "investigator", "analyst",
    "sutram", "investigation", "assistant", "digital", "financial",
    "intelligence", "evidence", "document", "documents", "draft", "final",
    "confidential", "urgent", "important", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday", "january", "february", "march",
    "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "please", "kindly", "attached", "attachment",
    "sent", "received", "reply", "forward", "whatsapp", "telegram", "signal",
    "chat", "conversation", "call", "calls", "source", "open", "closed",
    "high", "medium", "low", "risk", "alert", "review", "summary", "overview",
}

_NAME_RE = re.compile(
    r"\b([A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ'’-]+(?:\s+[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ'’-]+){1,2})\b"
)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_EMAIL_SKIP = {"noreply", "no-reply", "donotreply", "info", "support", "admin",
               "sales", "contact", "help", "security", "abuse", "postmaster"}


def extract_person_names(text: str) -> List[str]:
    """Lightweight person-name extraction for free text.

    Two- or three-token capitalized spans, excluding headers, months, roles and
    other common non-name phrases.  This is deliberately conservative: it only
    creates entities for names that recur or appear in email headers.
    """
    cleaned = _EMAIL_RE.sub(" ", text or "")
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    out: List[str] = []
    for match in _NAME_RE.finditer(cleaned):
        tokens = re.sub(r"\s+", " ", match.group(1)).strip().split()
        for size in (3, 2):
            cand = tokens[:size]
            if len(cand) < 2:
                continue
            if any(t.lower().strip(".,") in _NAME_STOP for t in cand):
                continue
            if all(t.isupper() for t in cand):
                continue
            if any(re.search(r"\d", t) for t in cand):
                continue
            name = " ".join(cand)
            if name not in out:
                out.append(name)
            break
    return out


def _name_tokens(name: str) -> Set[str]:
    return {
        normalize_text(t) for t in re.findall(r"[^\W\d_]+", name or "", flags=re.UNICODE)
        if len(t) >= 2
    }


def _email_tokens(address: str) -> Set[str]:
    local = (address or "").split("@")[0]
    return {
        normalize_text(t) for t in re.split(r"[^A-Za-z0-9]+", local)
        if len(t) >= 2
    }


def link_chunks(chunks: List[Chunk], entities: Dict[str, Entity],
                aliases: Dict[str, str], create_unknown: bool = True) -> Dict[str, Entity]:
    """Attach entity mentions and locations to chunks; create text entities.

    Text-only entities are created for:

    * identifier-shaped tokens (phones, handles, account numbers) appearing
      at least twice;
    * person names appearing at least twice or in an email From/To/Cc header;
    * email addresses, linked to a person when the local part matches the
      person's name tokens, otherwise created as their own entity.
    """
    alias_to_entity = dict(aliases)
    # normalized aliases let text names link to structured entities regardless
    # of casing/accents ("Yara Okafor" vs "yara okafor")
    for _alias, _eid in list(alias_to_entity.items()):
        _norm = normalize_text(_alias)
        if _norm:
            alias_to_entity.setdefault(_norm, _eid)
    if not create_unknown:
        for chunk in chunks:
            chunk.entity_ids = sorted(extract_mentions(chunk.text, alias_to_entity))
        return entities

    next_num = max(
        [int(re.sub(r"\D", "", eid) or 0) for eid in entities if str(eid).startswith("U")]
        or [0]
    ) + 1

    def new_id() -> str:
        nonlocal next_num
        eid = f"U{next_num:04d}"
        while eid in entities:
            next_num += 1
            eid = f"U{next_num:04d}"
        next_num += 1
        return eid

    # 1) identifier-shaped tokens
    unknown_ids: Dict[str, int] = {}
    for chunk in chunks:
        for token in _identifier_tokens(chunk.text):
            if token in alias_to_entity or token.lower() in alias_to_entity:
                continue
            if _EMAIL_RE.fullmatch(token):
                continue
            unknown_ids[token] = unknown_ids.get(token, 0) + 1
    for token, count in sorted(unknown_ids.items(), key=lambda kv: -kv[1]):
        if count < 2:
            continue
        eid = new_id()
        entities[eid] = Entity(
            entity_id=eid,
            names=[token],
            identifiers=[token],
            roles=["unknown"],
            sources=["text"],
            n_events=0,
            metadata={"from_text": True, "occurrences": count},
        )
        alias_to_entity[token] = eid
        alias_to_entity[token.lower()] = eid

    # 2) person names
    name_counts: Dict[str, int] = {}
    name_original: Dict[str, str] = {}
    name_sources: Dict[str, Set[str]] = {}
    header_names: Set[str] = set()
    for chunk in chunks:
        for name in extract_person_names(chunk.text):
            key = normalize_text(name)
            name_counts[key] = name_counts.get(key, 0) + 1
            name_original.setdefault(key, name)
            name_sources.setdefault(key, set()).add(chunk.source_type)
        if chunk.source_type == "email":
            for line in chunk.text.splitlines():
                m = re.match(r"^(From|To|Cc):\s*(.+?)\s*<", line, re.IGNORECASE)
                if m:
                    for name in extract_person_names(m.group(2)):
                        header_names.add(normalize_text(name))
                        name_original.setdefault(normalize_text(name), name)
                        name_sources.setdefault(normalize_text(name), set()).add("email")
    for key, count in sorted(name_counts.items(), key=lambda kv: -kv[1]):
        if key in alias_to_entity:
            continue
        if count < 2 and key not in header_names:
            continue
        name = name_original.get(key, key)
        eid = new_id()
        entities[eid] = Entity(
            entity_id=eid,
            names=[name],
            identifiers=[],
            roles=["unknown"],
            sources=sorted(name_sources.get(key, {"text"})),
            n_events=count,
            metadata={"from_text": True, "kind": "person", "occurrences": count},
        )
        alias_to_entity[name] = eid
        alias_to_entity[key] = eid
        alias_to_entity.setdefault(normalize_text(name), eid)

    # 3) email addresses -> person entity by local-part name match
    email_counts: Dict[str, int] = {}
    for chunk in chunks:
        for address in _EMAIL_RE.findall(chunk.text):
            email_counts[address] = email_counts.get(address, 0) + 1
    for address, count in sorted(email_counts.items(), key=lambda kv: -kv[1]):
        if address in alias_to_entity or address.lower() in alias_to_entity:
            continue
        local = address.split("@")[0].lower()
        if local in _EMAIL_SKIP:
            continue
        tokens = _email_tokens(address)
        best_eid = None
        best_overlap = 0
        for eid, ent in entities.items():
            for name in ent.names:
                nt = _name_tokens(name)
                if nt and nt <= tokens and len(nt) > best_overlap:
                    best_eid = eid
                    best_overlap = len(nt)
        if best_eid:
            ent = entities[best_eid]
            ent.identifiers = sorted(set(ent.identifiers) | {address})
            alias_to_entity[address] = best_eid
            alias_to_entity[address.lower()] = best_eid
        elif count >= 1:
            eid = new_id()
            entities[eid] = Entity(
                entity_id=eid,
                names=[address],
                identifiers=[address],
                roles=["unknown"],
                sources=["email"],
                n_events=count,
                metadata={"from_text": True, "kind": "email", "occurrences": count},
            )
            alias_to_entity[address] = eid
            alias_to_entity[address.lower()] = eid

    # 4) attach mentions / locations / amounts / timestamps
    for chunk in chunks:
        mentions = extract_mentions(chunk.text, alias_to_entity)
        chunk.entity_ids = sorted(mentions)
        locs = re.findall(r"\b(?:T|TOWER|BRANCH|LAT|LOC|CELL)[-_]?\d+\b",
                          chunk.text, re.IGNORECASE)
        if locs:
            chunk.location = locs[0]
        amounts = extract_amounts(chunk.text)
        if amounts:
            chunk.metadata["amounts"] = amounts[:10]
        if chunk.timestamp is None:
            chunk.timestamp = extract_timestamp(chunk.text)
        for eid in chunk.entity_ids:
            ent = entities.get(eid)
            if not ent or not (ent.metadata or {}).get("from_text"):
                continue
            ent.sources = sorted(set(ent.sources) | {chunk.source_type})
            if chunk.timestamp:
                ent.first_seen = min(filter(None, [ent.first_seen, chunk.timestamp])) \
                    if (ent.first_seen or chunk.timestamp) else None
                ent.last_seen = max(filter(None, [ent.last_seen, chunk.timestamp])) \
                    if (ent.last_seen or chunk.timestamp) else None
    return entities


# ---------------------------------------------------------------------------
# main ingestion
# ---------------------------------------------------------------------------
def ingest_paths(case: Case, store: CaseStore, paths: Sequence[str],
                 custodian: str = "", notes: str = "",
                 probe_cfg=None, min_events_per_entity: int = 30) -> IngestResult:
    """Add evidence files to a case, parse them and rebuild the index."""
    parsed: List[ParsedEvidence] = []
    added: List[str] = []
    for raw_path in paths:
        item = store.add_evidence(case, raw_path, source_type=classify_source_type(raw_path),
                                  custodian=custodian, notes=notes)
        if item.evidence_id not in added:
            added.append(item.evidence_id)
        # already parsed duplicate: do not double count
        if item.n_rows or item.n_chunks:
            continue
        path = store.resolve_evidence_path(case, item)
        result = parse_evidence(item, path, probe_cfg=probe_cfg,
                                chunk_start=sum(len(p.chunks) for p in parsed) + 1)
        item.source_type = result.detected_source
        item.n_rows = int(len(result.events)) if result.events is not None else 0
        item.n_chunks = len(result.chunks)
        item.warnings = list(dict.fromkeys(result.warnings))
        item.metadata = {**item.metadata, **result.metadata}
        parsed.append(result)

    # ---- structured events ------------------------------------------------
    frames = [p.events for p in parsed if p.events is not None and not p.events.empty]
    events = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not events.empty:
        from aml.entity_resolve import resolve_entities
        from aml.schema_detect import _clean_ids
        for col in events.columns:
            if events[col].dtype == object:
                events[col] = _clean_ids(events[col])
        # The same export often arrives in several formats (csv + json + jsonl).
        # Deduplicate on the canonical event fields, not on provenance columns,
        # so one logical event is not counted twice.
        canonical = [
            "actor_raw", "actor_name", "event_type", "timestamp", "source",
            "amount", "counterparty_id_raw", "location", "direction", "duration",
        ]
        subset = [c for c in canonical if c in events.columns]
        events = events.drop_duplicates(subset=subset if subset else None)
        if "timestamp" in events.columns:
            events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce", utc=True)
        em = resolve_entities(events)
        events["entity_id"] = events["actor_raw"].map(em.entity_id_by_identifier)
        events["counterparty_id"] = events["counterparty_id_raw"].map(em.entity_id_by_identifier)
        new_entities = _entities_from_events(events, em)
        merged, mapping = _merge_entity_maps(case.entities, new_entities)
        if mapping:
            events["entity_id"] = events["entity_id"].map(lambda x: mapping.get(x, x))
            events["counterparty_id"] = events["counterparty_id"].map(lambda x: mapping.get(x, x))
        case.entities = merged
    else:
        em = None
        case.entities = dict(case.entities)

    # ---- chunks -----------------------------------------------------------
    chunks: List[Chunk] = [c for p in parsed for c in p.chunks]
    aliases = build_alias_index(case.entities)
    if chunks:
        case.entities = link_chunks(chunks, case.entities, aliases, create_unknown=True)
        aliases = build_alias_index(case.entities)

    # ---- persist ----------------------------------------------------------
    store.save_index(case, events, chunks, aliases)
    case.metadata["index_version"] = utc_now()
    store.save_case(case)

    warnings: List[str] = []
    per_evidence: List[dict] = []
    for p in parsed:
        warnings.extend(p.warnings)
        per_evidence.append({
            "evidence_id": p.item.evidence_id,
            "filename": p.item.filename,
            "source_type": p.detected_source,
            "rows": int(len(p.events)) if p.events is not None else 0,
            "chunks": len(p.chunks),
            "warnings": p.warnings,
        })
    return IngestResult(
        events=events,
        chunks=chunks,
        entities=case.entities,
        aliases=aliases,
        warnings=list(dict.fromkeys(warnings)),
        per_evidence=per_evidence,
        added_evidence_ids=added,
    )

"""Tamper-evident case diary.

Every investigation action (ingest, analysis, query, finding, recommendation,
report) is appended to an ordered log.  Each entry stores the hash of the
previous entry and its own SHA-256, forming a hash chain:

    entry[n].prev_hash == entry[n-1].hash
    entry[n].hash      == sha256(canonical_payload(entry[n]))

This is not a blockchain and does not prevent deletion of the whole log, but
it makes silent edits detectable, which is the relevant property for a case
diary that may later be reviewed or disclosed.
"""
from __future__ import annotations

import hashlib
import json
from typing import List, Tuple

from .i18n import t
from .case import Case, DiaryEntry, utc_now

GENESIS_HASH = "0" * 64


def _payload(entry: DiaryEntry) -> str:
    """Canonical JSON payload covered by the hash (everything but ``hash``)."""
    data = {
        "entry_id": entry.entry_id,
        "seq": entry.seq,
        "timestamp": entry.timestamp,
        "actor": entry.actor,
        "entry_type": entry.entry_type,
        "text": entry.text,
        "evidence_ids": list(entry.evidence_ids),
        "prev_hash": entry.prev_hash,
    }
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _hash_entry(entry: DiaryEntry) -> str:
    return hashlib.sha256(_payload(entry).encode("utf-8")).hexdigest()


def append_entry(case: Case, entry_type: str, text: str, actor: str = "system",
                 evidence_ids: List[str] | None = None) -> DiaryEntry:
    """Append a diary entry and update the case's hash chain."""
    prev = case.diary[-1].hash if case.diary else GENESIS_HASH
    entry = DiaryEntry(
        entry_id=f"D{len(case.diary) + 1:05d}",
        seq=len(case.diary) + 1,
        timestamp=utc_now(),
        actor=actor or "system",
        entry_type=entry_type or "note",
        text=text or "",
        evidence_ids=list(evidence_ids or []),
        prev_hash=prev,
    )
    entry.hash = _hash_entry(entry)
    case.diary.append(entry)
    case.touch()
    return entry


def verify_chain(case: Case) -> Tuple[bool, List[str]]:
    """Verify ordering, prev-hash links and recomputed hashes.

    Returns ``(ok, errors)``.  An empty diary is considered valid.
    """
    errors: List[str] = []
    prev = GENESIS_HASH
    for i, entry in enumerate(case.diary):
        if entry.seq != i + 1:
            errors.append(f"entry {entry.entry_id}: seq {entry.seq} != {i + 1}")
        if entry.prev_hash != prev:
            errors.append(f"entry {entry.entry_id}: prev_hash mismatch")
        expected = _hash_entry(entry)
        if entry.hash != expected:
            errors.append(f"entry {entry.entry_id}: hash mismatch")
        prev = entry.hash
    return (not errors, errors)


def render_diary_md(case: Case, lang: str = "en") -> str:
    """Render the diary as Markdown for export / the case report."""
    ok, errors = verify_chain(case)
    status = t("diary.verified", lang) if ok else t("diary.tampered", lang)
    lines = [
        f"# {t('diary.title', lang)} — {case.title}",
        "",
        f"*{t('app.case', lang)}: {case.case_id} · {status}*",
        "",
    ]
    if errors:
        lines += ["> **Chain errors:** " + "; ".join(errors), ""]
    for entry in case.diary:
        lines.append(
            f"- **{entry.timestamp}** · `{entry.entry_type}` · {entry.actor} — "
            f"{entry.text}  "
        )
        if entry.evidence_ids:
            lines.append(f"  - {t('common.evidence', lang)}: {', '.join(entry.evidence_ids)}")
        lines.append(f"  - hash `{entry.hash[:16]}…` ← `{entry.prev_hash[:16]}…`")
    return "\n".join(lines) + "\n"


def write_audit_log(case: Case, path: str) -> None:
    """Write a human-readable mirror of the diary to ``audit.log``."""
    with open(path, "w", encoding="utf-8") as fh:
        for entry in case.diary:
            fh.write(
                f"{entry.seq:05d}\t{entry.timestamp}\t{entry.actor}\t"
                f"{entry.entry_type}\t{entry.hash}\t{entry.prev_hash}\t{entry.text}\n"
            )

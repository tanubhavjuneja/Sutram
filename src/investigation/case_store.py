"""Case persistence, evidence inventory and chain-of-custody metadata.

A case is a directory:

    <root>/<case_id>/
        case.json
        evidence/
        index/
        reports/
        audit.log

The store is intentionally simple (JSON + JSONL) so a case can be copied,
archived, hashed or reviewed without a database.  Evidence files are copied
into ``evidence/`` and identified by SHA-256; duplicates are detected by hash.
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import diary
from .models import Case, Chunk, Entity, EvidenceItem, slugify, utc_now

CASE_ROOT_DEFAULT = "data/cases"
CHUNK_FILE = "chunks.jsonl"
EVENT_FILE = "events.jsonl"
ENTITY_FILE = "entities.json"
ALIAS_FILE = "aliases.json"


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _mime(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


class CaseStore:
    def __init__(self, root: str | Path = CASE_ROOT_DEFAULT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths -------------------------------------------------------------
    def case_dir(self, case_id: str) -> Path:
        return self.root / case_id

    def evidence_dir(self, case: Case) -> Path:
        return Path(case.root) / "evidence"

    def index_dir(self, case: Case) -> Path:
        return Path(case.root) / "index"

    def reports_dir(self, case: Case) -> Path:
        return Path(case.root) / "reports"

    def analytics_dir(self, case: Case) -> Path:
        return self.reports_dir(case) / "analytics"

    def case_file(self, case: Case) -> Path:
        return Path(case.root) / "case.json"

    def resolve_evidence_path(self, case: Case, item: EvidenceItem) -> Path:
        path = Path(item.path)
        if not path.is_absolute():
            path = Path(case.root) / path
        return path

    # -- lifecycle ---------------------------------------------------------
    def create_case(self, title: str, description: str = "", language: str = "en") -> Case:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        base = f"{slugify(title)}-{stamp}"
        case_id = base
        n = 2
        while (self.root / case_id).exists():
            case_id = f"{base}-{n}"
            n += 1
        root = self.root / case_id
        for sub in ("evidence", "index", "reports", "reports/analytics"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        case = Case(
            case_id=case_id,
            title=title or "Untitled case",
            description=description,
            language=language or "en",
            root=str(root),
        )
        diary.append_entry(case, "note", f"Case created: {case.title}", actor="system")
        self.save_case(case)
        return case

    def save_case(self, case: Case) -> None:
        if not case.root:
            case.root = str(self.case_dir(case.case_id))
        root = Path(case.root)
        root.mkdir(parents=True, exist_ok=True)
        case.touch()
        (root / "case.json").write_text(
            json.dumps(case.to_dict(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        try:
            diary.write_audit_log(case, str(root / "audit.log"))
        except OSError:
            pass

    def load_case(self, case_id_or_path: str) -> Case:
        path = Path(case_id_or_path)
        if path.is_dir() and (path / "case.json").exists():
            root = path
        else:
            root = self.case_dir(case_id_or_path)
        case_file = root / "case.json"
        if not case_file.exists():
            raise FileNotFoundError(f"case not found: {case_id_or_path}")
        data = json.loads(case_file.read_text(encoding="utf-8"))
        case = Case.from_dict(data, root=str(root))
        return case

    def list_cases(self) -> List[dict]:
        out: List[dict] = []
        for path in sorted(self.root.iterdir(), reverse=True):
            if not path.is_dir():
                continue
            case_file = path / "case.json"
            if not case_file.exists():
                continue
            try:
                data = json.loads(case_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            out.append({
                "case_id": data.get("case_id", path.name),
                "title": data.get("title", path.name),
                "language": data.get("language", "en"),
                "status": data.get("status", "open"),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "stats": {
                    "evidence": len(data.get("evidence", [])),
                    "entities": len(data.get("entities", {})),
                    "findings": len(data.get("findings", [])),
                    "recommendations": len(data.get("recommendations", [])),
                    "diary_entries": len(data.get("diary", [])),
                },
            })
        return out

    def delete_case(self, case_id: str) -> bool:
        path = self.case_dir(case_id)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            return True
        return False

    # -- evidence ----------------------------------------------------------
    def add_evidence(self, case: Case, path: str | Path, source_type: str = "",
                     custodian: str = "", notes: str = "") -> EvidenceItem:
        """Copy an evidence file into the case and return its inventory item.

        Duplicate content (same SHA-256) returns the existing item instead of
        creating a second copy; this prevents double-counting the same export
        delivered in two formats.
        """
        src = Path(path)
        if not src.is_file():
            raise FileNotFoundError(f"evidence file not found: {src}")
        digest = _sha256(src)
        for existing in case.evidence:
            if existing.sha256 == digest:
                existing.metadata.setdefault("duplicates", []).append(str(src))
                return existing

        evidence_id = f"EV{len(case.evidence) + 1:04d}"
        target = self.evidence_dir(case) / f"{evidence_id}_{src.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        item = EvidenceItem(
            evidence_id=evidence_id,
            path=str(target.relative_to(Path(case.root))).replace("\\", "/"),
            filename=src.name,
            sha256=digest,
            size_bytes=src.stat().st_size,
            mime=_mime(src),
            source_type=source_type or "other",
            acquired_at=utc_now(),
            custodian=custodian,
            notes=notes,
            original_path=str(src.resolve()),
            metadata={"extension": src.suffix.lower()},
        )
        case.evidence.append(item)
        case.touch()
        return item

    # -- index persistence -------------------------------------------------
    def save_index(self, case: Case, events: pd.DataFrame, chunks: List[Chunk],
                   aliases: Dict[str, str] | None = None) -> None:
        index_dir = self.index_dir(case)
        index_dir.mkdir(parents=True, exist_ok=True)
        if events is not None and not events.empty:
            # JSONL keeps mixed types and is inspectable without pyarrow
            events.to_json(index_dir / EVENT_FILE, orient="records",
                           lines=True, date_format="iso", default_handler=str)
        else:
            (index_dir / EVENT_FILE).write_text("", encoding="utf-8")
        with open(index_dir / CHUNK_FILE, "w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False, default=str) + "\n")
        (index_dir / ENTITY_FILE).write_text(
            json.dumps({k: v.to_dict() for k, v in case.entities.items()},
                       indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        (index_dir / ALIAS_FILE).write_text(
            json.dumps(aliases or {}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_events(self, case: Case) -> pd.DataFrame:
        path = self.index_dir(case) / EVENT_FILE
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_json(path, orient="records", lines=True)
        except ValueError:
            return pd.DataFrame()

    def load_chunks(self, case: Case) -> List[Chunk]:
        path = self.index_dir(case) / CHUNK_FILE
        if not path.exists():
            return []
        chunks: List[Chunk] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunks.append(Chunk.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError):
                    continue
        return chunks

    def load_entities(self, case: Case) -> Dict[str, Entity]:
        path = self.index_dir(case) / ENTITY_FILE
        if not path.exists():
            return dict(case.entities)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return dict(case.entities)
        return {str(k): Entity.from_dict(v) for k, v in data.items()}

    def load_aliases(self, case: Case) -> Dict[str, str]:
        path = self.index_dir(case) / ALIAS_FILE
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return {str(k): str(v) for k, v in data.items()}

    # -- diary convenience -------------------------------------------------
    def append_diary(self, case: Case, entry_type: str, text: str,
                     actor: str = "system", evidence_ids: Optional[List[str]] = None):
        entry = diary.append_entry(case, entry_type, text, actor=actor,
                                   evidence_ids=evidence_ids)
        self.save_case(case)
        return entry

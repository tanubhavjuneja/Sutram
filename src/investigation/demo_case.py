"""Generate a complete demo case for the investigation assistant.

The demo combines the existing synthetic AML generator with investigation
artefacts that do not exist in the original project:

* emails between suspects (`.eml`)
* chat logs (`.txt`)
* an investigator note (`.md`)
* a PDF account-opening form
* a ground-truth query set for evaluation

It then creates a case, ingests everything, runs the analytics, generates
recommendations and writes the draft case report.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("AML_SKIP_TORCH", "1")

import pandas as pd

from .i18n import normalize_language
from .service import InvestigationConfig, InvestigationService


def _email_for(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", ".", name.lower()).strip(".")
    return f"{slug}@example.com"


def _first_name(name: str) -> str:
    return (name or "there").split()[0]


def _pick_storyline(key: dict, storyline: str) -> Tuple[Optional[str], Optional[dict]]:
    for pid, info in key.get("entities", {}).items():
        kind = info.get("storyline") or (info.get("details") or {}).get("type")
        if kind == storyline:
            return pid, info
    return None, None


def _entity_by_id(key: dict, pid: str) -> Optional[dict]:
    return key.get("entities", {}).get(pid)


def _counterparty_from_note(key: dict, info: dict) -> Optional[dict]:
    note = (info.get("details") or {}).get("note", "") or ""
    match = re.search(r"E\d+", note)
    if match:
        return _entity_by_id(key, match.group(0))
    return None


def _load_key(root: Path) -> dict:
    path = root / "answer_key.json"
    if not path.exists():
        return {"entities": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _case_dates(root: Path) -> Tuple[datetime, datetime]:
    """Read the generated bank export to anchor communications in the case period."""
    bank = root / "sources" / "bank_export.csv"
    fallback = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
    if not bank.exists():
        return fallback - timedelta(days=30), fallback
    try:
        df = pd.read_csv(bank, usecols=["timestamp"])
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dropna()
        if ts.empty:
            return fallback - timedelta(days=30), fallback
        return ts.min().to_pydatetime(), ts.max().to_pydatetime()
    except Exception:
        return fallback - timedelta(days=30), fallback


def _write_email(path: Path, sender: dict, recipient: dict, subject: str,
                 body: str, when: datetime) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"{sender.get('name', 'Unknown')} <{_email_for(sender.get('name', 'unknown'))}>"
    msg["To"] = f"{recipient.get('name', 'Unknown')} <{_email_for(recipient.get('name', 'unknown'))}>"
    msg["Date"] = when.strftime("%a, %d %b %Y %H:%M:%S +0000")
    msg.set_content(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(msg))


def _write_pdf(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(str(path))
        y = 760
        for line in lines:
            c.drawString(64, y, line)
            y -= 22
        c.save()
    except Exception:
        # reportlab is a project dependency, but keep a text fallback
        path.with_suffix(".txt").write_text("\n".join(lines), encoding="utf-8")


def _make_communications(root: Path, key: dict, case_title: str) -> Dict[str, Path]:
    comms = root / "communications"
    docs = root / "documents"
    comms.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)

    start, end = _case_dates(root)
    base = end - timedelta(days=3)
    if base < start:
        base = start + timedelta(days=1)

    s_pid, s = _pick_storyline(key, "structuring")
    c_pid, c = _pick_storyline(key, "colocation")
    ch_pid, ch = _pick_storyline(key, "chain")
    d_pid, d = _pick_storyline(key, "dormant_flip")
    f_pid, f = _pick_storyline(key, "fan_io")
    bg_pid, bg = _pick_storyline(key, "background")
    entities = [x for x in (s, c, ch, d, f, bg) if x]
    if not entities:
        raise RuntimeError("synthetic generator produced no entities")

    s = s or entities[0]
    c = c or entities[min(1, len(entities) - 1)]
    ch = ch or entities[min(2, len(entities) - 1)]
    d = d or entities[min(3, len(entities) - 1)]
    f = f or entities[min(4, len(entities) - 1)]
    bg = bg or entities[-1]

    # colocation counterparty from the generator's note, if available
    cp = _counterparty_from_note(key, c) if c else None
    if not cp:
        cp = ch
    # chain next hop
    chain_note = ((ch.get("details") or {}).get("note", "") or "") if ch else ""
    chain_ids = re.findall(r"E\d+", chain_note)
    next_hop = _entity_by_id(key, chain_ids[1]) if len(chain_ids) > 1 else bg
    next_hop = next_hop or bg

    location = "T854"
    amount = 9500.0
    threshold = 10000.0

    # 1) structuring instruction email
    email1 = comms / "email_structuring_instructions.eml"
    _write_email(
        email1, s, c,
        subject=f"Invoice and transfer — {s.get('account', '')}",
        body=(
            f"Hi {_first_name(c.get('name', ''))},\n\n"
            f"Please move {amount:,.0f} to {s.get('account', 'the usual account')} "
            f"before the {threshold:,.0f} reporting limit. Use the usual branch and "
            f"confirm from the {location} tower. Delete this thread after reading.\n\n"
            f"— {_first_name(s.get('name', ''))}"
        ),
        when=base - timedelta(days=2, hours=3),
    )

    # 2) layering instructions email
    email2 = comms / "email_layering_instructions.eml"
    _write_email(
        email2, ch, next_hop,
        subject="Layering instructions — next hop",
        body=(
            f"The funds are in {ch.get('account', 'the first account')}. Move them "
            f"through the usual intermediate accounts and keep each transfer below "
            f"the reporting threshold. Do not use the same branch twice. Confirm when "
            f"the second hop is complete."
        ),
        when=base - timedelta(days=1, hours=5),
    )

    # 3) tower confirmation email
    email3 = comms / "email_tower_confirmation.eml"
    _write_email(
        email3, c, s,
        subject="Tower log confirmation",
        body=(
            f"I was at tower {location} at {base.strftime('%H:%M')} on "
            f"{base.strftime('%d %B %Y')}. The call to {cp.get('phone', 'the number')} "
            f"was before the transfer. We should not discuss this on the usual phone again."
        ),
        when=base - timedelta(hours=2),
    )

    # 4) chat logs
    chat1 = comms / "chat_structuring.txt"
    chat1.write_text(
        "\n".join([
            f"[{base.strftime('%Y-%m-%d')} 10:00:00] {s.get('name', 'A')}: Did you send the {amount:,.0f}?",
            f"[{base.strftime('%Y-%m-%d')} 10:02:00] {c.get('name', 'B')}: Yes, to {s.get('account', 'the account')}. The branch was quiet.",
            f"[{base.strftime('%Y-%m-%d')} 10:03:00] {s.get('name', 'A')}: Good. Use tower {location} next time. Delete this chat.",
            f"[{base.strftime('%Y-%m-%d')} 10:05:00] {c.get('name', 'B')}: Understood.",
        ]),
        encoding="utf-8",
    )

    chat2 = comms / "chat_layering.txt"
    chat2.write_text(
        "\n".join([
            f"[{base.strftime('%Y-%m-%d')} 11:10:00] {ch.get('name', 'C')}: The first hop is done.",
            f"[{base.strftime('%Y-%m-%d')} 11:12:00] {next_hop.get('name', 'D')}: Received. Moving to the second account now.",
            f"[{base.strftime('%Y-%m-%d')} 11:13:00] {ch.get('name', 'C')}: Keep it under 10k.",
        ]),
        encoding="utf-8",
    )

    # 5) investigator note
    note = docs / "investigator_note.md"
    note.write_text(
        "\n".join([
            f"# Source report — {case_title}",
            "",
            f"Source reports that {s.get('name')} and {c.get('name')} met at tower "
            f"{location} on {base.strftime('%d %B %Y')}. {s.get('name')} is suspected "
            f"of structuring deposits just below the {threshold:,.0f} reporting limit.",
            "",
            f"{c.get('name')} called {cp.get('name')} minutes before a transfer. "
            f"{ch.get('name')} is linked to a three-hop layering path. "
            f"{d.get('name')} shows a dormant account suddenly reactivating. "
            f"{f.get('name')} shows a fan-in / fan-out pattern.",
            "",
            "Open questions: who controls the beneficiary accounts, and whether the "
            "tower location links the suspects to the transfers.",
        ]),
        encoding="utf-8",
    )

    # 6) account opening form (PDF)
    form = docs / "account_opening_form.pdf"
    _write_pdf(form, [
        "ACCOUNT OPENING FORM",
        f"Holder: {s.get('name', '')}",
        f"Account: {s.get('account', '')}",
        f"Phone: {s.get('phone', '')}",
        f"Handle: {s.get('handle', '')}",
        f"Date: {(base - timedelta(days=40)).strftime('%Y-%m-%d')}",
        "Address: 14 Harbour Road",
        "Risk rating: standard",
        "Relationship manager: M. Tan",
    ])

    return {
        "email_structuring": email1,
        "email_layering": email2,
        "email_tower": email3,
        "chat_structuring": chat1,
        "chat_layering": chat2,
        "note": note,
        "form": form,
        "s": s, "c": c, "ch": ch, "d": d, "f": f, "bg": bg, "cp": cp,
    }


def _ground_truth(case_title: str, comms: dict) -> dict:
    s, c, ch, d, f = comms["s"], comms["c"], comms["ch"], comms["d"], comms["f"]
    s_name, c_name, ch_name = s.get("name", ""), c.get("name", ""), ch.get("name", "")
    queries = [
        {
            "id": "q_search_en", "lang": "en", "intent": "search",
            "question": f"Show transfers above 9,000 from {s_name} in the last 90 days.",
            "entity_names": [s_name], "keywords": ["bank", "debit", "credit"],
            "evidence_files": ["bank_export.csv"],
        },
        {
            "id": "q_correlation_en", "lang": "en", "intent": "correlation",
            "question": f"Who did {c_name} call right before the transfer?",
            "entity_names": [c_name], "keywords": ["communication", "transfer", "cdr"],
            "evidence_files": ["cdr_export.csv"],
        },
        {
            "id": "q_relationship_en", "lang": "en", "intent": "relationship",
            "question": f"What is the connection between {s_name} and {c_name}?",
            "entity_names": [s_name, c_name], "keywords": ["relationship", "connection", "co-occur"],
            "evidence_files": [],
        },
        {
            "id": "q_profile_en", "lang": "en", "intent": "entity_profile",
            "question": f"Who is {s_name}?",
            "entity_names": [s_name], "keywords": ["risk"],
            "evidence_files": [],
        },
        {
            "id": "q_anomaly_en", "lang": "en", "intent": "anomaly",
            "question": "What looks suspicious in this case?",
            "entity_names": [], "keywords": ["suspicious"],
            "evidence_files": [],
        },
        {
            "id": "q_anomaly_hi", "lang": "hi", "intent": "anomaly",
            "question": "इस मामले में संदिग्ध गतिविधि क्या है?",
            "entity_names": [], "keywords": ["संदिग्ध"],
            "evidence_files": [],
        },
        {
            "id": "q_anomaly_es", "lang": "es", "intent": "anomaly",
            "question": "¿Cuáles son las transacciones sospechosas?",
            "entity_names": [], "keywords": ["sospech"],
            "evidence_files": [],
        },
        {
            "id": "q_anomaly_fr", "lang": "fr", "intent": "anomaly",
            "question": "Quelles transactions sont suspectes ?",
            "entity_names": [], "keywords": ["suspect"],
            "evidence_files": [],
        },
        {
            "id": "q_anomaly_de", "lang": "de", "intent": "anomaly",
            "question": "Was wirkt in diesem Fall verdächtig?",
            "entity_names": [], "keywords": ["verdächtig"],
            "evidence_files": [],
        },
        {
            "id": "q_anomaly_pt", "lang": "pt", "intent": "anomaly",
            "question": "O que parece suspeito neste caso?",
            "entity_names": [], "keywords": ["suspeit"],
            "evidence_files": [],
        },
        {
            "id": "q_search_de", "lang": "de", "intent": "search",
            "question": f"Zeige alle Überweisungen über 9.000 von {s_name}.",
            "entity_names": [s_name], "keywords": ["bank", "debit", "credit"],
            "evidence_files": ["bank_export.csv"],
        },
        {
            "id": "q_summary_en", "lang": "en", "intent": "summarize",
            "question": "Summarize the case.",
            "entity_names": [], "keywords": [case_title.split()[0]],
            "evidence_files": [],
        },
        {
            "id": "q_recommend_en", "lang": "en", "intent": "recommend",
            "question": "What should I investigate next?",
            "entity_names": [], "keywords": ["recommend", "investigat", "interview"],
            "evidence_files": [],
        },
        {
            "id": "q_diary_en", "lang": "en", "intent": "diary",
            "question": "Show the case diary.",
            "entity_names": [], "keywords": ["diary", "chain"],
            "evidence_files": [],
        },
        {
            "id": "q_report_en", "lang": "en", "intent": "report",
            "question": "Draft a case report.",
            "entity_names": [], "keywords": ["report"],
            "evidence_files": [],
        },
        {
            "id": "q_email_en", "lang": "en", "intent": "search",
            "question": "Find emails mentioning the reporting limit.",
            "entity_names": [], "keywords": ["reporting", "limit", "email"],
            "evidence_files": ["email_structuring_instructions.eml"],
        },
    ]
    return {
        "case_title": case_title,
        "entities": {
            "structuring": s,
            "colocation": c,
            "chain": ch,
            "dormant_flip": d,
            "fan_io": f,
            "counterparty": comms["cp"],
        },
        "queries": queries,
        "expected_recommendation_categories": [
            "evidence_gap", "interview", "records", "forensics",
            "corroboration", "analysis",
        ],
        "notes": "Ground truth for the demo case. Entities are keyed by role; "
                 "the eval harness maps names/identifiers to case entity ids.",
    }


def build_demo_case(root: str | Path = "data/demo_case",
                    case_root: str | Path = "data/cases",
                    lang: str = "en",
                    regenerate: bool = True) -> dict:
    """Generate the demo evidence, run the full workflow, return a summary."""
    root = Path(root)
    case_root = Path(case_root)
    lang = normalize_language(lang)

    # 1) structured synthetic data -----------------------------------------
    if regenerate or not (root / "sources" / "bank_export.csv").exists():
        from aml.data_generator.emit_multi import generate_all_mixed, make_social_text
        generate_all_mixed(
            root, seed=42, surprise_seed=99,
            n_background=24, n_per_storyline=2, n_surprise=4,
        )
        make_social_text(root, n=120)
    key = _load_key(root)
    if not key.get("entities"):
        raise RuntimeError(f"demo generator did not write an answer key under {root}")

    # 2) communications + documents ----------------------------------------
    title = "Operation Nightfall"
    comms = _make_communications(root, key, title)
    truth = _ground_truth(title, comms)
    truth_path = root / "ground_truth.json"
    truth_path.write_text(json.dumps(truth, indent=2, ensure_ascii=False, default=str),
                          encoding="utf-8")

    # 3) case + ingest + analyze + report ----------------------------------
    cfg = InvestigationConfig(
        case_root=str(case_root),
        language=lang,
        min_events_per_entity=8,
        tol_minutes=25,
        structuring_threshold=10000.0,
        top_k=10,
    )
    svc = InvestigationService(case_root=str(case_root), config=cfg, lang=lang)
    case = svc.create_case(
        title,
        description=("Synthetic multi-source investigation case: bank transactions, "
                     "CDR, social posts, emails, chat logs, an investigator note and "
                     "a PDF account-opening form."),
        language=lang,
    )
    case.metadata["ground_truth"] = str(truth_path)
    case.metadata["demo_dir"] = str(root)
    case.metadata["synthetic"] = True
    svc.save_case(case)
    evidence_paths: List[str] = []
    evidence_paths.extend(str(p) for p in sorted((root / "sources").glob("*")) if p.is_file())
    evidence_paths.extend(str(p) for p in sorted((root / "communications").glob("*")) if p.is_file())
    evidence_paths.extend(str(p) for p in sorted((root / "documents").glob("*")) if p.is_file())
    ingest = svc.ingest(case.case_id, evidence_paths, custodian="Demo custodian",
                        notes="Synthetic demo evidence")
    analysis = svc.analyze(case.case_id, run_report=True)
    svc.recommend(case.case_id, lang=lang)
    report = svc.report(case.case_id, lang=lang)

    case = svc.load_case(case.case_id)
    return {
        "case_id": case.case_id,
        "case_root": str(case.root),
        "demo_dir": str(root),
        "title": case.title,
        "language": case.language,
        "evidence": len(case.evidence),
        "events": ingest.get("events", 0),
        "chunks": ingest.get("chunks", 0),
        "entities": len(case.entities),
        "findings": len(case.findings),
        "recommendations": len(case.recommendations),
        "analysis": analysis,
        "report_files": report.get("files", {}),
        "ground_truth": str(truth_path),
        "sample_questions": [q["question"] for q in truth["queries"][:8]],
        "suggested_commands": [
            f"python run_case.py ask {case.case_id} \"What looks suspicious?\"",
            f"python run_case.py chat {case.case_id}",
            f"python run_case.py report {case.case_id}",
            f"python eval_investigation.py --case {case.case_id}",
        ],
    }

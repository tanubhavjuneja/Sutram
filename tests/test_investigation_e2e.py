"""End-to-end tests for case ingestion, retrieval, assistant, diary and report."""
from __future__ import annotations

import base64
from email.message import EmailMessage
from pathlib import Path
from types import SimpleNamespace

import pytest

from aml.case import Citation, Finding
from aml.config import InvestigationConfig
from aml.recommend import generate_recommendations
from aml.service import InvestigationService


def _write_case_files(tmp_path: Path) -> list:
    src = tmp_path / "sources"
    src.mkdir()
    bank = src / "bank.csv"
    rows = ["txn_type,amount,recipient_account,branch_code,account_name,account_no,timestamp"]
    for i in range(10):
        rows.append(
            f"debit,{9500 + i},ACCT1002,BR1,Alice,ACCT1001,2025-10-01 {10 + i}:00:00"
        )
        rows.append(
            f"credit,{9500 + i},ACCT1001,BR1,Bob,ACCT1002,2025-10-01 {11 + i}:00:00"
        )
    bank.write_text("\n".join(rows) + "\n", encoding="utf-8")

    cdr = src / "cdr.csv"
    cdr.write_text(
        "caller_msisdn,callee,cell_id,length,subscriber,timestamp\n"
        "6391000000001,6391000000002,T854,5.2,Alice,2025-10-01 09:55:00\n"
        "6391000000002,6391000000001,T854,3.1,Bob,2025-10-01 09:57:00\n",
        encoding="utf-8",
    )

    social = src / "social.csv"
    social.write_text(
        "username,hashtags,place,full_name,timestamp\n"
        "@alice,,T854,Alice,2025-10-01 09:50:00\n"
        "@bob,,T854,Bob,2025-10-01 09:51:00\n",
        encoding="utf-8",
    )

    mail = src / "mail.eml"
    msg = EmailMessage()
    msg["Subject"] = "Transfer instructions"
    msg["From"] = "Alice <alice@example.com>"
    msg["To"] = "Bob <bob@example.com>"
    msg["Date"] = "Wed, 01 Oct 2025 09:58:00 +0000"
    msg.set_content(
        "Send 9,500 to ACCT1002 before the 10,000 reporting limit. "
        "Use tower T854. Delete this message."
    )
    mail.write_bytes(bytes(msg))

    chat = src / "chat.txt"
    chat.write_text(
        "[2025-10-01 09:59:00] Alice: Did you send the 9,500?\n"
        "[2025-10-01 10:00:00] Bob: Yes, to ACCT1002. The branch was quiet.\n"
        "[2025-10-01 10:01:00] Alice: Good. Use tower T854 next time.\n",
        encoding="utf-8",
    )

    note = src / "note.md"
    note.write_text(
        "# Case note\n\nAlice and Bob met at tower T854 on 2025-10-01.",
        encoding="utf-8",
    )
    return [str(p) for p in sorted(src.iterdir())]


@pytest.fixture()
def service_case(tmp_path):
    cfg = InvestigationConfig(case_root=str(tmp_path / "cases"), min_events_per_entity=2)
    svc = InvestigationService(case_root=cfg.case_root, config=cfg)
    case = svc.create_case("E2E Test Case", description="small mixed case", language="en")
    paths = _write_case_files(tmp_path)
    ingest = svc.ingest(case.case_id, paths, custodian="Test custodian")
    # Never call the real Cloudflare API from the test suite.
    svc.llm = FakeLLM()
    return svc, svc.load_case(case.case_id), ingest, paths


def test_ingest_mixed_evidence(service_case):
    svc, case, ingest, paths = service_case
    assert len(case.evidence) == len(paths)
    assert ingest["events"] > 0
    assert ingest["chunks"] >= 3
    assert len(case.entities) >= 2
    assert any("Alice" in e.names for e in case.entities.values())
    assert any("Bob" in e.names for e in case.entities.values())
    assert svc.verify_diary(case.case_id)["ok"]


def test_search_and_citations(service_case):
    svc, case, _ingest, _paths = service_case
    answer = svc.ask(case.case_id, "Show transfers above 9,000 from Alice.")
    assert answer.intent == "search"
    assert answer.citations
    assert "Alice" in answer.answer or "ACCT" in answer.answer
    assert any(c.file == "bank.csv" for c in answer.citations)


def test_entity_profile_and_timeline(service_case):
    svc, case, _ingest, _paths = service_case
    profile = svc.ask(case.case_id, "Who is Alice?")
    assert profile.intent == "entity_profile"
    assert "Alice" in profile.answer
    assert profile.citations

    timeline = svc.ask(case.case_id, "What happened on 1 October 2025?")
    assert timeline.intent == "timeline"
    assert timeline.citations


def test_text_evidence_search(service_case):
    svc, case, _ingest, _paths = service_case
    answer = svc.ask(case.case_id, "Find emails mentioning the reporting limit.")
    assert answer.intent == "search"
    assert answer.citations
    assert any(c.file.endswith(".eml") for c in answer.citations)


def test_evidence_inventory_and_relationship(service_case):
    svc, case, _ingest, _paths = service_case
    inventory = svc.ask(case.case_id, "What evidence do we have?")
    assert inventory.intent == "evidence_inventory"
    assert "EV0001" in inventory.answer

    rel = svc.ask(case.case_id, "What is the connection between Alice and Bob?")
    assert rel.intent == "relationship"
    assert "Alice" in rel.answer and "Bob" in rel.answer


def test_multilingual_anomaly_answer_with_manual_finding(service_case):
    svc, case, _ingest, _paths = service_case
    index = svc.build_index(case)
    alice = next(e for e in case.entities.values() if "Alice" in e.names)
    citation = index.citation_for_event(index.events.iloc[0])
    case.findings.append(Finding(
        finding_id="F0001",
        kind="structuring",
        title="Structuring pattern",
        description="Repeated amounts just under the reporting limit.",
        severity="HIGH",
        confidence=0.9,
        entity_ids=[alice.entity_id],
        evidence_ids=[citation.evidence_id],
        citations=[citation],
        detected_by=["test"],
    ))
    svc.save_case(case)

    hi = svc.ask(case.case_id, "इस मामले में संदिग्ध गतिविधि क्या है?", lang="hi")
    assert hi.intent == "anomaly"
    assert "संदिग्ध" in hi.answer
    assert hi.citations

    es = svc.ask(case.case_id, "¿Cuáles son las transacciones sospechosas?", lang="es")
    assert es.intent == "anomaly"
    assert "sospech" in es.answer.lower()


def test_summary_recommendations_and_diary(service_case):
    svc, case, _ingest, _paths = service_case
    summary = svc.summarize(case.case_id)
    assert summary["text"]
    assert summary["stats"]["events"] > 0

    index = svc.build_index(case)
    recs = generate_recommendations(case, index, lang="en", min_events=2)
    assert isinstance(recs, list)
    assert all(r.priority in ("HIGH", "MEDIUM", "LOW") for r in recs)

    diary = svc.diary(case.case_id)
    assert diary["chain_ok"]
    assert any(e["entry_type"] == "ingest" for e in diary["entries"])


def test_gap_recommendations_for_bank_only_case(tmp_path):
    cfg = InvestigationConfig(case_root=str(tmp_path / "cases"), min_events_per_entity=2)
    svc = InvestigationService(case_root=cfg.case_root, config=cfg)
    case = svc.create_case("Bank Only", language="en")
    bank = tmp_path / "bank_only.csv"
    rows = ["txn_type,amount,recipient_account,account_name,account_no,timestamp"]
    for i in range(20):
        rows.append(f"debit,{100 + i},ACCT2,Alice,ACCT1,2025-10-01 {10 + i}:00:00")
    bank.write_text("\n".join(rows) + "\n", encoding="utf-8")
    svc.ingest(case.case_id, [str(bank)])
    case = svc.load_case(case.case_id)
    index = svc.build_index(case)
    recs = generate_recommendations(case, index, lang="en", min_events=2)
    assert any(r.category == "evidence_gap" for r in recs)
    assert any("CDR" in r.action or "call detail" in r.action.lower() for r in recs)


def test_case_report_generation(service_case, tmp_path):
    svc, case, _ingest, _paths = service_case
    result = svc.report(case.case_id, lang="en")
    files = result["files"]
    assert Path(files["md"]).exists()
    assert Path(files["html"]).exists()
    assert Path(files["json"]).exists()
    assert Path(files["pdf"]).exists()
    md = Path(files["md"]).read_text(encoding="utf-8")
    assert "Draft Case Report" in md
    assert "Evidence inventory" in md
    assert "Case diary" in md
    assert "Draft" in md or "draft" in md

    hindi = svc.report(case.case_id, lang="hi")
    hi_md = Path(hindi["files"]["md"]).read_text(encoding="utf-8")
    assert "ड्राफ्ट केस रिपोर्ट" in hi_md or "केस रिपोर्ट" in hi_md


# ---------------------------------------------------------------------------
# agentic tools: plot / rerun / recheck / deep dive / vision / image generation
# ---------------------------------------------------------------------------
_PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zl1sAAAAASUVORK5CYII="
)


class FakeLLM:
    """Deterministic stand-in so tool tests never hit the network."""

    def __init__(self):
        self.available = True
        self.config = SimpleNamespace(
            model="fake-vision",
            image_model="fake-image",
            to_dict=lambda: {"provider": "fake"},
        )

    def complete(self, *args, **kwargs):
        return "OK"

    def polish_answer(self, question, draft, citations, lang="en", max_citations=6):
        return draft

    def describe_image(self, image, question, lang="en", max_tokens=500):
        return "A test chart with a line."

    def generate_image(self, prompt, steps=4):
        return _PNG_1PX


def test_agent_plot_returns_inline_image(service_case):
    svc, case, _ingest, _paths = service_case
    svc.llm = FakeLLM()
    answer = svc.ask(case.case_id, "Plot the timeline.")
    assert answer.intent == "plot"
    assert answer.images
    assert Path(answer.images[0]["path"]).exists()
    assert answer.to_dict()["images"]


def test_agent_generate_image(service_case):
    svc, case, _ingest, _paths = service_case
    svc.llm = FakeLLM()
    answer = svc.ask(case.case_id, "Generate an image of a suspect board.")
    assert answer.intent == "generate_image"
    assert answer.images
    assert Path(answer.images[0]["path"]).exists()
    assert "synthetic illustration" in answer.answer.lower()


def test_agent_vision_reads_uploaded_image(service_case):
    svc, case, _ingest, _paths = service_case
    svc.llm = FakeLLM()
    data_url = "data:image/png;base64," + base64.b64encode(_PNG_1PX).decode("ascii")
    answer = svc.ask(case.case_id, "What is in this image?", images=[data_url])
    assert answer.intent == "vision"
    assert "Image analysis" in answer.answer
    assert answer.images
    assert Path(answer.images[0]["path"]).exists()


def test_agent_recheck_numbers(service_case):
    svc, case, _ingest, _paths = service_case
    svc.llm = FakeLLM()
    answer = svc.ask(case.case_id, "Recheck the numbers.")
    assert answer.intent == "recheck"
    assert "matching events" in answer.answer
    assert "| metric | value |" in answer.answer


def test_agent_deep_dive(service_case):
    svc, case, _ingest, _paths = service_case
    svc.llm = FakeLLM()
    answer = svc.ask(case.case_id, "Analyze further for structuring.")
    assert answer.intent == "deep_dive"
    assert "Deep dive" in answer.answer


def test_agent_rerun_analysis(service_case):
    svc, case, _ingest, _paths = service_case
    svc.llm = FakeLLM()
    answer = svc.ask(case.case_id, "Rerun the analysis.")
    assert answer.intent == "run_analysis"
    assert "Analysis rerun complete" in answer.answer
    assert answer.artifacts

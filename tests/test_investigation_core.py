"""Unit tests for the investigation layer's core building blocks."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aml.case_store import CaseStore
from aml.diary import append_entry, verify_chain
from aml.i18n import detect_language, normalize_language, t
from aml.index import BM25, tokenize
from aml.case import Case
from aml.query import QueryParser


def _parser() -> QueryParser:
    return QueryParser(
        alias_to_entity={
            "Viktor Duarte": "E1",
            "Elena Unger": "E2",
            "ACCT1001": "E1",
            "6391255614226": "E2",
        },
        entity_names={"E1": ["Viktor Duarte"], "E2": ["Elena Unger"]},
        case_start="2025-01-01T00:00:00+00:00",
        case_end="2025-12-31T23:59:59+00:00",
    )


# ---------------------------------------------------------------------------
# i18n
# ---------------------------------------------------------------------------
def test_language_detection_supported_locales():
    assert detect_language("Show me suspicious transactions") == "en"
    assert detect_language("इस मामले में संदिग्ध क्या है?") == "hi"
    assert detect_language("¿Cuáles son las transacciones sospechosas?") == "es"
    assert detect_language("Quelles transactions sont suspectes ?") == "fr"
    assert detect_language("Zeige alle Überweisungen über 10.000") == "de"
    assert detect_language("Quais transações são suspeitas?") == "pt"


def test_i18n_translation_and_fallback():
    assert "परिणाम" in t("answer.search.header", "hi", n=3)
    assert t("answer.search.header", "xx", n=2).startswith("I found")
    assert normalize_language("en-US") == "en"
    assert normalize_language("unknown") == "en"


# ---------------------------------------------------------------------------
# query parser
# ---------------------------------------------------------------------------
def test_query_parser_search_filters_and_entities():
    plan = _parser().parse("Show transfers above 10,000 from Viktor in the last 30 days.")
    assert plan.intent == "search"
    assert "bank" in plan.sources
    assert plan.amount_range.min == 10000
    assert "E1" in plan.entity_ids
    assert plan.time_range.start is not None
    assert plan.time_range.end is not None


def test_query_parser_intents():
    p = _parser()
    assert p.parse("What looks suspicious?").intent == "anomaly"
    assert p.parse("What is the connection between Viktor and Elena?").intent == "relationship"
    assert p.parse("Who is Viktor Duarte?").intent == "entity_profile"
    assert p.parse("How much money did Viktor send?").intent == "aggregate"
    assert p.parse("What should I investigate next?").intent == "recommend"
    assert p.parse("Summarize the case.").intent == "summarize"
    assert p.parse("Show the case diary.").intent == "diary"
    assert p.parse("Draft a case report.").intent == "report"
    assert p.parse("What evidence do we have?").intent == "evidence_inventory"
    assert p.parse("List all entities.").intent == "entity_list"


def test_query_parser_multilingual_intents():
    p = _parser()
    hi = p.parse("इस मामले में संदिग्ध गतिविधि क्या है?")
    assert hi.intent == "anomaly" and hi.language == "hi"
    es = p.parse("¿Cuáles son las transacciones sospechosas?")
    assert es.intent == "anomaly" and es.language == "es"
    fr = p.parse("Quelles transactions sont suspectes ?")
    assert fr.intent == "anomaly" and fr.language == "fr"
    de = p.parse("Zeige alle Überweisungen über 9.000 von Viktor Duarte.")
    assert de.intent == "search"
    assert "bank" in de.sources
    assert de.amount_range.min == 9000
    assert "E1" in de.entity_ids
    pt = p.parse("O que parece suspeito neste caso?")
    assert pt.intent == "anomaly" and pt.language == "pt"


def test_query_parser_time_ranges():
    p = _parser()
    single = p.parse("What happened on 3 October 2025?")
    assert single.intent == "timeline"
    assert single.time_range.start.startswith("2025-10-03")
    assert single.time_range.end.startswith("2025-10-03")

    month = p.parse("What happened in October 2025?")
    assert month.time_range.start.startswith("2025-10-01")
    assert month.time_range.end.startswith("2025-10-31")

    year = p.parse("What happened in 2025?")
    assert year.time_range.start.startswith("2025-01-01")
    assert year.time_range.end.startswith("2025-12-31")

    relative = p.parse("Show transactions in the last 30 days.")
    assert relative.time_range.start is not None


def test_query_parser_amount_ranges():
    p = _parser()
    between = p.parse("Show transactions between 1k and 2k.")
    assert between.amount_range.min == 1000
    assert between.amount_range.max == 2000
    under = p.parse("Show transactions under $5,000.")
    assert under.amount_range.max == 5000
    over = p.parse("Show transactions above 10,000.")
    assert over.amount_range.min == 10000


def test_query_parser_given_name_resolution():
    plan = _parser().parse("What is the connection between Viktor and Elena?")
    assert "E1" in plan.entity_ids
    assert "E2" in plan.entity_ids


def test_query_parser_agent_tool_intents():
    p = _parser()
    rerun = p.parse("Rerun the analysis.")
    assert rerun.intent == "run_analysis"
    recheck = p.parse("Recheck the numbers.")
    assert recheck.intent == "recheck"
    plot = p.parse("Plot the correlation network.")
    assert plot.intent == "plot" and plot.chart_kind == "network"
    timeline = p.parse("Plot the timeline.")
    assert timeline.intent == "plot" and timeline.chart_kind == "timeline"
    map_plan = p.parse("Show me the map.")
    assert map_plan.intent == "plot" and map_plan.chart_kind == "map"
    deep = p.parse("Analyze further for structuring.")
    assert deep.intent == "deep_dive"
    generated = p.parse("Generate an image of a suspect board.")
    assert generated.intent == "generate_image"
    assert "suspect board" in generated.image_prompt.lower()


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
def test_bm25_ranks_matching_documents():
    docs = [tokenize("alpha beta"), tokenize("beta gamma"), tokenize("delta")]
    bm25 = BM25(docs)
    results = bm25.search(["beta"], top_k=3)
    assert results
    assert {i for i, _ in results} == {0, 1}
    assert results[0][1] > 0


# ---------------------------------------------------------------------------
# diary + case store
# ---------------------------------------------------------------------------
def test_diary_hash_chain_and_tamper_detection(tmp_path):
    store = CaseStore(tmp_path)
    case = store.create_case("Diary Test", language="en")
    append_entry(case, "ingest", "Ingested 3 files", actor="system")
    append_entry(case, "query", "Asked about transfers", actor="analyst")
    store.save_case(case)

    loaded = store.load_case(case.case_id)
    ok, errors = verify_chain(loaded)
    assert ok and not errors

    loaded.diary[0].text = "tampered"
    ok, errors = verify_chain(loaded)
    assert not ok
    assert any("hash mismatch" in e for e in errors)


def test_case_store_evidence_hash_and_duplicate_detection(tmp_path):
    store = CaseStore(tmp_path / "cases")
    case = store.create_case("Evidence Test")
    source = tmp_path / "bank.csv"
    source.write_text("account,timestamp,amount\nA1,2025-01-01,100\n", encoding="utf-8")
    first = store.add_evidence(case, source, source_type="bank", custodian="Alice")
    assert first.sha256
    assert first.evidence_id == "EV0001"
    assert Path(case.root, first.path).exists()

    # same content under a different name is detected as a duplicate
    duplicate = tmp_path / "bank_copy.csv"
    duplicate.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    second = store.add_evidence(case, duplicate, source_type="bank")
    assert second.evidence_id == first.evidence_id
    assert second.metadata.get("duplicates")


def test_case_roundtrip_json(tmp_path):
    store = CaseStore(tmp_path)
    case = store.create_case("Roundtrip", description="desc", language="hi")
    case.entities["E1"] = __import__("aml.case", fromlist=["Entity"]).Entity(
        entity_id="E1", names=["Test Person"], identifiers=["ACCT1"], risk_score=0.8,
    )
    store.save_case(case)
    loaded = store.load_case(case.case_id)
    assert loaded.title == "Roundtrip"
    assert loaded.language == "hi"
    assert loaded.entities["E1"].display_name == "Test Person"
    assert loaded.entities["E1"].risk_score == 0.8

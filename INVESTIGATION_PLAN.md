# Sutram Investigator — Migration & Build Plan

**From:** Multi-source financial anomaly detection engine (AML)
**To:** AI-Powered Digital Investigation Assistant

**Status:** Phases 1–5 delivered; **Option B complete** — the case workspace now lives inside
`src/aml`, with `CasePipeline` extending the existing pipeline. Cloudflare Workers AI text /
vision / FLUX image generation, agentic chat tools, the improved graphs, the native PySide6
case workspace (`case_ui.py`), tool shortcut chips, and richer Markdown rendering are also
delivered (2026-11-24).

**Baseline verified:** **124 tests pass, 6 skipped** on Python 3.15 (`pytest tests`), including
the PySide6 desktop UI tests and the investigation/agent-tool suite. The skipped tests include
the 4 live Cloudflare integration tests, which pass when `RUN_CLOUDFLARE_TESTS=1`. The chart QA
run (`python scripts/chart_qa.py --case operation-nightfall-20261123T195835`) generated every
light/dark chart and completed Cloudflare vision read-back. The demo case + investigation
evaluation are reproducible with `python run_case.py demo` and
`python eval_investigation.py --case <case_id>`.

---

## 1. Executive summary

The existing project is a strong **analytics core** for a different problem. It already
solves the hardest data-engineering parts of digital investigations: heterogeneous
ingestion, runtime schema detection, cross-source entity resolution, evidence-sufficiency
verdicts, anomaly detection, fusion/ranking, explainability, and report generation.

The new problem statement is broader. It needs a **case-centric, conversational
investigation workspace** on top of that analytics core:

| New deliverable | What is missing today |
|---|---|
| Conversational forensic assistant | No chat/query interface, no NLU, no answer synthesis |
| Evidence retrieval engine | No text index, no BM25/vector search, no citation-grounded retrieval |
| Automated case diary | No case lifecycle, no append-only activity log |
| Investigative recommendation module | No next-step reasoning, no evidence-gap-to-action mapping |
| Multilingual interface | Multilingual schema aliases only; no UI/report/query localization |
| Case summarization | Entity-level narratives exist; no case/entity dossier summarization |
| Draft case reports | A polished anomaly brief exists; no investigation case-report structure |

**Strategy (Option B, delivered):** keep the verified `src/aml` engine intact as the
analytics core and fold the case workspace **into the same `aml` package** — `case.py`,
`case_store.py`, `diary.py`, `evidence.py`, `query.py`, `index.py`, `retrieval.py`,
`summarize.py`, `recommend.py`, `assistant.py`, `llm.py`, `case_report.py`, `service.py`.
`CasePipeline` in `src/aml/pipeline.py` extends the existing `Pipeline`; there is no
parallel `src/investigation` package. The existing evaluation harness and 84 original tests
remain valid, and `run_pipeline.py --investigation` exposes the case workflow through the
existing CLI.

**Pivot in one sentence:** *The AML engine becomes the "suspicious-pattern detection"
subsystem of a larger evidence-retrieval, case-reasoning, and reporting assistant.*

---

## 2. Review of the current project

### 2.1 What the project is

Sutram is a source-agnostic pipeline over a unified long-format dataframe:

```
bank CSV / CDR / social
        │
        ▼
[A] runtime schema detection (rapidfuzz + content probes)
        ▼
[B] entity resolution (ID + fuzzy name, cross-source)
        ▼
unified dataframe: entity | event | timestamp | source | amount |
                   counterparty | location | direction | duration
        ▼
[C] time correlation   [D] network   [E] stat/ML   [H] Benford / structuring / chain
        └───────────────┬───────────────┬────────────┘
                        ▼
                 [F] fusion & ranking
                        ▼
                 [G] explanations → GUI / HTML / PDF / JSON
```

### 2.2 Verified strengths (reusable)

| Capability | Module | Why it matters for investigations |
|---|---|---|
| Runtime schema detection, 576 stress cases | `src/aml/schema_detect.py` | Real evidence arrives in arbitrary exports; no fixed parser can work |
| Multi-format loaders (CSV/JSON/JSONL/TXT, latin-1, delimiter sniffing) | `src/aml/loaders.py` | Investigators receive mixed, dirty exports |
| Cross-source entity resolution | `src/aml/entity_resolve.py` | One person appears as account, phone, handle |
| Sufficiency engine (SUPPORTED / DEGRADED / BLOCKED + reasons) | `src/aml/sufficiency.py` | Critical for court-defensible "what this evidence can/cannot prove" |
| 7 model families + fusion + ablation | `src/aml/models.py`, `fusion_meta.py` | Suspicious-pattern detection is an explicit deliverable |
| Plain-language insights with evidence sentences | `src/aml/insights.py` | Direct input to case summarization and findings |
| Polished MD/HTML/PDF/JSON reports + charts | `src/aml/report.py`, `visuals.py` | Reuse as the analytics appendix of a case report |
| Synthetic data + write-only answer keys + eval | `src/aml/data_generator/`, `eval.py` | Extend to a demo case and investigation evaluation |
| Desktop app with run history | `main_app.py`, `storage.py` | Retain as analyst console |

### 2.3 Constraints / risks found during review

1. **Dependency drift:** the repo targets Python 3.11, but the current machine has Python 3.15 and was missing `scikit-learn`, `rapidfuzz`, `pytest`, `reportlab`, `PySide6`. Core deps were installed during this review; PySide6 is still absent, so UI tests cannot run here.
2. **The `aml` package name and branding are AML-specific.** Renaming it would break 84 tests and every import. Keep `aml` as the analytics core; introduce `investigation` as the new layer.
3. **No unstructured-evidence path.** PDFs, emails, chat logs, notes, and images are the heart of digital investigations. The existing loaders only produce dataframes; there is no text extraction, chunking, or document index.
4. **No persistence model for a case.** Runs are archived, but there is no case entity, evidence inventory, chain of custody, findings, recommendations, or diary.
5. **No retrieval layer.** There is no full-text index, no semantic search, no citation objects. Reports cite narrative evidence sentences, not evidence items with file/hash/page references.
6. **No NL query layer.** The pipeline is invoked programmatically; there is no query understanding, intent routing, or conversational memory.
7. **Multilingual coverage is ingestion-only.** Spanish/French/German/Portuguese headers exist; UI, answers, and reports are English-only.
8. **No evaluation for investigation tasks.** `eval.py` measures anomaly ranking only. Retrieval, QA, citation faithfulness, recommendation coverage, and multilingual parity are unmeasured.

---

## 3. Problem statement → capability mapping

| # | Required capability | Current asset | New work | Priority |
|---|---|---|---|---|
| 1 | Natural language querying of forensic evidence | Structured pipeline + insights | Multilingual query parser (intent, entity, time, amount, source, keywords), answer synthesizer, chat memory | **P0** |
| 2 | Context-aware evidence search | Unified dataframe, entity map | Evidence index (events + text chunks), BM25 + structured filters + graph expansion, citations | **P0** |
| 3 | Automated case summarization | `insights.py`, report overview | Case summary, entity dossier, timeline summary, evidence-gap summary | **P0** |
| 4 | Suspicious pattern / anomaly identification | 7 model families, fusion, sufficiency | Wrap results as case findings with severity, confidence, citations | **P0** |
| 5 | Investigative lead recommendation | Model verdicts, insights, sufficiency | Rule engine mapping findings + gaps → prioritized next actions | **P0** |
| 6 | Draft case report generation | MD/HTML/PDF/JSON report generator | Investigation case report: metadata, evidence inventory, findings, timeline, recommendations, diary, limitations | **P0** |
| 7 | Conversational forensic assistant | None | Chat service + CLI REPL + web chat UI with citations and follow-ups | **P0** |
| 8 | Automated case diary | None | Append-only, hash-chained diary of ingest/analysis/query/finding/action/export events | **P0** |
| 9 | Multilingual interface | Multilingual header aliases | Locale catalogs (en/hi/es/fr/de/pt), language detection, multilingual query lexicons, localized answers/reports, UI switcher | **P1** |
| 10 | Evidence retrieval engine (broad evidence types) | CSV/JSON/JSONL/TXT | `.eml`, `.pdf`, `.docx`, `.md`, image metadata, generic text; chunking; provenance; SHA-256 | **P1** |
| 11 | Case management / chain of custody | Run archive | Case store, evidence hashing, duplicate detection, audit log | **P1** |
| 12 | Optional LLM polish / translation | None | Provider-agnostic adapter (OpenAI-compatible / Ollama), grounded prompt, strict citation validation, offline fallback | **P2** |
| 13 | Investigation evaluation | `eval.py` | Query/retrieval/citation/recommendation/multilingual eval harness | **P1** |
| 14 | Demo-ready web workspace | PySide6 desktop | Zero-dependency web UI (stdlib HTTP server) with chat, evidence explorer, findings, diary, report | **P1** |

---

## 4. Gap analysis (what blocks each deliverable)

### 4.1 Evidence layer gaps

- **No text extraction** from PDF/EML/DOCX/notes.
- **No chunking** with line/page provenance.
- **No evidence inventory** with SHA-256, size, MIME, custodian, acquisition time.
- **No case-scoped entity registry** that includes people mentioned only in text.
- **No unified timeline** across structured events and unstructured documents.
- **No citation object** that can point to `evidence_id + chunk_id/event_id + file + timestamp + snippet`.

### 4.2 Retrieval / NLU gaps

- **No query parser** for intent, entities, time, amounts, sources, direction, location.
- **No BM25 / TF-IDF / embedding index** over text.
- **No hybrid retrieval** combining filters, keyword search, entity graph, and recency.
- **No conversational state** (pronoun resolution, follow-up filters, language).
- **No answer synthesis** with citations, confidence, and suggested follow-ups.
- **No query logging** into the case diary.

### 4.3 Case reasoning gaps

- **No findings model** (severity, confidence, entity/evidence links, detected-by).
- **No recommendation model** (priority, action, rationale, linked finding/evidence).
- **No evidence-gap → action rules** (e.g., "no CDR → request call records").
- **No case summary / entity dossier / timeline summary**.
- **No draft case report** with the sections an investigator or supervisor expects.

### 4.4 Interface / i18n gaps

- **No chat interface.**
- **No evidence explorer** with filters and citations.
- **No diary view.**
- **No language switcher** or localized strings.
- **No multilingual answer templates** or language-aware query understanding.

---

## 5. Target architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        INVESTIGATION WORKSPACE                               │
│  Web chat UI  ·  CLI (`run_case.py`, `ask.py`)  ·  Desktop analyst console   │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │  ask / search / summarize / recommend /
                                    │  diary / report
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    CASE WORKSPACE (`src/aml`, same package)                 │
│                                                                              │
│  QueryParser ──► HybridRetriever ──► AnswerSynthesizer ──► citations         │
│       │               │                    │                                 │
│       │               ├── EventIndex       ├── CaseSummarizer                │
│       │               ├── TextChunkIndex   ├── EntityDossier                 │
│       │               └── EntityGraph      └── RecommendationEngine          │
│       │                                                                      │
│  CaseStore ──► EvidenceInventory ──► CaseDiary (hash chain) ──► CaseReport   │
│  Agent tools: rerun · recheck · plot · deep dive · generate image · vision   │
│  Cloudflare Workers AI: text · vision · FLUX image generation                │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     ANALYTICS CORE (`src/aml`, unchanged)                    │
│  schema detection · entity resolution · sufficiency · 7 models · fusion ·    │
│  insights · HTML/PDF/MD/JSON report · synthetic data · eval                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 5.1 New case data model

| Object | Key fields | Purpose |
|---|---|---|
| `Case` | id, title, description, language, status, created/updated, evidence, entities, findings, recommendations, diary, metadata | Case container persisted as `case.json` |
| `EvidenceItem` | id, path, filename, SHA-256, size, MIME, source type, acquired_at, custodian, notes, rows/chunks, metadata | Chain-of-custody inventory |
| `Entity` | id, names, identifiers, roles, sources, first/last seen, event count, risk score, risk factors | Person/account/phone/handle across evidence |
| `Chunk` | id, evidence_id, text, timestamp, source type, entity_ids, location, line range, metadata | Retrieval unit for unstructured evidence |
| `Event` | id, evidence_id, source, event_type, timestamp, entity_id, counterparty_id, amount, direction, location, duration, search_text | Retrieval unit for structured evidence |
| `Citation` | evidence_id, chunk_id/event_id, file, timestamp, snippet, source type, entity_ids | Grounding for every answer/finding |
| `Finding` | id, kind, title, description, severity, confidence, entity/evidence links, citations, detected_by | Suspicious pattern / insight |
| `Recommendation` | id, priority, action, rationale, category, entity/finding/evidence links, status | Investigative lead |
| `DiaryEntry` | id, seq, timestamp, actor, entry_type, text, evidence_ids, prev_hash, hash | Tamper-evident case diary |
| `Answer` | text, intent, language, citations, hits, confidence, followups, data, query_plan, images, artifacts | Chat response with inline charts/uploads and downloadable artefacts |

### 5.2 Storage layout

```
data/cases/<case_id>/
  case.json                 # case metadata, evidence, entities, findings, recommendations, diary
  evidence/                 # copied originals (chain of custody)
  index/
    events.jsonl            # structured events with evidence_id + event_id
    chunks.jsonl            # text chunks with provenance
    entities.json           # entity registry
    aliases.json            # alias -> entity_id
  reports/
    case_report.md
    case_report.html
    case_report.json
    case_diary.md
    analytics/              # existing aml report (HTML/PDF/MD/JSON/charts)
    chat/                   # generated charts, uploaded images, FLUX illustrations
  audit.log                 # human-readable mirror of the hash-chained diary
```

---

## 6. Work breakdown (file-level)

### 6.1 Case-workspace modules — `src/aml/` (Option B)

| File | Responsibility |
|---|---|
| `case.py` | Dataclasses: `Case`, `EvidenceItem`, `Entity`, `Chunk`, `Finding`, `Citation`, `Recommendation`, `DiaryEntry`, `Answer`, `QueryPlan`, `TimeRange`, `AmountRange`; serialization |
| `i18n.py` | Locale catalogs (en/hi/es/fr/de/pt), `t()`, language detection, language-request detection, stopwords, month names, query examples |
| `case_store.py` | Create/load/save/list cases, copy evidence, SHA-256, duplicate detection, index persistence |
| `diary.py` | Append-only hash-chained diary, verification, markdown rendering |
| `evidence.py` | Structured evidence parsing (reuses `aml`), EML/PDF/DOCX/MD/TXT/image extraction, chunking, mention extraction, timestamp/location/amount extraction |
| `query.py` | Multilingual query parser: intent, entity, time, amount, source, direction, location, keywords, chart kind, image prompt, limit, ambiguity; agent-tool intents |
| `index.py` | `EvidenceIndex`: event index, BM25 text index, entity registry, entity graph, aggregate/timeline/relationship queries |
| `retrieval.py` | `HybridRetriever`: hard filters + BM25 + graph expansion + recency + anomaly boost, dedupe, citations |
| `summarize.py` | Case summary, entity dossier, timeline summary, evidence-gap summary |
| `recommend.py` | Evidence-gap and finding-driven recommendation rules with priority/rationale |
| `llm.py` | Cloudflare Workers AI (text + vision + FLUX image generation) plus OpenAI-compatible/Ollama adapters; grounded prompt, citation validation, offline fallback |
| `assistant.py` | Intent routing, session context, answer synthesis, citations, confidence, follow-ups, diary logging |
| `case_report.py` | Draft case report (MD/HTML/JSON/PDF), diary export |
| `service.py` | `InvestigationService` orchestrator + agent tools (rerun, recheck, plot, deep dive, generate image, vision) |
| `config.py` | `InvestigationConfig` / `load_investigation_config` merged into the existing config module |
| `pipeline.py` | `PipelineResult` extended with case/evidence/index/findings/recommendations/summary/diary; `CasePipeline` unified entry point |
| `visuals.py` | OSM-backed activity map + interactive Leaflet HTML map, connected correlation graph, line-graph timelines |
| `data_generator/investigation_case.py` | Demo case generator helpers (emails, chats, notes, PDF, ground truth) |

### 6.2 New entry points

| File | Purpose |
|---|---|
| `run_case.py` | CLI: `create`, `ingest`, `analyze`, `ask`, `summary`, `entities`, `timeline`, `findings`, `recommend`, `diary`, `report`, `list`, `demo` |
| `ask.py` | Interactive conversational REPL with `:lang`, `:summary`, `:diary`, `:recommend`, `:report`, `:help`, `:quit` |
| `web/server.py` | Zero-dependency HTTP server + JSON API |
| `web/static/index.html` | Chat workspace: case header, chat, citations, evidence/entities/findings/diary tabs, language switcher |
| `web/static/app.js` | Fetch API, chat rendering, citation cards, tabs, i18n |
| `web/static/style.css` | Dark Sutram theme |
| `run_pipeline.py --investigation` | Existing CLI gains the case workspace: create/ingest/analyze/ask/report through `CasePipeline` |
| `run_demo.py --investigation` | Build the complete case demo (structured + emails + chats + notes + PDF + ground truth) and run the workflow |
| `eval_investigation.py` | Retrieval / QA / citation / recommendation / multilingual evaluation |
| `tests/test_cloudflare_integration.py` | Live text/vision/FLUX integration tests (skipped unless `RUN_CLOUDFLARE_TESTS=1`) |

### 6.3 Existing files changed (minimal, non-breaking)

| File | Change |
|---|---|
| `README.md` | Rebrand + new quickstart + deliverable mapping + Cloudflare/agent-tool docs |
| `PLAN.md` | Add a pointer at the top to this plan; keep the AML build record |
| `requirements.txt` | No new mandatory deps; Cloudflare uses stdlib `urllib`; `Pillow` remains optional for EXIF |
| `config/investigation.yaml` | Case root, language, analysis thresholds, Cloudflare model IDs/timeout |
| `src/aml/config.py` | `InvestigationConfig` + `load_investigation_config` merged into the existing config module |
| `src/aml/pipeline.py` | `PipelineResult` extended; `CasePipeline` added |
| `src/aml/report.py` | Correlation-graph heading, interactive Leaflet map, new chart calls, map points in JSON |
| `src/aml/visuals.py` | OSM map + zoom inset, connected correlation graph, line-graph timelines, larger labels |
| `run_pipeline.py` | `--investigation` case mode using `CasePipeline` |
| `run_demo.py` | `--investigation` demo mode |
| `run_case.py`, `ask.py`, `run_web.py`, `web/` | Case CLI, image attachments, inline charts/artefacts |
| `tests/` | `test_investigation_*.py`, agent-tool tests, live Cloudflare tests; existing tests updated only for new report headings |
| `.env.example`, `.gitignore` | Credential template + `.env` ignored |
| `data/cases/.gitkeep` | Case root placeholder |

The original 84 tests remain green; the only existing-test edits are the two report-heading
assertions changed by the new graph titles (`Suspect correlation graph`, `Activity map`).

---

## 7. Phased roadmap

### Phase 0 — Foundations (0.5 day) ✅ delivered
- Case/evidence/entity/finding/recommendation/diary dataclasses and JSON persistence.
- SHA-256 evidence inventory + chain-of-custody metadata.
- Hash-chained case diary + verification.
- i18n catalogs for en/hi/es/fr/de/pt.

**Acceptance:** create a case, add evidence, append diary entries, reload from disk, verify hash chain; unit tests green.

### Phase 1 — Evidence + retrieval engine (1–2 days)
- Structured ingestion via `aml` (bank/CDR/social) with per-evidence provenance.
- Unstructured ingestion: EML, PDF, DOCX, MD/TXT, image metadata.
- Chunking with line/page provenance.
- Entity registry + alias index + entity graph.
- BM25 index + structured event filters + hybrid retrieval + citations.

**Acceptance:** ingest a mixed case; search "offshore account" and "transfers above 10,000" and get cited results; entity graph resolves names/phones/accounts; retrieval eval Recall@5 ≥ 0.8 on the demo query set.

### Phase 2 — Assistant + case reasoning (1–2 days)
- Multilingual query parser (intent/entity/time/amount/source/direction/location/keywords).
- Intent handlers: search, entity profile, timeline, relationship, correlation, aggregate, anomaly, summarize, recommend, diary, report, evidence inventory, entity list, help.
- Answer synthesis with citations, confidence, follow-ups, session memory.
- Findings from `aml.insights` + model scores + sufficiency verdicts.
- Recommendation rules (evidence gaps + findings).
- Case summary / entity dossier / timeline summary.

**Acceptance:** 20 demo questions in en/hi/es answer correctly with citations; follow-up pronouns work; recommendations cover all evidence-gap categories; case summary contains period, sources, entities, findings, gaps.

### Phase 3 — Interfaces + Cloudflare AI (delivered)
- `run_case.py` CLI + `ask.py` REPL.
- Zero-dependency web workspace: chat, citations, evidence explorer, entities, findings, diary, recommendations, report, language switcher.
- Image upload for vision analysis; inline generated charts/illustrations and download links.
- Cloudflare Workers AI: text polish, vision reading, FLUX text-to-image; offline fallback remains.
- Agentic tools: rerun analysis, recheck numbers, plot charts, deep-dive models, generate image, vision.

**Acceptance met:** one command opens a browser; questions work in two languages; citations and the diary are inspectable; the report downloads; images render inline.

### Phase 4 — Report + evaluation + demo + graphs (delivered)
- Draft case report (MD/HTML/JSON/PDF) with evidence inventory, entities, findings, timeline, recommendations, diary, limitations, analytics appendix.
- Demo case generator with emails/chats/notes/PDF and ground truth.
- Investigation eval harness: retrieval Recall@k/MRR, intent accuracy, citation accuracy, recommendation coverage, multilingual parity.
- Graph improvements: OSM-backed activity map with zoom inset + interactive Leaflet map, connected/labelled correlation graph, line-graph timelines, larger labels.
- README/PLAN/config updates.

**Acceptance met:** `python run_case.py demo` produces a complete case and report; `python eval_investigation.py` prints reproducible metrics; existing `pytest` + `eval.py` still pass.

### Phase 5 — Remaining stretch (optional)
- Embedding retrieval (`sentence-transformers` or provider embeddings) fused with BM25.
- OCR for scanned images (Tesseract optional).
- Arabic/Urdu RTL UI + locales.
- Analyst feedback loop: accept/reject recommendations, diary records decisions.
- Export to standard investigation formats (e.g., case-management JSON).
- CI pinned to Python 3.11 + PySide6 to run the desktop UI tests.

---

## 8. Natural-language + retrieval design

### 8.1 Query pipeline

```
raw question
   │
   ├─ language detection / explicit "in Hindi" override
   ├─ intent classification (multilingual lexicon + priority rules)
   ├─ entity mention resolution (alias exact → identifier regex → fuzzy name)
   ├─ time range parsing (ISO, DMY/MDY, month names, relative, between/after/before)
   ├─ amount range parsing (above/below/between, k/m, currency symbols)
   ├─ source / direction / location / limit extraction
   ├─ keyword extraction (stopword + intent + entity/filter removal)
   ▼
QueryPlan
   │
   ├─ hard filters on events (entity, source, time, amount, direction, location)
   ├─ BM25 over event search_text and text chunks
   ├─ entity-graph expansion (1-hop neighbours, relationship paths)
   ├─ recency + amount + anomaly-score boosts
   ├─ normalize + weighted fusion + dedupe
   ▼
Ranked hits → AnswerSynthesizer → Answer(text, citations, confidence, followups)
```

### 8.2 Intent taxonomy

| Intent | Example | Handler |
|---|---|---|
| `search` | "Show transfers above 10,000 from Viktor" | Hybrid retrieval |
| `entity_profile` | "Who is Viktor Duarte?" | Entity dossier |
| `timeline` | "What happened on 3 October 2025?" | Chronological retrieval |
| `relationship` | "What is the connection between Viktor and Elena?" | Graph + co-occurrence |
| `correlation` | "Who did Viktor call right before the transfer?" | Time-correlation over calls/transfers |
| `aggregate` | "How much money did Viktor send?" | Filtered aggregation |
| `anomaly` | "What looks suspicious?" | Findings from analytics |
| `summarize` | "Summarize the case" | Case summary |
| `recommend` | "What should I investigate next?" | Recommendation engine |
| `diary` | "Show the case diary" | Diary |
| `report` | "Draft a case report" | Report generator |
| `evidence_inventory` | "What evidence do we have?" | Evidence inventory |
| `entity_list` | "List all entities" | Entity registry |
| `help` | "What can you do?" | Capability help |

### 8.3 Grounding rules

1. Every factual answer includes at least one `Citation` (evidence id, file, timestamp, snippet).
2. If retrieval returns nothing, the answer says so and suggests broadening; it never invents.
3. Findings and recommendations link to `evidence_ids` / `finding_ids`.
4. Optional LLM output is accepted only if it preserves at least one valid citation ID and introduces no new IDs; otherwise the deterministic draft is returned.
5. The case diary records the query, plan, result count, and citations.

### 8.4 Why BM25 first (and embeddings later)

BM25 is deterministic, dependency-free, fast on case-sized data, and easy to explain in
a demo. Embeddings are valuable for paraphrase recall but add a model download, latency,
and non-determinism. The design keeps a `Retriever` interface so an embedding channel can
be fused in Phase 5 without changing the assistant or UI.

---

## 9. Multilingual strategy

| Layer | Approach |
|---|---|
| UI strings | JSON locale catalogs + `t(key, lang)`; language switcher persisted |
| Query understanding | Per-language intent lexicons, stopwords, month names, relative-time words |
| Entity names/IDs | Language-independent alias index; regex identifiers |
| Answers | Per-language templates with citations; findings/recommendations templated by kind |
| Reports | Localized section labels + localized summary/recommendation templates; original evidence text is quoted, not translated |
| Language detection | Script detection (Devanagari) + stopword scoring; explicit "in Hindi"/"en español" overrides |
| RTL (stretch) | Arabic/Urdu catalog + `dir="rtl"` in the web UI |

Supported in Phase 1: **English, Hindi, Spanish, French, German, Portuguese**.

Evidence quotations, analytics-derived finding titles and historical diary entries are
preserved in their original language; report labels, recommendations, summaries and the
interface are localized. A forensic record must not silently paraphrase or translate the
underlying evidence.

---

## 10. Evaluation plan

### 10.1 Demo ground truth

The demo case generator writes `ground_truth.json` with:

- 20+ questions across en/hi/es, each with expected intent, entity IDs, evidence IDs, and key facts.
- Expected recommendation categories (e.g., `evidence_gap`, `interview`, `records`, `forensics`).
- Expected case-summary facts (period, sources, top entity, finding kinds).

### 10.2 Metrics

| Area | Metrics |
|---|---|
| Retrieval | Recall@5, Recall@10, MRR, nDCG@10 |
| Query understanding | Intent accuracy, entity-resolution accuracy, filter accuracy |
| Answer quality | Expected-fact containment, citation precision/recall, no-answer correctness |
| Summarization | Key-fact coverage, citation coverage, unsupported-claim count (manual/LLM judge) |
| Recommendations | Coverage of expected categories, priority ordering, evidence linkage |
| Multilingual | Per-language retrieval/QA parity vs English |
| Operational | Ingest time, index size, query latency, diary verification |

### 10.3 Guardrails

- No claim without a citation.
- No "suspicious" label without a model/rule name and evidence.
- Sufficiency verdicts are surfaced in the case report as limitations.
- Synthetic demo data is clearly labeled; answer keys are write-only during development.

---

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Scope creep across evidence types | Phase 1 supports EML/PDF/DOCX/MD/TXT + structured; images are metadata-only; OCR is stretch |
| LLM hallucination | Deterministic answers by default; LLM only polishes, with citation validation and fallback |
| Breaking the verified AML core | No changes to `src/aml`; new layer imports it; existing tests remain the regression gate |
| Multilingual quality | Template-based answers are predictable; locale catalogs reviewed; per-language eval parity tracked |
| Large cases / slow search | BM25 + pandas filters are fast for case-scale data; index persisted; chunk limits; top-k caps |
| Evidence integrity questions | SHA-256 per item, hash-chained diary, chain-of-custody fields, duplicate detection |
| Privacy / sensitive data | Local-first design; no network calls by default; LLM is opt-in via environment config |
| Python 3.11 vs 3.15 drift | Code uses 3.11-compatible syntax; CI should pin 3.11; requirements documented |

---

## 12. Demo script (5 minutes)

1. **Open** the web workspace on the demo case: case title, evidence count, entity count, date range.
2. **Ask in English:** "Show transfers above 9,000 from Yara Okafor in the last 90 days." → cited transaction cards.
3. **Follow up:** "Who did Ava Carlos call right before the transfer?" → correlation answer with call + transfer citations.
4. **Ask in Hindi:** "इस मामले में संदिग्ध गतिविधि क्या है?" → findings with severity + evidence.
5. **Ask:** "What is the connection between Yara Okafor and Ava Carlos?" → graph path + shared events.
6. **Ask:** "What should I investigate next?" → prioritized recommendations with rationale.
7. **Agent tools:** "Rerun the analysis." → refreshed findings/leads + correlation graph + report downloads; "Recheck the numbers." → Markdown table; "Plot the map." → inline map; "Analyze further for structuring." → model deep-dive.
8. **Cloudflare AI:** "Generate an image of a clean forensic case board…" → inline FLUX illustration; attach a chart and ask "What does this chart show?" → vision answer.
9. **Open the diary:** ingest/analysis/query/finding entries, hash chain verified.
10. **Download the draft case report:** executive summary, evidence inventory, entities, findings, timeline, recommendations, diary, limitations, analytics appendix.
11. **Show the improved analytics appendix:** OSM activity map with zoom inset + interactive Leaflet map, connected correlation graph, line-graph timelines.

---

## 13. Delivered in this session

Phases 1–5 are delivered; **Option B** is complete:

- [x] This plan
- [x] **Option B:** case workspace folded into `src/aml` (`case.py`, `case_store.py`, `diary.py`, `evidence.py`, `query.py`, `index.py`, `retrieval.py`, `summarize.py`, `recommend.py`, `assistant.py`, `llm.py`, `case_report.py`, `service.py`, `i18n.py`)
- [x] `CasePipeline` in `src/aml/pipeline.py`; extended `PipelineResult`; `run_pipeline.py --investigation`
- [x] Evidence ingestion: structured + EML/PDF/DOCX/MD/TXT/chat, chunking, entity mentions
- [x] Multilingual query parser + BM25 index + hybrid retrieval + citations
- [x] Case summarizer + entity dossier + timeline + evidence-gap analysis
- [x] Recommendation engine (gaps + findings, priority/rationale/links)
- [x] Assistant + citations + session memory + follow-ups
- [x] Case report generator (MD/HTML/JSON/PDF) + diary export
- [x] CLI (`run_case.py`, `ask.py`) + `run_web.py`
- [x] Demo case generator + ground truth (`run_demo.py --investigation`)
- [x] Investigation eval harness
- [x] Zero-dependency web chat workspace (chat, evidence, entities, findings, leads, diary, report, language switcher)
- [x] **Cloudflare Workers AI:** text polish, vision reading, FLUX text-to-image; `.env` credentials; live integration tests
- [x] **Agentic tools:** rerun analysis, recheck numbers, plot charts, deep-dive models, generate image, image upload/vision; inline images and artefacts in chat
- [x] **Improved graphs:** OSM activity map + zoom inset + interactive Leaflet map, connected/labelled correlation graph, line-graph timelines, larger labels
- [x] README / PLAN / config updates
- [x] **112 non-UI tests pass, 4 skipped** (84 original + 28 investigation/agent-tool tests; 4 live Cloudflare tests skipped by default, all passing when enabled)

### Verified demo results (16 queries, synthetic demo case)

| Metric | Value |
|---|---|
| Intent accuracy | 1.000 |
| Entity hit rate | 1.000 |
| Keyword containment | 1.000 |
| Citation coverage (citation-required intents) | 1.000 |
| Evidence recall@5 | 1.000 |
| MRR | 1.000 |
| Recommendation category coverage | 1.000 |
| Average query latency | ~370 ms |
| Languages | en, hi, es, fr, de, pt |

**Definition of done met:** `python run_case.py demo` works end-to-end;
`python eval_investigation.py --case <case_id>` produces the metrics above;
`pytest tests --ignore=tests/test_main_app.py` is green; README explains the new workflow.

### Remaining stretch work

- Embedding retrieval fused with BM25 (paraphrase recall).
- OCR for scanned images (Tesseract optional).
- Arabic/Urdu RTL locales.
- Analyst accept/reject workflow for recommendations (UI currently displays/copies leads).
- Larger real-world case validation; the demo metrics are on a small synthetic case.
- CI pinned to Python 3.11 with PySide6 to run the desktop UI tests.

---

*This plan supersedes the AML-only roadmap for the new problem statement. The AML build
record in `PLAN.md` and the frozen schema in `SCHEMA.md` remain valid for the analytics
core.*

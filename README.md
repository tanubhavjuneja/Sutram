# Sutram Investigator — AI-Powered Digital Investigation Assistant

Sutram Investigator turns the project's verified multi-source anomaly engine into a
**case-centric investigation workspace**: ingest mixed evidence, ask questions in natural
language, retrieve cited evidence, summarize the case, surface suspicious patterns,
generate investigative leads, keep a tamper-evident case diary, and draft a case report.

It is built on top of the existing **Sutram analytics core** (`src/aml`), which remains the
suspicious-pattern detection subsystem. The analytics core is unchanged and still passes
its original evaluation harness; the new investigation layer lives in `src/investigation`.

> **Why this mapping works:** the hard part of a digital investigation assistant is not the
> chat bubble — it is trustworthy evidence handling. This project already had runtime schema
> detection, cross-source entity resolution, evidence-sufficiency verdicts, seven model
> families, fusion, explainability and reproducible evaluation. The new layer adds the case,
> retrieval, conversation, recommendation, diary and report capabilities on top.

---

## What it does

| Problem-statement deliverable | Implementation |
|---|---|
| **Conversational forensic assistant** | `src/investigation/assistant.py` + web chat + CLI REPL; intent routing, session memory, follow-ups, grounded answers |
| **Evidence retrieval engine** | `src/investigation/index.py` + `retrieval.py`: BM25 over events and text chunks, structured filters, entity-graph expansion, recency/amount/risk boosts, citations |
| **Automated case summarization** | `src/investigation/summarize.py`: case overview, entity dossier, timeline summary, evidence-gap summary |
| **Suspicious patterns / anomalies** | Existing `src/aml` 7 model families + fusion, converted into case findings with severity, confidence, evidence links and citations |
| **Investigative lead recommendation** | `src/investigation/recommend.py`: evidence-gap rules + finding-driven next actions with priority, rationale and linked evidence |
| **Draft case report generation** | `src/investigation/case_report.py`: Markdown / HTML / JSON / PDF case report with evidence inventory, entities, findings, timeline, recommendations, diary, limitations and analytics appendix |
| **Automated case diary** | `src/investigation/diary.py`: append-only, hash-chained log of ingest/analysis/query/finding/recommendation/export actions; tamper verification |
| **Multilingual interface** | `src/investigation/i18n.py`: English, Hindi, Spanish, French, German, Portuguese; language detection, multilingual query lexicons, localized answers and report labels |
| **Case / evidence / chain of custody** | `src/investigation/case_store.py`: case directory, SHA-256 evidence inventory, duplicate detection, audit log |
| **Broad evidence ingestion** | `src/investigation/evidence.py`: bank/CDR/social CSV/JSON/JSONL/TXT + `.eml`, `.pdf`, `.docx`, `.md`, chat logs, image metadata |

---

## Quickstart

Requires Python 3.11+ (the demo was validated on Python 3.15 with the dependencies below).

```bash
pip install -r requirements.txt

# 1) Generate the synthetic demo case, run analytics, recommendations and report
python run_case.py demo

# 2) Open the conversational web workspace
python run_web.py --open
#    or: python web/server.py --case-root data/cases --port 8000

# 3) Ask from the terminal
python run_case.py list
python run_case.py ask <case_id> "What looks suspicious in this case?"
python run_case.py chat <case_id> --lang hi
python run_case.py summary <case_id>
python run_case.py recommend <case_id>
python run_case.py diary <case_id>
python run_case.py report <case_id> --lang en

# 4) Evaluate the investigation assistant
python eval_investigation.py --case <case_id>

# 5) The original analytics core still works
python run_demo.py
python eval.py
pytest tests --ignore=tests/test_main_app.py
# install PySide6 to run the desktop UI tests as well
```

The demo generates `data/demo_case/` (structured + emails + chats + note + PDF + ground
truth) and `data/cases/<case_id>/` (case record, evidence copies, index, reports, audit log).

---

## Demo walkthrough (5 minutes)

1. `python run_case.py demo` — builds **Operation Nightfall** from synthetic bank/CDR/social
   data plus emails, chat logs, an investigator note and a PDF account-opening form.
2. `python run_web.py --open` — opens the dark investigation workspace.
3. Ask: **"Show transfers above 9,000 from Yara Okafor in the last 90 days."** → cited
   transaction cards and the exact evidence files.
4. Follow up: **"Who did Ava Carlos call right before the transfer?"** → call/transfer
   correlation with timestamps and citations.
5. Ask: **"What is the connection between Yara Okafor and Ava Carlos?"** → direct/indirect
   path, shared documents and co-occurrence.
6. Switch the language selector to **हिन्दी** and ask **"इस मामले में संदिग्ध गतिविधि क्या है?"**
   → localized findings with the same evidence links.
7. Open **Leads** → prioritized next steps: interview, records, tower/CCTV, corroboration,
   evidence gaps.
8. Open **Case diary** → ingest/analysis/query/finding/export entries with a verified hash chain.
9. Open **Report** → generate and download the draft case report (MD/HTML/PDF/JSON) plus the
   original analytics brief as an appendix.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Web chat workspace  ·  CLI (`run_case.py`, `ask.py`)  ·  Desktop console    │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ ask / search / summarize / recommend /
                                │ diary / report
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                 INVESTIGATION LAYER (`src/investigation`)                    │
│                                                                              │
│  QueryParser ──► HybridRetriever ──► AnswerSynthesizer ──► citations         │
│       │               │                    │                                 │
│       │               ├── EventIndex       ├── CaseSummarizer                │
│       │               ├── TextChunkIndex   ├── EntityDossier                 │
│       │               └── EntityGraph      └── RecommendationEngine          │
│       │                                                                      │
│  CaseStore ──► EvidenceInventory ──► CaseDiary (hash chain) ──► CaseReport   │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                   ANALYTICS CORE (`src/aml`, unchanged)                      │
│  runtime schema detection · entity resolution · sufficiency · 7 models ·     │
│  fusion · insights · HTML/PDF/MD/JSON anomaly brief · synthetic data · eval  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Case storage

```
data/cases/<case_id>/
  case.json                 # metadata, evidence, entities, findings, leads, diary
  evidence/                 # copied originals + SHA-256 inventory
  index/
    events.jsonl            # structured events with evidence provenance
    chunks.jsonl            # text chunks with line/page provenance
    entities.json           # entity registry
    aliases.json            # alias -> entity id
  reports/
    case_report.md / .html / .json / .pdf
    case_diary.md
    analytics/              # existing anomaly brief (HTML/PDF/MD/JSON/charts)
  audit.log                 # human-readable mirror of the hash-chained diary
```

---

## Evidence types

**Structured** (parsed by the existing adaptive schema engine):

- bank / transaction exports
- CDR / call detail records
- social-media posts

**Unstructured** (parsed by the investigation layer):

- `.eml` emails (headers, body, attachment metadata)
- `.pdf` documents (page-level text)
- `.docx` documents (stdlib ZIP/XML extraction)
- `.md` / `.txt` / `.log` notes and chat logs
- image metadata (EXIF when Pillow is installed; otherwise file facts)

Every evidence item gets a SHA-256 hash, source type, acquisition time, custodian and
duplicate detection. Report artefacts are served read-only; evidence originals are never
exposed by the web server.

---

## Natural-language querying

The parser is deterministic and explainable — no network call is required:

- **Intents:** search, entity profile, timeline, relationship, call/transfer correlation,
  aggregate, anomaly, summarize, recommend, diary, report, evidence inventory, entity list, help.
- **Filters:** entity/alias/identifier, source, event type, time range, amount range,
  direction, location, keyword, result limit.
- **Languages:** English, Hindi, Spanish, French, German, Portuguese. Language detection
  handles script, stopwords and disambiguating vocabulary; explicit requests ("in Hindi",
  "en español") override it.
- **Follow-ups:** session memory resolves pronouns and short references ("who did *he* call?",
  "connection between Viktor and Elena").
- **Grounding:** every factual answer carries citations (evidence id, file, timestamp, snippet);
  no evidence match produces an explicit no-result answer instead of a guess.

### Optional LLM polish

The assistant is fully offline by default. An optional OpenAI-compatible or Ollama endpoint
can rewrite the deterministic draft into more natural prose in the requested language, but
only under strict grounding rules: it must preserve at least one valid citation label and may
not invent evidence IDs. If the LLM is unavailable, fails, or violates the rules, the
deterministic answer is returned unchanged.

Configure via environment variables:

```bash
# OpenAI-compatible
set INVESTIGATION_LLM_PROVIDER=openai
set INVESTIGATION_LLM_MODEL=gpt-4o-mini
set INVESTIGATION_LLM_API_KEY=...
set INVESTIGATION_LLM_BASE_URL=https://api.openai.com/v1

# Local Ollama
set INVESTIGATION_LLM_PROVIDER=ollama
set INVESTIGATION_LLM_MODEL=llama3.1
set INVESTIGATION_LLM_BASE_URL=http://localhost:11434
```

Or set `llm_provider` in `config/investigation.yaml`.

**Multilingual note:** evidence quotations, analytics-derived finding titles and historical
diary entries are preserved in their original language; report labels, recommendations,
summaries and the interface are localized. This is deliberate — a forensic record should not
silently paraphrase or translate the underlying evidence.

---

## Measured performance

### Investigation assistant (demo case, 16 queries, synthetic data)

Reproducible with `python eval_investigation.py --case <case_id>`:

| Metric | Value |
|---|---|
| Intent accuracy | **1.000** |
| Entity-resolution hit rate | **1.000** |
| Expected-keyword containment | **1.000** |
| Citation coverage (citation-required intents) | **1.000** |
| Evidence recall@5 | **1.000** |
| MRR | **1.000** |
| Recommendation category coverage | **1.000** |
| Average query latency | **~370 ms** |
| Languages evaluated | en, hi, es, fr, de, pt |

These are demo-case numbers on a small synthetic case; they demonstrate the retrieval and
grounding pipeline, not real-world accuracy. The evaluation harness is the reproducible
baseline and will be extended as more cases are added.

### Analytics core (unchanged)

Reproducible with `python eval.py` on the original 21 main + 12 surprise entities:

| Metric | Value |
|---|---|
| recall@10 / precision@10 | 0.429 / 0.900 |
| Best single model recall@10 | network 0.333 |
| Fusion gain over best single | +0.096 |
| Surprise hold-out recall@10 | 0.071 |

Plus: **105 non-UI tests pass** (84 original + 21 investigation tests), 14/14 dirty-data
scenarios pass, 576/576 schema stress cases pass. The PySide6 desktop tests require PySide6,
which is not installed in the current environment.

---

## Project structure

```text
config/config.yaml              analytics-core settings
config/investigation.yaml       case workspace, thresholds, optional LLM
src/aml/                        verified analytics core (unchanged)
src/investigation/              new investigation layer
  models.py                     case / evidence / entity / finding / recommendation / diary / answer
  i18n.py                       six locales, language detection, query lexicons
  case_store.py                 case persistence, SHA-256 evidence inventory
  diary.py                      hash-chained case diary
  evidence.py                   structured + EML/PDF/DOCX/text ingestion, chunking, mentions
  query.py                      multilingual query parser
  index.py                      BM25 + event index + entity graph
  retrieval.py                  hybrid retrieval + citations + call/transfer correlation
  summarize.py                  case summary / entity dossier / timeline / gaps
  recommend.py                  evidence-gap + finding-driven recommendations
  assistant.py                  intent routing, grounded answers, session memory
  llm.py                        optional grounded LLM adapter
  case_report.py                draft case report (MD/HTML/JSON/PDF)
  service.py                    orchestrator
  demo_case.py                  demo evidence + ground truth generator
web/                            zero-dependency chat workspace
  server.py                     JSON API + static files
  static/index.html             workspace UI
  static/app.js                 chat, citations, tabs, i18n
  static/style.css              dark forensic console theme
run_case.py                     case CLI
ask.py                          interactive assistant wrapper
run_web.py                      launch the web workspace
eval_investigation.py           investigation evaluation harness
tests/test_investigation_*.py   new tests
data/cases/                     generated case workspaces
data/demo_case/                 generated demo evidence + ground truth
INVESTIGATION_PLAN.md           migration plan, gap analysis, roadmap
PLAN.md                         original analytics-core build record
SCHEMA.md                       frozen analytics dataframe spec
GAP_AUDIT.md                    original analytics gap audit
```

---

## Design principles

1. **Ground everything.** No factual answer or finding without a citation to an evidence item.
2. **Say what the evidence cannot prove.** The analytics sufficiency verdicts are surfaced in
   the case report and limitations; the assistant never invents missing coverage.
3. **Keep the analytics core stable.** The new layer imports `src/aml`; it does not fork or
   rewrite the verified models/evaluation.
4. **Local-first, offline by default.** No network calls unless the operator opts into an LLM.
5. **Auditable by construction.** Hash-chained diary, SHA-256 evidence inventory, deterministic
   query plans and reproducible evaluation.

---

## Documentation

- `INVESTIGATION_PLAN.md` — migration mapping, gap analysis, target architecture, phased roadmap,
  evaluation plan and demo script.
- `PLAN.md` — the original analytics-core build record and measured targets.
- `SCHEMA.md` — frozen analytics dataframe specification.
- `GAP_AUDIT.md` — original analytics coverage audit.
- `config/investigation.yaml` — investigation thresholds and optional LLM settings.

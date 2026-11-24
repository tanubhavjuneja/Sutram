# Sutram Investigator — AI-Powered Digital Investigation Assistant

Sutram Investigator turns the project's verified multi-source anomaly engine into a
**case-centric investigation workspace**: ingest mixed evidence, ask questions in natural
language, retrieve cited evidence, summarize the case, surface suspicious patterns,
generate investigative leads, keep a tamper-evident case diary, and draft a case report.

The investigation capabilities now live **inside the `aml` package** (`src/aml`) and are
exposed through the same pipeline entry point (`CasePipeline` in `src/aml/pipeline.py`).
There is no second `investigation` package and no parallel analytics implementation. The
existing analytics models, schema detection, entity resolution, sufficiency engine and
evaluation harness remain the detection core.

> **Why this mapping works:** the hard part of a digital investigation assistant is not the
> chat bubble — it is trustworthy evidence handling. This project already had runtime schema
> detection, cross-source entity resolution, evidence-sufficiency verdicts, seven model
> families, fusion, explainability and reproducible evaluation. The case, retrieval,
> conversation, recommendation, diary, report and Cloudflare AI capabilities are layered
> into that same package.

---

## What it does

| Problem-statement deliverable | Implementation |
|---|---|
| **Conversational forensic assistant** | `src/aml/assistant.py` + web chat + CLI REPL; intent routing, session memory, follow-ups, grounded answers, agentic tools |
| **Evidence retrieval engine** | `src/aml/index.py` + `retrieval.py`: BM25 over events and text chunks, structured filters, entity-graph expansion, recency/amount/risk boosts, citations |
| **Automated case summarization** | `src/aml/summarize.py`: case overview, entity dossier, timeline summary, evidence-gap summary |
| **Suspicious patterns / anomalies** | Existing `src/aml` 7 model families + fusion, converted into case findings with severity, confidence, evidence links and citations |
| **Investigative lead recommendation** | `src/aml/recommend.py`: evidence-gap rules + finding-driven next actions with priority, rationale and linked evidence |
| **Draft case report generation** | `src/aml/case_report.py`: Markdown / HTML / JSON / PDF case report with evidence inventory, entities, findings, timeline, recommendations, diary, limitations and analytics appendix |
| **Automated case diary** | `src/aml/diary.py`: append-only, hash-chained log of ingest/analysis/query/finding/recommendation/export actions; tamper verification |
| **Multilingual interface** | `src/aml/i18n.py`: English, Hindi, Spanish, French, German, Portuguese; language detection, multilingual query lexicons, localized answers, recommendations and report labels |
| **Case / evidence / chain of custody** | `src/aml/case_store.py`: case directory, SHA-256 evidence inventory, duplicate detection, audit log |
| **Broad evidence ingestion** | `src/aml/evidence.py`: bank/CDR/social CSV/JSON/JSONL/TXT + `.eml`, `.pdf`, `.docx`, `.md`, chat logs, image metadata |
| **Agentic investigation tools** | `src/aml/service.py`: rerun analysis, recheck numbers, deep-dive a model, plot a chart, generate an image, read an uploaded image |
| **Cloudflare Workers AI** | `src/aml/llm.py`: text model, vision model, FLUX text-to-image model; grounded polish, image reading, image generation |
| **Improved graphs** | `src/aml/visuals.py`: OSM-backed activity map with zoom inset + interactive Leaflet map, connected suspect-correlation graph with labelled edges, line-graph timelines |

---

## Quickstart

Requires Python 3.11+ (the demo was validated on Python 3.15 with the dependencies below).

```bash
pip install -r requirements.txt

# Optional: connect Cloudflare Workers AI (text + vision + image generation)
copy .env.example .env
# fill in:
#   Account ID=...
#   API KEY=...

# 1) Generate the synthetic demo case, run analytics, recommendations and report
python run_case.py demo

# 2) Open the conversational web workspace
python run_web.py --open
#    or: python web/server.py --case-root data/cases --port 8000

# 3) Ask from the terminal
python run_case.py list
python run_case.py ask <case_id> "What looks suspicious in this case?"
python run_case.py ask <case_id> "What is in this image?" --image evidence/photo.png
python run_case.py chat <case_id> --lang hi
python run_case.py summary <case_id>
python run_case.py recommend <case_id>
python run_case.py diary <case_id>
python run_case.py report <case_id> --lang en

# 4) The same workspace through the existing pipeline entry point
python run_pipeline.py --investigation --title "Operation Nightfall" evidence/
python run_pipeline.py --investigation --case-id <case_id> --ask "Recheck the numbers."

# 5) Evaluate the investigation assistant
python eval_investigation.py --case <case_id>

# 6) The original analytics core still works
python run_demo.py
python run_demo.py --investigation
python eval.py
pytest tests --ignore=tests/test_main_app.py
# install PySide6 to run the desktop UI tests as well
```

The demo generates `data/demo_case/` (structured + emails + chats + note + PDF + ground
truth) and `data/cases/<case_id>/` (case record, evidence copies, index, reports, audit log).

**Secrets:** `.env` is git-ignored. `.env.example` documents the expected keys. The
dashboard-style keys `Account ID` and `API KEY` are normalized automatically, as are
`CLOUDFLARE_ACCOUNT_ID` / `CLOUDFLARE_API_TOKEN`.

---

## Agentic chat tools

The assistant is not limited to reading the existing report. It can execute investigation
actions and return Markdown plus generated images/artefacts inline:

| Ask | What happens |
|---|---|
| “Rerun the analysis.” | Re-runs the analytics pipeline, refreshes findings/leads, returns a model-verdict table, a new correlation graph and analytics report downloads |
| “Recheck the numbers.” | Recomputes counts, totals, averages, min/max, source/direction breakdowns from the current index and returns a Markdown table with citations |
| “Plot the timeline.” / “Show me the map.” / “Plot the correlation network.” | Renders the requested chart and displays it inline in the chat |
| “Plot Yara Okafor’s activity.” | Renders the per-entity activity timeline |
| “Analyze further for structuring.” / “Deep dive into the network.” | Runs the selected model on the current evidence and returns a ranked table with entity dossiers/citations |
| “Generate an image of a clean forensic case board…” | Calls FLUX.1 [schnell] and displays the generated illustration in the chat |
| Attach an image + ask “What does this chart show?” | Sends the image to the Cloudflare vision model and returns a clearly labelled model observation |

Tool answers are recorded in the case diary. Generated images and reports are served from
the case `reports/` directory; the web chat renders images and download links inline. The
CLI prints the generated image path and file artefacts.

---

## Cloudflare Workers AI

Cloudflare is the default provider when credentials are present:

```dotenv
Account ID=your_cloudflare_account_id
API KEY=your_workers_ai_api_token
```

Models used:

| Role | Model |
|---|---|
| Text / grounded polish | `@cf/meta/llama-3.1-8b-instruct` |
| Vision / image reading | `@cf/meta/llama-3.2-11b-vision-instruct` |
| Text-to-image | `@cf/black-forest-labs/flux-1-schnell` |

Capabilities:

- **Text:** rewrites grounded drafts while preserving citation labels; if the model fails,
  times out, or invents a citation, the deterministic answer is returned unchanged.
- **Vision:** reads uploaded PNG/JPG evidence or charts; the answer is explicitly labelled
  as a model observation, not a forensic conclusion.
- **Image generation:** produces synthetic illustrations from a text prompt; clearly labelled
  as synthetic, not evidence.

The assistant still works fully offline without credentials. `.env` is never committed.

---

## Improved graphs

- **Activity map:** OpenStreetMap tile background when online, deterministic synthetic
  geo-projection otherwise; collision-aware labels, a zoom inset for the densest cluster,
  and an interactive Leaflet map with scroll/pinch zoom in `report.html`.
- **Suspect correlation graph:** one connected graph for the flagged suspects. Edges are
  direct transfers/calls/posts, call→transfer colocation, shared locations and shared
  counterparties. Edge types are colour-coded and labelled; node labels are larger and
  collision-checked; disconnected groups are called out explicitly.
- **Timeline charts:** daily traffic by modality is a line graph (not a stacked area); the
  per-entity activity timeline is also a line graph with integer day counts.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Web chat workspace  ·  CLI (`run_case.py`, `ask.py`)  ·  Desktop console    │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ ask / search / summarize / recommend /
                                │ diary / report / plot / rerun / vision
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                 CASE WORKSPACE (`src/aml`, same package)                     │
│                                                                              │
│  QueryParser ──► HybridRetriever ──► AnswerSynthesizer ──► citations         │
│       │               │                    │                                 │
│       │               ├── EventIndex       ├── CaseSummarizer                │
│       │               ├── TextChunkIndex   ├── EntityDossier                 │
│       │               └── EntityGraph      └── RecommendationEngine          │
│       │                                                                      │
│  CaseStore ──► EvidenceInventory ──► CaseDiary (hash chain) ──► CaseReport   │
│       │                                                                      │
│  Agent tools: rerun · recheck · plot · deep dive · generate image · vision   │
│  Cloudflare Workers AI: text · vision · FLUX image generation                │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                   ANALYTICS CORE (`src/aml`, unchanged)                      │
│  runtime schema detection · entity resolution · sufficiency · 7 models ·     │
│  fusion · insights · HTML/PDF/MD/JSON anomaly brief · synthetic data · eval  │
└──────────────────────────────────────────────────────────────────────────────┘
```

`CasePipeline` in `src/aml/pipeline.py` is the unified public entry point. It extends the
existing `Pipeline` and exposes `create_case`, `ingest_case`, `analyze_case`, `ask`,
`search`, `summarize`, `recommend`, `diary`, `report` and `run_case_result`.

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
    analytics/              # anomaly brief (HTML/PDF/MD/JSON/charts)
    chat/                   # generated charts, uploads, generated illustrations
  audit.log                 # human-readable mirror of the hash-chained diary
```

---

## Evidence types

**Structured** (parsed by the existing adaptive schema engine):

- bank / transaction exports
- CDR / call detail records
- social-media posts

**Unstructured** (parsed by the case workspace):

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
  aggregate, anomaly, summarize, recommend, diary, report, evidence inventory, entity list,
  help, plus agent tools: rerun analysis, recheck, plot, deep dive, generate image.
- **Filters:** entity/alias/identifier, source, event type, time range, amount range,
  direction, location, keyword, result limit.
- **Languages:** English, Hindi, Spanish, French, German, Portuguese. Language detection
  handles script, stopwords and disambiguating vocabulary; explicit requests (“in Hindi”,
  “en español”) override it.
- **Follow-ups:** session memory resolves pronouns and short references.
- **Grounding:** every factual answer carries citations (evidence id, file, timestamp,
  snippet); no evidence match produces an explicit no-result answer instead of a guess.

Evidence quotations, analytics-derived finding titles and historical diary entries are
preserved in their original language; report labels, recommendations, summaries and the
interface are localized. A forensic record must not silently paraphrase or translate the
underlying evidence.

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
grounding pipeline, not real-world accuracy.

### Analytics core (unchanged)

Reproducible with `python eval.py` on the original 21 main + 12 surprise entities:

| Metric | Value |
|---|---|
| recall@10 / precision@10 | 0.429 / 0.900 |
| Best single model recall@10 | network 0.333 |
| Fusion gain over best single | +0.096 |
| Surprise hold-out recall@10 | 0.071 |

Plus: **112 non-UI tests pass, 4 skipped** (84 original + 28 investigation tests; the four
skipped tests are live Cloudflare integration tests enabled with
`RUN_CLOUDFLARE_TESTS=1`). The PySide6 desktop tests require PySide6, which is not installed
in the current environment. The live Cloudflare suite passes all four tests when enabled.

---

## Project structure

```text
config/config.yaml              analytics-core settings
config/investigation.yaml       case workspace, thresholds, Cloudflare model IDs
.env.example                    Cloudflare credential template (never commit .env)
src/aml/                        ONE package: analytics + case workspace
  case.py                       case / evidence / entity / finding / recommendation / diary / answer
  case_store.py                 case persistence, SHA-256 evidence inventory
  diary.py                      hash-chained case diary
  evidence.py                   structured + EML/PDF/DOCX/text/chat ingestion, chunking, mentions
  query.py                      multilingual query parser + agent-tool intents
  index.py                      BM25 + event index + entity graph
  retrieval.py                  hybrid retrieval + citations + call/transfer correlation
  summarize.py                  case summary / entity dossier / timeline / gaps
  recommend.py                  evidence-gap + finding-driven recommendations
  assistant.py                  intent routing, grounded answers, session memory
  llm.py                        Cloudflare Workers AI + OpenAI/Ollama adapters, vision, FLUX
  case_report.py                draft case report (MD/HTML/JSON/PDF)
  service.py                    orchestrator + agent tools
  pipeline.py                   Pipeline + CasePipeline + extended PipelineResult
  visuals.py                    OSM map, correlation graph, line timelines
  i18n.py                       six locales, language detection, query lexicons
  data_generator/investigation_case.py  demo evidence + ground truth
web/                            zero-dependency chat workspace
run_pipeline.py                 analytics + `--investigation` case mode
run_case.py                     case CLI
ask.py                          interactive assistant wrapper
run_web.py                      launch the web workspace
eval_investigation.py           investigation evaluation harness
tests/test_investigation_*.py   investigation + agent-tool tests
tests/test_cloudflare_integration.py  live Cloudflare tests (skipped by default)
INVESTIGATION_PLAN.md           migration plan, gap analysis, roadmap
PLAN.md                         original analytics-core build record
SCHEMA.md                       frozen analytics dataframe spec
GAP_AUDIT.md                    original analytics gap audit
```

---

## Design principles

1. **Ground everything.** No factual answer or finding without a citation to an evidence item.
2. **Say what the evidence cannot prove.** Analytics sufficiency verdicts are surfaced in the
   case report and limitations; the assistant never invents missing coverage.
3. **One package, one pipeline.** Case features live in `src/aml`; `CasePipeline` extends the
   existing analytics pipeline.
4. **Local-first, offline fallback.** Deterministic answers work without a network; Cloudflare
   Workers AI is opt-in via `.env` and falls back cleanly.
5. **Auditable by construction.** Hash-chained diary, SHA-256 evidence inventory, deterministic
   query plans and reproducible evaluation.

---

## Documentation

- `INVESTIGATION_PLAN.md` — migration mapping, gap analysis, target architecture, phased roadmap,
  evaluation plan and demo script.
- `PLAN.md` — the original analytics-core build record and measured targets.
- `SCHEMA.md` — frozen analytics dataframe specification.
- `GAP_AUDIT.md` — original analytics coverage audit.
- `config/investigation.yaml` — investigation thresholds and Cloudflare model settings.
- `.env.example` — Cloudflare Workers AI credential template.

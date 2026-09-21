# Sutram — Multi-Source Financial Anomaly Detection

Sutram (formerly TraceWeave) detects anomalous patterns across **Call Detail Records (CDR)**, **bank transactions**, and **social media activity** using a dataframe-centric pipeline with a mixture of specialized models, an honest data-sufficiency engine, and a verified evaluation harness.

No fixed parser: schema is detected at runtime, entities are resolved across sources with no shared ID, only the models the data can support are run, and every result ships with an explanation.

## Features

- **Auto-adaptive ingestion** (`src/aml/schema_detect.py`): rapidfuzz header matching + 8-slot content-probe chain, unix-epoch + mixed datetime parsing, currency-mess parsing, CSV/JSON/JSONL/TXT, latin-1 fallback, delimiter sniffing
- **Entity resolution** (`src/aml/entity_resolve.py`): ID + name similarity, per-source fallback chain, cross-source linking
- **Sufficiency engine** (`src/aml/sufficiency.py`): per-model `SUPPORTED / DEGRADED / BLOCKED` verdicts with actionable reasons (e.g. "network skipped: no counterparty dimension; add a bank or CDR file")
- **Model zoo, 7 families on one dataframe** (`src/aml/models.py`): time-correlation (merge_asof + backward/anypair/velocity), network (burst, oddball, scatter/gather, fan-in/out, reciprocity), stat/ML (IF/EIF/LOF/OCSVM/GMM/HBOS/Mahalanobis/PCA/AE/zscore/KDE), Benford, structuring, behavioral (dormant/silence), chain/layering DFS
- **Fusion + explanations** (`src/aml/fusion_meta.py`): Borda / score-mean / rank-avg / RF-XGBoost meta-fusion, template narrations with fired models + evidence
- **GUI + reports** (`main_app.py`, `src/aml/report.py`): PySide6 drag-drop, run history, HTML/PDF/MD/JSON export
- **Synthetic data + write-only answer keys** (`src/aml/data_generator/`): agent-based normals, 6 typology storylines (chain/colocation/dormant_flip/fan_io/post_burst/silence/structuring) + surprise injector
- **One-command eval** (`eval.py`): recall/precision@k, per-type recall, leave-one-model-out ablation, surprise-leak exclusion

## Quickstart

Requires Python 3.11.

```bash
pip install -r requirements.txt

# 1. End-to-end demo: generate data -> run pipeline -> write report
python run_demo.py

# 2. Run on your own files (any CSV/JSON/JSONL/TXT)
python run_pipeline.py --config config/config.yaml data/sources/ file.csv

# 3. Desktop app
python main_app.py

# 4. Reproduce the numbers
python eval.py
pytest tests
```

Reports go to `data/report/` (git-ignored, regenerable). Config lives in `config/config.yaml`.

## Architecture

```text
sources/*.csv|json|jsonl|txt ──► [A] Schema detection ──► [B] Entity resolution
                                                                     │
                                                    [Unified long dataframe]
                        entity_id | event_type | timestamp | source | amount |
                        counterparty_id | location | direction | duration
                                                                     │
              ┌──────────────────────────────┬─────────────────────────┴────────────┐
              ▼                              ▼                                      ▼
 [C] Time correlation (merge_asof)  [D] Network correlation        [E] Stat/ML outliers
                                              │
                 H: Benford / structuring / chain rules            │
              └──────────────────────┬─────────────────────────────┘
                                     ▼
                     [F] Fusion & ranking (Borda / score-mean / meta)
                                     ▼
                     [G] Explanations ──► GUI / HTML / PDF / JSON
```

Rule-based + statistical + graph + ML models each capture a different complementary signal. No single model is trusted; output is an ensemble. See `SCHEMA.md` for the frozen dataframe spec.

## Measured performance

Reproducible with `python eval.py` (21 main + 12 surprise entities):

| Metric | Value |
|---|---|
| recall@10 / precision@10 | 0.429 / 0.900 |
| Best single model recall@10 | network 0.333 |
| Fusion gain over best single | +0.096 |
| Surprise hold-out recall@10 | 0.071 |

Plus: 94 tests pass, 14/14 dirty-data scenarios pass (`scripts/dirty_audit.py`), 576/576 schema stress cases pass.

## Project structure

```text
config/config.yaml      pipeline settings (thresholds, dirs, seeds)
src/aml/                pipeline: schema_detect, loaders, entity_resolve,
                        sufficiency, models, fusion_meta, report, visuals
src/aml/data_generator/ synthetic data + storylines + surprise injector
data/sources/           default input dir
data/datasets/          15 cross-source correlation variants (full/none/pair_*)
data/answer_key*.json   write-only ground truth
main_app.py             PySide6 GUI
run_pipeline.py         CLI analysis
run_demo.py             generate + run + report demo
eval.py                 evaluation harness
scripts/                dirty fixtures/audit, tuning, zoo/suite evals
tests/                  pytest suite
assets/charts/          architecture/coverage charts
PLAN.md                 full build plan + targets
SCHEMA.md               frozen dataframe spec
GAP_AUDIT.md            gap audit
```

## Docs

- `PLAN.md` — what is built, measured numbers, remaining work
- `SCHEMA.md` — unified dataframe columns, field slots, anomaly types
- `GAP_AUDIT.md` — coverage audit
- `config/config.yaml` — all tunable thresholds

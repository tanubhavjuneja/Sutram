# Sutram Investigator — Progress

**Updated:** 2026-11-24
**Branch:** `feat/console-ui-rebuild`
**Baseline commit:** `30085f1` (case workspace folded into `src/aml`)

## Current state

- Native PySide6 case workspace is integrated into `main_app.py` via
  `case_ui.py` (`CaseWorkspacePage` + `CaseWorker`), opened from the
  **Investigator** button on the Home screen.
- The case workspace covers case selection, evidence ingestion, analytics
  rerun, chat with image upload, findings, leads, diary, draft report, and
  language switching.
- Agent tools (rerun analysis, recheck numbers, plot, deep dive, generate
  image, vision) are wired through `src/aml/service.py`. Tool answers now
  carry `data.tool` and `data.elapsed_ms` for the UI badges.
- Qt and web chats have visible tool shortcut chips and richer Markdown
  rendering (headings, links, blockquotes, fenced code, horizontal rules,
  tables).
- Cloudflare Workers AI defaults: `llama-4-scout-17b-16e-instruct` (vision),
  `llama-3.3-70b-instruct-fp8-fast` (text), `flux-2-dev` (image generation).
- Chart QA completed with Cloudflare vision read-back for the demo case;
  light and dark charts render cleanly. Vision’s remaining notes are false
  positives from OpenStreetMap tile labels, not chart-label defects.
- `README.md` and `INVESTIGATION_PLAN.md` reflect the current models and
  test counts.

## Verification

```bash
# Full suite (PySide6 installed)
AML_SKIP_TORCH=1 python -m pytest -q tests
# -> 124 passed, 6 skipped

# Live Cloudflare integration (text, vision, image generation)
AML_SKIP_TORCH=1 RUN_CLOUDFLARE_TESTS=1 python -m pytest tests/test_cloudflare_integration.py -q
# -> 4 passed

# Regenerate every chart and run vision read-back
python scripts/chart_qa.py --case operation-nightfall-20261123T195835
# -> reports/chart_qa/chart_qa.md and .json
```

## Known caveats

- Cloudflare Workers AI rejects very large request bodies (roughly 15–16 MiB)
  even when the model’s 1M-token context is far from full. The `read` tool
  returns images as base64, and OpenCode re-sends history images on every
  turn, so image-heavy sessions can hit the byte cap.
- Mitigations: downscale images with `media.image` in `opencode.json`, read
  each screenshot once, delegate screenshot inspection to a subagent, and
  run `/compact` before the session grows too large.
- If a session is already stuck on the Cloudflare body-size error, switch it
  temporarily to `opencode/tidepool-alpha-free`, run a manual compaction,
  then switch back to the Cloudflare model.
- `RUN_CLOUDFLARE_TESTS=1` requires the credentials in `.env`; the suite is
  skipped by default so offline runs stay deterministic.

## Open items

- Commit the current working tree (UI integration, chart fixes, Cloudflare
  model updates, chart QA script, progress docs) and push
  `feat/console-ui-rebuild`.
- Optional: add a small OpenCode plugin that caps inline images per request
  for `cloudflare-workers-ai` (hard guard against the body-size failure).

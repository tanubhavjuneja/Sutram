"""Report generation (markdown + HTML + PDF + JSON) for a PipelineResult.

What a human reads from a report is *plain-language findings* — the top
findings are written as sentences about real people and real records, never
as model names or scores. Machine-readable detail (per-record scores, model
verdicts) lives in report.json only.

  * report.md   — plain markdown with embedded charts
  * report.html — self-contained styled HTML (charts inlined as base64)
  * report.pdf  — printable PDF built with reportlab
  * report.json — machine-readable payload
  * charts/     — the matplotlib PNGs referenced by all formats
"""
from __future__ import annotations

import base64
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from . import visuals
from . import insights as insights_mod

MINT = "#3fbf8f"
SKY = "#5aa7e8"
TEAL = "#35c7c2"
AMBER = "#e0a03d"
RED = "#e0566b"
NAVY = "#1e3b33"
MUTED = "#6b857d"
LIGHT = "#eef5f1"

RISK_COLORS = {
    "HIGH": RED,
    "MEDIUM": AMBER,
    "LOW": MINT,
}

# Judge-facing report branding (shared by markdown / HTML / PDF)
REPORT_TITLE = "Cross-Source Anomaly Intelligence Brief"
REPORT_SUBTITLE = ("Automatic screening of financial, telecom and social "
                   "activity — every flagged person is backed by concrete, "
                   "explainable evidence.")


def _generated_line(res) -> str:
    """Timestamp + scope line under the report title."""
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    n_files = len(res.per_source)
    n_src = len({rs.source for rs in res.per_source})
    return (f"Generated {ts} · {n_files} source file(s) across "
            f"{n_src} data type(s) · analysis depth auto-adapted to the data")


def _fmt(v: float) -> str:
    return f"{v:,.0f}"


def _unbold(s: str) -> str:
    """Strip **emphasis** markers (used by the insights engine)."""
    return s.replace("**", "")


def _htmlize(s: str) -> str:
    """Escape HTML + turn **emphasis** into <b> tags."""
    parts = re.split(r"\*\*(.+?)\*\*", s)
    out = []
    for i, part in enumerate(parts):
        cleaned = part.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        out.append(f"<b>{cleaned}</b>" if i % 2 else cleaned)
    return "".join(out)


def _pdfize(s: str) -> str:
    """Escape for reportlab Paragraph, with **emphasis** -> <b>."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>",
                  s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ---------------------------------------------------------------------------
# shared content
# ---------------------------------------------------------------------------
def _insights_for(res, cfg) -> List[dict]:
    return insights_mod.build_insights(res, top_k=cfg.top_k)


def _overview_lines(res, insight_list) -> List[str]:
    """Plain opening sentences: what was reviewed, what stood out."""
    unified = res.unified
    n_events = int(len(unified))
    n_people = int(unified["entity_id"].nunique()) if not unified.empty else 0
    n_files = len(res.per_source)
    lines = [
        f"We looked through **{_fmt(n_events)} records** across "
        f"**{n_files} file(s)** and mapped them to **{_fmt(n_people)} people**.",
    ]
    if insight_list:
        lines.append(f"**{len(insight_list)} people** stood out enough to be worth "
                     f"a closer look — they are listed below, most important first.")
    else:
        lines.append("**Nobody stood out** from the crowd in the records we reviewed.")
    return lines


def _has_timeline(res) -> bool:
    return (res.unified is not None and not res.unified.empty
            and "timestamp" in res.unified.columns
            and res.unified["timestamp"].notna().any())


# ---------------------------------------------------------------------------
# markdown (self-contained: charts inlined as base64 PNG data URIs so the
# .md file renders everywhere — email, GitHub, other machines — without the
# charts/ folder travelling with it)
# ---------------------------------------------------------------------------
def _md_chart(chart: Path) -> str:
    if not Path(chart).exists():
        return ""
    b64 = base64.b64encode(Path(chart).read_bytes()).decode("ascii")
    return f"![{Path(chart).stem}](data:image/png;base64,{b64})"


def _markdown(res, cfg, charts: Dict[str, Path]) -> str:
    insight = _insights_for(res, cfg)
    L: List[str] = []
    L.append(f"# {REPORT_TITLE}")
    L.append(f"*{_generated_line(res)}*")
    L.append("")
    L.append(REPORT_SUBTITLE)
    L.append("")
    for line in _overview_lines(res, insight):
        L.append(line)
    L.append("")
    L.append(_md_chart(charts["people"]))
    L.append("")

    L.append("## The people who stand out")
    L.append("")
    if insight:
        for it in insight:
            L.append(f"### {it['rank']}. {it['name']} — {it['risk']}")
            L.append("")
            for j, bullet in enumerate(it["evidence"]):
                prefix = "-" if j == 0 else "  -"
                L.append(f"{prefix} {_unbold(bullet)}")
            L.append("")
    else:
        L.append("None.")
        L.append("")

    L.append("## Where the records came from")
    L.append("")
    L.append("| file | source | records | people |")
    L.append("|---|---|---|---|")
    if res.per_source:
        for rs in res.per_source:
            L.append(f"| {Path(rs.file).name} | {rs.source} | {rs.n_rows:,} | "
                     f"{rs.n_entities:,} |")
    else:
        L.append("| — | — | 0 | 0 |")
    L.append("")
    if res.per_source:
        L.append(_md_chart(charts["sources"]))
        L.append("")

    L.append("## When it happened")
    L.append("")
    if _has_timeline(res):
        L.append(_md_chart(charts["timeline"]))
    else:
        L.append("No timestamped records were found in the files.")
    L.append("")

    if charts.get("network") and charts.get("people") is not None:
        L.append("## Who the flagged people are connected to")
        L.append("")
        L.append(_md_chart(charts["network"]))
        L.append("")

    if charts.get("map"):
        L.append("## The activity map")
        L.append("")
        L.append(_md_chart(charts["map"]))
        L.append("")

    if charts.get("drill"):
        L.append("## A closer look at the people who stand out")
        L.append("")
        for title, img in charts["drill"].items():
            L.append(f"### {title}")
            L.append("")
            L.append(_md_chart(img))
            L.append("")

    L.append("_The full technical detail behind this analysis is kept in "
             "`report.json` — this report is the human summary._")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# HTML — Sutram Case Console (terminal-ops shell, self-contained)
# ---------------------------------------------------------------------------
def _b64(path: Path | str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def _risk_cell(risk: str) -> str:
    r = (risk or "LOW").upper()
    cls = {"HIGH": "risk-high", "MEDIUM": "risk-med"}.get(r, "risk-low")
    return f'<span class="risk {cls}">{r}</span>'


def _html_register(res, insight_list, drill_dark, drill_light) -> str:
    """Suspect register: dense sortable table rows + expandable dossiers."""
    if not insight_list:
        return '<div class="empty">NO ENTITIES MET ALERT CRITERIA</div>'
    ranks = {r.get("entity_id"): r for r in (res.rankings or [])}
    expl = res.explanations or {}
    rows = []
    for it in insight_list:
        eid = it.get("entity_id", "")
        meta = ranks.get(eid, {})
        score = float(meta.get("score", 0.0))
        fired = meta.get("models_fired", [])
        n_ev = meta.get("n_events", 0)
        engines = " ".join(f'<span class="tag">{m.replace("_", "-")}</span>'
                           for m in fired) or '<span class="tag dim">ensemble</span>'
        bullets = "".join(f"<li>{_htmlize(b)}</li>" for b in it.get("evidence", []))
        rationale = expl.get(eid, "")
        dd_dark = ""
        dd_light = ""
        for title, pth in (drill_dark or {}).items():
            if it.get("name") in title or eid in title:
                dd_dark = _b64(pth)
                break
        for title, pth in (drill_light or {}).items():
            if it.get("name") in title or eid in title:
                dd_light = _b64(pth)
                break
        drill = ""
        if dd_dark or dd_light:
            drill = (
                '<div class="drill-label">ACTIVITY STRIP</div>'
                + (f'<img class="chart-dark" src="{dd_dark}" alt="activity strip for {it.get("name")}" loading="lazy">'
                   if dd_dark else "")
                + (f'<img class="chart-light" src="{dd_light}" alt="activity strip for {it.get("name")}" loading="lazy">'
                   if dd_light else "")
            )
        cls = "row-high" if it.get("risk") == "HIGH" else ("row-med" if it.get("risk") == "MEDIUM" else "")
        risk = _risk_cell(it.get("risk", ""))
        rows.append(
            f'<tr class="reg-row {cls}" data-eid="{eid}" data-score="{score:.3f}" '
            f'data-events="{n_ev}" data-risk="{(it.get("risk") or "LOW").upper()}">'
            f'<td class="mono dim">{it["rank"]:02d}</td>'
            f'<td class="mono amber">{eid}</td>'
            f'<td class="name">{_htmlize(it["name"])}</td>'
            f'<td>{risk}</td>'
            f'<td class="mono amber score">{score:.3f}</td>'
            f'<td class="engines">{engines}</td>'
            f'<td class="mono num">{n_ev}</td>'
            f'</tr>'
            f'<tr class="dossier" hidden><td colspan="7">'
            f'<ul class="evidence">{bullets}</ul>'
            + (f'<div class="rationale"><span class="dim mono">ENSEMBLE //</span> {_htmlize(rationale)}</div>'
               if rationale else "")
            + drill +
            f'</td></tr>'
        )
    return "\n".join(rows)


def _html_sufficiency_table(res) -> str:
    suff = getattr(res, "sufficiency", {}) or {}
    if not suff:
        return ""
    rows = []
    for name, v in suff.items():
        status = getattr(v, "status", "?")
        cls = {"SUPPORTED": "st-ok", "DEGRADED": "st-warn"}.get(status, "st-off")
        rows.append(
            f'<tr><td class="mono">{name.upper().replace("_", "-")}</td>'
            f'<td><span class="status {cls}">{status}</span></td>'
            f'<td class="dim small">{_htmlize(getattr(v, "reason", ""))}</td></tr>'
        )
    return ("<table class='grid'><thead><tr><th>ENGINE</th><th>STATUS</th>"
            "<th>RATIONALE</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def _html_sources_table(res) -> str:
    if not res.per_source:
        return '<div class="empty">NO SOURCE FILES LOADED</div>'
    rows = []
    for rs in res.per_source:
        slots = "".join(f'<span class="tag dim">{s}</span>'
                        for s in sorted(rs.resolved_fields))
        warn = f'<span class="warn-line">{_htmlize(rs.warnings[0])}</span>' if rs.warnings else ""
        rows.append(
            f'<tr><td class="mono">{Path(rs.file).name}</td>'
            f'<td><span class="src src-{rs.source}">{(rs.source or "?").upper()}</span></td>'
            f'<td class="mono num">{rs.n_rows:,}</td>'
            f'<td class="mono num">{rs.n_entities:,}</td>'
            f'<td><div class="slots">{slots}</div>{warn}</td></tr>'
        )
    return ("<table class='grid'><thead><tr><th>FILE</th><th>SRC</th>"
            "<th class='num'>ROWS</th><th class='num'>ENTS</th>"
            "<th>SLOTS</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def _visual_box(title: str, sub: str, dark: str, light: str) -> str:
    if not (dark or light):
        return ""
    imgs = ""
    if dark:
        imgs += f'<img class="chart-dark" src="{dark}" alt="{title}" loading="lazy">'
    if light:
        imgs += f'<img class="chart-light" src="{light}" alt="{title}" loading="lazy">'
    return (f'<section class="panel"><div class="panel-head"><h3>{title}</h3>'
            f'<span class="sub">{sub}</span></div>{imgs}</section>')


def _kpi_strip(res, insight_list) -> str:
    u = res.unified
    n_ev = len(u) if u is not None and not u.empty else 0
    n_pe = int(u["entity_id"].nunique()) if u is not None and not u.empty else 0
    n_hi = sum(1 for it in insight_list if it.get("risk") == "HIGH")
    n_md = sum(1 for it in insight_list if it.get("risk") == "MEDIUM")
    return (
        f'<div class="kpis">'
        f'<div class="kpi"><span class="k">{n_ev:,}</span><span class="l">RECORDS</span></div>'
        f'<div class="kpi"><span class="k">{n_pe:,}</span><span class="l">ENTITIES</span></div>'
        f'<div class="kpi"><span class="k">{len(res.per_source)}</span><span class="l">FILES</span></div>'
        f'<div class="kpi"><span class="k hi">{n_hi}</span><span class="l">HIGH RISK</span></div>'
        f'<div class="kpi"><span class="k md">{n_md}</span><span class="l">MEDIUM</span></div>'
        f'</div>'
    )


def _sidebar_suspects(insight_list) -> str:
    if not insight_list:
        return ""
    items = "".join(
        f'<a class="side-eid" href="#reg-{it["entity_id"]}" data-eid="{it["entity_id"]}">'
        f'<span class="mono amber">{it["entity_id"]}</span>'
        f'<span class="mono dim">{it.get("score", 0):.2f}</span></a>'
        for it in insight_list[:10]
    )
    return f'<div class="side-group"><div class="side-label">SUSPECTS</div>{items}</div>'


def _html(res, cfg, charts: Dict[str, Path]) -> str:
    insight_list = _insights_for(res, cfg)
    overview = _overview_lines(res, insight_list)
    overview_html = " ".join(_htmlize(line) for line in overview)

    c_dark = charts.get("dark") or charts
    c_light = charts.get("light") or charts
    drill_dark = c_dark.get("drill") or {}
    drill_light = c_light.get("drill") or {}

    tl_d = _b64(c_dark.get("timeline")) if _has_timeline(res) else ""
    tl_l = _b64(c_light.get("timeline")) if _has_timeline(res) else ""
    nw_d, nw_l = _b64(c_dark.get("network")), _b64(c_light.get("network"))
    mp_d, mp_l = _b64(c_dark.get("map")), _b64(c_light.get("map"))
    so_d, so_l = _b64(c_dark.get("sources")), _b64(c_light.get("sources"))
    pe_d, pe_l = _b64(c_dark.get("people")), _b64(c_light.get("people"))

    register = _html_register(res, insight_list, drill_dark, drill_light)
    suff = _html_sufficiency_table(res)
    sources = _html_sources_table(res)
    kpis = _kpi_strip(res, insight_list)
    side_sus = _sidebar_suspects(insight_list)
    n_high = sum(1 for it in insight_list if it.get("risk") == "HIGH")

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{REPORT_TITLE} — Sutram Console</title>
<style>
  :root, [data-theme="dark"] {{
    --bg: #020617; --panel: #0F172A; --raised: #1E293B;
    --line: #334155; --line2: #1E293B;
    --tx: #F8FAFC; --tx2: #94A3B8; --tx3: #64748B;
    --amber: #FFB000; --red: #F87171; --green: #34D399;
    --sky: #38BDF8; --yellow: #FDE047;
    --hl: rgba(255, 176, 0, 0.09);
  }}
  [data-theme="light"] {{
    --bg: #F5F5F0; --panel: #FFFFFF; --raised: #FAFAF7;
    --line: #D4D4D8; --line2: #E4E4E7;
    --tx: #18181B; --tx2: #3F3F46; --tx3: #71717A;
    --amber: #B45309; --red: #DC2626; --green: #047857;
    --sky: #0369A1; --yellow: #A16207;
    --hl: rgba(180, 83, 9, 0.08);
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: var(--bg); color: var(--tx); }}
  body {{
    font-family: ui-monospace, "IBM Plex Mono", SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12.5px; line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }}
  .sans {{ font-family: -apple-system, "Segoe UI", "IBM Plex Sans", Roboto, sans-serif; }}

  /* ---- shell ---- */
  .shell {{ display: flex; min-height: 100vh; }}
  .sidebar {{
    width: 208px; flex: 0 0 208px; position: fixed; top: 44px; bottom: 0; left: 0;
    background: var(--panel); border-right: 1px solid var(--line);
    padding: 14px 0; overflow-y: auto; z-index: 40;
  }}
  .side-label {{
    font-size: 9.5px; letter-spacing: 1.4px; color: var(--tx3);
    padding: 10px 16px 4px; font-weight: 700;
  }}
  .side-item {{
    display: flex; gap: 8px; align-items: center;
    padding: 5px 16px; color: var(--tx2); text-decoration: none;
    font-size: 11px; letter-spacing: 0.4px;
    border-left: 2px solid transparent;
  }}
  .side-item:hover {{ color: var(--tx); background: var(--raised); }}
  .side-item.active {{ color: var(--amber); border-left-color: var(--amber); background: var(--hl); }}
  .side-item .no {{ color: var(--tx3); font-size: 10px; }}
  .side-item.active .no {{ color: var(--amber); }}
  .side-eid {{
    display: flex; justify-content: space-between; padding: 4px 16px;
    color: var(--tx2); text-decoration: none; font-size: 11px;
    border-left: 2px solid transparent;
  }}
  .side-eid:hover {{ background: var(--raised); border-left-color: var(--line); }}
  .side-eid.active {{ background: var(--hl); border-left-color: var(--amber); }}
  .main {{
    margin-left: 208px; margin-top: 44px; flex: 1; min-width: 0;
    padding: 18px 22px 60px;
  }}

  /* ---- command bar ---- */
  .cmdbar {{
    position: fixed; top: 0; left: 0; right: 0; height: 44px; z-index: 50;
    background: var(--panel); border-bottom: 1px solid var(--line);
    display: flex; align-items: center; gap: 18px; padding: 0 16px;
  }}
  .wordmark {{ color: var(--amber); font-weight: 700; letter-spacing: 1px; font-size: 12.5px; white-space: nowrap; }}
  .wordmark .sep {{ color: var(--tx3); }}
  .cmd-stats {{ color: var(--tx2); font-size: 11px; letter-spacing: 0.3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .cmd-stats b {{ color: var(--tx); font-weight: 700; }}
  .cmd-stats .hi {{ color: var(--red); font-weight: 700; }}
  .cmd-right {{ margin-left: auto; display: flex; align-items: center; gap: 10px; }}

  .theme-sw {{
    display: inline-flex; border: 1px solid var(--line); font-size: 10.5px;
    letter-spacing: 0.8px; cursor: pointer; user-select: none;
  }}
  .theme-sw span {{ padding: 4px 10px; color: var(--tx3); }}
  .theme-sw span.on {{ background: var(--amber); color: #020617; font-weight: 700; }}
  [data-theme="dark"] .theme-sw .t-dark {{ background: var(--amber); color: #020617; font-weight: 700; }}
  [data-theme="light"] .theme-sw .t-dark {{ color: var(--tx3); }}
  [data-theme="light"] .theme-sw .t-light {{ background: var(--amber); color: #18181B; font-weight: 700; }}
  [data-theme="dark"] .theme-sw .t-light {{ color: var(--tx3); }}

  .chips {{ display: inline-flex; gap: 6px; }}
  .chip {{
    border: 1px solid var(--line); background: transparent; color: var(--tx2);
    font-family: inherit; font-size: 10.5px; letter-spacing: 0.8px;
    padding: 4px 10px; cursor: pointer;
  }}
  .chip:hover {{ color: var(--tx); border-color: var(--tx3); }}
  .chip.on {{ background: var(--amber); border-color: var(--amber); color: #020617; font-weight: 700; }}
  .chip[data-f="HIGH"]:not(.on) {{ color: var(--red); border-color: color-mix(in srgb, var(--red) 45%, var(--line)); }}
  .chip[data-f="MEDIUM"]:not(.on) {{ color: var(--yellow); border-color: color-mix(in srgb, var(--yellow) 45%, var(--line)); }}

  .sortsel {{
    background: transparent; border: 1px solid var(--line); color: var(--tx2);
    font-family: inherit; font-size: 10.5px; letter-spacing: 0.8px;
    padding: 4px 8px; cursor: pointer;
  }}
  .sortsel option {{ background: var(--panel); color: var(--tx); }}

  /* ---- sections & panels ---- */
  section.block {{ margin-bottom: 26px; scroll-margin-top: 56px; }}
  .block-head {{
    display: flex; align-items: baseline; gap: 10px;
    border-bottom: 1px solid var(--line); padding-bottom: 7px; margin-bottom: 12px;
  }}
  .block-no {{ color: var(--amber); font-size: 11px; font-weight: 700; }}
  .block-head h2 {{
    font-size: 12.5px; font-weight: 700; letter-spacing: 1.2px;
    color: var(--tx); text-transform: uppercase; font-family: inherit;
  }}
  .block-head .sub {{ margin-left: auto; color: var(--tx3); font-size: 10.5px; letter-spacing: 0.4px; text-align: right; }}

  .panel {{
    border: 1px solid var(--line); background: var(--panel);
    padding: 14px 16px; margin-bottom: 14px;
  }}
  .panel-head {{ margin-bottom: 10px; }}
  .panel-head h3 {{
    font-size: 11px; letter-spacing: 1px; color: var(--amber);
    text-transform: uppercase; font-weight: 700;
  }}
  .panel-head .sub {{ color: var(--tx3); font-size: 10.5px; letter-spacing: 0.3px; }}
  .panel img, .dossier img {{ max-width: 100%; display: block; border: 1px solid var(--line2); }}

  /* ---- KPI strip ---- */
  .kpis {{ display: grid; grid-template-columns: repeat(5, 1fr); border: 1px solid var(--line); background: var(--panel); }}
  .kpi {{ padding: 12px 14px; border-right: 1px solid var(--line2); }}
  .kpi:last-child {{ border-right: none; }}
  .kpi .k {{ display: block; font-size: 21px; font-weight: 700; color: var(--amber); line-height: 1.1; }}
  .kpi .k.hi {{ color: var(--red); }}
  .kpi .k.md {{ color: var(--yellow); }}
  .kpi .l {{ display: block; font-size: 9.5px; letter-spacing: 1.2px; color: var(--tx3); margin-top: 3px; }}

  .overview {{ border: 1px solid var(--line); background: var(--panel); padding: 12px 16px; margin-bottom: 14px; }}
  .overview p {{ color: var(--tx2); }}
  .overview b {{ color: var(--tx); font-weight: 700; }}

  /* ---- register table ---- */
  .reg-toolbar {{
    display: flex; align-items: center; gap: 10px; margin-bottom: 8px;
    color: var(--tx3); font-size: 10px; letter-spacing: 0.8px;
  }}
  .reg-count {{ margin-left: auto; }}
  table.reg {{ width: 100%; border-collapse: collapse; border: 1px solid var(--line); }}
  table.reg thead th {{
    background: var(--raised); color: var(--tx3); font-size: 9.5px;
    letter-spacing: 1.1px; text-align: left; padding: 7px 10px;
    border-bottom: 1px solid var(--line); cursor: pointer; user-select: none;
    white-space: nowrap;
  }}
  table.reg thead th:hover {{ color: var(--amber); }}
  table.reg thead th .arr {{ color: var(--amber); }}
  table.reg tbody td {{
    padding: 7px 10px; border-bottom: 1px solid var(--line2);
    color: var(--tx2); font-size: 12px; vertical-align: middle;
  }}
  tr.reg-row {{ cursor: pointer; }}
  tr.reg-row:hover td {{ background: var(--raised); color: var(--tx); }}
  tr.reg-row.open td {{ background: var(--hl); border-bottom: none; color: var(--tx); }}
  tr.reg-row.row-high td:first-child {{ box-shadow: inset 3px 0 0 var(--red); }}
  tr.reg-row.row-med td:first-child {{ box-shadow: inset 3px 0 0 var(--yellow); }}
  td.name {{ color: var(--tx); font-family: -apple-system, "Segoe UI", "IBM Plex Sans", sans-serif; }}
  td.score {{ font-weight: 700; }}
  td.num {{ text-align: right; }}
  .amber {{ color: var(--amber); }}
  .dim {{ color: var(--tx3); }}
  .small {{ font-size: 10.5px; }}
  .mono {{ font-family: inherit; }}
  .num {{ font-variant-numeric: tabular-nums; }}

  .risk {{
    display: inline-block; font-size: 9.5px; letter-spacing: 1px;
    padding: 1px 7px; font-weight: 700; border: 1px solid;
  }}
  .risk-high {{ color: var(--red); border-color: var(--red); background: color-mix(in srgb, var(--red) 12%, transparent); }}
  .risk-med {{ color: var(--yellow); border-color: var(--yellow); background: color-mix(in srgb, var(--yellow) 10%, transparent); }}
  .risk-low {{ color: var(--green); border-color: var(--green); }}

  .tag {{
    display: inline-block; font-size: 9.5px; letter-spacing: 0.6px;
    border: 1px solid var(--line); color: var(--tx2);
    padding: 1px 6px; margin: 1px 2px 1px 0;
  }}
  .tag.dim {{ border-color: var(--line2); color: var(--tx3); }}

  tr.dossier > td {{
    background: var(--raised); border-bottom: 1px solid var(--line);
    padding: 14px 18px 16px;
  }}
  tr.dossier .evidence {{ margin: 0 0 10px; padding-left: 18px; }}
  tr.dossier .evidence li {{ color: var(--tx2); margin: 3px 0; font-family: -apple-system, "Segoe UI", "IBM Plex Sans", sans-serif; }}
  tr.dossier .evidence b {{ color: var(--tx); }}
  .rationale {{
    border-left: 2px solid var(--amber); padding: 6px 10px; margin-bottom: 10px;
    color: var(--tx2); font-size: 11.5px; background: var(--panel);
    font-family: -apple-system, "Segoe UI", "IBM Plex Sans", sans-serif;
  }}
  .drill-label {{
    font-size: 9px; letter-spacing: 1.2px; color: var(--tx3); margin: 8px 0 6px;
  }}

  /* ---- generic tables ---- */
  table.grid {{ width: 100%; border-collapse: collapse; border: 1px solid var(--line); }}
  table.grid th {{
    background: var(--raised); color: var(--tx3); font-size: 9.5px;
    letter-spacing: 1.1px; text-align: left; padding: 7px 10px;
    border-bottom: 1px solid var(--line);
  }}
  table.grid td {{
    padding: 7px 10px; border-bottom: 1px solid var(--line2);
    color: var(--tx2); font-size: 12px; vertical-align: top;
  }}
  table.grid tr:last-child td {{ border-bottom: none; }}
  table.grid tr:hover td {{ background: var(--raised); }}
  th.num, td.num {{ text-align: right; }}

  .status {{
    display: inline-block; font-size: 9.5px; letter-spacing: 1px;
    padding: 1px 7px; font-weight: 700; border: 1px solid;
  }}
  .st-ok {{ color: var(--green); border-color: var(--green); }}
  .st-warn {{ color: var(--yellow); border-color: var(--yellow); }}
  .st-off {{ color: var(--tx3); border-color: var(--line); }}

  .src {{
    display: inline-block; font-size: 9.5px; letter-spacing: 1px;
    padding: 1px 7px; font-weight: 700; border: 1px solid var(--line);
  }}
  .src-bank {{ color: var(--amber); border-color: var(--amber); }}
  .src-cdr {{ color: var(--sky); border-color: var(--sky); }}
  .src-social {{ color: var(--yellow); border-color: var(--yellow); }}
  .slots {{ display: flex; flex-wrap: wrap; gap: 3px; }}
  .warn-line {{ display: block; color: var(--yellow); font-size: 10px; margin-top: 4px; }}

  .empty {{
    border: 1px dashed var(--line); color: var(--tx3);
    padding: 18px; text-align: center; letter-spacing: 1px; font-size: 11px;
  }}

  /* ---- charts: theme swap ---- */
  [data-theme="dark"] .chart-light {{ display: none !important; }}
  [data-theme="dark"] .chart-dark {{ display: block !important; }}
  [data-theme="light"] .chart-dark {{ display: none !important; }}
  [data-theme="light"] .chart-light {{ display: block !important; }}

  /* ---- footer ---- */
  .foot {{
    margin-top: 30px; border-top: 1px solid var(--line); padding-top: 10px;
    color: var(--tx3); font-size: 10.5px; letter-spacing: 0.4px;
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 6px;
  }}
  .foot b {{ color: var(--tx2); }}
  .foot code {{ color: var(--amber); }}

  /* ---- print: collapse shell to editorial flow ---- */
  @media print {{
    body {{ background: #fff; color: #000; }}
    .sidebar, .cmdbar {{ display: none !important; }}
    .main {{ margin: 0; padding: 0; }}
    section.block {{ break-inside: avoid; }}
    [data-theme="dark"] .chart-dark {{ display: block !important; }}
    tr.dossier > td {{ background: #f5f5f5; }}
  }}

  @media (max-width: 900px) {{
    .sidebar {{ display: none; }}
    .main {{ margin-left: 0; }}
    .cmd-stats {{ display: none; }}
    .kpis {{ grid-template-columns: repeat(2, 1fr); }}
    .kpi {{ border-bottom: 1px solid var(--line2); }}
  }}
  @media (prefers-reduced-motion: reduce) {{
    * {{ scroll-behavior: auto !important; transition: none !important; }}
  }}
</style>
</head>
<body>
<div class="shell">
  <nav class="sidebar" aria-label="Section index">
    <div class="side-label">INDEX</div>
    <a class="side-item" href="#sec-01" data-sec="sec-01"><span class="no">01</span> SUMMARY</a>
    <a class="side-item" href="#sec-02" data-sec="sec-02"><span class="no">02</span> SUSPECTS</a>
    <a class="side-item" href="#sec-03" data-sec="sec-03"><span class="no">03</span> TIMELINE</a>
    <a class="side-item" href="#sec-04" data-sec="sec-04"><span class="no">04</span> NETWORK</a>
    <a class="side-item" href="#sec-05" data-sec="sec-05"><span class="no">05</span> MAP</a>
    <a class="side-item" href="#sec-06" data-sec="sec-06"><span class="no">06</span> AUDIT</a>
    <a class="side-item" href="#sec-07" data-sec="sec-07"><span class="no">07</span> SOURCES</a>
    {side_sus}
  </nav>

  <header class="cmdbar">
    <span class="wordmark">SUTRAM<span class="sep">/</span>CASE</span>
    <span class="cmd-stats" id="cmd-stats"></span>
    <div class="cmd-right">
      <div class="chips" id="chips" role="group" aria-label="Severity filter">
        <button class="chip on" data-f="ALL">ALL</button>
        <button class="chip" data-f="HIGH">HIGH</button>
        <button class="chip" data-f="MEDIUM">MED</button>
      </div>
      <select class="sortsel" id="sortsel" aria-label="Sort register">
        <option value="score">SCORE ▾</option>
        <option value="events">EVENTS ▾</option>
        <option value="rank">RANK</option>
      </select>
      <div class="theme-sw" id="theme-sw" role="button" tabindex="0"
           aria-label="Toggle dark and light mode">
        <span class="t-dark">DARK</span><span class="t-light">LIGHT</span>
      </div>
    </div>
  </header>

  <main class="main">
    <section class="block" id="sec-01">
      <div class="block-head"><span class="block-no">01</span><h2>Executive Summary</h2>
        <span class="sub">{_generated_line(res)}</span></div>
      <div class="overview sans"><p>{overview_html}</p></div>
      {kpis}
    </section>

    <section class="block" id="sec-02">
      <div class="block-head"><span class="block-no">02</span><h2>The people who stand out</h2>
        <span class="sub">{n_high} HIGH · {len(insight_list)} FLAGGED — CLICK ROW TO OPEN DOSSIER</span></div>
      <table class="reg" id="regtable">
        <thead><tr>
          <th data-k="rank">#</th><th data-k="eid">ENTITY</th><th data-k="name">NAME</th>
          <th data-k="risk">RISK</th><th data-k="score">SCORE</th>
          <th>ENGINES</th><th data-k="events" class="num">EVENTS</th>
        </tr></thead>
        <tbody id="regbody">{register}</tbody>
      </table>
    </section>

    <section class="block" id="sec-03">
      <div class="block-head"><span class="block-no">03</span><h2>When it happened</h2>
        <span class="sub">EVENT VOLUME BY MODALITY</span></div>
      {_visual_box("", "", tl_d, tl_l) if (tl_d or tl_l) else '<div class="empty">NO TIMESTAMPED RECORDS</div>'}
    </section>

    <section class="block" id="sec-04">
      <div class="block-head"><span class="block-no">04</span><h2>Who the flagged people are connected to</h2>
        <span class="sub">EGO-NETWORK · TOP COUNTERPARTIES</span></div>
      {_visual_box("", "", nw_d, nw_l) if (nw_d or nw_l) else '<div class="empty">NO COUNTERPARTY DIMENSION</div>'}
    </section>

    <section class="block" id="sec-05">
      <div class="block-head"><span class="block-no">05</span><h2>The activity map</h2>
        <span class="sub">LOCATIONS &amp; TOWER TELEMETRY</span></div>
      {_visual_box("", "", mp_d, mp_l) if (mp_d or mp_l) else '<div class="empty">NO LOCATION DATA</div>'}
    </section>

    <section class="block" id="sec-06">
      <div class="block-head"><span class="block-no">06</span><h2>Detection engine sufficiency &amp; health audit</h2>
        <span class="sub">HONEST GATING — ONLY SUPPORTED ENGINES RAN</span></div>
      {suff or '<div class="empty">NO AUDIT DATA</div>'}
    </section>

    <section class="block" id="sec-07">
      <div class="block-head"><span class="block-no">07</span><h2>Where the records came from</h2>
        <span class="sub">PROVENANCE LEDGER</span></div>
      {sources}
      {_visual_box("", "", so_d, so_l) if (so_d or so_l) else ''}
      {_visual_box("", "", pe_d, pe_l) if (pe_d or pe_l) else ''}
    </section>

    <div class="foot">
      <span><b>SUTRAM</b> · multi-source anomaly console — full detail in <code>report.json</code></span>
      <span>SELF-CONTAINED · NO EXTERNAL REQUESTS</span>
    </div>
  </main>
</div>

<script>
(function() {{
  "use strict";
  var doc = document, root = doc.documentElement;

  /* ---- theme: dark default, persisted ---- */
  function setTheme(t) {{
    root.setAttribute("data-theme", t);
    try {{ localStorage.setItem("sutram-theme", t); }} catch (e) {{}}
  }}
  function curTheme() {{ return root.getAttribute("data-theme") || "dark"; }}
  function toggleTheme() {{ setTheme(curTheme() === "dark" ? "light" : "dark"); }}
  var sw = doc.getElementById("theme-sw");
  if (sw) {{
    sw.addEventListener("click", toggleTheme);
    sw.addEventListener("keydown", function(e) {{
      if (e.key === "Enter" || e.key === " ") {{ e.preventDefault(); toggleTheme(); }}
    }});
  }}
  try {{ setTheme(localStorage.getItem("sutram-theme") || "dark"); }} catch (e) {{}}

  /* ---- command bar stats ---- */
  var stats = doc.getElementById("cmd-stats");
  if (stats) {{
    var kp = doc.querySelectorAll(".kpi .k");
    if (kp.length >= 5) {{
      stats.innerHTML = "REC <b>" + kp[0].textContent + "</b> · ENT <b>" + kp[1].textContent +
        "</b> · FILES <b>" + kp[2].textContent + "</b> · <span class='hi'>HIGH " + kp[3].textContent + "</span>";
    }}
  }}

  /* ---- register: expand dossiers (one at a time) ---- */
  var body = doc.getElementById("regbody");
  if (body) {{
    body.addEventListener("click", function(e) {{
      var tr = e.target.closest("tr.reg-row");
      if (!tr) return;
      var open = tr.classList.contains("open");
      body.querySelectorAll("tr.reg-row.open").forEach(function(r) {{
        r.classList.remove("open");
        var d = r.nextElementSibling;
        if (d && d.classList.contains("dossier")) d.hidden = true;
      }});
      if (!open) {{
        tr.classList.add("open");
        var dd = tr.nextElementSibling;
        if (dd && dd.classList.contains("dossier")) {{
          dd.hidden = false;
          var img = dd.querySelector("img");
          if (img && img.complete) img.loading = "eager";
        }}
      }}
    }});
    /* sidebar suspect links open the matching dossier */
    doc.querySelectorAll(".side-eid").forEach(function(a) {{
      a.addEventListener("click", function(ev) {{
        ev.preventDefault();
        var eid = a.getAttribute("data-eid");
        var row = body.querySelector('tr.reg-row[data-eid="' + eid + '"]');
        if (!row) return;
        var wasOpen = row.classList.contains("open");
        body.querySelectorAll("tr.reg-row.open").forEach(function(r) {{
          r.classList.remove("open");
          var d = r.nextElementSibling;
          if (d && d.classList.contains("dossier")) d.hidden = true;
        }});
        if (!wasOpen) {{ row.classList.add("open"); var d2 = row.nextElementSibling; if (d2) d2.hidden = false; }}
        row.scrollIntoView({{ block: "center" }});
      }});
    }});
  }}

  /* ---- register: filter + sort ---- */
  function rowsArr() {{
    return Array.prototype.slice.call(body.querySelectorAll("tr.reg-row"));
  }}
  function applyReg() {{
    var f = "ALL", chip = doc.querySelector(".chip.on");
    if (chip) f = chip.getAttribute("data-f");
    var sort = (doc.getElementById("sortsel") || {{}}).value || "score";
    var pairs = []; /* row -> dossier */
    rowsArr().forEach(function(r) {{
      var d = r.nextElementSibling;
      pairs.push([r, d && d.classList.contains("dossier") ? d : null]);
    }});
    var keyf = {{
      score: function(r) {{ return parseFloat(r.getAttribute("data-score")) || 0; }},
      events: function(r) {{ return parseInt(r.getAttribute("data-events"), 10) || 0; }},
      rank: function(r) {{ return parseInt(r.textContent.trim(), 10) || 999; }}
    }}[sort] || function(r) {{ return parseFloat(r.getAttribute("data-score")) || 0; }};
    if (sort !== "rank") pairs.sort(function(a, b) {{ return keyf(b[0]) - keyf(a[0]); }});
    else pairs.sort(function(a, b) {{ return keyf(a[0]) - keyf(b[0]); }});
    pairs.forEach(function(p) {{ if (p[1]) body.appendChild(p[1]); body.appendChild(p[0]); }});
    var shown = 0;
    pairs.forEach(function(p) {{
      var ok = f === "ALL" || p[0].getAttribute("data-risk") === f;
      p[0].hidden = !ok;
      if (p[1]) {{ p[1].hidden = !ok || !p[0].classList.contains("open"); }}
      if (ok) shown++;
    }});
    var cnt = doc.getElementById("reg-count");
    if (cnt) cnt.textContent = shown + " / " + pairs.length;
  }}
  doc.querySelectorAll(".chip").forEach(function(c) {{
    c.addEventListener("click", function() {{
      doc.querySelectorAll(".chip").forEach(function(x) {{ x.classList.remove("on"); }});
      c.classList.add("on");
      applyReg();
    }});
  }});
  var ss = doc.getElementById("sortsel");
  if (ss) ss.addEventListener("change", applyReg);

  /* sort header arrows */
  var heads = doc.querySelectorAll("table.reg thead th[data-k]");
  heads.forEach(function(th) {{
    th.addEventListener("click", function() {{
      var k = th.getAttribute("data-k");
      var map = {{ score: "score", events: "events", rank: "rank", "#": "rank",
                  ENTITY: "rank", NAME: "rank", RISK: "rank" }};
      var sel = doc.getElementById("sortsel");
      var val = map[k] || (k === "events" ? "events" : k === "score" ? "score" : null);
      if (val && sel) {{ sel.value = val; applyReg(); }}
    }});
  }});

  /* ---- scroll-spy ---- */
  var secs = ["sec-01", "sec-02", "sec-03", "sec-04", "sec-05", "sec-06", "sec-07"];
  if ("IntersectionObserver" in window) {{
    var io = new IntersectionObserver(function(entries) {{
      entries.forEach(function(en) {{
        if (en.isIntersecting) {{
          doc.querySelectorAll(".side-item").forEach(function(a) {{
            a.classList.toggle("active", a.getAttribute("data-sec") === en.target.id);
          }});
        }}
      }});
    }}, {{ rootMargin: "-45% 0px -45% 0px" }});
    secs.forEach(function(id) {{
      var el = doc.getElementById(id);
      if (el) io.observe(el);
    }});
  }}
}})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# JSON (machine-readable; keeps the technical scores the report omits)
# ---------------------------------------------------------------------------
def _json(res, cfg) -> dict:
    unified = res.unified
    n_entities = int(unified["entity_id"].nunique()) if not unified.empty else 0
    scores = res.model_scores or {}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "records": int(len(unified)),
            "people": n_entities,
            "source_files": len(res.per_source),
            "people_who_stand_out": len(res.rankings),
        },
        "sources": [
            {
                "file": getattr(rs, "file", ""),
                "source": rs.source,
                "rows": rs.n_rows,
                "people": rs.n_entities,
                "resolved_slots": sorted(rs.resolved_fields),
                "warnings": rs.warnings,
            }
            for rs in res.per_source
        ],
        "model_scores": {m: dict(sorted(s.items(), key=lambda kv: -kv[1])[:20])
                         for m, s in scores.items()},
        "rankings": res.rankings,
    }


# ---------------------------------------------------------------------------
# PDF (reportlab)
# ---------------------------------------------------------------------------
def _pdf(res, cfg, report_dir: Path, charts: Dict[str, Path]) -> Path:
    """Build a printable PDF led by plain-language findings."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    path = report_dir / "report.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=14 * mm, rightMargin=14 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm,
                            title="Sutram report")
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Title"], textColor=colors.HexColor(NAVY),
                        fontSize=20, spaceAfter=2)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor(NAVY),
                        fontSize=13, spaceBefore=10, spaceAfter=6)
    h3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=10.5,
                        textColor=colors.HexColor(NAVY), spaceBefore=8, spaceAfter=3)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9,
                          textColor=colors.HexColor(NAVY))
    meta = ParagraphStyle("Meta", parent=styles["BodyText"], fontSize=8,
                          textColor=colors.HexColor(MUTED))
    risk_colors = {"HIGH": colors.HexColor(RED), "MEDIUM": colors.HexColor(AMBER),
                   "LOW": colors.HexColor(MINT)}

    insight_list = _insights_for(res, cfg)
    story = [
        Paragraph(REPORT_TITLE, h1),
        Paragraph(f"{_generated_line(res)} · {REPORT_SUBTITLE}", meta),
        Spacer(1, 4 * mm),
    ]
    for line in _overview_lines(res, insight_list):
        story.append(Paragraph(_pdfize(line), body))
    if charts.get("people"):
        story += [Spacer(1, 2 * mm),
                  Image(str(charts["people"]), width=170 * mm, height=64 * mm)]
    story.append(Spacer(1, 3 * mm))

    if insight_list:
        story.append(Paragraph("The people who stand out", h2))
        for it in insight_list:
            rc = risk_colors.get(it["risk"], colors.HexColor(MUTED))
            story.append(Paragraph(f"{it['rank']}. <b>{it['name']}</b> — "
                                   f"<font color='#{rc.hexval()[2:]}'>{it['risk']}</font>", h3))
            for bl in it["evidence"]:
                story.append(Paragraph(f"• {_pdfize(bl)}", body))
        story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("Where the records came from", h2))
    story.append(Table(
        [["file", "source", "records", "people"]] +
        [[Path(rs.file).name, rs.source, rs.n_rows, rs.n_entities]
         for rs in res.per_source] or [["—", "—", 0, 0]],
        colWidths=[78 * mm, 24 * mm, 30 * mm, 30 * mm],
        style=TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(NAVY)),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cfdcd6")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor(LIGHT)]),
        ]),
    ))
    if charts.get("sources"):
        story += [Spacer(1, 2 * mm),
                  Image(str(charts["sources"]), width=170 * mm, height=48 * mm)]
    story.append(Spacer(1, 3 * mm))

    if charts.get("timeline") and _has_timeline(res):
        story.append(Paragraph("When it happened", h2))
        story.append(Image(str(charts["timeline"]), width=170 * mm, height=54 * mm))
        story.append(Spacer(1, 3 * mm))

    if charts.get("network"):
        story.append(Paragraph("Who the flagged people are connected to", h2))
        story.append(Image(str(charts["network"]), width=170 * mm, height=90 * mm))
        story.append(Spacer(1, 3 * mm))

    if charts.get("map"):
        story.append(Paragraph("The activity map", h2))
        story.append(Image(str(charts["map"]), width=170 * mm, height=92 * mm))
        story.append(Spacer(1, 3 * mm))

    if charts.get("drill"):
        story.append(Paragraph("A closer look at the people who stand out", h2))
        for title, img in charts["drill"].items():
            story.append(Paragraph(f"<b>{_pdfize(title)}</b>", h3))
            story.append(Image(str(img), width=170 * mm, height=60 * mm))
            story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("The full technical detail behind this analysis is "
                           "kept with this report for the record — this page "
                           "is the human-readable summary.", meta))
    doc.build(story)
    return path


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def generate_report(result, config, report_dir: str | Path = "data/report") -> dict:
    """Build and write the full report. Returns paths of written files."""
    report_dir = Path(report_dir)
    charts_dir = report_dir / "charts"
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dark theme charts (default visual experience)
    charts_dark = {
        "people": visuals.render_person_attention(result, charts_dir, name="people_dark.png", theme="dark"),
        "sources": visuals.render_sources(result, charts_dir, name="sources_dark.png", theme="dark"),
        "timeline": visuals.render_timeline(result, charts_dir, name="timeline_dark.png", theme="dark"),
        "network": visuals.render_ego_graph(result, charts_dir, name="network_dark.png", theme="dark"),
        "map": visuals.render_location_map(result, charts_dir, name="map_dark.png", theme="dark"),
    }
    # 2. Light theme charts (standards & toggle support)
    charts_light = {
        "people": visuals.render_person_attention(result, charts_dir, name="people.png", theme="light"),
        "sources": visuals.render_sources(result, charts_dir, name="sources.png", theme="light"),
        "timeline": visuals.render_timeline(result, charts_dir, name="timeline.png", theme="light"),
        "network": visuals.render_ego_graph(result, charts_dir, name="network.png", theme="light"),
        "map": visuals.render_location_map(result, charts_dir, name="map.png", theme="light"),
    }
    top = _insights_for(result, config) or []
    charts_dark["drill"] = {
        f"{it['name']} ({it['risk']})": visuals.render_entity_timeline(
            result, charts_dir, it["entity_id"], name=f"entity_{it['entity_id']}_dark.png", theme="dark")
        for it in top[:3]
    }
    charts_light["drill"] = {
        f"{it['name']} ({it['risk']})": visuals.render_entity_timeline(
            result, charts_dir, it["entity_id"], name=f"entity_{it['entity_id']}.png", theme="light")
        for it in top[:3]
    }

    charts = {
        **charts_light,
        "dark": charts_dark,
        "light": charts_light,
    }

    markdown = _markdown(result, config, charts_dark)
    html = _html(result, config, charts)
    json_body = _json(result, config)
    pdf_path = _pdf(result, config, report_dir, charts_light)

    (report_dir / "report.md").write_text(markdown, encoding="utf-8")
    (report_dir / "report.html").write_text(html, encoding="utf-8")
    (report_dir / "report.json").write_text(
        json.dumps(json_body, indent=2, default=str), encoding="utf-8"
    )
    return {
        "markdown": str(report_dir / "report.md"),
        "html": str(report_dir / "report.html"),
        "pdf": str(pdf_path),
        "json": str(report_dir / "report.json"),
    }
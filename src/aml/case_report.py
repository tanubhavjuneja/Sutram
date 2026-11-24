"""Draft case report generation.

Produces a court-reviewable structure:

    metadata · executive summary · evidence inventory · entities · findings ·
    timeline · recommendations · case diary · methodology & limitations ·
    analytics appendix · query log · disclaimer

Markdown, HTML, JSON and (when reportlab is available) PDF are supported.  The
report is a **draft**: it links every finding to evidence and carries an
explicit verification disclaimer.
"""
from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .diary import render_diary_md, verify_chain
from .i18n import t
from .index import EvidenceIndex, _safe_ts, _short
from .case import Case, Finding, QueryPlan
from .summarize import case_summary, limitations

_SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
_PRIORITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def _fmt_ts(value: str) -> str:
    return (value or "").replace("T", " ")[:19] or "—"


def _fmt_money(value: float) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _short_hash(value: str) -> str:
    return (value or "")[:16] + ("…" if value else "")


def _key_timeline(index: EvidenceIndex, limit: int = 60) -> List[dict]:
    items: List[dict] = []
    if not index.events.empty:
        for _, row in index.events.dropna(subset=["event_ts"]).sort_values("event_ts").iterrows():
            items.append({
                "timestamp": row["event_ts"].isoformat(),
                "source": str(row.get("source", "")),
                "title": f"{index.entity_label(str(row.get('entity_id', '')))} → "
                         f"{index.entity_label(str(row.get('counterparty_id', '')))}",
                "snippet": index.citation_for_event(row).snippet,
                "evidence_id": str(row.get("evidence_id", "")),
            })
    for chunk in index.chunks:
        ts = _safe_ts(chunk.timestamp)
        if ts is None:
            continue
        item = index.case.evidence_by_id(chunk.evidence_id)
        items.append({
            "timestamp": ts.isoformat(),
            "source": chunk.source_type,
            "title": item.filename if item else chunk.evidence_id,
            "snippet": _short(chunk.text, 240),
            "evidence_id": chunk.evidence_id,
        })
    items.sort(key=lambda x: x["timestamp"])
    if len(items) <= limit:
        return items
    # evenly sample so the report shows beginning, middle and end
    step = max(1, len(items) // limit)
    return items[::step][:limit]


def _report_data(case: Case, index: EvidenceIndex, lang: str, min_events: int = 30) -> dict:
    summary = case_summary(case, index, lang=lang, min_events=min_events)
    findings = sorted(
        case.findings,
        key=lambda f: (_SEVERITY_ORDER.get(f.severity, 3), -f.confidence, f.finding_id),
    )
    recommendations = sorted(
        case.recommendations,
        key=lambda r: (_PRIORITY_ORDER.get(r.priority, 3), r.recommendation_id),
    )
    entities = sorted(
        case.entities.values(),
        key=lambda e: (-e.risk_score, -e.n_events, e.display_name),
    )
    chain_ok, chain_errors = verify_chain(case)
    analytics = (case.metadata or {}).get("analytics", {}) or {}
    return {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "classification": "DRAFT — INVESTIGATIVE USE",
        "prepared_by": "Sutram Investigator",
        "case": {
            "case_id": case.case_id,
            "title": case.title,
            "description": case.description,
            "language": case.language,
            "status": case.status,
            "created_at": case.created_at,
            "updated_at": case.updated_at,
        },
        "summary": summary,
        "evidence": index.evidence_summary(),
        "entities": [e.to_dict() for e in entities],
        "findings": [f.to_dict() for f in findings],
        "timeline": _key_timeline(index),
        "recommendations": [r.to_dict() for r in recommendations],
        "diary": [d.to_dict() for d in case.diary],
        "diary_chain_ok": chain_ok,
        "diary_chain_errors": chain_errors,
        "limitations": limitations(case, index, lang=lang),
        "analytics": {
            "report_files": analytics.get("report_files", {}),
            "sufficiency": analytics.get("sufficiency", {}),
            "warnings": analytics.get("warnings", []),
            "n_models": analytics.get("n_models", 0),
            "top_rankings": analytics.get("rankings", [])[:10],
        },
        "query_log": [
            d.to_dict() for d in case.diary if d.entry_type == "query"
        ],
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "_None recorded._\n"
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |")
    return "\n".join(lines) + "\n"


def _md_report(data: dict, case: Case, index: EvidenceIndex, lang: str) -> str:
    c = data["case"]
    s = data["summary"]
    lines: List[str] = [
        f"# {t('report.case_report', lang)} — {c['title']}",
        "",
        f"*{t('report.case_id', lang)}: `{c['case_id']}` · {t('report.generated', lang)}: "
        f"{_fmt_ts(data['generated_at'])} · {t('report.classification', lang)}: "
        f"{data['classification']}*",
        "",
        f"**{t('report.status', lang)}:** {c['status']} · "
        f"**{t('report.language', lang)}:** {c['language']} · "
        f"**{t('report.prepared_by', lang)}:** {data['prepared_by']}",
        "",
    ]
    if c.get("description"):
        lines += [c["description"], ""]

    # executive summary
    lines += [f"## {t('report.executive_summary', lang)}", "", s["text"], ""]
    if s.get("gaps"):
        lines += [f"**{t('report.evidence_gap', lang)}:** " +
                  "; ".join(g["key"].replace("_", " ") for g in s["gaps"]), ""]
    if s.get("recommendations"):
        lines += [f"**{t('report.next_steps', lang)}:** " +
                  s["recommendations"][0]["action"], ""]

    # evidence inventory
    lines += [f"## {t('report.evidence_inventory', lang)}", ""]
    lines.append(_md_table(
        ["ID", "File", "Type", "Rows", "Chunks", "SHA-256", "Acquired"],
        [[e["evidence_id"], e["filename"], e["source_type"], e["n_rows"],
          e["n_chunks"], f"`{_short_hash(e['sha256'])}`", _fmt_ts(e["acquired_at"])]
         for e in data["evidence"]],
    ))
    lines.append("")

    # entities
    lines += [f"## {t('report.entities', lang)}", ""]
    lines.append(_md_table(
        ["ID", "Name", "Risk", "Events", "Sources", "Identifiers"],
        [[e["entity_id"], e["display_name"] if "display_name" in e else (e.get("names") or [e["entity_id"]])[0],
          _fmt_money(e.get("risk_score", 0)), e.get("n_events", 0),
          ", ".join(e.get("sources", [])), ", ".join(e.get("identifiers", [])[:6])]
         for e in data["entities"][:40]],
    ))
    lines.append("")

    # findings
    lines += [f"## {t('report.findings', lang)}", ""]
    if not data["findings"]:
        lines += [t("report.no_findings", lang), ""]
    else:
        lines.append(_md_table(
            ["Severity", "Kind", "Title", "Confidence", "Entities", "Evidence"],
            [[f["severity"], f["kind"], f["title"], f"{f['confidence']:.2f}",
              ", ".join(f.get("entity_ids", [])), ", ".join(f.get("evidence_ids", [])[:6])]
             for f in data["findings"][:30]],
        ))
        lines.append("")
        for f in data["findings"][:15]:
            lines += [f"### [{f['severity']}] {f['title']}", "",
                      f"- **kind:** {f['kind']} · **confidence:** {f['confidence']:.2f} · "
                      f"**entities:** {', '.join(f.get('entity_ids', [])) or '—'}",
                      f"- {f['description']}"]
            for citation in f.get("citations", [])[:4]:
                lines.append(
                    f"- evidence `{citation.get('evidence_id', '')}` · "
                    f"{citation.get('file', '')} · {_fmt_ts(citation.get('timestamp', ''))} — "
                    f"{citation.get('snippet', '')}"
                )
            lines.append("")

    # timeline
    lines += [f"## {t('report.timeline', lang)}", ""]
    if not data["timeline"]:
        lines += ["_No dated events._", ""]
    else:
        lines.append(_md_table(
            ["When", "Source", "What", "Evidence"],
            [[_fmt_ts(item["timestamp"]), item["source"], item["title"], item["evidence_id"]]
             for item in data["timeline"]],
        ))
        lines.append("")

    # recommendations
    lines += [f"## {t('report.recommendations', lang)}", ""]
    if not data["recommendations"]:
        lines += [t("report.no_recommendations", lang), ""]
    else:
        lines.append(_md_table(
            ["Priority", "Category", "Action", "Rationale", "Links"],
            [[r["priority"], r["category"], r["action"], r["rationale"],
              ", ".join(r.get("finding_ids", []) + r.get("entity_ids", []) + r.get("evidence_ids", []))]
             for r in data["recommendations"]],
        ))
        lines.append("")

    # diary
    lines += [f"## {t('report.case_diary', lang)}", ""]
    status = t("report.diary_chain_ok", lang) if data["diary_chain_ok"] else t("report.diary_chain_bad", lang)
    lines += [f"*{status}*", ""]
    for entry in data["diary"]:
        lines.append(
            f"- **{_fmt_ts(entry['timestamp'])}** · `{entry['entry_type']}` · "
            f"{entry['actor']} — {entry['text']}"
        )
    lines.append("")

    # methodology / limitations
    lines += [f"## {t('report.methodology', lang)}", ""]
    lines += ["The case was assembled from the evidence inventory above. "
              "Structured records were normalized by runtime schema detection and "
              "cross-source entity resolution; documents were chunked with line/page "
              "provenance. Findings combine rule-based, statistical, graph and "
              "time-correlation signals. Every finding links to evidence; every "
              "query and action is recorded in the hash-chained case diary. "
              "Report labels and recommendations are localized; evidence quotations "
              "and historical diary entries are preserved in the language in which "
              "they were recorded.", ""]
    lines += [f"### {t('report.limitations', lang)}", ""]
    for item in data["limitations"]:
        lines.append(f"- {item}")
    lines.append("")

    # analytics appendix
    lines += [f"## {t('report.analytics_appendix', lang)}", ""]
    files = data["analytics"].get("report_files", {}) or {}
    if files.get("html"):
        lines.append(f"- Analytics report: `{Path(files['html']).name}`")
    if files.get("pdf"):
        lines.append(f"- Analytics PDF: `{Path(files['pdf']).name}`")
    lines.append(f"- Models run: {data['analytics'].get('n_models', 0)}")
    suff = data["analytics"].get("sufficiency", {}) or {}
    if suff:
        lines.append(_md_table(
            ["Model", "Verdict", "Reason"],
            [[m, (v.get("status", "") if isinstance(v, dict) else str(v)),
              (v.get("reason", "") if isinstance(v, dict) else "")]
             for m, v in sorted(suff.items())],
        ))
    if data["analytics"].get("warnings"):
        lines += ["**Warnings**", ""] + [f"- {w}" for w in data["analytics"]["warnings"][:20]] + [""]
    lines.append("")

    # query log
    lines += [f"## {t('report.query_log', lang)}", ""]
    if data["query_log"]:
        for entry in data["query_log"]:
            lines.append(f"- {_fmt_ts(entry['timestamp'])} — {entry['text']}")
    else:
        lines.append("_No queries recorded._")
    lines.append("")

    lines += ["---", "", f"*{t('report.disclaimer', lang)}*", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
_CSS = """
:root { --ink:#0f172a; --muted:#64748b; --line:#d8e0ea; --bg:#ffffff;
        --panel:#f8fafc; --amber:#b45309; --red:#b91c1c; --green:#047857; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; }
main { max-width:1040px; margin:0 auto; padding:38px 30px 80px; }
header { border-bottom:3px solid var(--ink); padding-bottom:16px; margin-bottom:26px; }
h1 { font-size:27px; margin:0 0 8px; letter-spacing:-0.3px; }
h2 { font-size:18px; margin:34px 0 10px; border-bottom:1px solid var(--line);
     padding-bottom:6px; }
h3 { font-size:14px; margin:20px 0 6px; }
p { margin:8px 0; }
.meta { color:var(--muted); font-size:12px; }
.badge { display:inline-block; border:1px solid var(--amber); color:var(--amber);
         padding:2px 8px; font-size:11px; font-weight:700; letter-spacing:.6px; }
table { border-collapse:collapse; width:100%; margin:10px 0 18px; font-size:12px; }
th { background:var(--panel); text-align:left; font-weight:700; }
th,td { border:1px solid var(--line); padding:6px 8px; vertical-align:top; }
tr:nth-child(even) td { background:#fcfdff; }
code { background:var(--panel); padding:1px 4px; border-radius:3px; font-size:12px; }
ul { margin:6px 0 12px 20px; padding:0; }
li { margin:3px 0; }
.sev-HIGH { color:var(--red); font-weight:700; }
.sev-MEDIUM { color:var(--amber); font-weight:700; }
.sev-LOW { color:var(--green); font-weight:700; }
.note { background:var(--panel); border-left:3px solid var(--amber);
        padding:10px 14px; margin:14px 0; font-size:12px; }
.cite { color:var(--muted); font-size:11px; }
footer { margin-top:40px; border-top:1px solid var(--line); padding-top:14px;
         color:var(--muted); font-size:11px; }
@media print { main { max-width:none; padding:0; } body { font-size:11px; }
                h2 { page-break-after:avoid; } table { page-break-inside:auto; } }
"""


def _md_inline(text: str) -> str:
    text = html.escape(str(text))
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", text)
    return text


def _md_to_html(md: str) -> str:
    """Minimal Markdown -> HTML for the report subset (headings, tables, lists)."""
    lines = md.split("\n")
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("### "):
            out.append(f"<h3>{_md_inline(line[4:])}</h3>")
            i += 1
            continue
        if line.startswith("## "):
            out.append(f"<h2>{_md_inline(line[3:])}</h2>")
            i += 1
            continue
        if line.startswith("# "):
            out.append(f"<h1>{_md_inline(line[2:])}</h1>")
            i += 1
            continue
        if line.strip() == "---":
            out.append("<hr>")
            i += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            if len(table_lines) >= 2:
                headers = [c.strip() for c in table_lines[0].strip("|").split("|")]
                rows = []
                for row in table_lines[2:]:
                    rows.append([c.strip() for c in row.strip("|").split("|")])
                out.append("<table><thead><tr>" +
                           "".join(f"<th>{_md_inline(h)}</th>" for h in headers) +
                           "</tr></thead><tbody>")
                for row in rows:
                    out.append("<tr>" + "".join(f"<td>{_md_inline(c)}</td>" for c in row) + "</tr>")
                out.append("</tbody></table>")
            continue
        if line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(lines[i][2:])
                i += 1
            out.append("<ul>" + "".join(f"<li>{_md_inline(item)}</li>" for item in items) + "</ul>")
            continue
        # paragraph (collect until blank / structural line)
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||- |---)", lines[i]):
            para.append(lines[i])
            i += 1
        out.append(f"<p>{_md_inline(' '.join(para))}</p>")
    return "\n".join(out)


def _html_report(md: str, data: dict, lang: str) -> str:
    c = data["case"]
    body = _md_to_html(md)
    return f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(t('report.case_report', lang))} — {html.escape(c['title'])}</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<header>
  <div class="badge">{html.escape(data['classification'])}</div>
  <h1>{html.escape(t('report.case_report', lang))} — {html.escape(c['title'])}</h1>
  <div class="meta">{html.escape(t('report.case_id', lang))}: {html.escape(c['case_id'])} ·
  {html.escape(t('report.generated', lang))}: {html.escape(_fmt_ts(data['generated_at']))} ·
  {html.escape(t('report.prepared_by', lang))}: {html.escape(data['prepared_by'])}</div>
</header>
{body}
<footer>{html.escape(t('report.disclaimer', lang))}</footer>
</main>
</body>
</html>"""


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def _pdf_report(data: dict, path: Path, lang: str) -> bool:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer,
                                        Table, TableStyle)
    except Exception:
        return False

    def esc(value: str) -> str:
        return html.escape(str(value or "")).replace("\n", "<br/>")

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Title"], fontSize=19, spaceAfter=4)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12,
                        spaceBefore=12, spaceAfter=5)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=8.5, leading=11)
    small = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=7.2, leading=9)
    try:
        doc = SimpleDocTemplate(str(path), pagesize=A4,
                                leftMargin=14 * mm, rightMargin=14 * mm,
                                topMargin=12 * mm, bottomMargin=12 * mm,
                                title=f"Case report — {data['case']['title']}")
        story = [
            Paragraph(esc(f"{t('report.case_report', lang)} — {data['case']['title']}"), h1),
            Paragraph(esc(f"{t('report.case_id', lang)}: {data['case']['case_id']} · "
                          f"{t('report.generated', lang)}: {_fmt_ts(data['generated_at'])} · "
                          f"{data['classification']}"), small),
            Spacer(1, 5 * mm),
            Paragraph(esc(t("report.executive_summary", lang)), h2),
            Paragraph(esc(data["summary"]["text"]), body),
        ]

        def table(headers, rows, widths=None):
            data_rows = [headers] + rows
            tbl = Table(data_rows, colWidths=widths, repeatRows=1)
            tbl.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 4 * mm))

        story.append(Paragraph(esc(t("report.evidence_inventory", lang)), h2))
        table(["ID", "File", "Type", "Rows", "Chunks"],
              [[e["evidence_id"], e["filename"], e["source_type"], e["n_rows"], e["n_chunks"]]
               for e in data["evidence"]],
              widths=[18 * mm, 88 * mm, 24 * mm, 16 * mm, 18 * mm])

        story.append(Paragraph(esc(t("report.entities", lang)), h2))
        table(["ID", "Name", "Risk", "Events", "Sources"],
              [[e["entity_id"], (e.get("names") or [e["entity_id"]])[0],
                f"{e.get('risk_score', 0):.2f}", e.get("n_events", 0),
                ", ".join(e.get("sources", []))]
               for e in data["entities"][:30]],
              widths=[18 * mm, 60 * mm, 14 * mm, 16 * mm, 60 * mm])

        story.append(Paragraph(esc(t("report.findings", lang)), h2))
        if data["findings"]:
            table(["Severity", "Kind", "Title", "Conf."],
                  [[f["severity"], f["kind"], f["title"], f"{f['confidence']:.2f}"]
                   for f in data["findings"][:25]],
                  widths=[18 * mm, 24 * mm, 120 * mm, 14 * mm])
        else:
            story.append(Paragraph(esc(t("report.no_findings", lang)), body))

        story.append(Paragraph(esc(t("report.recommendations", lang)), h2))
        for rec in data["recommendations"][:15]:
            story.append(Paragraph(
                esc(f"[{rec['priority']}] {rec['action']} — {rec['rationale']}"), body))
            story.append(Spacer(1, 1.5 * mm))

        story.append(Paragraph(esc(t("report.case_diary", lang)), h2))
        for entry in data["diary"][:40]:
            story.append(Paragraph(
                esc(f"{_fmt_ts(entry['timestamp'])} · {entry['entry_type']} · {entry['text']}"),
                small))

        story.append(Paragraph(esc(t("report.methodology", lang)), h2))
        for item in data["limitations"]:
            story.append(Paragraph(esc(f"• {item}"), body))
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(esc(t("report.disclaimer", lang)), small))
        doc.build(story)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def generate_case_report(case: Case, index: EvidenceIndex, lang: Optional[str] = None,
                         out_dir: str | Path = "", formats=("md", "html", "json", "pdf"),
                         min_events: int = 30) -> Dict[str, str]:
    lang = lang or case.language or "en"
    out_dir = Path(out_dir or (Path(case.root or ".") / "reports"))
    out_dir.mkdir(parents=True, exist_ok=True)
    data = _report_data(case, index, lang, min_events=min_events)
    md = _md_report(data, case, index, lang)

    written: Dict[str, str] = {}
    if "md" in formats:
        path = out_dir / "case_report.md"
        path.write_text(md, encoding="utf-8")
        written["md"] = str(path)
    if "html" in formats:
        path = out_dir / "case_report.html"
        path.write_text(_html_report(md, data, lang), encoding="utf-8")
        written["html"] = str(path)
    if "json" in formats:
        path = out_dir / "case_report.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
        written["json"] = str(path)
    if "pdf" in formats:
        path = out_dir / "case_report.pdf"
        if _pdf_report(data, path, lang):
            written["pdf"] = str(path)
    # diary export
    diary_path = out_dir / "case_diary.md"
    diary_path.write_text(render_diary_md(case, lang), encoding="utf-8")
    written["diary"] = str(diary_path)
    return written

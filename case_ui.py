"""Native PySide6 case workspace for main_app.py.

This page is the desktop equivalent of the web workspace and uses the exact
same ``aml`` pipeline/service:

* case selection / creation / demo
* evidence ingestion and analytics
* conversational assistant with citations, follow-ups, image attachments
* generated charts / FLUX illustrations rendered inline
* evidence, entities, findings, leads, diary and report tabs
* Cloudflare Workers AI status
"""
from __future__ import annotations

import base64
import html
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("AML_SKIP_TORCH", "1")

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QTextDocument
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from aml.i18n import SUPPORTED_LANGUAGES, language_name, t


class CaseWorker(QThread):
    """Run a potentially slow case operation off the UI thread."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn())
        except Exception as exc:  # surface the error in the UI
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class CaseWorkspacePage(QWidget):
    home_requested = Signal()
    toast_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CaseWorkspace")
        self.svc = None
        self.case = None
        self._chat: list = []
        self._attached: list = []
        self._worker = None
        self._pending_cb = None
        self._last_report_html = ""
        self._init_service()
        self._build_ui()
        self.refresh_cases()

    # -- service -----------------------------------------------------------
    def _init_service(self):
        from aml.config import load_investigation_config
        from aml.service import InvestigationService
        cfg = load_investigation_config(
            str(ROOT / "config" / "investigation.yaml"),
            case_root=str(ROOT / "data" / "cases"),
        )
        self.svc = InvestigationService(case_root=cfg.case_root, config=cfg)

    def _lang(self) -> str:
        return self.lang_combo.currentData() or "en"

    def _case_id(self) -> str:
        return self.case_combo.currentData() or ""

    def _url(self, path) -> str:
        try:
            return QUrl.fromLocalFile(str(Path(path).resolve())).toString()
        except Exception:
            return ""

    # -- UI ----------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        header = QFrame()
        header.setObjectName("GlassPanelStrong")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(14, 9, 14, 9)
        hl.setSpacing(8)
        title = QLabel("CASE WORKSPACE")
        title.setObjectName("SectionTitle")
        hl.addWidget(title)

        self.case_combo = QComboBox()
        self.case_combo.setMinimumWidth(300)
        self.case_combo.currentIndexChanged.connect(self._case_changed)
        hl.addWidget(self.case_combo, 1)

        self.btn_new = QPushButton("New case")
        self.btn_new.setObjectName("GhostBtn")
        self.btn_new.clicked.connect(self.new_case)
        hl.addWidget(self.btn_new)

        self.btn_demo = QPushButton("Load demo")
        self.btn_demo.setObjectName("AccentBtn")
        self.btn_demo.clicked.connect(self.load_demo)
        hl.addWidget(self.btn_demo)

        self.btn_ingest = QPushButton("Add evidence")
        self.btn_ingest.setObjectName("GhostBtn")
        self.btn_ingest.clicked.connect(self.ingest)
        hl.addWidget(self.btn_ingest)

        self.btn_analyze = QPushButton("Analyze")
        self.btn_analyze.setObjectName("PrimaryBtn")
        self.btn_analyze.clicked.connect(self.analyze)
        hl.addWidget(self.btn_analyze)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setObjectName("GhostBtn")
        self.btn_refresh.clicked.connect(self.refresh_views)
        hl.addWidget(self.btn_refresh)

        self.lang_combo = QComboBox()
        for code in SUPPORTED_LANGUAGES:
            self.lang_combo.addItem(language_name(code), code)
        self.lang_combo.setMinimumWidth(110)
        self.lang_combo.currentIndexChanged.connect(self._language_changed)
        hl.addWidget(self.lang_combo)

        self.btn_home = QPushButton("← Analytics")
        self.btn_home.setObjectName("GhostBtn")
        self.btn_home.clicked.connect(self.home_requested)
        hl.addWidget(self.btn_home)
        root.addWidget(header)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        root.addWidget(self.splitter, 1)

        # ---- left: case card + sources -----------------------------------
        left = QFrame()
        left.setObjectName("GlassPanel")
        left.setMinimumWidth(210)
        left.setMaximumWidth(280)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(14, 14, 14, 14)
        ll.setSpacing(8)
        self.case_title = QLabel("No case selected")
        self.case_title.setObjectName("CaseTitle")
        self.case_title.setWordWrap(True)
        ll.addWidget(self.case_title)
        self.case_meta = QLabel("")
        self.case_meta.setObjectName("CaseMeta")
        self.case_meta.setWordWrap(True)
        ll.addWidget(self.case_meta)
        ll.addSpacing(6)

        stats = QGridLayout()
        stats.setSpacing(4)
        self.stat_values = {}
        for row, (key, label) in enumerate((
            ("evidence", "EVIDENCE"), ("entities", "ENTITIES"),
            ("findings", "FINDINGS"), ("leads", "LEADS"),
        )):
            value = QLabel("0")
            value.setObjectName("CaseStatValue")
            caption = QLabel(label)
            caption.setObjectName("CaseStatLabel")
            stats.addWidget(value, row // 2 * 2, row % 2)
            stats.addWidget(caption, row // 2 * 2 + 1, row % 2)
            self.stat_values[key] = value
        ll.addLayout(stats)
        ll.addSpacing(6)

        self.llm_label = QLabel("")
        self.llm_label.setObjectName("LLMStatus")
        self.llm_label.setWordWrap(True)
        ll.addWidget(self.llm_label)

        source_title = QLabel("SOURCES")
        source_title.setObjectName("SectionSub")
        ll.addWidget(source_title)
        self.source_list = QListWidget()
        self.source_list.setObjectName("CaseSourceList")
        ll.addWidget(self.source_list, 1)
        self.splitter.addWidget(left)

        # ---- center: tabs ------------------------------------------------
        self.tabs = QTabWidget()
        self._build_assistant_tab()
        self._build_evidence_tab()
        self._build_entities_tab()
        self._build_findings_tab()
        self._build_leads_tab()
        self._build_diary_tab()
        self._build_report_tab()
        self.splitter.addWidget(self.tabs)

        # ---- right: citations / follow-ups ------------------------------
        right = QFrame()
        right.setObjectName("Inspector")
        right.setMinimumWidth(280)
        right.setMaximumWidth(380)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(12, 12, 12, 12)
        rl.setSpacing(8)
        head = QHBoxLayout()
        label = QLabel("CITATIONS")
        label.setObjectName("SectionSub")
        head.addWidget(label)
        head.addStretch(1)
        self.confidence = QLabel("—")
        self.confidence.setObjectName("Badge")
        head.addWidget(self.confidence)
        rl.addLayout(head)
        self.citation_view = QTextBrowser()
        self.citation_view.setObjectName("Inspector")
        self.citation_view.setOpenExternalLinks(True)
        rl.addWidget(self.citation_view, 2)

        follow_label = QLabel("FOLLOW-UPS")
        follow_label.setObjectName("SectionSub")
        rl.addWidget(follow_label)
        self.followup_box = QVBoxLayout()
        self.followup_box.setSpacing(4)
        follow_host = QWidget()
        follow_host.setLayout(self.followup_box)
        rl.addWidget(follow_host, 1)
        self.splitter.addWidget(right)
        self.splitter.setSizes([230, 780, 300])

        self.status = QLabel("Ready.")
        self.status.setObjectName("SectionSub")
        root.addWidget(self.status)

    def _make_table(self, headers):
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        return table

    def _build_assistant_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(7)
        self.chat_view = QTextBrowser()
        self.chat_view.setObjectName("Chat")
        self.chat_view.setOpenExternalLinks(True)
        lay.addWidget(self.chat_view, 1)

        tools = QHBoxLayout()
        tools.setSpacing(5)
        for label, prompt in (
            ("Rerun analysis", "Rerun the analysis."),
            ("Recheck numbers", "Recheck the numbers."),
            ("Plot network", "Plot the correlation network."),
            ("Generate image", "Generate an image of a clean forensic case board."),
        ):
            btn = QPushButton(label)
            btn.setObjectName("GhostBtn")
            btn.setStyleSheet("font-size:10px; padding:4px 7px;")
            btn.clicked.connect(lambda _=False, q=prompt: self.ask(q))
            tools.addWidget(btn)
        tools.addStretch(1)
        lay.addLayout(tools)

        attach = QHBoxLayout()
        self.btn_attach = QPushButton("Attach image")
        self.btn_attach.setObjectName("GhostBtn")
        self.btn_attach.clicked.connect(self.attach_images)
        attach.addWidget(self.btn_attach)
        self.attach_label = QLabel("No image attached")
        self.attach_label.setObjectName("SectionSub")
        attach.addWidget(self.attach_label)
        attach.addStretch(1)
        lay.addLayout(attach)

        row = QHBoxLayout()
        self.input = QTextEdit()
        self.input.setPlaceholderText(
            "Ask about the evidence, an entity, a timeline, or what to do next…")
        self.input.setMaximumHeight(88)
        row.addWidget(self.input, 1)
        self.btn_send = QPushButton("Send")
        self.btn_send.setObjectName("PrimaryBtn")
        self.btn_send.clicked.connect(self.send)
        row.addWidget(self.btn_send)
        lay.addLayout(row)

        self.suggestions_host = QWidget()
        self.suggestions = QHBoxLayout(self.suggestions_host)
        self.suggestions.setContentsMargins(0, 0, 0, 0)
        self.suggestions.setSpacing(5)
        lay.addWidget(self.suggestions_host)
        self.tabs.addTab(tab, "ASSISTANT")

    def _build_evidence_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(8, 8, 8, 8)
        self.evidence_table = self._make_table(
            ["ID", "File", "Type", "Rows", "Chunks", "SHA-256", "Acquired"])
        lay.addWidget(self.evidence_table)
        self.tabs.addTab(tab, "EVIDENCE")

    def _build_entities_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(8, 8, 8, 8)
        self.entity_table = self._make_table(
            ["ID", "Name", "Risk", "Events", "Sources", "Identifiers"])
        lay.addWidget(self.entity_table)
        self.tabs.addTab(tab, "ENTITIES")

    def _build_findings_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(8, 8, 8, 8)
        self.finding_table = self._make_table(
            ["Severity", "Kind", "Title", "Confidence", "Entities", "Evidence"])
        lay.addWidget(self.finding_table)
        self.tabs.addTab(tab, "FINDINGS")

    def _build_leads_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(8, 8, 8, 8)
        bar = QHBoxLayout()
        bar.addStretch(1)
        self.btn_leads = QPushButton("Refresh leads")
        self.btn_leads.setObjectName("GhostBtn")
        self.btn_leads.clicked.connect(self.recommend)
        bar.addWidget(self.btn_leads)
        lay.addLayout(bar)
        self.lead_table = self._make_table(
            ["Priority", "Category", "Action", "Rationale", "Links"])
        lay.addWidget(self.lead_table)
        self.tabs.addTab(tab, "LEADS")

    def _build_diary_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(8, 8, 8, 8)
        self.diary_view = QTextBrowser()
        self.diary_view.setObjectName("Inspector")
        lay.addWidget(self.diary_view)
        self.tabs.addTab(tab, "DIARY")

    def _build_report_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(8, 8, 8, 8)
        bar = QHBoxLayout()
        self.btn_gen_report = QPushButton("Generate report")
        self.btn_gen_report.setObjectName("PrimaryBtn")
        self.btn_gen_report.clicked.connect(self.generate_report)
        bar.addWidget(self.btn_gen_report)
        self.btn_open_report = QPushButton("Open HTML")
        self.btn_open_report.setObjectName("GhostBtn")
        self.btn_open_report.setEnabled(False)
        self.btn_open_report.clicked.connect(self.open_report)
        bar.addWidget(self.btn_open_report)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.report_files_label = QLabel("No report generated yet.")
        self.report_files_label.setObjectName("SectionSub")
        self.report_files_label.setWordWrap(True)
        lay.addWidget(self.report_files_label)
        self.summary_view = QTextBrowser()
        self.summary_view.setObjectName("Inspector")
        lay.addWidget(self.summary_view, 1)
        self.tabs.addTab(tab, "REPORT")

    # -- data loading ------------------------------------------------------
    def refresh_cases(self, select_id=None):
        current = select_id or self._case_id()
        cases = self.svc.list_cases()
        self.case_combo.blockSignals(True)
        self.case_combo.clear()
        for case in cases:
            self.case_combo.addItem(f"{case['title']} · {case['case_id']}",
                                    case["case_id"])
        self.case_combo.blockSignals(False)
        if not cases:
            self.case = None
            self._clear_views()
            self._set_status("No cases yet — click Load demo or New case.")
            return
        idx = self.case_combo.findData(current) if current else -1
        if idx < 0:
            idx = 0
        self.case_combo.setCurrentIndex(idx)
        self._case_changed(idx)

    def _case_changed(self, index):
        case_id = self.case_combo.itemData(index)
        if not case_id:
            return
        try:
            self.case = self.svc.load_case(case_id)
        except Exception as exc:
            self._error(str(exc))
            return
        self.case_title.setText(self.case.title)
        self.case_meta.setText(
            f"{self.case.case_id}\ncreated {self.case.created_at[:10]}")
        if self.case.language in SUPPORTED_LANGUAGES:
            self._set_lang(self.case.language)
        self._chat = []
        self._welcome()
        self.refresh_views()

    def _set_lang(self, code):
        idx = self.lang_combo.findData(code)
        if idx >= 0:
            self.lang_combo.blockSignals(True)
            self.lang_combo.setCurrentIndex(idx)
            self.lang_combo.blockSignals(False)

    def _language_changed(self, _index):
        self._refresh_suggestions()
        self.refresh_views()

    def refresh_views(self):
        if not self.case:
            return
        try:
            self.case = self.svc.load_case(self.case.case_id)
            index = self.svc.build_index(self.case)
        except Exception as exc:
            self._error(str(exc))
            return
        stats = index.stats()
        self.stat_values["evidence"].setText(str(len(self.case.evidence)))
        self.stat_values["entities"].setText(str(len(self.case.entities)))
        self.stat_values["findings"].setText(str(len(self.case.findings)))
        self.stat_values["leads"].setText(str(len(self.case.recommendations)))

        self.source_list.clear()
        for source, count in sorted((stats.get("sources") or {}).items()):
            self.source_list.addItem(f"{source}  ·  {count:,}")

        evidence = index.evidence_summary()
        self._fill_table(self.evidence_table, [
            [e["evidence_id"], e["filename"], e["source_type"], e["n_rows"],
             e["n_chunks"], (e["sha256"] or "")[:16] + "…",
             (e["acquired_at"] or "")[:19].replace("T", " ")]
            for e in evidence
        ])

        entities = sorted(self.case.entities.values(),
                          key=lambda e: (-e.risk_score, -e.n_events, e.display_name))
        self._fill_table(self.entity_table, [
            [e.entity_id, e.display_name, f"{e.risk_score:.2f}", e.n_events,
             ", ".join(e.sources), ", ".join(e.identifiers[:5])]
            for e in entities
        ])

        findings = sorted(self.case.findings,
                          key=lambda f: ({"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(f.severity, 3),
                                         -f.confidence))
        self._fill_table(self.finding_table, [
            [f.severity, f.kind, f.title, f"{f.confidence:.2f}",
             ", ".join(f.entity_ids), ", ".join(f.evidence_ids[:5])]
            for f in findings
        ])

        leads = sorted(self.case.recommendations,
                       key=lambda r: ({"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(r.priority, 3),
                                      r.recommendation_id))
        self._fill_table(self.lead_table, [
            [r.priority, r.category, r.action, r.rationale,
             ", ".join(r.finding_ids + r.entity_ids + r.evidence_ids)]
            for r in leads
        ])

        diary = self.svc.diary(self.case.case_id, lang=self._lang())
        chain = "verified" if diary["chain_ok"] else "TAMPERED"
        rows = []
        for entry in reversed(diary["entries"]):
            rows.append(
                f'<div class="diary"><b>{html.escape(entry["timestamp"][:19].replace("T", " "))}</b>'
                f' · <span class="kind">{html.escape(entry["entry_type"])}</span>'
                f' · {html.escape(entry["actor"])}<br>{html.escape(entry["text"])}</div>')
        self.diary_view.setHtml(
            f'<html><head><style>{self._chat_css()}</style></head><body>'
            f'<p><b>Hash chain: {chain}</b></p>{"".join(rows)}</body></html>')

        try:
            overview = self.svc.overview(self.case.case_id, lang=self._lang())
            summary = overview.get("summary", {})
            text = summary.get("text", "")
            gaps = summary.get("gaps", [])
            limitations = summary.get("limitations", [])
            if gaps:
                text += "\n\n**Evidence gaps**\n" + "\n".join(
                    f"- {g.get('key', '').replace('_', ' ')} ({g.get('severity', '')})"
                    for g in gaps)
            if limitations:
                text += "\n\n**Limitations**\n" + "\n".join(f"- {x}" for x in limitations)
            self.summary_view.setMarkdown(text)
        except Exception:
            self.summary_view.setPlainText("Summary unavailable.")

        llm = self.svc.llm
        if llm.available:
            model = getattr(llm.config, "text_model", "") or getattr(llm.config, "model", "")
            self.llm_label.setText(f"AI · {llm.config.provider} · {model}")
        else:
            self.llm_label.setText("OFFLINE · deterministic answers")
        self._refresh_suggestions()
        self._set_status("Ready.")

    def _fill_table(self, table, rows):
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                item = QTableWidgetItem("" if value is None else str(value))
                item.setToolTip("" if value is None else str(value))
                table.setItem(r, c, item)
        table.resizeColumnsToContents()

    def _clear_views(self):
        self.case_title.setText("No case selected")
        self.case_meta.setText("")
        for value in self.stat_values.values():
            value.setText("0")
        self.source_list.clear()
        for table in (self.evidence_table, self.entity_table,
                      self.finding_table, self.lead_table):
            table.setRowCount(0)
        self.chat_view.clear()
        self.citation_view.clear()
        self.diary_view.clear()
        self.summary_view.clear()
        self.confidence.setText("—")
        self.report_files_label.setText("No report generated yet.")
        self.btn_open_report.setEnabled(False)
        self._last_report_html = ""

    # -- chat --------------------------------------------------------------
    def _chat_css(self):
        return """
        body { background:#020617; color:#E2E8F0; font-family:'Segoe UI',Arial,sans-serif; font-size:13px; }
        .user { border-left:2px solid #FFB000; background:#111C31; padding:7px 10px; margin:8px 0; }
        .assistant { border-left:2px solid #38BDF8; background:#0F172A; padding:7px 10px; margin:8px 0; }
        .cites { border-top:1px solid #263449; margin-top:7px; padding-top:6px; color:#94A3B8; font-size:11px; }
        .diary { border-left:2px solid #FFB000; padding:5px 9px; margin:6px 0; background:#0F172A; }
        .kind { color:#38BDF8; font-family:monospace; font-size:10px; }
        .toolbar { margin:2px 0 7px; }
        .toolname { color:#38BDF8; font-family:monospace; font-size:10px; border:1px solid #38BDF8; padding:1px 5px; }
        .elapsed { color:#64748B; font-family:monospace; font-size:10px; margin-left:6px; }
        h1,h2,h3 { color:#FFB000; margin:10px 0 5px; }
        h1 { font-size:16px; } h2 { font-size:14px; } h3 { font-size:13px; }
        blockquote { border-left:2px solid #38BDF8; margin:8px 0; padding:4px 10px; color:#94A3B8; background:#0F172A; }
        pre { background:#111C31; border:1px solid #334155; padding:8px; margin:6px 0; }
        pre code { background:transparent; padding:0; }
        hr { border:none; border-top:1px solid #334155; margin:10px 0; }
        table { border-collapse:collapse; margin:6px 0; }
        th,td { border:1px solid #334155; padding:4px 6px; font-size:11px; }
        th { background:#1E293B; color:#FFB000; }
        code { color:#FDE047; background:#111C31; padding:1px 3px; }
        a { color:#FFB000; }
        img { border:1px solid #334155; margin-top:6px; }
        """

    def _md_to_html(self, text):
        try:
            doc = QTextDocument()
            doc.setMarkdown(text or "")
            raw = doc.toHtml()
            match = re.search(r"<body[^>]*>(.*)</body>", raw, re.S | re.I)
            return match.group(1) if match else html.escape(text or "").replace("\n", "<br>")
        except Exception:
            return html.escape(text or "").replace("\n", "<br>")

    def _welcome(self):
        self._chat = [{
            "role": "assistant",
            "text": t("chat.welcome", self._lang()),
            "images": [], "artifacts": [], "citations": [],
            "confidence": 0.9, "followups": [],
        }]
        self.render_chat()
        self._refresh_suggestions()

    def render_chat(self):
        blocks = []
        for msg in self._chat:
            if msg.get("role") == "user":
                text = html.escape(msg.get("text", "")).replace("\n", "<br>")
                attachments = "".join(
                    f'<br><span class="cites">📎 {html.escape(i.get("name", "image"))}</span>'
                    for i in msg.get("images", []))
                blocks.append(f'<div class="user"><b>YOU</b><br>{text}{attachments}</div>')
            else:
                body = self._md_to_html(msg.get("text", ""))
                images = "".join(
                    f'<br><a href="{self._url(i.get("path", ""))}">'
                    f'<img src="{self._url(i.get("path", ""))}" width="620"></a>'
                    f'<br><span class="cites">{html.escape(i.get("caption", ""))}</span>'
                    for i in msg.get("images", []) if i.get("path"))
                artifacts = "".join(
                    f'<br><a href="{self._url(a.get("path", ""))}">'
                    f'{html.escape(a.get("label", "file"))}</a>'
                    for a in msg.get("artifacts", []) if a.get("path"))
                citations = ""
                if msg.get("citations"):
                    lines = []
                    for i, c in enumerate(msg["citations"][:8], 1):
                        lines.append(
                            f'[C{i}] {html.escape(c.evidence_id)} · '
                            f'{html.escape(c.file)} · '
                            f'{html.escape((c.timestamp or "")[:19])} · '
                            f'{html.escape(c.snippet)}')
                    citations = '<div class="cites"><b>CITATIONS</b><br>' + "<br>".join(lines) + '</div>'
                tool = msg.get("tool")
                elapsed = msg.get("elapsed_ms")
                tool_badge = ""
                if tool:
                    elapsed_text = ""
                    if isinstance(elapsed, (int, float)):
                        elapsed_text = f'<span class="elapsed">{int(elapsed):,} ms</span>'
                    tool_badge = (
                        f'<div class="toolbar"><span class="toolname">'
                        f'{html.escape(str(tool))}</span>&nbsp;{elapsed_text}</div>')
                blocks.append(
                    f'<div class="assistant"><b>SUTRAM</b>{tool_badge}<br>{body}'
                    f'{images}{artifacts}{citations}</div>')
        self.chat_view.setHtml(
            f'<html><head><style>{self._chat_css()}</style></head>'
            f'<body>{"".join(blocks)}</body></html>')
        bar = self.chat_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _case_examples(self):
        """Build suggestions from the current case's highest-risk people."""
        if not self.case:
            return []
        people = sorted(
            self.case.entities.values(),
            key=lambda e: (-e.risk_score, -e.n_events, e.display_name),
        )
        names = [
            e.display_name for e in people
            if e.display_name and e.display_name != e.entity_id
        ][:2]
        if len(names) >= 2:
            return [
                f"Show transfers above 9,000 from {names[0]} in the last 90 days.",
                f"Who did {names[1]} call right before the transfer?",
                f"What is the connection between {names[0]} and {names[1]}?",
                "What looks suspicious in this case?",
            ]
        if names:
            return [
                f"Who is {names[0]}?",
                f"Show transfers involving {names[0]}.",
                "What looks suspicious in this case?",
                "Summarize the case.",
            ]
        return []

    def _refresh_suggestions(self):
        while self.suggestions.count():
            item = self.suggestions.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        examples = self._case_examples()
        if not examples:
            examples = t("query.examples", self._lang())
        if not isinstance(examples, list):
            return
        for question in examples[:4]:
            btn = QPushButton(question)
            btn.setObjectName("GhostBtn")
            btn.setStyleSheet("text-align:left; font-size:10px; padding:5px 8px;")
            btn.clicked.connect(lambda _=False, q=question: self.ask(q))
            self.suggestions.addWidget(btn)
        self.suggestions.addStretch(1)

    def attach_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Attach image", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)")
        if not files:
            return
        for path in files[:3 - len(self._attached)]:
            try:
                raw = Path(path).read_bytes()
            except OSError as exc:
                self._error(str(exc))
                continue
            mime = "image/jpeg" if raw[:2] == b"\xff\xd8" else "image/png"
            self._attached.append({
                "name": Path(path).name,
                "data_url": f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}",
            })
        self.attach_label.setText(
            ", ".join(x["name"] for x in self._attached) or "No image attached")

    def ask(self, question: str):
        """Send a suggested/follow-up question through the normal chat path."""
        if not self.case:
            self._toast("Load or create a case first.")
            return
        self.input.setPlainText(question or "")
        self.send()

    def send(self):
        if not self.case:
            self._toast("Load or create a case first.")
            return
        question = self.input.toPlainText().strip()
        images = list(self._attached)
        if not question and not images:
            return
        if not question:
            question = "Describe this image."
        self._chat.append({
            "role": "user", "text": question,
            "images": [{"name": x["name"]} for x in images],
        })
        self.render_chat()
        self.input.clear()
        self._attached = []
        self.attach_label.setText("No image attached")
        self.tabs.setCurrentIndex(0)
        payload = [x["data_url"] for x in images] or None
        self._run(
            lambda: self.svc.ask(self.case.case_id, question,
                                 lang=self._lang(), images=payload),
            self._after_answer,
            "Searching the evidence…",
        )

    def _after_answer(self, answer):
        self._chat.append({
            "role": "assistant",
            "text": answer.answer,
            "images": list(answer.images),
            "artifacts": list(answer.artifacts),
            "citations": list(answer.citations),
            "confidence": answer.confidence,
            "followups": list(answer.followups),
            "tool": (answer.data or {}).get("tool", ""),
            "elapsed_ms": (answer.data or {}).get("elapsed_ms"),
        })
        self.render_chat()
        self._show_inspector(answer)
        self.refresh_views()

    def _show_inspector(self, answer):
        cards = []
        for i, citation in enumerate(answer.citations[:10], 1):
            cards.append(
                f'<div style="border:1px solid #263449;background:#0F172A;'
                f'padding:6px;margin:5px 0;">'
                f'<b style="color:#FFB000;">C{i}</b> '
                f'<span style="color:#94A3B8;">{html.escape(citation.evidence_id)}</span><br>'
                f'<span style="color:#E2E8F0;">{html.escape(citation.file)}</span><br>'
                f'<span style="color:#64748B;">{html.escape((citation.timestamp or "")[:19])}</span><br>'
                f'<span style="color:#94A3B8;">{html.escape(citation.snippet)}</span></div>')
        self.citation_view.setHtml(
            f'<html><head><style>{self._chat_css()}</style></head>'
            f'<body>{"".join(cards) or "<p>No citations.</p>"}</body></html>')
        self.confidence.setText(f"{int(answer.confidence * 100)}%")
        while self.followup_box.count():
            item = self.followup_box.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        for question in answer.followups[:5]:
            btn = QPushButton(question)
            btn.setObjectName("GhostBtn")
            btn.setStyleSheet("text-align:left; font-size:10px; padding:5px 7px;")
            btn.clicked.connect(lambda _=False, q=question: self.ask(q))
            self.followup_box.addWidget(btn)
        self.followup_box.addStretch(1)

    # -- actions -----------------------------------------------------------
    def new_case(self):
        title, ok = QInputDialog.getText(self, "New case", "Case title:")
        if not ok or not title.strip():
            return
        case = self.svc.create_case(title.strip(), language=self._lang())
        self.refresh_cases(select_id=case.case_id)
        self._toast(f"Created case {case.case_id}")

    def load_demo(self):
        from aml.data_generator.investigation_case import build_demo_case
        self._run(
            lambda: build_demo_case(
                root=ROOT / "data" / "demo_case",
                case_root=self.svc.store.root,
                lang=self._lang(),
            ),
            self._after_demo,
            "Building the demo case (structured + emails + chats + PDF)…",
        )

    def ingest(self):
        if not self.case:
            self._toast("Create or load a case first.")
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add evidence", "",
            "Evidence (*.csv *.tsv *.json *.jsonl *.txt *.log *.eml *.pdf "
            "*.docx *.md *.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)")
        if not files:
            return
        self._run(
            lambda: self.svc.ingest(self.case.case_id, files),
            self._after_change,
            f"Ingesting {len(files)} evidence file(s)…",
        )

    def analyze(self):
        if not self.case:
            self._toast("Create or load a case first.")
            return
        self._run(
            lambda: self.svc.analyze(self.case.case_id),
            self._after_analyze,
            "Running the analytics ensemble…",
        )

    def recommend(self):
        if not self.case:
            return
        self._run(
            lambda: self.svc.recommend(self.case.case_id, lang=self._lang()),
            self._after_change,
            "Generating investigative leads…",
        )

    def generate_report(self):
        if not self.case:
            return
        self._run(
            lambda: self.svc.report(self.case.case_id, lang=self._lang()),
            self._after_report,
            "Generating the draft case report…",
        )

    def open_report(self):
        if self._last_report_html and Path(self._last_report_html).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._last_report_html))
        else:
            self._toast("Generate the report first.")

    def _after_demo(self, result):
        case_id = result.get("case_id") if isinstance(result, dict) else ""
        self.refresh_cases(select_id=case_id)
        self._toast("Demo case ready.")

    def _after_change(self, _result):
        self.refresh_views()
        self._toast("Case updated.")

    def _after_analyze(self, result):
        if isinstance(result, dict) and not result.get("ok", True):
            self._error(result.get("error", "Analysis failed."))
            return
        self.refresh_views()
        self.tabs.setCurrentIndex(3)  # findings
        self._toast("Analysis complete.")

    def _after_report(self, result):
        files = (result or {}).get("files", {}) or {}
        self._last_report_html = files.get("html", "")
        self.btn_open_report.setEnabled(bool(self._last_report_html))
        self.report_files_label.setText(
            "\n".join(f"{k}: {v}" for k, v in files.items()) or "No report files.")
        self._toast("Draft report generated.")

    # -- worker plumbing ---------------------------------------------------
    def _run(self, fn, callback, status):
        if self._worker is not None:
            self._toast("A case operation is already running.")
            return
        self._pending_cb = callback
        self._set_busy(True, status)
        self._worker = CaseWorker(fn, self)
        self._worker.done.connect(self._worker_done)
        self._worker.failed.connect(self._worker_failed)
        self._worker.finished.connect(self._worker_finished)
        self._worker.start()

    def _worker_done(self, result):
        callback = self._pending_cb
        if callback:
            try:
                callback(result)
            except Exception as exc:
                self._error(str(exc))

    def _worker_failed(self, message):
        self._error(message)

    def _worker_finished(self):
        self._worker = None
        self._pending_cb = None
        self._set_busy(False, "Ready.")

    def _set_busy(self, busy, status):
        for button in (self.btn_new, self.btn_demo, self.btn_ingest,
                       self.btn_analyze, self.btn_refresh, self.btn_leads,
                       self.btn_gen_report, self.btn_attach, self.btn_send):
            if button is not None:
                button.setEnabled(not busy)
        self.status.setText(status)
        if busy:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()

    def _set_status(self, text):
        self.status.setText(text)

    def _error(self, message):
        self._set_busy(False, "Error.")
        QMessageBox.critical(self, "Sutram Investigator", message)

    def _toast(self, message):
        self.toast_requested.emit(message)

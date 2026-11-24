"""Zero-dependency web workspace for Sutram Investigator.

    python web/server.py --case-root data/cases --port 8000

Serves the chat UI and a small JSON API over the investigation service.  No
FastAPI/Flask dependency: ``http.server`` is enough for a local hackathon demo
and keeps the offline-first promise.

API:
    GET  /api/cases
    GET  /api/languages
    GET  /api/i18n?lang=hi
    GET  /api/case_bundle?case_id=...&lang=...
    GET  /api/summary?case_id=...&lang=...
    GET  /api/evidence?case_id=...
    GET  /api/entities?case_id=...
    GET  /api/findings?case_id=...
    GET  /api/recommendations?case_id=...&lang=...
    GET  /api/diary?case_id=...&lang=...
    POST /api/ask            {case_id, question, lang, session_id}
    POST /api/recommend      {case_id, lang}
    POST /api/report         {case_id, lang}
    POST /api/analyze        {case_id}
    POST /api/note           {case_id, text, actor}
    GET  /files/<case_id>/<reports-relative-path>
"""
from __future__ import annotations

import json
import mimetypes
import os
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("AML_SKIP_TORCH", "1")

from aml.service import InvestigationService  # noqa: E402
from aml.config import load_config  # noqa: E402
from aml.diary import verify_chain  # noqa: E402
from aml.i18n import (CATALOGS, SUPPORTED_LANGUAGES, language_name,  # noqa: E402
                                normalize_language)
from aml.recommend import generate_recommendations  # noqa: E402
from aml.summarize import case_summary  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
_LOCK = threading.Lock()


class AppState:
    def __init__(self, case_root: str, config_path: str, default_lang: str = "en"):
        self.case_root = case_root
        self.config_path = config_path
        self.default_lang = default_lang
        self.service = InvestigationService(
            case_root=case_root,
            config=load_config(config_path, case_root=case_root, lang=default_lang),
            lang=default_lang,
        )


STATE: Optional[AppState] = None


def _json_default(value):
    return str(value)


def _safe_case_path(case, relative: str) -> Optional[Path]:
    base = Path(case.root).resolve()
    target = (base / relative).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        return None
    if not target.is_file():
        return None
    # only expose report artefacts, never the evidence originals
    try:
        target.relative_to((base / "reports").resolve())
    except ValueError:
        return None
    return target


def _bundle(svc: InvestigationService, case_id: str, lang: str) -> dict:
    case = svc.load_case(case_id)
    index = svc.build_index(case)
    ok, errors = verify_chain(case)
    # Localize the auto-generated leads for the selected UI language.  The
    # recommendation IDs are stable within a language; switching language
    # regenerates the action/rationale text, not the underlying evidence.
    target_lang = normalize_language(lang or case.language)
    auto_langs = {
        (r.metadata or {}).get("lang")
        for r in case.recommendations
        if (r.metadata or {}).get("auto")
    }
    if case.recommendations and target_lang not in auto_langs:
        generate_recommendations(case, index, lang=target_lang,
                                 min_events=svc.config.min_events_per_entity)
        svc.store.save_case(case)
    summary = case_summary(case, index, lang=target_lang,
                           min_events=svc.config.min_events_per_entity)
    entities = sorted(
        case.entities.values(),
        key=lambda e: (-e.risk_score, -e.n_events, e.display_name),
    )
    findings = sorted(
        case.findings,
        key=lambda f: ({"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(f.severity, 3), -f.confidence),
    )
    recommendations = sorted(
        case.recommendations,
        key=lambda r: ({"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(r.priority, 3), r.recommendation_id),
    )
    report_files: Dict[str, str] = {}
    reports_dir = Path(case.root) / "reports"
    for key, filename in (
        ("md", "case_report.md"),
        ("html", "case_report.html"),
        ("pdf", "case_report.pdf"),
        ("json", "case_report.json"),
        ("diary", "case_diary.md"),
    ):
        if (reports_dir / filename).exists():
            report_files[key] = f"/files/{case.case_id}/reports/{filename}"
    for key, filename in (("analytics_html", "analytics/report.html"),
                          ("analytics_pdf", "analytics/report.pdf")):
        if (reports_dir / filename).exists():
            report_files[key] = f"/files/{case.case_id}/reports/{filename}"
    return {
        "case": case.to_dict(),
        "stats": index.stats(),
        "summary": summary,
        "evidence": index.evidence_summary(),
        "entities": [e.to_dict() for e in entities],
        "findings": [f.to_dict() for f in findings],
        "recommendations": [r.to_dict() for r in recommendations],
        "diary": [d.to_dict() for d in case.diary],
        "chain_ok": ok,
        "chain_errors": errors,
        "report_files": report_files,
        "languages": [{"code": c, "name": language_name(c)} for c in SUPPORTED_LANGUAGES],
        "llm": svc.llm.config.to_dict(),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "SutramInvestigator/0.1"

    # -- helpers -----------------------------------------------------------
    def _send(self, status: int, body: bytes, content_type: str = "application/json; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, value: Any, status: int = 200):
        body = json.dumps(value, ensure_ascii=False, default=_json_default).encode("utf-8")
        self._send(status, body)

    def _error(self, message: str, status: int = 500):
        self._json({"error": message}, status=status)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def log_message(self, fmt, *args):
        # keep the console quiet; errors still surface through responses
        return

    # -- routing -----------------------------------------------------------
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        lang = (query.get("lang", [STATE.default_lang])[0] if STATE else "en")
        case_id = query.get("case_id", [""])[0]

        try:
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path in ("/app.js", "/style.css"):
                return self._static(path.lstrip("/"))
            if path == "/api/health":
                return self._json({"ok": True, "cases": len(STATE.service.list_cases())})
            if path == "/api/languages":
                return self._json([{"code": c, "name": language_name(c)}
                                   for c in SUPPORTED_LANGUAGES])
            if path == "/api/cases":
                return self._json(STATE.service.list_cases())
            if path == "/api/i18n":
                catalog = dict(CATALOGS.get("en", {}))
                catalog.update(CATALOGS.get(lang, {}))
                return self._json(catalog)
            if path == "/api/case_bundle":
                if not case_id:
                    cases = STATE.service.list_cases()
                    if not cases:
                        return self._json({"error": "no cases"}, status=404)
                    case_id = cases[0]["case_id"]
                with _LOCK:
                    return self._json(_bundle(STATE.service, case_id, lang))
            if path == "/api/summary":
                with _LOCK:
                    return self._json(STATE.service.summarize(case_id, lang=lang))
            if path == "/api/evidence":
                case = STATE.service.load_case(case_id)
                return self._json(STATE.service.build_index(case).evidence_summary())
            if path == "/api/entities":
                case = STATE.service.load_case(case_id)
                return self._json([e.to_dict() for e in sorted(
                    case.entities.values(), key=lambda e: (-e.risk_score, -e.n_events))])
            if path == "/api/findings":
                return self._json(STATE.service.findings(case_id))
            if path == "/api/recommendations":
                with _LOCK:
                    return self._json(STATE.service.recommend(case_id, lang=lang))
            if path == "/api/diary":
                with _LOCK:
                    return self._json(STATE.service.diary(case_id, lang=lang))
            if path.startswith("/files/"):
                return self._serve_file(path)
            return self._error("not found", status=404)
        except FileNotFoundError as exc:
            return self._error(str(exc), status=404)
        except Exception as exc:
            return self._error(f"{type(exc).__name__}: {exc}", status=500)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        payload = self._read_json()
        try:
            if path == "/api/ask":
                case_id = payload.get("case_id", "")
                question = payload.get("question", "")
                lang = payload.get("lang") or STATE.default_lang
                session_id = payload.get("session_id") or "web"
                if not case_id or not question:
                    return self._error("case_id and question are required", status=400)
                with _LOCK:
                    answer = STATE.service.ask(
                        case_id, question, lang=lang, session_id=session_id,
                        images=payload.get("images") or None,
                    )
                return self._json(answer.to_dict())
            if path == "/api/recommend":
                with _LOCK:
                    return self._json(STATE.service.recommend(
                        payload.get("case_id", ""), lang=payload.get("lang")))
            if path == "/api/report":
                with _LOCK:
                    return self._json(STATE.service.report(
                        payload.get("case_id", ""), lang=payload.get("lang")))
            if path == "/api/analyze":
                with _LOCK:
                    return self._json(STATE.service.analyze(payload.get("case_id", "")))
            if path == "/api/note":
                with _LOCK:
                    return self._json(STATE.service.add_note(
                        payload.get("case_id", ""), payload.get("text", ""),
                        actor=payload.get("actor", "analyst")))
            return self._error("not found", status=404)
        except FileNotFoundError as exc:
            return self._error(str(exc), status=404)
        except Exception as exc:
            return self._error(f"{type(exc).__name__}: {exc}", status=500)

    # -- static / files ----------------------------------------------------
    def _static(self, name: str):
        path = (STATIC_DIR / name).resolve()
        try:
            path.relative_to(STATIC_DIR.resolve())
        except ValueError:
            return self._error("not found", status=404)
        if not path.is_file():
            return self._error("not found", status=404)
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in ("application/javascript",):
            content_type += "; charset=utf-8"
        self._send(200, path.read_bytes(), content_type)

    def _serve_file(self, path: str):
        # /files/<case_id>/<reports-relative-path>
        parts = path.split("/", 3)
        if len(parts) < 4:
            return self._error("not found", status=404)
        case_id, relative = parts[2], parts[3]
        case = STATE.service.load_case(case_id)
        target = _safe_case_path(case, relative)
        if target is None:
            return self._error("not found", status=404)
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self._send(200, target.read_bytes(), content_type)


def main() -> int:
    import argparse
    import webbrowser
    global STATE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case-root", default="data/cases")
    ap.add_argument("--config", default=str(ROOT / "config" / "investigation.yaml"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--open", action="store_true", help="open the browser automatically")
    args = ap.parse_args()

    STATE = AppState(args.case_root, args.config, args.lang)
    url = f"http://{args.host}:{args.port}/"
    print(f"Sutram Investigator web workspace: {url}")
    print(f"Case root: {args.case_root}")
    print("Press Ctrl+C to stop.")
    if args.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

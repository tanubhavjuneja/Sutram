"""Multilingual catalogs and language detection.

Six locales are supported out of the box: English, Hindi, Spanish, French,
German and Portuguese.  Catalogs are partial: any missing key falls back to
English, which keeps translation work incremental without breaking the UI.

The module also exposes the per-language lexicons used by the query parser
(intent words, stopwords, month names, relative-time words).  Keeping them in
one place makes it easy to add a language later.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional

SUPPORTED_LANGUAGES: List[str] = ["en", "hi", "es", "fr", "de", "pt"]

LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "hi": "हिन्दी",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "pt": "Português",
}


# ---------------------------------------------------------------------------
# English master catalog
# ---------------------------------------------------------------------------
_EN: Dict[str, Any] = {
    # app / navigation
    "app.title": "Sutram Investigator",
    "app.tagline": "AI-Powered Digital Investigation Assistant",
    "app.case": "Case",
    "nav.overview": "Overview",
    "nav.assistant": "Assistant",
    "nav.evidence": "Evidence",
    "nav.entities": "Entities",
    "nav.findings": "Findings",
    "nav.diary": "Case diary",
    "nav.recommendations": "Leads",
    "nav.report": "Report",
    "common.loading": "Working…",
    "common.refresh": "Refresh",
    "common.error": "Something went wrong.",
    "common.close": "Close",
    "common.export": "Export",
    "common.download": "Download",
    "common.language": "Language",
    "common.search": "Search",
    "common.send": "Send",
    "common.clear": "Clear",
    "common.copy": "Copy",
    "common.yes": "Yes",
    "common.no": "No",
    "common.none": "None",
    "common.unknown": "Unknown",
    "common.confidence": "Confidence",
    "common.severity": "Severity",
    "common.priority": "Priority",
    "common.status": "Status",
    "common.source": "Source",
    "common.sources": "Sources",
    "common.timestamp": "Timestamp",
    "common.file": "File",
    "common.entity": "Entity",
    "common.evidence": "Evidence",
    "common.citations": "Citations",
    "common.actions": "Actions",
    "common.details": "Details",
    "common.summary": "Summary",
    "common.risk": "Risk",
    "common.total": "Total",
    "common.count": "Count",
    "common.average": "Average",
    "common.largest": "Largest",
    "common.open": "Open",
    "common.done": "Done",
    "common.accept": "Accept",
    "common.dismiss": "Dismiss",
    "lang.en": "English",
    "lang.hi": "Hindi",
    "lang.es": "Spanish",
    "lang.fr": "French",
    "lang.de": "German",
    "lang.pt": "Portuguese",

    # intents
    "intent.search": "Evidence search",
    "intent.entity_profile": "Entity profile",
    "intent.timeline": "Timeline",
    "intent.relationship": "Relationship",
    "intent.correlation": "Call/transfer correlation",
    "intent.aggregate": "Aggregate",
    "intent.anomaly": "Suspicious patterns",
    "intent.summarize": "Case summary",
    "intent.recommend": "Investigative leads",
    "intent.diary": "Case diary",
    "intent.report": "Draft report",
    "intent.evidence_inventory": "Evidence inventory",
    "intent.entity_list": "Entity list",
    "intent.help": "Help",

    # chat
    "chat.placeholder": "Ask about the evidence, an entity, a timeline, or what to do next…",
    "chat.welcome": "I am your forensic case assistant. I answer only from the indexed evidence and cite every factual claim.",
    "chat.grounded": "Answers are grounded in the indexed evidence; every factual claim is cited.",
    "chat.suggestions": "Suggested questions",
    "chat.thinking": "Searching the evidence…",
    "chat.followups": "You could ask next",
    "chat.no_answer": "I could not ground an answer in the indexed evidence.",
    "chat.session": "Session",

    # evidence / entities / findings / diary / recommendations
    "evidence.title": "Evidence inventory",
    "evidence.count": "{n} item(s)",
    "evidence.hash": "SHA-256",
    "evidence.custodian": "Custodian",
    "evidence.acquired": "Acquired",
    "evidence.rows": "Rows",
    "evidence.chunks": "Chunks",
    "evidence.type": "Type",
    "evidence.integrity": "Integrity",
    "evidence.integrity_ok": "Hash recorded",
    "entities.title": "Entities",
    "entities.risk": "Risk",
    "entities.sources": "Sources",
    "entities.first_seen": "First seen",
    "entities.last_seen": "Last seen",
    "entities.events": "Events",
    "entities.identifiers": "Identifiers",
    "entities.names": "Names",
    "findings.title": "Suspicious-pattern findings",
    "findings.none": "No findings yet. Run the analytics step.",
    "diary.title": "Case diary",
    "diary.verified": "Hash chain verified",
    "diary.tampered": "Hash chain TAMPERED",
    "diary.entry": "Entry",
    "diary.actor": "Actor",
    "recommendations.title": "Investigative leads",
    "recommendations.none": "No recommendations yet.",
    "recommendations.accept": "Accept",
    "recommendations.dismiss": "Dismiss",
    "report.title": "Draft case report",
    "report.generate": "Generate report",
    "report.generated": "Report generated",
    "report.hint": "Generate a draft case report from the indexed evidence, findings, recommendations and diary.",
    "report.open_html": "Open HTML",
    "report.download_md": "Download MD",
    "report.download_pdf": "Download PDF",
    "report.download_json": "Download JSON",
    "summary.title": "Case summary",
    "stats.evidence": "Evidence",
    "stats.entities": "Entities",
    "stats.events": "Events",
    "stats.findings": "Findings",

    # answer templates
    "answer.search.header": "I found **{n}** matching item(s). The strongest matches are:",
    "answer.search.none": "No evidence matched **{query}**. Try broadening the date range, removing a filter, or searching for a person, account, phone or handle instead.",
    "answer.search.footer": "Showing the top {n} of {total} matches.",
    "answer.entity.header": "**{name}** ({entity_id}) — {risk} risk",
    "answer.entity.profile": "{name} appears in **{n_events}** event(s) across {sources}. First seen {first_seen}; last seen {last_seen}.",
    "answer.entity.identifiers": "Known identifiers: {identifiers}.",
    "answer.entity.risk_factors": "Risk factors: {factors}.",
    "answer.entity.not_found": "I could not find an entity matching **{term}**. Known entities: {names}.",
    "answer.timeline.header": "Chronology — **{n}** item(s) in the selected window:",
    "answer.timeline.none": "No dated evidence was found for that window.",
    "answer.relationship.header": "Relationship between **{a}** and **{b}**:",
    "answer.relationship.direct": "Direct contact: **{n}** event(s) between them.",
    "answer.relationship.path": "Indirect path: {path}.",
    "answer.relationship.common": "Common counterparts: {names}.",
    "answer.relationship.shared": "No direct structured transaction/call edge was found, but the entities co-occur in **{n}** document(s):",
    "answer.relationship.none": "No direct or one-hop connection was found in the indexed evidence.",
    "answer.correlation.header": "I found **{n}** communication(s) within {tol} minutes of a transfer:",
    "answer.correlation.none": "No call/post was found within {tol} minutes of a transfer for this entity.",
    "answer.aggregate.header": "Aggregate over **{n}** matching event(s):",
    "answer.aggregate.total": "Total amount: **{total}**.",
    "answer.aggregate.count": "Count: **{count}**.",
    "answer.aggregate.avg": "Average: **{avg}**.",
    "answer.aggregate.largest": "Largest: **{largest}**.",
    "answer.anomaly.header": "I found **{n}** suspicious-pattern finding(s). Highest priority first:",
    "answer.anomaly.none": "No suspicious-pattern findings yet. Run the analytics step first.",
    "answer.summarize.header": "Case summary — **{title}**",
    "answer.recommend.header": "Recommended next steps, highest priority first:",
    "answer.recommend.none": "No recommendations yet.",
    "answer.diary.header": "Case diary — **{n}** entries. Chain status: {status}.",
    "answer.report.header": "Draft case report generated: {path}",
    "answer.evidence_inventory.header": "Evidence inventory — **{n}** item(s):",
    "answer.entity_list.header": "Entities — **{n}** resolved:",
    "answer.help": "I can search evidence, profile entities, build timelines, explain relationships, correlate calls with transfers, aggregate amounts, summarize the case, recommend next steps, show the diary, and draft a report. Ask in English, Hindi, Spanish, French, German or Portuguese.",
    "answer.citation_line": "[{label}] {file} · {timestamp} · {snippet}",
    "answer.confidence": "Confidence: {value}",
    "answer.followups": "You could ask:",

    # summaries
    "summary.overview": "**{title}** covers **{n_evidence}** evidence item(s) across {sources}, spanning {start} to {end}. The case has **{n_entities}** resolved entities and **{n_findings}** finding(s).",
    "summary.no_dates": "The case has **{n_evidence}** evidence item(s) across {sources}; no dated events were found.",
    "summary.top_entities": "Highest-risk entities: {names}.",
    "summary.findings": "Suspicious patterns: {kinds}.",
    "summary.gaps": "Evidence gaps: {gaps}.",
    "summary.next_steps": "Top recommendation: {action}",
    "summary.entity": "**{name}** ({entity_id}) — {risk} risk. **{n_events}** event(s) across {sources}. First seen {first_seen}; last seen {last_seen}. Identifiers: {identifiers}.",
    "summary.entity.findings": "Linked findings: {titles}.",
    "summary.entity.counterparts": "Top counterparts: {names}.",
    "summary.timeline": "Timeline: **{n}** dated item(s) from {start} to {end}.",
    "summary.limitations": "Limitations: {items}",
    "summary.none": "No evidence has been indexed yet.",

    # recommendations
    "rec.gap.bank.action": "Obtain complete bank/transaction records for the case period.",
    "rec.gap.bank.rationale": "No bank evidence is indexed, so financial flows, structuring and layering cannot be verified.",
    "rec.gap.cdr.action": "Obtain call detail records (CDR) for the relevant phone numbers.",
    "rec.gap.cdr.rationale": "No CDR evidence is indexed, so call-transfer colocation and communication networks cannot be tested.",
    "rec.gap.social.action": "Collect social-media posts and account identifiers for the relevant handles.",
    "rec.gap.social.rationale": "No social evidence is indexed; open-source context and weak-tie links are missing.",
    "rec.gap.email.action": "Collect email and chat exports for the relevant accounts.",
    "rec.gap.email.rationale": "No email/chat evidence is indexed; intent, coordination and instruction trails may be missing.",
    "rec.gap.timestamps.action": "Obtain dated records for the undated evidence.",
    "rec.gap.timestamps.rationale": "Some evidence has no timestamps, so it cannot be placed on the case timeline.",
    "rec.gap.volume.action": "Collect a longer period or more records for the thin sources.",
    "rec.gap.volume.rationale": "Per-entity volume is below the reliability threshold; statistical findings are weak.",
    "rec.gap.unresolved.action": "Resolve unmatched identifiers (accounts, phones, handles) to people.",
    "rec.gap.unresolved.rationale": "Unresolved identifiers prevent cross-source linking and hide relationships.",
    "rec.interview.action": "Interview {name} ({entity_id}) and record a statement.",
    "rec.interview.rationale": "This entity is ranked {risk} risk with {n_findings} linked finding(s).",
    "rec.records.action": "Request full account/phone records for {identifiers}.",
    "rec.records.rationale": "Complete records are needed to verify the flagged activity for {name}.",
    "rec.structuring.action": "Review all transactions in the 88–100% band below the {threshold} reporting limit for {name}.",
    "rec.structuring.rationale": "The structuring pattern suggests deliberate avoidance of reporting thresholds.",
    "rec.colocation.action": "Obtain tower dump / CCTV / access logs for {location} around {timestamp}.",
    "rec.colocation.rationale": "A call/post was followed by a transfer to the same party within the tolerance window.",
    "rec.chain.action": "Map the multi-hop flow and request KYC for the beneficiary accounts in the chain.",
    "rec.chain.rationale": "Rapid layering through intermediate accounts is a classic placement/layering indicator.",
    "rec.fan.action": "Screen the {n} counterparties of {name} for mule-account behaviour.",
    "rec.fan.rationale": "Fan-in/fan-out structure is consistent with collection and dispersal.",
    "rec.dormancy.action": "Check for account takeover, life events or incarceration around the reactivation date for {name}.",
    "rec.dormancy.rationale": "A long dormant account suddenly re-activated and moved funds.",
    "rec.corroborate.action": "Corroborate the {kind} finding for {name} with an independent source.",
    "rec.corroborate.rationale": "The finding is based on a single signal; corroboration strengthens the case.",
    "rec.integrity.action": "Verify the SHA-256 hash and chain-of-custody record for the flagged evidence.",
    "rec.integrity.rationale": "Evidence integrity must be documented before it is relied on in a report.",
    "rec.timeline_gap.action": "Request records covering the timeline gap from {start} to {end}.",
    "rec.timeline_gap.rationale": "There is a gap in the dated evidence that could hide relevant activity.",
    "rec.report.action": "Generate and review the draft case report.",
    "rec.report.rationale": "A structured report makes the findings, evidence links and limitations reviewable.",

    # diary
    "diary.ingest": "Ingested {n} evidence item(s): {files}.",
    "diary.analysis": "Analytics completed: {models} model(s), {findings} finding(s), top entity {top}.",
    "diary.query": "Query ({language}, {intent}): {query} — {n} result(s).",
    "diary.finding": "Finding {id}: {title} ({severity}).",
    "diary.recommendation": "Recommendation {id}: {action} ({priority}).",
    "diary.report": "Case report generated: {path}.",
    "diary.note": "{text}",

    # report
    "report.case_report": "Draft Case Report",
    "report.case_id": "Case ID",
    "report.generated": "Generated",
    "report.classification": "Classification",
    "report.status": "Status",
    "report.language": "Language",
    "report.executive_summary": "Executive summary",
    "report.evidence_inventory": "Evidence inventory",
    "report.entities": "Entities",
    "report.findings": "Suspicious-pattern findings",
    "report.timeline": "Timeline",
    "report.recommendations": "Investigative recommendations",
    "report.case_diary": "Case diary",
    "report.methodology": "Methodology and limitations",
    "report.limitations": "Limitations",
    "limitations.ocr": "Text evidence is quoted as-is; OCR was not applied to scanned images.",
    "limitations.review": "Answers are generated from the indexed evidence and must be reviewed by a qualified investigator.",
    "limitations.no_analytics": "Suspicious-pattern analytics have not been run yet.",
    "limitations.no_structured": "No structured events were indexed; timeline and financial analysis are limited.",
    "limitations.none": "No material limitations were identified from the indexed evidence.",
    "limitations.blocked": "{model}: BLOCKED — {reason}",
    "limitations.degraded": "{model}: DEGRADED — {reason}",
    "report.analytics_appendix": "Analytics appendix",
    "report.query_log": "Query log",
    "report.no_findings": "No findings recorded.",
    "report.no_recommendations": "No recommendations recorded.",
    "report.hash": "SHA-256",
    "report.size": "Size",
    "report.rows": "Rows",
    "report.chunks": "Chunks",
    "report.custodian": "Custodian",
    "report.acquired": "Acquired",
    "report.prepared_by": "Prepared by",
    "report.disclaimer": "This is a draft generated from indexed evidence. Every finding must be verified by a qualified investigator before use in legal proceedings.",
    "report.diary_chain_ok": "Diary hash chain verified.",
    "report.diary_chain_bad": "Diary hash chain verification FAILED — investigate tampering.",
    "report.evidence_gap": "Evidence gaps",
    "report.next_steps": "Next steps",
    "report.generated_files": "Generated files",
    "report.none": "None recorded.",

    # query examples
    "query.examples": [
        "Show me transfers above 10,000 from Viktor in the last 30 days.",
        "Who did Viktor call right before the transfer?",
        "What is the connection between Viktor and Elena?",
        "What looks suspicious in this case?",
        "Summarize the case.",
        "What should I investigate next?",
        "Show the case diary.",
        "Draft a case report.",
    ],
}


# ---------------------------------------------------------------------------
# Partial translations
# ---------------------------------------------------------------------------
_HI: Dict[str, Any] = {
    "app.tagline": "एआई-संचालित डिजिटल जांच सहायक",
    "nav.overview": "अवलोकन",
    "nav.assistant": "सहायक",
    "nav.evidence": "साक्ष्य",
    "nav.entities": "संस्थाएँ",
    "nav.findings": "निष्कर्ष",
    "nav.diary": "केस डायरी",
    "nav.recommendations": "जांच सुराग",
    "nav.report": "रिपोर्ट",
    "common.loading": "काम जारी है…",
    "common.search": "खोजें",
    "common.send": "भेजें",
    "common.clear": "साफ़ करें",
    "common.language": "भाषा",
    "common.confidence": "विश्वास",
    "common.severity": "गंभीरता",
    "common.priority": "प्राथमिकता",
    "common.source": "स्रोत",
    "common.timestamp": "समय",
    "common.file": "फ़ाइल",
    "common.entity": "संस्था",
    "common.evidence": "साक्ष्य",
    "common.citations": "उद्धरण",
    "common.risk": "जोखिम",
    "common.total": "कुल",
    "common.count": "संख्या",
    "common.average": "औसत",
    "common.largest": "सबसे बड़ा",
    "chat.placeholder": "साक्ष्य, किसी व्यक्ति, समयरेखा या अगले कदम के बारे में पूछें…",
    "chat.welcome": "मैं आपका फोरेंसिक केस सहायक हूँ। मैं केवल अनुक्रमित साक्ष्य से उत्तर देता हूँ और हर तथ्य का उद्धरण देता हूँ।",
    "chat.suggestions": "सुझाए गए प्रश्न",
    "chat.followups": "आगे आप पूछ सकते हैं",
    "answer.search.header": "मुझे **{n}** मिलते-जुलते परिणाम मिले। सबसे मजबूत परिणाम:",
    "answer.search.none": "**{query}** से मेल खाता कोई साक्ष्य नहीं मिला। तिथि सीमा बढ़ाएँ, फ़िल्टर हटाएँ, या किसी व्यक्ति/खाते/फ़ोन से खोजें।",
    "answer.entity.header": "**{name}** ({entity_id}) — {risk} जोखिम",
    "answer.entity.profile": "{name} **{n_events}** घटनाओं में {sources} पर दिखता है। पहली बार {first_seen}; अंतिम बार {last_seen}।",
    "answer.entity.not_found": "**{term}** से मेल खाती कोई संस्था नहीं मिली। ज्ञात संस्थाएँ: {names}।",
    "answer.timeline.header": "समयरेखा — चुनी गई अवधि में **{n}** परिणाम:",
    "answer.timeline.none": "उस अवधि के लिए कोई तिथि-युक्त साक्ष्य नहीं मिला।",
    "answer.relationship.header": "**{a}** और **{b}** के बीच संबंध:",
    "answer.relationship.none": "अनुक्रमित साक्ष्य में कोई सीधा या एक-कदम संबंध नहीं मिला।",
    "answer.correlation.header": "मुझे **{n}** संचार मिले जो किसी ट्रांसफर के {tol} मिनट के भीतर थे:",
    "answer.correlation.none": "इस संस्था के किसी ट्रांसफर के {tol} मिनट के भीतर कोई कॉल/पोस्ट नहीं मिला।",
    "answer.aggregate.header": "**{n}** मिलती-जुलती घटनाओं पर कुल:",
    "answer.aggregate.total": "कुल राशि: **{total}**।",
    "answer.anomaly.header": "मुझे **{n}** संदिग्ध पैटर्न मिले। सबसे उच्च प्राथमिकता पहले:",
    "answer.anomaly.none": "अभी कोई संदिग्ध-पैटर्न निष्कर्ष नहीं है। पहले विश्लेषण चलाएँ।",
    "answer.summarize.header": "केस सारांश — **{title}**",
    "answer.recommend.header": "अनुशंसित अगले कदम, उच्चतम प्राथमिकता पहले:",
    "answer.diary.header": "केस डायरी — **{n}** प्रविष्टियाँ। श्रृंखला स्थिति: {status}।",
    "answer.report.header": "ड्राफ्ट केस रिपोर्ट तैयार: {path}",
    "answer.evidence_inventory.header": "साक्ष्य सूची — **{n}** आइटम:",
    "answer.entity_list.header": "संस्थाएँ — **{n}** हल की गईं:",
    "answer.help": "मैं साक्ष्य खोज सकता हूँ, व्यक्ति की प्रोफ़ाइल बना सकता हूँ, समयरेखा बना सकता हूँ, संबंध समझा सकता हूँ, कॉल और ट्रांसफर को जोड़ सकता हूँ, राशि जोड़ सकता हूँ, केस सारांश दे सकता हूँ, अगले कदम सुझा सकता हूँ, डायरी दिखा सकता हूँ और रिपोर्ट तैयार कर सकता हूँ। अंग्रेज़ी, हिन्दी, स्पेनिश, फ़्रेंच, जर्मन या पुर्तगाली में पूछें।",
    "summary.overview": "**{title}** में {sources} पर **{n_evidence}** साक्ष्य आइटम हैं, जो {start} से {end} तक हैं। केस में **{n_entities}** संस्थाएँ और **{n_findings}** निष्कर्ष हैं।",
    "summary.entity": "**{name}** ({entity_id}) — {risk} जोखिम। {sources} पर **{n_events}** घटनाएँ। पहली बार {first_seen}; अंतिम बार {last_seen}। पहचानकर्ता: {identifiers}।",
    "summary.gaps": "साक्ष्य की कमियाँ: {gaps}।",
    "summary.next_steps": "शीर्ष सिफ़ारिश: {action}",
    "report.case_report": "ड्राफ्ट केस रिपोर्ट",
    "report.executive_summary": "कार्यकारी सारांश",
    "report.evidence_inventory": "साक्ष्य सूची",
    "report.entities": "संस्थाएँ",
    "report.findings": "संदिग्ध-पैटर्न निष्कर्ष",
    "report.timeline": "समयरेखा",
    "report.recommendations": "जांच संबंधी सिफ़ारिशें",
    "report.case_diary": "केस डायरी",
    "report.methodology": "कार्यप्रणाली और सीमाएँ",
    "report.analytics_appendix": "विश्लेषण परिशिष्ट",
    "report.query_log": "प्रश्न लॉग",
    "report.disclaimer": "यह अनुक्रमित साक्ष्य से तैयार एक ड्राफ्ट है। कानूनी कार्यवाही में उपयोग से पहले प्रत्येक निष्कर्ष को योग्य अन्वेषक द्वारा सत्यापित करना आवश्यक है।",
    "report.diary_chain_ok": "डायरी हैश श्रृंखला सत्यापित।",
    "report.diary_chain_bad": "डायरी हैश श्रृंखला सत्यापन विफल — छेड़छाड़ की जाँच करें।",
    "query.examples": [
        "पिछले 30 दिनों में विक्टर से 10,000 से अधिक के ट्रांसफर दिखाएँ।",
        "ट्रांसफर से ठीक पहले विक्टर ने किसे कॉल किया?",
        "विक्टर और एलेना के बीच क्या संबंध है?",
        "इस मामले में संदिग्ध क्या है?",
        "केस का सारांश दें।",
        "मुझे आगे क्या जांच करनी चाहिए?",
        "केस डायरी दिखाएँ।",
        "केस रिपोर्ट का ड्राफ्ट बनाएँ।",
    ],
}

_ES: Dict[str, Any] = {
    "app.tagline": "Asistente de investigación digital con IA",
    "nav.overview": "Resumen",
    "nav.assistant": "Asistente",
    "nav.evidence": "Evidencia",
    "nav.entities": "Entidades",
    "nav.findings": "Hallazgos",
    "nav.diary": "Diario del caso",
    "nav.recommendations": "Pistas",
    "nav.report": "Informe",
    "common.loading": "Procesando…",
    "common.search": "Buscar",
    "common.send": "Enviar",
    "common.clear": "Limpiar",
    "common.language": "Idioma",
    "common.confidence": "Confianza",
    "common.severity": "Gravedad",
    "common.priority": "Prioridad",
    "common.source": "Fuente",
    "common.timestamp": "Fecha y hora",
    "common.file": "Archivo",
    "common.entity": "Entidad",
    "common.evidence": "Evidencia",
    "common.citations": "Citas",
    "common.risk": "Riesgo",
    "common.total": "Total",
    "common.count": "Recuento",
    "common.average": "Promedio",
    "common.largest": "Mayor",
    "chat.placeholder": "Pregunte sobre la evidencia, una entidad, una cronología o los próximos pasos…",
    "chat.welcome": "Soy su asistente forense. Respondo solo con la evidencia indexada y cito cada afirmación.",
    "chat.suggestions": "Preguntas sugeridas",
    "chat.followups": "Podría preguntar",
    "answer.search.header": "Encontré **{n}** resultado(s). Las coincidencias más fuertes son:",
    "answer.search.none": "Ninguna evidencia coincide con **{query}**. Amplíe el rango de fechas, elimine un filtro o busque una persona, cuenta, teléfono o usuario.",
    "answer.entity.header": "**{name}** ({entity_id}) — riesgo {risk}",
    "answer.entity.profile": "{name} aparece en **{n_events}** evento(s) en {sources}. Primera aparición {first_seen}; última {last_seen}.",
    "answer.entity.not_found": "No encontré una entidad que coincida con **{term}**. Entidades conocidas: {names}.",
    "answer.timeline.header": "Cronología — **{n}** elemento(s) en la ventana seleccionada:",
    "answer.timeline.none": "No se encontró evidencia fechada para esa ventana.",
    "answer.relationship.header": "Relación entre **{a}** y **{b}**:",
    "answer.relationship.none": "No se encontró una conexión directa o de un paso en la evidencia indexada.",
    "answer.correlation.header": "Encontré **{n}** comunicación(es) dentro de {tol} minutos de una transferencia:",
    "answer.correlation.none": "No se encontró ninguna llamada/publicación dentro de {tol} minutos de una transferencia para esta entidad.",
    "answer.aggregate.header": "Agregado sobre **{n}** evento(s) coincidentes:",
    "answer.aggregate.total": "Importe total: **{total}**.",
    "answer.anomaly.header": "Encontré **{n}** hallazgo(s) de patrón sospechoso. Prioridad más alta primero:",
    "answer.anomaly.none": "Aún no hay hallazgos. Ejecute primero el análisis.",
    "answer.summarize.header": "Resumen del caso — **{title}**",
    "answer.recommend.header": "Próximos pasos recomendados, prioridad más alta primero:",
    "answer.diary.header": "Diario del caso — **{n}** entradas. Estado de la cadena: {status}.",
    "answer.report.header": "Borrador del informe generado: {path}",
    "answer.evidence_inventory.header": "Inventario de evidencia — **{n}** elemento(s):",
    "answer.entity_list.header": "Entidades — **{n}** resueltas:",
    "answer.help": "Puedo buscar evidencia, perfilar entidades, crear cronologías, explicar relaciones, correlacionar llamadas con transferencias, agregar importes, resumir el caso, recomendar próximos pasos, mostrar el diario y redactar un informe. Pregunte en inglés, hindi, español, francés, alemán o portugués.",
    "summary.overview": "**{title}** abarca **{n_evidence}** elemento(s) de evidencia en {sources}, desde {start} hasta {end}. El caso tiene **{n_entities}** entidades resueltas y **{n_findings}** hallazgo(s).",
    "summary.entity": "**{name}** ({entity_id}) — riesgo {risk}. **{n_events}** evento(s) en {sources}. Primera aparición {first_seen}; última {last_seen}. Identificadores: {identifiers}.",
    "summary.gaps": "Brechas de evidencia: {gaps}.",
    "summary.next_steps": "Recomendación principal: {action}",
    "report.case_report": "Borrador del informe del caso",
    "report.executive_summary": "Resumen ejecutivo",
    "report.evidence_inventory": "Inventario de evidencia",
    "report.entities": "Entidades",
    "report.findings": "Hallazgos de patrones sospechosos",
    "report.timeline": "Cronología",
    "report.recommendations": "Recomendaciones de investigación",
    "report.case_diary": "Diario del caso",
    "report.methodology": "Metodología y limitaciones",
    "report.analytics_appendix": "Anexo analítico",
    "report.query_log": "Registro de consultas",
    "report.disclaimer": "Este es un borrador generado a partir de evidencia indexada. Cada hallazgo debe ser verificado por un investigador cualificado antes de usarse en un procedimiento legal.",
    "report.diary_chain_ok": "Cadena hash del diario verificada.",
    "report.diary_chain_bad": "FALLO en la verificación de la cadena hash del diario: investigue una posible manipulación.",
    "query.examples": [
        "Muestre las transferencias superiores a 10.000 de Viktor en los últimos 30 días.",
        "¿A quién llamó Viktor justo antes de la transferencia?",
        "¿Cuál es la conexión entre Viktor y Elena?",
        "¿Qué parece sospechoso en este caso?",
        "Resuma el caso.",
        "¿Qué debería investigar a continuación?",
        "Muestre el diario del caso.",
        "Redacte un informe del caso.",
    ],
}

_FR: Dict[str, Any] = {
    "app.tagline": "Assistant d'enquête numérique propulsé par l'IA",
    "nav.overview": "Aperçu",
    "nav.assistant": "Assistant",
    "nav.evidence": "Preuves",
    "nav.entities": "Entités",
    "nav.findings": "Constats",
    "nav.diary": "Journal de l'affaire",
    "nav.recommendations": "Pistes",
    "nav.report": "Rapport",
    "common.loading": "Traitement…",
    "common.search": "Rechercher",
    "common.send": "Envoyer",
    "common.clear": "Effacer",
    "common.language": "Langue",
    "common.confidence": "Confiance",
    "common.severity": "Gravité",
    "common.priority": "Priorité",
    "common.source": "Source",
    "common.timestamp": "Horodatage",
    "common.file": "Fichier",
    "common.entity": "Entité",
    "common.evidence": "Preuve",
    "common.citations": "Citations",
    "common.risk": "Risque",
    "common.total": "Total",
    "common.count": "Nombre",
    "common.average": "Moyenne",
    "common.largest": "Plus grand",
    "chat.placeholder": "Posez une question sur les preuves, une entité, une chronologie ou les prochaines étapes…",
    "chat.welcome": "Je suis votre assistant forensique. Je réponds uniquement à partir des preuves indexées et je cite chaque fait.",
    "chat.suggestions": "Questions suggérées",
    "chat.followups": "Vous pourriez demander",
    "answer.search.header": "J'ai trouvé **{n}** résultat(s). Les correspondances les plus fortes sont :",
    "answer.search.none": "Aucune preuve ne correspond à **{query}**. Élargissez la période, retirez un filtre ou recherchez une personne, un compte, un téléphone ou un identifiant.",
    "answer.entity.header": "**{name}** ({entity_id}) — risque {risk}",
    "answer.entity.profile": "{name} apparaît dans **{n_events}** événement(s) sur {sources}. Première apparition {first_seen} ; dernière {last_seen}.",
    "answer.entity.not_found": "Je n'ai pas trouvé d'entité correspondant à **{term}**. Entités connues : {names}.",
    "answer.timeline.header": "Chronologie — **{n}** élément(s) dans la fenêtre sélectionnée :",
    "answer.timeline.none": "Aucune preuve datée n'a été trouvée pour cette fenêtre.",
    "answer.relationship.header": "Relation entre **{a}** et **{b}** :",
    "answer.relationship.none": "Aucune connexion directe ou à un saut n'a été trouvée dans les preuves indexées.",
    "answer.correlation.header": "J'ai trouvé **{n}** communication(s) dans les {tol} minutes d'un virement :",
    "answer.correlation.none": "Aucun appel/publication trouvé dans les {tol} minutes d'un virement pour cette entité.",
    "answer.aggregate.header": "Agrégat sur **{n}** événement(s) correspondant(s) :",
    "answer.aggregate.total": "Montant total : **{total}**.",
    "answer.anomaly.header": "J'ai trouvé **{n}** constat(s) de schéma suspect. Priorité la plus élevée d'abord :",
    "answer.anomaly.none": "Aucun constat pour l'instant. Lancez d'abord l'analyse.",
    "answer.summarize.header": "Résumé de l'affaire — **{title}**",
    "answer.recommend.header": "Prochaines étapes recommandées, priorité la plus élevée d'abord :",
    "answer.diary.header": "Journal de l'affaire — **{n}** entrées. État de la chaîne : {status}.",
    "answer.report.header": "Projet de rapport généré : {path}",
    "answer.evidence_inventory.header": "Inventaire des preuves — **{n}** élément(s) :",
    "answer.entity_list.header": "Entités — **{n}** résolues :",
    "answer.help": "Je peux rechercher des preuves, profiler des entités, construire des chronologies, expliquer des relations, corréler appels et virements, agréger des montants, résumer l'affaire, recommander les prochaines étapes, afficher le journal et rédiger un rapport. Posez vos questions en anglais, hindi, espagnol, français, allemand ou portugais.",
    "summary.overview": "**{title}** couvre **{n_evidence}** élément(s) de preuve sur {sources}, de {start} à {end}. L'affaire compte **{n_entities}** entités résolues et **{n_findings}** constat(s).",
    "summary.entity": "**{name}** ({entity_id}) — risque {risk}. **{n_events}** événement(s) sur {sources}. Première apparition {first_seen} ; dernière {last_seen}. Identifiants : {identifiers}.",
    "summary.gaps": "Lacunes de preuves : {gaps}.",
    "summary.next_steps": "Recommandation principale : {action}",
    "report.case_report": "Projet de rapport d'affaire",
    "report.executive_summary": "Résumé exécutif",
    "report.evidence_inventory": "Inventaire des preuves",
    "report.entities": "Entités",
    "report.findings": "Constats de schémas suspects",
    "report.timeline": "Chronologie",
    "report.recommendations": "Recommandations d'enquête",
    "report.case_diary": "Journal de l'affaire",
    "report.methodology": "Méthodologie et limites",
    "report.analytics_appendix": "Annexe analytique",
    "report.query_log": "Journal des requêtes",
    "report.disclaimer": "Ceci est un projet généré à partir de preuves indexées. Chaque constat doit être vérifié par un enquêteur qualifié avant toute utilisation judiciaire.",
    "report.diary_chain_ok": "Chaîne de hachage du journal vérifiée.",
    "report.diary_chain_bad": "ÉCHEC de la vérification de la chaîne de hachage du journal — recherchez une altération.",
    "query.examples": [
        "Montrez les virements supérieurs à 10 000 de Viktor sur les 30 derniers jours.",
        "Qui Viktor a-t-il appelé juste avant le virement ?",
        "Quel est le lien entre Viktor et Elena ?",
        "Qu'est-ce qui semble suspect dans cette affaire ?",
        "Résumez l'affaire.",
        "Que devrais-je enquêter ensuite ?",
        "Affichez le journal de l'affaire.",
        "Rédigez un rapport d'affaire.",
    ],
}

_DE: Dict[str, Any] = {
    "app.tagline": "KI-gestützter digitaler Ermittlungsassistent",
    "nav.overview": "Übersicht",
    "nav.assistant": "Assistent",
    "nav.evidence": "Beweise",
    "nav.entities": "Entitäten",
    "nav.findings": "Erkenntnisse",
    "nav.diary": "Falltagebuch",
    "nav.recommendations": "Hinweise",
    "nav.report": "Bericht",
    "common.loading": "Wird verarbeitet…",
    "common.search": "Suchen",
    "common.send": "Senden",
    "common.clear": "Leeren",
    "common.language": "Sprache",
    "common.confidence": "Konfidenz",
    "common.severity": "Schweregrad",
    "common.priority": "Priorität",
    "common.source": "Quelle",
    "common.timestamp": "Zeitstempel",
    "common.file": "Datei",
    "common.entity": "Entität",
    "common.evidence": "Beweis",
    "common.citations": "Zitate",
    "common.risk": "Risiko",
    "common.total": "Summe",
    "common.count": "Anzahl",
    "common.average": "Durchschnitt",
    "common.largest": "Größter",
    "chat.placeholder": "Fragen Sie nach Beweisen, einer Entität, einer Zeitlinie oder den nächsten Schritten…",
    "chat.welcome": "Ich bin Ihr forensischer Fallassistent. Ich antworte nur aus den indexierten Beweisen und zitiere jede Tatsache.",
    "chat.suggestions": "Vorgeschlagene Fragen",
    "chat.followups": "Sie könnten fragen",
    "answer.search.header": "Ich habe **{n}** Treffer gefunden. Die stärksten Übereinstimmungen sind:",
    "answer.search.none": "Keine Beweise passen zu **{query}**. Erweitern Sie den Zeitraum, entfernen Sie einen Filter oder suchen Sie nach Person, Konto, Telefon oder Handle.",
    "answer.entity.header": "**{name}** ({entity_id}) — Risiko {risk}",
    "answer.entity.profile": "{name} erscheint in **{n_events}** Ereignis(sen) über {sources}. Erstmals {first_seen}; zuletzt {last_seen}.",
    "answer.entity.not_found": "Ich konnte keine Entität zu **{term}** finden. Bekannte Entitäten: {names}.",
    "answer.timeline.header": "Zeitlinie — **{n}** Element(e) im gewählten Zeitraum:",
    "answer.timeline.none": "Für diesen Zeitraum wurden keine datierten Beweise gefunden.",
    "answer.relationship.header": "Beziehung zwischen **{a}** und **{b}**:",
    "answer.relationship.none": "In den indexierten Beweisen wurde keine direkte oder einstufige Verbindung gefunden.",
    "answer.correlation.header": "Ich habe **{n}** Kommunikation(en) innerhalb von {tol} Minuten um eine Überweisung gefunden:",
    "answer.correlation.none": "Für diese Entität wurde kein Anruf/Beitrag innerhalb von {tol} Minuten um eine Überweisung gefunden.",
    "answer.aggregate.header": "Aggregat über **{n}** passende Ereignisse:",
    "answer.aggregate.total": "Gesamtbetrag: **{total}**.",
    "answer.anomaly.header": "Ich habe **{n}** verdächtige Muster gefunden. Höchste Priorität zuerst:",
    "answer.anomaly.none": "Noch keine Erkenntnisse. Führen Sie zuerst die Analyse aus.",
    "answer.summarize.header": "Fallzusammenfassung — **{title}**",
    "answer.recommend.header": "Empfohlene nächste Schritte, höchste Priorität zuerst:",
    "answer.diary.header": "Falltagebuch — **{n}** Einträge. Kettenstatus: {status}.",
    "answer.report.header": "Berichtsentwurf erstellt: {path}",
    "answer.evidence_inventory.header": "Beweisbestand — **{n}** Element(e):",
    "answer.entity_list.header": "Entitäten — **{n}** aufgelöst:",
    "answer.help": "Ich kann Beweise durchsuchen, Entitäten profilieren, Zeitlinien erstellen, Beziehungen erklären, Anrufe mit Überweisungen korrelieren, Beträge aggregieren, den Fall zusammenfassen, nächste Schritte empfehlen, das Tagebuch anzeigen und einen Bericht entwerfen. Fragen Sie auf Englisch, Hindi, Spanisch, Französisch, Deutsch oder Portugiesisch.",
    "summary.overview": "**{title}** umfasst **{n_evidence}** Beweiselement(e) aus {sources}, von {start} bis {end}. Der Fall hat **{n_entities}** aufgelöste Entitäten und **{n_findings}** Erkenntnis(se).",
    "summary.entity": "**{name}** ({entity_id}) — Risiko {risk}. **{n_events}** Ereignis(se) über {sources}. Erstmals {first_seen}; zuletzt {last_seen}. Identifikatoren: {identifiers}.",
    "summary.gaps": "Beweislücken: {gaps}.",
    "summary.next_steps": "Top-Empfehlung: {action}",
    "report.case_report": "Berichtsentwurf zum Fall",
    "report.executive_summary": "Zusammenfassung",
    "report.evidence_inventory": "Beweisbestand",
    "report.entities": "Entitäten",
    "report.findings": "Verdächtige Muster",
    "report.timeline": "Zeitlinie",
    "report.recommendations": "Ermittlungsempfehlungen",
    "report.case_diary": "Falltagebuch",
    "report.methodology": "Methodik und Grenzen",
    "report.analytics_appendix": "Analyseanhang",
    "report.query_log": "Abfrageprotokoll",
    "report.disclaimer": "Dies ist ein Entwurf aus indexierten Beweisen. Jede Erkenntnis muss vor einer juristischen Verwendung von einer qualifizierten Ermittlungsperson überprüft werden.",
    "report.diary_chain_ok": "Hash-Kette des Tagebuchs verifiziert.",
    "report.diary_chain_bad": "Hash-Kettenprüfung des Tagebuchs FEHLGESCHLAGEN — auf Manipulation prüfen.",
    "query.examples": [
        "Zeige Überweisungen über 10.000 von Viktor in den letzten 30 Tagen.",
        "Wen hat Viktor direkt vor der Überweisung angerufen?",
        "Wie ist die Verbindung zwischen Viktor und Elena?",
        "Was wirkt in diesem Fall verdächtig?",
        "Fasse den Fall zusammen.",
        "Was sollte ich als Nächstes untersuchen?",
        "Zeige das Falltagebuch.",
        "Entwirf einen Fallbericht.",
    ],
}

_PT: Dict[str, Any] = {
    "app.tagline": "Assistente de investigação digital com IA",
    "nav.overview": "Visão geral",
    "nav.assistant": "Assistente",
    "nav.evidence": "Evidências",
    "nav.entities": "Entidades",
    "nav.findings": "Achados",
    "nav.diary": "Diário do caso",
    "nav.recommendations": "Pistas",
    "nav.report": "Relatório",
    "common.loading": "Processando…",
    "common.search": "Pesquisar",
    "common.send": "Enviar",
    "common.clear": "Limpar",
    "common.language": "Idioma",
    "common.confidence": "Confiança",
    "common.severity": "Gravidade",
    "common.priority": "Prioridade",
    "common.source": "Fonte",
    "common.timestamp": "Data e hora",
    "common.file": "Arquivo",
    "common.entity": "Entidade",
    "common.evidence": "Evidência",
    "common.citations": "Citações",
    "common.risk": "Risco",
    "common.total": "Total",
    "common.count": "Contagem",
    "common.average": "Média",
    "common.largest": "Maior",
    "chat.placeholder": "Pergunte sobre as evidências, uma entidade, uma linha do tempo ou os próximos passos…",
    "chat.welcome": "Sou seu assistente forense. Respondo apenas com as evidências indexadas e cito cada fato.",
    "chat.suggestions": "Perguntas sugeridas",
    "chat.followups": "Você poderia perguntar",
    "answer.search.header": "Encontrei **{n}** resultado(s). As correspondências mais fortes são:",
    "answer.search.none": "Nenhuma evidência corresponde a **{query}**. Amplie o período, remova um filtro ou pesquise uma pessoa, conta, telefone ou usuário.",
    "answer.entity.header": "**{name}** ({entity_id}) — risco {risk}",
    "answer.entity.profile": "{name} aparece em **{n_events}** evento(s) em {sources}. Primeira aparição {first_seen}; última {last_seen}.",
    "answer.entity.not_found": "Não encontrei uma entidade correspondente a **{term}**. Entidades conhecidas: {names}.",
    "answer.timeline.header": "Linha do tempo — **{n}** item(ns) na janela selecionada:",
    "answer.timeline.none": "Nenhuma evidência datada foi encontrada para essa janela.",
    "answer.relationship.header": "Relação entre **{a}** e **{b}**:",
    "answer.relationship.none": "Nenhuma conexão direta ou de um salto foi encontrada nas evidências indexadas.",
    "answer.correlation.header": "Encontrei **{n}** comunicação(ões) dentro de {tol} minutos de uma transferência:",
    "answer.correlation.none": "Nenhuma chamada/publicação foi encontrada dentro de {tol} minutos de uma transferência para esta entidade.",
    "answer.aggregate.header": "Agregado sobre **{n}** evento(s) correspondente(s):",
    "answer.aggregate.total": "Valor total: **{total}**.",
    "answer.anomaly.header": "Encontrei **{n}** achado(s) de padrão suspeito. Maior prioridade primeiro:",
    "answer.anomaly.none": "Ainda não há achados. Execute a análise primeiro.",
    "answer.summarize.header": "Resumo do caso — **{title}**",
    "answer.recommend.header": "Próximos passos recomendados, maior prioridade primeiro:",
    "answer.diary.header": "Diário do caso — **{n}** entradas. Status da cadeia: {status}.",
    "answer.report.header": "Relatório preliminar gerado: {path}",
    "answer.evidence_inventory.header": "Inventário de evidências — **{n}** item(ns):",
    "answer.entity_list.header": "Entidades — **{n}** resolvidas:",
    "answer.help": "Posso pesquisar evidências, criar perfis de entidades, montar linhas do tempo, explicar relações, correlacionar chamadas com transferências, agregar valores, resumir o caso, recomendar próximos passos, mostrar o diário e redigir um relatório. Pergunte em inglês, hindi, espanhol, francês, alemão ou português.",
    "summary.overview": "**{title}** abrange **{n_evidence}** item(ns) de evidência em {sources}, de {start} a {end}. O caso tem **{n_entities}** entidades resolvidas e **{n_findings}** achado(s).",
    "summary.entity": "**{name}** ({entity_id}) — risco {risk}. **{n_events}** evento(s) em {sources}. Primeira aparição {first_seen}; última {last_seen}. Identificadores: {identifiers}.",
    "summary.gaps": "Lacunas de evidência: {gaps}.",
    "summary.next_steps": "Recomendação principal: {action}",
    "report.case_report": "Relatório preliminar do caso",
    "report.executive_summary": "Resumo executivo",
    "report.evidence_inventory": "Inventário de evidências",
    "report.entities": "Entidades",
    "report.findings": "Achados de padrões suspeitos",
    "report.timeline": "Linha do tempo",
    "report.recommendations": "Recomendações de investigação",
    "report.case_diary": "Diário do caso",
    "report.methodology": "Metodologia e limitações",
    "report.analytics_appendix": "Anexo analítico",
    "report.query_log": "Registro de consultas",
    "report.disclaimer": "Este é um rascunho gerado a partir de evidências indexadas. Cada achado deve ser verificado por um investigador qualificado antes de uso em processos legais.",
    "report.diary_chain_ok": "Cadeia de hash do diário verificada.",
    "report.diary_chain_bad": "FALHA na verificação da cadeia de hash do diário — investigue possível adulteração.",
    "query.examples": [
        "Mostre transferências acima de 10.000 de Viktor nos últimos 30 dias.",
        "Quem Viktor ligou logo antes da transferência?",
        "Qual é a conexão entre Viktor e Elena?",
        "O que parece suspeito neste caso?",
        "Resuma o caso.",
        "O que devo investigar a seguir?",
        "Mostre o diário do caso.",
        "Redija um relatório do caso.",
    ],
}

CATALOGS: Dict[str, Dict[str, Any]] = {
    "en": _EN,
    "hi": _HI,
    "es": _ES,
    "fr": _FR,
    "de": _DE,
    "pt": _PT,
}

# A few keys added after the main catalogs were written; keeping them in one
# block avoids duplicating the whole catalog for a single sentence.
CATALOGS["hi"]["answer.relationship.shared"] = (
    "कोई सीधा संरचित लेन-देन/कॉल संबंध नहीं मिला, लेकिन ये संस्थाएँ "
    "**{n}** दस्तावेज़(ों) में साथ दिखती हैं:"
)
CATALOGS["es"]["answer.relationship.shared"] = (
    "No se encontró una arista estructurada directa de transacción/llamada, "
    "pero las entidades coaparecen en **{n}** documento(s):"
)
CATALOGS["fr"]["answer.relationship.shared"] = (
    "Aucune arête structurée directe transaction/appel n'a été trouvée, mais "
    "les entités co-apparaissent dans **{n}** document(s) :"
)
CATALOGS["de"]["answer.relationship.shared"] = (
    "Keine direkte strukturierte Transaktions-/Anrufkante gefunden, aber die "
    "Entitäten kommen in **{n}** Dokument(en) gemeinsam vor:"
)
CATALOGS["pt"]["answer.relationship.shared"] = (
    "Nenhuma aresta estruturada direta de transação/chamada foi encontrada, "
    "mas as entidades coocorrem em **{n}** documento(s):"
)

CATALOGS["hi"].update({
    "stats.evidence": "साक्ष्य", "stats.entities": "संस्थाएँ",
    "stats.events": "घटनाएँ", "stats.findings": "निष्कर्ष",
})
CATALOGS["es"].update({
    "stats.evidence": "Evidencia", "stats.entities": "Entidades",
    "stats.events": "Eventos", "stats.findings": "Hallazgos",
})
CATALOGS["fr"].update({
    "stats.evidence": "Preuves", "stats.entities": "Entités",
    "stats.events": "Événements", "stats.findings": "Constats",
})
CATALOGS["de"].update({
    "stats.evidence": "Beweise", "stats.entities": "Entitäten",
    "stats.events": "Ereignisse", "stats.findings": "Erkenntnisse",
})
CATALOGS["pt"].update({
    "stats.evidence": "Evidências", "stats.entities": "Entidades",
    "stats.events": "Eventos", "stats.findings": "Achados",
})

CATALOGS["hi"].update({
    "summary.top_entities": "सबसे अधिक जोखिम वाली संस्थाएँ: {names}।",
    "summary.findings": "संदिग्ध पैटर्न: {kinds}।",
    "summary.gaps": "साक्ष्य की कमियाँ: {gaps}।",
    "summary.next_steps": "शीर्ष सिफ़ारिश: {action}",
    "summary.limitations": "सीमाएँ: {items}",
    "summary.entity.findings": "जुड़े निष्कर्ष: {titles}।",
    "summary.entity.counterparts": "प्रमुख प्रतिपक्ष: {names}।",
})
CATALOGS["es"].update({
    "summary.top_entities": "Entidades de mayor riesgo: {names}.",
    "summary.findings": "Patrones sospechosos: {kinds}.",
    "summary.gaps": "Brechas de evidencia: {gaps}.",
    "summary.next_steps": "Recomendación principal: {action}",
    "summary.limitations": "Limitaciones: {items}",
    "summary.entity.findings": "Hallazgos vinculados: {titles}.",
    "summary.entity.counterparts": "Principales contrapartes: {names}.",
})
CATALOGS["fr"].update({
    "summary.top_entities": "Entités à risque le plus élevé : {names}.",
    "summary.findings": "Schémas suspects : {kinds}.",
    "summary.gaps": "Lacunes de preuves : {gaps}.",
    "summary.next_steps": "Recommandation principale : {action}",
    "summary.limitations": "Limites : {items}",
    "summary.entity.findings": "Constats liés : {titles}.",
    "summary.entity.counterparts": "Principales contreparties : {names}.",
})
CATALOGS["de"].update({
    "summary.top_entities": "Höchstrisiko-Entitäten: {names}.",
    "summary.findings": "Verdächtige Muster: {kinds}.",
    "summary.gaps": "Beweislücken: {gaps}.",
    "summary.next_steps": "Top-Empfehlung: {action}",
    "summary.limitations": "Grenzen: {items}",
    "summary.entity.findings": "Verknüpfte Erkenntnisse: {titles}.",
    "summary.entity.counterparts": "Wichtigste Gegenparteien: {names}.",
})
CATALOGS["pt"].update({
    "summary.top_entities": "Entidades de maior risco: {names}.",
    "summary.findings": "Padrões suspeitos: {kinds}.",
    "summary.gaps": "Lacunas de evidência: {gaps}.",
    "summary.next_steps": "Recomendação principal: {action}",
    "summary.limitations": "Limitações: {items}",
    "summary.entity.findings": "Achados vinculados: {titles}.",
    "summary.entity.counterparts": "Principais contrapartes: {names}.",
})

# Recommendation actions/rationales in the supported locales.  Keeping these
# in the catalog means a language switch localizes the leads, not just the UI.
CATALOGS["hi"].update({
    "rec.gap.bank.action": "मामले की अवधि के लिए पूर्ण बैंक/लेन-देन रिकॉर्ड प्राप्त करें।",
    "rec.gap.bank.rationale": "कोई बैंक साक्ष्य अनुक्रमित नहीं है, इसलिए वित्तीय प्रवाह, स्ट्रक्चरिंग और लेयरिंग सत्यापित नहीं की जा सकती।",
    "rec.gap.cdr.action": "प्रासंगिक फ़ोन नंबरों के लिए कॉल डिटेल रिकॉर्ड (CDR) प्राप्त करें।",
    "rec.gap.cdr.rationale": "कोई CDR साक्ष्य अनुक्रमित नहीं है, इसलिए कॉल-ट्रांसफर सह-स्थान और संचार नेटवर्क का परीक्षण नहीं किया जा सकता।",
    "rec.gap.social.action": "प्रासंगिक हैंडल के लिए सोशल-मीडिया पोस्ट और खाता पहचानकर्ता एकत्र करें।",
    "rec.gap.social.rationale": "कोई सोशल साक्ष्य अनुक्रमित नहीं है; ओपन-सोर्स संदर्भ और कमजोर संबंध गायब हैं।",
    "rec.gap.email.action": "प्रासंगिक खातों के लिए ईमेल और चैट निर्यात एकत्र करें।",
    "rec.gap.email.rationale": "कोई ईमेल/चैट साक्ष्य अनुक्रमित नहीं है; मंशा, समन्वय और निर्देश का रिकॉर्ड गायब हो सकता है।",
    "rec.gap.timestamps.action": "बिना तिथि वाले साक्ष्य के लिए तिथि-युक्त रिकॉर्ड प्राप्त करें।",
    "rec.gap.timestamps.rationale": "कुछ साक्ष्यों में तिथि नहीं है, इसलिए उन्हें केस टाइमलाइन पर नहीं रखा जा सकता।",
    "rec.gap.volume.action": "पतले स्रोतों के लिए लंबी अवधि या अधिक रिकॉर्ड एकत्र करें।",
    "rec.gap.volume.rationale": "प्रति-संस्था मात्रा विश्वसनीयता सीमा से नीचे है; सांख्यिकीय निष्कर्ष कमजोर हैं।",
    "rec.gap.unresolved.action": "अनजान पहचानकर्ताओं (खाते, फ़ोन, हैंडल) को व्यक्तियों से जोड़ें।",
    "rec.gap.unresolved.rationale": "अनजान पहचानकर्ता क्रॉस-सोर्स लिंकिंग रोकते हैं और संबंध छिपाते हैं।",
    "rec.interview.action": "{name} ({entity_id}) का साक्षात्कार लें और बयान रिकॉर्ड करें।",
    "rec.interview.rationale": "यह संस्था {risk} जोखिम पर है और इसके {n_findings} जुड़े निष्कर्ष हैं।",
    "rec.records.action": "{identifiers} के लिए पूर्ण खाता/फ़ोन रिकॉर्ड का अनुरोध करें।",
    "rec.records.rationale": "{name} की चिह्नित गतिविधि सत्यापित करने के लिए पूर्ण रिकॉर्ड आवश्यक हैं।",
    "rec.structuring.action": "{name} के लिए {threshold} रिपोर्टिंग सीमा से 88–100% नीचे के सभी लेन-देन की समीक्षा करें।",
    "rec.structuring.rationale": "स्ट्रक्चरिंग पैटर्न रिपोर्टिंग सीमा से जानबूझकर बचने का संकेत देता है।",
    "rec.colocation.action": "{location} के आसपास {timestamp} के लिए टावर डंप / CCTV / एक्सेस लॉग प्राप्त करें।",
    "rec.colocation.rationale": "सहनशीलता अवधि के भीतर उसी पक्ष को कॉल/पोस्ट के बाद ट्रांसफर हुआ।",
    "rec.chain.action": "मल्टी-हॉप प्रवाह का मानचित्रण करें और श्रृंखला में लाभार्थी खातों के लिए KYC का अनुरोध करें।",
    "rec.chain.rationale": "मध्यवर्ती खातों के माध्यम से तेजी से लेयरिंग प्लेसमेंट/लेयरिंग का संकेत है।",
    "rec.fan.action": "{name} के {n} प्रतिपक्षों को म्यूल-खाता व्यवहार के लिए जाँचें।",
    "rec.fan.rationale": "फैन-इन/फैन-आउट संरचना संग्रह और फैलाव के अनुरूप है।",
    "rec.dormancy.action": "{name} के पुनः सक्रिय होने की तिथि के आसपास खाता अधिग्रहण, जीवन घटनाओं या कारावास की जाँच करें।",
    "rec.dormancy.rationale": "लंबे समय तक निष्क्रिय खाता अचानक सक्रिय होकर धन स्थानांतरित करता है।",
    "rec.corroborate.action": "{name} के {kind} निष्कर्ष की स्वतंत्र स्रोत से पुष्टि करें।",
    "rec.corroborate.rationale": "यह निष्कर्ष एकल संकेत पर आधारित है; पुष्टि मामले को मजबूत करती है।",
    "rec.integrity.action": "चिह्नित साक्ष्य के SHA-256 हैश और चेन-ऑफ-कस्टडी रिकॉर्ड सत्यापित करें।",
    "rec.integrity.rationale": "रिपोर्ट में उपयोग से पहले साक्ष्य की अखंडता दर्ज होनी चाहिए।",
    "rec.timeline_gap.action": "{start} से {end} तक की समयरेखा अंतराल के लिए रिकॉर्ड का अनुरोध करें।",
    "rec.timeline_gap.rationale": "तिथि-युक्त साक्ष्य में अंतराल है जो प्रासंगिक गतिविधि छिपा सकता है।",
    "rec.report.action": "ड्राफ्ट केस रिपोर्ट तैयार करें और समीक्षा करें।",
    "rec.report.rationale": "संरचित रिपोर्ट से निष्कर्ष, साक्ष्य लिंक और सीमाएँ समीक्षा योग्य होती हैं।",
})

CATALOGS["es"].update({
    "rec.gap.bank.action": "Obtenga los registros bancarios/de transacciones completos del período del caso.",
    "rec.gap.bank.rationale": "No hay evidencia bancaria indexada, por lo que no se pueden verificar flujos financieros, estructuración ni capas.",
    "rec.gap.cdr.action": "Obtenga los registros de llamadas (CDR) de los números de teléfono relevantes.",
    "rec.gap.cdr.rationale": "No hay evidencia CDR indexada, por lo que no se pueden probar la colocalización llamada-transferencia ni las redes de comunicación.",
    "rec.gap.social.action": "Recopile publicaciones e identificadores de cuentas de redes sociales relevantes.",
    "rec.gap.social.rationale": "No hay evidencia social indexada; falta contexto de fuentes abiertas y vínculos débiles.",
    "rec.gap.email.action": "Recopile exportaciones de correo electrónico y chat de las cuentas relevantes.",
    "rec.gap.email.rationale": "No hay evidencia de correo/chat indexada; pueden faltar rastros de intención, coordinación e instrucciones.",
    "rec.gap.timestamps.action": "Obtenga registros fechados para la evidencia sin fecha.",
    "rec.gap.timestamps.rationale": "Parte de la evidencia no tiene fecha y no puede situarse en la cronología.",
    "rec.gap.volume.action": "Recopile un período más largo o más registros para las fuentes con poco volumen.",
    "rec.gap.volume.rationale": "El volumen por entidad está por debajo del umbral de fiabilidad; los hallazgos estadísticos son débiles.",
    "rec.gap.unresolved.action": "Resuelva los identificadores no coincidentes (cuentas, teléfonos, usuarios) a personas.",
    "rec.gap.unresolved.rationale": "Los identificadores sin resolver impiden el enlace entre fuentes y ocultan relaciones.",
    "rec.interview.action": "Entreviste a {name} ({entity_id}) y registre una declaración.",
    "rec.interview.rationale": "Esta entidad tiene riesgo {risk} y {n_findings} hallazgo(s) vinculado(s).",
    "rec.records.action": "Solicite los registros completos de cuenta/teléfono de {identifiers}.",
    "rec.records.rationale": "Se necesitan registros completos para verificar la actividad señalada de {name}.",
    "rec.structuring.action": "Revise todas las transacciones en la banda del 88–100% por debajo del límite de {threshold} para {name}.",
    "rec.structuring.rationale": "El patrón de estructuración sugiere una evitación deliberada de los umbrales de reporte.",
    "rec.colocation.action": "Obtenga el volcado de torre / CCTV / registros de acceso de {location} alrededor de {timestamp}.",
    "rec.colocation.rationale": "Una llamada/publicación fue seguida por una transferencia a la misma parte dentro de la ventana de tolerancia.",
    "rec.chain.action": "Mapee el flujo multi-salto y solicite KYC de las cuentas beneficiarias de la cadena.",
    "rec.chain.rationale": "El rápido apilamiento a través de cuentas intermedias es un indicador clásico de colocación/capas.",
    "rec.fan.action": "Investigue a los {n} contrapartes de {name} por comportamiento de cuentas mula.",
    "rec.fan.rationale": "La estructura fan-in/fan-out es consistente con recaudación y dispersión.",
    "rec.dormancy.action": "Verifique toma de control de cuenta, eventos vitales o encarcelamiento alrededor de la fecha de reactivación de {name}.",
    "rec.dormancy.rationale": "Una cuenta inactiva se reactivó repentinamente y movió fondos.",
    "rec.corroborate.action": "Corrobore el hallazgo de {kind} de {name} con una fuente independiente.",
    "rec.corroborate.rationale": "El hallazgo se basa en una sola señal; la corroboración fortalece el caso.",
    "rec.integrity.action": "Verifique el hash SHA-256 y el registro de cadena de custodia de la evidencia señalada.",
    "rec.integrity.rationale": "La integridad de la evidencia debe documentarse antes de usarla en un informe.",
    "rec.timeline_gap.action": "Solicite registros que cubran el vacío de la cronología de {start} a {end}.",
    "rec.timeline_gap.rationale": "Hay un vacío en la evidencia fechada que podría ocultar actividad relevante.",
    "rec.report.action": "Genere y revise el borrador del informe del caso.",
    "rec.report.rationale": "Un informe estructurado hace revisables los hallazgos, los vínculos de evidencia y las limitaciones.",
})

CATALOGS["fr"].update({
    "rec.gap.bank.action": "Obtenez les relevés bancaires/de transactions complets pour la période de l'affaire.",
    "rec.gap.bank.rationale": "Aucune preuve bancaire n'est indexée ; les flux financiers, la structuration et le layering ne peuvent pas être vérifiés.",
    "rec.gap.cdr.action": "Obtenez les relevés d'appels (CDR) des numéros concernés.",
    "rec.gap.cdr.rationale": "Aucune preuve CDR n'est indexée ; la colocation appel-virement et les réseaux de communication ne peuvent pas être testés.",
    "rec.gap.social.action": "Collectez les publications et identifiants de comptes des réseaux sociaux concernés.",
    "rec.gap.social.rationale": "Aucune preuve sociale n'est indexée ; le contexte open source et les liens faibles manquent.",
    "rec.gap.email.action": "Collectez les exports e-mail et chat des comptes concernés.",
    "rec.gap.email.rationale": "Aucune preuve e-mail/chat n'est indexée ; les traces d'intention, de coordination et d'instruction peuvent manquer.",
    "rec.gap.timestamps.action": "Obtenez des enregistrements datés pour les preuves non datées.",
    "rec.gap.timestamps.rationale": "Certaines preuves n'ont pas de date et ne peuvent pas être placées sur la chronologie.",
    "rec.gap.volume.action": "Collectez une période plus longue ou plus d'enregistrements pour les sources peu fournies.",
    "rec.gap.volume.rationale": "Le volume par entité est inférieur au seuil de fiabilité ; les constats statistiques sont faibles.",
    "rec.gap.unresolved.action": "Résolvez les identifiants non appariés (comptes, téléphones, handles) en personnes.",
    "rec.gap.unresolved.rationale": "Les identifiants non résolus empêchent le rapprochement entre sources et masquent les relations.",
    "rec.interview.action": "Interrogez {name} ({entity_id}) et consignez une déclaration.",
    "rec.interview.rationale": "Cette entité présente un risque {risk} avec {n_findings} constat(s) lié(s).",
    "rec.records.action": "Demandez les relevés complets de compte/téléphone pour {identifiers}.",
    "rec.records.rationale": "Des relevés complets sont nécessaires pour vérifier l'activité signalée de {name}.",
    "rec.structuring.action": "Examinez toutes les transactions dans la bande 88–100 % sous le seuil de {threshold} pour {name}.",
    "rec.structuring.rationale": "Le schéma de structuration suggère un contournement délibéré des seuils de déclaration.",
    "rec.colocation.action": "Obtenez le relevé d'antenne / la vidéosurveillance / les journaux d'accès de {location} autour de {timestamp}.",
    "rec.colocation.rationale": "Un appel/publication a été suivi d'un virement à la même partie dans la fenêtre de tolérance.",
    "rec.chain.action": "Cartographiez le flux multi-saut et demandez le KYC des comptes bénéficiaires de la chaîne.",
    "rec.chain.rationale": "Le layering rapide via des comptes intermédiaires est un indicateur classique de placement/layering.",
    "rec.fan.action": "Vérifiez les {n} contreparties de {name} pour un comportement de compte mule.",
    "rec.fan.rationale": "La structure fan-in/fan-out est cohérente avec une collecte puis une dispersion.",
    "rec.dormancy.action": "Vérifiez une prise de contrôle de compte, un événement de vie ou une incarcération autour de la date de réactivation de {name}.",
    "rec.dormancy.rationale": "Un compte longtemps dormant s'est réactivé soudainement et a déplacé des fonds.",
    "rec.corroborate.action": "Corroborez le constat {kind} de {name} avec une source indépendante.",
    "rec.corroborate.rationale": "Le constat repose sur un seul signal ; la corroboration renforce l'affaire.",
    "rec.integrity.action": "Vérifiez le hash SHA-256 et le registre de chaîne de custody de la preuve signalée.",
    "rec.integrity.rationale": "L'intégrité de la preuve doit être documentée avant utilisation dans un rapport.",
    "rec.timeline_gap.action": "Demandez les enregistrements couvrant la lacune chronologique du {start} au {end}.",
    "rec.timeline_gap.rationale": "Une lacune dans les preuves datées pourrait masquer une activité pertinente.",
    "rec.report.action": "Générez et révisez le projet de rapport d'affaire.",
    "rec.report.rationale": "Un rapport structuré rend les constats, les liens de preuve et les limites vérifiables.",
})

CATALOGS["de"].update({
    "rec.gap.bank.action": "Beschaffen Sie vollständige Bank-/Transaktionsunterlagen für den Fallzeitraum.",
    "rec.gap.bank.rationale": "Es sind keine Bankbeweise indexiert; Finanzströme, Strukturierung und Layering können nicht verifiziert werden.",
    "rec.gap.cdr.action": "Beschaffen Sie Call Detail Records (CDR) für die relevanten Telefonnummern.",
    "rec.gap.cdr.rationale": "Es sind keine CDR-Beweise indexiert; Anruf-Überweisungs-Kolokation und Kommunikationsnetze können nicht geprüft werden.",
    "rec.gap.social.action": "Sammeln Sie Social-Media-Beiträge und Kontoidentifikatoren für die relevanten Handles.",
    "rec.gap.social.rationale": "Es sind keine sozialen Beweise indexiert; Open-Source-Kontext und schwache Verbindungen fehlen.",
    "rec.gap.email.action": "Sammeln Sie E-Mail- und Chat-Exporte für die relevanten Konten.",
    "rec.gap.email.rationale": "Es sind keine E-Mail-/Chat-Beweise indexiert; Absichts-, Koordinations- und Anweisungsspuren können fehlen.",
    "rec.gap.timestamps.action": "Beschaffen Sie datierte Unterlagen für die undatierten Beweise.",
    "rec.gap.timestamps.rationale": "Einige Beweise sind undatiert und können nicht in die Zeitlinie eingeordnet werden.",
    "rec.gap.volume.action": "Sammeln Sie einen längeren Zeitraum oder mehr Datensätze für die dünnen Quellen.",
    "rec.gap.volume.rationale": "Das Volumen pro Entität liegt unter der Zuverlässigkeitsschwelle; statistische Erkenntnisse sind schwach.",
    "rec.gap.unresolved.action": "Lösen Sie nicht zugeordnete Identifikatoren (Konten, Telefone, Handles) zu Personen auf.",
    "rec.gap.unresolved.rationale": "Nicht aufgelöste Identifikatoren verhindern quellenübergreifende Verknüpfungen und verbergen Beziehungen.",
    "rec.interview.action": "Befragen Sie {name} ({entity_id}) und protokollieren Sie eine Aussage.",
    "rec.interview.rationale": "Diese Entität hat {risk}-Risiko mit {n_findings} verknüpften Erkenntnis(sen).",
    "rec.records.action": "Fordern Sie vollständige Konto-/Telefonunterlagen für {identifiers} an.",
    "rec.records.rationale": "Vollständige Unterlagen sind nötig, um die markierte Aktivität von {name} zu verifizieren.",
    "rec.structuring.action": "Prüfen Sie alle Transaktionen im 88–100%-Band unter der {threshold}-Meldegrenze für {name}.",
    "rec.structuring.rationale": "Das Strukturierungsmuster deutet auf eine bewusste Umgehung der Meldegrenzen hin.",
    "rec.colocation.action": "Beschaffen Sie Funkzellenauswertung / CCTV / Zugriffsprotokolle für {location} um {timestamp}.",
    "rec.colocation.rationale": "Auf einen Anruf/Beitrag folgte innerhalb des Toleranzfensters eine Überweisung an dieselbe Partei.",
    "rec.chain.action": "Kartieren Sie den Multi-Hop-Fluss und fordern Sie KYC für die Begünstigtenkonten der Kette an.",
    "rec.chain.rationale": "Schnelles Layering über Zwischenkonten ist ein klassischer Platzierungs-/Layering-Indikator.",
    "rec.fan.action": "Prüfen Sie die {n} Gegenparteien von {name} auf Mule-Account-Verhalten.",
    "rec.fan.rationale": "Die Fan-in/Fan-out-Struktur entspricht Sammlung und Streuung.",
    "rec.dormancy.action": "Prüfen Sie Kontoübernahme, Lebensereignisse oder Inhaftierung um das Reaktivierungsdatum von {name}.",
    "rec.dormancy.rationale": "Ein lange ruhendes Konto wurde plötzlich reaktiviert und bewegte Gelder.",
    "rec.corroborate.action": "Bestätigen Sie die {kind}-Erkenntnis zu {name} mit einer unabhängigen Quelle.",
    "rec.corroborate.rationale": "Die Erkenntnis beruht auf einem einzelnen Signal; Bestätigung stärkt den Fall.",
    "rec.integrity.action": "Verifizieren Sie SHA-256-Hash und Chain-of-Custody-Eintrag des markierten Beweises.",
    "rec.integrity.rationale": "Die Beweisintegrität muss vor der Verwendung in einem Bericht dokumentiert werden.",
    "rec.timeline_gap.action": "Fordern Sie Unterlagen für die Zeitlücke von {start} bis {end} an.",
    "rec.timeline_gap.rationale": "Eine Lücke in den datierten Beweisen könnte relevante Aktivitäten verbergen.",
    "rec.report.action": "Erstellen und prüfen Sie den Entwurf des Fallberichts.",
    "rec.report.rationale": "Ein strukturierter Bericht macht Erkenntnisse, Beweisverknüpfungen und Grenzen prüfbar.",
})

CATALOGS["hi"].update({
    "report.limitations": "सीमाएँ",
    "limitations.ocr": "पाठ साक्ष्य जैसा है वैसा उद्धृत है; स्कैन की गई छवियों पर OCR नहीं लगाया गया।",
    "limitations.review": "उत्तर अनुक्रमित साक्ष्य से बनाए गए हैं और कानूनी उपयोग से पहले योग्य अन्वेषक द्वारा समीक्षा आवश्यक है।",
    "limitations.no_analytics": "संदिग्ध-पैटर्न विश्लेषण अभी नहीं चलाया गया।",
    "limitations.no_structured": "कोई संरचित घटना अनुक्रमित नहीं हुई; समयरेखा और वित्तीय विश्लेषण सीमित हैं।",
    "limitations.none": "अनुक्रमित साक्ष्य से कोई महत्वपूर्ण सीमा नहीं मिली।",
})
CATALOGS["es"].update({
    "report.limitations": "Limitaciones",
    "limitations.ocr": "La evidencia textual se cita tal cual; no se aplicó OCR a imágenes escaneadas.",
    "limitations.review": "Las respuestas se generan a partir de la evidencia indexada y deben ser revisadas por un investigador cualificado.",
    "limitations.no_analytics": "El análisis de patrones sospechosos aún no se ha ejecutado.",
    "limitations.no_structured": "No se indexaron eventos estructurados; el análisis temporal y financiero es limitado.",
    "limitations.none": "No se identificaron limitaciones materiales en la evidencia indexada.",
})
CATALOGS["fr"].update({
    "report.limitations": "Limites",
    "limitations.ocr": "Les preuves textuelles sont citées telles quelles ; l'OCR n'a pas été appliqué aux images scannées.",
    "limitations.review": "Les réponses sont générées à partir des preuves indexées et doivent être revues par un enquêteur qualifié.",
    "limitations.no_analytics": "L'analyse des schémas suspects n'a pas encore été exécutée.",
    "limitations.no_structured": "Aucun événement structuré n'a été indexé ; l'analyse temporelle et financière est limitée.",
    "limitations.none": "Aucune limite matérielle n'a été identifiée dans les preuves indexées.",
})
CATALOGS["de"].update({
    "report.limitations": "Grenzen",
    "limitations.ocr": "Textbeweise werden unverändert zitiert; auf gescannte Bilder wurde keine OCR angewendet.",
    "limitations.review": "Antworten werden aus den indexierten Beweisen erzeugt und müssen von einer qualifizierten Ermittlungsperson geprüft werden.",
    "limitations.no_analytics": "Die Analyse verdächtiger Muster wurde noch nicht ausgeführt.",
    "limitations.no_structured": "Es wurden keine strukturierten Ereignisse indexiert; Zeitlinien- und Finanzanalyse sind begrenzt.",
    "limitations.none": "Es wurden keine wesentlichen Einschränkungen aus den indexierten Beweisen festgestellt.",
})
CATALOGS["pt"].update({
    "report.limitations": "Limitações",
    "limitations.ocr": "As evidências textuais são citadas como estão; o OCR não foi aplicado a imagens digitalizadas.",
    "limitations.review": "As respostas são geradas a partir das evidências indexadas e devem ser revisadas por um investigador qualificado.",
    "limitations.no_analytics": "A análise de padrões suspeitos ainda não foi executada.",
    "limitations.no_structured": "Nenhum evento estruturado foi indexado; a análise temporal e financeira é limitada.",
    "limitations.none": "Nenhuma limitação material foi identificada nas evidências indexadas.",
})

CATALOGS["pt"].update({
    "rec.gap.bank.action": "Obtenha registros bancários/de transações completos para o período do caso.",
    "rec.gap.bank.rationale": "Não há evidência bancária indexada; fluxos financeiros, estruturação e camadas não podem ser verificados.",
    "rec.gap.cdr.action": "Obtenha registros de chamadas (CDR) para os números relevantes.",
    "rec.gap.cdr.rationale": "Não há evidência CDR indexada; a colocalização chamada-transferência e as redes de comunicação não podem ser testadas.",
    "rec.gap.social.action": "Colete publicações e identificadores de contas de redes sociais para os perfis relevantes.",
    "rec.gap.social.rationale": "Não há evidência social indexada; contexto de fontes abertas e vínculos fracos estão ausentes.",
    "rec.gap.email.action": "Colete exportações de e-mail e chat das contas relevantes.",
    "rec.gap.email.rationale": "Não há evidência de e-mail/chat indexada; trilhas de intenção, coordenação e instrução podem estar ausentes.",
    "rec.gap.timestamps.action": "Obtenha registros datados para as evidências sem data.",
    "rec.gap.timestamps.rationale": "Algumas evidências não têm data e não podem ser colocadas na linha do tempo.",
    "rec.gap.volume.action": "Colete um período maior ou mais registros para as fontes com pouco volume.",
    "rec.gap.volume.rationale": "O volume por entidade está abaixo do limite de confiabilidade; os achados estatísticos são fracos.",
    "rec.gap.unresolved.action": "Resolva identificadores não correspondidos (contas, telefones, perfis) para pessoas.",
    "rec.gap.unresolved.rationale": "Identificadores não resolvidos impedem a ligação entre fontes e ocultam relações.",
    "rec.interview.action": "Entreviste {name} ({entity_id}) e registre um depoimento.",
    "rec.interview.rationale": "Esta entidade tem risco {risk} com {n_findings} achado(s) vinculado(s).",
    "rec.records.action": "Solicite registros completos de conta/telefone para {identifiers}.",
    "rec.records.rationale": "Registros completos são necessários para verificar a atividade sinalizada de {name}.",
    "rec.structuring.action": "Revise todas as transações na faixa de 88–100% abaixo do limite de {threshold} para {name}.",
    "rec.structuring.rationale": "O padrão de estruturação sugere evasão deliberada dos limites de reporte.",
    "rec.colocation.action": "Obtenha dump de torre / CCTV / registros de acesso de {location} em torno de {timestamp}.",
    "rec.colocation.rationale": "Uma chamada/publicação foi seguida por uma transferência para a mesma parte dentro da janela de tolerância.",
    "rec.chain.action": "Mapeie o fluxo multi-salto e solicite KYC das contas beneficiárias na cadeia.",
    "rec.chain.rationale": "A camada rápida por contas intermediárias é um indicador clássico de colocação/camadas.",
    "rec.fan.action": "Verifique as {n} contrapartes de {name} quanto a comportamento de conta laranja.",
    "rec.fan.rationale": "A estrutura fan-in/fan-out é consistente com coleta e dispersão.",
    "rec.dormancy.action": "Verifique tomada de conta, eventos de vida ou prisão em torno da data de reativação de {name}.",
    "rec.dormancy.rationale": "Uma conta dormente foi reativada repentinamente e moveu fundos.",
    "rec.corroborate.action": "Corrobore o achado de {kind} de {name} com uma fonte independente.",
    "rec.corroborate.rationale": "O achado baseia-se em um único sinal; a corroboração fortalece o caso.",
    "rec.integrity.action": "Verifique o hash SHA-256 e o registro de cadeia de custódia da evidência sinalizada.",
    "rec.integrity.rationale": "A integridade da evidência deve ser documentada antes do uso em um relatório.",
    "rec.timeline_gap.action": "Solicite registros cobrindo a lacuna da linha do tempo de {start} a {end}.",
    "rec.timeline_gap.rationale": "Há uma lacuna nas evidências datadas que pode ocultar atividade relevante.",
    "rec.report.action": "Gere e revise o relatório preliminar do caso.",
    "rec.report.rationale": "Um relatório estruturado torna os achados, vínculos de evidência e limitações revisáveis.",
})


# ---------------------------------------------------------------------------
# lookup + language detection
# ---------------------------------------------------------------------------
def normalize_language(lang: Optional[str]) -> str:
    """Return a supported language code, defaulting to English."""
    code = (lang or "en").lower().strip()
    if code in ("en-us", "en-gb", "english"):
        return "en"
    if code in ("hindi", "hi-in"):
        return "hi"
    if code in ("spanish", "es-es", "es-mx"):
        return "es"
    if code in ("french", "fr-fr"):
        return "fr"
    if code in ("german", "de-de"):
        return "de"
    if code in ("portuguese", "pt-br", "pt-pt"):
        return "pt"
    return code if code in CATALOGS else "en"


def t(key: str, lang: str = "en", **kwargs) -> Any:
    """Translate ``key`` into ``lang`` with English fallback.

    Missing keys return the key itself, which makes UI gaps visible without
    raising.  Formatting errors return the unformatted template.
    """
    lang = normalize_language(lang)
    catalog = CATALOGS.get(lang, {})
    value = catalog.get(key, _EN.get(key, key))
    if isinstance(value, str) and kwargs:
        try:
            return value.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return value
    return value


def language_name(lang: str) -> str:
    return LANGUAGE_NAMES.get(normalize_language(lang), "English")


# ---------------------------------------------------------------------------
# stopwords / intent lexicons / month names
# ---------------------------------------------------------------------------
STOPWORDS: Dict[str, set] = {
    "en": set("""a an and are as at be been being between by can could did do does for from
        had has have he her hers him his how i if in into is it its me my of on or our
        she should so than that the their them then there these they this those to was
        we were what when where which who whom why will with would you your show find
        list give tell get all any please about over under above below more less most
        least last past next recent within during after before since until top first
        evidence case please query search look""".split()),
    "hi": set("""और के की का में से को पर यह वह ये वे कि क्या कौन कब कहाँ क्यों कैसे है हैं था थे
        हो गया गई दिखाएं दिखाओ खोजें बताओ सभी कोई पिछले दिन महीने साल केस साक्ष्य जांच
        कृपया बारे में से अधिक कम""".split()),
    "es": set("""el la los las un una unos unas de del que y o en por para con sin sobre
        entre es son fue fueron ser estar hay ha han como cuando donde quien quienes
        cual cuales que mostrar buscar lista dame dime todos todas cualquier por favor
        ultimos ultimas pasado pasada dias meses anos caso evidencia mas menos sobre
        bajo encima debajo desde hasta""".split()),
    "fr": set("""le la les un une des de du que et ou en pour par avec sans sur entre
        est sont etait etaient etre avoir a ont comme quand ou qui quoi dont quel
        quelle quels quelles montrer chercher liste donne dis tous toutes tout toute
        dernier derniere derniers dernieres jours mois ans cas preuve plus moins sur
        sous depuis jusqu""".split()),
    "de": set("""der die das ein eine einen einem einer des dem den und oder in im auf
        mit ohne von zu zum zur fur für ist sind war waren sein haben hat haben wie
        wann wo wer wen was welche welcher welches zeigen suche liste gib sag alle
        jede jeder jedes bitte letzte letzten tag tage monat monate jahr jahre fall
        beweis mehr weniger uber über unter seit bis""".split()),
    "pt": set("""o a os as um uma uns umas de do da dos das que e ou em por para com sem
        sobre entre e sao foi foram ser estar ha hao como quando onde quem qual quais
        mostrar buscar lista me diga todos todas qualquer por favor ultimos ultimas
        passado passada dias meses anos caso evidencia mais menos sobre sob desde ate""".split()),
}

INTENT_LEXICON: Dict[str, Dict[str, List[str]]] = {
    "summarize": {
        "en": ["summarize", "summary", "overview", "brief", "case summary", "recap"],
        "hi": ["सारांश", "सार", "संक्षेप", "केस सारांश"],
        "es": ["resumir", "resumen", "síntesis", "panorama"],
        "fr": ["résumer", "résumé", "synthèse", "aperçu"],
        "de": ["zusammenfassen", "zusammenfassung", "überblick"],
        "pt": ["resumir", "resumo", "visão geral", "síntese"],
    },
    "anomaly": {
        "en": ["suspicious", "anomaly", "anomalies", "unusual", "fraud", "red flag",
               "red flags", "odd", "strange", "risk", "risky"],
        "hi": ["संदिग्ध", "असामान्य", "धोखाधड़ी", "जोखिम", "खतरा", "अजीब"],
        "es": ["sospechoso", "sospechosa", "sospechosos", "sospechosas",
               "anomalía", "anomalías", "inusual", "fraude", "riesgo", "extraño"],
        "fr": ["suspect", "suspecte", "suspects", "suspectes", "anomalie",
               "anomalies", "inhabituel", "fraude", "risque", "étrange"],
        "de": ["verdächtig", "anomalie", "anomalien", "ungewöhnlich", "betrug",
               "risiko", "seltsam"],
        "pt": ["suspeito", "suspeita", "anomalia", "anomalias", "incomum",
               "fraude", "risco", "estranho"],
    },
    "recommend": {
        "en": ["recommend", "recommendation", "next step", "next steps", "lead",
               "leads", "what should i", "what to do", "investigate next", "next"],
        "hi": ["सिफ़ारिश", "सुझाव", "अगला कदम", "अगले कदम", "जांच क्या", "सुराग"],
        "es": ["recomendar", "recomendación", "siguiente paso", "próximos pasos",
               "pista", "qué debo", "investigar después"],
        "fr": ["recommander", "recommandation", "prochaine étape", "prochaines étapes",
               "piste", "que dois-je", "enquêter ensuite"],
        "de": ["empfehlen", "empfehlung", "nächster schritt", "nächste schritte",
               "hinweis", "was soll ich", "als nächstes"],
        "pt": ["recomendar", "recomendação", "próximo passo", "próximos passos",
               "pista", "o que devo", "investigar a seguir"],
    },
    "diary": {
        "en": ["diary", "case diary", "log", "activity log", "journal", "chronology"],
        "hi": ["डायरी", "केस डायरी", "लॉग", "जर्नल"],
        "es": ["diario", "bitácora", "registro", "journal"],
        "fr": ["journal", "journal de l'affaire", "registre"],
        "de": ["tagebuch", "falltagebuch", "protokoll", "journal"],
        "pt": ["diário", "diário do caso", "registro", "log"],
    },
    "report": {
        "en": ["report", "draft report", "case report", "generate report", "export report"],
        "hi": ["रिपोर्ट", "केस रिपोर्ट", "ड्राफ्ट रिपोर्ट"],
        "es": ["informe", "reporte", "borrador", "informe del caso"],
        "fr": ["rapport", "projet de rapport", "rapport d'affaire"],
        "de": ["bericht", "fallbericht", "berichtsentwurf"],
        "pt": ["relatório", "relatório do caso", "rascunho"],
    },
    "timeline": {
        "en": ["timeline", "chronology", "when did", "what happened", "sequence",
               "order of events"],
        "hi": ["समयरेखा", "कालक्रम", "कब हुआ", "क्या हुआ", "घटनाओं का क्रम"],
        "es": ["línea de tiempo", "cronología", "qué pasó", "cuándo"],
        "fr": ["chronologie", "ligne du temps", "que s'est-il passé", "quand"],
        "de": ["zeitlinie", "chronologie", "was ist passiert", "wann"],
        "pt": ["linha do tempo", "cronologia", "o que aconteceu", "quando"],
    },
    "relationship": {
        "en": ["connection", "link", "linked", "related", "relationship", "between",
               "associate", "associated"],
        "hi": ["संबंध", "जुड़ाव", "कनेक्शन", "के बीच"],
        "es": ["conexión", "vínculo", "relación", "relacionado", "entre"],
        "fr": ["lien", "connexion", "relation", "lié", "entre"],
        "de": ["verbindung", "beziehung", "verknüpft", "zwischen"],
        "pt": ["conexão", "vínculo", "relação", "relacionado", "entre"],
    },
    "correlation": {
        "en": ["right before", "just before", "before the transfer", "after the call",
               "call then transfer", "correlate", "co-location", "colocation",
               "same time", "around the transfer"],
        "hi": ["ठीक पहले", "ट्रांसफर से पहले", "कॉल के बाद", "सहसंबंध", "उसी समय"],
        "es": ["justo antes", "antes de la transferencia", "después de la llamada",
               "correlacionar", "misma hora"],
        "fr": ["juste avant", "avant le virement", "après l'appel", "corréler",
               "même moment"],
        "de": ["kurz vor", "vor der überweisung", "nach dem anruf", "korrelieren",
               "gleiche zeit"],
        "pt": ["logo antes", "antes da transferência", "depois da chamada",
               "correlacionar", "mesmo horário"],
    },
    "aggregate": {
        "en": ["how much", "how many", "total", "sum", "count", "average", "largest",
               "biggest", "maximum", "minimum"],
        "hi": ["कितना", "कितने", "कुल", "योग", "संख्या", "औसत", "सबसे बड़ा"],
        "es": ["cuánto", "cuántos", "total", "suma", "recuento", "promedio", "mayor"],
        "fr": ["combien", "total", "somme", "nombre", "moyenne", "plus grand"],
        "de": ["wie viel", "wie viele", "summe", "anzahl", "durchschnitt", "größte"],
        "pt": ["quanto", "quantos", "total", "soma", "contagem", "média", "maior"],
    },
    "entity_profile": {
        "en": ["who is", "tell me about", "profile", "dossier", "background",
               "information about"],
        "hi": ["कौन है", "के बारे में", "प्रोफ़ाइल", "जानकारी"],
        "es": ["quién es", "perfil", "información sobre", "informaciones sobre"],
        "fr": ["qui est", "profil", "informations sur", "à propos de"],
        "de": ["wer ist", "profil", "informationen über"],
        "pt": ["quem é", "perfil", "informações sobre"],
    },
    "evidence_inventory": {
        "en": ["what evidence", "evidence inventory", "evidence list", "files do we have",
               "what files", "chain of custody"],
        "hi": ["कौन साक्ष्य", "साक्ष्य सूची", "कौन सी फाइलें", "कस्टडी"],
        "es": ["qué evidencia", "inventario de evidencia", "lista de evidencia",
               "qué archivos", "cadena de custodia"],
        "fr": ["quelles preuves", "inventaire des preuves", "liste des preuves",
               "quels fichiers", "chaîne de custody"],
        "de": ["welche beweeise", "beweisbestand", "beweisliste", "welche dateien",
               "beweiskette"],
        "pt": ["quais evidências", "inventário de evidências", "lista de evidências",
               "quais arquivos", "cadeia de custódia"],
    },
    "entity_list": {
        "en": ["list entities", "all entities", "people", "persons", "suspects",
               "who is in the case"],
        "hi": ["संस्थाएँ", "सभी लोग", "व्यक्ति", "कौन कौन"],
        "es": ["lista de entidades", "todas las entidades", "personas", "sospechosos"],
        "fr": ["liste des entités", "toutes les entités", "personnes", "suspects"],
        "de": ["entitäten auflisten", "alle entitäten", "personen", "verdächtige"],
        "pt": ["listar entidades", "todas as entidades", "pessoas", "suspeitos"],
    },
    "search": {
        "en": ["show", "find", "search", "list", "display", "get", "retrieve",
               "look for", "give me", "filter", "which", "who", "what", "where"],
        "hi": ["दिखाएं", "दिखाओ", "खोजें", "सूची", "ढूंढें", "कौन", "क्या", "कहाँ"],
        "es": ["mostrar", "buscar", "lista", "encontrar", "filtrar", "cuál", "quién",
               "qué", "dónde"],
        "fr": ["montrer", "chercher", "liste", "trouver", "filtrer", "quel", "qui",
               "quoi", "où"],
        "de": ["zeigen", "zeige", "such", "suchen", "liste", "finden", "filtern",
               "welche", "wer", "was", "wo"],
        "pt": ["mostrar", "buscar", "lista", "encontrar", "filtrar", "qual", "quem",
               "o que", "onde"],
    },
}

# source vocabulary
SOURCE_WORDS: Dict[str, Dict[str, List[str]]] = {
    "bank": {
        "en": ["bank", "transaction", "transactions", "transfer", "transfers", "payment",
               "payments", "account", "wire", "deposit", "withdrawal", "credit",
               "debit", "statement", "amount", "money"],
        "hi": ["बैंक", "लेनदेन", "ट्रांसफर", "भुगतान", "खाता", "राशि", "पैसा"],
        "es": ["banco", "transacción", "transacciones", "transferencia", "pago",
               "cuenta", "depósito", "retiro", "crédito", "débito", "monto", "dinero"],
        "fr": ["banque", "transaction", "transactions", "virement", "paiement",
               "compte", "dépôt", "retrait", "crédit", "débit", "montant", "argent"],
        "de": ["bank", "transaktion", "transaktionen", "überweisung",
               "überweisungen", "zahlung", "zahlungen", "konto", "konten",
               "einzahlung", "abhebung", "guthaben", "lastschrift",
               "betrag", "beträge", "geld"],
        "pt": ["banco", "transação", "transações", "transferência", "pagamento",
               "conta", "depósito", "saque", "crédito", "débito", "valor", "dinheiro"],
    },
    "cdr": {
        "en": ["call", "calls", "cdr", "phone", "telephone", "msisdn", "tower",
               "duration", "sms", "dialed", "called"],
        "hi": ["कॉल", "फोन", "सीडीआर", "टावर", "अवधि", "एसएमएस"],
        "es": ["llamada", "llamadas", "cdr", "teléfono", "torre", "duración", "sms"],
        "fr": ["appel", "appels", "cdr", "téléphone", "antenne", "durée", "sms"],
        "de": ["anruf", "anrufe", "cdr", "telefon", "mast", "dauer", "sms"],
        "pt": ["chamada", "chamadas", "cdr", "telefone", "torre", "duração", "sms"],
    },
    "social": {
        "en": ["social", "post", "posts", "handle", "hashtag", "mention", "mentions",
               "twitter", "facebook", "instagram", "telegram", "whatsapp"],
        "hi": ["सोशल", "पोस्ट", "हैशटैग", "मेंशन", "ट्विटर", "फेसबुक", "इंस्टाग्राम"],
        "es": ["social", "publicación", "publicaciones", "hashtag", "mención",
               "twitter", "facebook", "instagram"],
        "fr": ["social", "publication", "publications", "hashtag", "mention",
               "twitter", "facebook", "instagram"],
        "de": ["sozial", "beitrag", "beiträge", "hashtag", "erwähnung", "twitter",
               "facebook", "instagram"],
        "pt": ["social", "publicação", "publicações", "hashtag", "menção", "twitter",
               "facebook", "instagram"],
    },
    "email": {
        "en": ["email", "emails", "mail", "inbox", "message", "messages", "correspondence"],
        "hi": ["ईमेल", "मेल", "इनबॉक्स", "संदेश", "पत्राचार"],
        "es": ["correo", "correos", "email", "buzón", "mensaje", "mensajes"],
        "fr": ["courriel", "courriels", "email", "boîte", "message", "messages"],
        "de": ["e-mail", "email", "posteingang", "nachricht", "nachrichten"],
        "pt": ["email", "e-mail", "caixa de entrada", "mensagem", "mensagens"],
    },
    "chat": {
        "en": ["chat", "chats", "conversation", "conversations", "whatsapp", "telegram",
               "signal", "messaging"],
        "hi": ["चैट", "बातचीत", "व्हाट्सएप", "टेलीग्राम"],
        "es": ["chat", "conversación", "conversaciones", "whatsapp", "telegram"],
        "fr": ["chat", "conversation", "conversations", "whatsapp", "telegram"],
        "de": ["chat", "chats", "unterhaltung", "unterhaltungen", "whatsapp", "telegram"],
        "pt": ["chat", "conversa", "conversas", "whatsapp", "telegram"],
    },
    "document": {
        "en": ["document", "documents", "pdf", "report", "note", "notes", "file", "files"],
        "hi": ["दस्तावेज़", "पीडीएफ", "रिपोर्ट", "नोट", "फाइल"],
        "es": ["documento", "documentos", "pdf", "informe", "nota", "archivo"],
        "fr": ["document", "documents", "pdf", "rapport", "note", "fichier"],
        "de": ["dokument", "dokumente", "pdf", "bericht", "notiz", "datei"],
        "pt": ["documento", "documentos", "pdf", "relatório", "nota", "arquivo"],
    },
    "image": {
        "en": ["image", "images", "photo", "photos", "picture", "exif", "camera"],
        "hi": ["छवि", "तस्वीर", "फोटो", "कैमरा"],
        "es": ["imagen", "imágenes", "foto", "fotos", "cámara"],
        "fr": ["image", "images", "photo", "photos", "appareil"],
        "de": ["bild", "bilder", "foto", "fotos", "kamera"],
        "pt": ["imagem", "imagens", "foto", "fotos", "câmera"],
    },
}

MONTHS: Dict[str, List[str]] = {
    "en": ["january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december"],
    "es": ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
           "agosto", "septiembre", "octubre", "noviembre", "diciembre"],
    "fr": ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
           "août", "septembre", "octobre", "novembre", "décembre"],
    "de": ["januar", "februar", "märz", "april", "mai", "juni", "juli",
           "august", "september", "oktober", "november", "dezember"],
    "pt": ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho",
           "agosto", "setembro", "outubro", "novembro", "dezembro"],
    "hi": ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई",
           "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"],
}

RELATIVE_WORDS: Dict[str, Dict[str, List[str]]] = {
    "last": {
        "en": ["last", "past", "previous", "recent"],
        "hi": ["पिछले", "पिछला", "पिछली", "हाल के"],
        "es": ["último", "última", "últimos", "últimas", "pasado", "pasada"],
        "fr": ["dernier", "dernière", "derniers", "dernières", "passé", "passée"],
        "de": ["letzte", "letzten", "letzter", "vergangene", "vergangenen"],
        "pt": ["último", "última", "últimos", "últimas", "passado", "passada"],
    },
    "today": {
        "en": ["today"], "hi": ["आज"], "es": ["hoy"], "fr": ["aujourd'hui"],
        "de": ["heute"], "pt": ["hoje"],
    },
    "yesterday": {
        "en": ["yesterday"], "hi": ["कल"], "es": ["ayer"], "fr": ["hier"],
        "de": ["gestern"], "pt": ["ontem"],
    },
    "this_week": {
        "en": ["this week"], "hi": ["इस सप्ताह"], "es": ["esta semana"],
        "fr": ["cette semaine"], "de": ["diese woche"], "pt": ["esta semana"],
    },
    "this_month": {
        "en": ["this month"], "hi": ["इस महीने"], "es": ["este mes"],
        "fr": ["ce mois"], "de": ["diesen monat"], "pt": ["este mês"],
    },
    "last_week": {
        "en": ["last week"], "hi": ["पिछले सप्ताह"], "es": ["la semana pasada"],
        "fr": ["la semaine dernière"], "de": ["letzte woche"], "pt": ["semana passada"],
    },
    "last_month": {
        "en": ["last month"], "hi": ["पिछले महीने"], "es": ["el mes pasado"],
        "fr": ["le mois dernier"], "de": ["letzten monat"], "pt": ["mês passado"],
    },
}


# ---------------------------------------------------------------------------
# language detection
# ---------------------------------------------------------------------------
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_ARABIC = re.compile(r"[\u0600-\u06FF]")
_CYRILLIC = re.compile(r"[\u0400-\u04FF]")
_CJK = re.compile(r"[\u4E00-\u9FFF]")


def detect_language(text: str) -> str:
    """Best-effort language detection for supported languages.

    Script detection wins for Devanagari; otherwise stopword overlap plus
    language-specific characters decide.  Ambiguous text falls back to English.
    """
    if not text:
        return "en"
    if _DEVANAGARI.search(text):
        return "hi"
    if _ARABIC.search(text) or _CYRILLIC.search(text) or _CJK.search(text):
        return "en"

    low = text.lower()
    tokens = re.findall(r"[^\W\d_]+", low, flags=re.UNICODE)
    token_set = set(tokens)
    scores: Dict[str, float] = {}
    for lang, words in STOPWORDS.items():
        if not words:
            continue
        hits = len(token_set & words)
        scores[lang] = float(hits)
    # language-specific characters
    if any(ch in low for ch in "ñ¿¡"):
        scores["es"] = scores.get("es", 0.0) + 2.0
    if any(ch in low for ch in "ßäöü"):
        scores["de"] = scores.get("de", 0.0) + 1.5
    if any(ch in low for ch in "çàèêëîïôûùœ"):
        scores["fr"] = scores.get("fr", 0.0) + 1.5
    if any(ch in low for ch in "ãõ"):
        scores["pt"] = scores.get("pt", 0.0) + 1.5

    # disambiguating vocabulary (es/pt share many stopwords)
    def _has_word(words):
        return any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", low) for w in words)

    if _has_word(["neste", "nesta", "não", "nao", "são", "sao", "transações",
                  "transacoes", "suspeitas", "suspeito", "suspeita", "você",
                  "voce", "também", "tambem", "pelo", "pela"]):
        scores["pt"] = scores.get("pt", 0.0) + 2.5
    if _has_word(["el", "los", "las", "una", "está", "qué", "cuáles",
                  "transacciones", "sospechosas", "sospechoso", "evidencia"]):
        scores["es"] = scores.get("es", 0.0) + 2.0
    if _has_word(["les", "des", "une", "est", "sont", "quelles", "suspectes",
                  "pour", "avec", "dans"]):
        scores["fr"] = scores.get("fr", 0.0) + 1.5
    if _has_word(["der", "die", "das", "und", "über", "transaktionen",
                  "verdächtig", "zeige", "welche", "betrag"]):
        scores["de"] = scores.get("de", 0.0) + 1.5
    if not scores or max(scores.values()) < 1:
        return "en"
    best = max(scores.items(), key=lambda kv: kv[1])
    return best[0]


_LANGUAGE_REQUEST_PATTERNS: List[tuple] = [
    ("hi", ["in hindi", "hindi me", "हिंदी में", "हिन्दी में", "hindi mein"]),
    ("es", ["in spanish", "en español", "en espanol", "español", "espanol"]),
    ("fr", ["in french", "en français", "en francais", "français", "francais"]),
    ("de", ["in german", "auf deutsch", "deutsch"]),
    ("pt", ["in portuguese", "em português", "em portugues", "português", "portugues"]),
    ("en", ["in english", "auf englisch", "en anglais", "en inglés", "em inglês"]),
]


def detect_language_request(text: str) -> Optional[str]:
    """Return an explicit language request ("answer in Hindi") if present."""
    low = (text or "").lower()
    for lang, patterns in _LANGUAGE_REQUEST_PATTERNS:
        if any(p in low for p in patterns):
            return lang
    return None


def strip_language_request(text: str) -> str:
    """Remove the explicit language phrase so it does not pollute keywords."""
    out = text
    for _lang, patterns in _LANGUAGE_REQUEST_PATTERNS:
        for p in patterns:
            out = re.sub(re.escape(p), " ", out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


# ---------------------------------------------------------------------------
# normalization helpers used by the parser
# ---------------------------------------------------------------------------
def strip_accents(text: str) -> str:
    """Unicode-normalize and drop combining marks (for fuzzy matching only)."""
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    return re.sub(r"\s+", " ", strip_accents((text or "").lower())).strip()


def all_phrases(lang: str, lexicon: Dict[str, Dict[str, List[str]]]) -> Iterable[str]:
    """Yield every phrase in a per-language lexicon (English included)."""
    for _intent, by_lang in lexicon.items():
        for phrase in by_lang.get(lang, []):
            yield phrase
        for phrase in by_lang.get("en", []):
            yield phrase

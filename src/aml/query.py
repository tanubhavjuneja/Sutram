"""Multilingual natural-language query parser.

Turns an investigator's question into a deterministic :class:`QueryPlan`:

    intent, entities, sources, event types, time range, amount range,
    direction, locations, keywords, limit, ambiguity notes

The parser is deliberately rule-based and explainable.  It does not require a
network call or an LLM, which keeps the assistant usable offline and makes the
retrieval behaviour reproducible.  An optional LLM can later rephrase the
answer, but it never controls the filters.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:  # rapidfuzz is already a project dependency; keep a stdlib fallback
    from rapidfuzz import fuzz, process
except Exception:  # pragma: no cover - optional
    fuzz = None
    process = None

from .i18n import (
    INTENT_LEXICON,
    MONTHS,
    RELATIVE_WORDS,
    SOURCE_WORDS,
    STOPWORDS,
    detect_language,
    detect_language_request,
    normalize_text,
    strip_language_request,
)
from .case import AmountRange, QueryPlan, TimeRange

# intent priority when scores tie (higher wins)
_INTENT_PRIORITY = {
    "report": 100,
    "generate_image": 97,
    "diary": 95,
    "plot": 94,
    "run_analysis": 93,
    "recheck": 92,
    "deep_dive": 91,
    "recommend": 90,
    "summarize": 85,
    "anomaly": 80,
    "correlation": 75,
    "relationship": 70,
    "timeline": 65,
    "aggregate": 60,
    "entity_profile": 55,
    "evidence_inventory": 50,
    "entity_list": 45,
    "search": 10,
    "help": 0,
}

_UNIT_DAYS = {
    # English
    "day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30,
    "year": 365, "years": 365,
    # Spanish
    "día": 1, "días": 1, "dia": 1, "dias": 1, "semana": 7, "semanas": 7,
    "mes": 30, "meses": 30, "año": 365, "años": 365, "ano": 365, "anos": 365,
    # French
    "jour": 1, "jours": 1, "semaine": 7, "semaines": 7, "mois": 30,
    "an": 365, "ans": 365,
    # German
    "tag": 1, "tage": 1, "woche": 7, "wochen": 7, "monat": 30, "monate": 30,
    "jahr": 365, "jahre": 365,
    # Portuguese
    "dia": 1, "dias": 1, "mês": 30, "meses": 30, "ano": 365, "anos": 365,
    # Hindi
    "दिन": 1, "सप्ताह": 7, "महीना": 30, "महीने": 30, "महीन": 30,
    "साल": 365, "वर्ष": 365,
}

_COMPARISON_ABOVE = [
    "above", "over", "more than", "greater than", "at least", "minimum", "min",
    ">", "after", "exceeding", "exceeds", "upwards of",
    "más de", "mas de", "mayor que", "superior a", "por encima de",
    "plus de", "supérieur à", "superieur a", "au-dessus de",
    "über", "uber", "ueber", "mehr als", "größer als", "groesser als",
    "acima de", "mais de", "maior que",
    "से अधिक", "से ज्यादा", "से ऊपर", "अधिक",
]
_COMPARISON_BELOW = [
    "below", "under", "less than", "at most", "maximum", "max", "<", "before",
    "up to", "not exceeding",
    "menos de", "menor que", "inferior a", "por debajo de", "hasta",
    "moins de", "inférieur à", "inferieur a", "en dessous de", "jusqu'à",
    "unter", "weniger als", "kleiner als", "bis zu",
    "abaixo de", "menos de", "menor que", "até",
    "से कम", "से कमतर", "नीचे", "तक",
]
_COMPARISON_BETWEEN = [
    "between", "from", "range", "ranging",
    "entre", "desde", "de", "rango",
    "entre", "de", "plage",
    "zwischen", "von", "bereich",
    "entre", "de", "intervalo",
    "के बीच", "से", "रेंज",
]

_NUM = r"(?:\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)"
_MONEY = rf"(?:[$€£₹]\s*)?({_NUM})\s*([kKmM])?"
_DATE_ISO = re.compile(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b")
_DATE_DMY = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](20\d{2})\b")
_YEAR_ONLY = re.compile(r"\b(20\d{2})\b")
_PHONE_RE = re.compile(r"(?<!\d)\+?\d{10,15}(?!\d)")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_HANDLE_RE = re.compile(r"@[\w.]+")
_ACCT_RE = re.compile(r"\bACCT[-_]?[A-Za-z0-9]{3,}\b", re.IGNORECASE)
_LOCATION_RE = re.compile(r"\b(?:T|TOWER|BRANCH|LAT|LOC|CELL)[-_]?\d+\b", re.IGNORECASE)
_TOP_RE = re.compile(r"\b(?:top|first|limit|show|return|give me)\s+(\d{1,3})\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _contains(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    phrase = normalize_text(phrase)
    if not phrase:
        return False
    if re.search(r"\W", phrase):
        return phrase in text
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


def _contains_any(text: str, phrases: Iterable[str]) -> bool:
    return any(_contains(text, p) for p in phrases)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[^\W\d_]+", (text or "").lower(), flags=re.UNICODE)


def _parse_number(token: str, multiplier: str = "") -> Optional[float]:
    """Parse '10,000', '1.234,56', '10k', '1.5m' into a float."""
    token = (token or "").strip().replace(" ", "")
    token = re.sub(r"[$€£₹]", "", token)
    if not token:
        return None
    # European style: 1.234,56 -> 1234.56
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d{1,2})?", token):
        token = token.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?", token):
        token = token.replace(",", "")
    else:
        token = token.replace(",", ".")
    try:
        value = float(token)
    except ValueError:
        return None
    mult = (multiplier or "").lower()
    if mult == "k":
        value *= 1_000
    elif mult == "m":
        value *= 1_000_000
    return value


def _to_dt(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _day_start(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _day_end(dt: datetime) -> datetime:
    return dt.replace(hour=23, minute=59, second=59, microsecond=0)


def _month_bounds(year: int, month: int) -> Tuple[datetime, datetime]:
    first = datetime(year, month, 1, tzinfo=timezone.utc)
    last_day = calendar.monthrange(year, month)[1]
    last = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)
    return first, last


def _remove_spans(text: str, spans: List[Tuple[int, int]]) -> str:
    if not spans:
        return text
    chars = list(text)
    for start, end in spans:
        for i in range(max(0, start), min(len(chars), end)):
            chars[i] = " "
    return "".join(chars)


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
class QueryParser:
    def __init__(self, alias_to_entity: Optional[Dict[str, str]] = None,
                 entity_names: Optional[Dict[str, List[str]]] = None,
                 case_start=None, case_end=None, default_lang: str = "en",
                 known_locations: Optional[Sequence[str]] = None,
                 default_limit: int = 20):
        self.alias_to_entity = dict(alias_to_entity or {})
        self.entity_names = {k: list(v) for k, v in (entity_names or {}).items()}
        self.case_start = _to_dt(case_start)
        self.case_end = _to_dt(case_end)
        self.default_lang = default_lang or "en"
        self.known_locations = [str(x) for x in (known_locations or [])]
        self.default_limit = max(1, min(int(default_limit or 20), 100))
        self._alias_items = sorted(
            ((normalize_text(alias), eid) for alias, eid in self.alias_to_entity.items() if alias),
            key=lambda kv: -len(kv[0]),
        )
        self._name_items = sorted(
            ((normalize_text(name), eid) for eid, names in self.entity_names.items() for name in names if name),
            key=lambda kv: -len(kv[0]),
        )
        # given-name / surname index for short queries ("Viktor and Elena")
        self._given_names: Dict[str, List[str]] = {}
        for eid, names in self.entity_names.items():
            for name in names:
                for token in re.findall(r"[^\W\d_]+", name or "", flags=re.UNICODE):
                    key = normalize_text(token)
                    if len(key) >= 3:
                        bucket = self._given_names.setdefault(key, [])
                        if eid not in bucket:
                            bucket.append(eid)

    # -- public ------------------------------------------------------------
    def parse(self, text: str, lang: Optional[str] = None) -> QueryPlan:
        raw = (text or "").strip()
        requested = detect_language_request(raw)
        language = (lang or requested or detect_language(raw) or self.default_lang).lower()
        if language not in ("en", "hi", "es", "fr", "de", "pt"):
            language = "en"
        working = strip_language_request(raw) if requested else raw
        norm = normalize_text(working)

        intent, intent_notes = self._classify_intent(norm, language)
        entity_ids, entity_terms, ambiguous = self._extract_entities(working, norm, language)
        time_range = self._extract_time_range(working, norm, language)
        amount_range = self._extract_amount_range(norm)
        sources = self._extract_sources(norm, language)
        direction = self._extract_direction(norm)
        locations = self._extract_locations(working, norm)
        limit = self._extract_limit(norm)
        keywords = self._extract_keywords(
            working, norm, language, entity_terms, time_range, amount_range,
            sources, locations,
        )
        event_types = self._event_types_for_sources(sources)
        chart_kind = self._extract_chart_kind(norm, entity_ids)
        image_prompt = (self._extract_image_prompt(working)
                        if intent == "generate_image" else "")

        notes = list(intent_notes)
        if ambiguous:
            notes.append("ambiguous entity mention")
        if not entity_ids and not sources and not keywords and not time_range.start \
                and not amount_range.min and not amount_range.max:
            notes.append("no explicit filters; using case-wide search")
        if intent in ("entity_profile", "relationship", "correlation", "aggregate") and not entity_ids:
            notes.append("intent needs an entity but none was resolved")

        return QueryPlan(
            raw=raw,
            language=language,
            intent=intent,
            entity_terms=entity_terms,
            entity_ids=entity_ids,
            sources=sources,
            event_types=event_types,
            time_range=time_range,
            amount_range=amount_range,
            direction=direction,
            location_terms=locations,
            keywords=keywords,
            chart_kind=chart_kind,
            image_prompt=image_prompt,
            limit=limit,
            ambiguous=ambiguous,
            notes=notes,
        )

    # -- intent ------------------------------------------------------------
    def _classify_intent(self, norm: str, lang: str) -> Tuple[str, List[str]]:
        scores: Dict[str, float] = {}
        notes: List[str] = []
        for intent, by_lang in INTENT_LEXICON.items():
            score = 0.0
            seen: Set[str] = set()
            for phrase in list(by_lang.get(lang, [])) + list(by_lang.get("en", [])):
                p = normalize_text(phrase)
                if not p or p in seen:
                    continue
                seen.add(p)
                if _contains(norm, p):
                    score += 1.0 + 0.12 * len(p.split())
            if score:
                scores[intent] = score

        # Strong intents get a boost so a single explicit cue is not outvoted
        # by generic search words ("what", "show", "transactions").
        for intent, boost in (
            ("anomaly", 2.0), ("recommend", 1.5), ("summarize", 1.0),
            ("diary", 1.0), ("report", 1.0), ("timeline", 0.6),
            ("relationship", 0.6), ("entity_profile", 0.5),
            ("evidence_inventory", 0.6), ("entity_list", 0.4),
            ("plot", 2.0), ("run_analysis", 2.0), ("recheck", 2.0),
            ("deep_dive", 2.0), ("generate_image", 2.0),
        ):
            if scores.get(intent):
                scores[intent] += boost

        # explicit tool intents: these should beat generic search words
        if _contains_any(norm, ["generate an image", "create an image",
                                "make an image", "draw an image",
                                "draw a picture", "make a picture",
                                "generate a picture", "illustrate",
                                "text to image", "text-to-image",
                                "render an image", "generate a visual"]):
            scores["generate_image"] = scores.get("generate_image", 0.0) + 6.0
        if _contains_any(norm, ["plot", "chart", "graph", "visualize", "visualise",
                                "draw", "diagram", "map", "heatmap"]):
            scores["plot"] = scores.get("plot", 0.0) + 5.0
        if _contains_any(norm, ["rerun", "re-run", "run again", "run the analysis",
                                "run analysis", "refresh analysis", "reanalyze",
                                "re-analyze", "run the models", "run tests",
                                "run checks", "recompute"]):
            scores["run_analysis"] = scores.get("run_analysis", 0.0) + 5.0
        if _contains_any(norm, ["recheck", "re-check", "recalculate", "re-calculate",
                                "verify the numbers", "check the numbers",
                                "double-check", "double check"]):
            scores["recheck"] = scores.get("recheck", 0.0) + 5.0
        if _contains_any(norm, ["deep dive", "dig deeper", "analyze further",
                                "analyse further", "investigate further",
                                "run a model", "check for structuring",
                                "check for layering", "look deeper"]):
            scores["deep_dive"] = scores.get("deep_dive", 0.0) + 5.0

        # explicit priority rules (they encode product decisions, not just words)
        if _contains(norm, "report") and _contains_any(
                norm, ["draft", "generate", "create", "write", "export", "prepare"]):
            scores["report"] = scores.get("report", 0.0) + 6.0
        if _contains_any(norm, ["case diary", "activity log", "journal", "diary", "डायरी", "diario", "tagebuch"]):
            scores["diary"] = scores.get("diary", 0.0) + 5.0
        if _contains_any(norm, ["what should i", "what to do", "next step", "next steps",
                                "investigate next", "recommend", "lead", "leads",
                                "qué debo", "que dois-je", "was soll ich", "o que devo",
                                "अगला कदम", "सिफ़ारिश"]):
            scores["recommend"] = scores.get("recommend", 0.0) + 4.0
        if _contains_any(norm, ["right before", "just before", "before the transfer",
                                "after the call", "call then transfer", "correlate",
                                "colocation", "co-location", "same time",
                                "ठीक पहले", "justo antes", "juste avant", "kurz vor",
                                "logo antes"]):
            scores["correlation"] = scores.get("correlation", 0.0) + 4.0
        # "between A and B" is a strong relationship signal; the generic word
        # "between" alone is not (it also appears in amount ranges)
        has_pair_joiner = any(j in norm for j in (" and ", " और ", " y ", " et ", " und ", " e "))
        if has_pair_joiner and _contains_any(
                norm, ["connection", "relationship", "link", "related", "associate",
                       "संबंध", "जुड़ाव", "conexión", "vínculo", "relación",
                       "lien", "relation", "verbindung", "beziehung",
                       "conexão", "relação"]):
            scores["relationship"] = scores.get("relationship", 0.0) + 4.0
        if _contains_any(norm, ["what happened", "when did", "timeline", "chronology",
                                "order of events", "sequence of events",
                                "क्या हुआ", "कब हुआ", "समयरेखा",
                                "qué pasó", "cuándo", "cronología",
                                "que s'est-il passé", "chronologie",
                                "was ist passiert", "wann", "zeitlinie",
                                "o que aconteceu", "quando", "linha do tempo"]):
            scores["timeline"] = scores.get("timeline", 0.0) + 4.0
        if _contains_any(norm, ["who is", "tell me about", "profile", "dossier",
                                "background on", "information about",
                                "कौन है", "के बारे में", "प्रोफ़ाइल",
                                "quién es", "perfil", "qui est", "profil",
                                "wer ist", "profil", "quem é"]):
            scores["entity_profile"] = scores.get("entity_profile", 0.0) + 3.0
        if _contains_any(norm, ["what evidence", "evidence inventory", "evidence list",
                                "files do we have", "what files", "chain of custody",
                                "कौन साक्ष्य", "साक्ष्य सूची",
                                "qué evidencia", "inventario de evidencia",
                                "quelles preuves", "inventaire des preuves",
                                "welche beweeise", "beweisbestand",
                                "quais evidências", "inventário de evidências"]):
            scores["evidence_inventory"] = scores.get("evidence_inventory", 0.0) + 4.0
        if _contains_any(norm, ["list entities", "all entities", "all people", "persons",
                                "suspects", "who is in the case",
                                "संस्थाएँ", "सभी लोग", "संदिग्ध",
                                "todas las entidades", "personas", "sospechosos",
                                "toutes les entités", "personnes", "suspects",
                                "alle entitäten", "personen", "verdächtige",
                                "todas as entidades", "pessoas", "suspeitos"]):
            scores["entity_list"] = scores.get("entity_list", 0.0) + 3.0
        if _contains_any(norm, ["help", "what can you do", "capabilities", "how do i use",
                                "मदद", "क्या कर सकते", "ayuda", "aide", "hilfe", "ajuda"]):
            scores["help"] = scores.get("help", 0.0) + 3.0

        if not scores:
            # A question with no intent word is still a search if it has any
            # substantive token; a greeting becomes help.
            if _contains_any(norm, ["hello", "hi", "hey", "नमस्ते", "hola", "bonjour",
                                    "hallo", "olá", "ola"]):
                return "help", notes
            return "search", notes

        # score + priority tie-break
        best = max(
            scores.items(),
            key=lambda kv: (round(kv[1], 4), _INTENT_PRIORITY.get(kv[0], 0)),
        )[0]
        if best == "search" and scores.get("search", 0.0) < 1.0:
            # avoid classifying a lone "who/what" as a search when a stronger
            # specific intent exists (handled above); otherwise default search
            pass
        return best, notes

    # -- entities ----------------------------------------------------------
    def _extract_entities(self, raw: str, norm: str, lang: str) -> Tuple[List[str], List[str], List[dict]]:
        ids: List[str] = []
        terms: List[str] = []
        ambiguous: List[dict] = []
        spans: List[Tuple[int, int]] = []

        def add(eid: str, term: str) -> None:
            if eid and eid not in ids:
                ids.append(eid)
            if term and term not in terms:
                terms.append(term)

        # 1) exact alias matches (names, accounts, phones, handles, emails)
        for alias, eid in self._alias_items:
            if len(alias) < 2:
                continue
            pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
            for match in re.finditer(pattern, norm):
                add(eid, match.group(0))
                spans.append((match.start(), match.end()))
                break  # one mention per alias is enough for the plan

        # 2) identifier-shaped tokens (in case the alias index missed them)
        for regex in (_PHONE_RE, _EMAIL_RE, _HANDLE_RE, _ACCT_RE):
            for match in regex.finditer(raw):
                token = match.group(0)
                eid = self.alias_to_entity.get(token) or self.alias_to_entity.get(token.lower())
                if eid:
                    add(eid, token)
                spans.append((match.start(), match.end()))

        # 3) fuzzy name matching for spelling variants
        candidates = self._name_candidates(raw)
        if candidates and self._name_items:
            names = [n for n, _ in self._name_items]
            eid_of = {n: eid for n, eid in self._name_items}
            for cand in candidates:
                cand_norm = normalize_text(cand)
                if len(cand_norm) < 4:
                    continue
                if any(cand_norm in alias for alias, _ in self._alias_items):
                    continue
                if process is not None:
                    matches = process.extract(
                        cand_norm, names, scorer=fuzz.token_sort_ratio,
                        score_cutoff=88.0, limit=3,
                    )
                else:  # pragma: no cover - fallback
                    import difflib
                    scored = sorted(
                        ((difflib.SequenceMatcher(None, cand_norm, n).ratio() * 100, n) for n in names),
                        reverse=True,
                    )[:3]
                    matches = [(n, s, 0) for s, n in scored if s >= 88.0]
                if not matches:
                    continue
                best_name, best_score = matches[0][0], float(matches[0][1])
                best_eid = eid_of.get(best_name)
                if not best_eid:
                    continue
                add(best_eid, cand)
                if len(matches) > 1 and float(matches[1][1]) >= best_score - 2.0:
                    second_eid = eid_of.get(matches[1][0])
                    if second_eid and second_eid != best_eid:
                        ambiguous.append({
                            "term": cand,
                            "best": best_eid,
                            "alternative": second_eid,
                            "score": round(best_score, 1),
                        })

        # 4) given-name / surname matching for short references.  This is what
        # lets "connection between Viktor and Elena" resolve when the full
        # names are "Viktor Duarte" and "Elena Unger".
        for match in re.finditer(r"\b[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ'’-]{2,}\b", raw):
            token = match.group(0)
            key = normalize_text(token)
            eids = self._given_names.get(key, [])
            if not eids:
                continue
            add(eids[0], token)
            if len(eids) > 1:
                ambiguous.append({
                    "term": token,
                    "best": eids[0],
                    "alternatives": eids[1:],
                    "score": 100.0,
                })

        return ids, terms, ambiguous

    def _name_candidates(self, raw: str) -> List[str]:
        """Capitalized name-like spans + quoted spans, limited to 200."""
        candidates: List[str] = []
        for match in re.finditer(r"\b[A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+){0,2}\b", raw):
            candidates.append(match.group(0))
        for match in re.finditer(r"[\"“”'‘’](.{2,60}?)[\"“”'‘’]", raw):
            candidates.append(match.group(1).strip())
        # lowercase queries: 2-3 word spans that are not stopwords
        tokens = re.findall(r"[^\W\d_]+", raw, flags=re.UNICODE)
        stop = STOPWORDS.get("en", set()) | STOPWORDS.get("es", set()) | \
            STOPWORDS.get("fr", set()) | STOPWORDS.get("de", set()) | \
            STOPWORDS.get("pt", set()) | STOPWORDS.get("hi", set())
        for n in (2, 3):
            for i in range(0, max(0, len(tokens) - n + 1)):
                span = tokens[i:i + n]
                if span and not all(t.lower() in stop for t in span):
                    candidates.append(" ".join(span))
        # de-duplicate, keep order, cap
        out: List[str] = []
        seen = set()
        for cand in candidates:
            key = cand.lower().strip()
            if key and key not in seen:
                seen.add(key)
                out.append(cand)
            if len(out) >= 200:
                break
        return out

    # -- time --------------------------------------------------------------
    def _extract_time_range(self, raw: str, norm: str, lang: str) -> TimeRange:
        anchor = self.case_end or datetime.now(timezone.utc)
        if self.case_start and anchor < self.case_start:
            anchor = self.case_start

        # 1) explicit range: between/from X and/to Y
        for between_word in _COMPARISON_BETWEEN:
            pattern = (rf"{re.escape(between_word)}\s+"
                       rf"(.{{3,40}}?)\s+(?:and|to|a|et|und|e|-)\s+(.{{3,40}})")
            m = re.search(pattern, norm)
            if m:
                start = self._parse_date_expression(m.group(1), lang, anchor)
                end = self._parse_date_expression(m.group(2), lang, anchor)
                if start or end:
                    start_dt = _day_start(start) if start else (self.case_start or datetime(1970, 1, 1, tzinfo=timezone.utc))
                    end_dt = _day_end(end) if end else anchor
                    return TimeRange(_iso(start_dt), _iso(end_dt), m.group(0).strip())

        # 2) month/year ranges before single-date parsing: "October 2025"
        # means the whole month, not 1 October.
        month_hit = None
        for names in MONTHS.values():
            for name in names:
                if _contains(norm, normalize_text(name)):
                    month_hit = name
                    break
            if month_hit:
                break
        if month_hit:
            year_match = _YEAR_ONLY.search(norm)
            # a day number is only a day if it is not the 4-digit year
            day_candidates = [
                int(x) for x in re.findall(r"\b(\d{1,2})\b", norm)
                if int(x) <= 31
            ]
            if not day_candidates:
                year = int(year_match.group(1)) if year_match else anchor.year
                for i, names in MONTHS.items():
                    if month_hit in names:
                        start, end = _month_bounds(year, MONTHS[i].index(month_hit) + 1)
                        return TimeRange(_iso(start), _iso(end), month_hit)
        if not month_hit:
            year_match = _YEAR_ONLY.search(norm)
            # year-only query ("in 2025") spans the year
            if year_match and not _DATE_ISO.search(norm) and not _DATE_DMY.search(norm):
                year = int(year_match.group(1))
                start = datetime(year, 1, 1, tzinfo=timezone.utc)
                end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
                return TimeRange(_iso(start), _iso(end), str(year))

        # 3) explicit single date / month / year
        single = self._parse_date_expression(norm, lang, anchor)
        if single:
            # if the text says "before X" or "after X", use an open range
            if _contains_any(norm, _COMPARISON_BELOW):
                start = self.case_start or datetime(1970, 1, 1, tzinfo=timezone.utc)
                return TimeRange(_iso(start), _iso(_day_end(single)), "before")
            if _contains_any(norm, _COMPARISON_ABOVE):
                end = self.case_end or anchor
                return TimeRange(_iso(_day_start(single)), _iso(end), "after")
            return TimeRange(_iso(_day_start(single)), _iso(_day_end(single)), "")

        # 3) relative ranges
        rel = self._parse_relative(norm, lang, anchor)
        if rel:
            return rel
        return TimeRange()

    def _parse_date_expression(self, expr: str, lang: str, anchor: datetime) -> Optional[datetime]:
        expr = (expr or "").strip()
        if not expr:
            return None
        # ISO
        m = _DATE_ISO.search(expr)
        if m:
            try:
                return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
            except ValueError:
                pass
        # DMY / MDY
        m = _DATE_DMY.search(expr)
        if m:
            a, b, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if a > 12:
                day, month = a, b
            elif b > 12:
                month, day = a, b
            else:
                month, day = a, b  # US-style default, matching the AML parser
            try:
                return datetime(year, month, day, tzinfo=timezone.utc)
            except ValueError:
                return None
        # month name + year (and optional day).  A bare month word is only
        # accepted with a digit or an explicit date preposition, so "may" in
        # "who may have called" is not read as the month of May.
        low = normalize_text(expr)
        has_date_preposition = _contains_any(
            low, ["in", "en", "au", "im", "em", "no", "na", "में", "during", "of"]
        )
        for i, names in MONTHS.items():
            for name in names:
                n = normalize_text(name)
                if not _contains(low, n):
                    continue
                year_match = re.search(r"\b(20\d{2})\b", low)
                day_match = re.search(r"\b(\d{1,2})\b", low)
                if not (year_match or day_match or has_date_preposition):
                    continue
                month = MONTHS[i].index(name) + 1
                year = int(year_match.group(1)) if year_match else anchor.year
                day = 1
                if day_match:
                    candidate_day = int(day_match.group(1))
                    if candidate_day <= 31:
                        day = candidate_day
                try:
                    return datetime(year, month, day, tzinfo=timezone.utc)
                except ValueError:
                    try:
                        return datetime(year, month, 1, tzinfo=timezone.utc)
                    except ValueError:
                        return None
        # year only
        m = _YEAR_ONLY.search(expr)
        if m:
            year = int(m.group(1))
            return datetime(year, 1, 1, tzinfo=timezone.utc)
        return None

    def _parse_relative(self, norm: str, lang: str, anchor: datetime) -> Optional[TimeRange]:
        # today / yesterday
        for key, delta_days in (("today", 0), ("yesterday", -1)):
            words = RELATIVE_WORDS.get(key, {}).get(lang, []) + RELATIVE_WORDS.get(key, {}).get("en", [])
            if _contains_any(norm, words):
                day = anchor + timedelta(days=delta_days)
                return TimeRange(_iso(_day_start(day)), _iso(_day_end(day)), key)
        # this week / this month / last week / last month
        for key in ("this_week", "this_month", "last_week", "last_month"):
            words = RELATIVE_WORDS.get(key, {}).get(lang, []) + RELATIVE_WORDS.get(key, {}).get("en", [])
            if not _contains_any(norm, words):
                continue
            if key == "this_week":
                start = _day_start(anchor - timedelta(days=anchor.weekday()))
                return TimeRange(_iso(start), _iso(anchor), key)
            if key == "last_week":
                start = _day_start(anchor - timedelta(days=anchor.weekday() + 7))
                end = _day_end(anchor - timedelta(days=anchor.weekday() + 1))
                return TimeRange(_iso(start), _iso(end), key)
            if key == "this_month":
                start = datetime(anchor.year, anchor.month, 1, tzinfo=timezone.utc)
                return TimeRange(_iso(start), _iso(anchor), key)
            if key == "last_month":
                year, month = (anchor.year, anchor.month - 1) if anchor.month > 1 else (anchor.year - 1, 12)
                start, end = _month_bounds(year, month)
                return TimeRange(_iso(start), _iso(end), key)
        # last/past N days/weeks/months/years
        last_words = RELATIVE_WORDS["last"].get(lang, []) + RELATIVE_WORDS["last"].get("en", [])
        if _contains_any(norm, last_words):
            m = re.search(
                r"\b(\d{1,3})\s*(day|days|día|días|dia|dias|jour|jours|tag|tage|"
                r"week|weeks|semana|semanas|semaine|semaines|woche|wochen|"
                r"month|months|mes|meses|mois|monat|monate|year|years|año|años|"
                r"ano|anos|an|ans|jahr|jahre|दिन|सप्ताह|महीना|महीने|साल|वर्ष)\b",
                norm,
            )
            if m:
                n = int(m.group(1))
                unit = m.group(2)
                days = _UNIT_DAYS.get(unit, 1) * n
                start = anchor - timedelta(days=days)
                return TimeRange(_iso(start), _iso(anchor), f"last {n} {unit}")
        # "last year" / "past year" (no number)
        if _contains_any(norm, last_words) and _contains_any(norm, ["year", "año", "an", "jahr", "साल", "वर्ष"]):
            start = anchor - timedelta(days=365)
            return TimeRange(_iso(start), _iso(anchor), "last year")
        return None

    # -- amounts -----------------------------------------------------------
    def _extract_amount_range(self, norm: str) -> AmountRange:
        # strip date-like spans so 2025-10-03 is not read as an amount
        cleaned = _DATE_ISO.sub(" ", norm)
        cleaned = _DATE_DMY.sub(" ", cleaned)

        # between X and Y
        between_pattern = (
            rf"(?:between|entre|zwischen|के बीच)\s+{_MONEY}\s+"
            rf"(?:and|to|a|et|und|e|-)\s+{_MONEY}"
        )
        m = re.search(between_pattern, cleaned)
        if m:
            lo = _parse_number(m.group(1), m.group(2))
            hi = _parse_number(m.group(3), m.group(4))
            if lo is not None or hi is not None:
                return AmountRange(lo, hi, m.group(0).strip())

        # above / below
        above = "|".join(re.escape(x) for x in sorted(_COMPARISON_ABOVE, key=len, reverse=True))
        below = "|".join(re.escape(x) for x in sorted(_COMPARISON_BELOW, key=len, reverse=True))
        m = re.search(rf"(?:{above})\s+{_MONEY}", cleaned)
        if m:
            value = _parse_number(m.group(1), m.group(2))
            if value is not None:
                return AmountRange(value, None, m.group(0).strip())
        m = re.search(rf"(?:{below})\s+{_MONEY}", cleaned)
        if m:
            value = _parse_number(m.group(1), m.group(2))
            if value is not None:
                return AmountRange(None, value, m.group(0).strip())

        # bare range "10,000 to 20,000" (only if both sides look like money)
        m = re.search(rf"{_MONEY}\s*(?:-|to|a|et|und|e)\s*{_MONEY}", cleaned)
        if m:
            lo = _parse_number(m.group(1), m.group(2))
            hi = _parse_number(m.group(3), m.group(4))
            if lo is not None and hi is not None and lo < hi:
                return AmountRange(lo, hi, m.group(0).strip())
        return AmountRange()

    # -- sources / direction / locations ----------------------------------
    def _extract_sources(self, norm: str, lang: str) -> List[str]:
        found: List[str] = []
        for source, by_lang in SOURCE_WORDS.items():
            words = list(by_lang.get(lang, [])) + list(by_lang.get("en", []))
            if _contains_any(norm, words):
                found.append(source)
        # a query about "transfers" is bank evidence even if the word "bank"
        # is absent; the source vocabulary already covers that.  Keep order
        # stable for deterministic plans.
        return found

    def _extract_direction(self, norm: str) -> Optional[str]:
        if _contains_any(norm, ["incoming", "credit", "received", "deposit",
                                "entrante", "crédito", "recibido", "eingehend",
                                "gutschrift", "entrada", "recebido", "आवक", "जमा"]):
            return "credit"
        if _contains_any(norm, ["outgoing", "debit", "sent", "withdrawal", "paid",
                                "saliente", "débito", "enviado", "ausgehend",
                                "lastschrift", "saída", "enviado", "जावक", "निकासी"]):
            return "debit"
        return None

    def _extract_locations(self, raw: str, norm: str) -> List[str]:
        found: List[str] = []
        for match in _LOCATION_RE.finditer(raw):
            token = match.group(0)
            if token not in found:
                found.append(token)
        for loc in self.known_locations:
            if _contains(norm, normalize_text(loc)) and loc not in found:
                found.append(loc)
        # "at <location>" / "tower <location>" / "branch <location>"
        for match in re.finditer(r"\b(?:at|in|tower|branch|location|cell|en|à|au|in|bei|im|em|no|na)\s+([A-Za-z0-9_-]{2,20})", raw, re.IGNORECASE):
            token = match.group(1)
            if token.lower() in STOPWORDS.get("en", set()):
                continue
            if token not in found and (token.upper().startswith(("T", "BRANCH", "LAT", "LOC")) or token in self.known_locations):
                found.append(token)
        return found[:5]

    # -- limit / keywords --------------------------------------------------
    def _extract_limit(self, norm: str) -> int:
        m = _TOP_RE.search(norm)
        if m:
            try:
                return max(1, min(int(m.group(1)), 100))
            except ValueError:
                pass
        return self.default_limit

    def _extract_keywords(self, raw: str, norm: str, lang: str,
                          entity_terms: List[str], time_range: TimeRange,
                          amount_range: AmountRange, sources: List[str],
                          locations: List[str]) -> List[str]:
        stop: Set[str] = set()
        for words in STOPWORDS.values():
            stop |= {normalize_text(w) for w in words}
        # remove intent/source vocabulary so the BM25 query focuses on content
        remove: Set[str] = set()
        for by_lang in INTENT_LEXICON.values():
            for phrase in by_lang.get(lang, []) + by_lang.get("en", []):
                remove |= {normalize_text(w) for w in _tokenize(phrase)}
        for by_lang in SOURCE_WORDS.values():
            for phrase in by_lang.get(lang, []) + by_lang.get("en", []):
                remove |= {normalize_text(w) for w in _tokenize(phrase)}
        for term in entity_terms:
            remove |= {normalize_text(w) for w in _tokenize(term)}
        for loc in locations:
            remove |= {normalize_text(w) for w in _tokenize(loc)}
        # time/amount words
        for words in RELATIVE_WORDS.values():
            for phrase in words.get(lang, []) + words.get("en", []):
                remove |= {normalize_text(w) for w in _tokenize(phrase)}
        for phrase in _COMPARISON_ABOVE + _COMPARISON_BELOW + _COMPARISON_BETWEEN:
            remove |= {normalize_text(w) for w in _tokenize(phrase)}
        # month names and time units are filters, not content keywords
        for names in MONTHS.values():
            for name in names:
                remove |= {normalize_text(w) for w in _tokenize(name)}
        for unit in _UNIT_DAYS:
            remove |= {normalize_text(w) for w in _tokenize(unit)}

        # quoted phrases are strong keywords
        keywords: List[str] = []
        for match in re.finditer(r"[\"“”'‘’](.{2,60}?)[\"“”'‘’]", raw):
            phrase = match.group(1).strip()
            if phrase and normalize_text(phrase) not in keywords:
                keywords.append(normalize_text(phrase))

        for token in _tokenize(raw):
            tok = normalize_text(token)
            if not tok or tok in stop or tok in remove:
                continue
            if len(tok) < 3 and not re.search(r"[\u0900-\u097F]", tok):
                continue
            if tok.isdigit():
                continue
            if tok not in keywords:
                keywords.append(tok)
        # keep the most discriminative tokens first; cap at 12
        return keywords[:12]

    def _extract_image_prompt(self, text: str) -> str:
        """Strip the 'generate an image of ...' wrapper to get the prompt."""
        prompt = text or ""
        for trigger in (
            "generate an image of", "generate an image", "create an image of",
            "create an image", "make an image of", "make an image",
            "draw an image of", "draw an image", "draw a picture of",
            "draw a picture", "make a picture of", "make a picture",
            "generate a picture of", "generate a picture", "illustrate",
            "text to image", "text-to-image", "render an image of",
            "render an image", "generate a visual of", "generate a visual",
            "छवि बनाएं", "तस्वीर बनाएं", "चित्र बनाएं",
            "generar una imagen", "crear una imagen", "dibujar una imagen",
            "générer une image", "créer une image", "dessiner une image",
            "bild generieren", "bild erstellen", "gerar uma imagem",
            "criar uma imagem", "desenhar uma imagem",
        ):
            prompt = re.sub(rf"(?i)\b{re.escape(trigger)}\b", " ", prompt)
        prompt = re.sub(r"^\s*(of|for|about|showing|with|depicting)\s+", "",
                        prompt, flags=re.IGNORECASE)
        prompt = re.sub(r"\s+", " ", prompt).strip(" .,:;")
        return prompt or (text or "").strip()

    def _extract_chart_kind(self, norm: str, entity_ids: List[str]) -> str:
        """Pick the chart the user is asking for (used by the plot tool)."""
        if _contains_any(norm, ["map", "activity map", "location map"]):
            return "map"
        if _contains_any(norm, ["network", "correlation", "connection graph",
                                "suspect graph", "graph"]):
            if _contains_any(norm, ["timeline", "traffic", "over time"]):
                return "entity_timeline" if entity_ids else "timeline"
            return "network"
        if _contains_any(norm, ["timeline", "traffic", "activity over time",
                                "daily traffic"]):
            return "entity_timeline" if entity_ids else "timeline"
        if _contains_any(norm, ["people", "risk bar", "who stands out", "attention"]):
            return "people"
        if _contains_any(norm, ["sources", "ingestion", "files"]):
            return "sources"
        if _contains_any(norm, ["models", "engine", "scores"]):
            return "models"
        if entity_ids:
            return "entity_timeline"
        return "network"

    def _event_types_for_sources(self, sources: List[str]) -> List[str]:
        mapping = {
            "bank": "transaction",
            "cdr": "call",
            "social": "post",
            "email": "email",
            "chat": "message",
            "document": "document",
            "image": "image",
        }
        return [mapping[s] for s in sources if s in mapping]

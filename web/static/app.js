/* Sutram Investigator — vanilla JS workspace */
'use strict';

const state = {
  cases: [],
  languages: [],
  lang: 'en',
  caseId: '',
  bundle: null,
  i18n: {},
  sessionId: 'web-' + Math.random().toString(36).slice(2),
  activeTab: 'assistant',
  busy: false,
};

const FALLBACK_SUGGESTIONS = [
  'What evidence do we have?',
  'What looks suspicious in this case?',
  'Who is the main suspect?',
  'Show transfers above 10,000 in the last 30 days.',
  'What should I investigate next?',
  'Draft a case report.',
];

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function fmt(template, vars) {
  if (typeof template !== 'string') return template;
  return template.replace(/\{(\w+)\}/g, (_, key) => (vars && key in vars ? vars[key] : `{${key}}`));
}

function t(key, fallback) {
  const value = state.i18n[key];
  if (value !== undefined && value !== null) return value;
  return fallback !== undefined ? fallback : key;
}

function formatTime(value) {
  if (!value) return 'undated';
  return String(value).replace('T', ' ').slice(0, 19);
}

function shortHash(value) {
  if (!value) return '—';
  return value.slice(0, 16) + '…';
}

function badge(value, cls = '') {
  const v = String(value || '').toUpperCase();
  return `<span class="badge ${cls} ${escapeHtml(v)}">${escapeHtml(v)}</span>`;
}

function riskBadge(score) {
  const value = Number(score || 0);
  if (value >= 0.7) return badge('HIGH');
  if (value >= 0.45) return badge('MEDIUM');
  return badge('LOW');
}

function toast(message, ms = 3200) {
  const el = $('#toast');
  el.textContent = message;
  el.classList.add('show');
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => el.classList.remove('show'), ms);
}

async function api(path, options = {}) {
  const opts = { headers: { 'Content-Type': 'application/json' }, ...options };
  const res = await fetch(path, opts);
  let data;
  try {
    data = await res.json();
  } catch (err) {
    throw new Error(`HTTP ${res.status}`);
  }
  if (!res.ok || (data && data.error)) {
    throw new Error((data && data.error) || `HTTP ${res.status}`);
  }
  return data;
}

function post(path, body) {
  return api(path, { method: 'POST', body: JSON.stringify(body) });
}

/* ---------------- markdown-lite ---------------- */
function inlineMd(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/\[C(\d+)\]/g, '<span class="cite-label">C$1</span>');
}

function md(text) {
  const lines = escapeHtml(text || '').split('\n');
  let html = '';
  let inList = false;
  for (const raw of lines) {
    const line = raw.replace(/\s+$/, '');
    if (/^\s*[-•]\s+/.test(line)) {
      if (!inList) { html += '<ul>'; inList = true; }
      html += '<li>' + inlineMd(line.replace(/^\s*[-•]\s+/, '')) + '</li>';
      continue;
    }
    if (inList) { html += '</ul>'; inList = false; }
    if (!line.trim()) { html += '<br>'; continue; }
    if (/^\s*\d+\.\s+/.test(line)) {
      html += '<div class="numline">' + inlineMd(line) + '</div>';
      continue;
    }
    if (/^\[C\d+\]/.test(line)) {
      html += '<div class="cite-line">' + inlineMd(line) + '</div>';
      continue;
    }
    html += '<p>' + inlineMd(line) + '</p>';
  }
  if (inList) html += '</ul>';
  return html;
}

/* ---------------- i18n ---------------- */
function applyI18n() {
  $$('[data-i18n]').forEach((el) => {
    const value = t(el.dataset.i18n, el.textContent);
    if (typeof value === 'string') el.textContent = value;
  });
  $$('[data-i18n-placeholder]').forEach((el) => {
    const value = t(el.dataset.i18nPlaceholder, el.getAttribute('placeholder') || '');
    if (typeof value === 'string') el.setAttribute('placeholder', value);
  });
  document.documentElement.lang = state.lang;
}

async function loadI18n(lang) {
  state.i18n = await api(`/api/i18n?lang=${encodeURIComponent(lang)}`);
  applyI18n();
}

/* ---------------- render ---------------- */
function renderStats() {
  const b = state.bundle;
  if (!b) return;
  const stats = b.stats || {};
  const cards = [
    [stats.evidence ?? b.evidence?.length ?? 0, t('stats.evidence', 'Evidence')],
    [stats.entities ?? b.entities?.length ?? 0, t('stats.entities', 'Entities')],
    [stats.events ?? 0, t('stats.events', 'Events')],
    [b.findings?.length ?? 0, t('stats.findings', 'Findings')],
  ];
  $('#stats').innerHTML = cards.map(([v, k]) =>
    `<div class="stat"><div class="v">${escapeHtml(v)}</div><div class="k">${escapeHtml(k)}</div></div>`
  ).join('');
  $('#case-title').textContent = b.case?.title || '—';
  const start = b.stats?.start ? formatTime(b.stats.start).slice(0, 10) : '—';
  const end = b.stats?.end ? formatTime(b.stats.end).slice(0, 10) : '—';
  $('#case-meta').textContent = `${b.case?.case_id || ''} · ${start} → ${end}`;
  const sources = b.stats?.sources || {};
  $('#source-list').innerHTML = Object.keys(sources).length
    ? Object.entries(sources).sort((a, b) => b[1] - a[1]).map(([k, v]) =>
        `<div class="source-pill"><span>${escapeHtml(k)}</span><b>${escapeHtml(v)}</b></div>`).join('')
    : '<div class="empty">No sources indexed.</div>';
}

function renderSuggestions() {
  let examples = t('query.examples', FALLBACK_SUGGESTIONS);
  if (!Array.isArray(examples)) examples = FALLBACK_SUGGESTIONS;
  $('#suggestions').innerHTML = examples.slice(0, 6).map((q) =>
    `<button class="chip" data-question="${escapeHtml(q)}">${escapeHtml(q)}</button>`
  ).join('');
  $$('#suggestions .chip').forEach((el) => el.addEventListener('click', () => ask(el.dataset.question)));
}

function renderWelcome() {
  const chat = $('#chat');
  chat.innerHTML = '';
  addMessage('assistant', md(t('chat.welcome', 'I am your forensic case assistant. I answer only from the indexed evidence and cite every factual claim.')));
  const examples = t('query.examples', FALLBACK_SUGGESTIONS);
  if (Array.isArray(examples)) {
    const chips = examples.slice(0, 4).map((q) =>
      `<button class="chip" data-question="${escapeHtml(q)}">${escapeHtml(q)}</button>`).join('');
    const wrap = document.createElement('div');
    wrap.className = 'msg assistant';
    wrap.innerHTML = `<div class="avatar">S</div><div class="bubble"><div class="muted" style="margin-bottom:6px">${escapeHtml(t('chat.suggestions', 'Suggested questions'))}</div><div class="suggestions">${chips}</div></div>`;
    chat.appendChild(wrap);
    wrap.querySelectorAll('.chip').forEach((el) => el.addEventListener('click', () => ask(el.dataset.question)));
  }
}

function addMessage(role, html) {
  const chat = $('#chat');
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;
  wrap.innerHTML = `<div class="avatar">${role === 'user' ? 'A' : 'S'}</div><div class="bubble">${html}</div>`;
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
  return wrap;
}

function renderCitations(citations) {
  const list = $('#citation-list');
  if (!citations || !citations.length) {
    list.innerHTML = '<div class="empty">No citations for this answer.</div>';
    return;
  }
  list.innerHTML = citations.map((c, i) => `
    <div class="citation">
      <div class="cite-head"><span>C${i + 1}</span><span>${escapeHtml(c.evidence_id || '')}</span></div>
      <div class="cite-file">${escapeHtml(c.file || c.evidence_id || '')}</div>
      <div class="cite-time">${escapeHtml(formatTime(c.timestamp))} · ${escapeHtml(c.source_type || '')}</div>
      <div class="cite-snippet">${escapeHtml(c.snippet || '')}</div>
    </div>`).join('');
}

function renderFollowups(followups) {
  const box = $('#followups');
  if (!followups || !followups.length) {
    box.innerHTML = '<div class="empty">—</div>';
    return;
  }
  box.innerHTML = followups.map((q) =>
    `<button class="chip" data-question="${escapeHtml(q)}">${escapeHtml(q)}</button>`).join('');
  box.querySelectorAll('.chip').forEach((el) => el.addEventListener('click', () => ask(el.dataset.question)));
}

function renderAnswer(answer) {
  addMessage('assistant', md(answer.answer));
  renderCitations(answer.citations);
  renderFollowups(answer.followups);
  $('#confidence').textContent = `${Math.round((answer.confidence || 0) * 100)}%`;
  const plan = answer.query_plan || {};
  $('#answer-meta').innerHTML = [
    `intent: ${escapeHtml(answer.intent || '—')}`,
    `language: ${escapeHtml(answer.language || '—')}`,
    `citations: ${(answer.citations || []).length}`,
    plan.time_range?.start ? `from: ${escapeHtml(formatTime(plan.time_range.start))}` : '',
    plan.time_range?.end ? `to: ${escapeHtml(formatTime(plan.time_range.end))}` : '',
    plan.amount_range?.min != null ? `min amount: ${escapeHtml(plan.amount_range.min)}` : '',
    plan.amount_range?.max != null ? `max amount: ${escapeHtml(plan.amount_range.max)}` : '',
  ].filter(Boolean).join('<br>');
}

function renderEvidence() {
  const items = state.bundle?.evidence || [];
  $('#evidence-count').textContent = `${items.length} item(s)`;
  if (!items.length) {
    $('#evidence-list').innerHTML = '<div class="empty">No evidence indexed.</div>';
    return;
  }
  const rows = items.map((e) => `
    <tr>
      <td class="mono">${escapeHtml(e.evidence_id)}</td>
      <td>${escapeHtml(e.filename)}</td>
      <td>${escapeHtml(e.source_type)}</td>
      <td class="mono">${escapeHtml(e.n_rows)}</td>
      <td class="mono">${escapeHtml(e.n_chunks)}</td>
      <td class="mono">${escapeHtml(shortHash(e.sha256))}</td>
      <td class="mono">${escapeHtml(formatTime(e.acquired_at))}</td>
    </tr>`).join('');
  $('#evidence-list').innerHTML = `<table><thead><tr>
    <th>ID</th><th>File</th><th>Type</th><th>Rows</th><th>Chunks</th><th>SHA-256</th><th>Acquired</th>
  </tr></thead><tbody>${rows}</tbody></table>`;
}

function renderEntities() {
  const items = state.bundle?.entities || [];
  $('#entity-count').textContent = `${items.length} resolved`;
  if (!items.length) {
    $('#entity-list').innerHTML = '<div class="empty">No entities resolved.</div>';
    return;
  }
  $('#entity-list').innerHTML = items.slice(0, 120).map((e) => {
    const name = (e.names && e.names[0]) || e.entity_id;
    const ids = (e.identifiers || []).slice(0, 5).map((x) => `<span>${escapeHtml(x)}</span>`).join('');
    const sources = (e.sources || []).map((x) => `<span>${escapeHtml(x)}</span>`).join('');
    const factors = (e.risk_factors || []).slice(0, 3).map((x) => `<div class="muted">• ${escapeHtml(x)}</div>`).join('');
    return `<div class="card clickable" data-entity="${escapeHtml(name)}">
      <div class="card-title">${escapeHtml(name)} <span class="muted">(${escapeHtml(e.entity_id)})</span></div>
      <div class="card-sub">${riskBadge(e.risk_score)} · risk ${Number(e.risk_score || 0).toFixed(2)} · ${escapeHtml(e.n_events || 0)} events</div>
      <div class="kv">${sources}${ids}</div>
      ${factors}
    </div>`;
  }).join('');
  $$('#entity-list .card.clickable').forEach((el) => el.addEventListener('click', () => {
    ask(`Who is ${el.dataset.entity}?`);
    switchTab('assistant');
  }));
}

function renderFindings() {
  const items = state.bundle?.findings || [];
  $('#finding-count').textContent = `${items.length} finding(s)`;
  if (!items.length) {
    $('#finding-list').innerHTML = '<div class="empty">No findings yet. Run the analytics step.</div>';
    return;
  }
  $('#finding-list').innerHTML = items.map((f) => {
    const cites = (f.citations || []).slice(0, 4).map((c) =>
      `<div class="citation"><div class="cite-head"><span>${escapeHtml(c.evidence_id || '')}</span><span>${escapeHtml(formatTime(c.timestamp))}</span></div>
       <div class="cite-file">${escapeHtml(c.file || '')}</div><div class="cite-snippet">${escapeHtml(c.snippet || '')}</div></div>`).join('');
    return `<div class="card">
      <div class="card-title">${escapeHtml(f.title)}</div>
      <div class="card-sub">${badge(f.severity)} ${badge(f.kind, 'kind')} · confidence ${Number(f.confidence || 0).toFixed(2)}</div>
      <div class="card-body">${escapeHtml(f.description || '')}</div>
      <div class="kv">${(f.entity_ids || []).map((x) => `<span>${escapeHtml(x)}</span>`).join('')}</div>
      ${cites}
    </div>`;
  }).join('');
}

function renderLeads() {
  const items = state.bundle?.recommendations || [];
  if (!items.length) {
    $('#lead-list').innerHTML = '<div class="empty">No recommendations yet.</div>';
    return;
  }
  $('#lead-list').innerHTML = items.map((r) => `
    <div class="card">
      <div class="card-title">${escapeHtml(r.action)}</div>
      <div class="card-sub">${badge(r.priority)} · ${escapeHtml(r.category)} · ${escapeHtml(r.recommendation_id)}</div>
      <div class="card-body">${escapeHtml(r.rationale)}</div>
      <div class="kv">
        ${(r.entity_ids || []).map((x) => `<span>${escapeHtml(x)}</span>`).join('')}
        ${(r.finding_ids || []).map((x) => `<span>${escapeHtml(x)}</span>`).join('')}
      </div>
      <div class="card-actions">
        <button class="btn ghost" data-copy="${escapeHtml(r.action + ' — ' + r.rationale)}">Copy</button>
      </div>
    </div>`).join('');
  $$('#lead-list [data-copy]').forEach((el) => el.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(el.dataset.copy);
      toast('Copied to clipboard.');
    } catch (err) {
      toast('Clipboard unavailable.');
    }
  }));
}

function renderDiary() {
  const entries = state.bundle?.diary || [];
  const ok = state.bundle?.chain_ok;
  $('#chain-status').innerHTML = ok
    ? '<span style="color:var(--teal)">Hash chain verified</span>'
    : '<span style="color:var(--red)">Hash chain TAMPERED</span>';
  if (!entries.length) {
    $('#diary-list').innerHTML = '<div class="empty">No diary entries.</div>';
    return;
  }
  $('#diary-list').innerHTML = entries.slice().reverse().map((d) => `
    <div class="diary-entry">
      <div><span class="diary-time">${escapeHtml(formatTime(d.timestamp))}</span>
      <span class="diary-type">${escapeHtml(d.entry_type)}</span></div>
      <div class="diary-text">${escapeHtml(d.text)}</div>
      <div class="diary-hash">${escapeHtml(d.actor)} · ${escapeHtml((d.hash || '').slice(0, 24))}…</div>
    </div>`).join('');
}

function renderReport() {
  const b = state.bundle;
  if (!b) return;
  const summary = b.summary || {};
  const gaps = (summary.gaps || []).map((g) =>
    `<span class="badge ${escapeHtml(g.severity || '')}">${escapeHtml((g.key || '').replace(/_/g, ' '))}</span>`).join('');
  const limitations = (summary.limitations || []).map((x) => `<li>${escapeHtml(x)}</li>`).join('');
  $('#summary-panel').innerHTML = `
    <div class="label">Case summary</div>
    <p>${md(summary.text || '')}</p>
    ${gaps ? `<div class="label">Evidence gaps</div><div class="gap-list">${gaps}</div>` : ''}
    ${limitations ? `<div class="label" style="margin-top:12px">Limitations</div><ul>${limitations}</ul>` : ''}
  `;
  const files = b.report_files || {};
  const setLink = (id, key, download) => {
    const el = $(id);
    if (files[key]) {
      el.href = files[key];
      el.classList.remove('hidden');
      if (download) el.setAttribute('download', '');
      else el.removeAttribute('download');
    } else {
      el.classList.add('hidden');
      el.removeAttribute('href');
    }
  };
  setLink('#download-md', 'md', true);
  setLink('#download-html', 'html', false);
  setLink('#download-pdf', 'pdf', true);
  setLink('#download-json', 'json', true);
  setLink('#download-analytics', 'analytics_html', false);
}

function renderAll() {
  renderStats();
  renderEvidence();
  renderEntities();
  renderFindings();
  renderLeads();
  renderDiary();
  renderReport();
  renderSuggestions();
}

/* ---------------- data loading ---------------- */
async function loadBundle(caseId, { resetChat = false } = {}) {
  if (!caseId) return;
  state.caseId = caseId;
  const b = await api(`/api/case_bundle?case_id=${encodeURIComponent(caseId)}&lang=${encodeURIComponent(state.lang)}`);
  state.bundle = b;
  renderAll();
  if (resetChat) {
    renderWelcome();
    renderCitations([]);
    renderFollowups([]);
    $('#confidence').textContent = '—';
    $('#answer-meta').textContent = '—';
  }
  const url = new URL(window.location.href);
  url.searchParams.set('case', caseId);
  window.history.replaceState({}, '', url);
}

async function loadCases() {
  state.cases = await api('/api/cases');
  const sel = $('#case-select');
  sel.innerHTML = state.cases.map((c) =>
    `<option value="${escapeHtml(c.case_id)}">${escapeHtml(c.title)} · ${escapeHtml(c.case_id)}</option>`).join('');
  if (!state.cases.length) {
    sel.innerHTML = '<option value="">No cases — run python run_case.py demo</option>';
    return;
  }
  const params = new URLSearchParams(window.location.search);
  const requested = params.get('case');
  const chosen = state.cases.some((c) => c.case_id === requested) ? requested : state.cases[0].case_id;
  sel.value = chosen;
  state.caseId = chosen;
}

async function loadLanguages() {
  state.languages = await api('/api/languages');
  const sel = $('#lang-select');
  sel.innerHTML = state.languages.map((l) =>
    `<option value="${escapeHtml(l.code)}">${escapeHtml(l.name)}</option>`).join('');
  const stored = localStorage.getItem('sutram-lang');
  state.lang = stored && state.languages.some((l) => l.code === stored) ? stored : 'en';
  sel.value = state.lang;
}

/* ---------------- actions ---------------- */
async function ask(question) {
  if (!question || !state.caseId) {
    toast('Load a case first (python run_case.py demo).');
    return;
  }
  if (state.busy) return;
  state.busy = true;
  $('#send-btn').disabled = true;
  addMessage('user', `<p>${escapeHtml(question)}</p>`);
  const thinking = addMessage('assistant', `<span class="thinking">${escapeHtml(t('chat.thinking', 'Searching the evidence…'))}</span>`);
  try {
    const answer = await post('/api/ask', {
      case_id: state.caseId,
      question,
      lang: state.lang,
      session_id: state.sessionId,
    });
    thinking.remove();
    renderAnswer(answer);
    // refresh diary/findings without resetting the chat
    loadBundle(state.caseId, { resetChat: false }).catch(() => {});
  } catch (err) {
    thinking.remove();
    addMessage('assistant', `<p style="color:var(--red)">${escapeHtml(err.message)}</p>`);
    toast(err.message);
  } finally {
    state.busy = false;
    $('#send-btn').disabled = false;
    $('#question').focus();
  }
}

async function generateReport() {
  if (!state.caseId) return;
  const btn = $('#generate-report');
  btn.disabled = true;
  btn.textContent = t('common.loading', 'Working…');
  try {
    const result = await post('/api/report', { case_id: state.caseId, lang: state.lang });
    toast(t('report.generated', 'Report generated'));
    await loadBundle(state.caseId, { resetChat: false });
    if (result.files?.html) window.open(`/files/${encodeURIComponent(state.caseId)}/reports/case_report.html`, '_blank');
  } catch (err) {
    toast(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = t('report.generate', 'Generate report');
  }
}

async function regenLeads() {
  if (!state.caseId) return;
  const btn = $('#regen-leads');
  btn.disabled = true;
  try {
    await post('/api/recommend', { case_id: state.caseId, lang: state.lang });
    await loadBundle(state.caseId, { resetChat: false });
    toast(t('common.refresh', 'Refreshed'));
  } catch (err) {
    toast(err.message);
  } finally {
    btn.disabled = false;
  }
}

function switchTab(tab) {
  state.activeTab = tab;
  $$('.nav-btn').forEach((b) => b.classList.toggle('active', b.dataset.tab === tab));
  $$('.tab').forEach((s) => s.classList.toggle('active', s.id === `tab-${tab}`));
}

/* ---------------- init ---------------- */
async function init() {
  try {
    await Promise.all([loadLanguages(), loadCases()]);
    await loadI18n(state.lang);
    if (!state.cases.length) {
      renderWelcome();
      addMessage('assistant', '<p>No cases found. Generate the demo with <code>python run_case.py demo</code>, then refresh this page.</p>');
      return;
    }
    await loadBundle(state.caseId, { resetChat: true });
  } catch (err) {
    renderWelcome();
    addMessage('assistant', `<p style="color:var(--red)">Startup error: ${escapeHtml(err.message)}</p>`);
    toast(err.message);
  }

  $('#case-select').addEventListener('change', async (e) => {
    try {
      await loadBundle(e.target.value, { resetChat: true });
    } catch (err) {
      toast(err.message);
    }
  });
  $('#lang-select').addEventListener('change', async (e) => {
    state.lang = e.target.value;
    localStorage.setItem('sutram-lang', state.lang);
    try {
      await loadI18n(state.lang);
      await loadBundle(state.caseId, { resetChat: true });
    } catch (err) {
      toast(err.message);
    }
  });
  $('#refresh-btn').addEventListener('click', async () => {
    try {
      await loadBundle(state.caseId, { resetChat: false });
      toast(t('common.refresh', 'Refreshed'));
    } catch (err) {
      toast(err.message);
    }
  });
  $('#send-btn').addEventListener('click', () => {
    const q = $('#question').value.trim();
    if (!q) return;
    $('#question').value = '';
    ask(q);
  });
  $('#question').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      $('#send-btn').click();
    }
  });
  $('#generate-report').addEventListener('click', generateReport);
  $('#regen-leads').addEventListener('click', regenLeads);
  $$('.nav-btn').forEach((b) => b.addEventListener('click', () => switchTab(b.dataset.tab)));
}

document.addEventListener('DOMContentLoaded', init);

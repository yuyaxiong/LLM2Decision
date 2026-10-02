"""Lightweight debug page: hand-build a /v1/systemone request and inspect the readout.

Single-page HTML with inline CSS/JS, no build step, no external dependencies, no CDN.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>System One Debug Page</title>
<style>
  body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; margin: 0; padding: 24px; background: #f6f7f9; color: #1f2329; }
  h1 { font-size: 20px; margin: 0 0 16px; }
  .card { background: #fff; border: 1px solid #e3e6eb; border-radius: 8px; padding: 16px; margin-bottom: 16px; max-width: 880px; }
  label { display: block; font-size: 13px; color: #4b5563; margin: 12px 0 4px; }
  textarea, select, input[type=text] { width: 100%; box-sizing: border-box; font: inherit; padding: 8px; border: 1px solid #cfd4dc; border-radius: 6px; background: #fff; }
  textarea { min-height: 72px; resize: vertical; }
  .row { display: flex; gap: 16px; flex-wrap: wrap; }
  .row > div { flex: 1; min-width: 200px; }
  .hint { font-size: 12px; color: #8a9099; margin-top: 2px; }
  .inline { display: flex; align-items: center; gap: 8px; margin-top: 12px; }
  .inline label { margin: 0; }
  button { margin-top: 16px; font: inherit; padding: 9px 22px; border: 0; border-radius: 6px; background: #2563eb; color: #fff; cursor: pointer; }
  button:disabled { opacity: .6; cursor: default; }
  .muted { color: #8a9099; font-size: 13px; }
  .err { color: #b91c1c; white-space: pre-wrap; word-break: break-all; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 6px; }
  th, td { border: 1px solid #e3e6eb; padding: 4px 8px; text-align: left; }
  th { background: #f3f4f6; font-weight: 500; }
  .bar-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; font-size: 13px; }
  .bar-label { width: 160px; flex: none; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .bar-track { flex: 1; background: #eef1f5; border-radius: 4px; height: 16px; }
  .bar-fill { display: block; height: 100%; background: #2563eb; border-radius: 4px; min-width: 2px; }
  .bar-val { width: 90px; flex: none; text-align: right; color: #4b5563; }
  #model { min-width: 240px; }
  .hint { font-size: 12px; color: #6b7280; margin-top: 3px; }
  .kv { margin: 2px 0; font-size: 13px; }
  .kv b { display: inline-block; min-width: 120px; color: #4b5563; font-weight: 500; }
  h3 { font-size: 15px; margin: 16px 0 6px; }
</style>
</head>
<body>
<h1>System One Debug Page</h1>

<div class="card">
  <label for="state">Input state</label>
  <textarea id="state" placeholder="Raw input text to judge"></textarea>

  <div class="row">
    <div>
      <label for="qtype">Question type</label>
      <select id="qtype">
        <option value="choice">choice (pick one)</option>
        <option value="noul">noul (yes/no judgment)</option>
        <option value="score">score (scale rating)</option>
      </select>
    </div>
    <div>
      <label for="model">Model route</label>
      <select id="model"><option value="">(loading…)</option></select>
      <div id="model-hint" class="hint"></div>
    </div>
  </div>

  <label for="instructions">Task / statement to judge (choice optional, noul required, score optional)</label>
  <textarea id="instructions" placeholder="For noul, put the statement to judge, e.g. The customer explicitly requests a refund."></textarea>

  <label for="candidates">Candidates (choice / score)</label>
  <textarea id="candidates" placeholder="One per line: label=description; you can also separate multiple labels on a line with commas or spaces"></textarea>
  <div class="hint" id="cand-hint"></div>

  <div class="inline">
    <input type="checkbox" id="debug" checked>
    <label for="debug">Return raw_candidates (debug)</label>
  </div>

  <button id="run">Submit</button>
</div>

<div class="card" id="result">
  <div class="muted">Results appear here after you submit.</div>
</div>

<script>
const $ = (id) => document.getElementById(id);

function parseChoice(text) {
  const criteria = {};
  text.split('\n').map(s => s.trim()).filter(Boolean).forEach(line => {
    const idx = line.search(/[=＝]/);
    if (idx >= 0) {
      const label = line.slice(0, idx).trim();
      const desc = line.slice(idx + 1).trim();
      if (label) criteria[label] = desc;
    } else {
      line.split(/[,\s，、;；]+/).map(s => s.trim()).filter(Boolean).forEach(l => { criteria[l] = ''; });
    }
  });
  return criteria;
}

function parseLines(text) {
  return text.split('\n').map(s => s.trim()).filter(Boolean);
}

function syncForm() {
  const type = $('qtype').value;
  const cand = $('candidates');
  if (type === 'noul') {
    cand.disabled = true;
    cand.style.background = '#f3f4f6';
    $('cand-hint').textContent = 'noul needs no candidates; write what to judge in the "statement to judge" field.';
  } else {
    cand.disabled = false;
    cand.style.background = '#fff';
    $('cand-hint').textContent = type === 'score'
      ? 'One scale label per line (e.g. 0 / 1 / 2).'
      : 'One "label=description" per line; or separate multiple labels on a line with commas/spaces.';
  }
}

function loadModels() {
  fetch('/v1/models')
    .then(r => r.json())
    .then(data => {
      const sel = $('model');
      const byName = {};
      sel.innerHTML = '';
      (data.models || []).forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.name;
        // Show only the route name in the option (including the real model ID pushes the text past
        // 400px and gets truncated natively); the real model ID goes in the title and the hint line below
        opt.textContent = m.name;
        opt.title = m.model;
        byName[m.name] = m.model;
        if (m.name === data.default) opt.selected = true;
        sel.appendChild(opt);
      });
      const hint = $('model-hint');
      const renderHint = () => { hint.textContent = byName[sel.value] ? ('Actual model: ' + byName[sel.value]) : ''; };
      sel.onchange = renderHint;
      renderHint();
    })
    .catch(() => { $('model').innerHTML = '<option value="">(default)</option>'; });
}

function buildQuestion() {
  const type = $('qtype').value;
  const q = { type: type, instructions: $('instructions').value };
  if (type === 'choice') q.criteria = parseChoice($('candidates').value);
  if (type === 'score') q.scale = parseLines($('candidates').value);
  return q;
}

function bars(distribution) {
  const entries = Object.entries(distribution || {});
  if (!entries.length) return '<div class="muted">No distribution data</div>';
  const max = Math.max.apply(null, entries.map(e => e[1])) || 1;
  return entries.map(([label, value]) => {
    const pct = (value / max * 100).toFixed(1);
    return '<div class="bar-row">' +
      '<span class="bar-label" title="' + label + '">' + label + '</span>' +
      '<span class="bar-track"><span class="bar-fill" style="width:' + pct + '%"></span></span>' +
      '<span class="bar-val">' + (value * 100).toFixed(2) + '%</span>' +
      '</div>';
  }).join('');
}

function rawTable(candidates) {
  if (!candidates || !candidates.length) return '';
  const rows = candidates.map(c =>
    '<tr><td>' + c.token + '</td><td>' + c.logprob.toFixed(4) + '</td></tr>').join('');
  return '<h3>raw_candidates (token / logprob)</h3>' +
    '<table><thead><tr><th>token</th><th>logprob</th></tr></thead><tbody>' + rows + '</tbody></table>';
}

function render(data) {
  let html = '';
  const footer = '<div class="kv muted">route: ' + (data.route || '-') + '  model: ' + (data.model || '-') +
    '  latency: ' + (data.latency_ms != null ? data.latency_ms + ' ms' : '-') +
    '  calls: ' + (data.calls != null ? data.calls : '-') + '</div>';
  Object.entries(data.decisions || {}).forEach(([name, d]) => {
    html += '<h3>Question: ' + name + ' (' + d.type + ')</h3>';
    html += '<div class="kv"><b>label</b>' + (d.label != null ? d.label : '-') + '</div>';
    html += '<div class="kv"><b>confidence</b>' + (d.confidence != null ? d.confidence : '-') + '</div>';
    html += '<div class="kv"><b>coverage</b>' + (d.coverage != null ? d.coverage : '-') + '</div>';
    html += '<div class="kv"><b>reliable</b>' + (d.reliable != null ? d.reliable : '-') + '</div>';
    html += '<div class="kv"><b>generated_token</b>' + (d.generated_token != null ? d.generated_token : '-') + '</div>';
    // None of the three segments includes concurrency queueing; latency_ms minus the sum of this row's three segments is the queue/scheduling overhead
    if (d.timing) {
      const sum = d.timing.prepare_ms + d.timing.call_ms + d.timing.readout_ms;
      html += '<div class="kv"><b>Timing breakdown</b>prepare ' + d.timing.prepare_ms.toFixed(1) +
        '  call ' + d.timing.call_ms.toFixed(1) +
        '  readout ' + d.timing.readout_ms.toFixed(1) +
        '  total ' + sum.toFixed(1) + ' ms</div>';
    }
    if (d.calls != null) html += '<div class="kv"><b>Calls for this question</b>' + d.calls + '</div>';
    if (d.true_probability != null) html += '<div class="kv"><b>true_probability</b>' + d.true_probability + '</div>';
    if (d.expected != null) html += '<div class="kv"><b>expected</b>' + d.expected + '</div>';
    html += '<h3>Candidate probabilities</h3>' + bars(d.distribution);
    html += rawTable(d.raw_candidates);
  });
  html += footer;
  $('result').innerHTML = html;
}

function showError(status, detail) {
  // detail is usually a string (our 502 carries what the model actually produced), but a 422
  // validation error is an object/array, and concatenating it would yield [object Object], so handle
  // the two cases separately and escape the result before showing it in a <pre>.
  const raw = (detail === undefined || detail === null) ? '(no detail)'
    : (typeof detail === 'string' ? detail : JSON.stringify(detail, null, 2));
  const safe = String(raw).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  $('result').innerHTML = '<div class="err">HTTP ' + status +
    '<pre style="white-space:pre-wrap;margin:6px 0 0;font-size:12px">' + safe + '</pre></div>';
}

function run() {
  const payload = {
    state: $('state').value,
    model: $('model').value || null,
    debug: $('debug').checked,
    questions: { q: buildQuestion() }
  };
  $('run').disabled = true;
  fetch('/v1/decide', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
    .then(res => res.json().then(body => ({ ok: res.ok, status: res.status, body: body })))
    .then(r => {
      if (r.ok) { render(r.body); }
      else { showError(r.status, r.body && r.body.detail); }
    })
    .catch(err => { showError('Network error', String(err)); })
    .finally(() => { $('run').disabled = false; });
}

$('qtype').addEventListener('change', syncForm);
$('run').addEventListener('click', run);
syncForm();
loadModels();
</script>
</body>
</html>
"""


@router.get("/debug", response_class=HTMLResponse, include_in_schema=False)
async def debug_page() -> HTMLResponse:
    """Return the single-page debug UI."""
    return HTMLResponse(content=_PAGE)

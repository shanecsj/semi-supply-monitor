/* CHOKEPOINT digest UI. No framework, no build step - the DOM is small
 * enough that a plain render-on-state-change loop is the honest choice. */

const state = {
  digest: null,          // raw /api/digest response
  minSeverity: 0,
  commodity: 'ALL',
  confirmedOnly: false,
  selected: null,
  loading: true,
  error: null,
  running: false,
};

const el = (id) => document.getElementById(id);

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function showToast(text) {
  let toast = document.querySelector('.toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.className = 'toast';
    document.body.appendChild(toast);
  }
  toast.textContent = text;
  toast.classList.add('show');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => toast.classList.remove('show'), 1800);
}

// ---------------------------------------------------------------- fetching

async function loadDigest({ silent = false } = {}) {
  if (!silent) { state.loading = true; state.error = null; render(); }
  try {
    const res = await fetch('/api/digest?days=7');
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    state.digest = await res.json();
    state.loading = false;
    state.error = null;
    if (state.selected && !state.digest.entries.some((e) => e.id === state.selected)) {
      state.selected = null;
    }
  } catch (err) {
    state.loading = false;
    state.error = err.message || String(err);
  }
  render();
}

async function runNow() {
  if (state.running) return;
  state.running = true;
  render();
  try {
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ days: 7 }),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    state.digest = await res.json();
    state.error = null;
    showToast('run complete');
  } catch (err) {
    state.error = `run failed: ${err.message || err}`;
  }
  state.running = false;
  render();
}

// ---------------------------------------------------------------- derived

function visibleEntries() {
  if (!state.digest) return [];
  return state.digest.entries.filter((e) => {
    if (e.severity < state.minSeverity) return false;
    if (state.commodity !== 'ALL' && e.commodity !== state.commodity && e.commodity !== 'BOTH') return false;
    if (state.confirmedOnly && e.speculative) return false;
    return true;
  });
}

function selectedEntry() {
  const visible = visibleEntries();
  if (!visible.length) return null;
  return visible.find((e) => e.id === state.selected) || visible[0];
}

// ---------------------------------------------------------------- rendering

function render() {
  renderTopBar();
  renderFilters();
  renderList();
  renderDetail();
}

function renderTopBar() {
  const d = state.digest;
  el('range-label').textContent = d
    ? `DIGEST · LAST ${d.meta.days}D → ${d.meta.rangeEnd}`
    : 'DIGEST · —';

  const dot = el('sensor-dot');
  const label = el('sensor-label');
  if (state.error) {
    dot.className = 'dot bad';
    label.textContent = 'connection error';
  } else if (state.loading || state.running) {
    dot.className = 'dot busy';
    label.textContent = state.running ? 'running…' : 'loading…';
  } else if (d) {
    dot.className = 'dot';
    label.textContent = `${d.meta.sensorCount} sensors configured · ${d.meta.totalDocuments} documents stored`;
  }

  el('classifier-label').textContent = d ? `classifier: ${d.meta.classifier}` : 'classifier: —';
  el('run-btn').disabled = state.running;
  el('run-btn').textContent = state.running ? 'RUNNING…' : 'RUN NOW';
}

function chip(label, active, onClick) {
  const b = document.createElement('div');
  b.className = 'chip' + (active ? ' active' : '');
  b.textContent = label;
  b.onclick = onClick;
  return b;
}

function renderFilters() {
  const sevDefs = [
    { label: 'ALL', value: 0 }, { label: '3+', value: 3 },
    { label: '4+', value: 4 }, { label: '5', value: 5 },
  ];
  const sevRow = el('severity-chips');
  sevRow.innerHTML = '';
  sevDefs.forEach((def) => sevRow.appendChild(
    chip(def.label, state.minSeverity === def.value, () => {
      state.minSeverity = def.value; render();
    })
  ));

  const comDefs = [{ label: 'ALL' }, { label: 'RAM' }, { label: 'GPU' }];
  const comRow = el('commodity-chips');
  comRow.innerHTML = '';
  comDefs.forEach((def) => comRow.appendChild(
    chip(def.label, state.commodity === def.label, () => {
      state.commodity = def.label; render();
    })
  ));

  const toggle = el('confirmed-toggle');
  toggle.className = 'toggle' + (state.confirmedOnly ? ' active' : '');
  toggle.onclick = () => { state.confirmedOnly = !state.confirmedOnly; render(); };

  const cp = el('chokepoints');
  cp.innerHTML = '';
  const points = state.digest ? state.digest.chokepoints : [];
  const peak = points.length ? points[0].concentration : 1;
  points.forEach((p) => {
    const hot = p.concentration >= peak - 1e-9;
    const row = document.createElement('div');
    row.className = 'chokepoint-row';
    row.innerHTML = `
      <div class="chokepoint-labels">
        <span class="chokepoint-name">${escapeHtml(p.name)}</span>
        <span class="chokepoint-val${hot ? ' hot' : ''}">${p.concentration.toFixed(2)}</span>
      </div>
      <div class="chokepoint-bar"><div class="chokepoint-fill${hot ? ' hot' : ''}" style="width:${p.pct}%"></div></div>
    `;
    cp.appendChild(row);
  });

  el('scheduled-date').textContent = state.digest ? state.digest.meta.nextScheduledDate : '—';
  el('scheduled-note').textContent = state.digest ? state.digest.meta.nextScheduledNote : '';
}

function sevTag(entry) {
  const hi = entry.severity >= 4;
  const mid = entry.severity === 3;
  return `<div class="tag sev${hi ? ' hi' : mid ? ' mid' : ''}">SEV ${entry.severity}/5</div>`;
}

function statusTag(entry) {
  return entry.speculative
    ? '<div class="tag status speculative">SPECULATIVE</div>'
    : '<div class="tag status confirmed">CONFIRMED</div>';
}

function renderList() {
  const list = el('entry-list');
  list.innerHTML = '';

  if (state.error) {
    el('visible-label').textContent = 'CONNECTION ERROR';
    list.innerHTML = `
      <div class="error-list">
        <div class="error-list-title">COULDN'T REACH THE API</div>
        <div class="error-list-body">${escapeHtml(state.error)}. Check that the server is running (<code>python -m semimon.cli serve</code>) and reload.</div>
      </div>`;
    return;
  }

  if (state.loading && !state.digest) {
    el('visible-label').textContent = 'LOADING…';
    list.innerHTML = `
      <div class="loading-list">
        <div class="loading-list-title">BUILDING DIGEST</div>
        <div class="loading-list-body">Clustering and classifying documents from the last period. This can take a few seconds against a live classifier.</div>
      </div>`;
    return;
  }

  const all = state.digest.entries;
  const visible = visibleEntries();
  const sel = selectedEntry();

  el('visible-label').textContent =
    `${visible.length} OF ${all.length} SHOWN · ${state.digest.meta.surfacedThisPeriod} SURFACED THIS PERIOD`;

  if (!visible.length) {
    list.innerHTML = `
      <div class="empty-list">
        <div class="empty-list-title">NOTHING MATCHES THESE FILTERS</div>
        <div class="empty-list-body">That is a normal outcome, not a failure — every source below was polled successfully. Widen the severity floor to see the rest of the period.</div>
      </div>`;
    return;
  }

  visible.forEach((entry) => {
    const card = document.createElement('div');
    card.className = 'entry-card' + (sel && sel.id === entry.id ? ' selected' : '');
    card.style.borderLeftColor = entry.severity >= 4 ? 'var(--accent)'
      : entry.severity === 3 ? 'var(--border-hi)' : 'var(--border-mid)';
    card.onclick = () => { state.selected = entry.id; render(); };

    const sourceLine = `${entry.docCount} docs · ${entry.sources.map((s) => s.name).join(', ')}`;

    card.innerHTML = `
      <div class="entry-top">
        <div class="entry-chips">
          ${sevTag(entry)}
          ${statusTag(entry)}
          <div class="tag">${escapeHtml(entry.riskType)}</div>
          <div class="tag">${escapeHtml(entry.commodity)}</div>
        </div>
        <div class="entry-date">${escapeHtml(entry.date)}</div>
      </div>
      <div class="entry-title">${escapeHtml(entry.title)}</div>
      <div class="entry-path">${escapeHtml(entry.pathShort)}</div>
      <div class="entry-foot">
        <div class="entry-source-line">${escapeHtml(sourceLine)}</div>
        <div class="entry-conf">conf ${entry.confidence}</div>
      </div>
    `;
    list.appendChild(card);
  });
}

function renderDetail() {
  const sel = state.digest && !state.error ? selectedEntry() : null;
  const empty = el('empty-detail');
  const body = el('detail-body');

  if (!sel) {
    empty.hidden = false;
    body.hidden = true;
    return;
  }
  empty.hidden = true;
  body.hidden = false;

  const hops = sel.hops.map((hop, i) => {
    const isFirst = i === 0;
    const isLast = i === sel.hops.length - 1;
    const dot = isFirst ? 'var(--accent)' : isLast ? 'var(--good)' : 'var(--border-hi)';
    const line = isLast ? 'transparent' : 'var(--border-mid)';
    const fg = isFirst || isLast ? 'var(--text)' : 'var(--text-dim)';
    return `
      <div class="hop-row">
        <div class="hop-rail">
          <div class="hop-dot" style="background:${dot}"></div>
          <div class="hop-line" style="background:${line}"></div>
        </div>
        <div class="hop-body">
          <div class="hop-name" style="color:${fg}">${escapeHtml(hop.name)}</div>
          <div class="hop-lag">${escapeHtml(hop.lag)}</div>
        </div>
      </div>`;
  }).join('');

  const sources = sel.sources.map((s) => `
      <div class="source-row">
        <div class="source-name">${escapeHtml(s.name)}</div>
        <div class="source-note">${escapeHtml(s.note)}</div>
      </div>`).join('');

  body.innerHTML = `
    <div class="detail-head">
      <div>DRAFTED ALERT</div>
      <div class="detail-actions">
        <button class="btn" id="copy-btn">COPY</button>
        <button class="btn" id="export-one-btn">EXPORT MD</button>
      </div>
    </div>
    <div class="detail-scroll">
      <div>
        <div class="detail-title">${escapeHtml(sel.title)}</div>
        <div class="detail-tags">
          ${sevTag(sel)}
          ${statusTag(sel)}
          <div class="tag">${escapeHtml(sel.riskType)}</div>
          <div class="tag">HORIZON ${escapeHtml(sel.horizon)}</div>
        </div>
      </div>

      <div class="detail-draft">${escapeHtml(sel.draft)}</div>

      ${hops.length ? `
      <div class="panel">
        <div class="label">PROPAGATION PATH · GRAPH TRAVERSAL, NOT GENERATED</div>
        <div class="hops">${hops}</div>
      </div>` : ''}

      <div class="section">
        <div class="label">EVIDENCE QUOTE · VERBATIM</div>
        <div class="quote-block">${escapeHtml(sel.quote)}</div>
      </div>

      <div class="section">
        <div class="label">MARKET ANNOTATION · ATTACHED, NOT INTERPRETED</div>
        <div class="market-line">${escapeHtml(sel.market)}</div>
      </div>

      <div class="section section-bordered">
        <div class="label">SOURCES · ${sel.docCount} DOCUMENTS CLUSTERED</div>
        ${sources}
      </div>
    </div>
  `;

  el('copy-btn').onclick = () => {
    navigator.clipboard.writeText(sel.draft).then(() => showToast('copied'));
  };
  el('export-one-btn').onclick = () => downloadMarkdown(entryMarkdown(sel), `${sel.id}.md`);
}

// ---------------------------------------------------------------- export

function entryMarkdown(entry) {
  const flags = [`severity ${entry.severity}/5`, entry.speculative ? 'speculative' : 'confirmed',
    entry.riskType.toLowerCase(), entry.commodity.toLowerCase()];
  const lines = [
    `### ${entry.title}`, '',
    `\`${flags.join(' | ')}\`  ·  ${entry.docCount} source(s)`, '',
    entry.draft, '',
  ];
  if (entry.pathShort) lines.push(`**Path:** ${entry.pathShort}`, '');
  if (entry.market) lines.push(`**Market:** ${entry.market}`, '');
  if (entry.quote) lines.push(`> ${entry.quote}`, '');
  if (entry.urls && entry.urls.length) {
    lines.push('Sources: ' + entry.urls.map((u) => `<${u}>`).join(' · '), '');
  }
  return lines.join('\n');
}

function digestMarkdown() {
  const d = state.digest;
  const entries = visibleEntries();
  const lines = [
    '# RAM & GPU supply-chain digest', `_last ${d.meta.days} days -> ${d.meta.rangeEnd}_`, '',
  ];
  entries.forEach((e) => lines.push(entryMarkdown(e), ''));
  return lines.join('\n');
}

function downloadMarkdown(text, filename) {
  const blob = new Blob([text], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------- init

el('run-btn').onclick = runNow;
el('export-btn').onclick = () => {
  if (!state.digest) return;
  downloadMarkdown(digestMarkdown(), `chokepoint-digest-${state.digest.meta.rangeEnd}.md`);
};

loadDigest();

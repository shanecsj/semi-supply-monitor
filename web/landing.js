/* Wires the live strip + hero preview on the landing page to /api/digest.
 * No framework: this page is static marketing copy with a handful of numbers
 * that should never silently drift out of sync with the running pipeline. */

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function timeAgo(iso) {
  if (!iso) return 'never';
  const then = new Date(iso.replace(' ', 'T') + (iso.endsWith('Z') ? '' : 'Z'));
  const mins = Math.max(0, Math.round((Date.now() - then.getTime()) / 60000));
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

async function loadLive() {
  try {
    const res = await fetch('/api/digest?days=7');
    if (!res.ok) throw new Error(`${res.status}`);
    const d = await res.json();
    renderStrip(d);
    renderPreview(d);
  } catch (err) {
    document.getElementById('preview-body').innerHTML =
      `<div class="preview-title">Live data unavailable</div>
       <div class="preview-desc">Couldn't reach the API (${escapeHtml(err.message || err)}). The digest still works from the CLI — see the README.</div>`;
  }
}

function renderStrip(d) {
  document.getElementById('stat-severe').innerHTML = `${d.meta.openSevere}<div class="stat-dot"></div>`;
  document.getElementById('stat-docs').textContent = d.meta.documentsInWindow;
  document.getElementById('stat-surfaced').textContent = d.meta.surfacedThisPeriod;
  document.getElementById('stat-nodes').textContent = d.meta.nodesTracked;
  document.getElementById('stat-poll').textContent = timeAgo(d.meta.lastPollAt);
  document.getElementById('support-nodes').textContent = `${d.meta.nodesTracked} / ${d.meta.edgesTracked}`;
}

function renderPreview(d) {
  const body = document.getElementById('preview-body');
  const dateLabel = document.getElementById('preview-date');
  const entries = d.entries || [];

  if (!entries.length) {
    dateLabel.textContent = 'no data yet';
    body.className = 'preview-body empty';
    body.innerHTML = `
      <div class="preview-title">No entries surfaced yet</div>
      <div class="preview-desc">That's the normal state for a fresh install — run <code class="mono" style="color:var(--text-dim)">python -m semimon.cli run</code> or hit RUN NOW in the digest to poll every sensor.</div>
    `;
    return;
  }

  const top = entries[0];
  dateLabel.textContent = top.date || '';
  body.className = 'preview-body';

  const sevTag = top.severity >= 4
    ? `<div class="tag sev-hi">SEV ${top.severity}/5</div>`
    : `<div class="tag">SEV ${top.severity}/5</div>`;
  const statusTag = top.speculative
    ? '<div class="tag speculative">SPECULATIVE</div>'
    : '<div class="tag confirmed">CONFIRMED</div>';

  const pathHtml = top.pathShort
    ? `<div class="preview-path-wrap">
         <div class="stat-label">PROPAGATION</div>
         <div class="preview-path">${escapeHtml(top.pathShort)}</div>
       </div>`
    : '';
  const quoteHtml = top.quote
    ? `<div class="preview-quote">&ldquo;${escapeHtml(top.quote)}&rdquo;</div>`
    : '';

  body.innerHTML = `
    <div class="tag-row">
      ${sevTag}
      ${statusTag}
      <div class="tag">${escapeHtml(top.riskType)}</div>
      <div class="tag">${escapeHtml(top.commodity)}</div>
    </div>
    <div class="preview-title">${escapeHtml(top.title)}</div>
    <div class="preview-desc">${escapeHtml((top.draft || '').split(/(?<=[.!?])\s+/).slice(0, 2).join(' '))}</div>
    ${pathHtml}
    ${quoteHtml}
  `;
}

loadLive();

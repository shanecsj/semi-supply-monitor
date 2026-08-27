/* CHOKEPOINT chat UI. Plain fetch + DOM append - no framework, matching the
 * rest of web/. Each turn is one POST to /api/chat; the backend (chat.py)
 * owns retrieval, grounding and citation, this file only renders it. */

const thread = document.getElementById('chat-thread');
const form = document.getElementById('chat-form');
const input = document.getElementById('chat-input');
const sendBtn = document.getElementById('chat-send');
const suggest = document.getElementById('chat-suggest');

let asking = false;

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function scrollToEnd() {
  thread.scrollTop = thread.scrollHeight;
}

function addTurn(role, html) {
  const row = document.createElement('div');
  row.className = `chat-turn ${role}`;
  row.innerHTML = html;
  thread.appendChild(row);
  scrollToEnd();
  return row;
}

function renderSources(sources) {
  if (!sources || !sources.length) return '';
  const items = sources.slice(0, 6).map((s) => `
    <div class="chat-source">
      <span class="chat-source-idx">[${s.index}]</span>
      ${s.url
        ? `<a href="${escapeHtml(s.url)}" target="_blank" rel="noopener">${escapeHtml(s.title)}</a>`
        : `<span>${escapeHtml(s.title)}</span>`}
      <span class="chat-source-src">${escapeHtml(s.source || '')}</span>
    </div>`).join('');
  return `<div class="chat-sources"><div class="label">SOURCES</div>${items}</div>`;
}

async function ask(question) {
  if (asking || !question.trim()) return;
  asking = true;
  sendBtn.disabled = true;
  suggest.style.display = 'none';

  addTurn('user', `<div class="chat-bubble">${escapeHtml(question)}</div>`);
  const pending = addTurn('assistant', `<div class="chat-bubble pending">thinking&hellip;</div>`);

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    const data = await res.json();
    pending.innerHTML = `
      <div class="chat-bubble">${escapeHtml(data.answer).replace(/\n/g, '<br>')}</div>
      ${renderSources(data.sources)}
    `;
  } catch (err) {
    pending.innerHTML = `<div class="chat-bubble error">Couldn't reach the API (${escapeHtml(err.message || err)}). Check that the server is running.</div>`;
  }

  scrollToEnd();
  asking = false;
  sendBtn.disabled = false;
}

form.addEventListener('submit', (e) => {
  e.preventDefault();
  const q = input.value;
  input.value = '';
  ask(q);
});

suggest.querySelectorAll('.chip').forEach((chip) => {
  chip.addEventListener('click', () => ask(chip.dataset.q));
});

import './style.css';
import { advance, sounding, playbackRatio, Piano } from './player.js';

const $ = id => document.getElementById(id);
$('app').innerHTML = `
  <header><a class="brand" href="/">HMSM<span>THE ROLL ROOM</span></a><span class="edition">Historical sound, set in motion.</span><span class="local"><i></i> LOCAL SESSION</span></header>
  <main>
    <section class="intro"><div><div class="eyebrow">A DIGITAL PLAYER PIANO</div><h1>Let the paper sing.</h1><p>Open a roll scan. Watch its music come to life.</p></div><div class="archive-mark">EST. <b>♪</b> 2026<br><span>DIGITAL ORGANOLOGY</span></div></section>
    <div class="workspace">
      <aside>
        <div class="section-label">01 <span>THE RECORDING</span></div>
        <form id="setup">
          <label class="file-picker" id="drop"><input id="file" type="file" accept=".tif,.tiff,.png,.jpg,.jpeg,.bmp,.webp" required><span class="file-icon">↥</span><strong id="filename">Choose a roll scan</strong><span id="filesize">or drop an image here</span><small>TIFF, PNG, JPEG · up to 4 GiB</small></label>
          <label>Roll format<select id="profile" required></select></label>
          <label>Marked roll tempo<div class="input-unit"><input id="tempo" type="number" min="1" max="200" value="60" required><span>÷ 10 ft/min</span></div></label>
          <details><summary>Scan settings</summary><label>Skip head rows<input id="skip" type="number" min="0" step="1" value="0" required></label><label>Resolution (DPI)<input id="dpi" type="number" min="1" max="10000" placeholder="From scan, otherwise 300"></label></details>
          <button class="primary" id="begin" type="submit">Digitize & load roll <span>→</span></button>
        </form>
        <div class="job-progress"><div class="section-label">02 <span>DIGITIZATION</span></div><div id="status" role="status" aria-live="polite">Waiting for a roll</div><progress id="progress" max="100" value="0"></progress><div class="progress-meta"><span id="count">0 notes detected</span><span id="percent">—</span></div></div>
        <p class="hint">Music becomes playable as the scan is read. The original image stays on this computer.</p>
        <p class="error" id="error" role="alert"></p>
        <a id="download" class="download" hidden>↓ Download MIDI</a><button id="different" class="text-button" hidden>Load different roll</button><input id="replacement-file" type="file" accept=".tif,.tiff,.png,.jpg,.jpeg,.bmp,.webp" hidden><button id="close" class="text-button" hidden>Close roll</button>
      </aside>
      <section class="player-panel" aria-label="Piano roll player">
        <div class="machine">
          <div class="machine-header"><span>H M S M</span><span class="machine-subtitle">PNEUMATIC ROLL PLAYER</span><span class="serial">№ 001</span></div>
          <div class="spool top"><div class="spindle"></div><div class="paper-cylinder"></div><div class="spindle"></div></div>
          <div class="roll-window"><canvas id="roll" aria-label="Scanned piano roll with detected notes highlighted"></canvas><div id="empty"><div class="empty-note">♩</div><h2>A little history.<br>A living sound.</h2><p>Your roll will pass through here.</p></div><div class="tracker"><div class="tracker-holes"></div></div><div class="tracker-label">TRACKER BAR</div></div>
          <div class="spool bottom"><div class="spindle"></div><div class="paper-cylinder"></div><div class="spindle"></div></div>
          <div id="keys" class="keyboard" aria-hidden="true"></div>
        </div>
        <div class="transport">
          <div class="transport-top"><button id="restart" class="icon-button" aria-label="Rewind to beginning" disabled>↤</button><button id="play" disabled>▶ <span>Play</span></button><span id="clock">0:00 <span>/ 0:00</span></span><span id="playstate">NO ROLL LOADED</span></div>
          <label class="seek-label"><span class="sr-only">Playback position</span><input id="seek" type="range" min="0" max="1" value="0" step="1" disabled></label>
          <div class="transport-bottom"><span><i class="legend"></i> SOUNDING NOTES</span><div class="tempo-control"><div class="tempo-plate"><label for="speed">Playback tempo</label><output id="speedvalue" for="speed">6.00 ft/min</output><small>Drag dial · arrow keys to fine tune</small></div><div class="dial"><span class="dial-face" aria-hidden="true"></span><input id="speed" aria-label="Playback tempo in feet per minute" type="range" min="1.5" max="12" step="0.01" value="6"></div></div></div>
        </div>
        <p class="caption">The paper moves. The tracker listens. <span>Live synth preview · full expression in MIDI export</span></p>
      </section>
    </div>
    <footer><span>HISTORICAL MUSICAL STORAGE MEDIA</span><span>From the archive to the air.</span></footer>
  </main>`;

let file, job = null, state = null, row = 0, playing = false, buffering = false;
let last = performance.now(), xhr, opening = false;
const piano = new Piano(), images = new Map();
const canvas = $('roll'), ctx = canvas.getContext('2d');
const keyElements = new Map();
for (let tone = 21; tone <= 108; tone++) {
  const key = document.createElement('div');
  key.className = [1, 3, 6, 8, 10].includes(tone % 12) ? 'key black' : 'key white';
  key.dataset.tone = tone; $('keys').append(key); keyElements.set(tone, key);
}
function error(message = '') { $('error').textContent = message; }
async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail;
    try { detail = (await response.json()).detail; } catch { /* use HTTP status */ }
    throw new Error(typeof detail === 'string' ? detail : `Request failed (${response.status})`);
  }
  return response.json();
}
api('/api/profiles').then(profiles => {
  profiles.forEach(name => $('profile').add(new Option(name.replaceAll('_', ' '), name)));
  $('profile').value = profiles.includes('phonola') ? 'phonola' : profiles[0];
  return api('/api/session');
}).then(saved => {
  if (!saved) return;
  job = saved.id; state = { version: -1 };
  $('filename').textContent = saved.options.filename;
  $('filesize').textContent = 'Restored local session';
  $('profile').value = saved.options.profile;
  $('tempo').value = saved.options.tempo;
  resetTempo();
  $('skip').value = saved.options.skip_rows;
  $('dpi').value = saved.options.dpi ?? '';
  lockForm(true); $('different').hidden = $('close').hidden = false; poll(job);
}).catch(e => error(`Could not reach the local server. ${e.message}`));
function selectFile(selected) {
  if (!selected || job || opening) return;
  file = selected;
  $('filename').textContent = file.name;
  $('filesize').textContent = `${(file.size / 1024 ** 2).toFixed(1)} MiB selected`;
  error();
}
$('file').onchange = e => selectFile(e.target.files[0]);
$('drop').ondragover = e => { e.preventDefault(); };
$('drop').ondrop = e => {
  e.preventDefault();
  if (!job && !opening) { $('file').files = e.dataTransfer.files; selectFile(e.dataTransfer.files[0]); }
};
function lockForm(locked) {
  $('setup').querySelectorAll('input, select, button').forEach(el => el.disabled = locked);
}
$('setup').onsubmit = async e => {
  e.preventDefault();
  if (!file || job || opening) return;
  if (file.size > 4 * 1024 ** 3) return error('Please choose a scan smaller than 4 GiB.');
  error(); opening = true; resetTempo();
  const options = { filename: file.name, profile: $('profile').value, tempo: +$('tempo').value,
    skip_rows: +$('skip').value, dpi: $('dpi').value ? +$('dpi').value : null };
  lockForm(true);
  try {
    state = await api('/api/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(options) });
    job = state.id;
    $('status').textContent = 'Copying scan to the local workspace…';
    await new Promise((resolve, reject) => {
      xhr = new XMLHttpRequest();
      xhr.open('PUT', `/api/jobs/${job}/scan`);
      xhr.setRequestHeader('Content-Type', 'application/octet-stream');
      xhr.upload.onprogress = e => {
        if (e.lengthComputable) { $('progress').value = e.loaded / e.total * 100; $('percent').textContent = `${Math.round(e.loaded / e.total * 100)}%`; }
      };
      xhr.onload = () => xhr.status < 300 ? resolve() : reject(new Error('Upload failed. Close the roll and try again.'));
      xhr.onerror = () => reject(new Error('Upload connection lost.'));
      xhr.onabort = () => reject(new Error('Upload cancelled.'));
      xhr.send(file);
    });
    $('progress').value = 0;
    poll(job);
  } catch (e) { error(e.message); if (!job) lockForm(false); }
  finally { opening = false; $('different').hidden = $('close').hidden = !job; }
};
async function poll(id) {
  if (id !== job) return;
  try {
    const next = await api(`/api/jobs/${id}?version=${state?.version ?? -1}`);
    if (id !== job) return;
    if (next.status) {
      state = next;
      if (state.origin_row !== null && !row) row = state.origin_row;
      const ready = state.origin_row !== null && state.safe_row > state.origin_row;
      $('play').disabled = $('restart').disabled = $('seek').disabled = !ready;
      $('seek').min = state.origin_row ?? 0;
      $('seek').max = Math.max(state.origin_row ?? 0, state.safe_row);
      $('count').textContent = `${state.notes.filter(n => n[2] > 0).length.toLocaleString()} notes detected`;
      const percent = state.status === 'complete' ? 100 : state.rows_processed / (state.height || 1) * 100;
      $('progress').value = percent; $('percent').textContent = `${Math.round(percent)}%`;
      $('status').textContent = { processing: 'Reading the roll…', queued: 'Preparing scan…', complete: 'Digitization complete', error: 'Could not digitize this scan' }[state.status] ?? state.status;
      $('empty').hidden = state.tiles.length > 0;
      if (state.status === 'error') { error(state.error); pause(); }
      if (state.status === 'complete') {
        $('download').hidden = false; $('download').href = `/api/jobs/${id}/midi`;
      }
    }
    error(state.status === 'error' ? state.error : '');
  } catch (e) { if (id === job) error(`Connection interrupted; retrying. ${e.message}`); }
  if (id === job && !['complete', 'error', 'cancelled'].includes(state?.status)) setTimeout(() => poll(id), 500);
}
function pause() { playing = false; buffering = false; piano.stop(); $('play').innerHTML = '▶ <span>Play</span>'; }
$('play').onclick = async () => {
  if (playing) return pause();
  try {
    await piano.unlock();
    if (state.status === 'complete' && row >= state.safe_row) row = state.origin_row;
    playing = true; last = performance.now(); $('play').innerHTML = 'Ⅱ <span>Pause</span>';
  } catch (e) { error(`Audio could not start: ${e.message}`); }
};
$('restart').onclick = () => { row = state.origin_row; piano.stop(); };
$('seek').oninput = e => { row = +e.target.value; piano.stop(); };
function updateTempo() {
  const control = $('speed'), value = +control.value;
  $('speedvalue').textContent = `${value.toFixed(2)} ft/min`;
  control.setAttribute('aria-valuetext', `${value.toFixed(2)} feet per minute`);
  control.parentElement.style.setProperty('--angle', `${-135 + 270 * (value - +control.min) / (+control.max - +control.min)}deg`);
}
function resetTempo() {
  const base = +$('tempo').value / 10;
  $('speed').min = base * 0.25;
  $('speed').max = base * 2;
  $('speed').step = 'any';
  $('speed').value = base;
  updateTempo();
}
function speedRatio() { return playbackRatio(+$('speed').value, +$('tempo').value); }
$('tempo').addEventListener('input', () => { if ($('tempo').validity.valid) resetTempo(); });
$('speed').oninput = () => { piano.stop(); updateTempo(); };
// Retain native slider semantics and keyboard access; vertical drag turns the dial.
let dialDrag;
$('speed').onpointerdown = e => {
  if (e.button !== 0) return;
  e.preventDefault(); e.currentTarget.focus(); e.currentTarget.setPointerCapture(e.pointerId);
  dialDrag = { y: e.clientY, value: +e.currentTarget.value };
};
$('speed').onpointermove = e => {
  if (!dialDrag) return;
  const control = e.currentTarget;
  control.value = dialDrag.value + (dialDrag.y - e.clientY) * (+control.max - +control.min) / 180;
  control.dispatchEvent(new Event('input'));
};
$('speed').onpointerup = $('speed').onpointercancel = () => { dialDrag = null; };
$('speed').onkeydown = e => {
  const direction = { ArrowUp: 1, ArrowRight: 1, ArrowDown: -1, ArrowLeft: -1 }[e.key];
  if (!direction) return;
  e.preventDefault();
  e.currentTarget.value = +e.currentTarget.value + direction * (e.shiftKey ? 0.1 : 0.01);
  e.currentTarget.dispatchEvent(new Event('input'));
};
resetTempo();
document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
async function closeRoll() {
  pause(); $('different').disabled = $('close').disabled = true;
  try {
    await api(`/api/jobs/${job}`, { method: 'DELETE' });
    job = state = null; row = 0; images.clear(); lockForm(false);
    $('different').hidden = $('close').hidden = $('download').hidden = true; $('empty').hidden = false;
    $('play').disabled = $('restart').disabled = $('seek').disabled = true;
    $('status').textContent = 'Waiting for a roll'; $('progress').value = 0;
    $('percent').textContent = '—'; $('count').textContent = '0 notes detected'; error();
    file = null; $('file').value = '';
    $('filename').textContent = 'Choose a roll scan';
    $('filesize').textContent = 'or drop an image here';
    return true;
  } catch (e) { error(e.message); return false; }
  finally { $('different').disabled = $('close').disabled = false; }
}
$('close').onclick = closeRoll;
// Open the picker inside the click gesture, before asynchronous cleanup, so
// Firefox retains user activation. Cancelling the picker keeps this roll open.
$('different').onclick = () => $('replacement-file').click();
$('replacement-file').onchange = async e => {
  const files = e.target.files;
  const selected = files[0];
  if (selected && await closeRoll()) {
    const transfer = new DataTransfer();
    transfer.items.add(selected);
    $('file').files = transfer.files;
    selectFile(selected);
    $('begin').focus();
  }
  $('replacement-file').value = '';
};
function time(rows) {
  const seconds = Math.max(0, Math.floor(rows * (state?.seconds_per_row || 0)));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
}
function draw(active) {
  const width = canvas.clientWidth, height = canvas.clientHeight, dpr = devicePixelRatio || 1;
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
  if (!state?.width) return;
  const scale = width / state.width, tracker = height * 0.72;
  // Earlier rows fall below the bar; upcoming paper feeds down from the spool.
  const y = scanRow => tracker - (scanRow - row) * scale;
  const low = row - (height - tracker) / scale, high = row + tracker / scale;
  const visible = state.tiles.filter(tile => tile.stop >= low && tile.start <= high);
  const wanted = new Set(visible.map(t => t.index));
  // Keep only viewport tiles and neighbours resident in the browser.
  for (const key of images.keys()) if (!wanted.has(key)) images.delete(key);
  for (const tile of visible) {
    let img = images.get(tile.index);
    if (!img) { img = new Image(); img.src = `/api/jobs/${job}/tiles/${tile.index}.jpg`; images.set(tile.index, img); }
    if (img.complete && img.naturalWidth) {
      ctx.save(); ctx.translate(0, y(tile.start)); ctx.scale(1, -1);
      ctx.drawImage(img, 0, 0, width, (tile.stop - tile.start) * scale); ctx.restore();
    }
  }
  const activeKeys = new Set(active.map(n => `${n[0]}:${n[2]}`));
  for (const [start, end, tone] of state.notes) {
    if (tone <= 0 || end < low || start > high) continue;
    const track = state.tracks.find(t => t[2] === tone);
    if (!track) continue;
    const isActive = activeKeys.has(`${start}:${tone}`);
    ctx.fillStyle = isActive ? 'rgba(249,187,80,0.82)' : 'rgba(97,202,184,0.28)';
    ctx.strokeStyle = isActive ? '#ffe0a1' : 'rgba(77,167,151,0.8)';
    // Follow paper drift using per-row edges, including across tile boundaries.
    for (const tile of visible) {
      if (end < tile.start || start > tile.stop || !tile.edges.length) continue;
      for (let i = 0; i < tile.edges.length; i++) {
        const [edgeRow, left, right] = tile.edges[i];
        const a = Math.max(start, edgeRow), b = Math.min(end, tile.edges[i + 1]?.[0] ?? tile.stop);
        if (b <= a) continue;
        const x = (left + (right - left) * track[0]) * scale;
        const w = Math.max(2, (right - left) * (track[1] - track[0]) * scale);
        ctx.fillRect(x, y(b), w, Math.max(1, (b - a) * scale));
        ctx.strokeRect(x, y(b), w, Math.max(1, (b - a) * scale));
      }
    }
  }
}
function frame(now) {
  const elapsed = Math.max(0, (now - last) / 1000); last = now;
  if (playing && state) {
    row = advance(row, elapsed, speedRatio(), state.seconds_per_row, state.safe_row);
    buffering = row >= state.safe_row;
    if (buffering && state.status === 'complete') pause();
  }
  const active = playing && !buffering ? sounding(state.notes, row) : [];
  if (playing && !buffering) piano.schedule(state.notes, row, state.seconds_per_row, speedRatio(), state.safe_row);
  else piano.stop();
  const tones = new Set(active.map(n => n[2]));
  for (const [tone, element] of keyElements) element.classList.toggle('active', tones.has(tone));
  draw(active);
  $('seek').value = row;
  $('clock').innerHTML = `${time(row - (state?.origin_row || 0))} <span>/ ${time((state?.safe_row || 0) - (state?.origin_row || 0))}</span>`;
  $('playstate').textContent = !job ? 'NO ROLL LOADED' : buffering ? 'BUFFERING…' : playing ? 'PLAYING' : state?.status === 'complete' ? 'READY TO PLAY' : 'LIVE DIGITIZATION';
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

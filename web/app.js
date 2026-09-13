const $ = (id) => document.getElementById(id);
let currentView = 'overview', latest = {}, planDraft = null, alertRows = [], lastFrame = 0;
function el(tag, text, cls) { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (cls) node.className = cls; return node; }
function notice(message, error = false) { $('notice').textContent = message; $('notice').className = error ? 'error' : ''; $('notice').hidden = false; }
async function api(path, options = {}) {
  const response = await fetch('/api' + path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail || body));
  return body;
}
function post(path, data = {}) { return api(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)}); }
['image', 'video'].forEach(kind => $('sample-' + kind).addEventListener('click', event => action(event.currentTarget, async () => {
  notice('Loading your models and supplied ' + kind + '…');
  const result = await post('/samples/' + kind);
  $('shelf-name').value = 'sample'; $('plan-name').value = 'sample';
  notice(result.message || 'Supplied sample loaded.'); await refresh();
})));
async function action(button, fn) {
  button.disabled = true;
  try { await fn(); } catch (error) { notice(error.message, true); } finally { button.disabled = false; }
}
const titles = {overview: 'Shelf monitor', products: 'Products', planogram: 'Planogram', alerts: 'Alert history', models: 'Models & setup'};
document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => {
  currentView = button.dataset.view;
  document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === currentView));
  document.querySelectorAll('[data-view]').forEach(v => v.classList.toggle('active', v === button));
  $('page-title').textContent = titles[currentView];
  refresh().catch(error => notice(error.message, true));
}));
$('mode').addEventListener('change', () => {
  const demo = $('mode').value === 'demo';
  $('source-fields').hidden = demo; $('demo-controls').hidden = !demo;
});
$('start').addEventListener('click', event => action(event.currentTarget, async () => {
  let source = $('source').value.trim();
  if ($('mode').value === 'video' && $('video-upload').files.length) {
    const body = new FormData(); body.append('file', $('video-upload').files[0]);
    notice('Uploading video…'); const result = await api('/media', {method: 'POST', body}); source = result.path; $('source').value = source;
  }
  await post('/monitor/start', {mode: $('mode').value, source, plan: $('shelf-name').value.trim()});
  notice($('mode').value === 'demo' ? 'Synthetic demo started. Try the shelf scenarios.' : 'Source started. Waiting for the first analysis.'); await refresh();
}));
$('stop').addEventListener('click', event => action(event.currentTarget, async () => { await post('/monitor/stop'); notice('Monitoring stopped. You can update models, references, or shelf positions.'); await refresh(); }));
document.querySelectorAll('[data-phase]').forEach(button => button.addEventListener('click', () => action(button, async () => {
  await post('/demo/' + button.dataset.phase);
  document.querySelectorAll('[data-phase]').forEach(b => b.classList.toggle('selected', b === button));
})));
$('analyze').addEventListener('click', event => action(event.currentTarget, async () => {
  if (!$('image-upload').files.length) throw new Error('Choose a shelf image first.');
  const body = new FormData(); body.append('file', $('image-upload').files[0]); body.append('plan', $('shelf-name').value.trim());
  notice('Analyzing your shelf image…'); await api('/analyze', {method: 'POST', body}); notice('Image analyzed. Review the boxes before generating a planogram.'); await refresh();
}));
$('product-form').addEventListener('submit', event => {
  event.preventDefault(); const button = event.currentTarget.querySelector('button'); const body = new FormData(event.currentTarget);
  action(button, async () => { const result = await api('/catalog', {method: 'POST', body}); notice(result.message); $('product-form').reset(); await renderCatalog(); });
});
$('rebuild-index').addEventListener('click', event => action(event.currentTarget, async () => { notice('Building product embeddings and the reference index…'); const result = await post('/catalog/rebuild'); notice(`${result.images} images across ${result.products} products. ${result.message}`); await refresh(); }));
$('validate-models').addEventListener('click', event => action(event.currentTarget, async () => { notice('Loading and checking the model bundle…'); const result = await post('/models/validate'); notice(result.message); await refresh(); }));
async function renderCatalog() {
  const products = await api('/catalog'); const list = $('catalog-list'); list.replaceChildren(); $('product-total').textContent = products.length + ' products';
  if (!products.length) { list.append(el('p', 'No products registered yet. Add your first reference images to get started.', 'empty-text')); return; }
  products.forEach(product => { const row = el('div', undefined, 'catalog-item'); row.append(el('div', '▦', 'sku-icon')); const info = el('div'); info.append(el('h4', product.name), el('small', 'SKU ' + product.sku)); row.append(info, el('span', product.images + ' images', 'badge')); list.append(row); });
}
function renderPlan() {
  const body = $('plan-rows'); body.replaceChildren();
  for (const [index, slot] of (planDraft?.slots || []).entries()) {
    const row = el('tr');
    for (const [key, value] of [['id', slot.id], ['expected_sku', slot.expected_sku], ...slot.bbox.map((v, i) => [i, v])]) {
      const cell = el('td'); const input = el('input'); input.value = value; input.setAttribute('aria-label', `${slot.id || 'New slot'} ${typeof key === 'number' ? ['left', 'top', 'right', 'bottom'][key] : key}`);
      if (typeof key === 'number') { input.type = 'number'; input.step = '0.001'; input.min = '0'; input.max = '1'; }
      input.disabled = $('plan-name').value === 'demo';
      input.addEventListener('input', () => { if (typeof key === 'number') slot.bbox[key] = Number(input.value); else slot[key] = input.value; }); cell.append(input); row.append(cell);
    }
    const cell = el('td'); const remove = el('button', '×', 'remove-slot'); remove.setAttribute('aria-label', 'Remove ' + slot.id); remove.disabled = $('plan-name').value === 'demo'; remove.onclick = () => { planDraft.slots.splice(index, 1); renderPlan(); }; cell.append(remove); row.append(cell); body.append(row);
  }
  $('plan-note').textContent = planDraft ? `${planDraft.slots.length} positions. Review every expected SKU before saving.` : 'No saved planogram for this shelf.';
}
$('load-plan').addEventListener('click', event => action(event.currentTarget, async () => { planDraft = await api('/planogram/' + encodeURIComponent($('plan-name').value)); renderPlan(); }));
$('generate-plan').addEventListener('click', event => action(event.currentTarget, async () => { planDraft = await post('/planogram/' + encodeURIComponent($('plan-name').value) + '/generate'); renderPlan(); notice('Draft generated. Assign any missing SKU IDs and review the positions before saving.'); }));
$('add-slot').onclick = () => { if ($('plan-name').value === 'demo') return notice('Demo planograms are read-only.', true); planDraft ||= {slots: []}; planDraft.slots.push({id: 'slot-' + (planDraft.slots.length + 1), expected_sku: '', row: 1, bbox: [0, 0, .1, .1]}); renderPlan(); };
$('save-plan').addEventListener('click', event => action(event.currentTarget, async () => { if (!planDraft) throw new Error('Generate or load a planogram first.'); const result = await api('/planogram/' + encodeURIComponent($('plan-name').value), {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(planDraft)}); notice(result.message); }));
function renderAlerts() {
  const filter = $('alert-filter').value; const rows = alertRows.filter(row => filter === 'all' || (filter === 'demo' ? row.source === 'demo' : row.source !== 'demo'));
  const list = $('alerts-list'); list.replaceChildren();
  if (!rows.length) { list.append(el('p', 'No alerts here yet. Stable shelf changes will appear while monitoring.', 'empty-text')); return; }
  for (const row of rows) {
    const item = el('div', undefined, 'alert'); item.append(el('div', row.state === 'OOS' ? '−' : '↔', 'alert-mark')); const info = el('div');
    info.append(el('h4', `${row.source} / ${row.slot} · ${row.state === 'OOS' ? 'Missing product' : 'Misplaced product'}`), el('p', `Expected ${row.expected || '—'} · Observed ${row.observed || 'empty'}${row.source === 'demo' ? ' · Synthetic demo' : ''}`), el('small', new Date(row.created).toLocaleString() + (row.note ? ' · ' + row.note : '')));
    item.append(info); const actions = el('div', undefined, 'alert-action');
    if (row.resolved) actions.append(el('span', 'Resolved', 'badge live'));
    else if (row.acknowledged) actions.append(el('span', 'Acknowledged', 'badge'));
    else { const button = el('button', 'Acknowledge', 'secondary'); button.onclick = () => action(button, async () => { await post(`/alerts/${row.id}/acknowledge`, {note: 'Reviewed in dashboard'}); notice('Alert acknowledged. It will resolve when the shelf is corrected.'); await refresh(); }); actions.append(button); }
    item.append(actions); list.append(item);
  }
}
$('alert-filter').onchange = renderAlerts;
function renderMonitor(result) {
  latest = result; const detections = result.detections || [], cells = result.cells || [];
  $('metric-products').textContent = result.updated ? detections.length : '—'; $('metric-ok').textContent = cells.length ? cells.filter(c => c.state === 'OK').length : '—';
  $('metric-attention').textContent = cells.length ? cells.filter(c => ['OOS', 'MISPLACED'].includes(c.state)).length : '—'; $('metric-speed').textContent = result.inference_ms != null ? result.inference_ms + ' ms' : '—';
  $('feed-message').textContent = result.message || 'Ready when you are.';
  const badge = $('feed-badge'); badge.textContent = result.mode === 'demo' ? (result.running ? 'Synthetic demo' : 'Demo stopped') : result.running ? 'Monitoring' : result.mode === 'image' ? 'Image result' : 'Stopped'; badge.className = 'badge ' + (result.mode === 'demo' ? 'demo' : result.running ? 'live' : '');
  $('frame-age').textContent = result.updated ? Math.max(0, Math.round(Date.now()/1000 - result.updated)) + 's ago' : '';
  if (result.updated && result.updated !== lastFrame) { $('shelf-frame').src = '/api/frame?t=' + result.updated; $('shelf-frame').hidden = false; $('frame-empty').hidden = true; lastFrame = result.updated; }
  if (!result.updated) { $('shelf-frame').hidden = true; $('frame-empty').hidden = false; lastFrame = 0; }
  $('sample-image').disabled = !!result.running; $('sample-video').disabled = !!result.running;
  $('start').disabled = !!result.running;
  const grid = $('position-grid'); grid.replaceChildren();
  if (!cells.length) grid.append(el('p', 'Start the demo, or create a planogram for your shelf.', 'empty-text'));
  for (const cell of cells) { const box = el('div', undefined, 'position ' + cell.state); box.append(el('span', cell.id, 'slot'), el('strong', 'SKU ' + cell.expected_sku), el('div', (cell.state === 'OOS' ? 'Missing' : cell.state === 'OK' ? 'Correct position' : cell.state === 'MISPLACED' ? 'Wrong SKU: ' + cell.observed_sku : 'Unknown product') + (cell.stable ? '' : ' · checking'), 'state')); grid.append(box); }
}
async function refresh() {
  const [status, alerts] = await Promise.all([api('/status'), api('/alerts')]);
  renderMonitor(status.monitor); alertRows = alerts;
  $('nav-alert-count').textContent = alerts.filter(a => !a.resolved && !a.acknowledged && a.source !== 'demo').length;
  $('artifact-path').textContent = status.artifact_path; $('reference-path').textContent = status.reference_path;
  if (currentView === 'models') { const list = $('model-files'); list.replaceChildren(); for (const file of status.files) { const row = el('div', undefined, 'file-row'); row.append(el('span', '◇', 'file-icon')); const info = el('div'); info.append(el('h4', file.name), el('code', file.file)); row.append(info, el('span', file.present ? 'Found' : 'Waiting for export', 'badge ' + (file.present ? 'present' : 'missing'))); list.append(row); } }
  if (currentView === 'products') await renderCatalog();
  if (currentView === 'alerts') renderAlerts();
}
async function poll() { try { await refresh(); } catch (error) { notice('Connection to the local application failed. ' + error.message, true); } finally { setTimeout(poll, 1200); } }
poll();

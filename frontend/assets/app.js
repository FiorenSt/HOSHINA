
const state = {
  classes: [],
  queue: 'uncertain',
  page: 1,
  page_size: 300,
  tile_size: 'xlarge',
  prob_low: 0.3,
  prob_high: 0.8,
  items: [],
  keyMap: {},
  selectedItems: new Set(),
  stats: null,
  loading: false,
  total_items: 0,
  total_pages: 1,
  theme: 'dark',
  search_query: '',
  label_filter: '',
  dragSelection: {
    active: false,
    startX: 0,
    startY: 0,
    currentX: 0,
    currentY: 0,
    overlay: null
  },
  session: {
    startTime: Date.now(),
    labeledCount: 0,
    skippedCount: 0,
    unsureCount: 0,
    lastLabelTime: Date.now(),
    avgTimePerLabel: 0
  },
};

function el(sel){ return document.querySelector(sel); }
function els(sel){ return document.querySelectorAll(sel); }

function applyTileSize(size) {
  const grid = el('#grid');
  if (grid) {
    // Remove all tile size classes
    grid.classList.remove('tile-small', 'tile-medium', 'tile-large', 'tile-xlarge');
    // Add the selected size class
    grid.classList.add(`tile-${size}`);
    console.log(`Applied tile size: ${size}, grid classes:`, grid.className);
    
    // Force a re-layout by triggering a reflow
    grid.style.display = 'none';
    grid.offsetHeight; // Trigger reflow
    grid.style.display = 'grid';
  } else {
    console.log('Grid element not found - will apply on next load');
  }
}

async function api(path, opts={}){
  const r = await fetch('/api' + path, opts);
  if(!r.ok){ throw new Error(`API ${path} failed: ${r.status}`); }
  return await r.json();
}

function status(msg){ 
  el('#status').textContent = msg; 
  if (msg.includes('Loading') || msg.includes('Training')) {
    el('#status').classList.add('loading');
  } else {
    el('#status').classList.remove('loading');
  }
}

async function loadClasses(){
  const data = await api('/classes');
  state.classes = data;
  state.keyMap = {};
  data.forEach((c,i)=>{
    const k = (i+1).toString();
    state.keyMap[k] = c.name;
  });
}

async function loadStats(){
  try {
    const data = await api('/stats');
    state.stats = data;
    updateStatsDisplay();
  } catch (e) {
    console.error('Failed to load stats:', e);
  }
}

function updateStatsDisplay(){
  if (!state.stats) return;
  const stats = state.stats;
  const progressBar = el('#progress-bar');
  if (progressBar) {
    progressBar.style.width = `${stats.progress_percentage}%`;
    progressBar.textContent = `${stats.progress_percentage}%`;
  }
  
  // Update status with more detailed info
  // const statusText = `${stats.labeled_items}/${stats.total_items} labeled (${stats.progress_percentage}%)`;
  // if (!state.loading) {
  //   status(statusText);
  // }
}

async function saveClasses(list){
  await api('/classes', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify(list),
  });
  await loadClasses();
}

function tileTemplate(item){
  const predP = (item && item.pred_label && item.probs && item.probs[item.pred_label] != null) ? item.probs[item.pred_label] : item.max_proba;
  const predBadge = item.pred_label ? `<span class="badge pred">pred: ${item.pred_label}${(predP!=null)?` (${Number(predP).toFixed(2)})`:''}</span>` : '';
  const labelBadge = item.label ? `<span class="badge label">label: ${item.label}</span>` : '';
  const selected = state.selectedItems.has(item.id) ? 'selected' : '';
  const checkbox = `<input type="checkbox" class="tile-checkbox" data-id="${item.id}" ${selected ? 'checked' : ''}>`;
  return `
  <div class="tile ${selected}" data-id="${item.id}">
    ${checkbox}
    <img src="${item.thumb}" loading="lazy" />
    <div class="hoverbar">
      <button class="iconbtn similar" title="Find similar images">Similar</button>
      <button class="iconbtn unsure" title="Mark as unsure">Unsure</button>
      <button class="iconbtn skip" title="Skip this image">Skip</button>
    </div>
    <div class="meta">
      <div>${predBadge} ${labelBadge}</div>
      <div></div>
    </div>
  </div>`;
}

async function loadItems(){
  state.loading = true;
  status('Loading...');
  try {
    const q = state.queue;
    const params = new URLSearchParams({
      queue: q, page: state.page, page_size: state.page_size,
      prob_low: state.prob_low, prob_high: state.prob_high, unlabeled_only: true
    });
    // All groups are ready with on-the-fly generation
    // Stable ordering seed for deterministic 'all' queue order
    if (state.order_seed != null) params.set('seed', String(state.order_seed));
    // Use full mode so predictions (e.g., max_proba) are included in responses
    
    // Add search and filter parameters
    if (state.search_query) {
      params.set('search', state.search_query);
    }
    if (state.label_filter) {
      params.set('label_filter', state.label_filter);
    }
    const data = await api(`/items?${params.toString()}`);
    // Backend paginates by groups and returns all members for selected groups
    state.items = (data.items || []);
    const grid = el('#grid');
    grid.innerHTML = state.items.map(tileTemplate).join('');
    
    // Apply current tile size after grid is populated
    applyTileSize(state.tile_size);
    
    // attach listeners
    grid.querySelectorAll('.tile').forEach(node=>{
      const id = parseInt(node.dataset.id);
      node.querySelector('.similar').addEventListener('click', ()=>showSimilar(id));
      node.querySelector('.unsure').addEventListener('click', ()=>labelItems([id], null, true, false));
      node.querySelector('.skip').addEventListener('click', ()=>labelItems([id], null, false, true));
      
      // Add checkbox listener
      const checkbox = node.querySelector('.tile-checkbox');
      if (checkbox) {
        checkbox.addEventListener('change', (e) => {
          if (e.target.checked) {
            state.selectedItems.add(id);
            node.classList.add('selected');
          } else {
            state.selectedItems.delete(id);
            node.classList.remove('selected');
          }
          updateBatchControls();
        });
      }
      
      // Add image click listener for zoom
      const img = node.querySelector('img');
      if (img) {
        img.addEventListener('click', () => showImageZoom(id));
      }
    });
    
    // Update pagination state
    // Totals are in groups now
    state.total_items = data.total || 0;
    state.total_pages = Math.ceil((state.total_items || 0) / state.page_size);
    state.page = data.page || 1;
    
    updatePagination();
    setupDragSelection();
    await loadStats();
    // status(`📋 ${data.total} items in queue`);
  } catch (e) {
    status(`⚠️ Error loading items: ${e.message}`);
  } finally {
    state.loading = false;
  }
}

function updateSessionStats(count, type = 'labeled') {
  const now = Date.now();
  const timeSinceLastLabel = now - state.session.lastLabelTime;
  
  if (type === 'labeled') {
    state.session.labeledCount += count;
  } else if (type === 'skipped') {
    state.session.skippedCount += count;
  } else if (type === 'unsure') {
    state.session.unsureCount += count;
  }
  
  const totalLabeled = state.session.labeledCount + state.session.skippedCount + state.session.unsureCount;
  if (totalLabeled > 0) {
    const sessionDuration = now - state.session.startTime;
    state.session.avgTimePerLabel = sessionDuration / totalLabeled;
  }
  
  state.session.lastLabelTime = now;
  updateSessionDisplay();
}

function updateSessionDisplay() {
  const sessionTime = Date.now() - state.session.startTime;
  const hours = Math.floor(sessionTime / 3600000);
  const minutes = Math.floor((sessionTime % 3600000) / 60000);
  const seconds = Math.floor((sessionTime % 60000) / 1000);
  
  const timeStr = hours > 0 ? `${hours}h ${minutes}m` : minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
  const totalLabeled = state.session.labeledCount + state.session.skippedCount + state.session.unsureCount;
  const rate = totalLabeled > 0 ? (totalLabeled / (sessionTime / 60000)).toFixed(1) : '0';
  
  // Update status with session info when not loading
  // if (!state.loading && totalLabeled > 0) {
  //   const sessionInfo = `⏱️ ${timeStr} • ${totalLabeled} labeled • ${rate}/min`;
  //   const statusEl = el('#status');
  //   if (statusEl && !statusEl.textContent.includes('Loading') && !statusEl.textContent.includes('Error')) {
  //     statusEl.textContent = sessionInfo;
  //   }
  // }
}

async function labelItems(ids, label=null, unsure=false, skip=false){
  try {
    await api('/label', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({item_ids: ids, label, unsure, skip, user: null})
    });
    
    // Update session stats
    if (skip) {
      updateSessionStats(ids.length, 'skipped');
    } else if (unsure) {
      updateSessionStats(ids.length, 'unsure');
    } else if (label) {
      updateSessionStats(ids.length, 'labeled');
    }
    
    await loadItems();
  } catch (e) {
    status(`Error labeling items: ${e.message}`);
  }
}

async function batchLabelItems(ids, label=null, unsure=false, skip=false){
  if (ids.length === 0) return;
  try {
    const result = await api('/batch-label', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({item_ids: ids, label, unsure, skip, user: null})
    });
    if (result.ok) {
      // Update session stats
      if (skip) {
        updateSessionStats(result.labeled_count, 'skipped');
      } else if (unsure) {
        updateSessionStats(result.labeled_count, 'unsure');
      } else if (label) {
        updateSessionStats(result.labeled_count, 'labeled');
      }
      
      status(`Batch labeled ${result.labeled_count} items`);
      state.selectedItems.clear();
      await loadItems();
    } else {
      status(`Batch labeling failed: ${result.msg}`);
    }
  } catch (e) {
    status(`Error batch labeling: ${e.message}`);
  }
}

// retrain removed: training handled via TensorFlow dashboard

async function showSimilar(id){
  const data = await api(`/similar/${id}`);
  const grid = el('#grid');
  grid.innerHTML = data.neighbors.map(tileTemplate).join('');
  grid.querySelectorAll('.tile').forEach(node=>{
    const id = parseInt(node.dataset.id);
    node.querySelector('.similar').addEventListener('click', ()=>showSimilar(id));
    node.querySelector('.unsure').addEventListener('click', ()=>labelItems([id], null, true, false));
    node.querySelector('.skip').addEventListener('click', ()=>labelItems([id], null, false, true));
  });
  status(`Similar items`);
}

function openModal(id){ el(id).classList.remove('hidden'); }
function closeModal(id){ el(id).classList.add('hidden'); }

async function openClasses(){
  await loadClasses();
  const list = el('#classList');
  list.innerHTML = '';
  const colors = (typeof syncClassColorsWithClasses === 'function') ? syncClassColorsWithClasses() : [];
  state.classes.forEach((c, i)=>{
    const row = document.createElement('div');
    const color = colors[i] || '#38bdf8';
    row.innerHTML = `<input type="text" value="${c.name}" data-idx="${i}"/> <input type="color" class="class-color" value="${color}" data-idx="${i}" style="margin-left:8px; vertical-align: middle;"/> <kbd>${i+1}</kbd> <button class="remove-class-btn" data-idx="${i}" style="margin-left:8px; background: #ef4444; color: white; border: none; border-radius: 3px; padding: 2px 6px; cursor: pointer;" title="Remove class">×</button>`;
    list.appendChild(row);
  });
  openModal('#classesModal');
}

function addClassRow(){
  const list = el('#classList');
  const i = list.children.length;
  const row = document.createElement('div');
  let color = '#38bdf8';
  try {
    const n = (state.classes || []).length + 1;
    if (typeof syncClassColorsWithClasses === 'function'){
      const existing = syncClassColorsWithClasses();
      const palette = (typeof getClassColorByIndex === 'function') ? existing.concat('#10b981') : existing;
      color = existing[i] || ['#38bdf8','#ef4444','#f59e0b','#10b981','#8b5cf6'][i % 5];
    }
  } catch {}
  row.innerHTML = `<input type="text" value="class_${i+1}" data-idx="${i}"/> <input type="color" class="class-color" value="${color}" data-idx="${i}" style="margin-left:8px; vertical-align: middle;"/> <kbd>${i+1}</kbd> <button class="remove-class-btn" data-idx="${i}" style="margin-left:8px; background: #ef4444; color: white; border: none; border-radius: 3px; padding: 2px 6px; cursor: pointer;" title="Remove class">×</button>`;
  list.appendChild(row);
}

function removeClassRow(idx){
  const list = el('#classList');
  const rows = list.children;
  if (idx >= 0 && idx < rows.length) {
    rows[idx].remove();
    // Update data-idx attributes for remaining rows
    for (let i = idx; i < rows.length; i++) {
      const row = rows[i];
      const inputs = row.querySelectorAll('input, button, kbd');
      inputs.forEach(elem => {
        if (elem.hasAttribute('data-idx')) {
          elem.setAttribute('data-idx', i);
        }
      });
      // Update keyboard shortcut display
      const kbd = row.querySelector('kbd');
      if (kbd) kbd.textContent = i + 1;
    }
  }
}

async function saveClassesFromModal(){
  const nameInputs = [...el('#classList').querySelectorAll('input[type="text"]')];
  const colorInputs = [...el('#classList').querySelectorAll('input.class-color')];
  const classes = nameInputs.map((inp, i)=>({name: inp.value, key: (i+1).toString(), order: i}));
  await saveClasses(classes);
  // Persist colors alongside (frontend only)
  try {
    const colors = classes.map((_, i)=> colorInputs[i] ? colorInputs[i].value : '#38bdf8');
    if (typeof saveClassColors === 'function') saveClassColors(colors);
  } catch {}
  closeModal('#classesModal');
  await loadItems();
}

function drawMap(points){
  const canvas = el('#mapCanvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width, canvas.height);
  const xs = points.map(p=>p.x), ys = points.map(p=>p.y);
  const minx = Math.min(...xs), maxx = Math.max(...xs);
  const miny = Math.min(...ys), maxy = Math.max(...ys);
  const pad = 20;
  function sx(x){ return pad + (x - minx) / (maxx - minx + 1e-6) * (canvas.width - pad*2); }
  function sy(y){ return pad + (y - miny) / (maxy - miny + 1e-6) * (canvas.height - pad*2); }
  ctx.globalAlpha = 0.9;
  for(const p of points){
    ctx.beginPath();
    ctx.arc(sx(p.x), sy(p.y), 3, 0, Math.PI*2);
    ctx.fillStyle = p.label ? '#f59e0b' : '#38bdf8';
    ctx.fill();
  }
}

async function openMap(){
  openModal('#mapModal');
  status('Computing UMAP visualization...');
  try {
    const data = await api('/map');
    if (data.points && data.points.length > 0) {
      drawMap(data.points);
      status(`UMAP visualization with ${data.points.length} points`);
    } else {
      status(data.msg || 'No data available for visualization');
      // Clear the canvas when no real data is available
      const canvas = el('#mapCanvas');
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  } catch (e) {
    status(`Error loading map: ${e.message}`);
  }
}

async function openStats(){
  openModal('#statsModal');
  await refreshStatsDisplay();
}

async function openSuggestions(){
  openModal('#suggestionsModal');
  await loadSuggestions();
}

async function openBulkOps(){
  openModal('#bulkOpsModal');
}

function showImportStatus(message, type = 'loading') {
  const statusEl = el('#import-status');
  if (statusEl) {
    statusEl.textContent = message;
    statusEl.className = `import-status ${type}`;
    statusEl.style.display = 'block';
  }
}

async function exportData(type) {
  try {
    let url, filename;
    switch(type) {
      case 'labels':
        url = '/api/export-labels';
        filename = 'labels.csv';
        break;
      case 'full':
        url = '/api/export';
        filename = 'dataset_export.zip';
        break;
      case 'model':
        url = '/api/export-model';
        filename = 'model_artifacts.zip';
        break;
    }
    
    status(`📤 Exporting ${type}...`);
    
    // Create download link
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    status(`✅ Export started - check your downloads`);
  } catch (e) {
    status(`⚠️ Export failed: ${e.message}`);
  }
}

async function importLabels(file) {
  showImportStatus('📥 Uploading and processing file...', 'loading');
  
  try {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('/api/import-labels', {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    
    if (result.ok) {
      showImportStatus(`✅ Successfully imported ${result.imported_count} labels`, 'success');
      await loadItems(); // Refresh the view
    } else {
      showImportStatus(`❌ Import failed: ${result.msg}`, 'error');
    }
  } catch (e) {
    showImportStatus(`❌ Import error: ${e.message}`, 'error');
  }
}

async function performBulkOperation(operation) {
  const confirmations = {
    'reset-all-labels': 'Are you sure you want to reset ALL labels? This cannot be undone.',
    'auto-label-confident': 'Auto-label items with >90% confidence? This will apply predicted labels.',
    'backup-dataset': 'Create a backup of the current dataset?'
  };
  
  if (confirmations[operation] && !confirm(confirmations[operation])) {
    return;
  }
  
  try {
    status(`⚡ Performing ${operation.replace('-', ' ')}...`);
    const result = await api(`/bulk-${operation}`, { method: 'POST' });
    
    if (result.ok) {
      status(`✅ ${operation.replace('-', ' ')} completed: ${result.msg}`);
      try { alert(`✅ ${result.msg || 'Operation completed'}`); } catch(_) {}
      // Close modal if open
      const modal = document.querySelector('#bulkOpsModal');
      if (modal && !modal.classList.contains('hidden')) {
        closeModal('#bulkOpsModal');
      }
      await loadItems(); // Refresh the view
    } else {
      status(`⚠️ Operation failed: ${result.msg}`);
      try { alert(`❌ Operation failed: ${result.msg || 'Unknown error'}`); } catch(_) {}
    }
  } catch (e) {
    status(`⚠️ Operation error: ${e.message}`);
    try { alert(`❌ Operation error: ${e.message}`); } catch(_) {}
  }
}

async function loadSuggestions(type = 'uncertain', count = 24){
  status('🎯 Loading smart suggestions...');
  try {
    const data = await api(`/suggestions?type=${type}&count=${count}`);
    renderSuggestions(data.suggestions || []);
    status('🎯 Smart suggestions loaded');
  } catch (e) {
    status(`⚠️ Error loading suggestions: ${e.message}`);
  }
}

function renderSuggestions(suggestions){
  const container = el('#suggestions-content');
  if (!container) return;
  
  if (suggestions.length === 0) {
    container.innerHTML = '<div class="no-suggestions">No suggestions available. Train the model to generate predictions.</div>';
    return;
  }
  
  container.innerHTML = suggestions.map(item => `
    <div class="suggestion-item" data-id="${item.id}">
      <img src="${item.thumb}" loading="lazy" />
      <div class="suggestion-meta">
        <div class="suggestion-score">${item.score ? item.score.toFixed(3) : 'N/A'}</div>
        <div class="suggestion-reason">${item.reason || 'AI recommendation'}</div>
      </div>
    </div>
  `).join('');
  
  // Add click handlers
  container.querySelectorAll('.suggestion-item').forEach(item => {
    const id = parseInt(item.dataset.id);
    item.addEventListener('click', () => showImageZoom(id));
  });
}

async function refreshStatsDisplay(){
  status('📊 Loading statistics...');
  try {
    const stats = await api('/stats');
    renderStatsDisplay(stats);
    status('📊 Statistics updated');
  } catch (e) {
    status(`⚠️ Error loading stats: ${e.message}`);
  }
}

function renderStatsDisplay(stats){
  const container = el('#stats-content');
  if (!container) return;
  
  const progressPercent = stats.progress_percentage || 0;
  const totalItems = stats.total_items || 0;
  const labeledItems = stats.labeled_items || 0;
  const unlabeledItems = stats.unlabeled_items || 0;
  const skippedItems = stats.skipped_items || 0;
  const unsureItems = stats.unsure_items || 0;
  
  // Session stats
  const sessionTime = Date.now() - state.session.startTime;
  const sessionHours = Math.floor(sessionTime / 3600000);
  const sessionMinutes = Math.floor((sessionTime % 3600000) / 60000);
  const sessionTimeStr = sessionHours > 0 ? `${sessionHours}h ${sessionMinutes}m` : `${sessionMinutes}m`;
  const totalSessionLabeled = state.session.labeledCount + state.session.skippedCount + state.session.unsureCount;
  const sessionRate = totalSessionLabeled > 0 ? (totalSessionLabeled / (sessionTime / 60000)).toFixed(1) : '0';

  container.innerHTML = `
    <div class="stats-card">
      <h3>📈 Progress Overview</h3>
      <div class="stats-value">${progressPercent}%</div>
      <div class="stats-label">Dataset Completion</div>
      <div class="stats-progress">
        <div class="stats-progress-fill" style="width: ${progressPercent}%"></div>
      </div>
      <div class="stats-breakdown">
        <div class="stats-item">
          <span class="stats-item-label">Total Items</span>
          <span class="stats-item-value">${totalItems.toLocaleString()}</span>
        </div>
        <div class="stats-item">
          <span class="stats-item-label">✅ Labeled</span>
          <span class="stats-item-value">${labeledItems.toLocaleString()}</span>
        </div>
        <div class="stats-item">
          <span class="stats-item-label">⏳ Remaining</span>
          <span class="stats-item-value">${unlabeledItems.toLocaleString()}</span>
        </div>
      </div>
    </div>
    
    <div class="stats-card">
      <h3>⏱️ Session Statistics</h3>
      <div class="stats-value">${sessionRate}</div>
      <div class="stats-label">Items per minute</div>
      <div class="stats-breakdown">
        <div class="stats-item">
          <span class="stats-item-label">Session Time</span>
          <span class="stats-item-value">${sessionTimeStr}</span>
        </div>
        <div class="stats-item">
          <span class="stats-item-label">📝 Labeled</span>
          <span class="stats-item-value">${state.session.labeledCount}</span>
        </div>
        <div class="stats-item">
          <span class="stats-item-label">❓ Unsure</span>
          <span class="stats-item-value">${state.session.unsureCount}</span>
        </div>
        <div class="stats-item">
          <span class="stats-item-label">⏭️ Skipped</span>
          <span class="stats-item-value">${state.session.skippedCount}</span>
        </div>
      </div>
    </div>
    
    <div class="stats-card">
      <h3>🏷️ Label Distribution</h3>
      <div class="stats-breakdown">
        ${Object.entries(stats.class_distribution || {}).map(([className, count]) => `
          <div class="stats-item">
            <span class="stats-item-label">${className}</span>
            <span class="stats-item-value">${count.toLocaleString()}</span>
          </div>
        `).join('')}
        ${Object.keys(stats.class_distribution || {}).length === 0 ? 
          '<div class="stats-item"><span class="stats-item-label">No classes labeled yet</span><span class="stats-item-value">-</span></div>' : ''}
      </div>
    </div>
    
    <div class="stats-card">
      <h3>⚠️ Special Categories</h3>
      <div class="stats-breakdown">
        <div class="stats-item">
          <span class="stats-item-label">❓ Unsure</span>
          <span class="stats-item-value">${unsureItems.toLocaleString()}</span>
        </div>
        <div class="stats-item">
          <span class="stats-item-label">⏭️ Skipped</span>
          <span class="stats-item-value">${skippedItems.toLocaleString()}</span>
        </div>
      </div>
    </div>
    
    <div class="stats-card">
      <h3>🤖 AI Processing</h3>
      <div class="stats-breakdown">
        <div class="stats-item">
          <span class="stats-item-label">🧠 Embeddings</span>
          <span class="stats-item-value">${(stats.embedding_count || 0).toLocaleString()}</span>
        </div>
        <div class="stats-item">
          <span class="stats-item-label">🎯 Predictions</span>
          <span class="stats-item-value">${(stats.prediction_count || 0).toLocaleString()}</span>
        </div>
        <div class="stats-item">
          <span class="stats-item-label">Coverage</span>
          <span class="stats-item-value">${Math.round(((stats.embedding_count || 0) / totalItems * 100) || 0)}%</span>
        </div>
      </div>
    </div>
    
    <div class="stats-card">
      <h3>📊 Confidence Distribution</h3>
      <div class="confidence-chart" id="confidence-chart">
        <div class="confidence-bar">
          <div class="confidence-label">High (>0.8)</div>
          <div class="confidence-bar-fill confidence-high" style="width: ${(stats.confidence_high || 0)}%"></div>
          <div class="confidence-value">${stats.confidence_high || 0}%</div>
        </div>
        <div class="confidence-bar">
          <div class="confidence-label">Medium (0.5-0.8)</div>
          <div class="confidence-bar-fill confidence-medium" style="width: ${(stats.confidence_medium || 0)}%"></div>
          <div class="confidence-value">${stats.confidence_medium || 0}%</div>
        </div>
        <div class="confidence-bar">
          <div class="confidence-label">Low (<0.5)</div>
          <div class="confidence-bar-fill confidence-low" style="width: ${(stats.confidence_low || 0)}%"></div>
          <div class="confidence-value">${stats.confidence_low || 0}%</div>
        </div>
      </div>
    </div>
  `;
}

function updateBatchControls(){
  const selectedCount = state.selectedItems.size;
  const batchControls = el('#batch-controls');
  if (batchControls) {
    batchControls.style.display = selectedCount > 0 ? 'flex' : 'none';
    el('#selected-count').textContent = selectedCount;
    
    // Update batch class buttons
    updateBatchClassButtons();
  }
}

function updateBatchClassButtons(){
  const container = el('#batch-class-buttons');
  if (!container) return;
  
  container.innerHTML = '';
  
  if (state.classes.length === 0) return;
  
  // Add class buttons
  state.classes.forEach((cls, i) => {
    const btn = document.createElement('button');
    btn.className = 'batch-class-btn';
    btn.textContent = `${cls.key}: ${cls.name}`;
    btn.title = `Batch label as ${cls.name}`;
    btn.addEventListener('click', () => {
      batchLabelItems(Array.from(state.selectedItems), cls.name, false, false);
    });
    container.appendChild(btn);
  });
}

function updatePagination(){
  const showPagination = state.total_pages > 1;
  
  // Show/hide pagination controls
  const topPagination = el('#pagination-controls');
  const bottomPagination = el('#bottom-pagination');
  const pageIndicator = el('#page-indicator');
  
  if (topPagination) topPagination.style.display = showPagination ? 'flex' : 'none';
  if (bottomPagination) bottomPagination.style.display = showPagination ? 'flex' : 'none';
  if (pageIndicator) pageIndicator.style.display = showPagination ? 'block' : 'none';
  
  // Update header page indicator
  if (pageIndicator && showPagination) {
    const currentPageEl = el('#current-page');
    const totalPagesEl = el('#total-pages');
    if (currentPageEl) currentPageEl.textContent = state.page;
    if (totalPagesEl) totalPagesEl.textContent = state.total_pages;
  }
  
  if (!showPagination) return;
  
  // Update pagination info
  const startItem = (state.page - 1) * state.page_size + 1;
  const endItem = Math.min(state.page * state.page_size, state.total_items);
  
  const topInfo = el('#pagination-info-text');
  const bottomInfo = el('#bottom-pagination-info');
  if (topInfo) topInfo.textContent = `Page ${state.page} of ${state.total_pages}`;
  if (bottomInfo) bottomInfo.textContent = `Showing ${startItem}-${endItem} of ${state.total_items} groups`;
  
  // Update button states
  const isFirstPage = state.page <= 1;
  const isLastPage = state.page >= state.total_pages;
  
  // Top buttons
  const firstBtn = el('#first-page');
  const prevBtn = el('#prev-page');
  const nextBtn = el('#next-page');
  const lastBtn = el('#last-page');
  
  if (firstBtn) firstBtn.disabled = isFirstPage;
  if (prevBtn) prevBtn.disabled = isFirstPage;
  if (nextBtn) nextBtn.disabled = isLastPage;
  if (lastBtn) lastBtn.disabled = isLastPage;
  
  // Bottom buttons
  const bottomFirstBtn = el('#bottom-first-page');
  const bottomPrevBtn = el('#bottom-prev-page');
  const bottomNextBtn = el('#bottom-next-page');
  const bottomLastBtn = el('#bottom-last-page');
  
  if (bottomFirstBtn) bottomFirstBtn.disabled = isFirstPage;
  if (bottomPrevBtn) bottomPrevBtn.disabled = isFirstPage;
  if (bottomNextBtn) bottomNextBtn.disabled = isLastPage;
  if (bottomLastBtn) bottomLastBtn.disabled = isLastPage;
}

async function goToPage(page) {
  if (page < 1 || page > state.total_pages || page === state.page) return;
  state.page = page;
  state.selectedItems.clear();
  updateBatchControls();
  await loadItems();
}

function toggleTheme() {
  const newTheme = state.theme === 'dark' ? 'light' : 'dark';
  state.theme = newTheme;
  
  // Update document theme
  if (newTheme === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
  } else {
    document.documentElement.removeAttribute('data-theme');
  }
  
  // Update button icon
  const themeBtn = el('#theme-toggle');
  if (themeBtn) {
    themeBtn.textContent = newTheme === 'dark' ? '🌙' : '☀️';
    themeBtn.title = `Switch to ${newTheme === 'dark' ? 'light' : 'dark'} mode`;
  }
  
  // Save preference
  localStorage.setItem('al_theme', newTheme);
}

function initTheme() {
  const savedTheme = localStorage.getItem('al_theme') || 'dark';
  state.theme = savedTheme;
  
  if (savedTheme === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
  }
  
  const themeBtn = el('#theme-toggle');
  if (themeBtn) {
    themeBtn.textContent = savedTheme === 'dark' ? '🌙' : '☀️';
    themeBtn.title = `Switch to ${savedTheme === 'dark' ? 'light' : 'dark'} mode`;
  }
}

function createDragOverlay() {
  if (!state.dragSelection.overlay) {
    const overlay = document.createElement('div');
    overlay.className = 'drag-selection-overlay';
    document.body.appendChild(overlay);
    state.dragSelection.overlay = overlay;
  }
  return state.dragSelection.overlay;
}

function updateDragOverlay() {
  const overlay = state.dragSelection.overlay;
  if (!overlay) return;
  
  const startX = Math.min(state.dragSelection.startX, state.dragSelection.currentX);
  const startY = Math.min(state.dragSelection.startY, state.dragSelection.currentY);
  const width = Math.abs(state.dragSelection.currentX - state.dragSelection.startX);
  const height = Math.abs(state.dragSelection.currentY - state.dragSelection.startY);
  
  overlay.style.left = `${startX}px`;
  overlay.style.top = `${startY}px`;
  overlay.style.width = `${width}px`;
  overlay.style.height = `${height}px`;
  overlay.style.display = 'block';
}

function getItemsInSelection() {
  const startX = Math.min(state.dragSelection.startX, state.dragSelection.currentX);
  const startY = Math.min(state.dragSelection.startY, state.dragSelection.currentY);
  const endX = Math.max(state.dragSelection.startX, state.dragSelection.currentX);
  const endY = Math.max(state.dragSelection.startY, state.dragSelection.currentY);
  
  const selectedIds = new Set();
  const tiles = els('.tile');
  
  tiles.forEach(tile => {
    const rect = tile.getBoundingClientRect();
    const tileX = rect.left + window.scrollX;
    const tileY = rect.top + window.scrollY;
    const tileEndX = tileX + rect.width;
    const tileEndY = tileY + rect.height;
    
    // Check if tile intersects with selection rectangle
    if (tileX < endX && tileEndX > startX && tileY < endY && tileEndY > startY) {
      const id = parseInt(tile.dataset.id);
      if (id) selectedIds.add(id);
    }
  });
  
  return selectedIds;
}

function setupDragSelection() {
  const grid = el('#grid');
  if (!grid) return;
  
  grid.addEventListener('mousedown', (e) => {
    // Only start drag selection on empty areas (not on tiles or buttons)
    if (e.target.closest('.tile') || e.target.closest('button') || e.target.closest('input')) return;
    
    state.dragSelection.active = true;
    state.dragSelection.startX = e.pageX;
    state.dragSelection.startY = e.pageY;
    state.dragSelection.currentX = e.pageX;
    state.dragSelection.currentY = e.pageY;
    
    createDragOverlay();
    updateDragOverlay();
    
    e.preventDefault();
  });
  
  document.addEventListener('mousemove', (e) => {
    if (!state.dragSelection.active) return;
    
    state.dragSelection.currentX = e.pageX;
    state.dragSelection.currentY = e.pageY;
    updateDragOverlay();
    
    // Update selected items
    const selectedIds = getItemsInSelection();
    state.selectedItems = selectedIds;
    
    // Update UI
    els('.tile').forEach(tile => {
      const id = parseInt(tile.dataset.id);
      const checkbox = tile.querySelector('.tile-checkbox');
      if (selectedIds.has(id)) {
        tile.classList.add('selected');
        if (checkbox) checkbox.checked = true;
      } else {
        tile.classList.remove('selected');
        if (checkbox) checkbox.checked = false;
      }
    });
    
    updateBatchControls();
  });
  
  document.addEventListener('mouseup', () => {
    if (!state.dragSelection.active) return;
    
    state.dragSelection.active = false;
    if (state.dragSelection.overlay) {
      state.dragSelection.overlay.style.display = 'none';
    }
  });
}

function setupUI(){
  // Theme toggle
  el('#theme-toggle').addEventListener('click', toggleTheme);
  
  
  // Header bulk ops button
  const bulkHeaderBtn = el('#bulk-ops-header');
  if (bulkHeaderBtn) {
    bulkHeaderBtn.addEventListener('click', openBulkOps);
  }

  // Tab UI removed: all sections are visible at once
  
  el('#queue').addEventListener('change', async (e)=>{
    state.queue = e.target.value; state.page = 1; await loadItems();
  });
  el('#pull').addEventListener('click', async ()=>{
    state.prob_low = parseFloat(el('#low').value);
    state.prob_high = parseFloat(el('#high').value);
    await loadItems();
  });
  // retrain button removed
  el('#export').addEventListener('click', ()=>window.location.href='/api/export');
  el('#classes').addEventListener('click', openClasses);
  el('#map').addEventListener('click', openMap);
  el('#stats').addEventListener('click', openStats);
  el('#suggestions').addEventListener('click', openSuggestions);
  el('#bulk-ops').addEventListener('click', openBulkOps);
  el('#help').addEventListener('click', showHelp);
  // Remove legacy background thumbnail build controls
  // Any UI elements for start/cancel build are now no-ops
  
  // Modal event handlers - use event delegation since elements are in hidden modals
  document.addEventListener('click', (e) => {
    const is = (sel) => e.target.closest(sel);
    // Classes modal
    if (is('#addClass')) {
      addClassRow();
    } else if (is('#saveClasses')) {
      saveClassesFromModal();
    } else if (is('#closeClasses')) {
      closeModal('#classesModal');
    } else if (is('.remove-class-btn')) {
      const idx = parseInt(e.target.getAttribute('data-idx'));
      removeClassRow(idx);
    }
    // Map modal
    else if (is('#closeMap')) {
      closeModal('#mapModal');
    }
    // Stats modal
    else if (is('#closeStats')) {
      closeModal('#statsModal');
    } else if (is('#refreshStats')) {
      refreshStatsDisplay();
    }
    // Suggestions modal
    else if (is('#closeSuggestions')) {
      closeModal('#suggestionsModal');
    } else if (is('#refresh-suggestions')) {
      const activeBtn = el('.suggestion-type-btn.active');
      const type = activeBtn ? activeBtn.id.replace('suggest-', '') : 'uncertain';
      const count = parseInt(el('#suggestion-count').value) || 24;
      loadSuggestions(type, count);
    }
    // Image modal
    else if (is('#closeImage')) {
      closeModal('#imageModal');
    }
    // Bulk operations modal
    else if (is('#closeBulkOps')) {
      closeModal('#bulkOpsModal');
    } else if (is('#export-labels')) {
      exportData('labels');
    } else if (is('#export-full')) {
      exportData('full');
    } else if (is('#export-model')) {
      exportData('model');
    } else if (is('#import-labels') || is('#import-config')) {
      const importFile = el('#import-file');
      if (importFile) importFile.click();
    } else if (is('#reset-all-labels')) {
      performBulkOperation('reset-all-labels');
    } else if (is('#auto-label-confident')) {
      performBulkOperation('auto-label-confident');
    } else if (is('#backup-dataset')) {
      performBulkOperation('backup-dataset');
    }
  });

  // Suggestions controls - use event delegation for buttons in hidden modal
  document.addEventListener('click', (e) => {
    if (e.target.classList.contains('suggestion-type-btn')) {
      // Update active state
      els('.suggestion-type-btn').forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      
      // Get type from button id
      const type = e.target.id.replace('suggest-', '');
      const count = parseInt(el('#suggestion-count').value) || 24;
      loadSuggestions(type, count);
    }
  });
  
  // Import file change handler
  document.addEventListener('change', (e) => {
    if (e.target.id === 'import-file') {
      const file = e.target.files[0];
      if (file) {
        if (file.name.endsWith('.csv')) {
          importLabels(file);
        } else {
          showImportStatus('❌ Please select a CSV file for label import', 'error');
        }
      }
    } else if (e.target.id === 'suggestion-count') {
      const activeBtn = el('.suggestion-type-btn.active');
      const type = activeBtn ? activeBtn.id.replace('suggest-', '') : 'uncertain';
      const count = parseInt(e.target.value) || 24;
      loadSuggestions(type, count);
    }
  });

  // Page size selector
  const sizeSel = el('#pageSize');
  if (sizeSel) {
    // Force fixed page size of 300 and disable control
    state.page_size = 300;
    sizeSel.value = '300';
    sizeSel.disabled = true;
  }

  // Tile size selector
  const tileSizeSel = el('#tileSize');
  if (tileSizeSel) {
    // Restore previously chosen tile size if available
    const saved = localStorage.getItem('al_tile_size') || 'xlarge';
    state.tile_size = saved;
    tileSizeSel.value = saved;
    
    tileSizeSel.addEventListener('change', (e)=>{
      const v = e.target.value || 'medium';
      console.log('Tile size changed to:', v);
      state.tile_size = v;
      localStorage.setItem('al_tile_size', v);
      // Apply immediately if grid exists, otherwise it will be applied on next load
      applyTileSize(v);
    });
  }
  
  // Batch controls
  el('#select-all').addEventListener('click', ()=>{
    state.items.forEach(item => state.selectedItems.add(item.id));
    updateBatchControls();
    loadItems(); // Refresh to show selections
  });
  el('#select-none').addEventListener('click', ()=>{
    state.selectedItems.clear();
    updateBatchControls();
    loadItems(); // Refresh to hide selections
  });
  el('#batch-unsure').addEventListener('click', ()=>{
    batchLabelItems(Array.from(state.selectedItems), null, true, false);
  });
  el('#batch-skip').addEventListener('click', ()=>{
    batchLabelItems(Array.from(state.selectedItems), null, false, true);
  });

  // Pagination controls - Top
  el('#first-page').addEventListener('click', () => goToPage(1));
  el('#prev-page').addEventListener('click', () => goToPage(state.page - 1));
  el('#next-page').addEventListener('click', () => goToPage(state.page + 1));
  el('#last-page').addEventListener('click', () => goToPage(state.total_pages));
  
  // Pagination controls - Bottom
  el('#bottom-first-page').addEventListener('click', () => goToPage(1));
  el('#bottom-prev-page').addEventListener('click', () => goToPage(state.page - 1));
  el('#bottom-next-page').addEventListener('click', () => goToPage(state.page + 1));
  el('#bottom-last-page').addEventListener('click', () => goToPage(state.total_pages));

  // Search functionality
  const searchInput = el('#search-input');
  const searchClear = el('#search-clear');
  const labelFilter = el('#label-filter');
  
  if (searchInput) {
    let searchTimeout;
    searchInput.addEventListener('input', (e) => {
      const query = e.target.value.trim();
      state.search_query = query;
      
      // Show/hide clear button
      searchClear.style.display = query ? 'block' : 'none';
      
      // Debounced search
      clearTimeout(searchTimeout);
      searchTimeout = setTimeout(async () => {
        state.page = 1;
        await loadItems();
      }, 300);
    });
  }
  
  if (searchClear) {
    searchClear.addEventListener('click', async () => {
      searchInput.value = '';
      state.search_query = '';
      searchClear.style.display = 'none';
      state.page = 1;
      await loadItems();
    });
  }
  
  if (labelFilter) {
    labelFilter.addEventListener('change', async (e) => {
      state.label_filter = e.target.value;
      state.page = 1;
      await loadItems();
    });
  }

  document.addEventListener('keydown', async (e)=>{
    const k = e.key;
    const first = state.items[0];
    
    // Don't interfere with typing in inputs
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    
    if(!first) return;
    
    // Class labeling (1-9)
    if(state.keyMap[k]){
      if (state.selectedItems.size > 0) {
        await batchLabelItems(Array.from(state.selectedItems), state.keyMap[k], false, false);
      } else {
        // Legacy bundle lacks triplet grouping; fall back to assigning all visible tiles if possible
        try {
          const tiles = Array.from(document.querySelectorAll('#grid .tile'));
          const allIds = tiles.map(n => parseInt(n.dataset.id)).filter(Boolean);
          if (Array.isArray(allIds) && allIds.length > 1){
            const ok = confirm(`Assign ALL ${allIds.length} items on this page to ${state.keyMap[k]}?`);
            if (!ok) return;
            await batchLabelItems(allIds, state.keyMap[k], false, false);
          } else {
            await labelItems([first.id], state.keyMap[k], false, false);
          }
        } catch(_){
          await labelItems([first.id], state.keyMap[k], false, false);
        }
      }
    }
    // Unsure (0)
    else if(k==='0'){
      if (state.selectedItems.size > 0) {
        await batchLabelItems(Array.from(state.selectedItems), null, true, false);
      } else {
        await labelItems([first.id], null, true, false);
      }
    }
    // Skip (X)
    else if(k==='x' || k==='X'){
      if (state.selectedItems.size > 0) {
        await batchLabelItems(Array.from(state.selectedItems), null, false, true);
      } else {
        await labelItems([first.id], null, false, true);
      }
    }
    // Retrain (R) removed
    // Select all (Ctrl+A)
    else if(k==='a' && e.ctrlKey){
      e.preventDefault();
      state.items.forEach(item => state.selectedItems.add(item.id));
      updateBatchControls();
      loadItems();
    }
    // Clear selection (Escape)
    else if(k==='Escape'){
      state.selectedItems.clear();
      updateBatchControls();
      loadItems();
    }
    // Help (H or ?)
    else if(k==='h' || k==='?'){
      showHelp();
    }
    // Navigation shortcuts
    else if(k==='ArrowLeft' && state.total_pages > 1){
      e.preventDefault();
      await goToPage(state.page - 1);
    }
    else if(k==='ArrowRight' && state.total_pages > 1){
      e.preventDefault();
      await goToPage(state.page + 1);
    }
    // Quick stats (S)
    else if(k==='s' || k==='S'){
      await openStats();
    }
    // Quick map (V for visualize)
    else if(k==='v' || k==='V'){
      await openMap();
    }
  });
}

function showImageZoom(itemId){
  const item = state.items.find(i => i.id === itemId);
  if (!item) return;

  el('#imageModalTitle').textContent = `Image: ${item.path.split('/').pop()}`;

  // Reset UI
  const three = el('#tripletThreeUp');
  const single = el('#imageModalImg');
  const modeSel = el('#tripletMode');
  const toggleSel = el('#tripletToggle');
  const blinkRange = el('#blinkSpeed');
  let blinkTimer = null;

  function applyMode(mode, trip){
    // Cleanup previous blink
    if (blinkTimer) { clearInterval(blinkTimer); blinkTimer = null; }

    if (mode === 'three'){
      three.style.display = 'flex';
      single.style.display = 'none';
      toggleSel.disabled = true;
      blinkRange.disabled = true;
      el('#imgTarget').src = trip.items.target?.file || '';
      el('#imgRef').src = trip.items.ref?.file || '';
      el('#imgDiff').src = trip.items.diff?.file || '';
    } else if (mode === 'toggle'){
      three.style.display = 'none';
      single.style.display = 'block';
      toggleSel.disabled = false;
      blinkRange.disabled = true;
      const sel = toggleSel.value || 'target';
      single.src = trip.items[sel]?.file || '';
    } else if (mode === 'blink'){
      three.style.display = 'none';
      single.style.display = 'block';
      toggleSel.disabled = true;
      blinkRange.disabled = false;
      const order = ['target','ref','diff'].filter(k=>trip.items[k]);
      let idx = 0;
      single.src = trip.items[order[idx]]?.file || '';
      const startBlink = ()=>{
        if (blinkTimer) clearInterval(blinkTimer);
        blinkTimer = setInterval(()=>{
          idx = (idx + 1) % order.length;
          single.src = trip.items[order[idx]]?.file || '';
        }, parseInt(blinkRange.value || '500'));
      };
      startBlink();
      blinkRange.oninput = startBlink;
    }
  }

  // Load triplet
  api(`/triplet/${itemId}`).then(trip=>{
    // Fallback if not a recognized triplet: show single
    const hasAny = trip.items && (trip.items.target || trip.items.ref || trip.items.diff);
    if (!hasAny){
      three.style.display = 'none';
      single.style.display = 'block';
      single.src = `/api/file/${itemId}`;
      openModal('#imageModal');
      return;
    }
    // Initialize controls
    modeSel.value = 'three';
    toggleSel.value = 'target';
    applyMode('three', trip);

    // Wire events
    modeSel.onchange = ()=>applyMode(modeSel.value, trip);
    toggleSel.onchange = ()=>applyMode('toggle', trip);

    openModal('#imageModal');
  }).catch(()=>{
    // fallback single image
    three.style.display = 'none';
    single.style.display = 'block';
    single.src = `/api/file/${itemId}`;
    openModal('#imageModal');
  });
}

function showHelp(){
  const helpText = `
🏷️ HOSHINA - Keyboard Shortcuts:

📝 Labeling:
• 1-9: Label with class (or batch label if items selected)
• 0: Mark as unsure 🤔
• X: Skip item ⏭️

🚀 Actions:
• T: Open TF Train
• S: View statistics 📊
• V: View map/visualization 🗺️
• H or ?: Show this help

📄 Navigation:
• ← →: Previous/Next page
• Ctrl+A: Select all visible items
• Escape: Clear selection

✅ Selection & Interaction:
• Click checkboxes to select items for batch operations
• Click images to zoom 🔍
• Use quick controls in header for page size/tile size
• Drag to select multiple items (coming soon!)

💡 Pro Tips:
• Images are now extra large by default for better visibility
• Page size and tile size controls are always visible in header
• Batch operations work with keyboard shortcuts too
  `;
  alert(helpText);
}

// Moved to modular scripts under /assets/js. This file is kept for compatibility.

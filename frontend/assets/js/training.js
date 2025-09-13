// Training Dashboard JavaScript
// Handles the full-page TensorFlow training interface

let trainingState = {
  isTraining: false,
  currentEpoch: 0,
  totalEpochs: 0,
  currentStep: 0,
  totalSteps: 0,
  startTime: null,
  statusTimer: null,
  charts: {},
  metrics: {
    loss: [],
    accuracy: [],
    valLoss: [],
    valAccuracy: []
  },
  classMap: {}
};

// Determine the positive class name from training target mapping
function getPositiveClassName(){
  try{
    const entries = Object.entries(trainingState.classMap || {}).filter(([n,v])=> Number.isFinite(v));
    if (!entries.length) return '';
    const targets = Array.from(new Set(entries.map(([,v])=> Number(v))));
    if (targets.length === 2 && targets.includes(0) && targets.includes(1)){
      const pos = entries.find(([,v])=> Number(v) === 1);
      return pos ? String(pos[0]) : '';
    }
    // Fallback: choose label with highest numeric target
    entries.sort((a,b)=> Number(a[1]) - Number(b[1]));
    return String(entries[entries.length - 1][0]);
  } catch(_){ return ''; }
}

// Navigation functions
function showTrainingDashboard() {
  const mainContent = document.getElementById('main-content');
  const trainingDashboard = document.getElementById('training-dashboard');
  const header = document.querySelector('header');
  
  if (mainContent) mainContent.style.display = 'none';
  if (header) header.style.display = 'none';
  if (trainingDashboard) {
    trainingDashboard.style.display = 'block';
    trainingDashboard.classList.remove('hidden');
    initializeTrainingDashboard();
  }
}

function hideTrainingDashboard() {
  const mainContent = document.getElementById('main-content');
  const trainingDashboard = document.getElementById('training-dashboard');
  const header = document.querySelector('header');
  
  if (mainContent) mainContent.style.display = 'block';
  if (header) header.style.display = 'block';
  if (trainingDashboard) {
    trainingDashboard.classList.add('hidden');
    trainingDashboard.style.display = 'none';
  }
  
  // Stop status polling
  stopTrainingStatusPolling();
}

// Initialize the training dashboard
async function initializeTrainingDashboard() {
  try {
    // Load training options
    await loadTrainingOptions();
    
    // Load dataset summary
    await loadDatasetSummary();
    
    // Setup event listeners
    setupTrainingEventListeners();
    
    // Setup slider interactions
    setupSliderInteractions();
    
    // Initialize charts
    initializeCharts();
    // Initialize prediction panel
    await initPredictionPanel();
    
    // Start status polling
    startTrainingStatusPolling();
    
    // Add initial log entry
    addLogEntry('info', 'Training dashboard initialized successfully');
    
  } catch (error) {
    console.error('Failed to initialize training dashboard:', error);
    addLogEntry('error', `Failed to initialize dashboard: ${error.message}`);
  }
}

// Load training options from backend
async function loadTrainingOptions() {
  try {
    const response = await fetch('/api/train/options');
    const data = await response.json();
    
    if (!data.ok) {
      throw new Error(data.msg || 'Failed to load training options');
    }
    
    // Populate model dropdown
    const modelSelect = document.getElementById('train-model');
    if (modelSelect && data.models) {
      modelSelect.innerHTML = '';
      data.models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.id;
        option.textContent = model.name;
        modelSelect.appendChild(option);
      });
      
      // Set default
      if (data.defaults && data.defaults.model) {
        modelSelect.value = data.defaults.model;
      }
    }
    
    // Set other defaults
    if (data.defaults) {
      const inputMode = document.getElementById('train-input-mode');
      const epochs = document.getElementById('train-epochs');
      const batchSize = document.getElementById('train-batch-size');
      const augment = document.getElementById('train-augment');
      
      if (inputMode) inputMode.value = data.defaults.input_mode || 'single';
      if (epochs) {
        epochs.value = data.defaults.epochs || 3;
        updateSliderValue('epochs', epochs.value);
      }
      if (batchSize) {
        batchSize.value = data.defaults.batch_size || 32;
        updateSliderValue('batch-size', batchSize.value);
      }
      if (augment) augment.checked = data.defaults.augment || false;
      // show/hide single-role group based on default mode
      toggleSingleRoleVisibility();
      const sr = document.getElementById('single-role');
      if (sr && data.defaults.single_role) sr.value = data.defaults.single_role;
    }
    
  } catch (error) {
    console.error('Error loading training options:', error);
    throw error;
  }
}

// Load dataset summary
async function loadDatasetSummary() {
  try {
    const response = await fetch('/api/stats');
    const data = await response.json();
    
    if (data.error) {
      throw new Error(data.error);
    }
    
    // Update summary display - prefer group-level stats for TF training context
    const totalGroups = (typeof data.total_groups === 'number') ? data.total_groups : data.total_items || 0;
    const labeledGroups = (typeof data.labeled_groups === 'number') ? data.labeled_groups : data.labeled_items || 0;
    const dist = (data.class_distribution_groups || data.class_distribution || {});
    updateElement('total-images', totalGroups);
    updateElement('labeled-images', labeledGroups);
    updateElement('num-classes', Object.keys(dist).length);
    // Examples per Class summary (compact)
    try {
      const parts = Object.keys(dist).sort().map(k => `${k}: ${dist[k]}`);
      updateElement('examples-per-class', parts.length ? parts.join(', ') : '-');
    } catch(_) {
      updateElement('examples-per-class', '-');
    }
    // Build default label mapping UI from class distribution keys
    buildLabelMappingUI(Object.keys(data.class_distribution || {}));
    
  } catch (error) {
    console.error('Error loading dataset summary:', error);
    addLogEntry('warning', `Could not load dataset summary: ${error.message}`);
  }
}

function buildLabelMappingUI(labels) {
  const host = document.getElementById('label-mapping-list');
  if (!host) return;
  const existing = trainingState.classMap || {};
  // Default: enable all except Unknown/Unlabeled; assign incremental ints starting at 0
  const normalized = labels.slice();
  const defaultOrder = normalized;
  const defaultMap = {};
  let nextVal = 0;
  defaultOrder.forEach(name => {
    const lower = (name || '').toLowerCase();
    const enabled = !(lower.includes('unknown') || lower.includes('unlabeled'));
    if (enabled) defaultMap[name] = nextVal++;
  });
  trainingState.classMap = Object.keys(existing).length ? existing : defaultMap;
  host.innerHTML = '';
  // headers
  const header = document.createElement('div'); header.className = 'lm-row';
  header.innerHTML = '<div class="lm-header">Label</div><div class="lm-header">Include</div><div class="lm-header">Target</div>';
  host.appendChild(header);
  labels.forEach(name => {
    const row = document.createElement('div'); row.className = 'lm-row';
    const isEnabled = Object.prototype.hasOwnProperty.call(trainingState.classMap, name);
    const curVal = isEnabled ? trainingState.classMap[name] : '';
    row.innerHTML = `
      <div class="lm-label">${name}</div>
      <div class="lm-enable">
        <input type="checkbox" ${isEnabled ? 'checked' : ''} data-lm-enable="${name}">
      </div>
      <div class="lm-value">
        <input type="number" step="1" min="0" value="${curVal}" data-lm-value="${name}" ${isEnabled ? '' : 'disabled'}>
      </div>`;
    host.appendChild(row);
  });
  // wire events
  host.querySelectorAll('input[type="checkbox"]').forEach(cb => {
    cb.addEventListener('change', (e) => {
      const name = e.target.getAttribute('data-lm-enable');
      const valInput = host.querySelector(`input[data-lm-value="${CSS.escape(name)}"]`);
      if (e.target.checked) {
        const existingVal = parseInt(valInput.value, 10);
        const val = Number.isFinite(existingVal) ? existingVal : 0;
        trainingState.classMap[name] = val;
        if (valInput) valInput.disabled = false;
      } else {
        delete trainingState.classMap[name];
        if (valInput) valInput.disabled = true;
      }
      try { if (typeof drawTestHistogram === 'function') drawTestHistogram(); } catch(_){}
    });
  });
  host.querySelectorAll('input[type="number"]').forEach(inp => {
    inp.addEventListener('input', (e) => {
      const name = e.target.getAttribute('data-lm-value');
      const val = parseInt(e.target.value, 10);
      if (Number.isFinite(val)) {
        trainingState.classMap[name] = val;
      }
    });
  });
}

// Setup event listeners
function setupTrainingEventListeners() {
  // Navigation
  const backBtn = document.getElementById('back-to-main');
  if (backBtn) {
    backBtn.addEventListener('click', hideTrainingDashboard);
  }
  const returnBtn = document.getElementById('return-to-labeling');
  if (returnBtn){
    returnBtn.addEventListener('click', async ()=>{
      try{
        hideTrainingDashboard();
        if (typeof window.loadItems === 'function'){
          await window.loadItems();
        } else if (typeof loadItems === 'function'){
          await loadItems();
        }
        const grid = document.getElementById('grid');
        if (grid) grid.scrollIntoView({ behavior: 'smooth' });
      } catch(_){ hideTrainingDashboard(); }
    });
  }
  // Model Builder open button
  const mbBtn = document.getElementById('open-model-builder');
  if (mbBtn) {
    mbBtn.addEventListener('click', () => {
      if (window.modelBuilder && typeof window.modelBuilder.open === 'function') {
        window.modelBuilder.open();
      }
    });
  }
  
  // Training controls
  const startBtn = document.getElementById('start-training');
  const pauseBtn = document.getElementById('pause-training');
  const stopBtn = document.getElementById('stop-training');
  
  if (startBtn) startBtn.addEventListener('click', startTraining);
  if (pauseBtn) pauseBtn.addEventListener('click', pauseTraining);
  if (stopBtn) stopBtn.addEventListener('click', stopTraining);
  
  // Model management
  const saveModelBtn = document.getElementById('save-model');
  const exportModelBtn = document.getElementById('export-model');
  const testModelBtn = document.getElementById('test-model');
  
  if (saveModelBtn) saveModelBtn.addEventListener('click', saveModel);
  if (exportModelBtn) exportModelBtn.addEventListener('click', exportModel);
  if (testModelBtn) testModelBtn.addEventListener('click', testModel);
  
  // Log controls
  const clearLogsBtn = document.getElementById('clear-logs');
  const exportLogsBtn = document.getElementById('export-logs');
  
  if (clearLogsBtn) clearLogsBtn.addEventListener('click', clearLogs);
  if (exportLogsBtn) exportLogsBtn.addEventListener('click', exportLogs);
  
  // Metrics controls
  const exportMetricsBtn = document.getElementById('export-metrics');
  const resetChartsBtn = document.getElementById('reset-charts');
  
  if (exportMetricsBtn) exportMetricsBtn.addEventListener('click', exportMetrics);
  if (resetChartsBtn) resetChartsBtn.addEventListener('click', resetCharts);

  // Input mode toggle
  const inputModeSel = document.getElementById('train-input-mode');
  if (inputModeSel) {
    inputModeSel.addEventListener('change', toggleSingleRoleVisibility);
  }

  // Predict panel events
  const runPredictBtn = document.getElementById('run-predict');
  if (runPredictBtn){
    runPredictBtn.addEventListener('click', async ()=>{
      const repredict = document.getElementById('predict-repredict-all')?.checked || false;
      const allRemaining = document.getElementById('predict-all-remaining')?.checked || false;
      const limitInput = document.getElementById('predict-limit');
      const limitVal = (!allRemaining && limitInput) ? parseInt(limitInput.value || '300') : null;
      try {
        runPredictBtn.disabled = true;
        const body = { repredict_all: repredict };
        if (!allRemaining && Number.isFinite(limitVal) && limitVal > 0) { body.limit = limitVal; }
        const r = await fetch('/api/predictions/run', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
        const data = await r.json();
        if (!data.ok){ addLogEntry('warning', data.msg || 'Prediction run returned'); } else { addLogEntry('success', `Predicted ${data.predicted} triplet groups`); }
      } catch(e){ addLogEntry('error', `Prediction failed: ${e.message}`); }
      finally { runPredictBtn.disabled = false; }
      await refreshPredictionSummary();
      await drawPredictionHistogram();
      await drawTestHistogram();
    });
  }
  const thr = document.getElementById('pred-threshold');
  if (thr){
    // Initialize from saved threshold if available
    try {
      const savedThr = parseFloat(localStorage.getItem('al_pred_threshold') || '');
      if (Number.isFinite(savedThr)){
        thr.value = String(savedThr);
        const out0 = document.getElementById('pred-threshold-value');
        if (out0) out0.textContent = savedThr.toFixed(2);
      }
      // Persist current positive class name for cross-page display
      const pos0 = getPositiveClassName();
      if (pos0) localStorage.setItem('al_positive_class', pos0);
    } catch(_){ }
    thr.addEventListener('input', ()=>{
      const v = parseFloat(thr.value || '0');
      const out = document.getElementById('pred-threshold-value');
      if (out) out.textContent = v.toFixed(2);
      // Persist threshold and positive class for use on main grid
      try { localStorage.setItem('al_pred_threshold', String(v)); } catch(_){ }
      try { const pos = getPositiveClassName(); if (pos) localStorage.setItem('al_positive_class', pos); } catch(_){ }
      drawPredictionHistogram();
      drawTestHistogram();
    });
  }
  const applyBtn = document.getElementById('apply-threshold');
  if (applyBtn){
    applyBtn.addEventListener('click', async ()=>{
      const thrVal = parseFloat(document.getElementById('pred-threshold')?.value || '0.5');
      // Apply threshold only for the positive class from training mapping
      const pos = getPositiveClassName();
      if (!pos){ addLogEntry('warning', 'No positive class found in training mapping'); return; }
      // Persist selections
      try { localStorage.setItem('al_pred_threshold', String(thrVal)); } catch(_){ }
      try { localStorage.setItem('al_positive_class', pos); } catch(_){ }
      try{
        applyBtn.disabled = true;
        const r = await fetch('/api/predictions/apply-threshold', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ class_name: pos, negative_class: null, threshold: thrVal, unlabeled_only: true }) });
        const data = await r.json();
        if (!data.ok){ addLogEntry('error', data.msg || 'Apply threshold failed'); }
        else { addLogEntry('success', `Applied threshold ${thrVal.toFixed(2)} for ${pos}; updated ${data.updated} items`); }
      } catch(e){ addLogEntry('error', `Apply threshold failed: ${e.message}`); }
      finally{ applyBtn.disabled = false; }
    });
  }
}

// Setup slider interactions
function setupSliderInteractions() {
  const epochsSlider = document.getElementById('train-epochs');
  const batchSizeSlider = document.getElementById('train-batch-size');
  const splitSlider = document.getElementById('train-split');
  
  if (epochsSlider) {
    epochsSlider.addEventListener('input', (e) => {
      updateSliderValue('epochs', e.target.value);
    });
  }
  
  if (batchSizeSlider) {
    batchSizeSlider.addEventListener('input', (e) => {
      updateSliderValue('batch-size', e.target.value);
    });
  }

  if (splitSlider) {
    splitSlider.addEventListener('input', (e) => {
      updateSliderValue('train-split', e.target.value);
    });
  }
}

function toggleSingleRoleVisibility(){
  const mode = document.getElementById('train-input-mode')?.value || 'single';
  const group = document.getElementById('single-role-group');
  if (group){ group.style.display = (mode === 'single') ? 'block' : 'none'; }
}

// Update slider value display
function updateSliderValue(type, value) {
  const valueElement = document.getElementById(`${type}-value`);
  if (valueElement) {
    valueElement.textContent = value;
  }
}

// Initialize charts using Chart.js (if available) or simple canvas drawing
function initializeCharts() {
  const chartIds = ['loss-combined-chart', 'acc-combined-chart'];
  chartIds.forEach(chartId => {
    const canvas = document.getElementById(chartId);
    if (canvas) {
      const ctx = canvas.getContext('2d');
      trainingState.charts[chartId] = { canvas, ctx, data: [] };
      drawEmptyChart(ctx, canvas.width, canvas.height);
    }
  });
}

// ===== Prediction Panel =====
async function initPredictionPanel(){
  try{
    await refreshPredictionSummary();
    await drawPredictionHistogram();
    await drawTestHistogram();
  } catch(e){ console.warn('initPredictionPanel failed', e); }
}

async function refreshPredictionSummary(){
  try{
    const r = await fetch('/api/predictions/summary');
    const d = await r.json();
    updateElement('pred-remaining', d.remaining ?? '-');
    updateElement('pred-done', d.predicted ?? '-');
    updateElement('pred-total', d.total_items ?? '-');
  } catch(e){
    updateElement('pred-remaining', '-');
  }
}

async function drawPredictionHistogram(){
  try{
    const canvas = document.getElementById('pred-train-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0,0,canvas.width, canvas.height);
    const thrVal = parseFloat(document.getElementById('pred-threshold')?.value || '0.5');
    const pos = getPositiveClassName();
    const q = pos ? `?bins=40&class_name=${encodeURIComponent(pos)}` : `?bins=40`;
    const data = await (await fetch('/api/predictions/histogram' + q)).json();
    const edges = data.edges || []; const counts = data.counts || []; const total = data.total || 0;
    const w = canvas.width, h = canvas.height; const padL=30,padR=10,padT=10,padB=20; const plotW=w-padL-padR, plotH=h-padT-padB;
    // axes
    ctx.strokeStyle = '#374151'; ctx.beginPath(); ctx.moveTo(padL,padT); ctx.lineTo(padL,padT+plotH); ctx.lineTo(padL+plotW,padT+plotH); ctx.stroke();
    if (!(edges.length && counts.length)) return;
    const maxC = Math.max(1, ...counts); const binW = plotW / counts.length;
    for (let i=0;i<counts.length;i++){
      const bh = (counts[i]/maxC)*plotH; const x = padL + i*binW + 1; const y = padT + plotH - bh;
      ctx.fillStyle = 'rgba(56,189,248,0.75)'; ctx.fillRect(x,y, Math.max(1,binW-2), Math.max(1,bh));
    }
    // x-axis ticks 0..1
    ctx.save();
    ctx.strokeStyle = '#4b5563';
    ctx.fillStyle = '#9ca3af';
    ctx.font = '10px Arial';
    ctx.textAlign = 'center';
    for (let t=0; t<=5; t++){
      const v = t/5;
      const x = padL + v * plotW;
      const y = padT + plotH;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y + 4); ctx.stroke();
      const lab = v.toFixed(1);
      ctx.fillText(lab, x, y + 14);
    }
    ctx.restore();
    // threshold marker
    const xThr = padL + Math.max(0, Math.min(1, thrVal)) * plotW;
    ctx.save(); ctx.setLineDash([5,3]); ctx.strokeStyle = '#f59e0b'; ctx.beginPath(); ctx.moveTo(xThr,padT); ctx.lineTo(xThr,padT+plotH); ctx.stroke(); ctx.restore();
    // label
    const labelText = pos ? `${pos}  n=${total}  thr=${thrVal.toFixed(2)}` : `p_max  n=${total}  thr=${thrVal.toFixed(2)}`;
    ctx.fillStyle = '#9ca3af'; ctx.font = '12px Arial'; ctx.fillText(labelText, padL+4, padT+12);
  } catch(e){ /* ignore */ }
}

async function drawTestHistogram(){
  try{
    const canvas = document.getElementById('pred-test-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0,0,canvas.width, canvas.height);
    const thrVal = parseFloat(document.getElementById('pred-threshold')?.value || '0.5');
    const pos = getPositiveClassName();
    const q = pos ? `?bins=40&class_name=${encodeURIComponent(pos)}` : `?bins=40`;
    const resp = await fetch('/api/predictions/test-histogram' + q);
    const data = await resp.json();
    const edges = data.edges || []; let labels = data.labels || [];
    const byLabel = data.counts_by_label || {}; const total = data.total || 0;
    // Filter to active training labels only
    const activeNames = Object.keys(trainingState.classMap || {});
    labels = labels.filter(n => activeNames.includes(n));
    const w = canvas.width, h = canvas.height; const padL=30,padR=10,padT=10,padB=20; const plotW=w-padL-padR, plotH=h-padT-padB;
    // axes
    ctx.strokeStyle = '#374151'; ctx.beginPath(); ctx.moveTo(padL,padT); ctx.lineTo(padL,padT+plotH); ctx.lineTo(padL+plotW,padT+plotH); ctx.stroke();
    // prepare stacked bars
    const keys = labels.filter(l => Array.isArray(byLabel[l]));
    if (!(edges.length && keys.length)) return;
    const binCount = byLabel[keys[0]].length;
    const binW = plotW / Math.max(1, binCount);
    // compute max stack height
    const stackSums = new Array(binCount).fill(0);
    for (let i=0;i<binCount;i++){
      let s = 0; for (const k of keys){ s += (byLabel[k][i]||0); } stackSums[i]=s;
    }
    const maxC = Math.max(1, ...stackSums);
    // colors from class palette
    const getColor = (idx) => {
      if (typeof getClassColorByIndex === 'function'){ return getClassColorByIndex(idx); }
      const fallback = ['#38bdf8','#ef4444','#f59e0b','#10b981','#8b5cf6','#ec4899','#22c55e','#f97316'];
      return fallback[idx % fallback.length];
    };
    // x-axis ticks 0..1
    ctx.save();
    ctx.strokeStyle = '#4b5563';
    ctx.fillStyle = '#9ca3af';
    ctx.font = '10px Arial';
    ctx.textAlign = 'center';
    for (let t=0; t<=5; t++){
      const v = t/5;
      const x = padL + v * plotW;
      const y = padT + plotH;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y + 4); ctx.stroke();
      const lab = v.toFixed(1);
      ctx.fillText(lab, x, y + 14);
    }
    ctx.restore();
    // draw stacked bars
    for (let i=0;i<binCount;i++){
      let acc = 0;
      for (let li=0; li<keys.length; li++){
        const k = keys[li];
        const c = byLabel[k][i] || 0;
        const hpx = (c / maxC) * plotH;
        const x = padL + i*binW + 1;
        const y = padT + plotH - acc - hpx;
        const col = getColor(li);
        ctx.fillStyle = col;
        ctx.globalAlpha = 0.8;
        ctx.fillRect(x, y, Math.max(1, binW - 2), Math.max(1, hpx));
        ctx.globalAlpha = 1.0;
        acc += hpx;
      }
    }
    // threshold marker
    const xThr = padL + Math.max(0, Math.min(1, thrVal)) * plotW;
    ctx.save(); ctx.setLineDash([5,3]); ctx.strokeStyle = '#f59e0b'; ctx.beginPath(); ctx.moveTo(xThr,padT); ctx.lineTo(xThr,padT+plotH); ctx.stroke(); ctx.restore();
    // label
    const labelText = pos ? `${pos} test  n=${total}  thr=${thrVal.toFixed(2)}` : `p_max test  n=${total}  thr=${thrVal.toFixed(2)}`;
    ctx.fillStyle = '#9ca3af'; ctx.font = '12px Arial'; ctx.fillText(labelText, padL+4, padT+12);
    // legend
    const legend = document.getElementById('pred-test-legend');
    if (legend){
      legend.innerHTML = '';
      keys.forEach((k, li)=>{
        const sw = document.createElement('span');
        const col = getColor(li);
        sw.className = 'legend-item';
        sw.innerHTML = `<span class="legend-swatch" style="background:${col}"></span>${k}`;
        legend.appendChild(sw);
      });
    }
  } catch(e){ /* ignore */ }
}

// Draw empty chart
function drawEmptyChart(ctx, width, height) {
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = '#374151';
  ctx.lineWidth = 1;
  
  // Draw grid
  const gridSize = 20;
  for (let x = 0; x <= width; x += gridSize) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  
  for (let y = 0; y <= height; y += gridSize) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  
  // Draw placeholder text
  ctx.fillStyle = '#6b7280';
  ctx.font = '14px Arial';
  ctx.textAlign = 'center';
  ctx.fillText('No data yet', width / 2, height / 2);
}

// Start training
async function startTraining() {
  try {
    const config = getTrainingConfig();
    
    addLogEntry('info', `Starting training with config: ${JSON.stringify(config)}`);
    
    const response = await fetch('/api/train/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    
    const data = await response.json();
    
    if (!data.ok) {
      throw new Error(data.msg || 'Failed to start training');
    }
    
    trainingState.isTraining = true;
    trainingState.startTime = Date.now();
    trainingState.totalEpochs = config.epochs;
    
    updateTrainingControls();
    updateStatusIndicator('preparing');
    
    addLogEntry('success', 'Training started successfully');
    
  } catch (error) {
    console.error('Failed to start training:', error);
    addLogEntry('error', `Failed to start training: ${error.message}`);
  }
}

// Pause training
async function pauseTraining() {
  // Note: Backend doesn't support pause, so we'll implement stop instead
  await stopTraining();
}

// Stop training
async function stopTraining() {
  try {
    const response = await fetch('/api/train/cancel', { method: 'POST' });
    const data = await response.json();
    
    if (!data.ok) {
      throw new Error(data.msg || 'Failed to stop training');
    }
    
    trainingState.isTraining = false;
    updateTrainingControls();
    updateStatusIndicator('idle');
    
    addLogEntry('warning', 'Training stopped by user');
    
  } catch (error) {
    console.error('Failed to stop training:', error);
    addLogEntry('error', `Failed to stop training: ${error.message}`);
  }
}

// Get training configuration from form
function getTrainingConfig() {
  return {
    model: document.getElementById('train-model')?.value || 'mobilenet_v2',
    input_mode: document.getElementById('train-input-mode')?.value || 'single',
    epochs: parseInt(document.getElementById('train-epochs')?.value || '3'),
    batch_size: parseInt(document.getElementById('train-batch-size')?.value || '32'),
    augment: document.getElementById('train-augment')?.checked || false,
    class_map: trainingState.classMap || {},
    split_pct: parseInt(document.getElementById('train-split')?.value || '85'),
    split_strategy: document.getElementById('split-strategy')?.value || 'natural',
    single_role: document.getElementById('single-role')?.value || 'target'
  };
}

// Update training controls state
function updateTrainingControls() {
  const startBtn = document.getElementById('start-training');
  const pauseBtn = document.getElementById('pause-training');
  const stopBtn = document.getElementById('stop-training');
  
  if (trainingState.isTraining) {
    if (startBtn) startBtn.disabled = true;
    if (pauseBtn) pauseBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = false;
  } else {
    if (startBtn) startBtn.disabled = false;
    if (pauseBtn) pauseBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = true;
  }
}

// Update status indicator
function updateStatusIndicator(status) {
  const indicator = document.getElementById('train-status-indicator');
  const statusText = document.getElementById('train-status-text');
  
  if (indicator) {
    indicator.className = `status-indicator ${status}`;
  }
  
  if (statusText) {
    const statusMap = {
      idle: 'Ready to Train',
      preparing: 'Preparing...',
      training: 'Training in Progress',
      error: 'Error Occurred',
      done: 'Training Complete'
    };
    statusText.textContent = statusMap[status] || 'Unknown Status';
  }
}

// Start status polling
function startTrainingStatusPolling() {
  if (trainingState.statusTimer) {
    clearInterval(trainingState.statusTimer);
  }
  
  trainingState.statusTimer = setInterval(updateTrainingStatus, 1000);
}

// Stop status polling
function stopTrainingStatusPolling() {
  if (trainingState.statusTimer) {
    clearInterval(trainingState.statusTimer);
    trainingState.statusTimer = null;
  }
}

// Update training status from backend
async function updateTrainingStatus() {
  try {
    const response = await fetch('/api/train/status');
    const status = await response.json();
    
    // Update status indicator
    updateStatusIndicator(status.stage || 'idle');
    
    // Update device info
    updateElement('train-device-info', status.device || 'Unknown');
    
    // Update progress
    if (status.running) {
      trainingState.isTraining = true;
      trainingState.currentEpoch = status.epoch || 0;
      trainingState.totalEpochs = status.total_epochs || 0;
      
      updateProgress();
      updateMetrics(status);
      updateCharts(status);
    } else if (trainingState.isTraining) {
      // Training just finished
      trainingState.isTraining = false;
      updateTrainingControls();
      
      if (status.stage === 'done') {
        addLogEntry('success', 'Training completed successfully');
        // refresh predict panel after training
        try { await refreshPredictionSummary(); await drawPredictionHistogram(); await drawTestHistogram(); } catch(_){}
        // Reveal the return-to-main button after successful training
        try {
          const returnBtn2 = document.getElementById('return-to-labeling');
          if (returnBtn2){ returnBtn2.style.display = 'inline-block'; }
        } catch(_){ }
      } else if (status.stage === 'error') {
        addLogEntry('error', `Training failed: ${status.message || 'Unknown error'}`);
      } else if (status.stage === 'cancelled') {
        addLogEntry('warning', 'Training was cancelled');
      }
    }
    
  } catch (error) {
    console.error('Failed to update training status:', error);
  }
}

// Update progress bars
function updateProgress() {
  const overallProgress = trainingState.totalEpochs > 0 ? 
    (trainingState.currentEpoch / trainingState.totalEpochs) * 100 : 0;
  
  updateProgressBar('overall-progress', overallProgress);
  updateElement('current-epoch', trainingState.currentEpoch);
  updateElement('total-epochs', trainingState.totalEpochs);
  
  // Update time estimate
  if (trainingState.startTime && trainingState.currentEpoch > 0) {
    const elapsed = Date.now() - trainingState.startTime;
    const avgTimePerEpoch = elapsed / trainingState.currentEpoch;
    const remaining = (trainingState.totalEpochs - trainingState.currentEpoch) * avgTimePerEpoch;
    
    updateElement('time-remaining', formatTime(remaining));
  }
}

// Update progress bar
function updateProgressBar(id, percentage) {
  const progressBar = document.getElementById(id);
  if (progressBar) {
    progressBar.style.width = `${Math.max(0, Math.min(100, percentage))}%`;
    progressBar.textContent = `${Math.round(percentage)}%`;
  }
}

// Update metrics display
function updateMetrics(status) {
  if (status.loss !== undefined) {
    updateElement('current-loss', status.loss.toFixed(4));
    trainingState.metrics.loss.push(status.loss);
  }
  
  if (status.acc !== undefined) {
    updateElement('current-accuracy', (status.acc * 100).toFixed(1) + '%');
    trainingState.metrics.accuracy.push(status.acc);
  }
  
  if (status.val_loss !== undefined) {
    updateElement('current-val-loss', status.val_loss.toFixed(4));
    trainingState.metrics.valLoss.push(status.val_loss);
  }
  
  if (status.val_acc !== undefined) {
    updateElement('current-val-accuracy', (status.val_acc * 100).toFixed(1) + '%');
    trainingState.metrics.valAccuracy.push(status.val_acc);
    
    // Update best validation accuracy
    const bestValAcc = Math.max(...trainingState.metrics.valAccuracy);
    updateElement('best-val-acc', (bestValAcc * 100).toFixed(1) + '%');
  }
}

// Update charts with new data
function updateCharts(status) {
  if (!status.history) return;
  const h = status.history;
  if ((h.loss && h.loss.length) || (h.val_loss && h.val_loss.length)) {
    updateDualSeriesChart('loss-combined-chart', h.loss || [], h.val_loss || [], 'Loss', '#ef4444', '#f59e0b');
  }
  if ((h.acc && h.acc.length) || (h.val_acc && h.val_acc.length)) {
    updateDualSeriesChart('acc-combined-chart', h.acc || [], h.val_acc || [], 'Accuracy', '#10b981', '#3b82f6');
  }
}

// Update individual chart
function updateDualSeriesChart(chartId, seriesA, seriesB, label, colorA, colorB) {
  const chart = trainingState.charts[chartId];
  if (!chart) return;
  const { ctx, canvas } = chart;
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#1f2937';
  ctx.fillRect(0, 0, width, height);
  const padding = 40;
  const chartWidth = width - 2 * padding;
  const chartHeight = height - 2 * padding;
  const allValues = [...(seriesA || []), ...(seriesB || [])].filter(v => typeof v === 'number');
  const minValue = allValues.length ? Math.min(...allValues) : 0;
  const maxValue = allValues.length ? Math.max(...allValues) : 1;
  const valueRange = maxValue - minValue || 1;
  ctx.strokeStyle = '#374151';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = padding + (i / 5) * chartHeight;
    ctx.beginPath();
    ctx.moveTo(padding, y);
    ctx.lineTo(width - padding, y);
    ctx.stroke();
  }
  const drawSeries = (data, color) => {
    if (!data || data.length < 1) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((value, index) => {
      const x = padding + (index / Math.max(1, data.length - 1)) * chartWidth;
      const y = height - padding - ((value - minValue) / valueRange) * chartHeight;
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  drawSeries(seriesA, colorA);
  drawSeries(seriesB, colorB);
  ctx.fillStyle = '#9ca3af';
  ctx.font = '12px Arial';
  ctx.textAlign = 'left';
  const aText = seriesA && seriesA.length ? seriesA[seriesA.length - 1].toFixed(4) : 'N/A';
  const bText = seriesB && seriesB.length ? seriesB[seriesB.length - 1].toFixed(4) : 'N/A';
  ctx.fillText(`${label}: train ${aText} | val ${bText}`, padding, 20);
  // X-axis tick marks and labels
  const maxDataLength = Math.max(seriesA?.length || 0, seriesB?.length || 0);
  if (maxDataLength > 0) {
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#9ca3af';
    ctx.font = '10px Arial';
    ctx.textAlign = 'center';
    
    // Draw x-axis ticks and labels
    const numTicks = Math.min(10, maxDataLength); // Show up to 10 ticks
    for (let i = 0; i <= numTicks; i++) {
      const tickIndex = Math.round((i / numTicks) * (maxDataLength - 1));
      const x = padding + (tickIndex / Math.max(1, maxDataLength - 1)) * chartWidth;
      const y = height - padding;
      
      // Draw tick mark
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x, y + 4);
      ctx.stroke();
      
      // Draw label (epoch number)
      ctx.fillText(tickIndex.toString(), x, y + 16);
    }
  }
  
  // Legend
  const legendY = padding - 10;
  ctx.fillStyle = colorA; ctx.fillRect(width - padding - 180, legendY - 8, 12, 12);
  ctx.fillStyle = '#9ca3af'; ctx.fillText('train', width - padding - 160, legendY + 2);
  ctx.fillStyle = colorB; ctx.fillRect(width - padding - 110, legendY - 8, 12, 12);
  ctx.fillStyle = '#9ca3af'; ctx.fillText('val', width - padding - 90, legendY + 2);
}

// Model management functions
async function saveModel() {
  addLogEntry('info', 'Saving model...');
  // Implementation would depend on backend API
}

async function exportModel() {
  try {
    const response = await fetch('/api/export-model');
    if (response.ok) {
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'model_artifacts.zip';
      a.click();
      window.URL.revokeObjectURL(url);
      
      addLogEntry('success', 'Model exported successfully');
    } else {
      throw new Error('Failed to export model');
    }
  } catch (error) {
    console.error('Export error:', error);
    addLogEntry('error', `Export failed: ${error.message}`);
  }
}

async function testModel() {
  addLogEntry('info', 'Testing model performance...');
  // Implementation would test model on validation set
}

// Log management
function addLogEntry(type, message) {
  const logOutput = document.getElementById('training-logs');
  if (!logOutput) return;
  
  const entry = document.createElement('div');
  entry.className = `log-entry ${type}`;
  
  const time = document.createElement('span');
  time.className = 'log-time';
  time.textContent = `[${new Date().toLocaleTimeString()}]`;
  
  const msg = document.createElement('span');
  msg.className = 'log-message';
  msg.textContent = message;
  
  entry.appendChild(time);
  entry.appendChild(msg);
  
  logOutput.appendChild(entry);
  
  // Auto-scroll if enabled
  const autoScroll = document.getElementById('auto-scroll-logs');
  if (autoScroll && autoScroll.checked) {
    logOutput.scrollTop = logOutput.scrollHeight;
  }
}

function clearLogs() {
  const logOutput = document.getElementById('training-logs');
  if (logOutput) {
    logOutput.innerHTML = '';
    addLogEntry('info', 'Logs cleared');
  }
}

function exportLogs() {
  const logOutput = document.getElementById('training-logs');
  if (!logOutput) return;
  
  const logs = Array.from(logOutput.children).map(entry => {
    const time = entry.querySelector('.log-time')?.textContent || '';
    const message = entry.querySelector('.log-message')?.textContent || '';
    return `${time} ${message}`;
  }).join('\n');
  
  const blob = new Blob([logs], { type: 'text/plain' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `training_logs_${new Date().toISOString().split('T')[0]}.txt`;
  a.click();
  window.URL.revokeObjectURL(url);
  
  addLogEntry('success', 'Logs exported');
}

// Metrics management
function exportMetrics() {
  const metrics = {
    timestamp: new Date().toISOString(),
    loss: trainingState.metrics.loss,
    accuracy: trainingState.metrics.accuracy,
    valLoss: trainingState.metrics.valLoss,
    valAccuracy: trainingState.metrics.valAccuracy
  };
  
  const blob = new Blob([JSON.stringify(metrics, null, 2)], { type: 'application/json' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `training_metrics_${new Date().toISOString().split('T')[0]}.json`;
  a.click();
  window.URL.revokeObjectURL(url);
  
  addLogEntry('success', 'Metrics exported');
}

function resetCharts() {
  trainingState.metrics = {
    loss: [],
    accuracy: [],
    valLoss: [],
    valAccuracy: []
  };
  
  // Reinitialize charts
  initializeCharts();
  
  addLogEntry('info', 'Charts reset');
}

// Utility functions
function updateElement(id, value) {
  const element = document.getElementById(id);
  if (element) {
    element.textContent = value;
  }
}

function formatTime(milliseconds) {
  const seconds = Math.floor(milliseconds / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  
  if (hours > 0) {
    return `${hours}:${(minutes % 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`;
  } else {
    return `${minutes}:${(seconds % 60).toString().padStart(2, '0')}`;
  }
}

// Export functions for use in other scripts
window.trainingDashboard = {
  show: showTrainingDashboard,
  hide: hideTrainingDashboard
};

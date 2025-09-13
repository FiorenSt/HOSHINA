async function loadStats(){
  try {
    const data = await api('/stats');
    state.stats = data;
    updateStatsDisplay();
  } catch (e) {
    console.error('Failed to load stats:', e);
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
    container.innerHTML = '<div class="no-suggestions">No suggestions available. Try retraining the model first.</div>';
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
  container.querySelectorAll('.suggestion-item').forEach(item => {
    const id = parseInt(item.dataset.id);
    item.addEventListener('click', () => showImageZoom(id));
  });
}

function showImageZoom(itemId){
  const item = state.items.find(i => i.id === itemId);
  if (!item) return;
  const title = el('#imageModalTitle');
  if (title) title.textContent = `Image: ${item.path.split('/').pop()}`;
  const three = el('#tripletThreeUp');
  const single = el('#imageModalImg');
  // reset and draw prediction histogram overlay
  renderPredHistogramOverlay(item.max_proba);
  api(`/triplet/${itemId}`).then(trip=>{
    const hasAny = trip.items && (trip.items.target || trip.items.ref || trip.items.diff);
    if (!hasAny){
      if (three) three.style.display = 'none';
      if (single) { single.style.display = 'block'; single.src = `/api/file/${itemId}`; }
      openModal('#imageModal');
      return;
    }
    if (three && single) {
      three.style.display = 'grid';
      single.style.display = 'none';
      
      // Set image sources and remove any assignment colors, but don't add click handlers
      const targetImg = el('#imgTarget');
      const refImg = el('#imgRef');
      const diffImg = el('#imgDiff');
      
      if (targetImg) {
        targetImg.classList.remove('assign-blue','assign-red','assign-yellow');
        // Remove any existing click listeners by cloning
        const newTarget = targetImg.cloneNode(false);
        newTarget.src = trip.items.target?.file || '';
        targetImg.parentNode.replaceChild(newTarget, targetImg);
      }
      
      if (refImg) {
        refImg.classList.remove('assign-blue','assign-red','assign-yellow');
        // Remove any existing click listeners by cloning
        const newRef = refImg.cloneNode(false);
        newRef.src = trip.items.ref?.file || '';
        refImg.parentNode.replaceChild(newRef, refImg);
      }
      
      if (diffImg) {
        diffImg.classList.remove('assign-blue','assign-red','assign-yellow');
        // Remove any existing click listeners by cloning
        const newDiff = diffImg.cloneNode(false);
        newDiff.src = trip.items.diff?.file || '';
        diffImg.parentNode.replaceChild(newDiff, diffImg);
      }
    }
    openModal('#imageModal');
  }).catch(()=>{
    if (three) three.style.display = 'none';
    if (single) { single.style.display = 'block'; single.src = `/api/file/${itemId}`; }
    openModal('#imageModal');
  });
}

async function renderPredHistogramOverlay(currentProb){
  try {
    const overlay = el('#predHistOverlay');
    const canvas = el('#predHistCanvas');
    if (!overlay || !canvas) return;
    overlay.style.display = 'block';
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0,0,canvas.width, canvas.height);

    const data = await api(`/predictions/histogram?bins=20`);
    const edges = data.edges || [];
    const counts = data.counts || [];
    const total = data.total || 0;
    if (!edges.length || !counts.length) {
      // draw empty axes
      drawHistAxes(ctx, canvas.width, canvas.height);
      return;
    }

    const w = canvas.width, h = canvas.height;
    const padL = 20, padR = 10, padB = 16, padT = 10;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;
    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    ctx.strokeStyle = 'rgba(255,255,255,0.6)';
    ctx.lineWidth = 1;
    // axes
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, padT + plotH);
    ctx.lineTo(padL + plotW, padT + plotH);
    ctx.stroke();

    // x-axis ticks 0..1
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.6)';
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.font = '10px system-ui, sans-serif';
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

    const maxCount = Math.max(1, ...counts);
    const binCount = counts.length;
    const binW = plotW / binCount;
    for (let i=0;i<binCount;i++){
      const c = counts[i];
      const bh = (c / maxCount) * plotH;
      const x = padL + i * binW + 1;
      const y = padT + plotH - bh;
      ctx.fillStyle = 'rgba(56,189,248,0.7)';
      ctx.fillRect(x, y, Math.max(1, binW - 2), Math.max(1, bh));
    }

    // dashed line at current prob
    if (typeof currentProb === 'number' && currentProb >= 0 && currentProb <= 1){
      const x = padL + currentProb * plotW;
      ctx.save();
      ctx.strokeStyle = 'rgba(245,158,11,0.9)';
      ctx.setLineDash([4,3]);
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, padT + plotH);
      ctx.stroke();
      ctx.restore();
    }

    // label
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.font = '10px system-ui, sans-serif';
    const lab = typeof currentProb === 'number' ? `p_max=${currentProb.toFixed(2)}  n=${total}` : `n=${total}`;
    ctx.fillText(lab, padL + 4, padT + 12);
  } catch(_){
    const overlay = el('#predHistOverlay');
    if (overlay) overlay.style.display = 'none';
  }
}

function drawHistAxes(ctx, w, h){
  const padL = 20, padR = 10, padB = 16, padT = 10;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  ctx.strokeStyle = 'rgba(255,255,255,0.6)';
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + plotH);
  ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();
}

function updateStatsDisplay(){
  if (!state.stats) return;
  const stats = state.stats;
  const progressBar = el('#progress-bar');
  if (progressBar) {
    progressBar.style.width = `${stats.progress_percentage}%`;
    progressBar.textContent = `${stats.progress_percentage}%`;
  }
  // const statusText = `${stats.labeled_items}/${stats.total_items} labeled (${stats.progress_percentage}%)`;
  // if (!state.loading) status(statusText);
}

function openModal(id){
  el(id).classList.remove('hidden');
  document.body.classList.add('modal-open');
}
function closeModal(id){
  // cleanup histogram overlay when closing image modal
  if (id === '#imageModal'){
    const overlay = el('#predHistOverlay');
    const canvas = el('#predHistCanvas');
    if (overlay) overlay.style.display = 'none';
    if (canvas){
      const ctx = canvas.getContext('2d');
      ctx && ctx.clearRect(0,0,canvas.width, canvas.height);
    }
  }
  el(id).classList.add('hidden');
  document.body.classList.remove('modal-open');
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
  // Prefer group-based stats if present
  const totalGroups = (typeof stats.total_groups === 'number') ? stats.total_groups : null;
  const labeledGroups = (typeof stats.labeled_groups === 'number') ? stats.labeled_groups : null;
  const progressPercent = (totalGroups !== null && labeledGroups !== null && totalGroups > 0)
    ? Math.round((labeledGroups / totalGroups) * 100)
    : (stats.progress_percentage || 0);
  const totalItems = (totalGroups !== null) ? totalGroups : (stats.total_items || 0);
  const labeledItems = (labeledGroups !== null) ? labeledGroups : (stats.labeled_items || 0);
  const unlabeledItems = (totalGroups !== null && labeledGroups !== null)
    ? Math.max(0, totalGroups - labeledGroups)
    : (stats.unlabeled_items || 0);
  const skippedItems = stats.skipped_items || 0;
  const unsureItems = stats.unsure_items || 0;
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
        ${Object.entries((stats.class_distribution_groups || stats.class_distribution || {})).map(([className, count]) => `
          <div class="stats-item">
            <span class="stats-item-label">${className}</span>
            <span class="stats-item-value">${count.toLocaleString()}</span>
          </div>
        `).join('')}
        ${Object.keys(stats.class_distribution_groups || stats.class_distribution || {}).length === 0 ? 
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
    updateBatchClassButtons();
  }
}

function updateBatchClassButtons(){
  const container = el('#batch-class-buttons');
  if (!container) return;
  container.innerHTML = '';
  if (state.classes.length === 0) return;
  state.classes.forEach((cls, i) => {
    const btn = document.createElement('button');
    btn.className = 'batch-class-btn';
    btn.textContent = `${cls.key}: ${cls.name}`;
    btn.title = `Batch label as ${cls.name}`;
    // apply color if available
    if (typeof getClassColorByIndex === 'function'){
      const c = getClassColorByIndex(i);
      if (c){
        btn.style.borderColor = c;
        btn.style.background = hexToRgba(c, 0.15);
        btn.style.color = '#fff';
      }
    }
    btn.addEventListener('click', () => {
      batchLabelItems(Array.from(state.selectedItems), cls.name, false, false);
    });
    container.appendChild(btn);
  });
}

function updatePagination(){
  const showPagination = state.total_pages > 1;
  const topPagination = el('#pagination-controls');
  const bottomPagination = el('#bottom-pagination');
  const pageIndicator = el('#page-indicator');
  // Always show bottom pagination for controls, but hide top pagination and page indicator if only 1 page
  if (topPagination) topPagination.style.display = showPagination ? 'flex' : 'none';
  if (bottomPagination) bottomPagination.style.display = 'flex'; // Always show for page size/tile size controls
  if (pageIndicator) pageIndicator.style.display = showPagination ? 'block' : 'none';
  if (pageIndicator) {
    const currentPageEl = el('#current-page');
    const totalPagesEl = el('#total-pages');
    if (currentPageEl) currentPageEl.textContent = state.page;
    if (totalPagesEl) totalPagesEl.textContent = state.total_pages;
  }
  const startItem = (state.page - 1) * state.page_size + 1;
  const endItem = Math.min(state.page * state.page_size, state.total_items);
  const topInfo = el('#pagination-info-text');
  const bottomInfo = el('#bottom-pagination-info');
  if (topInfo) topInfo.textContent = `Page ${state.page} of ${state.total_pages}`;
  if (bottomInfo) bottomInfo.textContent = `Showing ${startItem}-${endItem} of ${state.total_items} groups`;
  const isFirstPage = state.page <= 1;
  const isLastPage = state.page >= state.total_pages;
  const firstBtn = el('#first-page');
  const prevBtn = el('#prev-page');
  const nextBtn = el('#next-page');
  const lastBtn = el('#last-page');
  if (firstBtn) firstBtn.disabled = isFirstPage;
  if (prevBtn) prevBtn.disabled = isFirstPage;
  if (nextBtn) nextBtn.disabled = isLastPage;
  if (lastBtn) lastBtn.disabled = isLastPage;
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

function showHelp(){
  const helpText = `
🏷️ HOSHINA - Keyboard Shortcuts:

📝 Labeling:
• 1-9: Label with class (or batch label if items selected)
• 0: Mark as unsure 🤔
• X: Skip item ⏭️

🚀 Actions:
• R: Retrain model
• S: View statistics 📊
• V: View map/visualization 🗺️
• H or ?: Show this help

📄 Navigation:
• ← →: Previous/Next page
• Ctrl+A: Select all visible items
• Escape: Clear selection

✅ Selection & Interaction:
• Click anywhere on a tile to select/deselect
• Click the small lens button to zoom 🔍
• Use quick controls in header for page size/tile size
• Drag on empty grid space to select multiple

💡 Pro Tips:
• Images are now extra large by default for better visibility
• Page size and tile size controls are always visible in header
• Batch operations work with keyboard shortcuts too

🎯 Single Mode:
• Numbers 1-9: Assign current item to class
• 0: Mark unsure
• X: Skip
• ← → or Enter: Previous/Next item
  `;
  alert(helpText);
}


// ===== Single Mode Helpers =====
function updateSingleClassButtons(){
  const container = el('#single-class-buttons');
  if (!container) return;
  container.innerHTML = '';
  if (!state.classes || state.classes.length === 0) return;
  state.classes.forEach((cls, i) => {
    const key = (i+1).toString();
    const btn = document.createElement('button');
    btn.className = 'batch-class-btn';
    btn.textContent = `${key}: ${cls.name}`;
    btn.title = `Assign ${cls.name}`;
    // apply color if available
    if (typeof getClassColorByIndex === 'function'){
      const c = getClassColorByIndex(i);
      if (c){
        btn.style.borderColor = c;
        btn.style.background = hexToRgba(c, 0.15);
        btn.style.color = '#fff';
      }
    }
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      // Prevent double-clicks and rapid firing
      if (btn.disabled) return;
      btn.disabled = true;
      setTimeout(() => { btn.disabled = false; }, 500);
      
      const grp = state.groups[state.groupIndex];
      if (!grp) return;
      // Fire-and-forget to keep UI responsive; load refresh happens in background
      batchLabelItems(grp.memberIds, cls.name, false, false).catch(()=>{});
      advanceSingleNext();
    });
    container.appendChild(btn);
  });
}

async function renderSingleView(){
  const wrapper = el('#single-view');
  const grid = el('#grid');
  if (!wrapper) return;
  if (state.mode === 'single') {
    if (grid) grid.style.display = 'none';
    wrapper.style.display = 'block';
    // Hide apply-assignments controls in single view
    try {
      const ac = el('#assignment-controls'); if (ac) ac.style.display = 'none';
      const bac = el('#bottom-assignment-controls'); if (bac) bac.style.display = 'none';
    } catch(_){ }
  } else {
    wrapper.style.display = 'none';
    if (grid) grid.style.display = '';
    // Restore apply-assignments controls outside single view
    try {
      const ac = el('#assignment-controls'); if (ac) ac.style.display = '';
      const bac = el('#bottom-assignment-controls'); if (bac) bac.style.display = '';
    } catch(_){ }
    return;
  }
  const grp = state.groups[state.groupIndex];
  const repId = grp ? grp.repId : undefined;
  // Track stable key so background reloads can re-align the current group
  if (grp && grp.key) state.currentGroupKey = grp.key;
  const item = repId ? state.items.find(i=>i.id===repId) || state.items[0] : state.items[0];
  if (!item){
    el('#single-img')?.setAttribute('src','');
    const fn = el('#single-filename'); if (fn) fn.textContent = '';
    const pr = el('#single-pred'); if (pr) pr.textContent = '';
    updateSingleClassButtons();
    return;
  }
  const filename = item.path ? item.path.split('/').pop() : `ID ${item.id}`;
  const pred = item.pred_label ? `pred: ${item.pred_label}` : '';
  const fn = el('#single-filename'); if (fn) fn.textContent = filename;
  const pr = el('#single-pred'); if (pr) pr.textContent = pred;

  // Show representative image immediately, then load triplet asynchronously and replace if available
  const baseId = repId || item.id;
  state.currentRenderId = baseId;
  try {
    const imgHost = el('.single-image');
    if (imgHost) {
      imgHost.innerHTML = '<img id="single-img" style="max-width:100%; max-height: 80vh; object-fit: contain;" />';
    }
    const img = el('#single-img');
    if (img) img.src = `/api/file/${baseId}`;
  } catch(_){ }

  // Fetch triplet in background; only render if still on same item
  api(`/triplet/${baseId}`).then(trip => {
    if (state.currentRenderId !== baseId) return;
    const hasAny = trip.items && (trip.items.target || trip.items.ref || trip.items.diff);
    if (!hasAny) return;
    const imgHost = el('.single-image');
    if (!imgHost) return;
    const size = (state && state.tile_size) || 'xlarge';
    const vh = size === 'small' ? '28vh' : size === 'medium' ? '36vh' : size === 'large' ? '44vh' : '50vh';
    try { imgHost.style.padding = '0'; imgHost.style.margin = '0'; imgHost.style.lineHeight = '0'; } catch(_){ }
    imgHost.innerHTML = `
      <div id="tripletGrid" style="position:relative; display:grid; grid-template-columns: repeat(3, 1fr); gap:2px; width:80%; margin:0 auto; padding:0 24px; line-height:0; font-size:0;">
        <img id=\"singleTarget\" style=\"display:block; width:100%; height:${vh}; object-fit: contain; margin:0; padding:0; border:0; outline:0; box-shadow:none; vertical-align:top;\" />
        <img id=\"singleRef\" style=\"display:block; width:100%; height:${vh}; object-fit: contain; margin:0; padding:0; border:0; outline:0; box-shadow:none; vertical-align:top;\" />
        <img id=\"singleDiff\" style=\"display:block; width:100%; height:${vh}; object-fit: contain; margin:0; padding:0; border:0; outline:0; box-shadow:none; vertical-align:top;\" />
        <span id="tripletLabel" style="position:absolute; right:12px; bottom:8px; padding:4px 8px; font-weight:700; font-size:14px; border-radius:4px; background: rgba(0,0,0,0.5); color:#fff; pointer-events:none; display:none;"></span>
      </div>`;
    const t = el('#singleTarget'); const r = el('#singleRef'); const d = el('#singleDiff');
    if (t) t.src = trip.items.target?.file || '';
    if (r) r.src = trip.items.ref?.file || '';
    if (d) d.src = trip.items.diff?.file || '';
    // Click-cycling for triplet with bottom-right label in class color
    try {
      let currentIdx = -1;
      const imgs = [t, r, d].filter(Boolean);
      const labelEl = el('#tripletLabel');
      const applyTripletClass = (idx)=>{
        currentIdx = idx;
        const color = (typeof getClassColorByIndex === 'function' && idx >= 0) ? getClassColorByIndex(idx) : null;
        imgs.forEach(img=>{
          if (!img) return;
          if (!color){
            img.style.outline = '0';
            img.style.outlineOffset = '0';
            img.style.boxShadow = 'none';
          } else {
            img.style.outline = `3px solid ${color}`;
            img.style.outlineOffset = '-3px';
            img.style.boxShadow = `0 0 0 2px ${hexToRgba(color, 0.25)} inset`;
          }
        });
        if (labelEl){
          const name = (idx >= 0 && state.classes && state.classes[idx] && state.classes[idx].name) ? state.classes[idx].name : '';
          if (name && color){
            labelEl.textContent = name;
            labelEl.style.color = color;
            labelEl.style.display = 'inline-block';
          } else {
            labelEl.textContent = '';
            labelEl.style.display = 'none';
          }
        }
      };
      const cycle = ()=>{
        const n = (state.classes || []).length;
        if (n <= 0){ 
          status('⚠️ Please define classes first before labeling images. Go to the Classes tab to set up your classification labels.');
          return; 
        }
        const next = (currentIdx + 1) >= n ? -1 : (currentIdx + 1);
        applyTripletClass(next);
      };
      imgs.forEach(img=>{ img && img.addEventListener('click', cycle); });
    } catch(_){ }
  }).catch(()=>{});

  // Prefetch next images to reduce latency
  try { prefetchNextGroupImages(); } catch(_){ }

  updateSingleClassButtons();
}

async function navigateSingle(delta){
  const nextIndex = state.groupIndex + delta;
  if (nextIndex >= 0 && nextIndex < state.groups.length){
    state.groupIndex = nextIndex;
    await renderSingleView();
    return;
  }
  // Cross page navigation if available
  if (delta > 0 && state.page < state.total_pages){
    state.groupIndex = 0;
    await goToPage(state.page + 1);
    await renderSingleView();
  } else if (delta < 0 && state.page > 1){
    await goToPage(state.page - 1);
    state.groupIndex = Math.max(0, (state.groups.length || 1) - 1);
    await renderSingleView();
  }
}

async function advanceSingleNext(){
  state.navigating = true;
  try {
    await navigateSingle(1);
  } finally {
    state.navigating = false;
  }
}

// Expose for other modules
if (typeof window !== 'undefined'){
  window.renderSingleView = renderSingleView;
  window.navigateSingle = navigateSingle;
  window.advanceSingleNext = advanceSingleNext;
}

// ===== Prefetch Helpers =====
function prefetchImage(url){
  if (!url) return;
  try { const img = new Image(); img.src = url; } catch(_){ }
}

function prefetchNextGroupImages(){
  try {
    const nextIndex = state.groupIndex + 1;
    if (!state.groups || nextIndex < 0 || nextIndex >= state.groups.length) return;
    const nextGrp = state.groups[nextIndex];
    const nextRepId = nextGrp && nextGrp.repId;
    if (!nextRepId) return;
    // Prefetch representative single image
    prefetchImage(`/api/file/${nextRepId}`);
    // Prefetch triplet members if available
    api(`/triplet/${nextRepId}`).then(trip => {
      if (trip && trip.items){
        prefetchImage(trip.items.target && trip.items.target.file);
        prefetchImage(trip.items.ref && trip.items.ref.file);
        prefetchImage(trip.items.diff && trip.items.diff.file);
      }
    }).catch(()=>{});
  } catch(_){ }
}


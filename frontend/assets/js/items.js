function tileTemplate(item){
  const predP = (item && item.pred_label && item.probs && item.probs[item.pred_label] != null) ? item.probs[item.pred_label] : item.max_proba;
  const predBadge = item.pred_label ? `<span class="badge pred">pred: ${item.pred_label}${(predP!=null)?` (${Number(predP).toFixed(2)})`:''}</span>` : '';
  const labelBadge = item.label ? `<span class="badge label">label: ${item.label}</span>` : '';
  // Threshold-based decision badge for clarity
  let decisionBadge = '';
  try {
    const thr = parseFloat(localStorage.getItem('al_pred_threshold') || '');
    const pos = localStorage.getItem('al_positive_class') || '';
    if (Number.isFinite(thr) && pos && item && item.probs && typeof item.probs === 'object'){
      const pPos = (pos in item.probs) ? Number(item.probs[pos]) : null;
      if (pPos != null && pPos >= 0 && pPos <= 1){
        const decision = (pPos >= thr) ? pos : `not ${pos}`;
        decisionBadge = `<span class="badge" style="background:#374151; color:#e5e7eb;">thr: ${decision} (${pPos.toFixed(2)} ≥ ${thr.toFixed(2)})</span>`;
      }
    }
  } catch(_){}
  const selected = state.selectedItems.has(item.id) ? 'selected' : '';
  const assignIdx = state.assignmentMap && state.assignmentMap.get(item.id);
  const assignClass = assignIdx===0 ? 'assign-blue' : assignIdx===1 ? 'assign-red' : assignIdx===2 ? 'assign-yellow' : '';
  const assignName = (typeof state.classes !== 'undefined' && Array.isArray(state.classes) && typeof assignIdx === 'number' && assignIdx >= 0 && state.classes[assignIdx]) ? (state.classes[assignIdx].name || '') : '';
  const zoomBtn = `<button class="zoom-btn" title="Zoom">🔍</button>`;
  return `
  <div class="tile ${selected} ${assignClass}" data-id="${item.id}">
    <input type="checkbox" class="tile-checkbox" data-id="${item.id}" ${selected ? 'checked' : ''}>
    ${zoomBtn}
    <img src="${item.thumb}" loading="lazy" />
    <div class="hoverbar"></div>
    <div class="meta">
      <div>${predBadge} ${labelBadge} ${decisionBadge}</div>
      <div></div>
    </div>
    <span class="assign-tag">${assignName}</span>
  </div>`;
}

async function loadItems(){
  state.loading = true;
  status('Loading...');
  try {
    const q = state.queue;
    const params = new URLSearchParams({
      queue: q, page: state.page, page_size: state.page_size, unlabeled_only: true
    });
    // Ensure we only request items with ready composite thumbnails to avoid placeholders
    params.set('only_ready', 'true');
    // Provide stable seed so 'all' queue order is deterministic across refreshes
    if (state.order_seed != null) params.set('seed', String(state.order_seed));
    
    // Only send probability parameters when in 'band' mode
    if (q === 'band') {
      params.set('prob_low', state.prob_low);
      params.set('prob_high', state.prob_high);
    }
    
    if (state.search_query) params.set('search', state.search_query);
    if (state.label_filter) params.set('label_filter', state.label_filter);
    // Use full mode so predictions (e.g., max_proba) are included in responses
    const data = await api(`/items?${params.toString()}`);
    // Backend already paginates by groups; do not slice here
    state.items = (data.items || []);
    
    // Handle empty results gracefully
    if (state.items.length === 0 && data.total === 0) {
      status('No items found - try ingesting some data first');
      return;
    }

    // Build groups for single mode navigation
    const keyToIds = new Map();
    const keyToRep = new Map();
    for (const it of state.items){
      const key = (typeof groupKeyFromPath === 'function') ? groupKeyFromPath(it.path) : it.path;
      if (!keyToIds.has(key)) keyToIds.set(key, []);
      keyToIds.get(key).push(it.id);
      // representative id: prefer target, else ref, else diff, else first
      const name = (it.path || '').toLowerCase();
      const score = name.endsWith('_target.fits') ? 3 : name.endsWith('_ref.fits') ? 2 : name.endsWith('_diff.fits') ? 1 : 0;
      const prev = keyToRep.get(key);
      if (!prev || score > prev.score){ keyToRep.set(key, { id: it.id, score }); }
    }
    state.groups = Array.from(keyToIds.keys()).map(k=>({ key: k, memberIds: keyToIds.get(k), repId: (keyToRep.get(k)||{}).id || keyToIds.get(k)[0] }));
    // Reset group index if out of bounds
    if (state.groupIndex >= state.groups.length) state.groupIndex = 0;
    // Build representative items for grouped grid view
    const idToItem = new Map(state.items.map(it=>[it.id, it]));
    const repToGroup = new Map(state.groups.map(g=>[g.repId, g]));
    const groupedTiles = state.groups.map(g=> idToItem.get(g.repId) || idToItem.get(g.memberIds[0])).filter(Boolean);
    const grid = el('#grid');
    grid.innerHTML = groupedTiles.map(tileTemplate).join('');
    applyTileSize(state.tile_size);
    grid.querySelectorAll('.tile').forEach(node=>{
      const id = parseInt(node.dataset.id);
      const group = repToGroup.get(id);
      const groupIds = group ? group.memberIds : [id];
      // Apply vibrant glow immediately if already assigned
      try {
        const initialIdx = state.assignmentMap && state.assignmentMap.get(id);
        if (typeof window.applyAssignmentClassToNode === 'function'){
          window.applyAssignmentClassToNode(node, (typeof initialIdx === 'number') ? initialIdx : -1);
        }
      } catch(_){ }
      const zoomBtn = node.querySelector('.zoom-btn');
      const checkbox = node.querySelector('.tile-checkbox');
      if (zoomBtn) zoomBtn.addEventListener('click', (e)=>{ e.stopPropagation(); showImageZoom(id); });
      if (checkbox) checkbox.addEventListener('change', (e)=>{
        if (e.target.checked){
          groupIds.forEach(gid => state.selectedItems.add(gid));
          node.classList.add('selected');
        } else {
          groupIds.forEach(gid => state.selectedItems.delete(gid));
          node.classList.remove('selected');
        }
        updateBatchControls();
      });
      // Click cycles assignment: cycles through available classes (N) then none
      node.addEventListener('click', (e) => {
        if (e.button !== 0) return;
        cycleAssignment(id, node);
        // applyAssignmentClassToNode already updates tag text/color/display
      });
    });
    // Use server-reported total groups
    state.total_items = data.total || state.groups.length;
    state.total_pages = Math.ceil((state.total_items || 0) / state.page_size);
    state.page = data.page || 1;
    updatePagination();
    setupDragSelection();
    await loadStats();
    // If in single mode, (re)render that view and ensure bounds
    if (state.mode === 'single'){
      // Re-align current group by stable key if available
      if (state.currentGroupKey){
        const idx = state.groups.findIndex(g => g.key === state.currentGroupKey);
        if (idx >= 0) state.groupIndex = idx;
      }
      if (state.groupIndex >= state.groups.length) state.groupIndex = 0;
      // Only re-render if not currently navigating to prevent double renders
      if (!state.navigating && typeof renderSingleView === 'function') {
        await renderSingleView();
      }
    }
    // Skip automatic background triplet verification to avoid unintended dataset scans
    // status(`📋 ${data.total} items in queue`);
  } catch (e) {
    status(`⚠️ Error loading items: ${e.message}`);
  } finally {
    state.loading = false;
  }
}

async function labelItems(ids, label=null, unsure=false, skip=false){
  try {
    await api('/label', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({item_ids: ids, label, unsure, skip, user: null})
    });
    if (skip) updateSessionStats(ids.length, 'skipped');
    else if (unsure) updateSessionStats(ids.length, 'unsure');
    else if (label) updateSessionStats(ids.length, 'labeled');
    // In single (one-by-one) mode, refresh in the background to keep UI snappy
    if (state && state.mode === 'single') { setTimeout(()=>{ loadItems(); }, 0); }
    else { await loadItems(); }
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
      if (skip) updateSessionStats(result.labeled_count, 'skipped');
      else if (unsure) updateSessionStats(result.labeled_count, 'unsure');
      else if (label) updateSessionStats(result.labeled_count, 'labeled');
      status(`Batch labeled ${result.labeled_count} items`);
      state.selectedItems.clear();
      // In single (one-by-one) mode, refresh in the background to avoid delay before advancing
      if (state && state.mode === 'single') { setTimeout(()=>{ loadItems(); }, 0); }
      else { await loadItems(); }
    } else {
      status(`Batch labeling failed: ${result.msg}`);
    }
  } catch (e) {
    status(`Error batch labeling: ${e.message}`);
  }
}

async function retrain(){
  status('🚀 Training model...');
  const r = await api('/train', {method:'POST'});
  status(r.ok ? `✅ Trained on ${r.labeled} labeled; predicted ${r.predicted}` : `⚠️ ${r.msg}`);
  await loadItems();
}

// Note: TensorFlow training is now handled by the training dashboard (training.js)


async function showSimilar(id){
  const data = await api(`/similar/${id}`);
  const grid = el('#grid');
  grid.innerHTML = data.neighbors.map(tileTemplate).join('');
  grid.querySelectorAll('.tile').forEach(node=>{
    const id = parseInt(node.dataset.id);
    const zoomBtn = node.querySelector('.zoom-btn');
    if (zoomBtn) zoomBtn.addEventListener('click', (e)=>{ e.stopPropagation(); showImageZoom(id); });
    // Click cycles assignment: blue -> red -> yellow -> none
    node.addEventListener('click', (e) => {
      if (e.button !== 0) return;
      cycleAssignment(id, node);
    });
  });
  status(`Similar items`);
}


// Verify triplets across full dataset (background)
async function verifyTripletsAcrossDataset(){
  try {
    let page = 1;
    const pageSize = 500;
    let total = Infinity;
    const seen = new Map(); // key -> {target,ref,diff,ids:Set}
    while ((page - 1) * pageSize < total){
      const params = new URLSearchParams({
        queue: 'all', page: page, page_size: pageSize,
        prob_low: state.prob_low, prob_high: state.prob_high, unlabeled_only: false
      });
      const data = await api(`/items?${params.toString()}`);
      total = data.total || 0;
      const arr = data.items || [];
      for (const it of arr){
        const key = (typeof groupKeyFromPath === 'function') ? groupKeyFromPath(it.path) : it.path;
        let roles = seen.get(key);
        if (!roles){ roles = {target:null, ref:null, diff:null, ids: new Set()}; seen.set(key, roles); }
        const name = (it.path || '').toLowerCase();
        if (name.endsWith('_target.fits')) roles.target = it.id;
        else if (name.endsWith('_ref.fits')) roles.ref = it.id;
        else if (name.endsWith('_diff.fits')) roles.diff = it.id;
        roles.ids.add(it.id);
      }
      const pageCount = data.page_size || pageSize;
      if (page * pageCount >= total) break;
      page++;
      // avoid hammering server
      await new Promise(r=>setTimeout(r, 10));
    }
    let complete = 0;
    for (const [, roles] of seen){
      if (roles.target && roles.ref && roles.diff && roles.ids.size === 3) complete++;
    }
    status(`✅ Found ${complete} complete triplets`);
    state.tripletCheckDone = true;
  } catch(e){
    console.warn('Triplet verification failed', e);
  }
}

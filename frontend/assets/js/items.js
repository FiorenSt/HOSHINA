function tileTemplate(item){
  // Prefer positive-class probability for decision when available; fallback to model's argmax
  let posName = '';
  let thr = null;
  try { posName = localStorage.getItem('al_positive_class') || ''; } catch(_){ posName = ''; }
  try { const t = parseFloat(localStorage.getItem('al_pred_threshold') || ''); if (Number.isFinite(t)) thr = t; } catch(_){ }
  let posProb = null;
  try {
    if (posName && item && item.probs && Object.prototype.hasOwnProperty.call(item.probs, posName)){
      const v = Number(item.probs[posName]);
      if (Number.isFinite(v)) posProb = v;
    }
  } catch(_){ }
  // Determine label and probability to display
  let decisionLabel = item && item.pred_label ? item.pred_label : '';
  let decisionProb = (posProb != null)
    ? posProb
    : ((item && item.pred_label && item.probs && item.probs[item.pred_label] != null)
        ? Number(item.probs[item.pred_label])
        : (typeof item.max_proba === 'number' ? Number(item.max_proba) : null));
  if (posName && posProb != null && thr != null){
    // In binary case, flip to the other class name if below threshold
    let negative = `not ${posName}`;
    try {
      const names = (Array.isArray(state.classes) ? state.classes.map(c=>c.name) : []);
      if (names.length === 2 && names.includes(posName)){
        const other = names.find(n => n !== posName);
        if (other) negative = other;
      }
    } catch(_){ }
    decisionLabel = (posProb >= thr) ? posName : negative;
  }
  const predBadge = decisionLabel ? `<span class="badge pred">pred: ${decisionLabel}${(decisionProb!=null)?` (${Number(decisionProb).toFixed(2)})`:''}</span>` : '';
  const labelBadge = item.label ? `<span class="badge label">label: ${item.label}</span>` : '';
  // Do not show extra badges to avoid confusion
  let decisionBadge = '';
  let posProbBadge = '';
  try { /* no-op */ } catch(_){ }
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
      <div></div>
      <div>${predBadge} ${labelBadge}</div>
    </div>
    <span class="assign-tag">${assignName}</span>
  </div>`;
}

async function loadItems(){
  // Cancel any in-flight request
  try { if (state.loadAbortController) state.loadAbortController.abort(); } catch(_){ }
  const myToken = ++state.loadToken;
  state.loadAbortController = (typeof AbortController !== 'undefined') ? new AbortController() : null;
  state.loading = true;
  status('Loading...');
  try {
    const q = state.queue;
    const params = new URLSearchParams({
      queue: q, page: state.page, page_size: state.page_size, unlabeled_only: true
    });
    // On-the-fly thumbnails: no readiness gating needed
    // Provide stable seed so 'all' queue order is deterministic across refreshes
    if (state.order_seed != null) params.set('seed', String(state.order_seed));
    // Global predicted sorting handled server-side when enabled
    if (state.sort_pred === 'asc' || state.sort_pred === 'desc'){
      params.set('sort_pred', state.sort_pred);
      try { const pc = (localStorage.getItem('al_positive_class') || '').trim(); if (pc) params.set('pos_class', pc); } catch(_){ }
    }
    
    // Only send probability parameters when in 'band' mode
    if (q === 'band') {
      params.set('prob_low', state.prob_low);
      params.set('prob_high', state.prob_high);
    }
    // 'certain' mode removed
    
    if (state.search_query) params.set('search', state.search_query);
    if (state.label_filter) params.set('label_filter', state.label_filter);
    // Use full mode so predictions (e.g., max_proba) are included in responses
    const url = `/items?${params.toString()}`;
    // Use fetch directly to support AbortController; fallback to api()
    let data;
    if (state.loadAbortController && typeof fetch === 'function'){
      const res = await fetch(API_BASE + url, { signal: state.loadAbortController.signal });
      if (!res.ok) throw new Error(`API ${url} failed: ${res.status}`);
      data = await res.json();
    } else {
      data = await api(url);
    }
    // Discard if a newer load has started
    if (myToken !== state.loadToken) return;
    // Backend already paginates by groups; do not slice here
    state.items = (data.items || []);
    
    // Handle empty results gracefully
    if (state.items.length === 0 && data.total === 0) {
      status('No items found - try ingesting some data first');
      return;
    }

    // Build groups for GRID view rendering
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

    // When server-side sort is active, keep order as received to preserve global ranking
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
    // Note: Single view is now fully decoupled and uses its own buffers
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
    // Inline updates are suppressed during atomic apply; otherwise update small ops
    if (!state.suppressInlineLabelUpdates){
      try { if (Array.isArray(ids) && ids.length){ updateTilesAfterLabel(ids, label, unsure, skip); } } catch(_){ }
      setTimeout(()=>{ loadItems(); }, 0);
    }
  } catch (e) {
    status(`Error labeling items: ${e.message}`);
  }
}

async function batchLabelItems(ids, label=null, unsure=false, skip=false, doReload=true){
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
      if (!state.suppressInlineLabelUpdates){
        try { updateTilesAfterLabel(ids, label, unsure, skip); } catch(_){ }
        if (doReload){ setTimeout(()=>{ loadItems(); }, 0); }
      }
    } else {
      status(`Batch labeling failed: ${result.msg}`);
    }
  } catch (e) {
    status(`Error batch labeling: ${e.message}`);
  }
}

// retrain removed: training handled via TensorFlow dashboard

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


// ===== Single View Loader (decoupled from grid) =====
const SINGLE_CHUNK_SIZE = 120;

function buildGroupsFromItems(items){
  const keyToIds = new Map();
  const keyToRep = new Map();
  for (const it of items){
    const key = (typeof groupKeyFromPath === 'function') ? groupKeyFromPath(it.path) : it.path;
    if (!keyToIds.has(key)) keyToIds.set(key, []);
    keyToIds.get(key).push(it.id);
    const name = (it.path || '').toLowerCase();
    const score = name.endsWith('_target.fits') ? 3 : name.endsWith('_ref.fits') ? 2 : name.endsWith('_diff.fits') ? 1 : 0;
    const prev = keyToRep.get(key);
    if (!prev || score > prev.score){ keyToRep.set(key, { id: it.id, score }); }
  }
  return Array.from(keyToIds.keys()).map(k=>({ key: k, memberIds: keyToIds.get(k), repId: (keyToRep.get(k)||{}).id || keyToIds.get(k)[0] }));
}

function resetSingleBuffer(){
  state.single_items = [];
  state.single_groups = [];
  state.single_next_page = 1;
  state.single_total_pages = null;
  state.single_total_groups = null;
  state.single_remaining_groups = null;
}

async function singleFetchNextPage(){
  if (state.single_loading) return false;
  if (state.single_total_pages != null && state.single_next_page > state.single_total_pages) return false;
  state.single_loading = true;
  try {
    const q = state.queue;
    const params = new URLSearchParams({
      queue: q, page: state.single_next_page, page_size: String(SINGLE_CHUNK_SIZE), unlabeled_only: true
    });
    if (state.order_seed != null) params.set('seed', String(state.order_seed));
    if (state.sort_pred === 'asc' || state.sort_pred === 'desc'){
      params.set('sort_pred', state.sort_pred);
      try { const pc = (localStorage.getItem('al_positive_class') || '').trim(); if (pc) params.set('pos_class', pc); } catch(_){ }
    }
    if (q === 'band') { params.set('prob_low', state.prob_low); params.set('prob_high', state.prob_high); }
    if (state.search_query) params.set('search', state.search_query);
    if (state.label_filter) params.set('label_filter', state.label_filter);
    const data = await api(`/items?${params.toString()}`);
    const newItems = (data.items || []);
    if (newItems.length === 0 && (data.total || 0) === 0 && state.single_next_page === 1){
      status('No items found - try ingesting some data first');
    }
    // Append unique items
    const existingIds = new Set(state.single_items.map(it=>it.id));
    const appendItems = newItems.filter(it => !existingIds.has(it.id));
    state.single_items = state.single_items.concat(appendItems);
    // Build groups from appended items and append unique groups
    const existingKeys = new Set(state.single_groups.map(g=>g.key));
    const newGroups = buildGroupsFromItems(appendItems).filter(g => !existingKeys.has(g.key));
    state.single_groups = state.single_groups.concat(newGroups);
    // Update paging trackers
    const total = data.total || 0;
    const totalPages = Math.max(1, Math.ceil(total / SINGLE_CHUNK_SIZE));
    state.single_total_pages = totalPages;
    state.single_total_groups = total;
  // Initialize remaining counter only once per session/reset
  if (state.single_remaining_groups == null) state.single_remaining_groups = total;
    state.single_next_page += 1;
    return newGroups.length > 0;
  } finally {
    state.single_loading = false;
  }
}

async function ensureSingleHasIndex(targetIndex){
  while (targetIndex >= state.single_groups.length){
    const advanced = await singleFetchNextPage();
    if (!advanced) break;
  }
}

if (typeof window !== 'undefined'){
  window.resetSingleBuffer = resetSingleBuffer;
  window.singleFetchNextPage = singleFetchNextPage;
  window.ensureSingleHasIndex = ensureSingleHasIndex;
}

// ===== Single View Local Updates After Labeling =====
function updateSingleBuffersAfterLabel(ids, label=null, unsure=false, skip=false){
  try {
    if (!Array.isArray(ids) || ids.length === 0) return;
    const idSet = new Set(ids.map(Number));
    // Determine if current group transitioned from unlabeled to labeled
    let groupNewlyLabeled = false;
    try {
      // Find the group that contains any of the ids
      const grp = (state.single_groups || []).find(g => Array.isArray(g.memberIds) && g.memberIds.some(id => idSet.has(Number(id))));
      if (grp){
        // A group is considered labeled if all members have a non-empty label or are marked unsure/skip
        const beforeLabeled = grp._labeled === true;
        // Update items first so we can recompute labeled status
        // (fall through)
      }
    } catch(_){ }
    // Update single buffer items
    if (Array.isArray(state.single_items) && state.single_items.length){
      state.single_items.forEach(it => {
        if (idSet.has(Number(it.id))){
          if (label) it.label = label;
          // Track unsure/skip flags if needed later
          if (unsure) it.unsure = true;
          if (skip) it.skipped = true;
        }
      });
    }
    // Also reflect in current page items for grid consistency
    if (Array.isArray(state.items) && state.items.length){
      state.items.forEach(it => {
        if (idSet.has(Number(it.id))){
          if (label) it.label = label;
          if (unsure) it.unsure = true;
          if (skip) it.skipped = true;
        }
      });
    }
    // Recompute labeled status for the affected group and decrement remaining if it transitioned
    try {
      const grp = (state.single_groups || []).find(g => Array.isArray(g.memberIds) && g.memberIds.some(id => idSet.has(Number(id))));
      if (grp){
        const members = (state.single_items || []).filter(it => Array.isArray(grp.memberIds) && grp.memberIds.includes(Number(it.id)));
        const isNowLabeled = members.length > 0 && members.every(it => (it.label && String(it.label).length > 0) || it.unsure === true || it.skipped === true);
        const wasLabeled = grp._labeled === true;
        if (!wasLabeled && isNowLabeled){
          grp._labeled = true;
          if (typeof state.single_remaining_groups === 'number' && state.single_remaining_groups > 0){
            state.single_remaining_groups -= 1;
          }
          try { if (typeof updatePagination === 'function') updatePagination(); } catch(_){ }
        }
      }
    } catch(_){ }
  } catch(_){ }
}

if (typeof window !== 'undefined'){
  window.updateSingleBuffersAfterLabel = updateSingleBuffersAfterLabel;
}


// Fast, in-place grid updates after labeling to avoid full-page re-render latency
function updateTilesAfterLabel(ids, label=null, unsure=false, skip=false){
  try {
    if (!Array.isArray(ids) || ids.length === 0) return;
    const repIds = new Set();
    ids.forEach((rawId) => {
      const id = Number(rawId);
      // If the representative tile exists directly, update that
      const direct = document.querySelector(`.tile[data-id="${id}"]`);
      if (direct) { repIds.add(id); return; }
      // Otherwise, try to find the group's representative
      try {
        if (Array.isArray(state.groups)){
          const grp = state.groups.find(g => Array.isArray(g.memberIds) && g.memberIds.includes(id));
          if (grp && grp.repId != null) repIds.add(grp.repId);
        }
      } catch(_){ }
    });
    repIds.forEach((rid) => {
      const node = document.querySelector(`.tile[data-id="${rid}"]`);
      if (!node) return;
      // Clear selection and disable interactions
      node.classList.remove('selected');
      const checkbox = node.querySelector('.tile-checkbox');
      if (checkbox){ checkbox.checked = false; checkbox.disabled = true; }
      // Update visible label badge if we know the label
      try {
        if (label){
          // Prefer the badges row (right side) so label appears consistently even without predictions
          const metaRow = node.querySelector('.meta div:last-child') || node.querySelector('.meta div');
          const existing = node.querySelector('.badge.label');
          const span = existing || document.createElement('span');
          span.className = 'badge label';
          span.textContent = `label: ${label}`;
          if (!existing && metaRow) metaRow.appendChild(span);
        }
      } catch(_){ }
      // Remove assignment glow and apply a quick "done" effect
      try { if (typeof window.applyAssignmentClassToNode === 'function'){ window.applyAssignmentClassToNode(node, -1); } } catch(_){ }
      node.style.opacity = '0.5';
      node.style.filter = 'grayscale(0.6)';
      node.style.pointerEvents = 'none';
    });
  } catch(_){ }
}

// Remove representative tiles for assigned items immediately after Apply Assignments
function removeTilesAfterApply(ids){
  try {
    if (!Array.isArray(ids) || !ids.length) return;
    const repIds = new Set();
    ids.forEach((rawId)=>{
      const id = Number(rawId);
      const direct = document.querySelector(`.tile[data-id="${id}"]`);
      if (direct) { repIds.add(id); return; }
      try {
        if (Array.isArray(state.groups)){
          const grp = state.groups.find(g => Array.isArray(g.memberIds) && g.memberIds.includes(id));
          if (grp && grp.repId != null) repIds.add(grp.repId);
        }
      } catch(_){ }
    });
    repIds.forEach((rid)=>{
      const node = document.querySelector(`.tile[data-id="${rid}"]`);
      if (!node) return;
      node.remove();
    });
  } catch(_){ }
}

// Replace only the selected tiles with new content in one atomic step
async function inPlaceRefreshAfterAssignments(assignedIds){
  try {
    const grid = el('#grid'); if (!grid) return;
    const currentTiles = Array.from(grid.querySelectorAll('.tile'));
    const currentRepIds = currentTiles.map(n => Number(n.dataset.id));
    // Compute representative IDs to replace based on assigned IDs
    const replaceRepIds = [];
    (assignedIds || []).forEach(rawId => {
      const id = Number(rawId);
      if (currentRepIds.includes(id)) { replaceRepIds.push(id); return; }
      try {
        const grp = (state.groups || []).find(g => Array.isArray(g.memberIds) && g.memberIds.includes(id));
        if (grp && grp.repId != null && currentRepIds.includes(grp.repId)) replaceRepIds.push(grp.repId);
      } catch(_){ }
    });
    if (replaceRepIds.length === 0) return;

    // Fetch a refreshed page snapshot (same params as loadItems)
    const q = state.queue;
    const params = new URLSearchParams({
      queue: q, page: state.page, page_size: state.page_size, unlabeled_only: true
    });
    // On-the-fly thumbnails: no readiness gating needed
    if (state.order_seed != null) params.set('seed', String(state.order_seed));
    if (q === 'band') { params.set('prob_low', state.prob_low); params.set('prob_high', state.prob_high); }
    if (state.sort_pred === 'asc' || state.sort_pred === 'desc'){
      params.set('sort_pred', state.sort_pred);
      try { const pc = (localStorage.getItem('al_positive_class') || '').trim(); if (pc) params.set('pos_class', pc); } catch(_){ }
    }
    if (state.search_query) params.set('search', state.search_query);
    if (state.label_filter) params.set('label_filter', state.label_filter);
    const data = await api(`/items?${params.toString()}`);

    // Rebuild grouping for the new snapshot (same as in loadItems)
    const newItems = (data.items || []);
    const keyToIds = new Map();
    const keyToRep = new Map();
    for (const it of newItems){
      const key = (typeof groupKeyFromPath === 'function') ? groupKeyFromPath(it.path) : it.path;
      if (!keyToIds.has(key)) keyToIds.set(key, []);
      keyToIds.get(key).push(it.id);
      const name = (it.path || '').toLowerCase();
      const score = name.endsWith('_target.fits') ? 3 : name.endsWith('_ref.fits') ? 2 : name.endsWith('_diff.fits') ? 1 : 0;
      const prev = keyToRep.get(key);
      if (!prev || score > prev.score){ keyToRep.set(key, { id: it.id, score }); }
    }
    const newGroups = Array.from(keyToIds.keys()).map(k=>({ key: k, memberIds: keyToIds.get(k), repId: (keyToRep.get(k)||{}).id || keyToIds.get(k)[0] }));
    const idToItem = new Map(newItems.map(it=>[it.id, it]));
    const repToGroup = new Map(newGroups.map(g=>[g.repId, g]));
    const newRepIds = newGroups.map(g => g.repId);

    // If in explicit sorted mode, rebuild the page to reflect shifted order and promotion
    const sortedMode = (state.sort_pred === 'asc' || state.sort_pred === 'desc');
    if (sortedMode){
      const groupedTiles = newGroups
        .map(g=> idToItem.get(g.repId) || idToItem.get((g.memberIds||[])[0]))
        .filter(Boolean);
      grid.innerHTML = groupedTiles.map(tileTemplate).join('');
      applyTileSize(state.tile_size);
      grid.querySelectorAll('.tile').forEach(node=>{
        const rid = parseInt(node.dataset.id);
        const group = repToGroup.get(rid);
        const groupIds = group ? group.memberIds : [rid];
        try {
          const initialIdx = state.assignmentMap && state.assignmentMap.get(rid);
          if (typeof window.applyAssignmentClassToNode === 'function'){
            window.applyAssignmentClassToNode(node, (typeof initialIdx === 'number') ? initialIdx : -1);
          }
        } catch(_){ }
        const zoomBtn = node.querySelector('.zoom-btn');
        const checkbox = node.querySelector('.tile-checkbox');
        if (zoomBtn) zoomBtn.addEventListener('click', (e)=>{ e.stopPropagation(); showImageZoom(rid); });
        if (checkbox) checkbox.addEventListener('change', (e)=>{
          if (e.target.checked){ groupIds.forEach(gid => state.selectedItems.add(gid)); node.classList.add('selected'); }
          else { groupIds.forEach(gid => state.selectedItems.delete(gid)); node.classList.remove('selected'); }
          updateBatchControls();
        });
        node.addEventListener('click', (e) => {
          if (e.button !== 0) return; cycleAssignment(rid, node);
        });
      });
    } else {
      // Non-sorted mode: replace only the assigned positions in place without shifting others
      // Determine which IDs are new in the refreshed page snapshot (to use as replacements)
      const addedRepIds = newRepIds.filter(id => !currentRepIds.includes(id));
      if (addedRepIds.length === 0) {
        // Nothing new to fill with; do a minimal full refresh as fallback
        await loadItems(); return;
      }

      // Find target positions to replace (indices of tiles whose repIds were assigned)
      const targetIndices = [];
      currentRepIds.forEach((rid, idx) => { if (replaceRepIds.includes(rid)) targetIndices.push(idx); });

      // Build replacement nodes for as many targets as we have new reps
      const replacements = [];
      for (let i=0; i<targetIndices.length && i<addedRepIds.length; i++){
        const rid = addedRepIds[i];
        const item = idToItem.get(rid);
        if (!item) continue;
        const html = tileTemplate(item);
        const wrapper = document.createElement('div');
        wrapper.innerHTML = html.trim();
        const node = wrapper.firstElementChild;
        // Wire listeners similarly to loadItems
        const group = repToGroup.get(rid);
        const groupIds = group ? group.memberIds : [rid];
        try {
          const initialIdx = state.assignmentMap && state.assignmentMap.get(rid);
          if (typeof window.applyAssignmentClassToNode === 'function'){
            window.applyAssignmentClassToNode(node, (typeof initialIdx === 'number') ? initialIdx : -1);
          }
        } catch(_){ }
        const zoomBtn = node.querySelector('.zoom-btn');
        const checkbox = node.querySelector('.tile-checkbox');
        if (zoomBtn) zoomBtn.addEventListener('click', (e)=>{ e.stopPropagation(); showImageZoom(rid); });
        if (checkbox) checkbox.addEventListener('change', (e)=>{
          if (e.target.checked){ groupIds.forEach(gid => state.selectedItems.add(gid)); node.classList.add('selected'); }
          else { groupIds.forEach(gid => state.selectedItems.delete(gid)); node.classList.remove('selected'); }
          updateBatchControls();
        });
        node.addEventListener('click', (e) => {
          if (e.button !== 0) return; cycleAssignment(rid, node);
        });
        replacements.push(node);
      }

      // Apply all replacements in place without shifting other tiles
      for (let i=0; i<replacements.length; i++){
        const idx = targetIndices[i];
        const oldNode = currentTiles[idx];
        if (!oldNode || !oldNode.parentNode) continue;
        oldNode.parentNode.replaceChild(replacements[i], oldNode);
      }
    }

    // Update state to reflect the refreshed snapshot
    state.items = newItems;
    state.groups = newGroups;
    state.total_items = data.total || newGroups.length;
    state.total_pages = Math.ceil((state.total_items || 0) / state.page_size);
    state.page = data.page || state.page;
    try { updatePagination(); } catch(_){ }
    try { applyTileSize(state.tile_size); } catch(_){ }
    try { await loadStats(); } catch(_){ }
  } catch(e){
    // Fallback to full refresh on any error
    try { await loadItems(); } catch(_){ }
  }
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

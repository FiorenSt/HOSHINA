function setupUI(){
  const addListener = (sel, event, handler) => {
    const elem = el(sel);
    if (elem) elem.addEventListener(event, handler);
    else console.warn(`Element ${sel} not found`);
  };
  
  addListener('#theme-toggle', 'click', toggleTheme);
  
  // Category dropdown functionality
  setupCategoryDropdowns();
  // Smart Suggestions buttons -> update main grid
  const setQueueAndRefresh = async (queue) => {
    state.queue = queue; state.page = 1;
    // Visual feedback during suggestion load
    const grid = el('#grid');
    if (grid) {
      grid.classList.add('loading');
      grid.setAttribute('aria-busy', 'true');
    }
    status(`🎯 Loading ${queue === 'band' ? 'borderline' : queue} suggestions...`);
    await loadItems();
    if (grid) {
      grid.classList.remove('loading');
      grid.removeAttribute('aria-busy');
    }
    // Toggle probability band controls visibility when in band mode
    const bandGroup = el('#header-probability-band');
    if (bandGroup) bandGroup.style.display = (queue === 'band') ? 'flex' : 'none';
  };
  addListener('#smart-uncertain', 'click', async ()=>{ await setQueueAndRefresh('uncertain'); });
  addListener('#smart-diverse', 'click', async ()=>{ await setQueueAndRefresh('diverse'); });
  addListener('#smart-odd', 'click', async ()=>{ await setQueueAndRefresh('odd'); });
  addListener('#smart-all', 'click', async ()=>{ await setQueueAndRefresh('all'); });
  addListener('#smart-borderline', 'click', async ()=>{
    // Map Borderline to a probability band around 0.5
    const lowInp = el('#low'); const highInp = el('#high');
    const low = 0.4; const high = 0.6;
    if (lowInp) lowInp.value = String(low);
    if (highInp) highInp.value = String(high);
    state.prob_low = low; state.prob_high = high;
    await setQueueAndRefresh('band');
  });
  // Apply probability band explicitly; ensure queue=band
  addListener('#pull', 'click', async ()=>{ 
    state.prob_low = parseFloat(el('#low').value); 
    state.prob_high = parseFloat(el('#high').value); 
    await setQueueAndRefresh('band'); 
  });
  addListener('#retrain', 'click', retrain);
  addListener('#train-tf', 'click', () => {
    if (window.trainingDashboard) {
      window.trainingDashboard.show();
    } else {
      console.error('Training dashboard not available');
    }
  });
  // Remove stale export button listener if element not present
  // (kept no-op to avoid console noise)
  addListener('#classes', 'click', openClasses);
  // Refresh class buttons when classes or colors change
  document.addEventListener('al:classes-updated', ()=>{
    if (typeof syncClassColorsWithClasses === 'function') syncClassColorsWithClasses();
    if (typeof updateBatchClassButtons === 'function') updateBatchClassButtons();
    if (typeof updateSingleClassButtons === 'function') updateSingleClassButtons();
    if (typeof refreshAssignHint === 'function') refreshAssignHint();
  });
  addListener('#map', 'click', openMap);
  addListener('#stats', 'click', openStats);
  // Removed suggestions modal; Smart Suggestions now updates main grid directly
  addListener('#bulk-ops', 'click', openBulkOps);
  addListener('#import-folder-labeled', 'click', () => {
    openModal('#importFolderModal');
    const st = el('#ifl-status'); if (st) { st.style.display = 'none'; st.textContent = ''; }
  });
  // Capture-phase fallback for overlay dropdown clicks
  try {
    document.addEventListener('click', (e) => {
      if (e && e.target && e.target.closest && e.target.closest('#import-folder-labeled')){
        openModal('#importFolderModal');
        const st = el('#ifl-status'); if (st) { st.style.display = 'none'; st.textContent = ''; }
      }
    }, true);
  } catch(_){}
  addListener('#ifl-close', 'click', () => closeModal('#importFolderModal'));
  addListener('#ifl-run', 'click', async () => {
    const folder = el('#ifl-folder')?.value.trim();
    const cls = el('#ifl-class')?.value.trim();
    const trip = !!el('#ifl-triplets')?.checked;
    const thumbs = false;
    const requireAll = !!el('#ifl-require-all')?.checked;
    const maxGroupsVal = (el('#ifl-max-groups')?.value || '').trim();
    const maxGroups = maxGroupsVal ? parseInt(maxGroupsVal, 10) : null;
    const st = el('#ifl-status');
    if (!folder) { alert('Please enter a folder path'); return; }
    if (!cls) { alert('Please enter a class name'); return; }
    try {
      // Client-side guard: do not attempt folder import while ingest is running
      try {
        const ist = await api('/ingest/status');
        if (ist && ist.ok && ist.running) {
          if (st) { st.className = 'import-status error'; st.textContent = '❌ Another ingest is running. Please wait for it to finish or cancel it before importing.'; st.style.display = 'block'; }
          return;
        }
      } catch(_){ }
      if (st) { st.className = 'import-status loading'; st.textContent = 'Importing...'; st.style.display = 'block'; }
      const res = await fetch('/api/import-folder-labeled', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder_path: folder, class_name: cls, create_triplets: trip, make_thumbs: false, group_require_all: requireAll, max_groups: maxGroups })
      });
      const data = await res.json();
      if (data.ok) {
        const labeledGroups = (typeof data.labeled_groups === 'number') ? data.labeled_groups : (typeof data.processed_groups === 'number' ? data.processed_groups : null);
        const labeledDisplay = (labeledGroups !== null) ? labeledGroups : data.labeled;
        if (st) { st.className = 'import-status success'; st.textContent = `✅ Ingested ${data.ingested}, labeled ${labeledDisplay}, groups ${data.triplet_groups}`; }
        await loadItems();
        try { await refreshStatsDisplay(); } catch(_){ }
      } else {
        if (st) { st.className = 'import-status error'; st.textContent = `❌ ${data.msg || 'Import failed'}`; st.style.display = 'block'; }
      }
    } catch (e) {
      if (st) { st.className = 'import-status error'; st.textContent = `❌ ${e.message}`; st.style.display = 'block'; }
    }
  });
  // Auto-load grouping config on page load
  const loadGroupingConfig = async () => {
    try {
      const g = await api('/grouping');
      const ta = el('#grouping-json'); 
      if (ta) ta.value = JSON.stringify({ roles: g.roles, suffix_map: g.suffix_map, require_all_roles: g.require_all_roles, unit_size: g.unit_size }, null, 2);
    } catch (e) {
      console.warn('Failed to load grouping config:', e);
    }
  };
  
  // Auto-save grouping config when user modifies it
  const autoSaveGroupingConfig = async () => {
    const ta = el('#grouping-json'); if (!ta) return;
    let payload;
    try { payload = JSON.parse(ta.value); } catch(e) { return; } // Invalid JSON, don't save
    try {
      await api('/grouping', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    } catch(e){ console.warn('Failed to auto-save grouping config:', e); }
  };
  
  // Load config on page load
  loadGroupingConfig();
  
  // Auto-save on textarea change (debounced)
  const ta = el('#grouping-json');
  if (ta) {
    let saveTimeout;
    ta.addEventListener('input', () => {
      clearTimeout(saveTimeout);
      saveTimeout = setTimeout(autoSaveGroupingConfig, 1000); // Save 1 second after user stops typing
    });
  }
  // Start Fresh handler
  addListener('#start-fresh', 'click', async () => {
    if (!confirm('This will completely wipe all data, labels, predictions, embeddings, and caches.\n\nYou will need to ingest data again after this.\n\nProceed?')) return;
    
    const startBtn = el('#start-fresh');
    const originalText = startBtn ? startBtn.textContent : '';
    let progressInterval;
    
    try {
      // Disable button and show progress
      if (startBtn) {
        startBtn.disabled = true;
        startBtn.textContent = '🧹 Wiping...';
        startBtn.classList.add('loading');
      }
      
      // Show detailed progress messages with steps
      const steps = [
        '🗑️ Starting complete data wipe...',
        '📊 Counting database records...',
        '🗂️ Deleting database tables...',
        '🖼️ Clearing thumbnails and caches...',
        '✅ Finalizing cleanup...'
      ];
      
      let currentStep = 0;
      status(steps[currentStep]);
      
      // Show progress with a timer (visual feedback while waiting)
      progressInterval = setInterval(() => {
        currentStep = Math.min(currentStep + 1, steps.length - 1);
        const progress = Math.round((currentStep / (steps.length - 1)) * 100);
        status(`${steps[currentStep]} (${progress}%)`);
      }, 2000);
      
      // Add a timeout to prevent hanging (increased to 60 seconds for large datasets)
      const timeoutPromise = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Wipe operation timed out after 60 seconds')), 60000)
      );
      
      const resetPromise = api('/dataset/reset', { 
        method:'POST', 
        headers:{'Content-Type':'application/json'}, 
        body: JSON.stringify({ 
          wipe_items: true, 
          wipe_embeddings: true, 
          wipe_predictions: true, 
          wipe_labels: true, 
          wipe_umap: true, 
          wipe_caches: true, 
          wipe_classes: false, 
          recreate_db: true,
          reingest: false 
        }) 
      });
      
      // Race between reset and timeout
      const r = await Promise.race([resetPromise, timeoutPromise]);
      
      if (!r.ok) { 
        throw new Error(r.msg || 'Reset failed'); 
      }
      
      // Clear progress timer
      clearInterval(progressInterval);
      status('✅ All data wiped successfully - ready for fresh start');
      
      // Refresh the UI
      await loadItems();
      try { await refreshStatsDisplay(); } catch(_){ }
      
      // Force full page reload so the cleared state is obvious everywhere
      try { setTimeout(() => { window.location.reload(); }, 50); } catch(_){ }
      
    } catch(e){
      // Clear progress timer on error
      clearInterval(progressInterval);
      const errorMsg = e.message || 'Unknown error';
      status('❌ Wipe failed: ' + errorMsg);
      console.error('Reset error:', e);
      
      // Provide helpful error message
      if (errorMsg.includes('timeout')) {
        alert('Wipe operation timed out. This can happen with very large datasets.\n\nTry:\n1. Restart the server\n2. Or try again (the operation may have partially completed)');
      } else {
        alert('Wipe failed: ' + errorMsg);
      }
    } finally {
      // Clear progress timer and re-enable button
      if (progressInterval) {
        clearInterval(progressInterval);
      }
      if (startBtn) {
        startBtn.disabled = false;
        startBtn.textContent = originalText;
        startBtn.classList.remove('loading');
      }
    }
  });
  addListener('#help', 'click', showHelp);
  // Single-mode controls (buttons may not exist; guard without logging)
  const singleUnsureBtn = el('#single-unsure');
  if (singleUnsureBtn) singleUnsureBtn.addEventListener('click', () => {
    if (state.mode !== 'single') return;
    const grp = state.groups[state.groupIndex];
    if (!grp) return;
    batchLabelItems(grp.memberIds, null, true, false).catch(()=>{});
    if (typeof advanceSingleNext === 'function') advanceSingleNext();
  });
  const singleSkipBtn = el('#single-skip');
  if (singleSkipBtn) singleSkipBtn.addEventListener('click', () => {
    if (state.mode !== 'single') return;
    const grp = state.groups[state.groupIndex];
    if (!grp) return;
    batchLabelItems(grp.memberIds, null, false, true).catch(()=>{});
    if (typeof advanceSingleNext === 'function') advanceSingleNext();
  });
  addListener('#single-prev', 'click', () => { if (state.mode === 'single' && typeof navigateSingle === 'function') navigateSingle(-1); });
  addListener('#single-next', 'click', () => { if (state.mode === 'single' && typeof navigateSingle === 'function') navigateSingle(1); });
  // Data directory controls - separate Set Directory and Ingest buttons
  addListener('#apply-data-dir', 'click', async () => {
    const inp = el('#data-dir-input');
    if (!inp) return;
    const val = inp.value.trim();
    if (!val) { status('Please enter a folder path'); return; }
    try {
      status('Setting data directory...');
      const body = { data_dir: val, ingest: false };
      const res = await api('/set-data-dir', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const cur = el('#data-dir-current'); if (cur) cur.textContent = `Current: ${res.data_dir}`;
      status('Data directory updated');
    } catch (e) {
      status('Failed to update data directory');
    }
  });
  
  addListener('#ingest-data-dir', 'click', async () => {
    const inp = el('#data-dir-input');
    if (!inp) return;
    const val = inp.value.trim();
    if (!val) { status('Please enter a folder path'); return; }
    try {
      const maxGroupsVal = (el('#apply-max-groups')?.value || '').trim();
      const maxGroups = maxGroupsVal ? parseInt(maxGroupsVal, 10) : null;
      const requireAll = !!el('#apply-require-all')?.checked;
      const generatePNGs = !!el('#apply-generate-pngs')?.checked;
      status('Ingesting data...');

      // Start async ingest with progress reporting
      const startBody = { data_dir: val, by_groups: !!(maxGroups && maxGroups > 0), max_groups: maxGroups, generate_pngs: generatePNGs, require_all_roles: requireAll };
      const start = await api('/ingest/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(startBody) });
      if (!start.ok) { status(start.msg || 'Failed to start ingest'); return; }
      if (start.queued) {
        const pos = (typeof start.position === 'number') ? start.position : 0;
        status(`Ingest queued (position ${pos}). It will start automatically...`);
      }

      // Poll status until completion
      const progressBar = el('#progress-bar');
      let timer;
      const stop = () => { if (timer) { clearInterval(timer); timer = null; } };
      const render = (st) => {
        const total = st.total || 0;
        const totalFiles = st.total_files || 0;
        const done = st.done || 0;
        const pct = total ? Math.round(done / total * 100) : 0;
        if (progressBar) { progressBar.style.width = `${pct}%`; progressBar.textContent = `${pct}%`; }
        const eta = (st.eta_sec || st.eta_sec === 0) ? ` • ETA ${Math.max(0, st.eta_sec)}s` : '';
        const pngs = st.pngs ? ` • PNGs ${st.pngs}/${totalFiles}` : '';
        const errs = st.errors ? ` • errors ${st.errors}` : '';
        
        // Show special notification when database is ready for classification
        let statusMsg = st.message || (st.running ? 'running' : 'idle');
        if (st.db_ready && st.running) {
          statusMsg = `✅ ${statusMsg} - You can start classifying now!`;
        }
        
        // Show appropriate units (triplets vs files)
        const unitLabel = st.by_groups ? 'triplets' : 'files';
        const progressInfo = st.by_groups && totalFiles ? 
          `${done}/${total} ${unitLabel} (${st.ingested || 0}/${totalFiles} files) (${pct}%)` :
          `${done}/${total} ${unitLabel} (${pct}%)`;
        
        status(`${statusMsg} • ${progressInfo}${eta}${pngs}${errs}`);
      };
      let dbReadyNotified = false;
      const tick = async () => {
        try {
          const st = await api('/ingest/status');
          if (!st.ok) { status(st.msg || 'status error'); return; }
          render(st);
          
          // Auto-refresh items list when database becomes ready for first time
          if (st.db_ready && !dbReadyNotified) {
            dbReadyNotified = true;
            await loadItems();
            try { await refreshStatsDisplay(); } catch(_){ }
          }
          
          if (!st.running) {
            stop();
            await loadItems();
            try { await refreshStatsDisplay(); } catch(_){ }
          }
        } catch(e){ status(`⚠️ ${e.message}`); }
      };
      await tick();
      timer = setInterval(tick, 1000);
    } catch (e) {
      status('Failed to ingest data');
    }
  });
  // Mode toggle
  const applyMode = async (newMode) => {
    if (newMode !== 'single' && newMode !== 'grid') return;
    state.mode = newMode;
    localStorage.setItem('al_mode', state.mode);
    state.currentIndex = 0;
    state.groupIndex = 0;
    
    // Set tile size to 'large' when switching to single mode
    if (state.mode === 'single') {
      state.tile_size = 'large';
      localStorage.setItem('al_tile_size', 'large');
      const tileSizeSel = el('#tileSize');
      if (tileSizeSel) tileSizeSel.value = 'large';
    }
    
    // Hide page size selector in single mode; keep size selector visible
    try {
      const ps = el('#pageSize');
      const pageSizeGroup = ps ? ps.closest('.quick-control-group') : null;
      if (pageSizeGroup) pageSizeGroup.style.display = state.mode === 'single' ? 'none' : '';
    } catch(_){ }
    if (state.mode === 'single') await renderSingleView();
    else { const grid = el('#grid'); if (grid) grid.style.display = ''; const sv = el('#single-view'); if (sv) sv.style.display = 'none'; try { const ac = el('#assignment-controls'); if (ac) ac.style.display = ''; const bac = el('#bottom-assignment-controls'); if (bac) bac.style.display = ''; } catch(_){ } }
    const modeSel = el('#mode-select'); if (modeSel) modeSel.value = state.mode;
  };
  // Mode select dropdown removed - using buttons instead
  addListener('#view-mode-grid', 'click', async ()=>{ await applyMode('grid'); });
  addListener('#view-mode-single', 'click', async ()=>{ await applyMode('single'); });
  // Assignment controls
  addListener('#clear-assignments', 'click', ()=>{ clearAssignments(); loadItems(); });
  const applyAssignments = async ()=>{
    // Map cycling indices to classes dynamically for N classes
    const classes = state.classes.map(c=>c.name);
    const byIdx = new Map();
    state.assignmentMap.forEach((idx, id)=>{
      if (typeof idx === 'number' && idx >= 0 && idx < classes.length){
        if (!byIdx.has(idx)) byIdx.set(idx, []);
        byIdx.get(idx).push(id);
      }
    });
    for (let i=0;i<classes.length;i++){
      const ids = byIdx.get(i) || [];
      const cls = classes[i];
      if (ids.length && cls){ await batchLabelItems(ids, cls, false, false); }
    }
    clearAssignments();
    await loadItems();
  };
  addListener('#apply-assignments', 'click', applyAssignments);
  addListener('#apply-assignments-bottom', 'click', applyAssignments);
  document.addEventListener('click', (e) => {
    const is = (sel) => e.target.closest(sel);
    console.log('Click event on:', e.target, 'ID:', e.target.id, 'Closest bulk:', e.target.closest('.bulk-action-btn'));
    if (is('#addClass')) { addClassRow(); }
    else if (is('#saveClasses')) { saveClassesFromModal(); }
    else if (is('#closeClasses')) { closeModal('#classesModal'); }
    else if (is('.remove-class-btn')) { 
      const idx = parseInt(e.target.getAttribute('data-idx'));
      removeClassRow(idx);
    }
    else if (is('#closeMap')) { closeModal('#mapModal'); }
    else if (is('#closeStats')) { closeModal('#statsModal'); }
    else if (is('#refreshStats')) { refreshStatsDisplay(); }
    else if (is('#closeImage')) { closeModal('#imageModal'); }
    else if (is('#closeBulkOps')) { closeModal('#bulkOpsModal'); }
    else if (is('#export-labels')) { console.log('Export labels clicked'); exportData('labels'); }
    else if (is('#export-full')) { console.log('Export full clicked'); exportData('full'); }
    else if (is('#export-model')) { console.log('Export model clicked'); exportData('model'); }
    else if (is('#import-labels') || is('#import-config')) { const importFile = el('#import-file'); if (importFile) importFile.click(); }
    else if (is('#reset-all-labels')) { console.log('Reset all labels clicked!'); performBulkOperation('reset-all-labels', is('#reset-all-labels')); }
    else if (is('#auto-label-confident')) { console.log('Auto-label confident clicked!'); performBulkOperation('auto-label-confident', is('#auto-label-confident')); }
    else if (is('#backup-dataset')) { console.log('Backup dataset clicked!'); performBulkOperation('backup-dataset', is('#backup-dataset')); }
  });
  // Remove old suggestions modal listeners
  document.removeEventListener?.('click', ()=>{});
  document.addEventListener('change', (e) => {
    if (e.target.id === 'import-file') {
      const file = e.target.files[0];
      if (file) {
        if (file.name.endsWith('.csv')) { importLabels(file); }
        else { showImportStatus('❌ Please select a CSV file for label import', 'error'); }
      }
    }
  });
  const sizeSel = el('#pageSize');
  if (sizeSel) {
    const saved = parseInt(localStorage.getItem('al_page_size') || '', 10);
    if (!Number.isNaN(saved) && saved > 0) { state.page_size = saved; }
    sizeSel.value = String(state.page_size);
    sizeSel.addEventListener('change', async (e)=>{
      const v = parseInt(e.target.value, 10) || 60; state.page_size = v; localStorage.setItem('al_page_size', String(v)); state.page = 1; await loadItems();
    });
  }
  const tileSizeSel = el('#tileSize');
  if (tileSizeSel) {
    const saved = localStorage.getItem('al_tile_size') || 'xlarge';
    state.tile_size = saved; tileSizeSel.value = saved;
    tileSizeSel.addEventListener('change', (e)=>{
      const v = e.target.value || 'medium'; state.tile_size = v; localStorage.setItem('al_tile_size', v); applyTileSize(v);
      if (state.mode === 'single' && typeof renderSingleView === 'function') { renderSingleView(); }
    });
  }
  addListener('#select-all', 'click', ()=>{ state.items.forEach(item => state.selectedItems.add(item.id)); updateBatchControls(); loadItems(); });
  addListener('#select-none', 'click', ()=>{ state.selectedItems.clear(); updateBatchControls(); loadItems(); });
  addListener('#batch-unsure', 'click', ()=>{ batchLabelItems(Array.from(state.selectedItems), null, true, false); });
  addListener('#batch-skip', 'click', ()=>{ batchLabelItems(Array.from(state.selectedItems), null, false, true); });
  // Pagination controls (bottom only)
  addListener('#bottom-first-page', 'click', () => goToPage(1));
  addListener('#bottom-prev-page', 'click', () => goToPage(state.page - 1));
  addListener('#bottom-next-page', 'click', () => goToPage(state.page + 1));
  addListener('#bottom-last-page', 'click', () => goToPage(state.total_pages));
  const searchInput = el('#search-input');
  const searchClear = el('#search-clear');
  const labelFilter = el('#label-filter');
  if (searchInput) {
    let searchTimeout;
    searchInput.addEventListener('input', (e) => {
      const query = e.target.value.trim(); state.search_query = query; searchClear.style.display = query ? 'block' : 'none';
      clearTimeout(searchTimeout); searchTimeout = setTimeout(async () => { state.page = 1; await loadItems(); }, 300);
    });
  }
  if (searchClear) {
    searchClear.addEventListener('click', async () => {
      searchInput.value = ''; state.search_query = ''; searchClear.style.display = 'none'; state.page = 1; await loadItems();
    });
  }
  if (labelFilter) {
    labelFilter.addEventListener('change', async (e) => { state.label_filter = e.target.value; state.page = 1; await loadItems(); });
  }
  document.addEventListener('keydown', async (e)=>{
    const k = e.key; const first = state.items[0];
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if(!first) return;
    if(state.keyMap[k]){
      if (state.mode === 'single'){
        const grp = state.groups[state.groupIndex];
        const ids = grp ? grp.memberIds : [(state.items[state.currentIndex] || first).id];
        // Fire-and-forget label to keep UI responsive; refresh happens in background
        batchLabelItems(ids, state.keyMap[k], false, false).catch(()=>{});
        await advanceSingleNext();
      } else {
        if (state.selectedItems.size > 0) await batchLabelItems(Array.from(state.selectedItems), state.keyMap[k], false, false);
        else await labelItems([first.id], state.keyMap[k], false, false);
      }
    }
    else if(k==='0'){
      if (state.mode === 'single'){
        const grp = state.groups[state.groupIndex];
        const ids = grp ? grp.memberIds : [(state.items[state.currentIndex] || first).id];
        batchLabelItems(ids, null, true, false).catch(()=>{});
        await advanceSingleNext();
      } else if (state.selectedItems.size > 0) await batchLabelItems(Array.from(state.selectedItems), null, true, false);
      else await labelItems([first.id], null, true, false);
    }
    else if(k==='x' || k==='X'){
      if (state.mode === 'single'){
        const grp = state.groups[state.groupIndex];
        const ids = grp ? grp.memberIds : [(state.items[state.currentIndex] || first).id];
        batchLabelItems(ids, null, false, true).catch(()=>{});
        await advanceSingleNext();
      } else if (state.selectedItems.size > 0) await batchLabelItems(Array.from(state.selectedItems), null, false, true);
      else await labelItems([first.id], null, false, true);
    }
    else if(k==='r' || k==='R'){ await retrain(); }
    else if(k==='a' && e.ctrlKey){ e.preventDefault(); state.items.forEach(item => state.selectedItems.add(item.id)); updateBatchControls(); loadItems(); }
    else if(k==='Escape'){ state.selectedItems.clear(); updateBatchControls(); loadItems(); }
    else if(k==='h' || k==='?'){ showHelp(); }
    else if(k==='ArrowLeft'){
      e.preventDefault();
      if (state.mode === 'single') await navigateSingle(-1); else if (state.total_pages > 1) await goToPage(state.page - 1);
    }
    else if(k==='ArrowRight' || k==='Enter'){
      e.preventDefault();
      if (state.mode === 'single') await navigateSingle(1); else if (state.total_pages > 1) await goToPage(state.page + 1);
    }
    else if(k==='s' || k==='S'){ await openStats(); }
    else if(k==='v' || k==='V'){ await openMap(); }
  });

  // Entire Page assignment inline buttons (bottom-right, same bar as Apply assignments)
  const pageAssignInline = el('#page-assign-inline');
  const rebuildPageAssignButtons = () => {
    if (!pageAssignInline) return;
    pageAssignInline.innerHTML = '';
    if (!state.classes || state.classes.length === 0) return;
    // Title-like badge to match bar context
    const title = document.createElement('span');
    title.className = 'assign-hint';
    title.textContent = 'Entire Page';
    pageAssignInline.appendChild(title);
    state.classes.forEach((cls, i) => {
      const key = (i+1).toString();
      const btn = document.createElement('button');
      btn.className = 'page-assign-btn';
      btn.textContent = `${key}: ${cls.name}`;
      btn.title = `Assign all items on this page to ${cls.name}`;
      try {
        if (typeof getClassColorByIndex === 'function'){
          const c = getClassColorByIndex(i); if (c) { btn.style.borderColor = c; btn.style.background = hexToRgba(c, 0.2); }
        }
      } catch(_){ }
      btn.addEventListener('click', async () => {
        const groups = Array.isArray(state.groups) ? state.groups : [];
        const ids = groups.flatMap(g => g.memberIds || []).filter(Boolean);
        const groupCount = groups.length || 0;
        if (ids.length === 0) return;
        if (!confirm(`Assign ALL ${groupCount} groups on this page to ${cls.name}?`)) return;
        // Build a single payload of all member IDs across the visible triplet groups
        const allIds = groups.flatMap(g => (g.memberIds || [])).filter(Boolean);
        await batchLabelItems(allIds, cls.name, false, false);
      });
      pageAssignInline.appendChild(btn);
    });
  };
  rebuildPageAssignButtons();
  document.addEventListener('al:classes-updated', rebuildPageAssignButtons);
}

function setupCategoryDropdowns() {
  // Get all category toggle buttons
  const categoryToggles = document.querySelectorAll('.category-toggle');
  let openOverlay = null; // { dropdown, originalParent, placeholder, cleanup }
  
  const closeOpenOverlay = () => {
    if (!openOverlay) return;
    const { dropdown, originalParent, placeholder } = openOverlay;
    try {
      dropdown.classList.remove('overlay');
      dropdown.style.position = '';
      dropdown.style.top = '';
      dropdown.style.left = '';
      dropdown.style.right = '';
      dropdown.style.width = '';
      dropdown.style.minWidth = '';
      dropdown.style.maxHeight = '';
      dropdown.style.overflowY = '';
      dropdown.style.zIndex = '';
      dropdown.classList.remove('show');
      if (placeholder && originalParent) {
        originalParent.insertBefore(dropdown, placeholder);
        placeholder.remove();
      }
    } catch(_){}
    openOverlay = null;
  };
  
  const openAsOverlay = (toggle, dropdown) => {
    // Reattach as fixed overlay under body, positioned under toggle
    const rect = toggle.getBoundingClientRect();
    const placeholder = document.createElement('span');
    const originalParent = dropdown.parentElement;
    originalParent.insertBefore(placeholder, dropdown);
    document.body.appendChild(dropdown);
    dropdown.classList.add('overlay');
    dropdown.classList.add('show');
    // Positioning
    dropdown.style.position = 'fixed';
    const margin = 6;
    const left = Math.min(
      Math.max(10, rect.left),
      Math.max(10, window.innerWidth - (dropdown.offsetWidth || 240) - 10)
    );
    dropdown.style.top = `${Math.min(rect.bottom + margin, window.innerHeight - 10)}px`;
    dropdown.style.left = `${left}px`;
    dropdown.style.right = 'auto';
    dropdown.style.minWidth = `${Math.max(rect.width, 200)}px`;
    dropdown.style.maxHeight = '60vh';
    dropdown.style.overflowY = 'auto';
    dropdown.style.zIndex = '9000';
    
    openOverlay = { dropdown, originalParent, placeholder };
  };
  
  categoryToggles.forEach(toggle => {
    toggle.addEventListener('click', (e) => {
      e.stopPropagation();
      const category = toggle.getAttribute('data-category');
      const dropdown = document.getElementById(`${category}-dropdown`);
      
      // Close all other dropdowns
      document.querySelectorAll('.category-dropdown').forEach(dd => {
        if (dd !== dropdown) {
          dd.classList.remove('show');
        }
      });
      
      // Remove active state from all other toggles
      document.querySelectorAll('.category-toggle').forEach(t => {
        if (t !== toggle) {
          t.classList.remove('active');
        }
      });
      
      // Toggle current dropdown
      if (dropdown) {
        const isVisible = dropdown.classList.contains('show') && openOverlay?.dropdown === dropdown;
        if (isVisible) {
          closeOpenOverlay();
          toggle.classList.remove('active');
        } else {
          closeOpenOverlay();
          openAsOverlay(toggle, dropdown);
          toggle.classList.add('active');
        }
      }
    });
  });
  
  // Close dropdowns when clicking outside
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.nav-category-dropdown') && !e.target.closest('.category-dropdown')) {
      closeOpenOverlay();
      document.querySelectorAll('.category-toggle').forEach(toggle => {
        toggle.classList.remove('active');
      });
    }
  });
  
  // Prevent dropdown from closing when clicking inside it
  document.querySelectorAll('.category-dropdown').forEach(dropdown => {
    dropdown.addEventListener('click', (e) => {
      e.stopPropagation();
    });
  });
  
  // Close on escape / resize / scroll for robustness
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeOpenOverlay();
      document.querySelectorAll('.category-toggle').forEach(toggle => toggle.classList.remove('active'));
    }
  });
  window.addEventListener('resize', () => closeOpenOverlay());
  window.addEventListener('scroll', () => closeOpenOverlay(), true);
}

(async function(){
  await loadClasses().catch(()=>{});
  initTheme();
  setupUI();
  // Initialize data directory UI
  try {
    const cfg = await api('/config');
    const inp = el('#data-dir-input'); if (inp) inp.value = cfg.data_dir || '';
    const cur = el('#data-dir-current'); if (cur) cur.textContent = cfg.data_dir ? `Current: ${cfg.data_dir}` : 'Current: (not set)';
  } catch {}
  const savedMode = localStorage.getItem('al_mode') || 'grid';
  state.mode = savedMode;
  // Set tile size to 'large' if mode is 'single', otherwise use saved tile size
  const savedTileSize = localStorage.getItem('al_tile_size') || 'xlarge';
  state.tile_size = savedMode === 'single' ? 'large' : savedTileSize;
  // Mode select dropdown removed - mode is set via buttons
  const tileSizeSel = el('#tileSize'); if (tileSizeSel) tileSizeSel.value = state.tile_size;
  // Default to random order of triplet groups on initial load
  if (!state.queue) state.queue = 'all';
  
  // Load items with error handling to prevent UI blocking
  try {
    await loadItems();
    if (state.mode === 'single') await renderSingleView();
  } catch (e) {
    console.warn('Initial load failed:', e);
    status('Ready - no items loaded yet');
  }
  if (typeof refreshAssignHint === 'function') refreshAssignHint();
  try { const rf = el('#page-assign-fab'); if (rf) rf.style.display = (state.mode === 'single') ? 'none' : ''; } catch(_){ }
  // Apply initial visibility for page size group based on mode
  try {
    const ps = el('#pageSize');
    const pageSizeGroup = ps ? ps.closest('.quick-control-group') : null;
    if (pageSizeGroup) pageSizeGroup.style.display = state.mode === 'single' ? 'none' : '';
  } catch(_){}
  // Hide probability band controls initially unless queue=band
  try {
    const bandGroup = el('#header-probability-band');
    if (bandGroup) bandGroup.style.display = (state.queue === 'band') ? 'flex' : 'none';
  } catch(_){ }
  // Ensure apply-assignments controls visibility matches initial mode
  try {
    const ac = el('#assignment-controls'); if (ac) ac.style.display = state.mode === 'single' ? 'none' : '';
    const bac = el('#bottom-assignment-controls'); if (bac) bac.style.display = state.mode === 'single' ? 'none' : '';
  } catch(_){ }
})();



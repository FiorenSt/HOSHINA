async function loadClasses(){
  const data = await api('/classes');
  state.classes = data;
  state.keyMap = {};
  data.forEach((c,i)=>{
    const k = (i+1).toString();
    state.keyMap[k] = c.name;
  });
  // Notify UI to refresh any class-dependent controls
  document.dispatchEvent(new CustomEvent('al:classes-updated'));
  if (typeof syncClassColorsWithClasses === 'function') {
    syncClassColorsWithClasses();
  }
}

async function saveClasses(list){
  await api('/classes', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify(list),
  });
  await loadClasses();
}

async function openClasses(){
  await loadClasses();
  const list = el('#classList');
  list.innerHTML = '';
  const colors = (typeof syncClassColorsWithClasses === 'function') ? syncClassColorsWithClasses() : [];
  state.classes.forEach((c, i)=>{
    const row = document.createElement('div');
    const color = colors[i] || '#38bdf8';
    row.innerHTML = `
      <div class="class-row" style="display:flex; align-items:center; gap:8px;">
        <input type="text" value="${c.name}" data-idx="${i}"/>
        <button class="color-button" data-idx="${i}" title="Choose color" aria-label="Choose color" style="width:18px; height:18px; border:1px solid var(--border-color); border-radius:3px; background:${color}; cursor:pointer;"></button>
        <input type="color" class="class-color" value="${color}" data-idx="${i}" title="Custom color" style="display:none;"/>
        <kbd>${i+1}</kbd>
        <button class="remove-class-btn" data-idx="${i}" title="Remove class">×</button>
      </div>`;
    list.appendChild(row);
  });
  openModal('#classesModal');
  // Bind Save/Close explicitly to ensure they trigger even inside modal overlay
  try {
    const saveBtn = document.getElementById('saveClasses');
    if (saveBtn) {
      saveBtn.onclick = (e)=>{ e.preventDefault(); e.stopPropagation(); saveClassesFromModal(); };
    }
    const closeBtn = document.getElementById('closeClasses');
    if (closeBtn) {
      closeBtn.onclick = (e)=>{ e.preventDefault(); e.stopPropagation(); closeModal('#classesModal'); };
    }
  } catch(_){ }
  // Color popover logic
  let openPopover = null;
  const closePopover = () => { if (openPopover && openPopover.parentNode) { openPopover.parentNode.removeChild(openPopover); openPopover = null; } };
  const openColorPopover = (btn, idx) => {
    closePopover();
    const palette = (typeof getDefaultPalette === 'function' ? getDefaultPalette(12) : ['#3b82f6','#ef4444','#f59e0b','#10b981','#8b5cf6','#ec4899','#06b6d4','#22c55e','#f97316','#eab308','#14b8a6','#6366f1']);
    const pop = document.createElement('div');
    pop.className = 'color-popover';
    pop.style.position = 'fixed';
    pop.style.zIndex = '100000';
    pop.style.background = 'var(--bg-panel, #111827)';
    pop.style.border = '1px solid var(--border-color, #374151)';
    pop.style.borderRadius = '6px';
    pop.style.padding = '8px';
    pop.style.boxShadow = '0 6px 22px rgba(0,0,0,0.35)';
    // Palette grid
    const grid = document.createElement('div');
    grid.style.display = 'grid';
    grid.style.gridTemplateColumns = 'repeat(6, 18px)';
    grid.style.gap = '6px';
    palette.forEach(col => {
      const sw = document.createElement('button');
      sw.type = 'button';
      sw.title = col;
      sw.style.width = '18px'; sw.style.height = '18px'; sw.style.borderRadius = '3px';
      sw.style.border = '1px solid var(--border-color, #374151)';
      sw.style.background = col; sw.style.cursor = 'pointer';
      sw.addEventListener('click', () => {
        const inp = list.querySelector(`input.class-color[data-idx="${idx}"]`);
        if (inp) inp.value = col;
        btn.style.background = col;
        closePopover();
      });
      grid.appendChild(sw);
    });
    pop.appendChild(grid);
    // Custom picker row
    const customRow = document.createElement('div');
    customRow.style.display = 'flex'; customRow.style.alignItems = 'center'; customRow.style.gap = '8px'; customRow.style.marginTop = '8px';
    const label = document.createElement('span'); label.textContent = 'Custom:'; label.style.fontSize = '12px'; label.style.color = 'var(--text-secondary, #9ca3af)';
    const picker = document.createElement('input'); picker.type = 'color'; picker.value = (list.querySelector(`input.class-color[data-idx="${idx}"]`)?.value) || '#38bdf8';
    picker.addEventListener('input', () => {
      const inp = list.querySelector(`input.class-color[data-idx="${idx}"]`);
      if (inp) inp.value = picker.value;
      btn.style.background = picker.value;
    });
    const applyBtn = document.createElement('button'); applyBtn.type = 'button'; applyBtn.textContent = 'Apply';
    applyBtn.style.fontSize = '12px'; applyBtn.style.padding = '2px 6px'; applyBtn.style.border = '1px solid var(--border-color, #374151)'; applyBtn.style.borderRadius = '4px'; applyBtn.style.background = 'var(--bg-elev, #1f2937)'; applyBtn.style.color = 'var(--text, #e5e7eb)';
    applyBtn.addEventListener('click', () => { const clr = picker.value; const inp = list.querySelector(`input.class-color[data-idx="${idx}"]`); if (inp) inp.value = clr; btn.style.background = clr; closePopover(); });
    customRow.appendChild(label); customRow.appendChild(picker); customRow.appendChild(applyBtn);
    pop.appendChild(customRow);
    const modalContainer = document.querySelector('#classesModal .modal-content') || document.body;
    modalContainer.appendChild(pop);
    // Position under the button
    const r = btn.getBoundingClientRect();
    pop.style.top = `${Math.min(r.bottom + 6, window.innerHeight - pop.offsetHeight - 10)}px`;
    pop.style.left = `${Math.min(Math.max(10, r.left), window.innerWidth - pop.offsetWidth - 10)}px`;
    openPopover = pop;
    // Close on outside click or escape
    const onDocClick = (e) => { if (!pop.contains(e.target) && e.target !== btn) { closePopover(); document.removeEventListener('click', onDocClick, true); document.removeEventListener('keydown', onEsc, true); } };
    const onEsc = (e) => { if (e.key === 'Escape') { closePopover(); document.removeEventListener('click', onDocClick, true); document.removeEventListener('keydown', onEsc, true); } };
    setTimeout(()=>{ document.addEventListener('click', onDocClick, true); document.addEventListener('keydown', onEsc, true); }, 0);
  };
  list.addEventListener('click', (e)=>{
    const btn = e.target.closest('.color-button');
    if (!btn) return;
    e.preventDefault();
    e.stopPropagation();
    const idx = parseInt(btn.getAttribute('data-idx'));
    openColorPopover(btn, idx);
  });
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
      const palette = (typeof getDefaultPalette === 'function') ? getDefaultPalette(n) : existing;
      color = existing[i] || palette[i % palette.length] || '#3b82f6';
    }
  } catch {}
  row.innerHTML = `
    <div class="class-row" style="display:flex; align-items:center; gap:8px;">
      <input type="text" value="class_${i+1}" data-idx="${i}"/>
      <button class="color-button" data-idx="${i}" title="Choose color" aria-label="Choose color" style="width:18px; height:18px; border:1px solid var(--border-color); border-radius:3px; background:${color}; cursor:pointer;"></button>
      <input type="color" class="class-color" value="${color}" data-idx="${i}" title="Custom color" style="display:none;"/>
      <kbd>${i+1}</kbd>
      <button class="remove-class-btn" data-idx="${i}" title="Remove class">×</button>
    </div>`;
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
  // Debounce concurrent saves (avoid double POST from multiple handlers)
  if (saveClassesFromModal._saving) return;
  saveClassesFromModal._saving = true;
  try {
    const listEl = el('#classList');
    const nameInputs = [...listEl.querySelectorAll('input[type="text"]')];
    const colorInputs = [...listEl.querySelectorAll('input.class-color')];
    const classes = nameInputs.map((inp, i)=>({name: (inp.value || '').trim(), key: (i+1).toString(), order: i}));
    // Validate unique names to satisfy DB UNIQUE constraint
    const seen = new Set();
    const dups = new Set();
    classes.forEach(c=>{ const n = c.name; if (seen.has(n)) dups.add(n); else seen.add(n); });
    if (dups.size > 0){
      alert(`Duplicate class names are not allowed: ${Array.from(dups).join(', ')}`);
      return;
    }
    // Disable Save button during request
    const saveBtn = document.getElementById('saveClasses');
    const oldText = saveBtn ? saveBtn.textContent : '';
    if (saveBtn){ saveBtn.disabled = true; saveBtn.textContent = 'Saving...'; }
    try {
      await saveClasses(classes);
    } catch (e) {
      const msg = (e && e.message) ? e.message : 'Failed to save classes';
      alert(msg);
      return;
    }
    // Persist colors alongside (frontend only)
    try {
      const colors = classes.map((_, i)=> colorInputs[i] ? colorInputs[i].value : '#38bdf8');
      if (typeof saveClassColors === 'function') saveClassColors(colors);
      // Immediately reflect color changes in UI elements without waiting for next refresh
      try { if (typeof syncClassColorsWithClasses === 'function') syncClassColorsWithClasses(); } catch(_){ }
      try { if (typeof updateBatchClassButtons === 'function') updateBatchClassButtons(); } catch(_){ }
      try { if (typeof updateSingleClassButtons === 'function') updateSingleClassButtons(); } catch(_){ }
      try { if (typeof refreshAssignHint === 'function') refreshAssignHint(); } catch(_){ }
    } catch {}
    closeModal('#classesModal');
    await loadItems();
    if (saveBtn){ saveBtn.disabled = false; saveBtn.textContent = oldText; }
  } finally {
    saveClassesFromModal._saving = false;
  }
}



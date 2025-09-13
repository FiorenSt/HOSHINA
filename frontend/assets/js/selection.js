function updateSessionStats(count, type = 'labeled') {
  const now = Date.now();
  const timeSinceLastLabel = now - state.session.lastLabelTime;
  if (type === 'labeled') state.session.labeledCount += count;
  else if (type === 'skipped') state.session.skippedCount += count;
  else if (type === 'unsure') state.session.unsureCount += count;
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
  // if (!state.loading && totalLabeled > 0) {
  //   const sessionInfo = `⏱️ ${timeStr} • ${totalLabeled} labeled • ${rate}/min`;
  //   const statusEl = el('#status');
  //   if (statusEl && !statusEl.textContent.includes('Loading') && !statusEl.textContent.includes('Error')) {
  //     statusEl.textContent = sessionInfo;
  //   }
  // }
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
    const selectedIds = getItemsInSelection();
    state.selectedItems = selectedIds;
    // When dragging, cycle-assignment is not used; keep classic selection visuals only
    els('.tile').forEach(tile => {
      const id = parseInt(tile.dataset.id);
      if (selectedIds.has(id)) {
        tile.classList.add('selected');
      } else {
        tile.classList.remove('selected');
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



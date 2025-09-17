const state = {
  classes: [],
  queue: 'all',
  page: 1,
  page_size: 60,
  tile_size: 'xlarge',
  prob_low: 0.3,
  prob_high: 0.8,
  items: [],
  // Stable ordering seed used when requesting items to keep grid order deterministic
  order_seed: null,
  // Modes: 'grid' (existing) or 'single' for one-by-one
  mode: 'grid',
  currentIndex: 0, // legacy single-item index (grid-based)
  // Grouped single-mode navigation (triplet-level)
  groups: [],            // Array of { key, memberIds:number[], repId:number }
  groupIndex: 0,         // index in groups for single mode
  currentGroupKey: null, // currently displayed group's stable key (single mode)
  navigating: false,     // prevent duplicate renders during navigation
  keyMap: {},
  // Colors assigned per class index (aligned with state.classes order)
  classColors: [],
  // ===== Single view buffer (decoupled from grid pagination) =====
  single_items: [],          // appended items for single view only
  single_groups: [],         // appended groups for single view only
  single_next_page: 1,       // next page to fetch for single view
  single_total_pages: null,  // total pages available for single view (from backend)
  single_loading: false,     // to avoid concurrent fetches in single view
  single_total_groups: null, // total groups available for single view (from backend)
  single_remaining_groups: null, // remaining unlabeled groups counter for single view
  selectedItems: new Set(),
  // Click-cycling assignment map: itemId -> 0 (blue), 1 (red), 2 (yellow)
  assignmentMap: new Map(),
  stats: null,
  loading: false,
  // Sequencing/cancellation for rapid reloads (e.g., fast sort toggles)
  loadToken: 0,
  loadAbortController: null,
  total_items: 0,
  total_pages: 1,
  theme: 'dark',
  search_query: '',
  label_filter: '',
  // Suppress inline tile updates during atomic apply to avoid multi-step flashes
  suppressInlineLabelUpdates: false,
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
  tripletCheckDone: false,
  // Sorting by predicted probability: 'asc' | 'desc' | '' (unsorted)
  sort_pred: ''
};

function el(sel){ return document.querySelector(sel); }
function els(sel){ return document.querySelectorAll(sel); }

function applyTileSize(size) {
  const grid = el('#grid');
  if (grid) {
    grid.classList.remove('tile-small', 'tile-medium', 'tile-large', 'tile-xlarge');
    grid.classList.add(`tile-${size}`);
    // Avoid flashing the grid visible when in single (one-by-one) mode
    const shouldRemainHidden = state && state.mode === 'single';
    grid.style.display = 'none';
    grid.offsetHeight;
    grid.style.display = shouldRemainHidden ? 'none' : 'grid';
  }
}

// ===== Class colors helpers =====
function getDefaultPalette(n){
  // Vibrant, high-contrast palette
  const base = [
    '#3b82f6', // Blue-500
    '#ef4444', // Red-500
    '#f59e0b', // Amber-500
    '#10b981', // Emerald-500
    '#8b5cf6', // Violet-500
    '#ec4899', // Pink-500
    '#06b6d4', // Cyan-500
    '#22c55e', // Green-500
    '#f97316', // Orange-500
    '#eab308', // Yellow-500
    '#14b8a6', // Teal-500
    '#6366f1', // Indigo-500
  ];
  const out = [];
  for (let i=0;i<n;i++){ out.push(base[i % base.length]); }
  return out;
}

function loadClassColors(length){
  try {
    const raw = localStorage.getItem('al_class_colors');
    const parsed = raw ? JSON.parse(raw) : [];
    const out = Array.from({length}, (_,i)=> parsed[i] || getDefaultPalette(length)[i]);
    state.classColors = out;
    return out;
  } catch {
    const out = getDefaultPalette(length);
    state.classColors = out;
    return out;
  }
}

function saveClassColors(colors){
  try { localStorage.setItem('al_class_colors', JSON.stringify(colors)); } catch {}
  state.classColors = colors.slice();
}

function syncClassColorsWithClasses(){
  const n = (state.classes || []).length;
  const colors = loadClassColors(n);
  if (colors.length !== n){
    const palette = getDefaultPalette(n);
    const adj = Array.from({length:n}, (_,i)=> colors[i] || palette[i]);
    saveClassColors(adj);
  }
  return state.classColors;
}

function getClassColorByIndex(i){
  if (i == null || i < 0) return null;
  const colors = state.classColors && state.classColors.length ? state.classColors : syncClassColorsWithClasses();
  return colors[i] || getDefaultPalette(i+1)[i];
}

function hexToRgba(hex, alpha){
  let h = (hex || '').replace('#','');
  if (h.length === 3){ h = h.split('').map(c=>c+c).join(''); }
  if (h.length !== 6) return `rgba(0,0,0,${alpha})`;
  const r = parseInt(h.slice(0,2),16);
  const g = parseInt(h.slice(2,4),16);
  const b = parseInt(h.slice(4,6),16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// ===== API base autodetect (supports file:// usage) =====
const API_BASE = (() => {
  try {
    const saved = (typeof localStorage !== 'undefined') ? (localStorage.getItem('al_api_base') || '').trim() : '';
    if (saved) {
      // allow saving either full base or root; normalize trailing slash
      return saved.replace(/\/$/, '') + (saved.endsWith('/api') ? '' : (saved.endsWith('/api/') ? '' : (saved.endsWith('/') ? 'api' : '/api')));
    }
  } catch {}
  if (typeof window !== 'undefined'){
    if (window.location.protocol === 'file:'){
      return 'http://127.0.0.1:8000/api';
    }
    return (window.location.origin || '') + '/api';
  }
  return '/api';
})();

// Rewrite bare '/api...' fetches from other modules to absolute API_BASE when needed
if (typeof window !== 'undefined' && typeof window.fetch === 'function'){
  const __origFetch = window.fetch.bind(window);
  window.fetch = (input, init) => {
    try {
      if (typeof input === 'string' && input.startsWith('/api')){
        input = API_BASE + input.substring(4);
      }
    } catch {}
    return __origFetch(input, init);
  };
}

async function api(path, opts={}){
  const url = (path.startsWith('http://') || path.startsWith('https://')) ? path : (API_BASE + path);
  const r = await fetch(url, opts);
  if(!r.ok){ throw new Error(`API ${path} failed: ${r.status}`); }
  return await r.json();
}

function status(msg){ 
  el('#status').textContent = msg; 
  if (msg.includes('Loading') || msg.includes('Training') || msg.includes('Ingesting') || msg.includes('running') || msg.includes('scanning')) {
    el('#status').classList.add('loading');
  } else {
    el('#status').classList.remove('loading');
  }
}

function toggleTheme() {
  const newTheme = state.theme === 'dark' ? 'light' : 'dark';
  state.theme = newTheme;
  if (newTheme === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
  } else {
    document.documentElement.removeAttribute('data-theme');
  }
  const themeBtn = el('#theme-toggle');
  if (themeBtn) {
    themeBtn.textContent = newTheme === 'dark' ? '🌙' : '☀️';
    themeBtn.title = `Switch to ${newTheme === 'dark' ? 'light' : 'dark'} mode`;
  }
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


// Helpers for click-cycling assignments
function getAssignmentCssClass(itemId){
  const idx = state.assignmentMap ? state.assignmentMap.get(itemId) : undefined;
  if (idx === 0) return 'assign-blue';
  if (idx === 1) return 'assign-red';
  if (idx === 2) return 'assign-yellow';
  return '';
}

function applyAssignmentClassToNode(node, idx){
  if (!node) return;
  node.classList.remove('assign-blue','assign-red','assign-yellow');
  // Only keep legacy CSS classes for the first three indices if present; otherwise rely on inline styles
  if (idx === 0) node.classList.add('assign-blue');
  else if (idx === 1) node.classList.add('assign-red');
  else if (idx === 2) node.classList.add('assign-yellow');

  const tag = node.querySelector && node.querySelector('.assign-tag');
  const removeAssignedBadge = () => {
    try {
      const b = node.querySelector('.badge.assigned');
      if (b && b.parentNode) b.parentNode.removeChild(b);
    } catch(_){}
  };
  const metaRight = node.querySelector && (node.querySelector('.meta div:last-child') || node.querySelector('.meta div'));
  const ensureAssignOverlayRight = (text, color) => {
    try {
      let ov = node.querySelector('.assign-overlay-right');
      if (!ov){
        ov = document.createElement('span');
        ov.className = 'assign-overlay-right';
        node.appendChild(ov);
      }
      ov.textContent = text || '';
      ov.style.color = color || '';
      ov.style.display = text ? 'inline-block' : 'none';
    } catch(_){ }
  };
  const removeAssignOverlayRight = () => {
    try {
      const ov = node.querySelector('.assign-overlay-right');
      if (ov) ov.parentNode.removeChild(ov);
    } catch(_){ }
  };
  // dynamic color styling
  if (idx == null || idx < 0){
    node.style.borderColor = '';
    node.style.boxShadow = '';
    if (tag){
      tag.textContent = '';
      tag.style.color = '';
      tag.style.display = 'none';
    }
    // also remove any assigned badge/overlay when clearing assignment
    removeAssignedBadge();
    removeAssignOverlayRight();
    return;
  }
  const color = getClassColorByIndex(idx);
  if (color){
    node.style.borderColor = color;
    // Stronger, more vibrant dual halo
    const outer = hexToRgba(color, 0.65);
    const glow = hexToRgba(color, 0.45);
    node.style.boxShadow = `0 0 0 3px ${outer}, 0 0 14px ${glow}`;
    if (tag){
      const name = (state.classes && state.classes[idx] && state.classes[idx].name) ? state.classes[idx].name : '';
      tag.textContent = name;
      tag.style.color = color;
      // We'll hide this tag if we render a bottom-right badge to avoid duplicates
      tag.style.display = name ? 'inline-block' : 'none';
    }
    // Render an in-tile bottom-left overlay with the class name (always show when assigned)
    try {
      const name = (state.classes && state.classes[idx] && state.classes[idx].name) ? state.classes[idx].name : '';
      if (name){
        ensureAssignOverlayRight(name, color);
        removeAssignedBadge();
        if (tag) tag.style.display = 'none';
      } else {
        removeAssignOverlayRight();
        removeAssignedBadge();
        if (tag) tag.style.display = 'none';
      }
    } catch(_){ }
  } else if (tag){
    const name = (state.classes && state.classes[idx] && state.classes[idx].name) ? state.classes[idx].name : '';
    tag.textContent = name;
    tag.style.color = '';
    // Force visible even when CSS hides tags without legacy assign-* classes
    tag.style.display = name ? 'inline-block' : 'none';
    // Mirror logic without color: always show overlay when assigned
    try {
      if (name){
        ensureAssignOverlayRight(name, '');
        removeAssignedBadge();
        if (tag) tag.style.display = 'none';
      } else {
        removeAssignOverlayRight();
        removeAssignedBadge();
      }
    } catch(_){ }
  }
}

function cycleAssignment(itemId, node){
  const current = state.assignmentMap.has(itemId) ? state.assignmentMap.get(itemId) : -1;
  const totalClasses = (state.classes && state.classes.length) ? state.classes.length : 0;
  
  // Check if no classes are defined and show message to user
  if (totalClasses <= 0) {
    status('⚠️ Please define classes first before labeling images. Go to the Classes tab to set up your classification labels.');
    return; // Don't cycle assignment if no classes defined
  }
  
  const next = (current + 1 >= totalClasses ? -1 : current + 1);
  if (next === -1) state.assignmentMap.delete(itemId); else state.assignmentMap.set(itemId, next);
  applyAssignmentClassToNode(node, next);
}

// Ensure global access for event handlers in other files
if (typeof window !== 'undefined') {
  window.cycleAssignment = cycleAssignment;
  window.getClassColorByIndex = getClassColorByIndex;
  window.syncClassColorsWithClasses = syncClassColorsWithClasses;
  window.saveClassColors = saveClassColors;
  window.applyAssignmentClassToNode = applyAssignmentClassToNode;
  // Initialize or load stable order seed
  try {
    const savedSeed = localStorage.getItem('al_order_seed');
    if (savedSeed && /^\d+$/.test(savedSeed)) {
      state.order_seed = parseInt(savedSeed, 10);
    } else {
      // Derive a seed from current time once and persist; different datasets will reshuffle when desired via reset
      const newSeed = Math.floor(Date.now() % 2147483647);
      state.order_seed = newSeed;
      localStorage.setItem('al_order_seed', String(newSeed));
    }
  } catch(_) {
    state.order_seed = 12345;
  }
  window.refreshAssignHint = function refreshAssignHint(){
    try {
      const top = document.getElementById('assign-hint-top');
      const bottom = document.getElementById('assign-hint-bottom');
      const n = (state.classes || []).length;
      const parts = [];
      for (let i=0;i<n;i++){
        const key = (i+1).toString();
        const name = state.classes[i]?.name || `Class ${i+1}`;
        const color = typeof getClassColorByIndex === 'function' ? getClassColorByIndex(i) : null;
        const swatch = color ? `<span style="display:inline-block; width:10px; height:10px; background:${color}; border-radius:2px; vertical-align:middle; margin-right:4px;"></span>` : '';
        parts.push(`${key}: ${swatch}${name}`);
      }
      parts.push('None');
      const html = `Click cycles: ${parts.join(' → ')}`;
      if (top) top.innerHTML = html;
      if (bottom) bottom.innerHTML = html;
    } catch(_){ }
  };
}

function clearAssignments(){
  state.assignmentMap = new Map();
}


// ===== Grouping helpers =====
function groupKeyFromPath(path){
  try {
    const lower = (path || '').toLowerCase();
    const name = lower.split('/').pop();
    if (!name) return lower;
    const suffixes = ['_target.fits','_ref.fits','_diff.fits'];
    for (const suf of suffixes){
      if (name.endsWith(suf)){
        return name.substring(0, name.length - suf.length);
      }
    }
    // Fallback to stem without extension
    const dot = name.lastIndexOf('.');
    return dot > 0 ? name.substring(0, dot) : name;
  } catch(_){
    return path || '';
  }
}

if (typeof window !== 'undefined'){
  window.groupKeyFromPath = groupKeyFromPath;
}

// Helpers to persist whether predictions are available
function setPredictionsAvailableFlag(available){
  try { localStorage.setItem('al_predictions_available', available ? '1' : '0'); } catch(_){ }
}
function getPredictionsAvailableFlag(){
  try { return localStorage.getItem('al_predictions_available') === '1'; } catch(_){ return false; }
}

if (typeof window !== 'undefined'){
  window.getPredictionsAvailableFlag = getPredictionsAvailableFlag;
  window.setPredictionsAvailableFlag = setPredictionsAvailableFlag;
}



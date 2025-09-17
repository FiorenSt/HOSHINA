// Simple Model Builder UI (nodes, edges, params)
// Provides: window.modelBuilder.open()

(function(){
  const state = {
    graph: { nodes: [], edges: [], input_shape: [224,224,3] },
    selectedNodeId: null,
  };

  // Preset graphs (fully working architectures)
  const PRESETS = {
    'Simple CNN': {
      input_shape: [224,224,3],
      nodes: [
        { id: 'input', type: 'Input', params: {}, ui: { x: 40, y: 120, w: 140, h: 56, d: 12 } },
        { id: 'conv1', type: 'Conv2D', params: { filters: 32, kernel_size: 3, strides: 1, padding: 'same', activation: 'relu' }, ui: { x: 220, y: 120, w: 170, h: 56, d: 12 } },
        { id: 'pool1', type: 'MaxPool2D', params: { pool_size: 2, strides: 2, padding: 'same' }, ui: { x: 410, y: 120, w: 150, h: 56, d: 12 } },
        { id: 'conv2', type: 'Conv2D', params: { filters: 64, kernel_size: 3, strides: 1, padding: 'same', activation: 'relu' }, ui: { x: 580, y: 120, w: 170, h: 56, d: 12 } },
        { id: 'pool2', type: 'MaxPool2D', params: { pool_size: 2, strides: 2, padding: 'same' }, ui: { x: 770, y: 120, w: 150, h: 56, d: 12 } },
        { id: 'gap', type: 'GlobalAveragePooling2D', params: {}, ui: { x: 940, y: 120, w: 150, h: 56, d: 12 } },
      ],
      edges: [
        { from: 'input', to: 'conv1' },
        { from: 'conv1', to: 'pool1' },
        { from: 'pool1', to: 'conv2' },
        { from: 'conv2', to: 'pool2' },
        { from: 'pool2', to: 'gap' },
      ]
    },
    'Tiny SepConv': {
      input_shape: [224,224,3],
      nodes: [
        { id: 'input', type: 'Input', params: {}, ui: { x: 40, y: 120, w: 140, h: 56, d: 12 } },
        { id: 'sc1', type: 'SeparableConv2D', params: { filters: 32, kernel_size: 3, strides: 1, padding: 'same', activation: 'relu' }, ui: { x: 220, y: 120, w: 190, h: 56, d: 12 } },
        { id: 'pool1', type: 'MaxPool2D', params: { pool_size: 2, strides: 2, padding: 'same' }, ui: { x: 430, y: 120, w: 150, h: 56, d: 12 } },
        { id: 'sc2', type: 'SeparableConv2D', params: { filters: 64, kernel_size: 3, strides: 1, padding: 'same', activation: 'relu' }, ui: { x: 600, y: 120, w: 190, h: 56, d: 12 } },
        { id: 'gap', type: 'GlobalAveragePooling2D', params: {}, ui: { x: 810, y: 120, w: 150, h: 56, d: 12 } },
      ],
      edges: [
        { from: 'input', to: 'sc1' },
        { from: 'sc1', to: 'pool1' },
        { from: 'pool1', to: 'sc2' },
        { from: 'sc2', to: 'gap' },
      ]
    },
    'Mini ResNet': {
      input_shape: [224,224,3],
      nodes: [
        { id: 'input', type: 'Input', params: {}, ui: { x: 40, y: 120, w: 140, h: 56, d: 12 } },
        { id: 'conv1', type: 'Conv2D', params: { filters: 16, kernel_size: 3, strides: 1, padding: 'same', activation: 'relu' }, ui: { x: 220, y: 120, w: 170, h: 56, d: 12 } },
        { id: 'bn1', type: 'BatchNormalization', params: {}, ui: { x: 410, y: 120, w: 150, h: 56, d: 12 } },
        { id: 'act1', type: 'Activation', params: { activation: 'relu' }, ui: { x: 580, y: 120, w: 150, h: 56, d: 12 } },
        { id: 'conv2', type: 'Conv2D', params: { filters: 16, kernel_size: 3, strides: 1, padding: 'same', activation: 'linear' }, ui: { x: 740, y: 120, w: 170, h: 56, d: 12 } },
        { id: 'bn2', type: 'BatchNormalization', params: {}, ui: { x: 930, y: 120, w: 150, h: 56, d: 12 } },
        { id: 'proj', type: 'Conv2D', params: { filters: 16, kernel_size: 1, strides: 1, padding: 'same', activation: 'linear' }, ui: { x: 580, y: 260, w: 150, h: 56, d: 12 } },
        { id: 'add', type: 'Add', params: {}, ui: { x: 1100, y: 140, w: 130, h: 56, d: 12 } },
        { id: 'act2', type: 'Activation', params: { activation: 'relu' }, ui: { x: 1250, y: 140, w: 150, h: 56, d: 12 } },
        { id: 'gap', type: 'GlobalAveragePooling2D', params: {}, ui: { x: 1410, y: 140, w: 150, h: 56, d: 12 } },
      ],
      edges: [
        { from: 'input', to: 'conv1' },
        { from: 'conv1', to: 'bn1' },
        { from: 'bn1', to: 'act1' },
        { from: 'act1', to: 'conv2' },
        { from: 'conv2', to: 'bn2' },
        // skip path
        { from: 'input', to: 'proj' },
        // merge
        { from: 'bn2', to: 'add' },
        { from: 'proj', to: 'add' },
        { from: 'add', to: 'act2' },
        { from: 'act2', to: 'gap' },
      ]
    },
    'GAP Head': {
      input_shape: [224,224,3],
      nodes: [
        { id: 'input', type: 'Input', params: {}, ui: { x: 40, y: 120, w: 140, h: 56, d: 12 } },
        { id: 'conv', type: 'Conv2D', params: { filters: 32, kernel_size: 3, strides: 1, padding: 'same', activation: 'relu' }, ui: { x: 220, y: 120, w: 170, h: 56, d: 12 } },
        { id: 'gap', type: 'GlobalAveragePooling2D', params: {}, ui: { x: 410, y: 120, w: 150, h: 56, d: 12 } },
      ],
      edges: [
        { from: 'input', to: 'conv' },
        { from: 'conv', to: 'gap' },
      ]
    },
    'Tiny MLP': {
      input_shape: [224,224,3],
      nodes: [
        { id: 'input', type: 'Input', params: {}, ui: { x: 40, y: 120, w: 140, h: 56, d: 12 } },
        { id: 'flatten', type: 'Flatten', params: {}, ui: { x: 220, y: 120, w: 150, h: 56, d: 12 } },
        { id: 'dense1', type: 'Dense', params: { units: 256, activation: 'relu' }, ui: { x: 390, y: 120, w: 150, h: 56, d: 12 } },
        { id: 'drop', type: 'Dropout', params: { rate: 0.3 }, ui: { x: 550, y: 120, w: 130, h: 56, d: 12 } },
        { id: 'dense2', type: 'Dense', params: { units: 128, activation: 'relu' }, ui: { x: 690, y: 120, w: 150, h: 56, d: 12 } },
      ],
      edges: [
        { from: 'input', to: 'flatten' },
        { from: 'flatten', to: 'dense1' },
        { from: 'dense1', to: 'drop' },
        { from: 'drop', to: 'dense2' },
      ]
    },
    'VGG Small': {
      input_shape: [224,224,3],
      nodes: [
        { id: 'input', type: 'Input', params: {}, ui: { x: 40, y: 120, w: 140, h: 56, d: 12 } },
        { id: 'c11', type: 'Conv2D', params: { filters: 32, kernel_size: 3, padding: 'same', activation: 'relu' }, ui: { x: 220, y: 80, w: 170, h: 56, d: 12 } },
        { id: 'c12', type: 'Conv2D', params: { filters: 32, kernel_size: 3, padding: 'same', activation: 'relu' }, ui: { x: 400, y: 80, w: 170, h: 56, d: 12 } },
        { id: 'p1', type: 'MaxPool2D', params: { pool_size: 2, strides: 2, padding: 'same' }, ui: { x: 580, y: 80, w: 150, h: 56, d: 12 } },
        { id: 'c21', type: 'Conv2D', params: { filters: 64, kernel_size: 3, padding: 'same', activation: 'relu' }, ui: { x: 760, y: 80, w: 170, h: 56, d: 12 } },
        { id: 'c22', type: 'Conv2D', params: { filters: 64, kernel_size: 3, padding: 'same', activation: 'relu' }, ui: { x: 940, y: 80, w: 170, h: 56, d: 12 } },
        { id: 'p2', type: 'MaxPool2D', params: { pool_size: 2, strides: 2, padding: 'same' }, ui: { x: 1120, y: 80, w: 150, h: 56, d: 12 } },
        { id: 'gap', type: 'GlobalAveragePooling2D', params: {}, ui: { x: 1300, y: 80, w: 150, h: 56, d: 12 } },
      ],
      edges: [
        { from: 'input', to: 'c11' },
        { from: 'c11', to: 'c12' },
        { from: 'c12', to: 'p1' },
        { from: 'p1', to: 'c21' },
        { from: 'c21', to: 'c22' },
        { from: 'c22', to: 'p2' },
        { from: 'p2', to: 'gap' },
      ]
    },
    'Inception Mini': {
      input_shape: [224,224,3],
      nodes: [
        { id: 'input', type: 'Input', params: {}, ui: { x: 40, y: 160, w: 140, h: 56, d: 12 } },
        { id: 'stem', type: 'Conv2D', params: { filters: 32, kernel_size: 3, strides: 1, padding: 'same', activation: 'relu' }, ui: { x: 220, y: 160, w: 170, h: 56, d: 12 } },
        { id: 'b1_1x1', type: 'Conv2D', params: { filters: 16, kernel_size: 1, padding: 'same', activation: 'relu' }, ui: { x: 420, y: 40, w: 150, h: 56, d: 12 } },
        { id: 'b2_3x3', type: 'Conv2D', params: { filters: 24, kernel_size: 3, padding: 'same', activation: 'relu' }, ui: { x: 420, y: 140, w: 170, h: 56, d: 12 } },
        { id: 'b3_5x5', type: 'Conv2D', params: { filters: 16, kernel_size: 5, padding: 'same', activation: 'relu' }, ui: { x: 420, y: 240, w: 170, h: 56, d: 12 } },
        { id: 'b4_pool', type: 'MaxPool2D', params: { pool_size: 3, strides: 1, padding: 'same' }, ui: { x: 420, y: 340, w: 150, h: 56, d: 12 } },
        { id: 'b4_1x1', type: 'Conv2D', params: { filters: 16, kernel_size: 1, padding: 'same', activation: 'relu' }, ui: { x: 600, y: 340, w: 150, h: 56, d: 12 } },
        { id: 'concat', type: 'Concat', params: { axis: -1 }, ui: { x: 820, y: 160, w: 150, h: 56, d: 12 } },
        { id: 'gap', type: 'GlobalAveragePooling2D', params: {}, ui: { x: 1000, y: 160, w: 150, h: 56, d: 12 } },
      ],
      edges: [
        { from: 'input', to: 'stem' },
        { from: 'stem', to: 'b1_1x1' },
        { from: 'stem', to: 'b2_3x3' },
        { from: 'stem', to: 'b3_5x5' },
        { from: 'stem', to: 'b4_pool' },
        { from: 'b4_pool', to: 'b4_1x1' },
        { from: 'b1_1x1', to: 'concat' },
        { from: 'b2_3x3', to: 'concat' },
        { from: 'b3_5x5', to: 'concat' },
        { from: 'b4_1x1', to: 'concat' },
        { from: 'concat', to: 'gap' },
      ]
    },
    'Two-Branch Concat': {
      input_shape: [224,224,3],
      nodes: [
        { id: 'input', type: 'Input', params: {}, ui: { x: 40, y: 140, w: 140, h: 56, d: 12 } },
        { id: 'path1_conv', type: 'Conv2D', params: { filters: 32, kernel_size: 3, padding: 'same', activation: 'relu' }, ui: { x: 220, y: 80, w: 170, h: 56, d: 12 } },
        { id: 'path1_pool', type: 'MaxPool2D', params: { pool_size: 2, strides: 2, padding: 'same' }, ui: { x: 410, y: 80, w: 150, h: 56, d: 12 } },
        { id: 'path2_conv', type: 'Conv2D', params: { filters: 16, kernel_size: 1, padding: 'same', activation: 'relu' }, ui: { x: 220, y: 200, w: 150, h: 56, d: 12 } },
        { id: 'concat', type: 'Concat', params: { axis: -1 }, ui: { x: 600, y: 140, w: 150, h: 56, d: 12 } },
        { id: 'gap', type: 'GlobalAveragePooling2D', params: {}, ui: { x: 780, y: 140, w: 150, h: 56, d: 12 } },
      ],
      edges: [
        { from: 'input', to: 'path1_conv' },
        { from: 'path1_conv', to: 'path1_pool' },
        { from: 'input', to: 'path2_conv' },
        { from: 'path1_pool', to: 'concat' },
        { from: 'path2_conv', to: 'concat' },
        { from: 'concat', to: 'gap' },
      ]
    },
  };

  function ensureModal(){
    if (document.getElementById('modelBuilderModal')) return;
    const m = document.createElement('div');
    m.id = 'modelBuilderModal';
    m.className = 'modal hidden';
    m.innerHTML = `
      <div class="modal-content fullscreen">
        <div style="display:flex; align-items:center; justify-content:space-between; padding:10px 14px; border-bottom:1px solid #1f2937; background: var(--card);">
          <h2 style="margin:0;">🧩 Model Builder</h2>
          <div style="display:flex; gap:8px; align-items:center;">
            <label for="mb-preset" style="font-size:13px; color:#9ca3af;">Preset</label>
            <select id="mb-preset" class="config-select" style="min-width:220px;">
              <option value="">Choose preset…</option>
              <option value="Simple CNN">Simple CNN</option>
              <option value="Tiny SepConv">Tiny SepConv</option>
              <option value="Mini ResNet">Mini ResNet</option>
              <option value="GAP Head">GAP Head</option>
              <option value="Tiny MLP">Tiny MLP</option>
              <option value="VGG Small">VGG Small</option>
              <option value="Inception Mini">Inception Mini</option>
              <option value="Two-Branch Concat">Two-Branch Concat</option>
            </select>
            <button id="mb-load-preset" class="secondary-btn">📦 Load</button>
            <button id="mb-save" class="primary-btn">💾 Save Graph</button>
            <button id="mb-close" class="secondary-btn">Close</button>
          </div>
        </div>
        <div class="mb-body" style="display:flex; gap:12px; padding:12px; height: calc(100vh - 56px);">
          <div class="mb-left" style="flex:2; display:flex; flex-direction:column; min-width:0;">
            <div class="mb-toolbar" style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:8px;">
              <button id="mb-add-input" class="secondary-btn">Add Input</button>
              <button id="mb-add-conv" class="secondary-btn">Add Conv2D</button>
              <button id="mb-add-gap" class="secondary-btn">Add GlobalAvgPool</button>
              <button id="mb-add-dense" class="secondary-btn">Add Dense</button>
              <button id="mb-add-dropout" class="secondary-btn">Add Dropout</button>
              <button id="mb-add-bn" class="secondary-btn">Add BatchNorm</button>
              <button id="mb-add-activation" class="secondary-btn">Add Activation</button>
              <button id="mb-add-pool" class="secondary-btn">Add MaxPool</button>
              <button id="mb-add-flatten" class="secondary-btn">Add Flatten</button>
              <button id="mb-add-add" class="secondary-btn">Add Add</button>
              <button id="mb-add-concat" class="secondary-btn">Add Concat</button>
            </div>
            <div id="mb-canvas" style="flex:1; border:1px dashed #374151; border-radius:8px; padding:12px; overflow:auto; background:#0b1220; min-height:360px;"></div>
          </div>
          <div class="mb-right" style="flex:1; min-width:280px;">
            <div class="mb-panel" style="border:1px solid #374151; border-radius:8px; padding:12px; height:100%; overflow:auto;">
              <h3 style="margin-top:0;">Properties</h3>
              <div class="mb-props" id="mb-props"></div>
              <div class="assign-hint" style="margin-top:8px;">Tip: Click a node to edit; use Connect to link nodes.</div>
            </div>
          </div>
        </div>
      </div>`;
    document.body.appendChild(m);
    wireModal();
  }

  function open(){
    ensureModal();
    fetch('/api/model-builder/graph').then(r=>r.json()).then(d=>{
      if (d.ok && d.graph){ state.graph = d.graph; }
      draw();
      document.getElementById('modelBuilderModal').classList.remove('hidden');
      document.body.classList.add('modal-open');
    }).catch(()=>{
      state.graph = { nodes: [], edges: [], input_shape: [224,224,3] };
      draw();
      document.getElementById('modelBuilderModal').classList.remove('hidden');
      document.body.classList.add('modal-open');
    });
  }

  function close(){
    const el = document.getElementById('modelBuilderModal');
    if (el){ el.classList.add('hidden'); document.body.classList.remove('modal-open'); }
  }

  function wireModal(){
    const root = document.getElementById('modelBuilderModal');
    if (!root) return;
    const map = [
      ['mb-add-input','Input',{}],
      ['mb-add-conv','Conv2D',{filters:32,kernel_size:3,strides:1,padding:'same',activation:'relu'}],
      ['mb-add-gap','GlobalAveragePooling2D',{}],
      ['mb-add-dense','Dense',{units:128,activation:'relu'}],
      ['mb-add-dropout','Dropout',{rate:0.2}],
      ['mb-add-bn','BatchNormalization',{}],
      ['mb-add-activation','Activation',{activation:'relu'}],
      ['mb-add-pool','MaxPool2D',{pool_size:2,strides:2,padding:'same'}],
      ['mb-add-flatten','Flatten',{}],
      ['mb-add-add','Add',{}],
      ['mb-add-concat','Concat',{}],
    ];
    map.forEach(([id, type, params])=>{
      const btn = root.querySelector('#'+id);
      if (btn){ btn.addEventListener('click', ()=>{ addNode(type, params); }); }
    });
    root.querySelector('#mb-save')?.addEventListener('click', saveGraph);
    root.querySelector('#mb-load-preset')?.addEventListener('click', ()=>{
      const sel = root.querySelector('#mb-preset');
      const name = sel ? sel.value : '';
      if (!name) return;
      // confirm overwrite if existing graph is non-empty
      const hasExisting = (state.graph.nodes && state.graph.nodes.length) || (state.graph.edges && state.graph.edges.length);
      if (hasExisting && !confirm('Replace current graph with preset "'+name+'"?')) return;
      loadPreset(name);
    });
    root.querySelector('#mb-close')?.addEventListener('click', close);
  }

  function addNode(type, params){
    const id = `${type}_${Math.random().toString(36).slice(2,7)}`;
    state.graph.nodes.push({ id, type, params: { ...params } });
    state.selectedNodeId = id;
    draw();
  }

  function removeNode(id){
    state.graph.nodes = state.graph.nodes.filter(n=>n.id!==id);
    state.graph.edges = state.graph.edges.filter(e=>e.from!==id && e.to!==id);
    if (state.selectedNodeId===id) state.selectedNodeId=null;
    draw();
  }

  function connect(from, to){
    if (from===to) return;
    if (!state.graph.edges.find(e=>e.from===from && e.to===to)){
      state.graph.edges.push({ from, to });
      draw();
    }
  }

  function draw(){
    drawSVG();
    drawProps();
  }

  function loadPreset(name){
    const spec = PRESETS[name];
    if (!spec) return;
    // Deep clone to avoid sharing references
    state.graph = JSON.parse(JSON.stringify({ nodes: spec.nodes || [], edges: spec.edges || [], input_shape: spec.input_shape || [224,224,3] }));
    // Ensure node IDs are unique and edges reference them; keep provided IDs
    draw();
  }

  // SVG editor state and helpers
  let svgCtx = null;
  function drawSVG(){
    const host = document.getElementById('mb-canvas');
    if (!host) return;
    if (!svgCtx) svgCtx = initSVG(host);

    state.graph.nodes.forEach((n,i)=>{
      n.ui = n.ui || {};
      if (typeof n.ui.x !== 'number') n.ui.x = 60 + (i % 5) * 240;
      if (typeof n.ui.y !== 'number') n.ui.y = 60 + Math.floor(i/5) * 200;
      if (typeof n.ui.w !== 'number') n.ui.w = guessWidth(n);
      if (typeof n.ui.h !== 'number') n.ui.h = 96;
      if (typeof n.ui.d !== 'number') n.ui.d = 24;
      if (!n.ui.color) n.ui.color = pickColor(n.type);
    });

    svgCtx.edges.innerHTML = '';
    svgCtx.nodes.innerHTML = '';

    // edges
    state.graph.edges.forEach(e => {
      const a = state.graph.nodes.find(n=>n.id===e.from);
      const b = state.graph.nodes.find(n=>n.id===e.to);
      if (!a || !b) return;
      const p1 = portPos(a,'out');
      const p2 = portPos(b,'in');
      const path = makeEdge(svgCtx.edges, p1, p2, e.id || (e.id = 'e_'+rand5()));
      path.classList.add('edge');
      path.addEventListener('click', evt => selectEdge(e, path, evt));
    });

    // nodes
    state.graph.nodes.forEach(n => renderNode(svgCtx.nodes, n));
  }

  function initSVG(host){
    const NS = 'http://www.w3.org/2000/svg';
    host.innerHTML = '';
    const svg = document.createElementNS(NS,'svg'); svg.id='mb-svg'; host.appendChild(svg);

    // defs
    const defs = document.createElementNS(NS,'defs'); svg.appendChild(defs);
    const marker = document.createElementNS(NS,'marker'); marker.setAttribute('id','arrow'); marker.setAttribute('markerWidth','12'); marker.setAttribute('markerHeight','12'); marker.setAttribute('refX','12'); marker.setAttribute('refY','6'); marker.setAttribute('orient','auto');
    const arrow = document.createElementNS(NS,'path'); arrow.setAttribute('d','M0,0 L12,6 L0,12 Z'); arrow.setAttribute('class','edge-arrow'); marker.appendChild(arrow); defs.appendChild(marker);

    const gridLayer = document.createElementNS(NS,'g'); svg.appendChild(gridLayer); renderGrid(gridLayer);

    const viewport = document.createElementNS(NS,'g'); viewport.setAttribute('id','viewport'); svg.appendChild(viewport);
    const edges = document.createElementNS(NS,'g'); viewport.appendChild(edges);
    const nodes = document.createElementNS(NS,'g'); viewport.appendChild(nodes);

    const view = { x:0, y:0, k:1 };
    applyView();

    // pan
    let panning=false, panStart=null;
    svg.addEventListener('mousedown', e=>{
      if (e.target.closest('.node') || e.target.closest('.port') || e.target.closest('.edge')) return;
      panning=true; panStart={x:e.clientX,y:e.clientY,vx:view.x,vy:view.y}; svg.style.cursor='grabbing';
    });
    window.addEventListener('mousemove', e=>{
      if (!panning) return; const dx=e.clientX-panStart.x; const dy=e.clientY-panStart.y; view.x=panStart.vx+dx; view.y=panStart.vy+dy; applyView();
    });
    window.addEventListener('mouseup', ()=>{ if (panning){ panning=false; svg.style.cursor='default'; } });

    // zoom
    svg.addEventListener('wheel', e=>{
      e.preventDefault(); const scaleBy=1.1; const rect=svg.getBoundingClientRect(); const mx=e.clientX-rect.left; const my=e.clientY-rect.top; const delta=e.deltaY<0?1/scaleBy:scaleBy; const k2=clamp(view.k*delta,0.2,3); const p0x=(mx-view.x)/view.k; const p0y=(my-view.y)/view.k; view.k=k2; view.x=mx-p0x*view.k; view.y=my-p0y*view.k; applyView();
    }, { passive:false });

    function applyView(){ viewport.setAttribute('transform', `translate(${view.x},${view.y}) scale(${view.k})`); }
    function svgPoint(clientX, clientY){ const pt=svg.createSVGPoint(); pt.x=clientX; pt.y=clientY; const inv=viewport.getScreenCTM().inverse(); const p=pt.matrixTransform(inv); return {x:p.x, y:p.y}; }

    let tempEdge=null;
    function beginTempEdge(p1){ if (tempEdge) tempEdge.remove(); tempEdge=document.createElementNS(NS,'path'); tempEdge.setAttribute('class','edge'); tempEdge.setAttribute('stroke-dasharray','5,5'); edges.appendChild(tempEdge); updateTempEdge(p1,p1); }
    function updateTempEdge(p1,p2){ if (tempEdge) tempEdge.setAttribute('d', edgePath(p1,p2)); }
    function endTempEdge(){ if (tempEdge){ tempEdge.remove(); tempEdge=null; } }

    function onStartConnect(fromNode){
      const p1 = portPos(fromNode,'out'); beginTempEdge(p1);
      const move = e=>{ const p=svgPoint(e.clientX,e.clientY); updateTempEdge(p1,p); };
      const up = e=>{ window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up); endTempEdge(); const target=e.target.closest('.port[data-port="in"]'); if (!target) return; const toId=target.closest('.node').getAttribute('data-id'); if (fromNode.id===toId) return; if (wouldCreateCycle(state.graph, fromNode.id, toId)) { alert('Connection would create a cycle; disallowed.'); return; } if (!state.graph.edges.find(ed=>ed.from===fromNode.id && ed.to===toId)) { state.graph.edges.push({ from: fromNode.id, to: toId, id: 'e_'+rand5() }); drawSVG(); } };
      window.addEventListener('mousemove', move); window.addEventListener('mouseup', up, { once:true });
    }

    return { svg, viewport, nodes, edges, svgPoint, onStartConnect };
  }

  function renderGrid(layer){
    const NS='http://www.w3.org/2000/svg'; layer.innerHTML=''; const g=document.createElementNS(NS,'g'); g.setAttribute('class','grid'); layer.appendChild(g); const step=20, size=2000; for (let x=-size; x<=size; x+=step){ const l=document.createElementNS(NS,'line'); l.setAttribute('x1',x); l.setAttribute('y1',-size); l.setAttribute('x2',x); l.setAttribute('y2',size); g.appendChild(l);} for (let y=-size; y<=size; y+=step){ const l=document.createElementNS(NS,'line'); l.setAttribute('x1',-size); l.setAttribute('y1',y); l.setAttribute('x2',size); l.setAttribute('y2',y); g.appendChild(l);} }

  function renderNode(layer, node){
    const NS='http://www.w3.org/2000/svg'; const {x,y,w,h,d,color}=node.ui; const g=document.createElementNS(NS,'g'); g.setAttribute('class','node'); g.setAttribute('data-id', node.id); g.style.cursor='move'; layer.appendChild(g);
    const cFront=color, cTop=shade(color,22), cSide=shade(color,-10);
    const top=document.createElementNS(NS,'polygon'); top.setAttribute('points', [[x,y],[x+d,y-d],[x+w+d,y-d],[x+w,y]].map(p=>p.join(',')).join(' ')); top.setAttribute('fill', cTop); g.appendChild(top);
    const side=document.createElementNS(NS,'polygon'); side.setAttribute('points', [[x+w,y],[x+w+d,y-d],[x+w+d,y+h-d],[x+w,y+h]].map(p=>p.join(',')).join(' ')); side.setAttribute('fill', cSide); g.appendChild(side);
    const front=document.createElementNS(NS,'rect'); front.setAttribute('class','front'); front.setAttribute('x',x); front.setAttribute('y',y); front.setAttribute('width',w); front.setAttribute('height',h); front.setAttribute('rx',6); front.setAttribute('fill', cFront); front.setAttribute('stroke','rgba(0,0,0,0.25)'); g.appendChild(front);

    // Tooltip (native title) for minimal UI
    const title=document.createElementNS(NS,'title');
    title.textContent = `${node.type}${Object.keys(node.params||{}).length? ' — '+summarizeParams(node.params):''}`;
    g.appendChild(title);
    const inPort=document.createElementNS(NS,'circle'); inPort.setAttribute('class','port'); inPort.setAttribute('data-port','in'); inPort.setAttribute('cx',x-6); inPort.setAttribute('cy',y+h/2); inPort.setAttribute('r',5); inPort.setAttribute('fill','#60a5fa'); g.appendChild(inPort);
    const outPort=document.createElementNS(NS,'circle'); outPort.setAttribute('class','port'); outPort.setAttribute('data-port','out'); outPort.setAttribute('cx',x+w+6); outPort.setAttribute('cy',y+h/2); outPort.setAttribute('r',5); outPort.setAttribute('fill','#38bdf8'); g.appendChild(outPort);

    g.addEventListener('click', (e)=>{ if (e.target.classList.contains('port')) return; state.selectedNodeId=node.id; drawProps(); document.querySelectorAll('.node').forEach(n=>n.classList.remove('mb-selected')); g.classList.add('mb-selected'); });
    g.addEventListener('mousedown', e=>{ if (e.target.classList.contains('port')) return; const start=svgCtx.svgPoint(e.clientX,e.clientY); const ox=node.ui.x, oy=node.ui.y; g.classList.add('dragging'); const move=ev=>{ const p=svgCtx.svgPoint(ev.clientX,ev.clientY); const dx=p.x-start.x; const dy=p.y-start.y; node.ui.x=snap(ox+dx,10); node.ui.y=snap(oy+dy,10); drawSVG(); }; const up=()=>{ window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up); g.classList.remove('dragging'); }; window.addEventListener('mousemove', move); window.addEventListener('mouseup', up, { once:true }); });
    outPort.addEventListener('mousedown', e=>{ e.stopPropagation(); svgCtx.onStartConnect(node); });
  }

  function portPos(node, which){ const {x,y,w,h}=node.ui; return which==='in' ? {x:x-6,y:y+h/2} : {x:x+w+6,y:y+h/2}; }
  function makeEdge(layer, p1, p2, id){ const NS='http://www.w3.org/2000/svg'; const path=document.createElementNS(NS,'path'); path.setAttribute('id',id); path.setAttribute('class','edge'); path.setAttribute('d', edgePath(p1,p2)); path.setAttribute('marker-end','url(#arrow)'); layer.appendChild(path); return path; }
  function edgePath(p1,p2){ const dx=Math.max(40, Math.abs(p2.x-p1.x)*0.4); return `M ${p1.x},${p1.y} C ${p1.x+dx},${p1.y} ${p2.x-dx},${p2.y} ${p2.x},${p2.y}`; }
  function selectEdge(edge, pathEl, evt){ evt.stopPropagation(); document.querySelectorAll('.edge').forEach(p=>p.classList.remove('selected')); pathEl.classList.add('selected'); state.selectedNodeId=null; drawProps(); }
  function summarizeParams(p){ if (!p) return ''; const keys=Object.keys(p); return keys.slice(0,3).map(k=>`${k}:${p[k]}`).join(', ') + (keys.length>3?'…':''); }
  function guessWidth(n){ const base=120; const extra=/Conv|Concat|Add|Dense/.test(n.type)?20:0; return base+extra; }
  function pickColor(type){
    // Distinct palette per type for quick recognition
    if (/Input/i.test(type)) return '#4b5563'; // slate gray
    if (/SeparableConv/i.test(type)) return '#d97706'; // amber
    if (/Conv/i.test(type)) return '#2563eb'; // blue
    if (/Pool/i.test(type)) return '#059669'; // emerald
    if (/Batch|Norm/i.test(type)) return '#9333ea'; // purple
    if (/Activation|Act/i.test(type)) return '#f43f5e'; // rose
    if (/Dropout/i.test(type)) return '#f59e0b'; // amber light
    if (/Flatten/i.test(type)) return '#ef4444'; // red
    if (/Dense/i.test(type)) return '#16a34a'; // green
    if (/Add/i.test(type)) return '#64748b'; // slate
    if (/Concat/i.test(type)) return '#0ea5e9'; // sky
    return '#334155';
  }
  function shade(hex, percent){ let f=parseInt(hex.slice(1),16), t=percent<0?0:255, p=Math.abs(percent)/100; let R=f>>16, G=f>>8&0x00FF, B=f&0x0000FF; const to=x=>Math.round((t-x)*p)+x; return '#' + (0x1000000 + (to(R)<<16) + (to(G)<<8) + (to(B))).toString(16).slice(1); }
  function snap(v, step){ return Math.round(v/step)*step; }
  function clamp(v,a,b){ return Math.max(a, Math.min(b, v)); }
  function rand5(){ return Math.random().toString(36).slice(2,7); }
  function wouldCreateCycle(graph, from, to){ const adj=new Map(); graph.nodes.forEach(n=>adj.set(n.id,[])); graph.edges.forEach(e=>adj.get(e.from)?.push(e.to)); adj.get(from).push(to); const seen=new Set(); function dfs(v,target){ if (v===target && seen.size) return true; if (seen.has(v)) return false; seen.add(v); for (const w of adj.get(v)||[]) if (dfs(w,target)) return true; return false; } seen.clear(); return dfs(from, from); }

  // Delete selected node/edge
  window.addEventListener('keydown', (e)=>{ if (e.key!=='Delete' && e.key!=='Backspace') return; const sel=state.graph.nodes.find(n=>n.id===state.selectedNodeId); if (sel){ removeNode(sel.id); return; } const selEdge=document.querySelector('.edge.selected'); if (selEdge){ const id=selEdge.getAttribute('id'); state.graph.edges=state.graph.edges.filter(e=>e.id!==id); drawSVG(); } });

  function drawProps(){
    const host = document.getElementById('mb-props');
    if (!host) return;
    host.innerHTML='';
    const cur = state.graph.nodes.find(n=>n.id===state.selectedNodeId);
    if (!cur){ host.innerHTML = '<div style="color:#9ca3af;">Select a node to edit parameters</div>'; return; }
    const wrap = document.createElement('div');
    wrap.innerHTML = `<div style="margin-bottom:6px;"><strong>${cur.type}</strong> <span style="color:#9ca3af">(${cur.id})</span></div>`;
    const params = cur.params || {};
    Object.keys(params).forEach(k=>{
      const row = document.createElement('div'); row.style.marginBottom='6px';
      row.innerHTML = `<label style="display:block; font-size:12px; color:#9ca3af;">${k}</label><input data-k="${k}" value="${params[k]}" style="width:100%;">`;
      wrap.appendChild(row);
    });
    // Add parameter
    const addRow = document.createElement('div'); addRow.style.marginTop='8px';
    addRow.innerHTML = `<input id="mb-new-key" placeholder="param name" style="width:48%"> <input id="mb-new-val" placeholder="value" style="width:48%"> <button id="mb-add-param" class="secondary-btn" style="margin-top:6px;">Add Param</button>`;
    wrap.appendChild(addRow);
    // Save handler
    wrap.querySelectorAll('input[data-k]').forEach(inp=>{
      inp.addEventListener('input', (e)=>{
        const key = e.target.getAttribute('data-k');
        cur.params[key] = coerceValue(e.target.value);
      });
    });
    wrap.querySelector('#mb-add-param')?.addEventListener('click', ()=>{
      const k = wrap.querySelector('#mb-new-key').value.trim();
      const v = coerceValue(wrap.querySelector('#mb-new-val').value.trim());
      if (!k) return; cur.params[k] = v; drawProps(); draw();
    });
    host.appendChild(wrap);
  }

  function coerceValue(v){
    if (v === '') return v;
    if (!isNaN(Number(v))) return Number(v);
    if (v === 'true') return true;
    if (v === 'false') return false;
    return v;
  }

  async function saveGraph(){
    try{
      const r = await fetch('/api/model-builder/graph', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(state.graph) });
      const d = await r.json();
      if (!d.ok){ alert('Save failed: ' + (d.msg || '')); return; }
      alert('Graph saved. Choose model: Custom (Model Builder) in Training.');
    } catch(e){ alert('Save failed: ' + e.message); }
  }

  window.modelBuilder = { open };
})();



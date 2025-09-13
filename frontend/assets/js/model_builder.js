// Simple Model Builder UI (nodes, edges, params)
// Provides: window.modelBuilder.open()

(function(){
  const state = {
    graph: { nodes: [], edges: [], input_shape: [224,224,3] },
    selectedNodeId: null,
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
          <div style="display:flex; gap:8px;">
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
    const label=document.createElementNS(NS,'text'); label.setAttribute('x',x+14); label.setAttribute('y',y+26); label.setAttribute('fill','#e5e7eb'); label.setAttribute('font-size','14'); label.textContent=node.type; g.appendChild(label);
    const params=document.createElementNS(NS,'text'); params.setAttribute('x',x+14); params.setAttribute('y',y+48); params.setAttribute('fill','#cdd5e0'); params.setAttribute('font-size','12'); params.textContent=summarizeParams(node.params); g.appendChild(params);
    const inPort=document.createElementNS(NS,'circle'); inPort.setAttribute('class','port'); inPort.setAttribute('data-port','in'); inPort.setAttribute('cx',x-8); inPort.setAttribute('cy',y+h/2); inPort.setAttribute('r',7); inPort.setAttribute('fill','#60a5fa'); g.appendChild(inPort);
    const outPort=document.createElementNS(NS,'circle'); outPort.setAttribute('class','port'); outPort.setAttribute('data-port','out'); outPort.setAttribute('cx',x+w+8); outPort.setAttribute('cy',y+h/2); outPort.setAttribute('r',7); outPort.setAttribute('fill','#38bdf8'); g.appendChild(outPort);

    g.addEventListener('click', (e)=>{ if (e.target.classList.contains('port')) return; state.selectedNodeId=node.id; drawProps(); document.querySelectorAll('.node').forEach(n=>n.classList.remove('mb-selected')); g.classList.add('mb-selected'); });
    g.addEventListener('mousedown', e=>{ if (e.target.classList.contains('port')) return; const start=svgCtx.svgPoint(e.clientX,e.clientY); const ox=node.ui.x, oy=node.ui.y; g.classList.add('dragging'); const move=ev=>{ const p=svgCtx.svgPoint(ev.clientX,ev.clientY); const dx=p.x-start.x; const dy=p.y-start.y; node.ui.x=snap(ox+dx,10); node.ui.y=snap(oy+dy,10); drawSVG(); }; const up=()=>{ window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up); g.classList.remove('dragging'); }; window.addEventListener('mousemove', move); window.addEventListener('mouseup', up, { once:true }); });
    outPort.addEventListener('mousedown', e=>{ e.stopPropagation(); svgCtx.onStartConnect(node); });
  }

  function portPos(node, which){ const {x,y,w,h}=node.ui; return which==='in' ? {x:x-6,y:y+h/2} : {x:x+w+6,y:y+h/2}; }
  function makeEdge(layer, p1, p2, id){ const NS='http://www.w3.org/2000/svg'; const path=document.createElementNS(NS,'path'); path.setAttribute('id',id); path.setAttribute('class','edge'); path.setAttribute('d', edgePath(p1,p2)); path.setAttribute('marker-end','url(#arrow)'); layer.appendChild(path); return path; }
  function edgePath(p1,p2){ const dx=Math.max(40, Math.abs(p2.x-p1.x)*0.4); return `M ${p1.x},${p1.y} C ${p1.x+dx},${p1.y} ${p2.x-dx},${p2.y} ${p2.x},${p2.y}`; }
  function selectEdge(edge, pathEl, evt){ evt.stopPropagation(); document.querySelectorAll('.edge').forEach(p=>p.classList.remove('selected')); pathEl.classList.add('selected'); state.selectedNodeId=null; drawProps(); }
  function summarizeParams(p){ if (!p) return ''; const keys=Object.keys(p); return keys.slice(0,3).map(k=>`${k}:${p[k]}`).join(', ') + (keys.length>3?'…':''); }
  function guessWidth(n){ const base=140; const extra=/Conv|Concat|Add|Dense/.test(n.type)?40:0; return base+extra; }
  function pickColor(type){ if (/Input/i.test(type)) return '#2d3748'; if (/Conv/i.test(type)) return '#1f6feb'; if (/Pool/i.test(type)) return '#0ea5e9'; if (/Dense/i.test(type)) return '#16a34a'; if (/Batch|Norm/i.test(type)) return '#7c3aed'; if (/Dropout/i.test(type)) return '#f59e0b'; if (/Flatten/i.test(type)) return '#ef4444'; if (/Add|Concat/i.test(type)) return '#71717a'; return '#334155'; }
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



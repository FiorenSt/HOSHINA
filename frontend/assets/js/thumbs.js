(function(){
  function $(sel){ return document.querySelector(sel); }

  async function getJSON(path){
    const r = await fetch(path);
    if(!r.ok) throw new Error(`GET ${path} ${r.status}`);
    return await r.json();
  }

  async function postJSON(path, body){
    const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
    if(!r.ok) throw new Error(`POST ${path} ${r.status}`);
    return await r.json();
  }

  let timer = null;
  async function refresh(){
    const el = $('#thumb-build-status');
    if(!el) return;
    try{
      const st = await getJSON('/api/thumbs/build/status');
      if(st && st.ok){
        const running = !!st.running;
        const total = st.total || 0;
        const done = st.done || 0;
        const pct = total ? Math.round(done/total*100) : 0;
        const eta = (st.eta_sec || st.eta_sec === 0) ? ` • ETA ${Math.max(0, st.eta_sec)}s` : '';
        const msg = running ? '⏳ running' : (st.message || 'idle');
        el.textContent = `${msg} • ${done}/${total} (${pct}%)${eta}`;
      } else {
        el.textContent = `⚠️ ${st && st.msg ? st.msg : 'status error'}`;
      }
    }catch(e){
      el.textContent = `⚠️ ${e.message}`;
    }
  }

  async function start(){
    try{
      const r = await postJSON('/api/thumbs/build/start', { mode: 'composite', size: 256, only_missing: true });
      if(!r.ok){ alert(r.msg || 'Failed to start'); return; }
      if(timer) clearInterval(timer);
      timer = setInterval(refresh, 1000);
      refresh();
    }catch(e){ alert(e.message); }
  }

  async function cancel(){
    try{
      const r = await postJSON('/api/thumbs/build/cancel', {});
      if(!r.ok){ alert(r.msg || 'Failed to cancel'); return; }
      if(timer) clearInterval(timer);
      refresh();
    }catch(e){ alert(e.message); }
  }

  function init(){
    const startBtn = $('#start-thumb-build');
    const cancelBtn = $('#cancel-thumb-build');
    if(startBtn) startBtn.addEventListener('click', start);
    if(cancelBtn) cancelBtn.addEventListener('click', cancel);
    const statusEl = $('#thumb-build-status');
    if(statusEl){ refresh(); timer = setInterval(refresh, 3000); }
  }

  if(document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();



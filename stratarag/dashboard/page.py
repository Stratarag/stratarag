"""Single-file playground page. No external assets — works fully offline."""

PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>stratarag playground</title>
<style>
:root{
  --paper:#F3F5F9; --card:#FFFFFF; --ink:#141B29; --ink-soft:#5A6478;
  --line:#D9DEE9; --indigo:#3D4FC4; --indigo-soft:#E4E7FA;
  --recall:#B07C00; --recall-soft:#FBF3DC; --ok:#1F7A4D; --warn:#B3423A;
  --mono:ui-monospace,'SF Mono',Menlo,Consolas,monospace;
  --sans:'Avenir Next','Segoe UI',system-ui,sans-serif;
}
*{box-sizing:border-box;margin:0}
body{background:var(--paper);color:var(--ink);font-family:var(--sans);
  background-image:linear-gradient(var(--line) 1px,transparent 1px),
    linear-gradient(90deg,var(--line) 1px,transparent 1px);
  background-size:28px 28px;background-position:-1px -1px;min-height:100vh}
header{display:flex;align-items:baseline;gap:14px;padding:20px 28px 12px}
.wordmark{font-family:Iowan Old Style,Palatino,Georgia,serif;font-size:26px;
  letter-spacing:.5px}
.wordmark b{color:var(--indigo)}
.tag{font-family:var(--mono);font-size:12px;color:var(--ink-soft)}
.user{margin-left:auto;font-family:var(--mono);font-size:12px;color:var(--ink-soft)}
.user input{font-family:var(--mono);font-size:12px;border:1px solid var(--line);
  border-radius:6px;padding:4px 8px;background:var(--card);color:var(--ink);width:110px}
main{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(280px,1fr);
  gap:18px;padding:0 28px 28px;max-width:1200px}
@media(max-width:860px){main{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;
  box-shadow:0 1px 2px rgba(20,27,41,.05)}
#chat{display:flex;flex-direction:column;min-height:70vh}
#log{flex:1;overflow-y:auto;padding:18px;display:flex;flex-direction:column;gap:14px}
.msg{max-width:92%;line-height:1.5;font-size:15px}
.msg.you{align-self:flex-end;background:var(--indigo);color:#fff;
  padding:10px 14px;border-radius:12px 12px 3px 12px}
.msg.agent{align-self:flex-start;width:92%}
.bubble{background:var(--paper);border:1px solid var(--line);
  padding:12px 14px;border-radius:12px 12px 12px 3px}
.bubble.gated{border-color:var(--warn);background:#FBEFEE}
.recall{margin-top:8px;border-left:3px solid var(--recall);
  background:var(--recall-soft);border-radius:0 8px 8px 0;
  padding:8px 12px;font-size:12.5px}
.recall h4{font-family:var(--mono);font-size:10.5px;text-transform:uppercase;
  letter-spacing:.12em;color:var(--recall);margin-bottom:5px}
.pill{display:inline-block;background:#fff;border:1px solid var(--line);
  border-radius:999px;padding:2px 9px;margin:2px 3px 2px 0;font-size:12px}
.src{margin:5px 0;padding:6px 8px;background:#fff;border:1px solid var(--line);
  border-radius:6px;font-size:12px;color:var(--ink-soft)}
.src b{color:var(--ink);font-family:var(--mono);font-size:11px}
.gauge{display:inline-flex;align-items:center;gap:6px;font-family:var(--mono);
  font-size:11px;margin-top:6px}
.gauge .bar{width:90px;height:6px;background:var(--line);border-radius:3px;overflow:hidden}
.gauge .fill{height:100%;border-radius:3px}
form{display:flex;gap:10px;padding:14px;border-top:1px solid var(--line)}
form input{flex:1;font-size:15px;padding:11px 14px;border:1px solid var(--line);
  border-radius:9px;background:var(--paper);color:var(--ink)}
form input:focus,button:focus,.user input:focus{outline:2px solid var(--indigo);outline-offset:1px}
button{font-size:14px;font-weight:600;padding:11px 20px;border:0;border-radius:9px;
  background:var(--indigo);color:#fff;cursor:pointer}
button:disabled{opacity:.5;cursor:wait}
aside .panel{margin-bottom:18px;padding:16px}
aside h3{font-family:var(--mono);font-size:11px;text-transform:uppercase;
  letter-spacing:.14em;color:var(--ink-soft);margin-bottom:10px}
#trace{font-family:var(--mono);font-size:12px}
.trow{display:flex;justify-content:space-between;gap:8px;padding:5px 0;
  border-bottom:1px dashed var(--line)}
.trow:last-child{border-bottom:0}
.trow .ms{color:var(--ink-soft)}
.stage-tool{color:var(--recall)} .stage-llm{color:var(--indigo)}
.empty{color:var(--ink-soft);font-size:13px;font-style:italic}
#remember{display:flex;gap:8px;margin-top:10px}
#remember input{flex:1;font-size:13px;padding:7px 10px;border:1px solid var(--line);
  border-radius:7px;background:var(--paper);color:var(--ink)}
#remember button{padding:7px 12px;font-size:13px;background:var(--recall)}
@media(prefers-reduced-motion:no-preference){
  .msg{animation:rise .18s ease-out}
  @keyframes rise{from{opacity:0;transform:translateY(4px)}to{opacity:1}}
}
</style>
</head>
<body>
<header>
  <div class="wordmark"><b>stratarag</b> playground</div>
  <div class="tag">agents that remember</div>
  <label class="user">user_id <input id="uid" value="default" aria-label="user id"></label>
</header>
<main>
  <section id="chat" class="panel" aria-label="chat">
    <div id="log"></div>
    <form id="f">
      <input id="q" placeholder="Ask the agent…" autocomplete="off" aria-label="message">
      <button id="send" type="submit">Send</button>
    </form>
  </section>
  <aside>
    <div class="panel">
      <h3>Semantic memory · this user</h3>
      <div id="facts" class="empty">Nothing learned yet.</div>
      <div id="remember">
        <input id="fact" placeholder="Teach it a fact…" aria-label="fact">
        <button type="button" id="teach">Remember</button>
      </div>
    </div>
    <div class="panel">
      <h3>Last run · trace</h3>
      <div id="trace" class="empty">Send a message to see the pipeline.</div>
    </div>
  </aside>
</main>
<script>
const $=id=>document.getElementById(id);
const log=$('log'), factsBox=$('facts'), traceBox=$('trace');
const esc=s=>s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let knownFacts=[];

function addYou(text){
  const d=document.createElement('div');d.className='msg you';d.textContent=text;
  log.appendChild(d);log.scrollTop=log.scrollHeight;
}
function gaugeColor(c){return c>=.66?'var(--ok)':c>=.33?'var(--recall)':'var(--warn)';}
function addAgent(r){
  const d=document.createElement('div');d.className='msg agent';
  let recall='';
  const facts=(r.memory&&r.memory.semantic)||[];
  const eps=(r.memory&&r.memory.episodic)||[];
  if(facts.length||eps.length){
    recall+='<div class="recall"><h4>Recalled</h4>'
      + facts.map(f=>'<span class="pill">'+esc(f)+'</span>').join('')
      + eps.map(e=>'<span class="pill">↺ '+esc(e.slice(0,70))+'</span>').join('')
      + '</div>';
  }
  let sources='';
  if(r.sources&&r.sources.length){
    sources='<div class="recall" style="border-color:var(--indigo);background:var(--indigo-soft)">'
      +'<h4 style="color:var(--indigo)">Sources</h4>'
      + r.sources.map((s,i)=>'<div class="src"><b>['+(i+1)+(s.section?' · '+esc(s.section):'')
        +']</b> '+esc(s.text.slice(0,180))+(s.text.length>180?'…':'')+'</div>').join('')
      +'</div>';
  }
  const pct=Math.round((r.confidence||0)*100);
  d.innerHTML='<div class="bubble'+(r.gated?' gated':'')+'">'+esc(r.answer)+'</div>'
    +'<div class="gauge">confidence <span class="bar"><span class="fill" style="width:'
    +pct+'%;background:'+gaugeColor(r.confidence)+'"></span></span> '+pct+'%'
    +(r.gated?' · <span style="color:var(--warn)">gated</span>':'')+'</div>'
    +recall+sources;
  log.appendChild(d);log.scrollTop=log.scrollHeight;
}
function renderFacts(){
  factsBox.classList.toggle('empty',!knownFacts.length);
  factsBox.innerHTML=knownFacts.length
    ? knownFacts.map(f=>'<span class="pill">'+esc(f)+'</span>').join('')
    : 'Nothing learned yet.';
}
function renderTrace(trace){
  traceBox.classList.remove('empty');
  traceBox.innerHTML=(trace||[]).map(t=>{
    const cls=t.stage.startsWith('tool')?'stage-tool':(t.stage==='llm'?'stage-llm':'');
    return '<div class="trow"><span class="'+cls+'">'+esc(t.stage)
      +'</span><span class="ms">'+(t.ms||0).toFixed(1)+' ms</span></div>';
  }).join('')||'<span class="empty">no trace</span>';
}
async function post(path,body){
  const res=await fetch(path,{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await res.json();
  if(!res.ok)throw new Error(data.error||res.status);
  return data;
}
$('f').addEventListener('submit',async e=>{
  e.preventDefault();
  const q=$('q').value.trim();if(!q)return;
  $('q').value='';addYou(q);$('send').disabled=true;
  try{
    const r=await post('/api/chat',{message:q,user_id:$('uid').value||'default'});
    addAgent(r);renderTrace(r.trace);
    (r.memory&&r.memory.semantic||[]).forEach(f=>{
      if(!knownFacts.includes(f))knownFacts.push(f);});
    renderFacts();
  }catch(err){
    const d=document.createElement('div');d.className='msg agent';
    d.innerHTML='<div class="bubble gated">Error: '+esc(err.message)+'</div>';
    log.appendChild(d);
  }finally{$('send').disabled=false;$('q').focus();}
});
$('teach').addEventListener('click',async()=>{
  const fact=$('fact').value.trim();if(!fact)return;
  try{
    await post('/api/remember',{fact,user_id:$('uid').value||'default'});
    if(!knownFacts.includes(fact))knownFacts.push(fact);
    renderFacts();$('fact').value='';
  }catch(err){alert(err.message);}
});
$('q').focus();
</script>
</body>
</html>"""

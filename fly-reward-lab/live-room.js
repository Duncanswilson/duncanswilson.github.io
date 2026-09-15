/* Latest real server states only. No replay clock or procedural body motion. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const canvas = $('room-canvas');
  const ctx = canvas.getContext('2d');
  const room = document.querySelector('.room');
  const state = {clean:false};
  const colors=['#69d8bc','#eda0a2','#d8c184'];
  const TAU=Math.PI*2;
  const clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));
  const finite=v=>typeof v==='number'&&Number.isFinite(v);
  let latest=null,metadata=null,connection=null,client=null,drawQueued=false;
  let width=0,height=0,pixelRatio=1;
  const background=document.createElement('canvas');background.width=1440;background.height=1300;
  const bg=background.getContext('2d');
  const backend=window.FlyLiveConfig?.endpoint||window.location.origin;

  function polygon(c, points, fill, stroke) {
    c.beginPath(); c.moveTo(...points[0]);
    for (let i = 1; i < points.length; i++) c.lineTo(...points[i]);
    c.closePath();
    if (fill) {c.fillStyle = fill; c.fill();}
    if (stroke) {c.strokeStyle = stroke; c.stroke();}
  }
  function line(c, x1, y1, x2, y2, color, weight = 1) {
    c.beginPath(); c.moveTo(x1,y1); c.lineTo(x2,y2); c.strokeStyle = color; c.lineWidth = weight; c.stroke();
  }
  function rounded(c, x,y,w,h,r, fill, stroke) {
    c.beginPath(); c.roundRect(x,y,w,h,r);
    if (fill) {c.fillStyle=fill;c.fill();}
    if (stroke) {c.strokeStyle=stroke;c.stroke();}
  }
  function ellipse(c, x,y,rx,ry,fill) {
    c.beginPath(); c.ellipse(x,y,rx,ry,0,0,TAU); c.fillStyle=fill; c.fill();
  }
  function text(c, value, x,y,size,color,font='sans-serif') {
    c.font=`${size}px ${font}`;c.fillStyle=color;c.fillText(value,x,y);
  }
  function linear(c,x,y,xx,yy, stops) {
    const g=c.createLinearGradient(x,y,xx,yy);stops.forEach(([p,col])=>g.addColorStop(p,col));return g;
  }

  function buildRoom() {
    bg.fillStyle='#a9b5a8';bg.fillRect(0,0,1440,1300);
    // Three walls converge on the large, softly lit rear wall.
    polygon(bg,[[0,0],[250,55],[250,365],[0,552]],linear(bg,0,0,250,500,[[0,'#9ba99c'],[1,'#aebaae']]));
    polygon(bg,[[1270,55],[1440,0],[1440,530],[1270,365]],linear(bg,1250,0,1440,450,[[0,'#9ca999'],[1,'#83978b']]));
    bg.fillStyle=linear(bg,200,70,1240,460,[[0,'#c5cbb9'],[.55,'#c2cbbd'],[1,'#9fac9e']]);
    bg.fillRect(250,55,1020,310);
    polygon(bg,[[0,552],[250,365],[1270,365],[1440,530],[1440,1300],[0,1300]],linear(bg,800,365,700,1180,[[0,'#bcc5b3'],[.5,'#d1d1bc'],[1,'#aeb9a5']]));
    // A broad light shaft and perspective floor seams place every object on the ground.
    polygon(bg,[[440,365],[780,365],[1140,1150],[150,1150]],'#f6ebc917');
    const vanishing={x:760,y:240};
    for(let x=-2500;x<4500;x+=345){
      const at=(yy)=>vanishing.x+(x-vanishing.x)*(yy-vanishing.y)/(1300-vanishing.y);
      line(bg,at(368),368,x,1300,'#72816d24',1);
    }
    [390,423,469,532,620,747,941,1230].forEach(y=>line(bg,0,y,1440,y,'#7b897422',1));
    // Edge molding; the warm line beneath it is reflected on the floor.
    line(bg,250,365,1270,365,'#758772',6);
    line(bg,250,360,1270,360,'#e2e4c7',2);
    line(bg,0,552,250,365,'#899880',5);
    line(bg,1270,365,1440,530,'#657e6e',5);
    const light=bg.createLinearGradient(0,366,0,410);light.addColorStop(0,'#eaf1c439');light.addColorStop(1,'#f2e9cc00');bg.fillStyle=light;bg.fillRect(250,366,1020,45);
    // Recessed light panel, a quiet wall vent, and material seams.
    rounded(bg,470,105,400,154,5,'#839b8c','#d4d8bc');
    rounded(bg,480,113,380,136,3,linear(bg,480,113,860,249,[[0,'#e1e5c7'],[.5,'#cadbc5'],[1,'#b7c7b1']]));
    const glow=bg.createRadialGradient(650,184,10,650,184,255);glow.addColorStop(0,'#e2f0cf28');glow.addColorStop(1,'#edf8d300');bg.fillStyle=glow;bg.fillRect(390,0,570,430);
    for(let i=0;i<6;i++)line(bg,495+i*69,115,495+i*69,247,'#7e9a8233',1);
    line(bg,481,181,859,181,'#a5b39b77');
    rounded(bg,930,255,173,54,2,'#92a591','#bbc4b0');
    for(let i=0;i<7;i++)line(bg,947,265+i*5,1087,265+i*5,'#6d837055',2);
    text(bg,'01',1120,306,45,'#81947e','Georgia');
    text(bg,'SIGNAL ROOM',1122,324,8,'#6a806b');
    line(bg,300,70,300,351,'#f1f0d329');
    line(bg,1240,70,1240,351,'#6e876729');
    // Reproducible fine material speckle; no per-frame random flicker.
    let seed=31817;function noise(){seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;}
    for(let i=0;i<6000;i++){const x=noise()*1440,y=noise()*1300;bg.fillStyle=noise()>.5?'#183c2010':'#ffffe915';bg.fillRect(x,y,1,1);}
    // Wall-mounted cable tray with a subtle cast shadow.
    line(bg,310,338,410,338,'#59766125',8);line(bg,310,334,410,334,'#869a84',3);
  }

  function drawDevice(x,y, signal, t, mobile) {
    const s=mobile?.78:1;
    ctx.save();ctx.translate(x,y);ctx.scale(s,s);
    const shadow=ctx.createRadialGradient(100,151,5,100,151,175);shadow.addColorStop(0,'#273e2b50');shadow.addColorStop(1,'#33472d00');
    ctx.save();ctx.translate(0,123);ctx.scale(1,.3);ellipse(ctx,105,80,170,120,shadow);ctx.restore();
    // Three glass cartridges sit in sockets on the stimulation unit.
    const values=[signal.dopamine,signal.npf,signal.rate/Math.max(1,metadata?.config?.max_rate||100)];
    const labels=['DA','NPF','DRIVE'];
    for(let i=0;i<3;i++){
      const bx=24+i*62;
      rounded(ctx,bx,-95,38,103,8,'#6b827578','#cad4b3');
      const level=12+values[i]*65;
      rounded(ctx,bx+5,3-level,28,level,4,colors[i]+'b8');
      rounded(ctx,bx+4,-91,30,87,6,linear(ctx,bx,-70,bx+32,-70,[[0,'#e9f9d640'],[.5,'#e6f7dd08'],[1,'#142e2930']]));
      line(ctx,bx+8,-84,bx+8,-17,'#edf7de7a',2);
      rounded(ctx,bx-1,-100,40,13,3,'#354e45','#778d76');
      rounded(ctx,bx-1,-4,40,13,3,'#657961','#b6c0a1');
      text(ctx,labels[i],bx+7,-107,8,'#536c55');
      // Exposed leads from each cartridge into the box.
      ctx.beginPath();ctx.moveTo(bx+18,-100);ctx.bezierCurveTo(bx+18,-123,bx+53,-123,bx+53,-16);ctx.strokeStyle='#536d594f';ctx.lineWidth=3;ctx.stroke();
    }
    polygon(ctx,[[0,9],[185,9],[213,29],[28,29]],'#b8c5a9','#667b63');
    polygon(ctx,[[185,9],[213,29],[213,142],[185,126]],'#5f7965','#6d856e');
    rounded(ctx,0,27,193,117,5,linear(ctx,0,20,0,140,[[0,'#a5b496'],[1,'#8b9f82']]),'#596f59');
    rounded(ctx,12,43,167,47,3,'#27493e','#c4caae');
    text(ctx,'LIVE MODEL SIGNALS',21,57,6,'#a0bfa3');
    text(ctx,`${signal.rate.toFixed(0).padStart(3,'0')}`,21,79,21,'#bee7bd','monospace');
    text(ctx,'Hz / DA',67,78,8,'#88aa89','monospace');
    for(let j=0;j<15;j++){
      const a=j/15<signal.dopamine?1:.2;ctx.fillStyle=`rgba(142,218,167,${a*(.3+signal.dopamine*.7)})`;ctx.fillRect(121+j*3,77-(j%5+2)*2,1.5,(j%5+2)*2);
    }
    text(ctx,'MODEL INPUTS',14,113,8,'#3b5742');
    for(let i=0;i<3;i++){
      ellipse(ctx,125+i*24,111,6,6,'#354e3b');ellipse(ctx,125+i*24,111,3.5,3.5,colors[i]);
      text(ctx,labels[i],116+i*24,131,5,'#40583e');
    }
    [[7,35],[184,35],[7,137],[184,137]].forEach(([sx,sy])=>{ellipse(ctx,sx,sy,2,2,'#607657');line(ctx,sx-1,sy,sx+1,sy,'#c6d2b5');});
    rounded(ctx,10,144,25,8,2,'#526a52');rounded(ctx,161,144,25,8,2,'#526a52');
    ctx.restore();
    return colors.map((color,i)=>({x:x+(125+i*24)*s,y:y+111*s,color}));
  }

  function bezier(p0,p1,p2,p3,u) {
    const v=1-u;return{x:v*v*v*p0.x+3*v*v*u*p1.x+3*v*u*u*p2.x+u*u*u*p3.x,y:v*v*v*p0.y+3*v*v*u*p1.y+3*v*u*u*p2.y+u*u*u*p3.y};
  }
  function drawWire(start,end,index,signal,t) {
    const value=index===0?signal.dopamine:index===1?signal.npf:signal.rate/Math.max(1,metadata?.config?.max_rate||100);
    const c1={x:start.x+95+index*38,y:start.y+165+index*23};
    const c2={x:end.x-166-index*33,y:end.y+175+index*30};
    const path=(dy)=>{ctx.beginPath();ctx.moveTo(start.x,start.y+dy);ctx.bezierCurveTo(c1.x,c1.y+dy,c2.x,c2.y+dy,end.x,end.y+dy);};
    ctx.lineCap='round';path(10);ctx.strokeStyle='#33482e1c';ctx.lineWidth=9;ctx.stroke();
    path(0);ctx.strokeStyle='#394f43';ctx.lineWidth=7;ctx.stroke();
    path(-1);ctx.strokeStyle=colors[index]+'b0';ctx.lineWidth=3;ctx.stroke();
    path(-1);ctx.strokeStyle=colors[index]+'35';ctx.lineWidth=1;ctx.stroke();
    // Static cable colors indicate current signals; no travelling pulse clock.
    const dx=end.x-c2.x,dy=end.y-c2.y,rot=Math.atan2(dy,dx);
    ctx.save();ctx.translate(end.x,end.y);ctx.rotate(rot);rounded(ctx,-12,-4,15,8,3,'#aebfaf','#5e7561');rounded(ctx,-8,-3,5,6,1,colors[index]);ctx.restore();
  }


  function drawScene() {
    ctx.setTransform(pixelRatio,0,0,pixelRatio,0,0);
    ctx.fillStyle='#a9b5a8';ctx.fillRect(0,0,width,height);
    const mobile=width<560;
    const s=Math.max(width/1440,Math.min(height/900,width/950));
    const ox=(width-1440*s)/2,oy=height*.60-560*s;
    ctx.save();ctx.translate(ox,oy);ctx.scale(s,s);ctx.drawImage(background,0,0);
    if(!latest){ctx.restore();return;}
    const signal={dopamine:latest.dopamine,npf:latest.npf,rate:latest.dopamine_rate_hz};
    const fx=mobile?828:870,fy=mobile?727:596,fs=mobile?1.15:1.28;
    const strength=clamp((signal.dopamine+signal.npf)/2,0,1);
    const halo=ctx.createRadialGradient(fx,fy+90,12,fx,fy+90,300);
    halo.addColorStop(0,`rgba(216,231,162,${.08+.1*strength})`);halo.addColorStop(1,'#d8e7a200');
    ctx.save();ctx.translate(fx,fy+80);ctx.scale(1,.44);ctx.translate(-fx,-fy-80);ellipse(ctx,fx,fy+80,320,250,halo);ctx.restore();
    const inputs=drawDevice(mobile?405:280,mobile?612:486,signal,0,mobile);
    ctx.save();ctx.translate(fx,fy);ctx.scale(fs,fs);
    // There is deliberately no time/motion argument: each body transform comes
    // exclusively from the newest validated server motorPose.
    const pose=FlyActor.draw(ctx,{motorPose:latest.motorPose,dopamine:latest.dopamine,npf:latest.npf});
    ctx.restore();
    pose.ports.forEach((port,i)=>drawWire(inputs[i],{x:fx+port.x*fs,y:fy+port.y*fs},i,signal,0));
    const shade=ctx.createLinearGradient(0,880,0,1300);shade.addColorStop(0,'#344e3600');shade.addColorStop(1,'#344e3620');ctx.fillStyle=shade;ctx.fillRect(0,880,1440,420);
    ctx.restore();
  }

  function scheduleDraw(){
    if(drawQueued)return;
    drawQueued=true;
    requestAnimationFrame(()=>{drawQueued=false;drawScene();});
  }

  function duration(seconds){
    if(!finite(seconds))return '—';
    const whole=Math.max(0,Math.floor(seconds)),days=Math.floor(whole/86400),hours=Math.floor(whole%86400/3600),minutes=Math.floor(whole%3600/60),s=whole%60;
    return (days?`${days}d `:'')+(days||hours?`${hours}h `:'')+`${minutes}m ${s}s`;
  }
  function updateReadout(){
    if(!latest)return;
    $('dopamine-value').textContent=latest.dopamine.toFixed(3);
    $('npf-value').textContent=latest.npf.toFixed(3);
    $('rate-value').textContent=latest.motor_mean_hz.toFixed(2);
    $('dopamine-meter').style.width=`${clamp(latest.dopamine,0,1)*100}%`;
    $('npf-meter').style.width=`${clamp(latest.npf,0,1)*100}%`;
    $('sim-time').textContent=latest.sim_time.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})+' s';
    $('step-value').textContent=latest.step.toLocaleString('en-US');
    $('realtime-factor').textContent=finite(latest.realtime_factor)?latest.realtime_factor.toFixed(3)+'×':'—';
    $('uptime-value').textContent=duration(latest.uptime_seconds);
    $('server-clock').textContent=new Date(latest.server_time*1000).toLocaleTimeString();

  }

  function modelRecord(){
    $('model-record').textContent=JSON.stringify({run_id:latest?.run_id??metadata?.run_id??null,stream_id:latest?.stream_id??null,
      model:metadata?.model??null,config:metadata?.config??null,mechanics:metadata?.mechanics??null,
      assumptions:metadata?.assumptions??null,recorded_neuron_count:metadata?.recorded_neuron_count??null,
      mapped_neuron_count:metadata?.mapped_neuron_count??null},null,2);
  }
  function updateConnection(value){
    connection=value;
    document.body.dataset.livePhase=value.phase;
    const names={connecting:'Connecting',connected:'Live',stale:'Stale',disconnected:'Disconnected',closed:'Closed'};
    $('connection-badge').textContent=names[value.phase]||value.phase;
    $('header-status').textContent=({connecting:'Connecting to simulation',connected:'Receiving continuous motor output',
      stale:'Simulation stream is stale',disconnected:'Disconnected from simulation',closed:'Live connection closed'})[value.phase]||value.phase;
    $('signal-state').textContent=value.phase==='connected'?'LIVE':value.phase==='stale'?'STALE':latest?'LAST STATE':'WAITING';
    let note;
    if(value.phase==='connected')note='Receiving advancing server states.';
    else if(value.phase==='stale')note=latest?`No simulated-time progress for ${(value.progressAgeMs/1000).toFixed(1)} seconds. Last received pose held.`:'No advancing simulation state has arrived.';
    else if(value.phase==='disconnected')note=latest?'Connection lost. Last received pose held while reconnecting.':'The live backend is not connected; no simulation state is displayed.';
    else if(value.phase==='closed')note='The live connection is closed.';
    else note=latest?'Snapshot received; waiting for advancing stream states.':'Waiting for the first valid server state.';
    if(value.resumedAt!==null&&Date.now()-value.resumedAt<30000)note+=' The backend resumed from a checkpoint.';
    if(value.error)note+=' '+value.error;
    $('connection-note').textContent=note;
    const age=value.lastReceivedAt===null?null:Math.max(0,(Date.now()-value.lastReceivedAt)/1000);
    $('last-received').textContent=age===null?'No state received':age<1?'Just received':`Last received ${age.toFixed(0)}s ago`;
    if(!latest)$('loading').firstElementChild.textContent=value.phase==='disconnected'?'Live backend unavailable. Waiting to reconnect…':'Waiting for the first server state…';
  }

  function resize(){
    const bounds=canvas.getBoundingClientRect();width=bounds.width;height=bounds.height;pixelRatio=Math.min(window.devicePixelRatio||1,2);
    canvas.width=Math.round(width*pixelRatio);canvas.height=Math.round(height*pixelRatio);scheduleDraw();
  }
  function unavailable(error){
    document.body.dataset.livePhase='unavailable';$('connection-badge').textContent='Unavailable';
    $('header-status').textContent='Live viewer unavailable';$('connection-note').textContent=error.message||String(error);
    $('loading').firstElementChild.textContent='Live state could not be loaded. '+(error.message||String(error));
  }

  if(!ctx||!bg||!window.FlyActor||!window.FlyLive){unavailable(new Error('Required live viewer assets did not load.'));return;}
  $('scene-mode').addEventListener('click',()=>{state.clean=!state.clean;room.classList.toggle('clean',state.clean);$('scene-mode').setAttribute('aria-pressed',String(state.clean));$('scene-mode').textContent=state.clean?'Show labels ↙':'Hide labels ↗';scheduleDraw();});
  buildRoom();resize();new ResizeObserver(resize).observe(room);
  try{
    const base=new URL(backend,window.location.href);base.pathname=base.pathname.replace(/\/+$/,'')+'/';
    $('backend-endpoint').textContent='Backend: '+base.href;
    client=FlyLive.connect(backend,{
      staleAfterMs:window.FlyLiveConfig?.staleAfterMs??15000,
      onMeta(meta){metadata=meta;
        const recorded=meta.recorded_neuron_count,mapped=meta.mapped_neuron_count;
        $('motor-counts').textContent=Number.isInteger(recorded)&&Number.isInteger(mapped)?`${recorded} MOTOR NEURONS · ${mapped} MAPPED TO LEG JOINTS`:'MOTOR COUNTS NOT PROVIDED';
        modelRecord();updateReadout();scheduleDraw();},
      onState(frame,receipt){
        latest=frame;
        $('loading').hidden=true;updateReadout();if(receipt.newRun||receipt.resumed)modelRecord();scheduleDraw();},
      onStatus:updateConnection,
    });
  }catch(error){unavailable(error);}
  window.addEventListener('beforeunload',()=>client?.close());
})();

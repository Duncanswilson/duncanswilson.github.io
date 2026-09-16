/* A one-bit canvas. Every fly pose still comes directly from the live server. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const canvas = $('room-canvas');
  const ctx = canvas.getContext('2d', {alpha:false, willReadFrequently:true});
  const room = document.querySelector('.room');
  const clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));
  let latest=null,metadata=null,client=null,drawQueued=false;
  let width=0,height=0;
  const background=document.createElement('canvas');
  const bg=background.getContext('2d');
  const backend=window.FlyLiveConfig?.endpoint||window.location.origin;
  const bayer=[0,48,12,60,3,51,15,63,32,16,44,28,35,19,47,31,
    8,56,4,52,11,59,7,55,40,24,36,20,43,27,39,23,
    2,50,14,62,1,49,13,61,34,18,46,30,33,17,45,29,
    10,58,6,54,9,57,5,53,42,26,38,22,41,25,37,21];
  const patterns=new Map();

  function pattern(coverage) {
    if(patterns.has(coverage))return patterns.get(coverage);
    const tile=document.createElement('canvas');tile.width=8;tile.height=8;
    const c=tile.getContext('2d');c.fillStyle='#fff';c.fillRect(0,0,8,8);c.fillStyle='#000';
    bayer.forEach((value,i)=>{if(value<coverage*64)c.fillRect(i%8,Math.floor(i/8),1,1);});
    const result=ctx.createPattern(tile,'repeat');patterns.set(coverage,result);return result;
  }
  function polygon(c,points,fill,stroke='#000') {
    c.beginPath();c.moveTo(...points[0]);points.slice(1).forEach(p=>c.lineTo(...p));c.closePath();
    if(fill){c.fillStyle=fill;c.fill();}
    if(stroke){c.strokeStyle=stroke;c.lineWidth=1;c.stroke();}
  }
  function line(c,x1,y1,x2,y2,color='#000',weight=1) {
    c.beginPath();c.moveTo(Math.round(x1)+.5,Math.round(y1)+.5);c.lineTo(Math.round(x2)+.5,Math.round(y2)+.5);
    c.strokeStyle=color;c.lineWidth=weight;c.stroke();
  }
  function box(c,x,y,w,h,fill='#fff',stroke='#000') {
    x=Math.round(x);y=Math.round(y);w=Math.round(w);h=Math.round(h);
    if(fill){c.fillStyle=fill;c.fillRect(x,y,w,h);}
    if(stroke){c.strokeStyle=stroke;c.lineWidth=1;c.strokeRect(x+.5,y+.5,w,h);}
  }
  function bitmapText(c,value,x,y,size=8,color='#000') {
    c.font=`${size}px Monaco, monospace`;c.textBaseline='alphabetic';c.fillStyle=color;
    c.fillText(value,Math.round(x),Math.round(y));
  }

  function buildRoom() {
    background.width=width;background.height=height;
    const left=Math.round(width*.15),right=Math.round(width*.84),horizon=Math.round(height*.42);
    const edge=horizon+Math.round(height*.14);
    bg.fillStyle='#fff';bg.fillRect(0,0,width,height);
    polygon(bg,[[0,0],[left,0],[left,horizon],[0,edge]],pattern(.0625));
    polygon(bg,[[right,0],[width,0],[width,edge],[right,horizon]],pattern(.125));
    polygon(bg,[[0,edge],[left,horizon],[right,horizon],[width,edge],[width,height],[0,height]],pattern(.03125));
    line(bg,left,0,left,horizon);line(bg,right,0,right,horizon);
    line(bg,left,horizon,right,horizon);
    line(bg,left,horizon-3,right,horizon-3);
    line(bg,0,edge-3,left,horizon-3);line(bg,right,horizon-3,width,edge-3);
    // Perspective floor lines, interrupted by the ordered one-bit screen.
    bg.save();bg.beginPath();bg.moveTo(0,edge);bg.lineTo(left,horizon);bg.lineTo(right,horizon);
    bg.lineTo(width,edge);bg.lineTo(width,height);bg.lineTo(0,height);bg.closePath();bg.clip();
    for(let x=-width*2;x<width*3;x+=Math.max(45,width*.2))
      line(bg,width*.51,horizon-35,x,height,'#b8b8b8');
    [.06,.17,.35,.61,.94].forEach(t=>line(bg,0,horizon+(height-horizon)*t,width,horizon+(height-horizon)*t,'#b8b8b8'));
    bg.restore();
    // A punched-metal wall panel and the observation window.
    const wx=Math.round(width*.30),wy=Math.round(height*.10),ww=Math.round(width*.28),wh=Math.round(height*.19);
    box(bg,wx+3,wy+3,ww,wh,pattern(.5));box(bg,wx,wy,ww,wh);
    box(bg,wx+4,wy+4,ww-8,wh-8,pattern(.0625));
    line(bg,wx+ww/2,wy+4,wx+ww/2,wy+wh-4);
    line(bg,wx+4,wy+wh/2,wx+ww-4,wy+wh/2);
    const vx=Math.round(width*.66),vy=Math.round(height*.26),vw=Math.round(width*.13);
    box(bg,vx,vy,vw,15);
    for(let y=vy+3;y<vy+14;y+=3)line(bg,vx+3,y,vx+vw-3,y);
    bitmapText(bg,'01',right-16,horizon-9,11);
    if(width>390)bitmapText(bg,'OBSERVATION ROOM',left+7,horizon-9,7);
  }

  function drawDevice(x,y,signal,scale) {
    ctx.save();ctx.translate(Math.round(x),Math.round(y));ctx.scale(scale,scale);
    polygon(ctx,[[0,48],[90,48],[104,57],[17,57]],pattern(.125));
    polygon(ctx,[[90,48],[104,57],[104,106],[90,98]],pattern(.5));
    box(ctx,0,54,91,51);box(ctx,5,59,81,20,'#000');
    bitmapText(ctx,'SIGNAL',9,66,6,'#fff');
    bitmapText(ctx,String(Math.round(signal.rate)).padStart(3,'0'),9,76,11,'#fff');
    bitmapText(ctx,'Hz',36,76,7,'#fff');
    for(let i=0;i<12;i++)box(ctx,54+i*2,64,1,11,i/12<signal.dopamine?'#fff':'#555',null);
    bitmapText(ctx,'DA',9,91,7);bitmapText(ctx,'NPF',35,91,7);bitmapText(ctx,'OUT',65,91,7);
    [13,42,73].forEach(bx=>box(ctx,bx,95,7,4,pattern(.5)));
    [5,80].forEach(bx=>box(ctx,bx,106,9,4,'#000'));
    const values=[signal.dopamine,signal.npf,clamp(signal.rate/Math.max(1,metadata?.config?.max_rate||100),0,1)];
    const labels=['DA','NPF','DRV'];
    values.forEach((value,i)=>{
      const bx=7+i*29;
      box(ctx,bx,9,20,42);box(ctx,bx+3,12,14,35);
      const level=Math.round(clamp(value,0,1)*32);
      box(ctx,bx+4,46-level,12,level,pattern([.5,.25,.75][i]),null);
      box(ctx,bx-1,5,22,5,pattern(.5));box(ctx,bx-1,49,22,4,'#000');
      line(ctx,bx+5,13,bx+5,43,'#fff');
      bitmapText(ctx,labels[i],bx+1,0,7);
    });
    ctx.restore();
    return [13,42,73].map(bx=>({x:x+(bx+3)*scale,y:y+98*scale}));
  }

  function drawWire(start,end,index) {
    const bottom=Math.min(height-9,Math.max(start.y,end.y)+35+index*9);
    const path=()=>{ctx.beginPath();ctx.moveTo(start.x,start.y);ctx.bezierCurveTo(start.x+30,bottom,end.x-40-index*9,bottom,end.x,end.y);};
    ctx.lineCap='butt';ctx.lineJoin='miter';ctx.setLineDash([]);
    path();ctx.strokeStyle='#fff';ctx.lineWidth=4;ctx.stroke();
    path();ctx.strokeStyle='#000';ctx.lineWidth=1;
    ctx.setLineDash(index===1?[3,2]:index===2?[1,2]:[]);ctx.stroke();ctx.setLineDash([]);
    box(ctx,end.x-2,end.y-2,4,4,'#fff');
  }

  function oneBit() {
    const raster=ctx.getImageData(0,0,width,height),pixels=raster.data;
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){
      const i=(y*width+x)*4;
      const luminance=pixels[i]*.2126+pixels[i+1]*.7152+pixels[i+2]*.0722;
      const threshold=(bayer[(y%8)*8+x%8]+.5)*255/64;
      const value=luminance>threshold?255:0;
      pixels[i]=pixels[i+1]=pixels[i+2]=value;pixels[i+3]=255;
    }
    ctx.putImageData(raster,0,0);
  }
  function drawScene() {
    ctx.setTransform(1,0,0,1,0,0);ctx.imageSmoothingEnabled=false;
    ctx.drawImage(background,0,0);
    if(latest){
      const scale=Math.min(width/530,height/295);
      const deviceScale=clamp(width/530,.65,1.05);
      const fx=width*.69,fy=height*.68,fs=scale*.78;
      const inputs=drawDevice(width*.095,height*.48,{dopamine:latest.dopamine,npf:latest.npf,rate:latest.dopamine_rate_hz},deviceScale);
      ctx.save();ctx.translate(fx,fy);ctx.scale(fs,fs);
      // Both joint geometry and body translation come from this server state.
      const pose=FlyActor.draw(ctx,{motorPose:latest.motorPose,dopamine:latest.dopamine,npf:latest.npf});
      ctx.restore();
      pose.ports.forEach((port,i)=>drawWire(inputs[i],{x:fx+port.x*fs,y:fy+port.y*fs},i));
    }
    oneBit();
  }
  function scheduleDraw(){
    if(drawQueued)return;
    drawQueued=true;
    requestAnimationFrame(()=>{drawQueued=false;drawScene();});
  }

  function updateConnection(value){
    document.body.dataset.livePhase=value.phase;
    const status=$('header-status');
    status.textContent=({connecting:'Connecting to simulation',connected:'Live',
      stale:'Last pose · stream stale',disconnected:'Reconnecting · last pose held',
      closed:'Live connection closed'})[value.phase]||value.phase;
    status.title=value.error||'';
    if(!latest)$('loading').firstElementChild.textContent=value.phase==='disconnected'
      ?'Live backend unavailable. Waiting to reconnect…':'Waiting for the first server state…';
  }

  function resize(){
    const bounds=canvas.getBoundingClientRect();
    const pixelScale=bounds.width<540?1:2;
    width=Math.max(1,Math.round(bounds.width/pixelScale));
    height=Math.max(1,Math.round(bounds.height/pixelScale));
    canvas.width=width;canvas.height=height;ctx.imageSmoothingEnabled=false;
    buildRoom();scheduleDraw();
  }
  function unavailable(error){
    document.body.dataset.livePhase='unavailable';
    $('header-status').textContent='Live viewer unavailable';
    $('header-status').title=error.message||String(error);
    $('loading').firstElementChild.textContent='Live state could not be loaded. '+(error.message||String(error));
  }

  if(!ctx||!bg||!window.FlyActor||!window.FlyLive){unavailable(new Error('Required live viewer assets did not load.'));return;}
  resize();new ResizeObserver(resize).observe(room);
  try{
    client=FlyLive.connect(backend,{
      staleAfterMs:window.FlyLiveConfig?.staleAfterMs??15000,
      onMeta(meta){metadata=meta;scheduleDraw();},
      onState(frame){latest=frame;$('loading').hidden=true;scheduleDraw();},
      onStatus:updateConnection,
    });
  }catch(error){unavailable(error);}
  window.addEventListener('beforeunload',()=>client?.close());
})();

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

  // Actual 5×7 glyphs: each stroke lands on the final canvas pixel grid.
  // Canvas font antialiasing followed by dithering destroys text at this size.
  const deviceGlyphs={
    '0':['01110','10001','10011','10101','11001','10001','01110'],
    '1':['00100','01100','00100','00100','00100','00100','01110'],
    '2':['01110','10001','00001','00010','00100','01000','11111'],
    '3':['11110','00001','00001','01110','00001','00001','11110'],
    '4':['00010','00110','01010','10010','11111','00010','00010'],
    '5':['11111','10000','10000','11110','00001','00001','11110'],
    '6':['01110','10000','10000','11110','10001','10001','01110'],
    '7':['11111','00001','00010','00100','01000','01000','01000'],
    '8':['01110','10001','10001','01110','10001','10001','01110'],
    '9':['01110','10001','10001','01111','00001','00001','01110'],
    A:['01110','10001','10001','11111','10001','10001','10001'],
    D:['11110','10001','10001','10001','10001','10001','11110'],
    E:['11111','10000','10000','11110','10000','10000','11111'],
    F:['11111','10000','10000','11110','10000','10000','10000'],
    H:['10001','10001','10001','11111','10001','10001','10001'],
    L:['10000','10000','10000','10000','10000','10000','11111'],
    N:['10001','11001','11001','10101','10011','10011','10001'],
    P:['11110','10001','10001','11110','10000','10000','10000'],
    R:['11110','10001','10001','11110','10100','10010','10001'],
    T:['11111','00100','00100','00100','00100','00100','00100'],
    V:['10001','10001','10001','10001','10001','01010','00100'],
    Z:['11111','00001','00010','00100','01000','10000','11111'],
    '-':['00000','00000','00000','11111','00000','00000','00000'],
  };
  function deviceText(value,x,y,scale=1,color='#000'){
    ctx.fillStyle=color;x=Math.round(x);y=Math.round(y);
    [...String(value).toUpperCase()].forEach((letter,index)=>{
      const glyph=deviceGlyphs[letter];if(!glyph)return;
      glyph.forEach((row,dy)=>[...row].forEach((bit,dx)=>{
        if(bit==='1')ctx.fillRect(x+(index*6+dx)*scale,y+dy*scale,scale,scale);
      }));
    });
  }
  function centeredDeviceText(value,x,y){
    deviceText(value,x-(value.length*6-1)/2,y);
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

  function drawDevice(signal) {
    const full=width>=360&&height>=250;
    const faceWidth=full?128:94,side=full?10:8,deviceHeight=full?160:120;
    const x=Math.round(width*.095);
    const y=Math.max(8,Math.round(Math.min(height*.48,height-deviceHeight-12)));
    const centers=full?[24,64,104]:[18,47,76];
    const tubeY=full?22:18,tubeHeight=full?48:29,tubeWidth=full?24:18;
    const faceY=full?80:56,faceHeight=full?73:57;
    const socketY=full?139:105,socketWidth=full?12:10,socketHeight=full?9:7;
    ctx.save();ctx.translate(x,y);
    // No fractional scaling: labels, glass highlights and gauges stay crisp.
    polygon(ctx,[[0,faceY-5],[faceWidth,faceY-5],[faceWidth+side,faceY+3],[side,faceY+3]],pattern(.125));
    polygon(ctx,[[faceWidth,faceY-5],[faceWidth+side,faceY+3],
      [faceWidth+side,faceY+faceHeight+2],[faceWidth,faceY+faceHeight]],pattern(.125));
    box(ctx,0,faceY,faceWidth,faceHeight);
    for(let vent=faceY+12;vent<faceY+faceHeight-8;vent+=6)
      line(ctx,faceWidth+3,vent,faceWidth+side-2,vent+3);
    [4,faceWidth-5].forEach(screw=>{
      box(ctx,screw,faceY+3,3,3,'#000',null);
      box(ctx,screw+1,faceY+4,1,1,'#fff',null);
    });
    [7,faceWidth-16].forEach(foot=>box(ctx,foot,faceY+faceHeight+1,10,5,'#000',null));

    const screenX=full?7:5,screenY=faceY+7,screenWidth=faceWidth-2*screenX;
    box(ctx,screenX,screenY,screenWidth,full?33:31,'#000');
    const textX=screenX+5,textY=screenY+4;
    deviceText('DA RATE',textX,textY,1,'#fff');
    const rate=String(Math.round(signal.rate)).padStart(3,'0');
    deviceText(rate,textX,textY+11,2,'#fff');
    deviceText('HZ',textX+rate.length*12+4,textY+18,1,'#fff');
    const meterX=full?86:68,barCount=full?10:6;
    deviceText(full?'LEVEL':'DA',full?85:72,textY,1,'#fff');
    line(ctx,meterX-7,textY,meterX-7,textY+23,'#fff');
    for(let i=0;i<barCount;i++){
      const active=i/barCount<clamp(signal.dopamine,0,1);
      box(ctx,meterX+i*3,textY+12,2,active?11:1,'#fff',null);
    }

    const values=[signal.dopamine,signal.npf,clamp(signal.rate/Math.max(1,metadata?.config?.max_rate||100),0,1)];
    const labels=['DA','NPF','RATE'];
    centers.forEach((center,i)=>{
      const tubeX=center-tubeWidth/2,plateWidth=full?36:26;
      // White nameplates separate lettering from the room's stippled floor.
      box(ctx,center-plateWidth/2,0,plateWidth,11);
      centeredDeviceText(labels[i],center,2);
      box(ctx,tubeX-2,tubeY-7,tubeWidth+4,5,'#000');
      for(let ridge=tubeX+1;ridge<tubeX+tubeWidth;ridge+=4)
        box(ctx,ridge,tubeY-6,1,3,'#fff',null);
      box(ctx,tubeX,tubeY,tubeWidth,tubeHeight);
      const insideY=tubeY+3,insideHeight=tubeHeight-6;
      const level=Math.round(clamp(values[i],0,1)*insideHeight);
      box(ctx,tubeX+3,insideY+insideHeight-level,tubeWidth-6,level,pattern([.5,.25,.75][i]),null);
      if(level>0)line(ctx,tubeX+3,insideY+insideHeight-level,tubeX+tubeWidth-3,insideY+insideHeight-level);
      box(ctx,tubeX+3,insideY,2,insideHeight,'#fff',null);
      [0,.25,.5,.75,1].forEach((fraction,tick)=>{
        const tickY=Math.round(insideY+fraction*insideHeight);
        line(ctx,tubeX+tubeWidth-2,tickY,tubeX+tubeWidth+(tick%2===0?3:1),tickY);
      });
      if(full){deviceText('1',tubeX+tubeWidth+5,insideY-1);deviceText('0',tubeX+tubeWidth+5,insideY+insideHeight-6);}
      box(ctx,tubeX-2,tubeY+tubeHeight+2,tubeWidth+4,5,'#000');
      line(ctx,tubeX+1,tubeY+tubeHeight+3,tubeX+tubeWidth-1,tubeY+tubeHeight+3,'#fff');
      centeredDeviceText(labels[i],center,full?128:96);
      box(ctx,center-socketWidth/2,socketY,socketWidth,socketHeight,'#000',null);
      box(ctx,center-socketWidth/2+2,socketY+2,socketWidth-4,socketHeight-4,'#fff',null);
      box(ctx,center-2,socketY+socketHeight-3,4,3,'#000',null);
    });
    ctx.restore();
    // Wire roots and socket bottoms share exactly the same snapped coordinates.
    return centers.map(center=>({x:x+center,y:y+socketY+socketHeight-1}));
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
      const fx=width*.69,fy=height*.68,fs=scale*.78;
      const inputs=drawDevice({dopamine:latest.dopamine,npf:latest.npf,rate:latest.dopamine_rate_hz});
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

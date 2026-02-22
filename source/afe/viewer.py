"""Three.js 3D fly-through viewer for Arrow Field Embeddings.

Generates a standalone HTML file with:
- WASD + mouse fly controls (click to capture, ESC to release)
- Round points colored by label
- 3D cone arrows (instanced mesh) per arrow index, toggleable
- Full viewport rendering with lighting
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional


def _spherical_to_cartesian(arrows: np.ndarray, scale: float) -> np.ndarray:
    theta = arrows[:, :, 0]
    phi = arrows[:, :, 1]
    mag = arrows[:, :, 2]
    cos_phi = np.cos(phi)
    dx = mag * cos_phi * np.cos(theta) * scale
    dy = mag * cos_phi * np.sin(theta) * scale
    dz = mag * np.sin(phi) * scale
    return np.stack([dx, dy, dz], axis=-1).astype(np.float32)


def save_viewer(
    spatial: np.ndarray,
    arrows: np.ndarray,
    labels: Optional[np.ndarray] = None,
    path: str = "viewer.html",
    arrow_scale: float = 1.0,
    point_size: float = 10.0,
    title: str = "AFE Viewer",
    open_browser: bool = True,
) -> Path:
    """Generate and save an interactive Three.js viewer.

    Parameters
    ----------
    spatial : ndarray (n, 3)
    arrows : ndarray (n, k, 3) -- spherical (azimuth, elevation, magnitude)
    labels : ndarray (n,), optional -- integer class labels for coloring
    path : str -- output HTML file path
    arrow_scale : float -- arrow length multiplier (1.0 = auto)
    point_size : float -- screen-space point size in pixels
    title : str -- viewer title
    open_browser : bool -- open in default browser after saving
    """
    n, k, _ = arrows.shape

    spatial_range = np.ptp(spatial, axis=0).mean()

    # Density-aware arrow scaling: target p95 arrow length = 60% of median NN distance
    from sklearn.neighbors import NearestNeighbors
    sample_idx = np.random.default_rng(0).choice(n, min(n, 1000), replace=False)
    nn = NearestNeighbors(n_neighbors=2).fit(spatial)
    dists, _ = nn.kneighbors(spatial[sample_idx])
    median_nn = float(np.median(dists[:, 1]))
    target_p95 = median_nn * 0.6 * arrow_scale

    # Compute raw directions (scale=1), then rescale
    directions = _spherical_to_cartesian(arrows, 1.0)
    mags = np.linalg.norm(directions, axis=-1)  # (n, k)
    p95 = np.percentile(mags[mags > 1e-8], 95) if np.any(mags > 1e-8) else 1.0
    rescale = target_p95 / max(p95, 1e-8)
    directions = directions * rescale

    data = {
        "n": int(n),
        "k": int(k),
        "title": title,
        "pointSize": float(point_size),
        "range": float(spatial_range),
        "spatial": [round(float(x), 4) for x in spatial.flatten()],
    }

    if labels is not None:
        data["labels"] = [int(x) for x in labels]
        data["nLabels"] = int(np.max(labels) + 1)
    else:
        data["labels"] = [0] * n
        data["nLabels"] = 1

    for ai in range(k):
        data[f"arrow_{ai}"] = [round(float(x), 4) for x in directions[:, ai].flatten()]

    data_json = json.dumps(data, separators=(",", ":"))
    html = _TEMPLATE.replace("/*__DATA__*/", f"const DATA = {data_json};")

    out = Path(path)
    out.write_text(html)

    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{out.resolve()}")

    return out


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AFE Viewer</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0a;overflow:hidden;color:#eee;font-family:-apple-system,'Segoe UI',Roboto,monospace}
canvas{display:block}
#hud{position:fixed;top:0;left:0;right:0;bottom:0;pointer-events:none;z-index:10}
#title{position:absolute;top:14px;left:14px;font-size:15px;font-weight:600;opacity:.9}
#arrow-status{position:absolute;top:38px;left:14px;color:#aaa;font-size:12px}
#speed-status{position:absolute;top:56px;left:14px;color:#666;font-size:11px}
#controls{position:absolute;bottom:14px;left:14px;color:#555;font-size:11px;line-height:1.8}
#legend{position:absolute;top:14px;right:14px;color:#bbb;font-size:12px;line-height:2}
#click-prompt{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:20px;opacity:.5;transition:opacity .4s;letter-spacing:1px}
</style>
</head>
<body>
<div id="hud">
  <div id="title"></div>
  <div id="arrow-status"></div>
  <div id="speed-status"></div>
  <div id="controls">
    WASD move | QE up/down | Mouse look | Scroll speed | Shift boost<br>
    Left/Right cycle arrows | PgUp/Dn jump 10 | 1-0 toggle | backtick hide all | T all<br>
    O orbit/fly | B background | G grid box<br>
    Orbit: drag rotate, right-drag pan, scroll zoom
  </div>
  <div id="legend"></div>
  <div id="click-prompt">Click to fly</div>
</div>

<script src="three.min.js"></script>
<script>
/*__DATA__*/

var LABEL_PALETTE = [
  [0.90,0.10,0.10],[0.20,0.47,0.72],[0.30,0.68,0.29],[0.59,0.30,0.64],
  [1.00,0.50,0.00],[0.95,0.95,0.20],[0.65,0.34,0.16],[0.97,0.51,0.75],
  [0.55,0.55,0.55],[0.10,0.75,0.75]
];
var ARROW_HEX = [
  0xff3333,0x3388ff,0x33dd33,0xaa55ff,0xff8800,
  0x00cccc,0xeedd33,0xff55aa,0x77ee77,0xffaa33,
  0x33eeff,0xff33ff
];

// Scene
var scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0a);

var camera = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.1, 50000);
var renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);

// Lighting
scene.add(new THREE.AmbientLight(0x666666));
scene.add(new THREE.HemisphereLight(0xddeeff, 0x334455, 1.2));
var dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(1, 2, 3);
scene.add(dirLight);

// Bounds
var mnX=1e9,mnY=1e9,mnZ=1e9,mxX=-1e9,mxY=-1e9,mxZ=-1e9;
for(var i=0;i<DATA.n;i++){
  var x=DATA.spatial[i*3],y=DATA.spatial[i*3+1],z=DATA.spatial[i*3+2];
  if(x<mnX)mnX=x;if(x>mxX)mxX=x;
  if(y<mnY)mnY=y;if(y>mxY)mxY=y;
  if(z<mnZ)mnZ=z;if(z>mxZ)mxZ=z;
}
var cx=(mnX+mxX)/2,cy=(mnY+mxY)/2,cz=(mnZ+mxZ)/2;
var range=Math.max(mxX-mnX,mxY-mnY,mxZ-mnZ)||1;
camera.position.set(cx, cy, cz + range*1.3);
camera.lookAt(cx, cy, cz);

// 3D sphere points (instanced)
var sphereR = range * 0.0025 * DATA.pointSize / 10.0 * 0.5;
var sphereGeo = new THREE.SphereGeometry(1, 16, 12);
var sphereMat = new THREE.MeshPhongMaterial({
  shininess: 60,
  transparent: true,
  opacity: 0.92
});
var points = new THREE.InstancedMesh(sphereGeo, sphereMat, DATA.n);
var ptDummy = new THREE.Object3D();
var ptColor = new THREE.Color();
for (var i = 0; i < DATA.n; i++) {
  ptDummy.position.set(DATA.spatial[i*3], DATA.spatial[i*3+1], DATA.spatial[i*3+2]);
  ptDummy.scale.set(sphereR, sphereR, sphereR);
  ptDummy.updateMatrix();
  points.setMatrixAt(i, ptDummy.matrix);
  var c = LABEL_PALETTE[DATA.labels[i] % LABEL_PALETTE.length];
  ptColor.setRGB(c[0], c[1], c[2]);
  points.setColorAt(i, ptColor);
}
points.instanceMatrix.needsUpdate = true;
points.instanceColor.needsUpdate = true;
scene.add(points);

// Single merged arrow geometry: shaft + head as one object
function makeArrowGeo(shaftR, headR, shaftFrac) {
  var segs = 8;
  var headFrac = 1.0 - shaftFrac;
  var shaft = new THREE.CylinderGeometry(shaftR, shaftR, shaftFrac, segs);
  shaft.translate(0, shaftFrac / 2, 0);
  var head = new THREE.ConeGeometry(headR, headFrac, segs);
  head.translate(0, shaftFrac + headFrac / 2, 0);
  shaft.rotateX(-Math.PI / 2);
  head.rotateX(-Math.PI / 2);
  var merged = new THREE.BufferGeometry();
  var sPos = shaft.getAttribute('position').array;
  var sNor = shaft.getAttribute('normal').array;
  var hPos = head.getAttribute('position').array;
  var hNor = head.getAttribute('normal').array;
  var sIdx = shaft.index ? Array.from(shaft.index.array) : [];
  var hIdx = head.index ? Array.from(head.index.array) : [];
  var allPos = new Float32Array(sPos.length + hPos.length);
  var allNor = new Float32Array(sNor.length + hNor.length);
  allPos.set(sPos, 0); allPos.set(hPos, sPos.length);
  allNor.set(sNor, 0); allNor.set(hNor, sNor.length);
  var off = sPos.length / 3;
  var allIdx = sIdx.concat(hIdx.map(function(v){ return v + off; }));
  merged.setAttribute('position', new THREE.BufferAttribute(allPos, 3));
  merged.setAttribute('normal', new THREE.BufferAttribute(allNor, 3));
  merged.setIndex(allIdx);
  return merged;
}
var arrowGeo = makeArrowGeo(0.025, 0.08, 0.7);

var arrowGroups = [];
var zAxis = new THREE.Vector3(0, 0, 1);
var dummy = new THREE.Object3D();

for (var ai = 0; ai < DATA.k; ai++) {
  var dirs = DATA['arrow_' + ai];
  var hex = ARROW_HEX[ai % ARROW_HEX.length];
  var mat = new THREE.MeshLambertMaterial({ color: hex, transparent: true, opacity: 0.85 });
  var mesh = new THREE.InstancedMesh(arrowGeo, mat, DATA.n);

  for (var i = 0; i < DATA.n; i++) {
    var px = DATA.spatial[i*3], py = DATA.spatial[i*3+1], pz = DATA.spatial[i*3+2];
    var dx = dirs[i*3], dy = dirs[i*3+1], dz = dirs[i*3+2];
    var len = Math.sqrt(dx*dx + dy*dy + dz*dz);

    if (len < 1e-7) {
      dummy.scale.set(0, 0, 0);
      dummy.position.set(px, py, pz);
      dummy.quaternion.identity();
    } else {
      var dirV = new THREE.Vector3(dx/len, dy/len, dz/len);
      var quat = new THREE.Quaternion().setFromUnitVectors(zAxis, dirV);
      dummy.position.set(px + dirV.x * sphereR, py + dirV.y * sphereR, pz + dirV.z * sphereR);
      dummy.quaternion.copy(quat);
      dummy.scale.set(len, len, len);
    }
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
  }

  mesh.instanceMatrix.needsUpdate = true;
  mesh.visible = (ai === 0);
  scene.add(mesh);
  arrowGroups.push({ mesh: mesh, vis: ai === 0 });
}

var curArrow = 0;

// ===== Camera mode: 'fly' or 'orbit' =====
var camMode = 'fly';
var orbitGroup = new THREE.Group();  // holds bounding box + grid + labels
scene.add(orbitGroup);
orbitGroup.visible = true;

// --- Build Plotly-style bounding box with grids and axis labels ---
(function buildOrbitBox() {
  var pad = range * 0.02;
  var bx0 = mnX - pad, bx1 = mxX + pad;
  var by0 = mnY - pad, by1 = mxY + pad;
  var bz0 = mnZ - pad, bz1 = mxZ + pad;

  // Wireframe box edges
  var boxMat = new THREE.LineBasicMaterial({ color: 0x555555, transparent: true, opacity: 0.6 });
  var edges = [
    [bx0,by0,bz0, bx1,by0,bz0],[bx0,by1,bz0, bx1,by1,bz0],
    [bx0,by0,bz1, bx1,by0,bz1],[bx0,by1,bz1, bx1,by1,bz1],
    [bx0,by0,bz0, bx0,by1,bz0],[bx1,by0,bz0, bx1,by1,bz0],
    [bx0,by0,bz1, bx0,by1,bz1],[bx1,by0,bz1, bx1,by1,bz1],
    [bx0,by0,bz0, bx0,by0,bz1],[bx1,by0,bz0, bx1,by0,bz1],
    [bx0,by1,bz0, bx0,by1,bz1],[bx1,by1,bz0, bx1,by1,bz1]
  ];
  for (var ei = 0; ei < edges.length; ei++) {
    var eg = new THREE.BufferGeometry();
    var ev = new Float32Array(edges[ei]);
    eg.setAttribute('position', new THREE.BufferAttribute(ev, 3));
    orbitGroup.add(new THREE.Line(eg, boxMat));
  }

  // Grid lines on 3 back faces (like Plotly)
  var gridMat = new THREE.LineBasicMaterial({ color: 0x333333, transparent: true, opacity: 0.4 });
  var nGrid = 5;
  // XY back face (z = bz0)
  for (var gi = 0; gi <= nGrid; gi++) {
    var t = gi / nGrid;
    var gx = bx0 + t * (bx1 - bx0);
    var gy = by0 + t * (by1 - by0);
    var g1 = new THREE.BufferGeometry();
    g1.setAttribute('position', new THREE.BufferAttribute(new Float32Array([gx,by0,bz0, gx,by1,bz0]), 3));
    orbitGroup.add(new THREE.Line(g1, gridMat));
    var g2 = new THREE.BufferGeometry();
    g2.setAttribute('position', new THREE.BufferAttribute(new Float32Array([bx0,gy,bz0, bx1,gy,bz0]), 3));
    orbitGroup.add(new THREE.Line(g2, gridMat));
  }
  // XZ bottom face (y = by0)
  for (var gi = 0; gi <= nGrid; gi++) {
    var t = gi / nGrid;
    var gx = bx0 + t * (bx1 - bx0);
    var gz = bz0 + t * (bz1 - bz0);
    var g1 = new THREE.BufferGeometry();
    g1.setAttribute('position', new THREE.BufferAttribute(new Float32Array([gx,by0,bz0, gx,by0,bz1]), 3));
    orbitGroup.add(new THREE.Line(g1, gridMat));
    var g2 = new THREE.BufferGeometry();
    g2.setAttribute('position', new THREE.BufferAttribute(new Float32Array([bx0,by0,gz, bx1,by0,gz]), 3));
    orbitGroup.add(new THREE.Line(g2, gridMat));
  }
  // YZ left face (x = bx0)
  for (var gi = 0; gi <= nGrid; gi++) {
    var t = gi / nGrid;
    var gy = by0 + t * (by1 - by0);
    var gz = bz0 + t * (bz1 - bz0);
    var g1 = new THREE.BufferGeometry();
    g1.setAttribute('position', new THREE.BufferAttribute(new Float32Array([bx0,gy,bz0, bx0,gy,bz1]), 3));
    orbitGroup.add(new THREE.Line(g1, gridMat));
    var g2 = new THREE.BufferGeometry();
    g2.setAttribute('position', new THREE.BufferAttribute(new Float32Array([bx0,by0,gz, bx0,by1,gz]), 3));
    orbitGroup.add(new THREE.Line(g2, gridMat));
  }

  // Axis labels using sprite text
  function makeLabel(text, pos) {
    var c = document.createElement('canvas');
    c.width = 128; c.height = 48;
    var ctx = c.getContext('2d');
    ctx.fillStyle = '#999';
    ctx.font = 'bold 28px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 64, 24);
    var tex = new THREE.CanvasTexture(c);
    var spMat = new THREE.SpriteMaterial({ map: tex, transparent: true });
    var sp = new THREE.Sprite(spMat);
    sp.position.copy(pos);
    sp.scale.set(range * 0.12, range * 0.045, 1);
    return sp;
  }
  orbitGroup.add(makeLabel('X', new THREE.Vector3((bx0+bx1)/2, by0 - range*0.06, bz0)));
  orbitGroup.add(makeLabel('Y', new THREE.Vector3(bx0 - range*0.06, (by0+by1)/2, bz0)));
  orbitGroup.add(makeLabel('Z', new THREE.Vector3(bx0, by0 - range*0.06, (bz0+bz1)/2)));

  // Tick labels
  function makeTickLabel(val, pos) {
    var c = document.createElement('canvas');
    c.width = 96; c.height = 32;
    var ctx = c.getContext('2d');
    ctx.fillStyle = '#666';
    ctx.font = '20px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(val.toFixed(1), 48, 16);
    var tex = new THREE.CanvasTexture(c);
    var spMat = new THREE.SpriteMaterial({ map: tex, transparent: true });
    var sp = new THREE.Sprite(spMat);
    sp.position.copy(pos);
    sp.scale.set(range * 0.08, range * 0.025, 1);
    return sp;
  }
  for (var ti = 0; ti <= nGrid; ti++) {
    var t = ti / nGrid;
    orbitGroup.add(makeTickLabel(bx0+t*(bx1-bx0), new THREE.Vector3(bx0+t*(bx1-bx0), by0-range*0.03, bz0)));
    orbitGroup.add(makeTickLabel(by0+t*(by1-by0), new THREE.Vector3(bx0-range*0.04, by0+t*(by1-by0), bz0)));
    orbitGroup.add(makeTickLabel(bz0+t*(bz1-bz0), new THREE.Vector3(bx0-range*0.02, by0-range*0.03, bz0+t*(bz1-bz0))));
  }
})();

// --- Orbit state ---
var orbitTheta = 0.6, orbitPhi = 0.4, orbitDist = range * 1.8;
var orbitTarget = new THREE.Vector3(cx, cy, cz);
var orbitDragging = false, orbitPanning = false;
var orbitLastX = 0, orbitLastY = 0;

function updateOrbitCamera() {
  var cp = Math.cos(orbitPhi), sp2 = Math.sin(orbitPhi);
  var ct = Math.cos(orbitTheta), st = Math.sin(orbitTheta);
  camera.position.set(
    orbitTarget.x + orbitDist * cp * st,
    orbitTarget.y + orbitDist * sp2,
    orbitTarget.z + orbitDist * cp * ct
  );
  camera.lookAt(orbitTarget);
}

// --- Fly state ---
var keys = {};
var euler = new THREE.Euler(0, 0, 0, 'YXZ');
var flySpeed = range * 0.3;
var locked = false;

document.addEventListener('keydown', function(e) { keys[e.code] = true; });
document.addEventListener('keyup', function(e) { keys[e.code] = false; });

// --- Background (independent of camera mode) ---
var darkBg = true;
function toggleBackground() {
  darkBg = !darkBg;
  scene.background = new THREE.Color(darkBg ? 0x0a0a0a : 0xffffff);
  updateHUD();
}


// --- Mode switching (camera only, not background) ---
function setMode(mode) {
  if (mode === camMode) return;
  if (camMode === 'fly' && locked) {
    document.exitPointerLock();
  }
  camMode = mode;

  if (mode === 'orbit') {
    var dx2 = camera.position.x - cx, dy2 = camera.position.y - cy, dz2 = camera.position.z - cz;
    orbitDist = Math.sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2);
    orbitPhi = Math.asin(Math.max(-1, Math.min(1, dy2 / orbitDist)));
    orbitTheta = Math.atan2(dx2, dz2);
    orbitTarget.set(cx, cy, cz);
    updateOrbitCamera();
  }
  updateHUD();
}

// --- Mouse handlers (both modes) ---
renderer.domElement.addEventListener('mousedown', function(e) {
  if (camMode === 'fly') {
    renderer.domElement.requestPointerLock();
  } else {
    if (e.button === 0) { orbitDragging = true; }
    if (e.button === 2) { orbitPanning = true; }
    orbitLastX = e.clientX; orbitLastY = e.clientY;
  }
});
renderer.domElement.addEventListener('mouseup', function(e) {
  orbitDragging = false; orbitPanning = false;
});
renderer.domElement.addEventListener('contextmenu', function(e) {
  if (camMode === 'orbit') e.preventDefault();
});

document.addEventListener('pointerlockchange', function() {
  locked = document.pointerLockElement === renderer.domElement;
  document.getElementById('click-prompt').style.opacity = (camMode === 'fly' && !locked) ? '0.5' : '0';
});

document.addEventListener('mousemove', function(e) {
  if (camMode === 'fly') {
    if (!locked) return;
    euler.setFromQuaternion(camera.quaternion);
    euler.y -= e.movementX * 0.002;
    euler.x -= e.movementY * 0.002;
    euler.x = Math.max(-1.55, Math.min(1.55, euler.x));
    camera.quaternion.setFromEuler(euler);
  } else {
    var dx2 = e.clientX - orbitLastX, dy2 = e.clientY - orbitLastY;
    orbitLastX = e.clientX; orbitLastY = e.clientY;
    if (orbitDragging) {
      orbitTheta -= dx2 * 0.005;
      orbitPhi += dy2 * 0.005;
      orbitPhi = Math.max(-1.5, Math.min(1.5, orbitPhi));
      updateOrbitCamera();
    }
    if (orbitPanning) {
      var panScale = orbitDist * 0.001;
      var right = new THREE.Vector3();
      var up = new THREE.Vector3();
      camera.getWorldDirection(new THREE.Vector3());
      right.crossVectors(camera.up, new THREE.Vector3().subVectors(camera.position, orbitTarget).normalize()).normalize();
      up.crossVectors(new THREE.Vector3().subVectors(camera.position, orbitTarget).normalize(), right).normalize();
      orbitTarget.add(right.multiplyScalar(-dx2 * panScale));
      orbitTarget.add(up.multiplyScalar(dy2 * panScale));
      updateOrbitCamera();
    }
  }
});

renderer.domElement.addEventListener('wheel', function(e) {
  e.preventDefault();
  if (camMode === 'fly') {
    flySpeed *= e.deltaY > 0 ? 0.85 : 1.18;
    flySpeed = Math.max(range * 0.005, Math.min(range * 10, flySpeed));
  } else {
    orbitDist *= e.deltaY > 0 ? 1.1 : 0.9;
    orbitDist = Math.max(range * 0.1, Math.min(range * 20, orbitDist));
    updateOrbitCamera();
  }
  updateHUD();
}, { passive: false });

function flyUpdate(dt) {
  if (camMode !== 'fly' || !locked) return;
  var sp = flySpeed * dt;
  var boost = (keys['ShiftLeft'] || keys['ShiftRight']) ? 3 : 1;
  var v = new THREE.Vector3();
  if (keys['KeyW']) v.z -= 1;
  if (keys['KeyS']) v.z += 1;
  if (keys['KeyA']) v.x -= 1;
  if (keys['KeyD']) v.x += 1;
  if (keys['KeyE']) v.y += 1;
  if (keys['KeyQ']) v.y -= 1;
  if (v.lengthSq() > 0) {
    v.normalize().multiplyScalar(sp * boost);
    v.applyQuaternion(camera.quaternion);
    camera.position.add(v);
  }
}

// Arrow toggling
function showSingle(idx){
  idx = ((idx % DATA.k) + DATA.k) % DATA.k;
  for(var i=0;i<arrowGroups.length;i++){
    arrowGroups[i].mesh.visible = (i===idx);
    arrowGroups[i].vis = (i===idx);
  }
  curArrow = idx;
  updateHUD();
}

document.addEventListener('keydown', function(e){
  if(e.code==='ArrowRight'){ showSingle(curArrow+1); e.preventDefault(); }
  else if(e.code==='ArrowLeft'){ showSingle(curArrow-1); e.preventDefault(); }
  else if(e.code==='PageDown'){ showSingle(curArrow+10); e.preventDefault(); }
  else if(e.code==='PageUp'){ showSingle(curArrow-10); e.preventDefault(); }
  else if(e.code==='Home'){ showSingle(0); }
  else if(e.code==='End'){ showSingle(DATA.k-1); }
  else if(e.code==='Backquote'){
    for(var j=0;j<arrowGroups.length;j++){arrowGroups[j].mesh.visible=false;arrowGroups[j].vis=false;}
    updateHUD();
  }
  else if(e.code==='KeyT'&&!locked){
    var all=true;
    for(var j=0;j<arrowGroups.length;j++){if(!arrowGroups[j].vis)all=false;}
    for(var j=0;j<arrowGroups.length;j++){arrowGroups[j].mesh.visible=!all;arrowGroups[j].vis=!all;}
    updateHUD();
  }
  else if(e.code==='KeyO'){ setMode(camMode==='fly'?'orbit':'fly'); }
  else if(e.code==='KeyB'&&!locked){ toggleBackground(); }
  else if(e.code==='KeyG'){ orbitGroup.visible=!orbitGroup.visible; }
  else if(e.code==='KeyP'&&!locked){ points.visible=!points.visible; }
  else if(e.code.match(/^Digit[0-9]$/)&&!locked){
    var num=parseInt(e.code[5]);
    var idx=num===0?9:num-1;
    if(idx<arrowGroups.length){
      arrowGroups[idx].vis=!arrowGroups[idx].vis;
      arrowGroups[idx].mesh.visible=arrowGroups[idx].vis;
      updateHUD();
    }
  }
});

// HUD
function updateHUD(){
  document.getElementById('title').textContent = DATA.title;
  var vis=[];
  for(var i=0;i<arrowGroups.length;i++) if(arrowGroups[i].vis) vis.push(i);
  var el = document.getElementById('arrow-status');
  if(vis.length===0) el.textContent='Arrows: hidden (Left/Right to cycle)';
  else if(vis.length===1) el.textContent='Arrow '+vis[0]+' of '+DATA.k+' (Left/Right to cycle)';
  else el.textContent=vis.length+'/'+DATA.k+' arrows visible';
  var modeStr = camMode === 'orbit' ? 'ORBIT' : 'FLY';
  var speedStr = camMode === 'fly' ? ' | Speed: ' + flySpeed.toFixed(1) : '';
  document.getElementById('speed-status').textContent = modeStr + speedStr;
  // Adapt HUD colors for background
  document.getElementById('hud').style.color = darkBg ? '#eee' : '#222';
  document.getElementById('arrow-status').style.color = darkBg ? '#aaa' : '#555';
  document.getElementById('speed-status').style.color = darkBg ? '#666' : '#777';
  document.getElementById('controls').style.color = darkBg ? '#555' : '#999';
  document.getElementById('legend').style.color = darkBg ? '#bbb' : '#444';
  document.getElementById('click-prompt').style.color = darkBg ? '#eee' : '#222';
  document.getElementById('click-prompt').style.opacity = (camMode==='fly'&&!locked) ? '0.5' : '0';
}

// Build legend with safe DOM methods
var leg = document.getElementById('legend');
var nL = Math.min(DATA.nLabels, LABEL_PALETTE.length);
for(var i=0;i<nL;i++){
  var c=LABEL_PALETTE[i];
  var row = document.createElement('div');
  var dot = document.createElement('span');
  dot.style.display = 'inline-block';
  dot.style.width = '10px';
  dot.style.height = '10px';
  dot.style.borderRadius = '50%';
  dot.style.marginRight = '8px';
  dot.style.verticalAlign = 'middle';
  dot.style.background = 'rgb('+Math.round(c[0]*255)+','+Math.round(c[1]*255)+','+Math.round(c[2]*255)+')';
  row.appendChild(dot);
  row.appendChild(document.createTextNode(String(i)));
  leg.appendChild(row);
}
updateHUD();

// Resize
addEventListener('resize', function(){
  camera.aspect=innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// Animation loop
var clock = new THREE.Clock();
(function animate(){
  requestAnimationFrame(animate);
  var dt = Math.min(clock.getDelta(), 0.1);
  flyUpdate(dt);
  renderer.render(scene, camera);
})();
</script>
</body>
</html>
"""

import { useState, useCallback } from 'react';
import { useViewerStore } from '../store/useViewerStore';
import { buildShareUrl } from '../systems/bookmark';
import type { BookmarkState } from '../systems/bookmark';
import type { AFEDataset } from '../types/dataset';

/**
 * Generate a self-contained HTML string that embeds the dataset and
 * renders it with Three.js loaded from a CDN. The file works offline
 * (apart from the initial CDN fetch) and in any modern browser.
 */
function generateStandaloneHtml(
  dataset: AFEDataset,
  cameraPos: [number, number, number],
  cameraTarget: [number, number, number],
  activeArrowIndex: number | 'all',
  arrowScale: number,
): string {
  // Compact JSON: round positions/arrows to 4 decimal places to save space
  const compactDataset = {
    dataset: dataset.dataset,
    n_points: dataset.n_points,
    n_arrows: dataset.n_arrows,
    positions: dataset.positions.map((v) => Math.round(v * 10000) / 10000),
    arrows: dataset.arrows.map((v) => Math.round(v * 10000) / 10000),
    label_indices: dataset.label_indices,
    label_names: dataset.label_names,
    clusters: dataset.clusters.map((c) => ({ id: c.id, label: c.label, size: c.size })),
  };

  const dataJson = JSON.stringify(compactDataset);
  const initialArrow = activeArrowIndex === 'all' ? 0 : activeArrowIndex;

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AFE - ${dataset.dataset}</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0a;overflow:hidden;color:#eee;font-family:-apple-system,'Segoe UI',Roboto,monospace}
canvas{display:block}
#hud{position:fixed;top:0;left:0;right:0;bottom:0;pointer-events:none;z-index:10}
#info{position:absolute;top:14px;left:14px}
#info h1{font-size:14px;font-weight:600;opacity:.9;margin:0 0 4px 0}
#info .meta{font-size:11px;color:#888;line-height:1.6}
#arrow-hud{position:absolute;top:14px;right:14px;font-size:12px;color:#aaa;text-align:right}
#controls-hint{position:absolute;bottom:14px;left:14px;color:#555;font-size:10px;line-height:1.7}
</style>
</head>
<body>
<div id="hud">
  <div id="info">
    <h1 id="title"></h1>
    <div class="meta" id="meta"></div>
  </div>
  <div id="arrow-hud"></div>
  <div id="controls-hint">
    Drag to orbit | Scroll to zoom | Right-drag to pan<br>
    Left/Right arrows to cycle arrow fields
  </div>
</div>

<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.170.0/build/three.module.min.js",
    "three/addons/": "https://unpkg.com/three@0.170.0/examples/jsm/"
  }
}
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// --- Embedded dataset ---
const DS = ${dataJson};

// --- Constants ---
const GOLDEN_ANGLE = 137.508;
const CONE_SEGMENTS = 8;
const BASE_CONE_RADIUS = 0.12;
const BASE_CONE_HEIGHT = 1.0;
const POINT_SIZE_SCALE = 150;
const POINT_SIZE_LOG_BASE = 8;
const SCREEN_SCALE = 48.0;

// --- Initial state from viewer ---
const initialCameraPos = [${cameraPos[0]},${cameraPos[1]},${cameraPos[2]}];
const initialCameraTarget = [${cameraTarget[0]},${cameraTarget[1]},${cameraTarget[2]}];
let currentArrow = ${initialArrow};
const userArrowScale = ${arrowScale};

// =========================================================
// Utility: HSL to RGB
// =========================================================
function hslToRgb(h, s, l) {
  let r, g, b;
  if (s === 0) {
    r = g = b = l;
  } else {
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1/3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1/3);
  }
  return [r, g, b];
}
function hue2rgb(p, q, t) {
  if (t < 0) t += 1;
  if (t > 1) t -= 1;
  if (t < 1/6) return p + (q - p) * 6 * t;
  if (t < 1/2) return q;
  if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
  return p;
}

// =========================================================
// Utility: Spherical to Cartesian (Y-up, matching app)
// =========================================================
function sphericalToYUp(theta, phi, r) {
  const cosPhi = Math.cos(phi);
  return [
    r * cosPhi * Math.cos(theta),
    r * Math.sin(phi),
    r * cosPhi * Math.sin(theta),
  ];
}

// =========================================================
// Build cluster color palette (golden angle hue spacing)
// =========================================================
function buildClusterPalette(clusters) {
  const palette = new Map();
  clusters.forEach((c, i) => {
    const hue = (i * GOLDEN_ANGLE) % 360;
    const [r, g, b] = hslToRgb(hue / 360, 0.8, 0.65);
    palette.set(c.id, [r, g, b]);
  });
  palette.set(-1, [0.55, 0.55, 0.55]);
  return palette;
}

// =========================================================
// Build arrow color palette
// =========================================================
function buildArrowPalette(nArrows) {
  const colors = [];
  for (let i = 0; i < nArrows; i++) {
    const hue = (i * GOLDEN_ANGLE) % 360;
    const [r, g, b] = hslToRgb(hue / 360, 0.85, 0.6);
    colors.push(new THREE.Color(r, g, b));
  }
  return colors;
}

// =========================================================
// Auto-scale arrows (same algorithm as ArrowField.tsx)
// =========================================================
function computeAutoScale(positions, arrows, nPoints, nArrows) {
  if (nPoints < 2 || nArrows === 0) return 1;
  const sampleN = Math.min(nPoints, 2000);
  const step = Math.max(1, Math.floor(nPoints / sampleN));

  const magnitudes = [];
  for (let i = 0; i < nPoints; i += step) {
    for (let j = 0; j < nArrows; j++) {
      const r = Math.abs(arrows[i * nArrows * 3 + j * 3 + 2]);
      if (r > 0) magnitudes.push(r);
    }
  }
  if (magnitudes.length === 0) return 1;
  magnitudes.sort((a, b) => a - b);
  const p95 = magnitudes[Math.floor(magnitudes.length * 0.95)];
  if (p95 < 1e-10) return 1;

  const nnDists = [];
  for (let i = 0; i < nPoints && nnDists.length < sampleN; i += step) {
    const px = positions[i * 3];
    const py = positions[i * 3 + 1];
    const pz = positions[i * 3 + 2];
    let minDist = Infinity;
    for (let j = Math.max(0, i - 10); j < Math.min(nPoints, i + 10); j++) {
      if (j === i) continue;
      const dx = positions[j * 3] - px;
      const dy = positions[j * 3 + 1] - py;
      const dz = positions[j * 3 + 2] - pz;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d < minDist) minDist = d;
    }
    if (minDist < Infinity) nnDists.push(minDist);
  }
  if (nnDists.length === 0) return 1;
  nnDists.sort((a, b) => a - b);
  const medianNN = nnDists[Math.floor(nnDists.length / 2)];
  return (medianNN * 0.6) / p95;
}

// =========================================================
// Direction quaternion (rotate +Y to arrow direction)
// =========================================================
const _UP = new THREE.Vector3(0, 1, 0);
function directionQuaternion(dx, dy, dz) {
  const q = new THREE.Quaternion();
  const dir = new THREE.Vector3(dx, dy, dz);
  const len = dir.length();
  if (len < 1e-8) return q.identity();
  dir.divideScalar(len);
  const dot = _UP.dot(dir);
  if (dot < -0.9999) {
    q.set(1, 0, 0, 0);
    return q;
  }
  return q.setFromUnitVectors(_UP, dir);
}

// =========================================================
// Scene setup
// =========================================================
const n = DS.n_points;
const k = DS.n_arrows;
const palette = buildClusterPalette(DS.clusters);
const arrowPalette = buildArrowPalette(k);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0a);

const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 50000);
camera.position.set(initialCameraPos[0], initialCameraPos[1], initialCameraPos[2]);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);

// Lighting
scene.add(new THREE.AmbientLight(0x666666));
scene.add(new THREE.HemisphereLight(0xddeeff, 0x334455, 1.2));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(1, 2, 3);
scene.add(dirLight);

// OrbitControls
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(initialCameraTarget[0], initialCameraTarget[1], initialCameraTarget[2]);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.update();

// =========================================================
// Point cloud (custom shader matching the app)
// =========================================================
const pointSize = POINT_SIZE_SCALE / Math.log(n) / Math.log(POINT_SIZE_LOG_BASE);

const positions = new Float32Array(DS.positions);
const colors = new Float32Array(n * 3);
for (let i = 0; i < n; i++) {
  const labelIdx = DS.label_indices[i];
  const rgb = palette.get(labelIdx) || [0.55, 0.55, 0.55];
  colors[i * 3] = rgb[0];
  colors[i * 3 + 1] = rgb[1];
  colors[i * 3 + 2] = rgb[2];
}

const pointGeo = new THREE.BufferGeometry();
pointGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
pointGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

const pointMat = new THREE.ShaderMaterial({
  vertexShader: \`
    attribute vec3 color;
    varying vec3 vColor;
    uniform float pointSize;
    uniform float screenScale;
    void main() {
      vColor = color;
      vec4 cameraSpacePos = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * cameraSpacePos;
      float dist = length(cameraSpacePos.xyz);
      gl_PointSize = max(pointSize * screenScale / dist, 2.0);
    }
  \`,
  fragmentShader: \`
    varying vec3 vColor;
    void main() {
      vec2 center = gl_PointCoord - vec2(0.5);
      float r2 = dot(center, center);
      if (r2 > 0.25) discard;
      float outerR = sqrt(r2) * 2.27;
      if (r2 > 0.19) {
        float haloFade = (sqrt(r2) - 0.44) / 0.06;
        float haloAlpha = (1.0 - haloFade) * 0.35;
        gl_FragColor = vec4(0.0, 0.0, 0.0, haloAlpha);
        return;
      }
      float r = outerR;
      float r3 = r * r * r;
      float diffuse = 1.0 - r3 * 0.65;
      float specDist = length(center - vec2(-0.1, -0.1));
      float spec = smoothstep(0.25, 0.0, specDist);
      float rim = smoothstep(0.25, 0.4, sqrt(r2));
      vec3 shaded = vColor * diffuse * (1.0 - rim * 0.4) + vec3(1.0) * spec * 0.25;
      gl_FragColor = vec4(shaded, 1.0);
    }
  \`,
  uniforms: {
    pointSize: { value: pointSize },
    screenScale: { value: SCREEN_SCALE },
  },
  transparent: true,
  depthWrite: true,
});

const points = new THREE.Points(pointGeo, pointMat);
points.frustumCulled = false;
scene.add(points);

// =========================================================
// Arrow field (instanced cones, same approach as ArrowField.tsx)
// =========================================================
const autoScale = computeAutoScale(DS.positions, DS.arrows, n, k);
const effectiveScale = autoScale * userArrowScale;

const coneGeo = new THREE.ConeGeometry(BASE_CONE_RADIUS, BASE_CONE_HEIGHT, CONE_SEGMENTS);
coneGeo.translate(0, BASE_CONE_HEIGHT / 2, 0);

const _matrix = new THREE.Matrix4();
const _position = new THREE.Vector3();
const _scale = new THREE.Vector3();

const arrowMeshes = [];
for (let ai = 0; ai < k; ai++) {
  const mat = new THREE.MeshStandardMaterial({
    color: arrowPalette[ai] || new THREE.Color(1, 1, 1),
    roughness: 0.4,
    metalness: 0.1,
    transparent: true,
    opacity: 0.85,
  });
  const mesh = new THREE.InstancedMesh(coneGeo, mat, n);
  mesh.frustumCulled = false;

  let instanceCount = 0;
  for (let i = 0; i < n; i++) {
    const theta = DS.arrows[i * k * 3 + ai * 3];
    const phi = DS.arrows[i * k * 3 + ai * 3 + 1];
    const r = DS.arrows[i * k * 3 + ai * 3 + 2];
    const mag = Math.abs(r) * effectiveScale;

    if (mag < 1e-6) {
      _matrix.makeScale(0, 0, 0);
      mesh.setMatrixAt(instanceCount, _matrix);
      instanceCount++;
      continue;
    }

    const [dx, dy, dz] = sphericalToYUp(theta, phi, 1);
    const quat = directionQuaternion(dx, dy, dz);

    _position.set(DS.positions[i * 3], DS.positions[i * 3 + 1], DS.positions[i * 3 + 2]);
    _scale.set(mag, mag, mag);
    _matrix.compose(_position, quat, _scale);
    mesh.setMatrixAt(instanceCount, _matrix);
    instanceCount++;
  }

  mesh.count = instanceCount;
  mesh.instanceMatrix.needsUpdate = true;
  mesh.visible = (ai === currentArrow);
  scene.add(mesh);
  arrowMeshes.push(mesh);
}

// =========================================================
// HUD
// =========================================================
function updateHUD() {
  document.getElementById('title').textContent = DS.dataset;
  document.getElementById('meta').textContent =
    n.toLocaleString() + ' points | ' + k + ' arrow fields | ' +
    DS.clusters.length + ' clusters';
  const el = document.getElementById('arrow-hud');
  if (k === 0) {
    el.textContent = 'No arrows';
  } else {
    el.textContent = 'Arrow ' + currentArrow + ' / ' + k;
  }
}
updateHUD();

// =========================================================
// Arrow cycling (keyboard)
// =========================================================
function showArrow(idx) {
  idx = ((idx % k) + k) % k;
  for (let i = 0; i < arrowMeshes.length; i++) {
    arrowMeshes[i].visible = (i === idx);
  }
  currentArrow = idx;
  updateHUD();
}

document.addEventListener('keydown', function(e) {
  if (e.code === 'ArrowRight') { showArrow(currentArrow + 1); e.preventDefault(); }
  else if (e.code === 'ArrowLeft') { showArrow(currentArrow - 1); e.preventDefault(); }
});

// =========================================================
// Resize
// =========================================================
addEventListener('resize', function() {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// =========================================================
// Render loop
// =========================================================
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();
</script>
</body>
</html>`;
}

/**
 * Trigger a file download in the browser by creating a temporary link.
 */
function downloadHtmlBlob(html: string, filename: string): void {
  const blob = new Blob([html], { type: 'text/html' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  // Clean up
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 100);
}

export function ShareButton() {
  const [copied, setCopied] = useState(false);
  const [downloading, setDownloading] = useState(false);

  const handleShare = useCallback(async () => {
    const store = useViewerStore.getState();

    // Camera position/target aren't in Zustand — read from the Three.js canvas
    // We access the R3F store via the canvas's __r3f internals
    const canvas = document.querySelector('canvas');
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const r3fStore = (canvas as any)?.__r3f?.store?.getState();
    const camera = r3fStore?.camera;
    const controls = r3fStore?.controls;

    if (!camera) return;

    const target = controls?.target;
    const cameraTarget: [number, number, number] = target
      ? [target.x, target.y, target.z]
      : [0, 0, 0];

    const state: BookmarkState = {
      datasetUrl: store.datasetUrl,
      cameraPos: [camera.position.x, camera.position.y, camera.position.z],
      cameraTarget,
      spaceScale: store.spaceScale,
      colorMode: store.colorMode,
      controlMode: store.controlMode,
      selectedIndex: store.selectedIndex,
    };

    const url = buildShareUrl(state);

    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback: update URL bar
      window.history.replaceState(null, '', url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, []);

  const handleDownload = useCallback(() => {
    const store = useViewerStore.getState();
    const dataset = store.dataset;
    if (!dataset) return;

    // Read camera state from the Three.js canvas
    const canvas = document.querySelector('canvas');
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const r3fStore = (canvas as any)?.__r3f?.store?.getState();
    const camera = r3fStore?.camera;
    const controls = r3fStore?.controls;

    const cameraPos: [number, number, number] = camera
      ? [camera.position.x, camera.position.y, camera.position.z]
      : [0, 0, 10];

    const target = controls?.target;
    const cameraTarget: [number, number, number] = target
      ? [target.x, target.y, target.z]
      : [0, 0, 0];

    setDownloading(true);

    // Use requestAnimationFrame to let the UI update before the
    // potentially heavy HTML generation runs.
    requestAnimationFrame(() => {
      try {
        const html = generateStandaloneHtml(
          dataset,
          cameraPos,
          cameraTarget,
          store.activeArrowIndex,
          store.arrowScale,
        );

        const safeName = dataset.dataset
          .replace(/[^a-zA-Z0-9_-]/g, '_')
          .toLowerCase();
        downloadHtmlBlob(html, `afe_${safeName}.html`);
      } finally {
        setDownloading(false);
      }
    });
  }, []);

  const btnClass =
    'rounded-full bg-black/60 px-3 py-1.5 text-xs font-mono text-white/70 backdrop-blur-sm hover:bg-black/80 hover:text-white border border-white/10';

  return (
    <div className="flex gap-2">
      <button
        onClick={handleDownload}
        className={btnClass}
        title="Download self-contained HTML viewer"
        disabled={downloading}
      >
        {downloading ? 'GENERATING...' : 'DOWNLOAD'}
      </button>
      <button
        onClick={handleShare}
        className={btnClass}
        title="Copy shareable link to current view"
      >
        {copied ? 'COPIED!' : 'SHARE'}
      </button>
    </div>
  );
}

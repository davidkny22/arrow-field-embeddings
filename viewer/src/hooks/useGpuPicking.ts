import { useRef, useEffect } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useViewerStore } from '../store/useViewerStore';

const pickingVS = /* glsl */ `
attribute vec3 pickingColor;
attribute float scaleFactor;
varying vec3 vPickingColor;
uniform float pointSize;
uniform float screenScale;

void main() {
  vPickingColor = pickingColor;
  vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mvPos;
  float dist = length(mvPos.xyz);
  gl_PointSize = max(pointSize * screenScale / dist * scaleFactor, 2.0);
}
`;

const pickingFS = /* glsl */ `
varying vec3 vPickingColor;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  if (dot(c, c) > 0.25) discard;
  gl_FragColor = vec4(vPickingColor, 1.0);
}
`;

// Module-level bridge for rectangle picking
let _bridge: {
  gl: THREE.WebGLRenderer;
  camera: THREE.Camera;
  target: THREE.WebGLRenderTarget;
  material: THREE.ShaderMaterial;
  scene: THREE.Scene;
  pickPoints: THREE.Points;
  src: THREE.Points;
  pointSize: number;
  screenScale: number;
} | null = null;

function renderPickingPass(b: typeof _bridge) {
  if (!b) return;
  const { gl, camera, target, material, scene, pickPoints, src, pointSize, screenScale } = b;

  if (pickPoints.geometry !== src.geometry) {
    pickPoints.geometry = src.geometry;
  }

  src.updateWorldMatrix(true, false);
  pickPoints.matrix.copy(src.matrixWorld);
  pickPoints.matrixWorld.copy(src.matrixWorld);

  material.uniforms.pointSize.value = pointSize;
  material.uniforms.screenScale.value = screenScale;

  const prevTarget = gl.getRenderTarget();
  const prevClear = new THREE.Color();
  gl.getClearColor(prevClear);
  const prevAlpha = gl.getClearAlpha();

  gl.setRenderTarget(target);
  gl.setClearColor(0x000000, 0);
  gl.clear();
  gl.render(scene, camera);

  gl.setRenderTarget(prevTarget);
  gl.setClearColor(prevClear, prevAlpha);
}

export function pickRectangle(cssRect: { x: number; y: number; w: number; h: number }): Set<number> {
  if (!_bridge) return new Set();

  const { gl, target } = _bridge;

  renderPickingPass(_bridge);

  const dpr = gl.getPixelRatio();
  const canvasH = gl.domElement.clientHeight;

  const x = Math.round(cssRect.x * dpr);
  const y = Math.round((canvasH - cssRect.y - cssRect.h) * dpr);
  const w = Math.max(1, Math.round(cssRect.w * dpr));
  const h = Math.max(1, Math.round(cssRect.h * dpr));

  const tw = target.width;
  const th = target.height;
  const cx = Math.max(0, Math.min(x, tw - 1));
  const cy = Math.max(0, Math.min(y, th - 1));
  const cw = Math.min(w, tw - cx);
  const ch = Math.min(h, th - cy);

  if (cw <= 0 || ch <= 0) return new Set();

  const pixels = new Uint8Array(cw * ch * 4);
  gl.readRenderTargetPixels(target, cx, cy, cw, ch, pixels);

  const indices = new Set<number>();
  for (let i = 0; i < cw * ch; i++) {
    const off = i * 4;
    const id = (pixels[off]! << 16) | (pixels[off + 1]! << 8) | pixels[off + 2]!;
    if (id > 0) indices.add(id - 1);
  }

  return indices;
}

export function useGpuPicking(
  pointsRef: React.RefObject<THREE.Points | null>,
  pointSize: number,
  screenScale: number,
): React.RefObject<number | null> {
  const { gl, camera } = useThree();

  const targetRef = useRef<THREE.WebGLRenderTarget | null>(null);
  const materialRef = useRef<THREE.ShaderMaterial | null>(null);
  const sceneRef = useRef(new THREE.Scene());
  const pickPointsRef = useRef<THREE.Points | null>(null);
  const pixel = useRef(new Uint8Array(4));
  const mouse = useRef({ x: -1, y: -1, dirty: false });
  const pickedIndex = useRef<number | null>(null);
  const lastReported = useRef<number | null>(null);

  useEffect(() => {
    const w = gl.domElement.width || 1;
    const h = gl.domElement.height || 1;
    const scene = sceneRef.current;

    targetRef.current = new THREE.WebGLRenderTarget(w, h, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
    });

    materialRef.current = new THREE.ShaderMaterial({
      vertexShader: pickingVS,
      fragmentShader: pickingFS,
      uniforms: {
        pointSize: { value: pointSize },
        screenScale: { value: screenScale },
      },
    });

    return () => {
      _bridge = null;
      targetRef.current?.dispose();
      materialRef.current?.dispose();
      if (pickPointsRef.current) {
        scene.remove(pickPointsRef.current);
        pickPointsRef.current = null;
      }
      targetRef.current = null;
      materialRef.current = null;
    };
  }, [gl]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const canvas = gl.domElement;
    const dpr = gl.getPixelRatio();
    const onMove = (e: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouse.current.x = (e.clientX - rect.left) * dpr;
      mouse.current.y = (rect.bottom - e.clientY) * dpr;
      mouse.current.dirty = true;
    };
    const onLeave = () => {
      mouse.current.dirty = false;
      pickedIndex.current = null;
      if (lastReported.current !== null) {
        lastReported.current = null;
        useViewerStore.getState().hoverPoint(null);
        document.body.style.cursor = 'auto';
      }
    };
    canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerleave', onLeave);
    return () => {
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerleave', onLeave);
    };
  }, [gl]);

  useEffect(() => {
    const observer = new ResizeObserver(() => {
      if (targetRef.current) {
        targetRef.current.setSize(
          gl.domElement.width || 1,
          gl.domElement.height || 1,
        );
      }
    });
    observer.observe(gl.domElement);
    return () => observer.disconnect();
  }, [gl]);

  useFrame(() => {
    if (!pointsRef.current || !targetRef.current || !materialRef.current) return;

    const src = pointsRef.current;

    if (!pickPointsRef.current || pickPointsRef.current.geometry !== src.geometry) {
      if (pickPointsRef.current) sceneRef.current.remove(pickPointsRef.current);
      pickPointsRef.current = new THREE.Points(src.geometry, materialRef.current);
      pickPointsRef.current.frustumCulled = false;
      pickPointsRef.current.matrixAutoUpdate = false;
      sceneRef.current.add(pickPointsRef.current);
    }

    _bridge = {
      gl,
      camera,
      target: targetRef.current,
      material: materialRef.current,
      scene: sceneRef.current,
      pickPoints: pickPointsRef.current,
      src,
      pointSize,
      screenScale,
    };

    if (!mouse.current.dirty) return;
    mouse.current.dirty = false;

    renderPickingPass(_bridge);

    gl.readRenderTargetPixels(
      targetRef.current,
      Math.round(mouse.current.x),
      Math.round(mouse.current.y),
      1, 1,
      pixel.current,
    );

    const id = (pixel.current[0]! << 16) | (pixel.current[1]! << 8) | pixel.current[2]!;
    const newIndex = id > 0 ? id - 1 : null;
    pickedIndex.current = newIndex;

    if (newIndex !== lastReported.current) {
      lastReported.current = newIndex;
      const store = useViewerStore.getState();
      if (newIndex != null && store.dataset && newIndex < store.dataset.n_points) {
        store.hoverPoint(newIndex);
        document.body.style.cursor = 'pointer';
      } else {
        store.hoverPoint(null);
        document.body.style.cursor = 'auto';
      }
    }
  });

  return pickedIndex;
}

export function buildPickingColors(numPoints: number): Float32Array {
  const colors = new Float32Array(numPoints * 3);
  for (let i = 0; i < numPoints; i++) {
    const id = i + 1;
    colors[i * 3] = ((id >> 16) & 0xFF) / 255;
    colors[i * 3 + 1] = ((id >> 8) & 0xFF) / 255;
    colors[i * 3 + 2] = (id & 0xFF) / 255;
  }
  return colors;
}

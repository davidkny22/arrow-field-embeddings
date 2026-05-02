import { useRef, useEffect, useMemo, useCallback } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { useViewerStore } from '../store/useViewerStore';
import { computeColors, buildClusterPalette } from '../systems/colorSystem';
import { useGpuPicking, buildPickingColors } from '../hooks/useGpuPicking';

const DRAG_THRESHOLD_PX = 3;

// Inverse-log point sizing (TF projector formula)
const POINT_SIZE_SCALE = 200;
const POINT_SIZE_LOG_BASE = 8;
const SCREEN_SCALE = 48.0;

const vertexShader = /* glsl */ `
attribute vec3 color;
attribute float scaleFactor;

varying vec3 vColor;

uniform float pointSize;
uniform float screenScale;

#include <fog_pars_vertex>

void main() {
  vColor = color;
  vec4 cameraSpacePos = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * cameraSpacePos;

  float dist = length(cameraSpacePos.xyz);
  float outputPointSize = pointSize * screenScale / dist;
  gl_PointSize = max(outputPointSize * scaleFactor, 2.0);

  vec4 mvPosition = cameraSpacePos;
  #include <fog_vertex>
}
`;

const fragmentShader = /* glsl */ `
varying vec3 vColor;

#include <common>
#include <fog_pars_fragment>

void main() {
  vec2 center = gl_PointCoord - vec2(0.5);
  float r2 = dot(center, center);

  if (r2 > 0.25) discard;

  float outerR = sqrt(r2) * 2.27;
  if (r2 > 0.19) {
    float haloFade = (sqrt(r2) - 0.44) / 0.06;
    float haloAlpha = (1.0 - haloFade) * 0.35;
    gl_FragColor = vec4(0.0, 0.0, 0.0, haloAlpha);
    #include <fog_fragment>
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

  #include <fog_fragment>
}
`;

export function PointCloud() {
  const pointsRef = useRef<THREE.Points>(null);
  const dataset = useViewerStore((s) => s.dataset);
  const colorMode = useViewerStore((s) => s.colorMode);
  const paletteName = useViewerStore((s) => s.palette);
  const highlightedIndices = useViewerStore((s) => s.highlightedIndices);
  const neighborIndices = useViewerStore((s) => s.neighborIndices);
  const neighborCenter = useViewerStore((s) => s.neighborCenter);
  const activeArrowIndex = useViewerStore((s) => s.activeArrowIndex);
  const pulseIndex = useViewerStore((s) => s.pulseIndex);
  const pointSizeMultiplier = useViewerStore((s) => s.pointSizeMultiplier);
  const { gl } = useThree();
  const pulseTime = useRef(0);
  const pointerDownPos = useRef<{ x: number; y: number } | null>(null);

  const palette = useMemo(() => {
    if (!dataset) return new Map<number, [number, number, number]>();
    return buildClusterPalette(dataset.clusters, paletteName);
  }, [dataset, paletteName]);

  const pointSize = useMemo(() => {
    if (!dataset) return 10;
    return POINT_SIZE_SCALE / Math.log(dataset.n_points) / Math.log(POINT_SIZE_LOG_BASE);
  }, [dataset]);

  // Build geometry buffers
  const geometry = useMemo(() => {
    if (!dataset) return null;
    const n = dataset.n_points;
    const positions = new Float32Array(dataset.positions);
    const scaleFactors = new Float32Array(n).fill(1.0);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('scaleFactor', new THREE.BufferAttribute(scaleFactors, 1));
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    geo.setAttribute('pickingColor', new THREE.BufferAttribute(buildPickingColors(n), 3));
    return geo;
  }, [dataset]);

  // Dispose geometry on unmount or dataset change
  useEffect(() => {
    return () => {
      geometry?.dispose();
    };
  }, [geometry]);

  // GPU picking — O(1) hover detection
  const pickedIndex = useGpuPicking(pointsRef, pointSize, SCREEN_SCALE);

  // Track pointer-down for click vs drag
  useEffect(() => {
    const canvas = gl.domElement;
    const handlePointerDown = (e: PointerEvent) => {
      pointerDownPos.current = { x: e.clientX, y: e.clientY };
    };
    canvas.addEventListener('pointerdown', handlePointerDown);
    return () => canvas.removeEventListener('pointerdown', handlePointerDown);
  }, [gl]);

  const wasDrag = useCallback((e: MouseEvent) => {
    if (!pointerDownPos.current) return false;
    const dx = e.clientX - pointerDownPos.current.x;
    const dy = e.clientY - pointerDownPos.current.y;
    return Math.sqrt(dx * dx + dy * dy) > DRAG_THRESHOLD_PX;
  }, []);

  // Click handling
  useEffect(() => {
    const canvas = gl.domElement;

    const handleClick = (e: MouseEvent) => {
      if (wasDrag(e)) return;

      // Arrow click was handled by R3F event system in ArrowField
      if ((window as any).__arrowClickHandled) {
        (window as any).__arrowClickHandled = false;
        return;
      }

      const idx = pickedIndex.current;
      if (idx != null && dataset && idx < dataset.n_points) {
        useViewerStore.getState().selectPoint(idx);
        useViewerStore.getState().setClickedArrow(null);
      } else {
        const store = useViewerStore.getState();
        store.selectPoint(null);
        store.setClickedArrow(null);
        if (store.highlightedIndices.size > 0) {
          store.setHighlightedIndices(new Set());
        }
        if (store.neighborCenter != null) {
          store.setNeighborhood(null, []);
        }
        if (store.colorMode !== 'cluster') {
          store.setColorMode('cluster');
        }
      }
    };
    canvas.addEventListener('click', handleClick);
    return () => canvas.removeEventListener('click', handleClick);
  }, [gl, dataset, wasDrag, pickedIndex]);

  // Update colors and scale factors
  useEffect(() => {
    if (!geometry || !dataset) return;

    const colors = computeColors(dataset, colorMode, {
      clusterPalette: palette,
      palette: paletteName,
      highlightedIndices: highlightedIndices.size > 0 ? highlightedIndices : undefined,
      neighborIndices,
      neighborCenter,
      activeArrowIndex,
      arrowData: dataset.arrows,
      nArrows: dataset.n_arrows,
      reconError: dataset.recon_error,
    });

    const colorAttr = geometry.getAttribute('color') as THREE.BufferAttribute;
    (colorAttr.array as Float32Array).set(colors);
    colorAttr.needsUpdate = true;

    const scaleAttr = geometry.getAttribute('scaleFactor') as THREE.BufferAttribute;
    const scales = scaleAttr.array as Float32Array;

    if (highlightedIndices.size > 0) {
      for (let i = 0; i < dataset.n_points; i++) {
        scales[i] = highlightedIndices.has(i) ? 2.0 : 0.6;
      }
    } else {
      scales.fill(1.0);
    }
    scaleAttr.needsUpdate = true;
  }, [geometry, dataset, palette, paletteName, colorMode, highlightedIndices, neighborIndices, neighborCenter, activeArrowIndex]);

  // Pulse animation
  useEffect(() => {
    if (pulseIndex != null) pulseTime.current = 0;
  }, [pulseIndex]);

  const PULSE_DURATION = 4.0;
  const PULSE_SPEED = 5.0;
  useFrame((_, delta) => {
    if (pulseIndex == null || !geometry) return;
    pulseTime.current += delta;

    if (pulseTime.current > PULSE_DURATION) {
      const scaleAttr = geometry.getAttribute('scaleFactor') as THREE.BufferAttribute;
      (scaleAttr.array as Float32Array)[pulseIndex] = 1.0;
      scaleAttr.needsUpdate = true;
      useViewerStore.getState().setPulseIndex(null);
      return;
    }

    const wave = Math.sin(pulseTime.current * PULSE_SPEED);
    const scale = 2.5 + 1.5 * wave;

    const scaleAttr = geometry.getAttribute('scaleFactor') as THREE.BufferAttribute;
    (scaleAttr.array as Float32Array)[pulseIndex] = scale;
    scaleAttr.needsUpdate = true;
  });

  const uniforms = useMemo(
    () =>
      THREE.UniformsUtils.merge([
        THREE.UniformsLib.fog,
        {
          pointSize: { value: pointSize },
          screenScale: { value: SCREEN_SCALE },
        },
      ]),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  useEffect(() => {
    uniforms.pointSize.value = pointSize * pointSizeMultiplier;
  }, [pointSize, pointSizeMultiplier, uniforms]);

  if (!dataset || !geometry) return null;

  return (
    <points
      ref={pointsRef}
      geometry={geometry}
      frustumCulled={false}
    >
      <shaderMaterial
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={uniforms}
        fog
        transparent
        depthWrite={false}
      />
    </points>
  );
}

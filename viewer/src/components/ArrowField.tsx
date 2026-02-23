import { useRef, useEffect, useMemo } from 'react';
import * as THREE from 'three';
import { useViewerStore } from '../store/useViewerStore';
import { sphericalToThreeJS, directionQuaternion } from '../utils/spherical';
import { hslToRgb } from '../utils/color';

const GOLDEN_ANGLE = 137.508;
const CONE_SEGMENTS = 8;
const BASE_CONE_RADIUS = 0.12;
const BASE_CONE_HEIGHT = 1.0;

const _matrix = new THREE.Matrix4();
const _position = new THREE.Vector3();
const _quaternion = new THREE.Quaternion();
const _scale = new THREE.Vector3();

/**
 * Compute auto-scale so that the p95 arrow magnitude × autoScale ≈ 60%
 * of the median nearest-neighbor distance. This keeps arrows visible
 * but not overwhelming, matching the approach in source/afe/viewer.py.
 */
function computeAutoScale(
  positions: number[],
  arrows: number[],
  nPoints: number,
  nArrows: number,
): number {
  if (nPoints < 2 || nArrows === 0) return 1;

  // Sample up to 2000 points for performance
  const sampleN = Math.min(nPoints, 2000);
  const step = Math.max(1, Math.floor(nPoints / sampleN));

  // Collect arrow magnitudes
  const magnitudes: number[] = [];
  for (let i = 0; i < nPoints; i += step) {
    for (let j = 0; j < nArrows; j++) {
      const r = Math.abs(arrows[i * nArrows * 3 + j * 3 + 2]!);
      if (r > 0) magnitudes.push(r);
    }
  }
  if (magnitudes.length === 0) return 1;

  magnitudes.sort((a, b) => a - b);
  const p95 = magnitudes[Math.floor(magnitudes.length * 0.95)]!;
  if (p95 < 1e-10) return 1;

  // Estimate median NN distance from sampled points
  const nnDists: number[] = [];
  for (let i = 0; i < nPoints && nnDists.length < sampleN; i += step) {
    const px = positions[i * 3]!;
    const py = positions[i * 3 + 1]!;
    const pz = positions[i * 3 + 2]!;
    let minDist = Infinity;

    // Check a few nearby points (not a full kNN, just an estimate)
    for (let j = Math.max(0, i - 10); j < Math.min(nPoints, i + 10); j++) {
      if (j === i) continue;
      const dx = positions[j * 3]! - px;
      const dy = positions[j * 3 + 1]! - py;
      const dz = positions[j * 3 + 2]! - pz;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d < minDist) minDist = d;
    }
    if (minDist < Infinity) nnDists.push(minDist);
  }

  if (nnDists.length === 0) return 1;
  nnDists.sort((a, b) => a - b);
  const medianNN = nnDists[Math.floor(nnDists.length / 2)]!;

  // Target: p95 magnitude × scale ≈ 60% of median NN distance
  return (medianNN * 0.6) / p95;
}

/**
 * Build a distinct color palette for arrow indices using golden angle spacing.
 */
function arrowPalette(nArrows: number): THREE.Color[] {
  const colors: THREE.Color[] = [];
  for (let i = 0; i < nArrows; i++) {
    const hue = (i * GOLDEN_ANGLE) % 360;
    const [r, g, b] = hslToRgb(hue / 360, 0.85, 0.6);
    colors.push(new THREE.Color(r, g, b));
  }
  return colors;
}

/**
 * Renders arrow field as instanced cones. One InstancedMesh per arrow index.
 * Only the active arrow index (or all) is visible.
 */
export function ArrowField() {
  const dataset = useViewerStore((s) => s.dataset);
  const activeArrowIndex = useViewerStore((s) => s.activeArrowIndex);
  const arrowScale = useViewerStore((s) => s.arrowScale);
  const arrowDensity = useViewerStore((s) => s.arrowDensity);
  const arrowsVisible = useViewerStore((s) => s.arrowsVisible);

  const meshRefs = useRef<(THREE.InstancedMesh | null)[]>([]);
  const geometryRef = useRef<THREE.ConeGeometry | null>(null);

  // Compute auto-scale once per dataset
  const autoScale = useMemo(() => {
    if (!dataset) return 1;
    return computeAutoScale(
      dataset.positions,
      dataset.arrows,
      dataset.n_points,
      dataset.n_arrows,
    );
  }, [dataset]);

  // Build palette once per dataset
  const palette = useMemo(() => {
    if (!dataset) return [];
    return arrowPalette(dataset.n_arrows);
  }, [dataset]);

  // Create cone geometry (base at origin, tip along +Y)
  const coneGeometry = useMemo(() => {
    const geo = new THREE.ConeGeometry(
      BASE_CONE_RADIUS,
      BASE_CONE_HEIGHT,
      CONE_SEGMENTS,
    );
    // Translate so the base sits at origin and tip points along +Y
    geo.translate(0, BASE_CONE_HEIGHT / 2, 0);
    geometryRef.current = geo;
    return geo;
  }, []);

  // Dispose geometry on unmount
  useEffect(() => {
    return () => {
      geometryRef.current?.dispose();
    };
  }, []);

  // Update instance matrices whenever relevant state changes
  useEffect(() => {
    if (!dataset || dataset.n_arrows === 0) return;

    const n = dataset.n_points;
    const k = dataset.n_arrows;
    const scale = autoScale * arrowScale;
    const density = Math.max(1, Math.round(arrowDensity));

    for (let arrowIdx = 0; arrowIdx < k; arrowIdx++) {
      const mesh = meshRefs.current[arrowIdx];
      if (!mesh) continue;

      let instanceCount = 0;

      for (let i = 0; i < n; i += density) {
        const theta = dataset.arrows[i * k * 3 + arrowIdx * 3]!;
        const phi = dataset.arrows[i * k * 3 + arrowIdx * 3 + 1]!;
        const r = dataset.arrows[i * k * 3 + arrowIdx * 3 + 2]!;

        // Skip near-zero arrows
        const mag = Math.abs(r) * scale;
        if (mag < 1e-6) {
          // Set to zero-scale to hide
          _matrix.makeScale(0, 0, 0);
          mesh.setMatrixAt(instanceCount, _matrix);
          instanceCount++;
          continue;
        }

        // Arrow direction in Three.js Y-up coordinates
        const [dx, dy, dz] = sphericalToThreeJS(theta, phi, 1);
        directionQuaternion(dx, dy, dz, _quaternion);

        // Position at point location
        _position.set(
          dataset.positions[i * 3]!,
          dataset.positions[i * 3 + 1]!,
          dataset.positions[i * 3 + 2]!,
        );

        // Uniform scale by magnitude
        _scale.set(mag, mag, mag);

        _matrix.compose(_position, _quaternion, _scale);
        mesh.setMatrixAt(instanceCount, _matrix);
        instanceCount++;
      }

      mesh.count = instanceCount;
      mesh.instanceMatrix.needsUpdate = true;
    }
  }, [dataset, autoScale, arrowScale, arrowDensity]);

  // Update visibility when active arrow changes
  useEffect(() => {
    if (!dataset) return;
    for (let i = 0; i < dataset.n_arrows; i++) {
      const mesh = meshRefs.current[i];
      if (!mesh) continue;
      mesh.visible =
        arrowsVisible && (activeArrowIndex === 'all' || activeArrowIndex === i);
    }
  }, [dataset, activeArrowIndex, arrowsVisible]);

  if (!dataset || dataset.n_arrows === 0) return null;

  const k = dataset.n_arrows;
  const maxInstances = Math.ceil(dataset.n_points / Math.max(1, arrowDensity));

  return (
    <>
      {Array.from({ length: k }, (_, arrowIdx) => (
        <instancedMesh
          key={arrowIdx}
          ref={(el) => {
            meshRefs.current[arrowIdx] = el;
          }}
          args={[coneGeometry, undefined, maxInstances]}
          frustumCulled={false}
          visible={
            arrowsVisible &&
            (activeArrowIndex === 'all' || activeArrowIndex === arrowIdx)
          }
        >
          <meshStandardMaterial
            color={palette[arrowIdx] ?? new THREE.Color(1, 1, 1)}
            roughness={0.4}
            metalness={0.1}
            transparent
            opacity={0.85}
          />
        </instancedMesh>
      ))}
    </>
  );
}

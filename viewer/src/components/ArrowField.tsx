import { useRef, useEffect, useMemo } from 'react';
import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { useViewerStore } from '../store/useViewerStore';
import { sphericalToThreeJS, directionQuaternion } from '../utils/spherical';
import { hslToRgb } from '../utils/color';

const GOLDEN_ANGLE = 137.508;

// Arrow geometry proportions (unit arrow: total height = 1.0)
const SHAFT_RADIUS = 0.025;
const SHAFT_HEIGHT = 0.7;
const SHAFT_SEGMENTS = 5;
const HEAD_RADIUS = 0.08;
const HEAD_HEIGHT = 0.3;
const HEAD_SEGMENTS = 6;

const _matrix = new THREE.Matrix4();
const _position = new THREE.Vector3();
const _quaternion = new THREE.Quaternion();
const _scale = new THREE.Vector3();

/**
 * Compute auto-scale so the median arrow is ~3% of the cloud diameter.
 *
 * Arrow magnitudes are already normalized [0,1] by AFE. This converts them
 * to world-space lengths proportional to the point cloud's visual size.
 */
function computeAutoScale(
  positions: number[],
  arrows: number[],
  nPoints: number,
  nArrows: number,
): number {
  if (nPoints < 2 || nArrows === 0) return 1;

  // Cloud diameter: bounding box diagonal
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < nPoints; i++) {
    const x = positions[i * 3]!;
    const y = positions[i * 3 + 1]!;
    const z = positions[i * 3 + 2]!;
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  const dx = maxX - minX, dy = maxY - minY, dz = maxZ - minZ;
  const diameter = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (diameter < 1e-10) return 1;

  // Median non-zero arrow magnitude (sampled)
  const step = Math.max(1, Math.floor(nPoints / 2000));
  const magnitudes: number[] = [];
  for (let i = 0; i < nPoints; i += step) {
    for (let j = 0; j < nArrows; j++) {
      const r = Math.abs(arrows[i * nArrows * 3 + j * 3 + 2]!);
      if (r > 1e-8) magnitudes.push(r);
    }
  }
  if (magnitudes.length === 0) return 1;
  magnitudes.sort((a, b) => a - b);
  const median = magnitudes[Math.floor(magnitudes.length / 2)]!;
  if (median < 1e-10) return 1;

  // Scale so the median arrow is TARGET_FRACTION of the cloud diameter
  const TARGET_FRACTION = 0.02;
  return (diameter * TARGET_FRACTION) / median;
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
 * Create a merged arrow geometry: thin cylinder shaft + cone head.
 * Base at origin, tip at y = SHAFT_HEIGHT + HEAD_HEIGHT = 1.0.
 */
function createArrowGeometry(): THREE.BufferGeometry {
  const shaft = new THREE.CylinderGeometry(
    SHAFT_RADIUS,  // top radius (near head)
    0,             // bottom radius (at point) — tapers to nothing
    SHAFT_HEIGHT,
    SHAFT_SEGMENTS,
  );
  shaft.translate(0, SHAFT_HEIGHT / 2, 0);

  const head = new THREE.ConeGeometry(HEAD_RADIUS, HEAD_HEIGHT, HEAD_SEGMENTS);
  // Center of head above the shaft
  head.translate(0, SHAFT_HEIGHT + HEAD_HEIGHT / 2, 0);

  const merged = mergeGeometries([shaft, head]);
  shaft.dispose();
  head.dispose();

  if (!merged) {
    // Fallback: just a cone
    const fallback = new THREE.ConeGeometry(HEAD_RADIUS, 1.0, HEAD_SEGMENTS);
    fallback.translate(0, 0.5, 0);
    return fallback;
  }

  return merged;
}

/**
 * Renders arrow field as instanced shaft+cone arrows.
 * One InstancedMesh per arrow index. Only the active arrow index (or all) is visible.
 */
export function ArrowField() {
  const dataset = useViewerStore((s) => s.dataset);
  const activeArrowIndex = useViewerStore((s) => s.activeArrowIndex);
  const arrowScale = useViewerStore((s) => s.arrowScale);
  const arrowDensity = useViewerStore((s) => s.arrowDensity);
  const arrowsVisible = useViewerStore((s) => s.arrowsVisible);
  const spaceScale = useViewerStore((s) => s.spaceScale);
  const autoSpaceScale = useViewerStore((s) => s.autoSpaceScale);
  const totalScale = autoSpaceScale * spaceScale;

  const meshRefs = useRef<(THREE.InstancedMesh | null)[]>([]);
  const geometryRef = useRef<THREE.BufferGeometry | null>(null);
  // Maps instanceId → point index for each arrow mesh (for click handling)
  const instanceToPointRef = useRef<number[][]>([]);

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

  // Create arrow geometry (shaft + cone head, base at origin, tip along +Y)
  const arrowGeometry = useMemo(() => {
    const geo = createArrowGeometry();
    geo.computeBoundingSphere();
    geo.computeBoundingBox();
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

    // Rebuild instance-to-point mapping
    instanceToPointRef.current = Array.from({ length: k }, () => []);

    // Minimum visible size: 15% of the median-scaled arrow
    // (autoScale already normalizes so median ≈ TARGET_FRACTION of diameter)
    const minMag = autoScale * arrowScale * 0.15;

    for (let arrowIdx = 0; arrowIdx < k; arrowIdx++) {
      const mesh = meshRefs.current[arrowIdx];
      if (!mesh) continue;

      const pointMap: number[] = [];
      let instanceCount = 0;

      for (let i = 0; i < n; i += density) {
        const theta = dataset.arrows[i * k * 3 + arrowIdx * 3]!;
        const phi = dataset.arrows[i * k * 3 + arrowIdx * 3 + 1]!;
        const r = dataset.arrows[i * k * 3 + arrowIdx * 3 + 2]!;

        pointMap.push(i);

        // Skip truly zero arrows (no data)
        if (Math.abs(r) < 1e-8) {
          _matrix.makeScale(0, 0, 0);
          mesh.setMatrixAt(instanceCount, _matrix);
          instanceCount++;
          continue;
        }

        // Clamp to minimum visible size
        const mag = Math.max(Math.abs(r) * scale, minMag);

        // Arrow direction in Three.js Y-up coordinates
        const [dx, dy, dz] = sphericalToThreeJS(theta, phi, 1);
        directionQuaternion(dx, dy, dz, _quaternion);

        // Position at point location (scaled to match point cloud group)
        _position.set(
          dataset.positions[i * 3]! * totalScale,
          dataset.positions[i * 3 + 1]! * totalScale,
          dataset.positions[i * 3 + 2]! * totalScale,
        );

        _scale.set(mag, mag, mag);

        _matrix.compose(_position, _quaternion, _scale);
        mesh.setMatrixAt(instanceCount, _matrix);
        instanceCount++;
      }

      mesh.count = instanceCount;
      mesh.instanceMatrix.needsUpdate = true;
      mesh.computeBoundingSphere();
      instanceToPointRef.current[arrowIdx] = pointMap;
    }
  }, [dataset, autoScale, arrowScale, arrowDensity, totalScale]);

  // Update visibility when active arrow or visibility changes
  useEffect(() => {
    if (!dataset) return;
    for (let i = 0; i < dataset.n_arrows; i++) {
      const mesh = meshRefs.current[i];
      if (!mesh) continue;
      mesh.visible =
        arrowsVisible && (activeArrowIndex === 'all' || activeArrowIndex === i);
    }
  }, [dataset, activeArrowIndex, arrowsVisible]);

  // Clean up global flag on unmount
  useEffect(() => {
    return () => {
      delete (window as any).__arrowClickHandled;
    };
  }, []);

  if (!dataset || dataset.n_arrows === 0) return null;

  const k = dataset.n_arrows;
  const maxInstances = Math.ceil(dataset.n_points / Math.max(1, arrowDensity));

  return (
    <>
      {Array.from({ length: k }, (_, arrowIdx) => {
        const isVisible =
          arrowsVisible &&
          (activeArrowIndex === 'all' || activeArrowIndex === arrowIdx);
        return (
          <instancedMesh
            key={arrowIdx}
            ref={(el) => {
              meshRefs.current[arrowIdx] = el;
            }}
            args={[arrowGeometry, undefined, maxInstances]}
            frustumCulled={false}
            visible={isVisible}
            onClick={
              isVisible
                ? (e) => {
                    e.stopPropagation();
                    const instId = e.instanceId;
                    if (instId != null) {
                      const pointMap = instanceToPointRef.current[arrowIdx];
                      if (pointMap && instId < pointMap.length) {
                        const store = useViewerStore.getState();
                        store.setClickedArrow({
                          arrowIdx,
                          pointIdx: pointMap[instId]!,
                        });
                        store.selectPoint(pointMap[instId]!);
                      }
                    }
                    (window as any).__arrowClickHandled = true;
                  }
                : undefined
            }
          >
            <meshStandardMaterial
              color={palette[arrowIdx] ?? new THREE.Color(1, 1, 1)}
              roughness={0.4}
              metalness={0.1}
            />
          </instancedMesh>
        );
      })}
    </>
  );
}

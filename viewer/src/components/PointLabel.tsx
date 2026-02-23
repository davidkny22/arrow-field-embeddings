import { useRef, useState } from 'react';
import { Html } from '@react-three/drei';
import { useFrame, useThree } from '@react-three/fiber';
import { useViewerStore } from '../store/useViewerStore';

const PROXIMITY_DISTANCE = 12;
const MAX_PROXIMITY_LABELS = 15;
const UPDATE_INTERVAL = 0.25; // seconds between proximity checks

export function PointLabel() {
  const hoveredIndex = useViewerStore((s) => s.hoveredIndex);
  const dataset = useViewerStore((s) => s.dataset);
  const { camera } = useThree();
  const [nearbyPoints, setNearbyPoints] = useState<
    { label: string; pos: [number, number, number]; dist: number }[]
  >([]);
  const elapsed = useRef(0);

  useFrame((_, delta) => {
    if (!dataset) return;
    elapsed.current += delta;
    if (elapsed.current < UPDATE_INTERVAL) return;
    elapsed.current = 0;

    const cx = camera.position.x;
    const cy = camera.position.y;
    const cz = camera.position.z;
    const threshold = PROXIMITY_DISTANCE * PROXIMITY_DISTANCE;
    const nearby: typeof nearbyPoints = [];

    for (let i = 0; i < dataset.n_points; i++) {
      const px = dataset.positions[i * 3]!;
      const py = dataset.positions[i * 3 + 1]!;
      const pz = dataset.positions[i * 3 + 2]!;
      const dx = px - cx;
      const dy = py - cy;
      const dz = pz - cz;
      const distSq = dx * dx + dy * dy + dz * dz;

      if (distSq < threshold) {
        const label = dataset.label_names[dataset.label_indices[i]!] ?? `Point ${i}`;
        nearby.push({ label, pos: [px, py, pz] as [number, number, number], dist: Math.sqrt(distSq) });
        if (nearby.length > MAX_PROXIMITY_LABELS * 2) break;
      }
    }

    nearby.sort((a, b) => a.dist - b.dist);
    const limited = nearby.slice(0, MAX_PROXIMITY_LABELS);

    const changed =
      limited.length !== nearbyPoints.length ||
      limited.some((p, i) => p.label !== nearbyPoints[i]?.label);
    if (changed) setNearbyPoints(limited);
  });

  // Compute hovered point label text for dedup in proximity labels
  const hoveredLabel =
    hoveredIndex != null && dataset && hoveredIndex < dataset.n_points
      ? dataset.label_names[dataset.label_indices[hoveredIndex]!]
      : null;

  return (
    <>
      {/* Hovered point label */}
      {hoveredIndex != null && dataset && hoveredIndex < dataset.n_points && (
        <Html
          position={[
            dataset.positions[hoveredIndex * 3]!,
            dataset.positions[hoveredIndex * 3 + 1]!,
            dataset.positions[hoveredIndex * 3 + 2]!,
          ]}
          distanceFactor={10}
          style={{ pointerEvents: 'none' }}
        >
          <div className="rounded-md bg-black/80 px-3 py-1.5 text-sm text-white backdrop-blur-sm whitespace-nowrap">
            <div className="font-medium">{dataset.label_names[dataset.label_indices[hoveredIndex]!]}</div>
            {(() => {
              const cluster = dataset.clusters.find((c) => c.id === dataset.label_indices[hoveredIndex]!);
              return cluster ? (
                <div className="text-xs text-white/50">{cluster.label}</div>
              ) : null;
            })()}
          </div>
        </Html>
      )}

      {/* Proximity labels */}
      {nearbyPoints.map((p, idx) => {
        if (hoveredLabel && p.label === hoveredLabel) return null;
        const opacity = Math.max(0.2, 1 - p.dist / PROXIMITY_DISTANCE);
        return (
          <Html
            key={`${p.label}-${idx}`}
            position={p.pos}
            distanceFactor={10}
            style={{ pointerEvents: 'none' }}
          >
            <div
              className="whitespace-nowrap text-xs text-white select-none"
              style={{ opacity }}
            >
              {p.label}
            </div>
          </Html>
        );
      })}
    </>
  );
}

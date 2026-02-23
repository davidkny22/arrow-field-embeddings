import { useMemo } from 'react';
import * as THREE from 'three';
import { useViewerStore } from '../store/useViewerStore';

export function NeighborLines() {
  const dataset = useViewerStore((s) => s.dataset);
  const neighborIndices = useViewerStore((s) => s.neighborIndices);
  const neighborCenter = useViewerStore((s) => s.neighborCenter);

  const geometry = useMemo(() => {
    if (!dataset || neighborCenter == null || neighborIndices.length === 0) return null;

    const centerPos: [number, number, number] = [
      dataset.positions[neighborCenter * 3]!,
      dataset.positions[neighborCenter * 3 + 1]!,
      dataset.positions[neighborCenter * 3 + 2]!,
    ];

    // 2 vertices per line (center → neighbor)
    const positions = new Float32Array(neighborIndices.length * 6);
    const opacities = new Float32Array(neighborIndices.length * 2);

    for (let i = 0; i < neighborIndices.length; i++) {
      const ni = neighborIndices[i]!;
      const nPos: [number, number, number] = [
        dataset.positions[ni * 3]!,
        dataset.positions[ni * 3 + 1]!,
        dataset.positions[ni * 3 + 2]!,
      ];

      const offset = i * 6;
      positions[offset] = centerPos[0];
      positions[offset + 1] = centerPos[1];
      positions[offset + 2] = centerPos[2];
      positions[offset + 3] = nPos[0];
      positions[offset + 4] = nPos[1];
      positions[offset + 5] = nPos[2];

      // Distance fade: closer neighbors = more opaque
      const dx = nPos[0] - centerPos[0];
      const dy = nPos[1] - centerPos[1];
      const dz = nPos[2] - centerPos[2];
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const fade = Math.max(0.15, 1 - dist / 50);
      opacities[i * 2] = fade;
      opacities[i * 2 + 1] = fade * 0.5;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    return geo;
  }, [dataset, neighborIndices, neighborCenter]);

  if (!geometry) return null;

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color="#ffffff" transparent opacity={0.25} />
    </lineSegments>
  );
}

import { useViewerStore } from '../store/useViewerStore';

const RINGS = [
  { radius: 5, color: '#22c55e', label: 'Close neighbors' },   // green
  { radius: 15, color: '#f59e0b', label: 'Related' },           // amber
] as const;

export function DistanceRings() {
  const selectedIndex = useViewerStore((s) => s.selectedIndex);
  const dataset = useViewerStore((s) => s.dataset);
  const spaceScale = useViewerStore((s) => s.spaceScale);

  if (selectedIndex == null || !dataset) return null;

  const x = dataset.positions[selectedIndex * 3]!;
  const y = dataset.positions[selectedIndex * 3 + 1]!;
  const z = dataset.positions[selectedIndex * 3 + 2]!;

  return (
    <group position={[x, y, z]}>
      {RINGS.map((ring) => (
        <mesh key={ring.radius} scale={[1 / spaceScale, 1 / spaceScale, 1 / spaceScale]}>
          <sphereGeometry args={[ring.radius, 32, 16]} />
          <meshBasicMaterial
            wireframe
            transparent
            opacity={0.06}
            color={ring.color}
            depthWrite={false}
            fog={false}
          />
        </mesh>
      ))}
    </group>
  );
}

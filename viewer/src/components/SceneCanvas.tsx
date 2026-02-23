import { useState, useEffect, useMemo } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Stats } from '@react-three/drei';
import * as THREE from 'three';
import { PointCloud } from './PointCloud';
import { PointLabel } from './PointLabel';
import { ClusterLabels } from './ClusterLabels';
import { CameraAnimator } from './CameraAnimator';
import { ScrollZoom } from './ScrollZoom';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import { NeighborLines } from './NeighborLines';
import { IntroAnimation } from './IntroAnimation';
import { FlyControls } from './FlyControls';
import { DistanceRings } from './DistanceRings';
import { CameraLight } from './CameraLight';
import { BookmarkRestore } from './BookmarkRestore';
import { useViewerStore } from '../store/useViewerStore';

const FOG_COLOR = '#0a0a0a';
const NUM_POINTS_FOG_THRESHOLD = 50000;

/** Ctrl+drag swaps orbit ↔ pan on OrbitControls (trackpad-friendly). */
function CtrlPanSwap() {
  const { controls } = useThree();

  useEffect(() => {
    if (!controls) return;
    const orbit = controls as unknown as { mouseButtons: { LEFT: number; RIGHT: number } };

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Control') {
        orbit.mouseButtons.LEFT = THREE.MOUSE.PAN;
        orbit.mouseButtons.RIGHT = THREE.MOUSE.ROTATE;
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.key === 'Control') {
        orbit.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
        orbit.mouseButtons.RIGHT = THREE.MOUSE.PAN;
      }
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      orbit.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
      orbit.mouseButtons.RIGHT = THREE.MOUSE.PAN;
    };
  }, [controls]);

  return null;
}

export function SceneCanvas() {
  const dataset = useViewerStore((s) => s.dataset);
  const spaceScale = useViewerStore((s) => s.spaceScale);
  const controlMode = useViewerStore((s) => s.controlMode);
  const [showStats, setShowStats] = useState(false);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === '`') setShowStats((s) => !s);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  // Dynamic fog from point count + coordinate extent
  const { fogNear, fogFar } = useMemo(() => {
    if (!dataset) return { fogNear: 60, fogFar: 200 };

    const n = dataset.n_points;
    let maxDist = 0;
    for (let i = 0; i < n; i++) {
      const x = dataset.positions[i * 3]!;
      const y = dataset.positions[i * 3 + 1]!;
      const z = dataset.positions[i * 3 + 2]!;
      const dist = Math.sqrt(x * x + y * y + z * z);
      if (dist > maxDist) maxDist = dist;
    }

    const multiplier = 2 - Math.min(n, NUM_POINTS_FOG_THRESHOLD) / NUM_POINTS_FOG_THRESHOLD;
    const isLarge = n > 50000;
    return {
      fogNear: isLarge ? maxDist * 0.3 : maxDist * 1.2,
      fogFar: isLarge ? maxDist * 3.0 : maxDist * 4 * multiplier,
    };
  }, [dataset]);

  if (!dataset) return null;

  return (
    <div className="fixed inset-0">
      <Canvas
        camera={{ position: [0, 0, 120], fov: 60, near: 0.1, far: 1500 }}
        gl={{ antialias: true }}
        style={{ background: FOG_COLOR }}
      >
        <fog attach="fog" args={[FOG_COLOR, fogNear * spaceScale, fogFar * spaceScale]} />
        <ambientLight intensity={0.4} />
        <directionalLight position={[50, 50, 50]} intensity={0.5} />
        <directionalLight position={[-50, -30, -50]} intensity={0.2} />
        <CameraLight />
        <group scale={[spaceScale, spaceScale, spaceScale]}>
          <PointCloud />
          <NeighborLines />
          <DistanceRings />
          <PointLabel />
          <ClusterLabels />
        </group>
        <CameraAnimator />
        <OrbitControls
          makeDefault
          enabled={controlMode === 'orbit'}
          enableDamping={false}
          rotateSpeed={0.6}
          panSpeed={0.7}
          minDistance={0}
          maxDistance={500}
          enableZoom={false}
        />
        <CtrlPanSwap />
        <IntroAnimation />
        <ScrollZoom />
        <FlyControls />
        <EffectComposer>
          <Bloom
            luminanceThreshold={0.8}
            luminanceSmoothing={0.3}
            intensity={0.6}
            radius={0.4}
          />
        </EffectComposer>
        <BookmarkRestore />
        {showStats && <Stats />}
      </Canvas>
    </div>
  );
}

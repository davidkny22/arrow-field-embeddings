import { useRef, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { useViewerStore } from '../store/useViewerStore';
import { easeOutCubic, clamp } from '../utils/math';

const OFFSET_DISTANCE = 5;
const SETTLE_FRAMES = 10;
const AUTO_ROTATE_SPEED = 1.0;

interface AnimationState {
  startPos: THREE.Vector3;
  startTarget: THREE.Vector3;
  endPos: THREE.Vector3;
  endTarget: THREE.Vector3;
  duration: number;
  elapsed: number;
  settleCount: number;
}

export function CameraAnimator() {
  const { camera, controls, gl } = useThree();
  const animRef = useRef<AnimationState | null>(null);
  const interruptedRef = useRef(false);
  const autoOrbitRef = useRef(true);

  const flyToTarget = useViewerStore((s) => s.flyToTarget);
  const flyToState = useViewerStore((s) => s.flyToState);
  const dataset = useViewerStore((s) => s.dataset);

  // Enable auto-orbit when a new dataset loads (temporarily enables orbit controls)
  useEffect(() => {
    if (!dataset || !controls) return;
    const orbitControls = controls as unknown as { autoRotate: boolean; autoRotateSpeed: number; enabled: boolean };
    autoOrbitRef.current = true;
    orbitControls.enabled = true;
    orbitControls.autoRotate = true;
    orbitControls.autoRotateSpeed = AUTO_ROTATE_SPEED;
  }, [dataset, controls]);

  // Stop auto-orbit on first user interaction (mouse, scroll, or keyboard)
  useEffect(() => {
    const canvas = gl.domElement;

    const stopAutoOrbit = () => {
      if (!autoOrbitRef.current || !controls) return;
      autoOrbitRef.current = false;
      const orbitControls = controls as unknown as { autoRotate: boolean; enabled: boolean };
      orbitControls.autoRotate = false;
      // Restore the user's chosen control mode (orbit controls may need to disable for fly)
      const mode = useViewerStore.getState().controlMode;
      if (mode === 'fly') orbitControls.enabled = false;
    };

    canvas.addEventListener('pointerdown', stopAutoOrbit);
    canvas.addEventListener('wheel', stopAutoOrbit);
    window.addEventListener('keydown', stopAutoOrbit);
    return () => {
      canvas.removeEventListener('pointerdown', stopAutoOrbit);
      canvas.removeEventListener('wheel', stopAutoOrbit);
      window.removeEventListener('keydown', stopAutoOrbit);
    };
  }, [gl, controls]);

  // Detect user interruption during fly-to animation
  useEffect(() => {
    const handlePointerDown = () => {
      if (flyToState === 'animating') {
        interruptedRef.current = true;
      }
    };
    const handleWheel = () => {
      if (flyToState === 'animating') {
        interruptedRef.current = true;
      }
    };

    window.addEventListener('pointerdown', handlePointerDown);
    window.addEventListener('wheel', handleWheel);
    return () => {
      window.removeEventListener('pointerdown', handlePointerDown);
      window.removeEventListener('wheel', handleWheel);
    };
  }, [flyToState]);

  // Start fly-to animation — also stops auto-orbit
  useEffect(() => {
    if (!flyToTarget || flyToState !== 'animating') return;

    // Stop auto-orbit when flying to a point
    if (autoOrbitRef.current && controls) {
      autoOrbitRef.current = false;
      const orbitControls = controls as unknown as { autoRotate: boolean };
      orbitControls.autoRotate = false; // eslint-disable-line react-hooks/immutability
    }

    // flyToTarget is in data coordinates; multiply by spaceScale for world-space
    const s = useViewerStore.getState().spaceScale;
    const destination = new THREE.Vector3(...flyToTarget).multiplyScalar(s);
    const currentPos = camera.position.clone();

    const direction = currentPos.clone().sub(destination).normalize();
    if (direction.length() < 0.001) {
      direction.set(0, 0, 1);
    }

    const endPos = destination.clone().add(direction.multiplyScalar(OFFSET_DISTANCE));
    const distance = currentPos.distanceTo(endPos);
    const duration = clamp(distance * 0.03, 0.8, 3.0);

    const currentTarget = new THREE.Vector3();
    camera.getWorldDirection(currentTarget);
    currentTarget.multiplyScalar(OFFSET_DISTANCE).add(currentPos);

    animRef.current = {
      startPos: currentPos,
      startTarget: currentTarget,
      endPos,
      endTarget: destination,
      duration,
      elapsed: 0,
      settleCount: 0,
    };

    // Start pulsing the target point immediately so it's visible during flight
    const store = useViewerStore.getState();
    if (store.dataset) {
      const [tx, ty, tz] = flyToTarget;
      let bestIdx = -1;
      let bestDist = Infinity;
      for (let i = 0; i < store.dataset.n_points; i++) {
        const px = store.dataset.positions[i * 3]!;
        const py = store.dataset.positions[i * 3 + 1]!;
        const pz = store.dataset.positions[i * 3 + 2]!;
        const d = (px - tx) ** 2 + (py - ty) ** 2 + (pz - tz) ** 2;
        if (d < bestDist) { bestDist = d; bestIdx = i; }
      }
      if (bestIdx >= 0 && bestDist < 4) {
        store.setPulseIndex(bestIdx);
      }
    }

    interruptedRef.current = false;
  }, [flyToTarget, flyToState, camera, controls]);

  useFrame((state, delta) => {
    const anim = animRef.current;
    if (!anim) return;

    const currentFlyToState = useViewerStore.getState().flyToState;

    if (interruptedRef.current) {
      animRef.current = null;
      interruptedRef.current = false;
      useViewerStore.getState().cancelFlyTo();
      return;
    }

    if (currentFlyToState === 'animating') {
      anim.elapsed += delta;
      const t = clamp(anim.elapsed / anim.duration, 0, 1);
      const eased = easeOutCubic(t);

      camera.position.lerpVectors(anim.startPos, anim.endPos, eased);

      const lookTarget = new THREE.Vector3().lerpVectors(
        anim.startTarget,
        anim.endTarget,
        eased
      );
      camera.lookAt(lookTarget);

      const ctrl = state.controls as { target?: THREE.Vector3 } | null;
      if (ctrl?.target) {
        ctrl.target.copy(lookTarget);
      }

      if (t >= 1) {
        useViewerStore.getState().setFlyToState('settling');
      }
    } else if (currentFlyToState === 'settling') {
      anim.settleCount++;
      if (anim.settleCount >= SETTLE_FRAMES) {
        animRef.current = null;
        const store = useViewerStore.getState();
        store.setFlyToState('idle');

        // Auto-select the nearest point to the fly target
        if (store.flyToTarget && store.dataset) {
          const [tx, ty, tz] = store.flyToTarget;
          let bestIdx = -1;
          let bestDist = Infinity;
          for (let i = 0; i < store.dataset.n_points; i++) {
            const px = store.dataset.positions[i * 3]!;
            const py = store.dataset.positions[i * 3 + 1]!;
            const pz = store.dataset.positions[i * 3 + 2]!;
            const d = (px - tx) ** 2 + (py - ty) ** 2 + (pz - tz) ** 2;
            if (d < bestDist) {
              bestDist = d;
              bestIdx = i;
            }
          }
          if (bestIdx >= 0 && bestDist < 4) {
            store.selectPoint(bestIdx);
          }
        }
      }
    }
  });

  return null;
}

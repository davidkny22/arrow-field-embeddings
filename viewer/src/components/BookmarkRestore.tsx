import { useEffect, useRef } from 'react';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { useViewerStore } from '../store/useViewerStore';
import { decodeBookmark } from '../systems/bookmark';

/**
 * Restores visualization state from a URL hash bookmark.
 * Runs once after the dataset loads. Skips intro animation when restoring.
 * Must be placed inside the R3F Canvas to access camera/controls.
 */
export function BookmarkRestore() {
  const { camera, controls } = useThree();
  const applied = useRef(false);
  const dataset = useViewerStore((s) => s.dataset);
  const datasetUrl = useViewerStore((s) => s.datasetUrl);

  useEffect(() => {
    applied.current = false;
  }, [datasetUrl]);

  useEffect(() => {
    if (applied.current || !dataset) return;

    const bookmark = decodeBookmark(window.location.hash);
    if (!bookmark) return;

    // Only apply if the bookmark matches the loaded dataset
    if (bookmark.datasetUrl !== datasetUrl) return;

    applied.current = true;

    const store = useViewerStore.getState();

    // Skip intro animation — jump straight to final state
    store.setIntroState('done');

    // Apply camera position + target
    camera.position.set(...bookmark.cameraPos);
    camera.lookAt(new THREE.Vector3(...bookmark.cameraTarget));

    // Sync OrbitControls target
    const ctrl = controls as unknown as { target?: THREE.Vector3 } | null;
    if (ctrl?.target) {
      ctrl.target.set(...bookmark.cameraTarget);
    }

    // Apply state
    store.setColorMode(bookmark.colorMode);
    store.setControlMode(bookmark.controlMode);

    // Apply scale (set directly since cycleSpaceScale only cycles)
    if (bookmark.spaceScale !== store.spaceScale) {
      // Cycle until we hit the right scale
      const scales = [0.5, 1, 1.5, 2];
      const targetIdx = scales.indexOf(bookmark.spaceScale);
      const currentIdx = scales.indexOf(store.spaceScale);
      if (targetIdx >= 0 && currentIdx >= 0) {
        let steps = (targetIdx - currentIdx + scales.length) % scales.length;
        while (steps-- > 0) store.cycleSpaceScale();
      }
    }

    // Select point by index if specified
    if (bookmark.selectedIndex != null && bookmark.selectedIndex < dataset.n_points) {
      store.selectPoint(bookmark.selectedIndex);
    }

    // Clear hash after applying so refreshing starts fresh
    // Use replaceState to avoid adding a history entry
    window.history.replaceState(null, '', window.location.pathname + window.location.search);
  }, [dataset, datasetUrl, camera, controls]);

  return null;
}

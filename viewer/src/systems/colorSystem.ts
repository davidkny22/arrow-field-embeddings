import type { AFEDataset, ClusterData, ColorMode } from '../types/dataset';
import { hslToRgb } from '../utils/color';

export interface ColorParams {
  clusterPalette?: Map<number, [number, number, number]>;
  highlightedIndices?: Set<number>;
  dimColor?: [number, number, number];
  neighborIndices?: number[];
  neighborCenter?: number | null;
  // Arrow-specific
  activeArrowIndex?: number | 'all';
  arrowData?: number[];       // flat arrows array from dataset
  nArrows?: number;
  // Reconstruction
  reconError?: number[];
}

const NOISE_COLOR: [number, number, number] = [0.55, 0.55, 0.55];
const DIM_COLOR: [number, number, number] = [0.12, 0.12, 0.15];
const GOLDEN_ANGLE = 137.508;

export function buildClusterPalette(
  clusters: ClusterData[],
): Map<number, [number, number, number]> {
  const palette = new Map<number, [number, number, number]>();
  clusters.forEach((c, i) => {
    const hue = (i * GOLDEN_ANGLE) % 360;
    const [r, g, b] = hslToRgb(hue / 360, 0.8, 0.65);
    palette.set(c.id, [r, g, b]);
  });
  palette.set(-1, NOISE_COLOR);
  return palette;
}

/** Viridis-like colormap: blue → green → yellow. */
function viridis(t: number): [number, number, number] {
  // Simplified viridis approximation (0→dark purple, 0.5→teal, 1→yellow)
  const r = Math.max(0, Math.min(1, -0.3 + 2.8 * t * t));
  const g = Math.max(0, Math.min(1, -0.15 + 1.5 * t - 0.5 * t * t));
  const b = Math.max(0, Math.min(1, 0.55 - 0.8 * t + 0.4 * t * t));
  return [r, g, b];
}

export function computeColors(
  dataset: AFEDataset,
  mode: ColorMode,
  params: ColorParams,
): Float32Array {
  const n = dataset.n_points;
  const colors = new Float32Array(n * 3);
  const palette = params.clusterPalette ?? buildClusterPalette(dataset.clusters);

  if (mode === 'cluster') {
    for (let i = 0; i < n; i++) {
      const labelIdx = dataset.label_indices[i]!;
      const rgb = palette.get(labelIdx) ?? NOISE_COLOR;
      colors[i * 3] = rgb[0];
      colors[i * 3 + 1] = rgb[1];
      colors[i * 3 + 2] = rgb[2];
    }
  } else if (mode === 'highlight') {
    const highlighted = params.highlightedIndices ?? new Set<number>();
    const dim = params.dimColor ?? DIM_COLOR;

    for (let i = 0; i < n; i++) {
      if (highlighted.has(i)) {
        const labelIdx = dataset.label_indices[i]!;
        const rgb = palette.get(labelIdx) ?? NOISE_COLOR;
        colors[i * 3] = rgb[0];
        colors[i * 3 + 1] = rgb[1];
        colors[i * 3 + 2] = rgb[2];
      } else {
        colors[i * 3] = dim[0];
        colors[i * 3 + 1] = dim[1];
        colors[i * 3 + 2] = dim[2];
      }
    }
  } else if (mode === 'neighborhood') {
    const neighborSet = new Set(params.neighborIndices ?? []);
    const center = params.neighborCenter;
    const dim = params.dimColor ?? DIM_COLOR;

    for (let i = 0; i < n; i++) {
      if (center != null && i === center) {
        colors[i * 3] = 1.0;
        colors[i * 3 + 1] = 1.0;
        colors[i * 3 + 2] = 1.0;
      } else if (neighborSet.has(i)) {
        const labelIdx = dataset.label_indices[i]!;
        const rgb = palette.get(labelIdx) ?? NOISE_COLOR;
        colors[i * 3] = rgb[0];
        colors[i * 3 + 1] = rgb[1];
        colors[i * 3 + 2] = rgb[2];
      } else {
        colors[i * 3] = dim[0];
        colors[i * 3 + 1] = dim[1];
        colors[i * 3 + 2] = dim[2];
      }
    }
  } else if (mode === 'arrow_magnitude') {
    // Color by current arrow's magnitude (r component) using viridis
    const arrows = params.arrowData;
    const k = params.nArrows ?? 1;
    const arrowIdx =
      params.activeArrowIndex === 'all' ? 0 : (params.activeArrowIndex ?? 0);

    if (arrows && arrows.length === n * k * 3) {
      // Find min/max magnitude for normalization
      let minR = Infinity;
      let maxR = -Infinity;
      for (let i = 0; i < n; i++) {
        const r = arrows[i * k * 3 + arrowIdx * 3 + 2]!;
        if (r < minR) minR = r;
        if (r > maxR) maxR = r;
      }
      const range = maxR - minR || 1;

      for (let i = 0; i < n; i++) {
        const r = arrows[i * k * 3 + arrowIdx * 3 + 2]!;
        const t = (r - minR) / range;
        const [cr, cg, cb] = viridis(t);
        colors[i * 3] = cr;
        colors[i * 3 + 1] = cg;
        colors[i * 3 + 2] = cb;
      }
    } else {
      // Fallback to cluster
      for (let i = 0; i < n; i++) {
        const labelIdx = dataset.label_indices[i]!;
        const rgb = palette.get(labelIdx) ?? NOISE_COLOR;
        colors[i * 3] = rgb[0];
        colors[i * 3 + 1] = rgb[1];
        colors[i * 3 + 2] = rgb[2];
      }
    }
  } else if (mode === 'arrow_direction') {
    // Color by arrow direction: hue from θ (azimuth), brightness from φ (elevation)
    const arrows = params.arrowData;
    const k = params.nArrows ?? 1;
    const arrowIdx =
      params.activeArrowIndex === 'all' ? 0 : (params.activeArrowIndex ?? 0);

    if (arrows && arrows.length === n * k * 3) {
      for (let i = 0; i < n; i++) {
        const theta = arrows[i * k * 3 + arrowIdx * 3]!;       // [-π, π]
        const phi = arrows[i * k * 3 + arrowIdx * 3 + 1]!;     // [-π/2, π/2]
        const hue = (theta + Math.PI) / (2 * Math.PI);          // [0, 1]
        const lightness = 0.3 + 0.4 * ((phi + Math.PI / 2) / Math.PI); // [0.3, 0.7]
        const [r, g, b] = hslToRgb(hue, 0.85, lightness);
        colors[i * 3] = r;
        colors[i * 3 + 1] = g;
        colors[i * 3 + 2] = b;
      }
    } else {
      for (let i = 0; i < n; i++) {
        const labelIdx = dataset.label_indices[i]!;
        const rgb = palette.get(labelIdx) ?? NOISE_COLOR;
        colors[i * 3] = rgb[0];
        colors[i * 3 + 1] = rgb[1];
        colors[i * 3 + 2] = rgb[2];
      }
    }
  } else if (mode === 'recon_error') {
    // Green (low error) → Red (high error)
    const errors = params.reconError;
    if (errors && errors.length === n) {
      let maxErr = 0;
      for (let i = 0; i < n; i++) {
        if (errors[i]! > maxErr) maxErr = errors[i]!;
      }
      const scale = maxErr || 1;

      for (let i = 0; i < n; i++) {
        const t = Math.min(errors[i]! / scale, 1.0);
        // Green → Yellow → Red
        colors[i * 3] = t;                            // R: 0→1
        colors[i * 3 + 1] = 1.0 - t * 0.8;           // G: 1→0.2
        colors[i * 3 + 2] = 0.1;                      // B: constant low
      }
    } else {
      for (let i = 0; i < n; i++) {
        const labelIdx = dataset.label_indices[i]!;
        const rgb = palette.get(labelIdx) ?? NOISE_COLOR;
        colors[i * 3] = rgb[0];
        colors[i * 3 + 1] = rgb[1];
        colors[i * 3 + 2] = rgb[2];
      }
    }
  } else {
    // Unknown mode — fallback to cluster
    for (let i = 0; i < n; i++) {
      const labelIdx = dataset.label_indices[i]!;
      const rgb = palette.get(labelIdx) ?? NOISE_COLOR;
      colors[i * 3] = rgb[0];
      colors[i * 3 + 1] = rgb[1];
      colors[i * 3 + 2] = rgb[2];
    }
  }

  return colors;
}

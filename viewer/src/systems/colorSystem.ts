import type { AFEDataset, ClusterData, ColorMode } from '../types/dataset';
import { hslToRgb } from '../utils/color';

export type PaletteName = 'golden' | 'okabe' | 'tableau';

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
  // Palette
  palette?: PaletteName;
}

const NOISE_COLOR: [number, number, number] = [0.55, 0.55, 0.55];
const DIM_COLOR: [number, number, number] = [0.12, 0.12, 0.15];
const GOLDEN_ANGLE = 137.508;

// Okabe-Ito colorblind-safe palette (normalized 0–1)
const OKABE_ITO: [number, number, number][] = [
  [0.902, 0.624, 0.000],   // orange
  [0.337, 0.706, 0.914],   // sky blue
  [0.000, 0.620, 0.451],   // bluish green
  [0.941, 0.894, 0.259],   // yellow
  [0.000, 0.447, 0.698],   // blue
  [0.835, 0.369, 0.000],   // vermillion
  [0.800, 0.475, 0.655],   // reddish purple
];

// Tableau 10 palette (normalized 0–1)
const TABLEAU10: [number, number, number][] = [
  [0.306, 0.475, 0.655],   // blue
  [0.949, 0.557, 0.169],   // orange
  [0.882, 0.341, 0.349],   // red
  [0.463, 0.718, 0.698],   // cyan
  [0.349, 0.631, 0.310],   // green
  [0.929, 0.788, 0.282],   // yellow
  [0.690, 0.478, 0.631],   // purple
  [1.000, 0.616, 0.655],   // pink
  [0.612, 0.459, 0.373],   // brown
  [0.729, 0.690, 0.675],   // gray
];

export function buildClusterPalette(
  clusters: ClusterData[],
  paletteName: PaletteName = 'golden',
): Map<number, [number, number, number]> {
  const palette = new Map<number, [number, number, number]>();

  if (paletteName === 'okabe') {
    clusters.forEach((c, i) => {
      const rgb = OKABE_ITO[i % OKABE_ITO.length]!;
      palette.set(c.id, rgb);
    });
  } else if (paletteName === 'tableau') {
    clusters.forEach((c, i) => {
      const rgb = TABLEAU10[i % TABLEAU10.length]!;
      palette.set(c.id, rgb);
    });
  } else {
    clusters.forEach((c, i) => {
      const hue = (i * GOLDEN_ANGLE) % 360;
      const [r, g, b] = hslToRgb(hue / 360, 0.8, 0.65);
      palette.set(c.id, [r, g, b]);
    });
  }

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

/** Blue-orange colorblind-safe gradient. */
function blueOrange(t: number): [number, number, number] {
  const r = 0.05 + 0.90 * t;
  const g = 0.25 + 0.20 * t;
  const b = 0.65 - 0.60 * t;
  return [Math.max(0, Math.min(1, r)), Math.max(0, Math.min(1, g)), Math.max(0, Math.min(1, b))];
}

export function computeColors(
  dataset: AFEDataset,
  mode: ColorMode,
  params: ColorParams,
): Float32Array {
  const n = dataset.n_points;
  const colors = new Float32Array(n * 3);
  const palette = params.clusterPalette ?? buildClusterPalette(dataset.clusters, params.palette);

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
    // Blue (low error) → Orange (high error) — colorblind-safe
    const errors = params.reconError;
    if (errors && errors.length === n) {
      let maxErr = 0;
      for (let i = 0; i < n; i++) {
        if (errors[i]! > maxErr) maxErr = errors[i]!;
      }
      const scale = maxErr || 1;

      for (let i = 0; i < n; i++) {
        const t = Math.min(errors[i]! / scale, 1.0);
        const [cr, cg, cb] = blueOrange(t);
        colors[i * 3] = cr;
        colors[i * 3 + 1] = cg;
        colors[i * 3 + 2] = cb;
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

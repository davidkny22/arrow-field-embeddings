import { create } from 'zustand';
import type { AFEDataset, ColorMode, DatasetEntry, ClusterData } from '../types/dataset';

/**
 * Compute how much to spread the point cloud so arrows and orbs don't overlap.
 *
 * 1. Sample ~500 points, brute-force median NN distance
 * 2. Compute arrow auto-scale (median magnitude → 3% of diameter)
 * 3. If the p90 arrow world-length exceeds half the median NN gap, expand space
 */
function computeAutoSpaceScale(dataset: AFEDataset): number {
  const { positions, arrows, n_points: n, n_arrows: k } = dataset;
  if (n < 2 || k === 0) return 1;

  // --- Median NN distance (sampled) ---
  const nSamples = Math.min(500, n);
  const sampleStep = Math.max(1, Math.floor(n / nSamples));
  const sIdx: number[] = [];
  for (let i = 0; i < n; i += sampleStep) sIdx.push(i);

  const nnDists: number[] = [];
  for (let a = 0; a < sIdx.length; a++) {
    const ai = sIdx[a]!;
    const ax = positions[ai * 3]!, ay = positions[ai * 3 + 1]!, az = positions[ai * 3 + 2]!;
    let best = Infinity;
    for (let b = 0; b < sIdx.length; b++) {
      if (a === b) continue;
      const bi = sIdx[b]!;
      const dx = ax - positions[bi * 3]!;
      const dy = ay - positions[bi * 3 + 1]!;
      const dz = az - positions[bi * 3 + 2]!;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < best) best = d2;
    }
    if (best < Infinity) nnDists.push(Math.sqrt(best));
  }
  if (nnDists.length === 0) return 1;
  nnDists.sort((a, b) => a - b);
  const medianNN = nnDists[Math.floor(nnDists.length / 2)]!;
  if (medianNN < 1e-10) return 1;

  // --- Arrow auto-scale (same formula as ArrowField) ---
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = positions[i * 3]!, y = positions[i * 3 + 1]!, z = positions[i * 3 + 2]!;
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  const dx = maxX - minX, dy = maxY - minY, dz = maxZ - minZ;
  const diameter = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (diameter < 1e-10) return 1;

  const magStep = Math.max(1, Math.floor(n / 2000));
  const mags: number[] = [];
  for (let i = 0; i < n; i += magStep) {
    for (let j = 0; j < k; j++) {
      const r = Math.abs(arrows[i * k * 3 + j * 3 + 2]!);
      if (r > 1e-8) mags.push(r);
    }
  }
  if (mags.length === 0) return 1;
  mags.sort((a, b) => a - b);
  const medianMag = mags[Math.floor(mags.length / 2)]!;
  if (medianMag < 1e-10) return 1;

  const arrowAutoScale = (diameter * 0.02) / medianMag;

  // p90 arrow world-space length
  const p90Mag = mags[Math.floor(mags.length * 0.9)]!;
  const p90Length = p90Mag * arrowAutoScale;

  // If p90 arrow > half the median NN gap, expand space
  const halfGap = medianNN * 0.5;
  if (p90Length <= halfGap) return 1;
  return p90Length / halfGap;
}

interface ViewerState {
  // Data
  availableDatasets: DatasetEntry[];
  datasetUrl: string;
  dataset: AFEDataset | null;
  loading: boolean;
  error: string | null;

  // Selection
  selectedIndex: number | null;
  hoveredIndex: number | null;

  // Search
  highlightedIndices: Set<number>;
  searchQuery: string;

  // Camera
  flyToTarget: [number, number, number] | null;
  flyToState: 'idle' | 'animating' | 'settling';
  controlMode: 'orbit' | 'fly';
  spaceScale: number;
  autoSpaceScale: number;

  // Color
  colorMode: ColorMode;

  // Neighborhood
  neighborIndices: number[];
  neighborCenter: number | null;

  // Lookup maps (computed when dataset loads)
  labelToIndices: Map<number, number[]>;

  // Intro animation
  introState: 'pending' | 'animating' | 'done';

  // Pulse effect on fly-to target
  pulseIndex: number | null;

  // Arrow state
  activeArrowIndex: number | 'all';
  arrowScale: number;
  arrowDensity: number; // 1 = every point, 2 = every 2nd point, etc.
  arrowsVisible: boolean;

  // Clicked arrow info (for arrow info panel)
  clickedArrow: { arrowIdx: number; pointIdx: number } | null;

  // Actions
  setAvailableDatasets: (datasets: DatasetEntry[]) => void;
  setDatasetUrl: (url: string) => void;
  setDataset: (dataset: AFEDataset) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  selectPoint: (index: number | null) => void;
  hoverPoint: (index: number | null) => void;
  setHighlightedIndices: (indices: Set<number>) => void;
  setSearchQuery: (query: string) => void;
  flyTo: (target: [number, number, number]) => void;
  cancelFlyTo: () => void;
  setFlyToState: (state: 'idle' | 'animating' | 'settling') => void;
  setColorMode: (mode: ColorMode) => void;
  setControlMode: (mode: 'orbit' | 'fly') => void;
  setNeighborhood: (center: number | null, indices: number[]) => void;
  setPulseIndex: (index: number | null) => void;
  cycleSpaceScale: () => void;
  setIntroState: (state: 'pending' | 'animating' | 'done') => void;

  // Arrow actions
  setActiveArrowIndex: (index: number | 'all') => void;
  nextArrow: () => void;
  prevArrow: () => void;
  setArrowScale: (scale: number) => void;
  setArrowDensity: (density: number) => void;
  setArrowsVisible: (visible: boolean) => void;
  setClickedArrow: (info: { arrowIdx: number; pointIdx: number } | null) => void;
}

export const useViewerStore = create<ViewerState>((set, get) => ({
  availableDatasets: [],
  datasetUrl: '',
  dataset: null,
  loading: true,
  error: null,

  selectedIndex: null,
  hoveredIndex: null,

  highlightedIndices: new Set<number>(),
  searchQuery: '',

  flyToTarget: null,
  flyToState: 'idle',
  controlMode: 'fly',
  spaceScale: 0.5,
  autoSpaceScale: 1,

  colorMode: 'cluster',

  neighborIndices: [],
  neighborCenter: null,

  labelToIndices: new Map<number, number[]>(),

  introState: 'pending',
  pulseIndex: null,

  activeArrowIndex: 0,
  arrowScale: 1.0,
  arrowDensity: 1,
  arrowsVisible: true,
  clickedArrow: null,

  setAvailableDatasets: (datasets) =>
    set((state) => {
      const needsDefault = !state.datasetUrl && datasets.length > 0;
      return {
        availableDatasets: datasets,
        ...(needsDefault ? { datasetUrl: datasets[0]!.url } : {}),
      };
    }),

  setDatasetUrl: (url) =>
    set({
      datasetUrl: url,
      dataset: null,
      loading: true,
      error: null,
      selectedIndex: null,
      hoveredIndex: null,
      highlightedIndices: new Set<number>(),
      searchQuery: '',
      flyToTarget: null,
      flyToState: 'idle',
      colorMode: 'cluster',
      introState: 'pending',
      neighborIndices: [],
      neighborCenter: null,
      activeArrowIndex: 0,
    }),

  setDataset: (dataset) => {
    const labelToIndices = new Map<number, number[]>();
    for (let i = 0; i < dataset.n_points; i++) {
      const labelIdx = dataset.label_indices[i]!;
      const arr = labelToIndices.get(labelIdx);
      if (arr) arr.push(i);
      else labelToIndices.set(labelIdx, [i]);
    }
    const autoSpaceScale = computeAutoSpaceScale(dataset);
    return set({
      dataset,
      loading: false,
      error: null,
      introState: 'animating',
      labelToIndices,
      autoSpaceScale,
    });
  },

  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error, loading: false }),

  selectPoint: (index) => set({ selectedIndex: index }),
  hoverPoint: (index) => set({ hoveredIndex: index }),

  setHighlightedIndices: (indices) => set({ highlightedIndices: indices }),
  setSearchQuery: (query) => set({ searchQuery: query }),

  flyTo: (target) => set({ flyToTarget: target, flyToState: 'animating' }),
  cancelFlyTo: () => set({ flyToTarget: null, flyToState: 'idle' }),
  setFlyToState: (state) => set({ flyToState: state }),

  setColorMode: (mode) => set({ colorMode: mode }),
  setControlMode: (mode) => set({ controlMode: mode }),
  setNeighborhood: (center, indices) =>
    set({ neighborCenter: center, neighborIndices: indices }),

  setPulseIndex: (index) => set({ pulseIndex: index }),
  cycleSpaceScale: () =>
    set((s) => {
      const scales = [0.5, 1, 1.5, 2];
      const next = scales[(scales.indexOf(s.spaceScale) + 1) % scales.length]!;
      return { spaceScale: next ?? 0.5 };
    }),
  setIntroState: (state) => set({ introState: state }),

  // Arrow actions
  setActiveArrowIndex: (index) => set({ activeArrowIndex: index }),
  nextArrow: () =>
    set((s) => {
      if (!s.dataset) return {};
      const k = s.dataset.n_arrows;
      if (s.activeArrowIndex === 'all') return { activeArrowIndex: 0 };
      return { activeArrowIndex: (s.activeArrowIndex + 1) % k };
    }),
  prevArrow: () =>
    set((s) => {
      if (!s.dataset) return {};
      const k = s.dataset.n_arrows;
      if (s.activeArrowIndex === 'all') return { activeArrowIndex: k - 1 };
      return { activeArrowIndex: (s.activeArrowIndex - 1 + k) % k };
    }),
  setArrowScale: (scale) => set({ arrowScale: scale }),
  setArrowDensity: (density) => set({ arrowDensity: density }),
  setArrowsVisible: (visible) => set({ arrowsVisible: visible }),
  setClickedArrow: (info) => set({ clickedArrow: info }),
}));

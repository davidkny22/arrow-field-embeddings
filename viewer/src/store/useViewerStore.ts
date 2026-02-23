import { create } from 'zustand';
import type { AFEDataset, ColorMode, DatasetEntry, ClusterData } from '../types/dataset';

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
  spaceScale: 1,

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
    return set({
      dataset,
      loading: false,
      error: null,
      introState: 'animating',
      labelToIndices,
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
      const scales = [0.5, 1, 2, 3];
      const next = scales[(scales.indexOf(s.spaceScale) + 1) % scales.length]!;
      return { spaceScale: next };
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
}));

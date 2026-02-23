import { useMemo, useCallback } from 'react';
import Fuse from 'fuse.js';
import type { AFEDataset, ClusterData } from '../types/dataset';
import { useViewerStore } from '../store/useViewerStore';

interface LabelMatch {
  type: 'label';
  label: string;
  labelIndex: number;
  indices: number[]; // all point indices with this label
}

interface ClusterMatch {
  type: 'cluster';
  cluster: ClusterData;
}

export type SearchResult = LabelMatch | ClusterMatch;

function searchClusters(query: string, clusters: ClusterData[]): ClusterMatch[] {
  const q = query.toLowerCase();
  return clusters
    .filter((c) => c.label.toLowerCase().includes(q))
    .map((c) => ({ type: 'cluster' as const, cluster: c }));
}

export function useSearch(dataset: AFEDataset | null) {
  const fuse = useMemo(() => {
    if (!dataset) return null;
    const items = dataset.label_names.map((name, i) => ({ label: name, labelIndex: i }));
    return new Fuse(items, {
      keys: ['label'],
      threshold: 0.4,
      includeScore: true,
      shouldSort: true,
    });
  }, [dataset]);

  const search = useCallback(
    (query: string): SearchResult[] => {
      if (!query.trim() || !dataset || !fuse) return [];

      const clusterMatches = searchClusters(query, dataset.clusters);
      const labelResults = fuse.search(query, { limit: 20 });

      const labelToIndices = useViewerStore.getState().labelToIndices;
      const labelMatches: LabelMatch[] = labelResults.map((r) => ({
        type: 'label',
        label: r.item.label,
        labelIndex: r.item.labelIndex,
        indices: labelToIndices.get(r.item.labelIndex) ?? [],
      }));

      return [...clusterMatches, ...labelMatches];
    },
    [dataset, fuse],
  );

  const getHighlightIndices = useCallback(
    (results: SearchResult[]): Set<number> => {
      if (!dataset) return new Set();
      const indices = new Set<number>();
      const labelToIndices = useViewerStore.getState().labelToIndices;

      for (const result of results) {
        if (result.type === 'label') {
          for (const i of result.indices) indices.add(i);
        } else {
          // Add all points in this cluster — cluster id maps to label index
          const members = labelToIndices.get(result.cluster.id) ?? [];
          for (const i of members) indices.add(i);
        }
      }

      return indices;
    },
    [dataset],
  );

  return { search, getHighlightIndices };
}

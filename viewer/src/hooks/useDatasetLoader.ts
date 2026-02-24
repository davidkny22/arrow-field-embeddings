import { useEffect } from 'react';
import pako from 'pako';
import type { AFEDataset } from '../types/dataset';
import { useViewerStore } from '../store/useViewerStore';

async function loadDataset(url: string): Promise<AFEDataset> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load dataset: ${response.status} ${response.statusText}`);
  }

  const buffer = await response.arrayBuffer();
  const bytes = new Uint8Array(buffer);

  // Detect gzip magic bytes (0x1f 0x8b)
  const isGzip = bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;

  let jsonString: string;
  if (isGzip) {
    try {
      const decompressed = pako.inflate(bytes);
      jsonString = new TextDecoder().decode(decompressed);
    } catch {
      throw new Error('Failed to decompress dataset — file may be corrupted');
    }
  } else {
    jsonString = new TextDecoder().decode(bytes);
  }

  let data: AFEDataset;
  try {
    data = JSON.parse(jsonString) as AFEDataset;
  } catch {
    throw new Error('Dataset file contains invalid JSON');
  }

  // Validate essential fields
  if (!data.positions?.length) throw new Error('Dataset has no positions');
  if (!data.arrows?.length) throw new Error('Dataset has no arrows');
  if (data.positions.length !== data.n_points * 3) {
    throw new Error(`Position count mismatch: ${data.positions.length} vs expected ${data.n_points * 3}`);
  }
  if (data.arrows.length !== data.n_points * data.n_arrows * 3) {
    throw new Error(`Arrow count mismatch: ${data.arrows.length} vs expected ${data.n_points * data.n_arrows * 3}`);
  }

  return data;
}

export function useDatasetLoader(url: string) {
  const setDataset = useViewerStore((s) => s.setDataset);
  const setLoading = useViewerStore((s) => s.setLoading);
  const setError = useViewerStore((s) => s.setError);

  useEffect(() => {
    if (!url) return;
    let cancelled = false;

    setLoading(true);
    loadDataset(url)
      .then((dataset) => {
        if (!cancelled) setDataset(dataset);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Unknown error loading dataset');
        }
      });

    return () => {
      cancelled = true;
    };
  }, [url, setDataset, setLoading, setError]);
}

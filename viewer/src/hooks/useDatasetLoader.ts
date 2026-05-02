import { useEffect, useRef } from 'react';
import pako from 'pako';
import type { AFEDataset } from '../types/dataset';
import { useViewerStore } from '../store/useViewerStore';

async function loadDataset(url: string, signal: AbortSignal, onProgress: (pct: number) => void): Promise<AFEDataset> {
  const response = await fetch(url, { signal });
  if (!response.ok) {
    throw new Error(`Failed to load dataset: ${response.status} ${response.statusText}`);
  }

  const contentLength = Number(response.headers.get('Content-Length') || '0');
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('ReadableStream not supported');
  }

  const chunks: Uint8Array[] = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (signal.aborted) {
      reader.cancel();
      throw new DOMException('Aborted', 'AbortError');
    }
    chunks.push(value);
    received += value.length;
    if (contentLength > 0) {
      onProgress(Math.min(100, Math.round((received / contentLength) * 100)));
    }
  }

  // Concatenate chunks
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }

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
  const setLoadingProgress = useViewerStore((s) => s.setLoadingProgress);
  const setError = useViewerStore((s) => s.setError);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!url) return;

    // Abort any previous fetch
    if (abortRef.current) {
      abortRef.current.abort();
    }
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setLoadingProgress(0);
    loadDataset(url, controller.signal, setLoadingProgress)
      .then((dataset) => {
        if (!controller.signal.aborted) {
          setLoadingProgress(100);
          setDataset(dataset);
        }
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setError(err instanceof Error ? err.message : 'Unknown error loading dataset');
      });

    return () => {
      controller.abort();
    };
  }, [url, setDataset, setLoading, setLoadingProgress, setError]);
}

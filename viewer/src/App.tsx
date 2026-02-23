import { useEffect } from 'react';
import { useDatasetLoader } from './hooks/useDatasetLoader';
import { SceneCanvas } from './components/SceneCanvas';
import { LoadingScreen } from './components/LoadingScreen';
import { InfoPanel } from './components/InfoPanel';
import { SearchBar } from './components/SearchBar';
import { DatasetSelector } from './components/DatasetSelector';
import { ControlModeToggle } from './components/ControlModeToggle';
import { SpaceScaleToggle } from './components/SpaceScaleToggle';
import { ShareButton } from './components/ShareButton';
import { RectangleSelector } from './components/RectangleSelector';
import { ControlsHint } from './components/ControlsHint';
import { useViewerStore } from './store/useViewerStore';
import type { DatasetEntry } from './types/dataset';

function App() {
  const datasetUrl = useViewerStore((s) => s.datasetUrl);
  useDatasetLoader(datasetUrl);

  // Discover available presets from /presets/index.json
  useEffect(() => {
    fetch('/presets/index.json')
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((entries: DatasetEntry[]) => {
        if (Array.isArray(entries) && entries.length > 0) {
          useViewerStore.getState().setAvailableDatasets(entries);
        } else {
          throw new Error('empty');
        }
      })
      .catch(() => {
        // Try server fallback
        const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        fetch(`${apiUrl}/health`)
          .then((r) => r.json())
          .then((data: { datasets?: string[] }) => {
            if (data.datasets && data.datasets.length > 0) {
              const entries: DatasetEntry[] = data.datasets.map((id) => ({
                id,
                label: id.replace(/[-_]/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
                url: `/presets/${id}.json.gz`,
              }));
              useViewerStore.getState().setAvailableDatasets(entries);
            } else {
              useViewerStore.getState().setError(
                'No datasets found. Generate presets with: python viewer/scripts/generate_presets.py',
              );
            }
          })
          .catch(() => {
            useViewerStore.getState().setError(
              'No datasets found. Generate presets or start the server.',
            );
          });
      });
  }, []);

  return (
    <>
      <LoadingScreen />
      <SceneCanvas />
      <DatasetSelector />
      <SearchBar />
      <InfoPanel />
      <ShareButton />
      <SpaceScaleToggle />
      <ControlModeToggle />
      <ControlsHint />
      <RectangleSelector />
    </>
  );
}

export default App;

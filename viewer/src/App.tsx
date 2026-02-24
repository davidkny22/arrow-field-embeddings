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
import { ArrowControls } from './components/ArrowControls';
import { ArrowInfoPanel } from './components/ArrowInfoPanel';
import { ColorModeSelector } from './components/ColorModeSelector';
import { MetricsBar } from './components/MetricsBar';
import { ReconPanel } from './components/ReconPanel';
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

      {/* Top-left: dataset selector + metrics */}
      <DatasetSelector />
      <MetricsBar />

      {/* Top-center: search */}
      <SearchBar />

      {/* Right side: stacked panels */}
      <div className="fixed right-4 top-4 z-40 flex flex-col gap-3 max-h-[calc(100vh-2rem)] overflow-y-auto pointer-events-none">
        <div className="pointer-events-auto"><InfoPanel /></div>
        <div className="pointer-events-auto"><ArrowInfoPanel /></div>
        <div className="pointer-events-auto"><ReconPanel /></div>
      </div>

      {/* Bottom-left: color modes, controls hint */}
      <ColorModeSelector />
      <ControlsHint />

      {/* Bottom-center: arrow controls */}
      <ArrowControls />

      {/* Bottom-right: scale, mode, share/download */}
      <div className="fixed bottom-4 right-4 z-40 flex items-center gap-2">
        <SpaceScaleToggle />
        <ControlModeToggle />
        <ShareButton />
      </div>

      <RectangleSelector />
    </>
  );
}

export default App;

import { useEffect, useState } from 'react';
import { useDatasetLoader } from './hooks/useDatasetLoader';
import { SceneCanvas, takeScreenshot } from './components/SceneCanvas';
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
import { ShortcutsModal } from './components/ShortcutsModal';
import { useViewerStore } from './store/useViewerStore';
import type { DatasetEntry } from './types/dataset';

function App() {
  const datasetUrl = useViewerStore((s) => s.datasetUrl);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  useDatasetLoader(datasetUrl);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === '?' && !e.shiftKey && !e.ctrlKey && !e.metaKey && !e.altKey) {
        const active = document.activeElement;
        if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA')) return;
        setShortcutsOpen((prev) => !prev);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

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
        <button
          onClick={() => setShortcutsOpen(true)}
          className="rounded-full bg-black/60 px-3 py-1.5 text-xs font-mono text-white/70 backdrop-blur-sm hover:bg-black/80 hover:text-white border border-white/10"
          aria-label="Keyboard shortcuts"
          title="Keyboard shortcuts (?)"
        >
          ?
        </button>
        <button
          onClick={takeScreenshot}
          className="rounded-full bg-black/60 px-3 py-1.5 text-xs font-mono text-white/70 backdrop-blur-sm hover:bg-black/80 hover:text-white border border-white/10"
          aria-label="Download screenshot"
          title="Download screenshot (P)"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M13.997 4a2 2 0 0 1 1.76 1.05l.486.9A2 2 0 0 0 18.003 7H20a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2h1.997a2 2 0 0 0 1.759-1.048l.489-.904A2 2 0 0 1 9.004 4z"/><circle cx="12" cy="13" r="3"/></svg>
        </button>
        <ShareButton />
      </div>

      <RectangleSelector />
      <ShortcutsModal open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
    </>
  );
}

export default App;

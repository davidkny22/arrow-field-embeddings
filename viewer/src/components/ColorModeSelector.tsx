import { useViewerStore } from '../store/useViewerStore';
import type { ColorMode } from '../types/dataset';

interface ModeOption {
  mode: ColorMode;
  label: string;
  title: string;
}

const COLOR_MODES: ModeOption[] = [
  { mode: 'cluster', label: 'Cluster', title: 'Color by cluster label' },
  {
    mode: 'arrow_magnitude',
    label: 'Magnitude',
    title: 'Color by arrow magnitude',
  },
  {
    mode: 'arrow_direction',
    label: 'Direction',
    title: 'Color by arrow direction',
  },
  {
    mode: 'recon_error',
    label: 'Error',
    title: 'Color by reconstruction error',
  },
];

export function ColorModeSelector() {
  const dataset = useViewerStore((s) => s.dataset);
  const colorMode = useViewerStore((s) => s.colorMode);

  if (!dataset) return null;

  return (
    <div className="fixed bottom-20 left-4 z-40 flex flex-col gap-1">
      <span className="text-[10px] font-mono text-white/30 uppercase tracking-wider px-1 mb-0.5">
        Color
      </span>
      <div className="flex flex-col gap-1">
        {COLOR_MODES.map(({ mode, label, title }) => {
          const active = colorMode === mode;
          return (
            <button
              key={mode}
              onClick={() => useViewerStore.getState().setColorMode(mode)}
              className={`rounded-md px-3 py-1.5 text-xs font-mono backdrop-blur-sm border transition-colors duration-150 text-left ${
                active
                  ? 'bg-white/15 text-white border-white/30'
                  : 'bg-black/60 text-white/50 border-white/10 hover:bg-black/80 hover:text-white/70'
              }`}
              title={title}
            >
              {label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

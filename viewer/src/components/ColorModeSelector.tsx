import { useViewerStore } from '../store/useViewerStore';
import type { ColorMode } from '../types/dataset';
import type { PaletteName } from '../systems/colorSystem';

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

const PALETTES: { name: PaletteName; label: string }[] = [
  { name: 'okabe', label: 'Okabe' },
  { name: 'tableau', label: 'Tableau' },
  { name: 'golden', label: 'Golden' },
];

function ColorModeButton({ mode, label, title, active }: ModeOption & { active: boolean }) {
  return (
    <button
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
}

function PaletteButton({ name, label, active }: { name: PaletteName; label: string; active: boolean }) {
  return (
    <button
      onClick={() => useViewerStore.getState().setPalette(name)}
      className={`rounded px-2 py-0.5 text-[10px] font-mono border transition-colors duration-150 ${
        active
          ? 'bg-white/15 text-white border-white/30'
          : 'bg-black/60 text-white/50 border-white/10 hover:bg-black/80 hover:text-white/70'
      }`}
      title={`Use ${label} palette`}
    >
      {label}
    </button>
  );
}

export function ColorModeSelector() {
  const dataset = useViewerStore((s) => s.dataset);
  const colorMode = useViewerStore((s) => s.colorMode);
  const palette = useViewerStore((s) => s.palette);

  if (!dataset) return null;

  return (
    <div className="fixed bottom-20 left-4 z-40 flex flex-col gap-1">
      <span className="text-[10px] font-mono text-white/30 uppercase tracking-wider px-1 mb-0.5">
        Color
      </span>
      <div className="flex flex-col gap-1">
        {COLOR_MODES.map((option) => (
          <ColorModeButton
            key={option.mode}
            {...option}
            active={colorMode === option.mode}
          />
        ))}
      </div>
      <div className="flex gap-1 mt-1">
        {PALETTES.map((p) => (
          <PaletteButton
            key={p.name}
            name={p.name}
            label={p.label}
            active={palette === p.name}
          />
        ))}
      </div>
    </div>
  );
}

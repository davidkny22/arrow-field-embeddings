import { useViewerStore } from '../store/useViewerStore';

export function SpaceScaleToggle() {
  const spaceScale = useViewerStore((s) => s.spaceScale);
  const cycleSpaceScale = useViewerStore((s) => s.cycleSpaceScale);

  return (
    <button
      onClick={cycleSpaceScale}
      className="rounded-full bg-black/60 px-3 py-1.5 text-xs font-mono text-white/70 backdrop-blur-sm hover:bg-black/80 hover:text-white border border-white/10"
      title="Scale expansion (0.5x, 1x, 2x, 3x)"
      aria-label="Cycle space scale"
    >
      {spaceScale}x
    </button>
  );
}

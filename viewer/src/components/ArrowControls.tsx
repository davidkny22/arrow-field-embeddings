import { useEffect } from 'react';
import { useViewerStore } from '../store/useViewerStore';

/**
 * Arrow cycling UI + keyboard shortcuts.
 *
 * Shortcuts (when not typing in an input):
 *   ←/→   Cycle arrow index
 *   Q      Toggle all arrows (disabled in fly mode)
 *   H      Toggle arrow visibility (disabled in fly mode)
 *   1-9    Jump to arrow index (disabled in fly mode)
 */
export function ArrowControls() {
  const dataset = useViewerStore((s) => s.dataset);
  const activeArrowIndex = useViewerStore((s) => s.activeArrowIndex);
  const arrowsVisible = useViewerStore((s) => s.arrowsVisible);
  const arrowScale = useViewerStore((s) => s.arrowScale);
  const pointSizeMultiplier = useViewerStore((s) => s.pointSizeMultiplier);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      const tag = el.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable) return;

      const store = useViewerStore.getState();
      if (!store.dataset || store.dataset.n_arrows === 0) return;
      const isFly = store.controlMode === 'fly';

      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        store.prevArrow();
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        store.nextArrow();
      } else if ((e.key === 'q' || e.key === 'Q') && !isFly) {
        // Toggle all arrows (Q to avoid conflict with WASD fly controls)
        store.setActiveArrowIndex(
          store.activeArrowIndex === 'all' ? 0 : 'all',
        );
      } else if ((e.key === 'h' || e.key === 'H') && !isFly) {
        store.setArrowsVisible(!store.arrowsVisible);
      } else if (e.key >= '1' && e.key <= '9' && !isFly) {
        const idx = parseInt(e.key, 10) - 1;
        if (idx < store.dataset.n_arrows) {
          store.setActiveArrowIndex(idx);
        }
      }
    };

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  if (!dataset || dataset.n_arrows === 0) return null;

  const k = dataset.n_arrows;
  const label =
    activeArrowIndex === 'all'
      ? `ALL (${k})`
      : `${activeArrowIndex + 1} / ${k}`;

  return (
    <div className="fixed bottom-4 left-1/2 z-40 -translate-x-1/2 flex items-center gap-2">
      {/* Visibility toggle */}
      <button
        onClick={() => useViewerStore.getState().setArrowsVisible(!arrowsVisible)}
        className={`rounded-full px-3 py-1.5 text-xs font-mono backdrop-blur-sm border border-white/10 ${
          arrowsVisible
            ? 'bg-black/60 text-white/70 hover:bg-black/80 hover:text-white'
            : 'bg-black/40 text-white/30 hover:bg-black/60 hover:text-white/50'
        }`}
        title="Toggle arrows (H)"
        aria-label="Toggle arrow visibility"
      >
        {arrowsVisible ? 'ARROWS' : 'HIDDEN'}
      </button>

      {arrowsVisible && (
        <>
          {/* Previous */}
          <button
            onClick={() => useViewerStore.getState().prevArrow()}
            className="rounded-full bg-black/60 px-2.5 py-1.5 text-xs font-mono text-white/70 backdrop-blur-sm hover:bg-black/80 hover:text-white border border-white/10"
            title="Previous arrow (←)"
            aria-label="Previous arrow"
          >
            &#9664;
          </button>

          {/* Current label */}
          <div className="rounded-full bg-black/60 px-3 py-1.5 text-xs font-mono text-white/80 backdrop-blur-sm border border-white/10 min-w-[80px] text-center">
            {label}
          </div>

          {/* Next */}
          <button
            onClick={() => useViewerStore.getState().nextArrow()}
            className="rounded-full bg-black/60 px-2.5 py-1.5 text-xs font-mono text-white/70 backdrop-blur-sm hover:bg-black/80 hover:text-white border border-white/10"
            title="Next arrow (→)"
            aria-label="Next arrow"
          >
            &#9654;
          </button>

          {/* All toggle */}
          <button
            onClick={() =>
              useViewerStore.getState().setActiveArrowIndex(
                activeArrowIndex === 'all' ? 0 : 'all',
              )
            }
            className={`rounded-full px-3 py-1.5 text-xs font-mono backdrop-blur-sm border border-white/10 ${
              activeArrowIndex === 'all'
                ? 'bg-white/20 text-white'
                : 'bg-black/60 text-white/70 hover:bg-black/80 hover:text-white'
            }`}
            title="Show all arrows (Q)"
            aria-label="Toggle all arrows"
          >
            ALL
          </button>

          {/* Scale slider */}
          <div className="flex items-center gap-1.5 rounded-full bg-black/60 px-3 py-1.5 backdrop-blur-sm border border-white/10">
            <span className="text-xs font-mono text-white/40">Scale</span>
            <input
              type="range"
              min="0.1"
              max="5"
              step="0.1"
              value={arrowScale}
              onChange={(e) =>
                useViewerStore.getState().setArrowScale(parseFloat(e.target.value))
              }
              className="w-16 h-1 accent-white/60"
            />
            <span className="text-xs font-mono text-white/60 w-8 text-right">
              {arrowScale.toFixed(1)}
            </span>
          </div>

          {/* Point size slider */}
          <div className="flex items-center gap-1.5 rounded-full bg-black/60 px-3 py-1.5 backdrop-blur-sm border border-white/10">
            <span className="text-xs font-mono text-white/40">Points</span>
            <input
              type="range"
              min="0.1"
              max="5"
              step="0.1"
              value={pointSizeMultiplier}
              onChange={(e) =>
                useViewerStore.getState().setPointSizeMultiplier(parseFloat(e.target.value))
              }
              className="w-16 h-1 accent-white/60"
            />
            <span className="text-xs font-mono text-white/60 w-8 text-right">
              {pointSizeMultiplier.toFixed(1)}
            </span>
          </div>
        </>
      )}
    </div>
  );
}

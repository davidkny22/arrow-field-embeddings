import { useState, useCallback } from 'react';
import { useViewerStore } from '../store/useViewerStore';

export function InfoPanel() {
  const selectedIndex = useViewerStore((s) => s.selectedIndex);
  const dataset = useViewerStore((s) => s.dataset);
  const selectPoint = useViewerStore((s) => s.selectPoint);
  const activeArrowIndex = useViewerStore((s) => s.activeArrowIndex);

  const [arrowsExpanded, setArrowsExpanded] = useState(false);

  const handleClose = useCallback(() => {
    selectPoint(null);
  }, [selectPoint]);

  if (!dataset || selectedIndex == null) return null;

  // --- Derived data for the selected point ---

  const label = dataset.label_names[dataset.label_indices[selectedIndex]!] ?? 'Unknown';

  const clusterIdx = dataset.label_indices[selectedIndex]!;
  const cluster = dataset.clusters.find((c) => c.id === clusterIdx);

  const px = dataset.positions[selectedIndex * 3]!;
  const py = dataset.positions[selectedIndex * 3 + 1]!;
  const pz = dataset.positions[selectedIndex * 3 + 2]!;

  // Gather arrow values: n_arrows arrows, each with (theta, phi, r)
  const arrows: { theta: number; phi: number; r: number }[] = [];
  for (let j = 0; j < dataset.n_arrows; j++) {
    const base = selectedIndex * dataset.n_arrows * 3 + j * 3;
    arrows.push({
      theta: dataset.arrows[base]!,
      phi: dataset.arrows[base + 1]!,
      r: dataset.arrows[base + 2]!,
    });
  }

  const reconError = dataset.recon_error?.[selectedIndex];

  return (
    <div className="w-80 rounded-lg bg-black/80 p-4 text-white backdrop-blur-sm ring-1 ring-white/10">
      {/* Header: label + close */}
      <div className="mb-3 flex items-start justify-between gap-2">
        <h2 className="text-lg font-semibold leading-tight break-words">{label}</h2>
        <button
          onClick={handleClose}
          className="ml-2 shrink-0 text-white/40 hover:text-white/80 text-lg leading-none"
        >
          &times;
        </button>
      </div>

      {/* Cluster */}
      {cluster && (
        <div className="mb-3">
          <div className="text-xs uppercase tracking-wider text-white/40 mb-1">Cluster</div>
          <div className="text-sm font-medium">{cluster.label}</div>
          <div className="mt-0.5 text-xs text-white/40">{cluster.size} points</div>
        </div>
      )}

      {/* Position */}
      <div className="mb-3">
        <div className="text-xs uppercase tracking-wider text-white/40 mb-1">Position</div>
        <div className="font-mono text-xs text-white/60">
          [{px.toFixed(2)}, {py.toFixed(2)}, {pz.toFixed(2)}]
        </div>
      </div>

      {/* Point index */}
      <div className="mb-3">
        <div className="text-xs uppercase tracking-wider text-white/40 mb-1">Point Index</div>
        <div className="font-mono text-xs text-white/60">{selectedIndex}</div>
      </div>

      {/* Arrow values */}
      {arrows.length > 0 && (
        <div className="mb-3">
          <button
            onClick={() => setArrowsExpanded(!arrowsExpanded)}
            className="flex w-full items-center justify-between text-xs uppercase tracking-wider text-white/40 hover:text-white/60 mb-1"
          >
            <span>Arrows ({arrows.length})</span>
            <span className="text-sm">{arrowsExpanded ? '\u25B4' : '\u25BE'}</span>
          </button>

          {arrowsExpanded && (
            <div className="max-h-64 overflow-y-auto rounded bg-white/5 p-2 space-y-1">
              <div className="grid grid-cols-[auto_1fr_1fr_1fr] gap-x-3 text-xs text-white/40 mb-1 px-1">
                <span>#</span>
                <span>{'\u03B8'}</span>
                <span>{'\u03C6'}</span>
                <span>r</span>
              </div>
              {arrows.map((a, j) => {
                const isActive = activeArrowIndex === j || activeArrowIndex === 'all';
                const dimLabels = dataset.arrow_dim_labels?.[j];
                return (
                  <div
                    key={j}
                    className={`rounded px-1 py-0.5 cursor-pointer hover:bg-white/10 ${
                      isActive ? 'bg-white/10' : ''
                    }`}
                    onClick={() => useViewerStore.getState().setActiveArrowIndex(j)}
                  >
                    <div className={`grid grid-cols-[auto_1fr_1fr_1fr] gap-x-3 font-mono text-xs ${
                      isActive ? 'text-white' : 'text-white/60'
                    }`}>
                      <span className={isActive ? 'text-white/60' : 'text-white/30'}>{j}</span>
                      <span>{a.theta.toFixed(3)}</span>
                      <span>{a.phi.toFixed(3)}</span>
                      <span>{a.r.toFixed(3)}</span>
                    </div>
                    {dimLabels && dimLabels.length > 0 && (
                      <div className="text-[10px] text-white/40 mt-0.5 pl-4 break-words">
                        {dimLabels.join(', ')}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Reconstruction error */}
      {reconError != null && (
        <div className="mb-3">
          <div className="text-xs uppercase tracking-wider text-white/40 mb-1">Reconstruction Error</div>
          <div className="font-mono text-xs text-white/60">{reconError.toFixed(4)}</div>
        </div>
      )}

      {/* TODO: Re-add neighborhood feature when server-side k-NN endpoint is available.
         Previously this used embeddingService.neighbors() — will be replaced with
         a direct positional k-NN computation or a server call. */}
    </div>
  );
}

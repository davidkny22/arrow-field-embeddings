import { useCallback } from 'react';
import { useViewerStore } from '../store/useViewerStore';

/**
 * Decode spherical (θ, φ, r) to the 3 Cartesian dimension values.
 *
 * Math convention (Z-up, same as AFE export):
 *   dim[0] = r * cos(φ) * cos(θ)
 *   dim[1] = r * cos(φ) * sin(θ)
 *   dim[2] = r * sin(φ)
 */
function decodeDimValues(
  theta: number,
  phi: number,
  r: number,
): [number, number, number] {
  const cosPhi = Math.cos(phi);
  return [
    r * cosPhi * Math.cos(theta),
    r * cosPhi * Math.sin(theta),
    r * Math.sin(phi),
  ];
}

export function ArrowInfoPanel() {
  const clickedArrow = useViewerStore((s) => s.clickedArrow);
  const dataset = useViewerStore((s) => s.dataset);

  const handleClose = useCallback(() => {
    useViewerStore.getState().setClickedArrow(null);
  }, []);

  if (!clickedArrow || !dataset) return null;

  const { arrowIdx, pointIdx } = clickedArrow;
  const k = dataset.n_arrows;

  // Arrow spherical values for this point
  const base = pointIdx * k * 3 + arrowIdx * 3;
  const theta = dataset.arrows[base]!;
  const phi = dataset.arrows[base + 1]!;
  const r = dataset.arrows[base + 2]!;

  // Decode to Cartesian dimension values
  const decoded = decodeDimValues(theta, phi, r);

  // Point label
  const label = dataset.label_names[dataset.label_indices[pointIdx]!] ?? 'Unknown';

  // Dimension labels for this arrow
  const dimLabels = dataset.arrow_dim_labels?.[arrowIdx] ?? [];
  const mode = dataset.encoding_mode;

  // Formula labels for the 3 spherical→cartesian components
  const formulaLabels = [
    `r cos\u03C6 cos\u03B8`,
    `r cos\u03C6 sin\u03B8`,
    `r sin\u03C6`,
  ];

  return (
    <div className="w-80 rounded-lg bg-black/85 p-4 text-white backdrop-blur-sm ring-1 ring-white/10">
      {/* Header */}
      <div className="mb-3 flex items-start justify-between gap-2">
        <h2 className="text-lg font-semibold leading-tight">
          Arrow {arrowIdx}
        </h2>
        <button
          onClick={handleClose}
          className="ml-2 shrink-0 text-white/40 hover:text-white/80 text-lg leading-none"
        >
          &times;
        </button>
      </div>

      {/* Attached point */}
      <div className="mb-3">
        <div className="text-xs uppercase tracking-wider text-white/40 mb-1">
          Point
        </div>
        <div className="text-sm">
          <span className="font-medium">{label}</span>
          <span className="text-white/40 ml-2">#{pointIdx}</span>
        </div>
      </div>

      {/* Spherical encoding */}
      <div className="mb-3">
        <div className="text-xs uppercase tracking-wider text-white/40 mb-1">
          Spherical encoding
        </div>
        <div className="grid grid-cols-3 gap-2 font-mono text-xs">
          <div>
            <span className="text-white/40">{'\u03B8'} </span>
            <span>{theta.toFixed(3)}</span>
          </div>
          <div>
            <span className="text-white/40">{'\u03C6'} </span>
            <span>{phi.toFixed(3)}</span>
          </div>
          <div>
            <span className="text-white/40">r </span>
            <span>{r.toFixed(3)}</span>
          </div>
        </div>
      </div>

      {/* Decoded dimensions — varies by encoding mode */}
      {mode === 'direct' ? (
        // Direct mode: each arrow encodes exactly 3 (or fewer) residual dims
        <div>
          <div className="text-xs uppercase tracking-wider text-white/40 mb-1">
            Decoded dimensions
          </div>
          <div className="rounded bg-white/5 p-2 space-y-1">
            {[0, 1, 2].map((c) => {
              const dimName = dimLabels[c];
              if (!dimName) return null;
              return (
                <div
                  key={c}
                  className="grid grid-cols-[1fr_auto_auto] gap-x-2 items-center font-mono text-xs"
                >
                  <span className="text-white/50 truncate" title={dimName}>
                    {dimName}
                  </span>
                  <span className="text-white/25 text-[10px]">
                    {formulaLabels[c]}
                  </span>
                  <span className="text-right tabular-nums w-16">
                    {decoded[c].toFixed(4)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      ) : mode === 'pca' ? (
        // PCA mode: each arrow is one principal component
        <div>
          <div className="text-xs uppercase tracking-wider text-white/40 mb-1">
            Component
          </div>
          <div className="rounded bg-white/5 p-2 font-mono text-xs">
            <span className="text-white/50">{dimLabels[0] ?? `PC${arrowIdx + 1}`}</span>
            <span className="ml-2 tabular-nums">{r.toFixed(4)}</span>
          </div>
        </div>
      ) : (
        // Adaptive mode: each arrow encodes a group of dimensions
        <div>
          <div className="text-xs uppercase tracking-wider text-white/40 mb-1">
            Dimension group ({dimLabels.length} dims)
          </div>
          <div className="rounded bg-white/5 p-2">
            <div className="max-h-48 overflow-y-auto">
              <div className="flex flex-wrap gap-1">
                {dimLabels.map((dim, i) => (
                  <span
                    key={i}
                    className="rounded bg-white/10 px-1.5 py-0.5 text-xs text-white/70"
                  >
                    {dim}
                  </span>
                ))}
              </div>
            </div>
            {/* Show magnitude as summary metric */}
            <div className="mt-2 pt-2 border-t border-white/10 font-mono text-xs">
              <span className="text-white/40">magnitude </span>
              <span className="tabular-nums">{r.toFixed(4)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

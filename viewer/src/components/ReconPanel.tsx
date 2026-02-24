import { useViewerStore } from '../store/useViewerStore';

export function ReconPanel() {
  const selectedIndex = useViewerStore((s) => s.selectedIndex);
  const dataset = useViewerStore((s) => s.dataset);

  if (!dataset || selectedIndex == null) return null;

  const spatialMSE = dataset.recon_error_spatial?.[selectedIndex];
  const fullMSE = dataset.recon_error?.[selectedIndex];

  // Need both recon arrays to show this panel
  if (spatialMSE == null || fullMSE == null) return null;

  const improvement =
    spatialMSE > 0 ? ((1 - fullMSE / spatialMSE) * 100) : 0;

  // For the visual bar, normalize relative to spatial (the larger value)
  const maxError = Math.max(spatialMSE, fullMSE, 1e-9);
  const spatialBarPct = (spatialMSE / maxError) * 100;
  const fullBarPct = (fullMSE / maxError) * 100;

  const { gap_report } = dataset;
  const gapPct = gap_report.information_gap_score * 100;

  return (
    <div className="fixed right-4 bottom-20 z-40 w-80 rounded-lg bg-black/80 p-4 text-white backdrop-blur-sm ring-1 ring-white/10">
      {/* Header */}
      <div className="mb-3 text-xs uppercase tracking-wider text-white/40">
        Reconstruction Quality
      </div>

      {/* Spatial-only MSE */}
      <div className="mb-2">
        <div className="flex items-center justify-between text-xs">
          <span className="text-white/50">Spatial-only MSE</span>
          <span className="font-mono text-white/70">{spatialMSE.toFixed(4)}</span>
        </div>
      </div>

      {/* Full AFE MSE */}
      <div className="mb-2">
        <div className="flex items-center justify-between text-xs">
          <span className="text-white/50">Full AFE MSE</span>
          <span className="font-mono text-white/70">{fullMSE.toFixed(4)}</span>
        </div>
      </div>

      {/* Improvement */}
      <div className="mb-3">
        <div className="flex items-center justify-between text-xs">
          <span className="text-white/50">Improvement</span>
          <span
            className={`font-mono font-semibold ${
              improvement > 0 ? 'text-emerald-400' : 'text-red-400'
            }`}
          >
            {improvement > 0 ? '+' : ''}{improvement.toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Visual error bar */}
      <div className="mb-4">
        <div className="text-xs text-white/40 mb-1.5">Error Comparison</div>
        <div className="relative h-4 w-full rounded bg-white/5 overflow-hidden">
          {/* Spatial bar (background, gray) */}
          <div
            className="absolute inset-y-0 left-0 rounded bg-white/15"
            style={{ width: `${spatialBarPct}%` }}
          />
          {/* Full AFE bar (overlay, colored) */}
          <div
            className="absolute inset-y-0 left-0 rounded bg-indigo-500/60"
            style={{ width: `${fullBarPct}%` }}
          />
        </div>
        <div className="mt-1 flex justify-between text-[10px] text-white/30">
          <span>Spatial</span>
          <span>Full AFE</span>
        </div>
      </div>

      {/* Gap Score */}
      <div className="mb-3">
        <div className="flex items-center justify-between text-xs mb-1.5">
          <span className="text-white/50">Information Gap</span>
          <span className="font-mono text-white/70">{gapPct.toFixed(1)}%</span>
        </div>
        <div className="relative h-2 w-full rounded-full bg-white/5 overflow-hidden">
          <div
            className="absolute inset-y-0 left-0 rounded-full bg-amber-500/60"
            style={{ width: `${Math.min(gapPct, 100)}%` }}
          />
        </div>
      </div>

      {/* Captured / Residual Dims */}
      <div className="flex items-center gap-4">
        <div className="text-xs">
          <span className="text-white/40 mr-1">Captured</span>
          <span className="font-mono text-white/70">{gap_report.n_captured_dims}</span>
        </div>
        <div className="text-xs">
          <span className="text-white/40 mr-1">Residual</span>
          <span className="font-mono text-white/70">{gap_report.n_residual_dims}</span>
        </div>
      </div>
    </div>
  );
}

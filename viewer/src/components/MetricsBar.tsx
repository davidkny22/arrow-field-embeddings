import { useViewerStore } from '../store/useViewerStore';

function fmt(n: number, decimals: number): string {
  return n.toFixed(decimals);
}

function pct(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}

function commas(n: number): string {
  return n.toLocaleString();
}

export function MetricsBar() {
  const dataset = useViewerStore((s) => s.dataset);

  if (!dataset) return null;

  const { metrics } = dataset;

  return (
    <div className="fixed top-14 left-4 z-40 flex flex-wrap items-center gap-2 max-w-[calc(100vw-22rem)]">
      {/* Encoding mode badge */}
      <span className="rounded-lg bg-indigo-500/30 px-2.5 py-1 text-xs font-semibold uppercase tracking-wider text-indigo-300 border border-indigo-500/30 backdrop-blur-sm">
        {dataset.encoding_mode}
      </span>

      {/* Points */}
      <span className="rounded-lg bg-black/60 px-2.5 py-1 text-xs text-white/70 backdrop-blur-sm border border-white/10">
        <span className="text-white/40 mr-1">Points</span>
        <span className="font-mono">{commas(dataset.n_points)}</span>
      </span>

      {/* Arrows */}
      <span className="rounded-lg bg-black/60 px-2.5 py-1 text-xs text-white/70 backdrop-blur-sm border border-white/10">
        <span className="text-white/40 mr-1">Arrows</span>
        <span className="font-mono">{commas(dataset.n_arrows)}</span>
      </span>

      {/* Dims */}
      <span className="rounded-lg bg-black/60 px-2.5 py-1 text-xs text-white/70 backdrop-blur-sm border border-white/10">
        <span className="text-white/40 mr-1">Dims</span>
        <span className="font-mono">{dataset.embedding_dim}</span>
      </span>

      {/* kNN@10 */}
      <span className="rounded-lg bg-black/60 px-2.5 py-1 text-xs text-white/70 backdrop-blur-sm border border-white/10">
        <span className="text-white/40 mr-1">kNN@10</span>
        <span className="font-mono">{pct(metrics.knn_recall_k10)}</span>
      </span>

      {/* Arrow kNN@10 */}
      <span className="rounded-lg bg-black/60 px-2.5 py-1 text-xs text-white/70 backdrop-blur-sm border border-white/10">
        <span className="text-white/40 mr-1">Arrow kNN@10</span>
        <span className="font-mono">{pct(metrics.arrow_knn_recall_k10)}</span>
      </span>

      {/* Info Gain */}
      <span className="rounded-lg bg-black/60 px-2.5 py-1 text-xs text-white/70 backdrop-blur-sm border border-white/10">
        <span className="text-white/40 mr-1">Info Gain</span>
        <span className="font-mono">{pct(metrics.arrow_spatial_information_gain)}</span>
      </span>

      {/* Recon MSE */}
      <span className="rounded-lg bg-black/60 px-2.5 py-1 text-xs text-white/70 backdrop-blur-sm border border-white/10">
        <span className="text-white/40 mr-1">Recon MSE</span>
        <span className="font-mono">{fmt(metrics.reconstruction_mse, 4)}</span>
      </span>

      {/* Distance Corr */}
      <span className="rounded-lg bg-black/60 px-2.5 py-1 text-xs text-white/70 backdrop-blur-sm border border-white/10">
        <span className="text-white/40 mr-1">Distance Corr</span>
        <span className="font-mono">{fmt(metrics.spearman_dist_corr, 3)}</span>
      </span>
    </div>
  );
}

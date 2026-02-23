import { useViewerStore } from '../store/useViewerStore';

export function DatasetSelector() {
  const availableDatasets = useViewerStore((s) => s.availableDatasets);
  const datasetUrl = useViewerStore((s) => s.datasetUrl);
  const setDatasetUrl = useViewerStore((s) => s.setDatasetUrl);
  const loading = useViewerStore((s) => s.loading);

  if (availableDatasets.length === 0) return null;

  return (
    <div className="fixed left-4 top-4 z-40">
      <select
        value={datasetUrl}
        onChange={(e) => setDatasetUrl(e.target.value)}
        disabled={loading}
        className="rounded-md bg-black/70 px-3 py-1.5 text-sm text-white backdrop-blur-sm outline-none ring-1 ring-white/10 focus:ring-white/30 disabled:opacity-40"
      >
        {availableDatasets.map((s) => (
          <option key={s.id} value={s.url}>
            {s.label}
          </option>
        ))}
      </select>
    </div>
  );
}

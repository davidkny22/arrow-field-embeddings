import { useViewerStore } from '../store/useViewerStore';

export function LoadingScreen() {
  const loading = useViewerStore((s) => s.loading);
  const loadingProgress = useViewerStore((s) => s.loadingProgress);
  const error = useViewerStore((s) => s.error);
  const availableDatasets = useViewerStore((s) => s.availableDatasets);
  const datasetUrl = useViewerStore((s) => s.datasetUrl);

  if (!loading && !error) return null;

  const datasetName = availableDatasets.find((s) => s.url === datasetUrl)?.label ?? 'dataset';
  const showProgress = loadingProgress > 0 && loadingProgress < 100;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#0a0a0a]">
      <div className="text-center">
        {loading && !error && (
          <>
            <div className="mb-4 text-2xl font-light tracking-widest text-white/80">
              AFE VIEWER
            </div>
            <div className="text-sm text-white/40">Loading {datasetName}...</div>
            <div className="mt-6 h-0.5 w-32 mx-auto overflow-hidden rounded bg-white/10">
              {showProgress ? (
                <div
                  className="h-full rounded bg-white/60 transition-all duration-150"
                  style={{ width: `${loadingProgress}%` }}
                />
              ) : (
                <div className="h-full w-1/3 animate-pulse rounded bg-white/30" />
              )}
            </div>
            {showProgress && (
              <div className="mt-2 text-[10px] font-mono text-white/30">
                {loadingProgress}%
              </div>
            )}
          </>
        )}
        {error && (
          <>
            <div className="mb-4 text-2xl font-light tracking-widest text-red-400/80">
              ERROR
            </div>
            <div className="max-w-md text-sm text-white/60">{error}</div>
          </>
        )}
      </div>
    </div>
  );
}

import { useState, useRef, useEffect, useCallback } from 'react';
import { useViewerStore } from '../store/useViewerStore';
import { useSearch, type SearchResult } from '../hooks/useSearch';

export function SearchBar() {
  const dataset = useViewerStore((s) => s.dataset);
  const setSearchQuery = useViewerStore((s) => s.setSearchQuery);
  const setHighlightedIndices = useViewerStore((s) => s.setHighlightedIndices);
  const setColorMode = useViewerStore((s) => s.setColorMode);
  const flyTo = useViewerStore((s) => s.flyTo);
  const selectPoint = useViewerStore((s) => s.selectPoint);

  const { search, getHighlightIndices } = useSearch(dataset);

  const [input, setInput] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedIdx, setSelectedIdx] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(undefined);

  // Debounced search
  useEffect(() => {
    clearTimeout(debounceRef.current);

    if (!input.trim()) {
      setResults([]);
      setShowDropdown(false);
      setHighlightedIndices(new Set());
      setColorMode('cluster');
      setSearchQuery('');
      return;
    }

    debounceRef.current = setTimeout(() => {
      const hits = search(input);
      setResults(hits);
      setShowDropdown(hits.length > 0);
      setSelectedIdx(-1);
      setSearchQuery(input);

      if (hits.length > 0) {
        const indices = getHighlightIndices(hits);
        setHighlightedIndices(indices);
        setColorMode('highlight');
      } else {
        setHighlightedIndices(new Set());
        setColorMode('cluster');
      }
    }, 200);

    return () => clearTimeout(debounceRef.current);
  }, [input, search, getHighlightIndices, setHighlightedIndices, setColorMode, setSearchQuery]);

  const selectResult = useCallback(
    (result: SearchResult) => {
      if (!dataset) return;

      if (result.type === 'label') {
        // Highlight all matching points; fly to and select the first one
        const firstIndex = result.indices[0];
        if (firstIndex != null) {
          selectPoint(firstIndex);
          const px = dataset.positions[firstIndex * 3]!;
          const py = dataset.positions[firstIndex * 3 + 1]!;
          const pz = dataset.positions[firstIndex * 3 + 2]!;
          flyTo([px, py, pz]);
        }
      } else {
        // Cluster result — fly to centroid, deselect point
        flyTo(result.cluster.centroid);
        selectPoint(null);
      }

      setShowDropdown(false);
    },
    [dataset, flyTo, selectPoint],
  );

  const clearSearch = useCallback(() => {
    setInput('');
    setResults([]);
    setShowDropdown(false);
    setHighlightedIndices(new Set());
    setColorMode('cluster');
    setSearchQuery('');
    selectPoint(null);
  }, [setHighlightedIndices, setColorMode, setSearchQuery, selectPoint]);

  // Keyboard shortcuts
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      clearSearch();
      inputRef.current?.blur();
      return;
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIdx((prev) => Math.min(prev + 1, results.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIdx((prev) => Math.max(prev - 1, -1));
    } else if (e.key === 'Enter' && selectedIdx >= 0 && results[selectedIdx]) {
      e.preventDefault();
      selectResult(results[selectedIdx]);
    } else if (e.key === 'Enter' && results.length > 0 && results[0]) {
      e.preventDefault();
      selectResult(results[0]);
    }
  };

  // Global "/" shortcut to focus search
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === '/' && document.activeElement !== inputRef.current) {
        e.preventDefault();
        inputRef.current?.focus();
      }
      if (e.key === 'Escape' && document.activeElement === inputRef.current) {
        clearSearch();
        inputRef.current?.blur();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [clearSearch]);

  if (!dataset) return null;

  return (
    <div className="fixed left-1/2 top-4 z-40 w-96 -translate-x-1/2">
      <div className="relative">
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => results.length > 0 && setShowDropdown(true)}
          placeholder='Search labels... (press "/")'
          className="w-full rounded-lg bg-black/70 px-4 py-2.5 text-sm text-white placeholder-white/30 backdrop-blur-sm outline-none ring-1 ring-white/10 focus:ring-white/30"
        />
        {input && (
          <button
            onClick={clearSearch}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-white/30 hover:text-white/60"
          >
            &times;
          </button>
        )}
      </div>

      {showDropdown && (
        <div className="mt-1 max-h-80 overflow-y-auto rounded-lg bg-black/80 py-1 backdrop-blur-sm ring-1 ring-white/10">
          {results.map((result, i) => (
            <button
              key={
                result.type === 'label'
                  ? `l-${result.labelIndex}`
                  : `c-${result.cluster.id}`
              }
              onClick={() => selectResult(result)}
              className={`w-full px-4 py-2 text-left text-sm ${
                i === selectedIdx
                  ? 'bg-white/10 text-white'
                  : 'text-white/70 hover:bg-white/5'
              }`}
            >
              {result.type === 'label' ? (
                <span>
                  {result.label}
                  <span className="ml-2 text-white/30">
                    ({result.indices.length} point{result.indices.length !== 1 ? 's' : ''})
                  </span>
                </span>
              ) : (
                <span>
                  <span className="text-white/40">Cluster: </span>
                  {result.cluster.label}
                  <span className="text-white/30">
                    {' '}({result.cluster.size} points)
                  </span>
                </span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

import { useEffect, useRef } from 'react';

const GLOBAL_SHORTCUTS = [
  ['/', 'Focus search'],
  ['`', 'Toggle stats'],
  ['P', 'Download screenshot'],
  ['?', 'Open shortcuts'],
];

const FLY_SHORTCUTS = [
  ['W A S D', 'Move'],
  ['E / Space', 'Up'],
  ['Q / Ctrl', 'Down'],
  ['Mouse drag', 'Look around'],
  ['Scroll', 'Zoom'],
  ['Shift + drag', 'Rectangle select'],
  ['Click point', 'Select & inspect'],
  ['← / →', 'Cycle arrows'],
  ['Q', 'Show all arrows'],
  ['H', 'Toggle arrows'],
];

const ORBIT_SHORTCUTS = [
  ['Left drag', 'Rotate'],
  ['Right drag', 'Pan'],
  ['Ctrl + drag', 'Swap rotate/pan'],
  ['Scroll', 'Zoom'],
  ['Shift + drag', 'Rectangle select'],
  ['Click point', 'Select & inspect'],
  ['← / →', 'Cycle arrows'],
  ['Q', 'Show all arrows'],
  ['H', 'Toggle arrows'],
];

interface ShortcutsModalProps {
  open: boolean;
  onClose: () => void;
}

export function ShortcutsModal({ open, onClose }: ShortcutsModalProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        ref={ref}
        role="dialog"
        aria-modal="true"
        aria-label="Keyboard shortcuts"
        className="max-h-[80vh] w-full max-w-lg overflow-y-auto rounded-xl bg-black/80 p-6 backdrop-blur-md ring-1 ring-white/10"
      >
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-white/80">
            Keyboard Shortcuts
          </h2>
          <button
            onClick={onClose}
            className="text-white/40 hover:text-white"
            aria-label="Close shortcuts"
          >
            ✕
          </button>
        </div>

        <div className="space-y-4">
          <ShortcutSection title="Global" items={GLOBAL_SHORTCUTS} />
          <ShortcutSection title="Orbit Mode" items={ORBIT_SHORTCUTS} />
          <ShortcutSection title="Fly Mode" items={FLY_SHORTCUTS} />
        </div>
      </div>
    </div>
  );
}

function ShortcutSection({ title, items }: { title: string; items: string[][] }) {
  return (
    <div>
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-white/50">
        {title}
      </h3>
      <div className="space-y-1">
        {items.map(([key, action]) => (
          <div key={key + action} className="flex items-center gap-3 text-xs">
            <span className="min-w-[110px] font-mono text-white/80">{key}</span>
            <span className="text-white/40">{action}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

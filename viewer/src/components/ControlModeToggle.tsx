import { useViewerStore } from '../store/useViewerStore';

export function ControlModeToggle() {
  const controlMode = useViewerStore((s) => s.controlMode);
  const setControlMode = useViewerStore((s) => s.setControlMode);

  // Available in all modes (simple + advanced)

  return (
    <button
      onClick={() => setControlMode(controlMode === 'orbit' ? 'fly' : 'orbit')}
      className="rounded-full bg-black/60 px-3 py-1.5 text-xs font-mono text-white/70 backdrop-blur-sm hover:bg-black/80 hover:text-white border border-white/10"
      title={controlMode === 'orbit' ? 'Switch to fly controls (WASD)' : 'Switch to orbit controls'}
      aria-label={controlMode === 'orbit' ? 'Switch to fly controls' : 'Switch to orbit controls'}
    >
      {controlMode === 'orbit' ? 'ORBIT' : 'FLY'}
    </button>
  );
}

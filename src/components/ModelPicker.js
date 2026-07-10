import React, { useState, useRef, useEffect } from 'react';
import ReactDOM from 'react-dom';
import { Bot, ChevronDown } from 'lucide-react';

const ANTHROPIC_MODELS = [
  { value: 'claude-sonnet-4-20250514', label: 'Claude Sonnet 4.5' },
  { value: 'claude-haiku-4-5-20251001', label: 'Claude Haiku 4.5' },
  { value: 'claude-opus-4-20250514', label: 'Claude Opus 4' },
];

function ModelPicker({ llmStatus, ollamaModels, onFetchModels, onSwitch }) {
  const [open, setOpen] = useState(false);
  const [rect, setRect] = useState(null);
  const triggerRef = useRef(null);
  const dropdownRef = useRef(null);

  useEffect(() => {
    const handler = (e) => {
      if (
        triggerRef.current &&
        !triggerRef.current.contains(e.target) &&
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target)
      )
        setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const handleOpen = () => {
    if (triggerRef.current) setRect(triggerRef.current.getBoundingClientRect());
    if (!open) onFetchModels();
    setOpen((o) => !o);
  };

  const backend = llmStatus?.backend || 'anthropic';
  const activeModel = llmStatus?.model || '';
  const shortName = activeModel.length > 20 ? activeModel.slice(0, 18) + '\u2026' : activeModel || '\u2026';

  const modelList =
    backend === 'anthropic'
      ? ANTHROPIC_MODELS
      : [{ value: '', label: `default (${activeModel})` }, ...ollamaModels.map((m) => ({ value: m, label: m }))];

  const dropdown =
    open &&
    rect &&
    ReactDOM.createPortal(
      <div
        ref={dropdownRef}
        style={{
          position: 'fixed',
          bottom: window.innerHeight - rect.top + 6,
          left: Math.max(8, rect.left),
          minWidth: 230,
          zIndex: 9999,
          background: 'var(--bg-panel)',
          border: '1px solid var(--border-neon)',
          borderRadius: 'var(--radius-md)',
          overflow: 'hidden',
          boxShadow: '0 -8px 32px rgba(0,0,0,0.7)',
        }}
      >
        {/* Backend tabs */}
        <div style={{ display: 'flex', borderBottom: '1px solid var(--border)' }}>
          {[
            { k: 'anthropic', label: '\u2601 Anthropic' },
            { k: 'ollama', label: '\u2B21 Ollama' },
          ].map(({ k, label }) => (
            <button
              key={k}
              onMouseDown={(e) => {
                e.preventDefault();
                onSwitch(k, null);
              }}
              style={{
                flex: 1,
                padding: '7px 0',
                fontSize: 11,
                border: 'none',
                cursor: 'pointer',
                background: backend === k ? 'var(--accent-soft)' : 'transparent',
                color: backend === k ? 'var(--neon-cyan)' : 'var(--text-tertiary)',
                fontFamily: 'var(--font-mono)',
                transition: 'background 0.15s',
              }}
            >
              {label}
            </button>
          ))}
        </div>
        {/* Model list */}
        <div style={{ maxHeight: 200, overflowY: 'auto', padding: '4px 0' }}>
          {modelList.map((m) => {
            const isActive = m.value ? m.value === activeModel : backend === 'ollama';
            return (
              <div
                key={m.value}
                onMouseDown={(e) => {
                  e.preventDefault();
                  onSwitch(backend, m.value || null);
                  setOpen(false);
                }}
                style={{
                  padding: '6px 14px',
                  fontSize: 11,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  color: isActive ? 'var(--neon-cyan)' : 'var(--text-secondary)',
                  background: isActive ? 'var(--accent-soft)' : 'transparent',
                }}
                onMouseEnter={(e) => {
                  if (!isActive) e.currentTarget.style.background = 'var(--bg-hover)';
                }}
                onMouseLeave={(e) => {
                  if (!isActive) e.currentTarget.style.background = 'transparent';
                }}
              >
                <span style={{ width: 12, fontSize: 10, color: 'var(--neon-cyan)' }}>{isActive ? '\u2713' : ''}</span>
                {m.label}
              </div>
            );
          })}
        </div>
        {/* Status bar */}
        {llmStatus?.backend === 'ollama' && (
          <div
            style={{
              padding: '5px 14px',
              fontSize: 10,
              color: llmStatus.ollama_reachable ? 'var(--neon-green)' : 'var(--warm)',
              borderTop: '1px solid var(--border)',
              fontFamily: 'var(--font-mono)',
            }}
          >
            {llmStatus.ollama_reachable ? '\u25CF Ollama reachable' : '\u25CB Ollama unreachable'}
          </div>
        )}
      </div>,
      document.body,
    );

  return (
    <>
      <button
        ref={triggerRef}
        onClick={handleOpen}
        className="voice-btn"
        title="Switch model"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 3,
          padding: '0 7px',
          fontSize: 10,
          fontFamily: 'var(--font-mono)',
          color: 'var(--text-secondary)',
          minWidth: 0,
        }}
      >
        <Bot size={12} style={{ flexShrink: 0 }} />
        <span style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {shortName}
        </span>
        <ChevronDown
          size={9}
          style={{ flexShrink: 0, transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.15s' }}
        />
      </button>
      {dropdown}
    </>
  );
}

export default React.memo(ModelPicker);

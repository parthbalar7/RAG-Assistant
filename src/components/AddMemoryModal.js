import React, { useState } from 'react';
import { Brain } from 'lucide-react';

function AddMemoryModal({ onClose, onAdd }) {
  const [content, setContent] = useState('');
  const [memType, setMemType] = useState('fact');
  const [importance, setImportance] = useState(0.7);
  const types = ['fact', 'pref', 'decision', 'insight'];
  return (
    <div className="modal-overlay" onClick={onClose} style={{ zIndex: 150 }}>
      <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 440 }}>
        <h2>Add Memory</h2>
        <p>Manually teach the assistant a fact, preference, or insight to remember.</p>
        <textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="e.g., The project uses FastAPI with PostgreSQL..."
          style={{
            width: '100%',
            minHeight: 80,
            padding: 12,
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border)',
            background: 'var(--surface-2)',
            color: 'var(--text-1)',
            fontSize: 13,
            fontFamily: 'var(--font-body)',
            resize: 'vertical',
            outline: 'none',
            boxSizing: 'border-box',
          }}
        />
        <div style={{ display: 'flex', gap: 6, margin: '12px 0', flexWrap: 'wrap' }}>
          {types.map((t) => (
            <button
              key={t}
              onClick={() => setMemType(t)}
              className={memType === t ? 'modal-btn confirm' : 'modal-btn cancel'}
              style={{ fontSize: 11, padding: '5px 12px' }}
            >
              {t.replace(/_/g, ' ')}
            </button>
          ))}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
          <span style={{ fontSize: 11, color: 'var(--text-2)' }}>Importance:</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={importance}
            onChange={(e) => setImportance(parseFloat(e.target.value))}
            style={{ flex: 1, accentColor: 'var(--accent)' }}
          />
          <span style={{ fontSize: 11, color: 'var(--accent)', fontFamily: 'var(--font-mono)', minWidth: 30 }}>
            {(importance * 100).toFixed(0)}%
          </span>
        </div>
        <div className="modal-actions">
          <button className="modal-btn cancel" onClick={onClose}>
            Cancel
          </button>
          <button
            className="modal-btn confirm"
            onClick={() => onAdd(content, memType, importance)}
            disabled={!content.trim()}
          >
            <Brain size={12} style={{ verticalAlign: -2 }} /> Store
          </button>
        </div>
      </div>
    </div>
  );
}

export default React.memo(AddMemoryModal);

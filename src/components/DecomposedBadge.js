import React, { useState } from 'react';
import { Layers } from 'lucide-react';

function DecomposedBadge({ subQueries }) {
  const [open, setOpen] = useState(false);
  if (!subQueries || subQueries.length === 0) return null;
  return (
    <div className="gpath-container">
      <button
        className="gpath-summary-btn"
        onClick={() => setOpen((o) => !o)}
        title="Query was decomposed into sub-queries"
      >
        <Layers size={10} style={{ color: 'var(--accent)' }} />
        <span style={{ color: 'var(--accent)', fontSize: 10 }}>Decomposed</span>
        <span className="gpath-seed-chips">
          {subQueries.map((_sq, i) => (
            <span key={i} className="gpath-chip gpath-seed">
              {i + 1}
            </span>
          ))}
        </span>
        <span style={{ fontSize: 9, color: 'var(--text-3)', marginLeft: 'auto' }}>
          {subQueries.length} sub-queries {open ? '\u25B2' : '\u25BC'}
        </span>
      </button>
      {open && (
        <div className="gpath-drawer">
          <div className="gpath-nodes-section">
            <span className="gpath-section-label">Sub-queries retrieved independently</span>
            {subQueries.map((sq, i) => (
              <div key={i} className="gpath-edge-row" style={{ alignItems: 'flex-start', gap: 6 }}>
                <span className="gpath-chip gpath-seed" style={{ minWidth: 18, textAlign: 'center' }}>
                  {i + 1}
                </span>
                <span style={{ fontSize: 10, color: 'var(--text-2)', lineHeight: 1.4 }}>{sq}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default React.memo(DecomposedBadge);

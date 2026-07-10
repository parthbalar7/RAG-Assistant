import React, { useState } from 'react';
import { Network } from 'lucide-react';

function GraphPathBadge({ graphPath }) {
  const [open, setOpen] = useState(false);
  if (!graphPath) return null;

  const seeds = graphPath.seeds || [];
  const nodes = graphPath.nodes || [];
  const edges = graphPath.edges || [];
  const chunks_found = graphPath.chunks_found || 0;
  const noSeeds = seeds.length === 0;
  const seedKeys = new Set(seeds.map((s) => s.key));

  return (
    <div className="gpath-container">
      <button
        className="gpath-summary-btn"
        onClick={() => !noSeeds && setOpen((o) => !o)}
        style={{ cursor: noSeeds ? 'default' : 'pointer' }}
        title="Graph traversal path"
      >
        <Network size={10} style={{ color: noSeeds ? 'var(--text-3)' : 'var(--accent)' }} />
        <span style={{ color: noSeeds ? 'var(--text-3)' : 'var(--accent)', fontSize: 10 }}>Graph path</span>
        {noSeeds ? (
          <span style={{ fontSize: 9, color: 'var(--text-3)' }}>
            {graphPath.message || 'No matching entities found'}
          </span>
        ) : (
          <span className="gpath-seed-chips">
            {seeds.map((s) => (
              <span key={s.key} className="gpath-chip gpath-seed">
                {s.display}
              </span>
            ))}
          </span>
        )}
        {!noSeeds && (
          <span style={{ fontSize: 9, color: 'var(--text-3)', marginLeft: 'auto' }}>
            {nodes.length} nodes · {chunks_found} chunks {open ? '\u25B2' : '\u25BC'}
          </span>
        )}
      </button>

      {open && (
        <div className="gpath-drawer">
          {/* Traversal edge list */}
          {edges.length > 0 && (
            <div className="gpath-edges">
              {edges.slice(0, 15).map((e, i) => (
                <div key={i} className="gpath-edge-row">
                  <span className={'gpath-chip ' + (seedKeys.has(e.from?.toLowerCase()) ? 'gpath-seed' : 'gpath-node')}>
                    {e.from}
                  </span>
                  <span className="gpath-rel-label">{e.rel.replace(/_/g, ' ')}</span>
                  <span className="gpath-arrow">{'\u2192'}</span>
                  <span className={'gpath-chip ' + (seedKeys.has(e.to?.toLowerCase()) ? 'gpath-seed' : 'gpath-node')}>
                    {e.to}
                  </span>
                  <span className="gpath-hop">hop {e.hop}</span>
                </div>
              ))}
            </div>
          )}

          {/* All visited nodes */}
          <div className="gpath-nodes-section">
            <span className="gpath-section-label">Visited entities</span>
            <div className="gpath-nodes-wrap">
              {nodes.map((n) => (
                <span
                  key={n.key}
                  className={'gpath-chip ' + (n.is_seed ? 'gpath-seed' : 'gpath-node')}
                  title={n.type + ' \u00B7 ' + n.degree + ' connections'}
                >
                  {n.display}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default React.memo(GraphPathBadge);

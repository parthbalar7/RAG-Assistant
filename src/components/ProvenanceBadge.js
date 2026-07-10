import React, { useState } from 'react';
import { GitBranch, ShieldCheck, ShieldAlert, ShieldOff } from 'lucide-react';

const RISK_META = {
  sourced: { icon: ShieldCheck, color: 'var(--ok)', label: 'sourced' },
  inferred: { icon: ShieldAlert, color: 'var(--warn)', label: 'inferred' },
  orphan: { icon: ShieldOff, color: 'var(--danger)', label: 'orphan' },
};

function ProvenanceBadge({ provenance }) {
  const [open, setOpen] = useState(false);
  const [activeSent, setActiveSent] = useState(null);
  if (!provenance) return null;
  const { sentences, sourced_count, inferred_count, orphan_count } = provenance;
  const total = sentences.length;
  if (total === 0) return null;

  const riskColor =
    orphan_count / total > 0.4 ? 'var(--danger)' : orphan_count / total > 0.15 ? 'var(--warn)' : 'var(--ok)';

  return (
    <div className="prov-container">
      <button
        className="prov-summary-btn"
        onClick={() => setOpen((o) => !o)}
        title="Ancestry Trace — sentence-level attribution"
      >
        <GitBranch size={10} style={{ color: riskColor }} />
        <span className="prov-pill prov-sourced">{sourced_count} sourced</span>
        {inferred_count > 0 && <span className="prov-pill prov-inferred">{inferred_count} inferred</span>}
        {orphan_count > 0 && <span className="prov-pill prov-orphan">{orphan_count} orphan</span>}
        <span style={{ marginLeft: 'auto', fontSize: 9, color: 'var(--text-3)' }}>{open ? '\u25B2' : '\u25BC'}</span>
      </button>

      {open && (
        <div className="prov-drawer">
          <div className="prov-sentences">
            {sentences.map((sp, i) => {
              const meta = RISK_META[sp.risk] || RISK_META.inferred;
              const Icon = meta.icon;
              const isActive = activeSent === i;
              return (
                <div
                  key={i}
                  className={'prov-sent ' + (isActive ? 'prov-sent-active' : '')}
                  onClick={() => setActiveSent(isActive ? null : i)}
                >
                  <div className="prov-sent-header">
                    <Icon size={10} style={{ color: meta.color, flexShrink: 0 }} />
                    <span className="prov-sent-text">
                      {sp.text.length > 100 ? sp.text.slice(0, 100) + '\u2026' : sp.text}
                    </span>
                    <span className="prov-novel-score" style={{ color: meta.color }}>
                      {(sp.novel_score * 100).toFixed(0)}% novel
                    </span>
                  </div>
                  {isActive && sp.attributions.length > 0 && (
                    <div className="prov-attr-list">
                      {sp.attributions.slice(0, 3).map((a, j) => (
                        <div key={j} className="prov-attr">
                          <span className="prov-attr-type">{a.source_type}</span>
                          <span className="prov-attr-id" title={a.source_id}>
                            {a.source_id.length > 30 ? '\u2026' + a.source_id.slice(-28) : a.source_id}
                          </span>
                          <span className="prov-attr-sim">{(a.similarity * 100).toFixed(0)}%</span>
                          <span className="prov-attr-preview">{a.source_preview}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
          <div className="prov-legend">
            <span>
              <ShieldCheck size={9} style={{ color: 'var(--ok)' }} /> sourced &lt;35% novel
            </span>
            <span>
              <ShieldAlert size={9} style={{ color: 'var(--warn)' }} /> inferred 35-65%
            </span>
            <span>
              <ShieldOff size={9} style={{ color: 'var(--danger)' }} /> orphan &gt;65%
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

export default React.memo(ProvenanceBadge);

import React, { useState } from 'react';
import { ChevronDown, ChevronRight, FileCode, BookOpen } from 'lucide-react';

function SourcesPanel({ sources, onViewPdf }) {
  const [open, setOpen] = useState(false);
  const [expanded, setExpanded] = useState({});
  if (!sources || sources.length === 0) return null;

  const toggle = (i) => setExpanded((p) => ({ ...p, [i]: !p[i] }));

  return (
    <div className="sources-panel">
      <button className="sources-toggle" onClick={() => setOpen(!open)}>
        {open ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
        <BookOpen size={11} /> {sources.length} source{sources.length !== 1 ? 's' : ''} referenced
      </button>
      {open && (
        <div className="sources-list">
          {sources.map((s, i) => (
            <div key={i} className={'source-card' + (expanded[i] ? ' expanded' : '')} onClick={() => toggle(i)}>
              <div className="source-card-header">
                <FileCode size={12} style={{ color: 'var(--neon-purple)', flexShrink: 0 }} />
                <span className="source-card-file">{s.file}</span>
                {s.page && (
                  <span
                    className="source-card-page"
                    onClick={(e) => {
                      e.stopPropagation();
                      onViewPdf && onViewPdf(s);
                    }}
                  >
                    p.{s.page}
                  </span>
                )}
                <span className="source-card-score">{(s.score * 100).toFixed(0)}%</span>
                {s.search_type && (
                  <span style={{ fontSize: 9, color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>
                    {s.search_type}
                  </span>
                )}
              </div>
              <div className="source-card-preview">
                {s.preview || `Lines: ${s.lines || '?'} · Language: ${s.language || '?'}`}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default React.memo(SourcesPanel);

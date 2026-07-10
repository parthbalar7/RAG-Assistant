import React from 'react';
import { FileText, X } from 'lucide-react';

function PdfViewerPanel({ source, onClose }) {
  if (!source) return null;
  return (
    <div className="pdf-viewer-panel">
      <div className="pdf-viewer-header">
        <span>
          <FileText size={12} style={{ verticalAlign: -2 }} /> {source.file} — Page {source.page || '?'}
        </span>
        <button
          onClick={onClose}
          style={{ background: 'none', border: 'none', color: 'var(--text-3)', cursor: 'pointer' }}
        >
          <X size={16} />
        </button>
      </div>
      <div className="pdf-viewer-content">
        <div className="pdf-page-preview">{source.preview || 'No preview available for this page.'}</div>
      </div>
    </div>
  );
}

export default React.memo(PdfViewerPanel);

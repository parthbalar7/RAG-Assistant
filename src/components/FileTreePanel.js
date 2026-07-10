import React, { useState, useEffect, useCallback } from 'react';
import { FileCode, Trash2 } from 'lucide-react';
import { api } from '../utils/api';

function FileTreePanel({ refreshKey, token, onToast }) {
  const [files, setFiles] = useState([]);
  const [deleting, setDeleting] = useState(null);

  const load = useCallback(() => {
    api
      .get('/api/files', token)
      .then((d) => setFiles(d.files || []))
      .catch(() => {});
  }, [token]);

  useEffect(() => {
    load();
  }, [refreshKey, load]);

  const handleDelete = async (path) => {
    if (!window.confirm(`Remove "${path.split('/').pop()}" from the index?`)) return;
    setDeleting(path);
    try {
      await api.del('/api/files?path=' + encodeURIComponent(path), token);
      onToast('success', 'File removed from index');
      load();
    } catch (e) {
      onToast('error', e.message);
    } finally {
      setDeleting(null);
    }
  };

  const tree = {};
  files.forEach(function (f) {
    const parts = f.path.split('/');
    const dir = parts.length > 1 ? parts.slice(0, -1).join('/') : '.';
    if (!tree[dir]) tree[dir] = [];
    tree[dir].push(f);
  });

  if (files.length === 0)
    return <div style={{ padding: 16, fontSize: 12, color: 'var(--text-3)' }}>No files indexed yet.</div>;
  return (
    <div>
      <div
        style={{
          fontSize: 11,
          color: 'var(--text-3)',
          padding: '4px 8px',
          marginBottom: 4,
          fontFamily: 'var(--font-mono)',
        }}
      >
        {files.length} files indexed
      </div>
      {Object.entries(tree).map(([dir, items]) => (
        <div key={dir}>
          <div className="pr-section-title" style={{ paddingLeft: 8 }}>
            {dir}
          </div>
          {items.map((f, i) => (
            <div key={i} className="file-item" style={{ gap: 4 }}>
              <FileCode size={12} style={{ color: 'var(--accent)', flexShrink: 0 }} />
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
                {f.path.split('/').pop()}
              </span>
              <span className="file-lang">{f.language}</span>
              <button
                onClick={() => handleDelete(f.path)}
                disabled={deleting === f.path}
                title="Remove from index"
                style={{
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  color: 'var(--text-3)',
                  padding: '0 2px',
                  flexShrink: 0,
                  opacity: deleting === f.path ? 0.4 : 1,
                }}
              >
                <Trash2 size={11} />
              </button>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

export default React.memo(FileTreePanel);

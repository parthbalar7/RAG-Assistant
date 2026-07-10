import React, { useState, useRef, useMemo } from 'react';
import { X, FolderOpen, FileCode, Trash2, CheckCircle2, AlertCircle } from 'lucide-react';
import { api } from '../utils/api';

function IngestModal({ onClose, onToast, onRefresh, token }) {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [dirPath, setDirPath] = useState('');
  const [showManualPath, setShowManualPath] = useState(false);
  const fileRef = useRef(null);
  const folderRef = useRef(null);

  // Read all files from a dropped directory entry recursively
  const readEntryRecursive = (entry) => {
    return new Promise((resolve) => {
      if (entry.isFile) {
        entry.file(
          (f) => {
            // Preserve relative path
            const path = entry.fullPath.startsWith('/') ? entry.fullPath.slice(1) : entry.fullPath;
            const fileWithPath = new File([f], path, { type: f.type });
            resolve([fileWithPath]);
          },
          () => resolve([]),
        );
      } else if (entry.isDirectory) {
        const reader = entry.createReader();
        const allEntries = [];
        const readBatch = () => {
          reader.readEntries(
            (entries) => {
              if (entries.length === 0) {
                Promise.all(allEntries.map(readEntryRecursive)).then((results) => resolve(results.flat()));
              } else {
                allEntries.push(...entries);
                readBatch(); // Keep reading (readEntries returns max 100)
              }
            },
            () => resolve([]),
          );
        };
        readBatch();
      } else {
        resolve([]);
      }
    });
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setDragging(false);
    const items = e.dataTransfer.items;
    if (!items) return;

    const allFiles = [];
    const entries = [];
    for (let i = 0; i < items.length; i++) {
      const entry = items[i].webkitGetAsEntry ? items[i].webkitGetAsEntry() : null;
      if (entry) entries.push(entry);
    }

    for (const entry of entries) {
      const result = await readEntryRecursive(entry);
      allFiles.push(...result);
    }

    if (allFiles.length > 0) {
      setFiles((prev) => [...prev, ...allFiles]);
      onToast('info', `Found ${allFiles.length} files`);
    }
  };

  // Native folder picker (showDirectoryPicker API)
  const pickFolder = async () => {
    if (!('showDirectoryPicker' in window)) {
      setShowManualPath(true);
      return;
    }
    try {
      const dirHandle = await window.showDirectoryPicker();
      const collected = [];

      const readDir = async (handle, path = '') => {
        for await (const [name, entry] of handle.entries()) {
          const fullPath = path ? path + '/' + name : name;
          if (entry.kind === 'file') {
            try {
              const file = await entry.getFile();
              const fileWithPath = new File([file], fullPath, { type: file.type });
              collected.push(fileWithPath);
            } catch {
              /* skip unreadable */
            }
          } else if (entry.kind === 'directory') {
            // Skip common junk directories
            const skip = [
              'node_modules',
              '__pycache__',
              '.git',
              '.next',
              'dist',
              'build',
              '.venv',
              'venv',
              '.idea',
              'target',
              'bin',
              'obj',
            ];
            if (!skip.includes(name) && !name.startsWith('.')) {
              await readDir(entry, fullPath);
            }
          }
        }
      };

      await readDir(dirHandle);
      if (collected.length > 0) {
        setFiles((prev) => [...prev, ...collected]);
        onToast('info', `Found ${collected.length} files in "${dirHandle.name}"`);
      } else {
        onToast('error', 'No files found in selected folder');
      }
    } catch (e) {
      if (e.name !== 'AbortError') onToast('error', e.message);
    }
  };

  // Upload via webkitdirectory input fallback
  const handleFolderInput = (e) => {
    const newFiles = Array.from(e.target.files).map((f) => {
      const path = f.webkitRelativePath || f.name;
      return new File([f], path, { type: f.type });
    });
    setFiles((prev) => [...prev, ...newFiles]);
    if (newFiles.length > 0) onToast('info', `Found ${newFiles.length} files`);
  };

  // Manual path ingest (original method)
  const ingestByPath = async () => {
    if (!dirPath.trim()) return;
    setLoading(true);
    setResult(null);
    try {
      const r = await api.post('/api/ingest', { directory: dirPath.trim() }, token);
      setResult({ type: 'success', message: `Indexed ${r.chunks_indexed} chunks from ${r.documents_processed} files` });
      onToast('success', `Ingested ${r.documents_processed} files`);
      onRefresh();
    } catch (e) {
      setResult({ type: 'error', message: e.message });
    }
    setLoading(false);
  };

  // Upload collected files
  const uploadFiles = async () => {
    if (files.length === 0) return;
    setLoading(true);
    setResult(null);
    setProgress(0);

    const BATCH_SIZE = 200;
    let totalChunks = 0,
      totalFiles = [];

    for (let i = 0; i < files.length; i += BATCH_SIZE) {
      const batch = files.slice(i, i + BATCH_SIZE);

      try {
        const data = await api.upload(batch, token);
        totalChunks += data.chunks_indexed;
        totalFiles.push(...(data.files_processed || []));
      } catch (e) {
        setResult({ type: 'error', message: `Batch error: ${e.message}` });
        setLoading(false);
        return;
      }
      setProgress(Math.round(((i + batch.length) / files.length) * 100));
    }

    setResult({ type: 'success', message: `Indexed ${totalChunks} chunks from ${totalFiles.length} files` });
    onToast('success', `Ingested ${totalFiles.length} files (${totalChunks} chunks)`);
    onRefresh();
    setLoading(false);
  };

  const clearFiles = () => {
    setFiles([]);
    setResult(null);
    setProgress(0);
  };

  // Group files by top-level directory for display
  const fileTree = useMemo(() => {
    const tree = {};
    files.forEach((f) => {
      const parts = f.name.split('/');
      const dir = parts.length > 1 ? parts[0] : '(root)';
      if (!tree[dir]) tree[dir] = [];
      tree[dir].push(parts[parts.length - 1]);
    });
    return tree;
  }, [files]);

  const dirCount = Object.keys(fileTree).length;
  const ext = useMemo(() => {
    const m = {};
    files.forEach((f) => {
      const e = f.name.split('.').pop();
      m[e] = (m[e] || 0) + 1;
    });
    return Object.entries(m)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8);
  }, [files]);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 520 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2>Index Documents</h2>
          <button
            onClick={onClose}
            style={{ background: 'none', border: 'none', color: 'var(--text-3)', cursor: 'pointer' }}
          >
            <X size={16} />
          </button>
        </div>
        <p>Drag & drop a folder, or use the buttons below. Supports 80+ file types.</p>

        {/* Drop Zone */}
        <div
          className={'upload-zone' + (dragging ? ' dragging' : '')}
          onDragEnter={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
        >
          <FolderOpen
            size={28}
            style={{
              color: dragging ? 'var(--accent)' : 'var(--text-3)',
              marginBottom: 8,
            }}
          />
          <p style={{ fontSize: 13, color: dragging ? 'var(--accent)' : 'var(--text-2)', fontWeight: 500 }}>
            {dragging ? 'Drop folder or files here' : 'Drag & drop a project folder here'}
          </p>
          <p style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 4 }}>
            .py .js .ts .java .kt .cs .go .rs .html .css .sql .md .json .xml and 60+ more
          </p>
        </div>

        {/* Action Buttons */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
          <button
            className="modal-btn cancel"
            style={{ flex: 1, justifyContent: 'center', display: 'flex', alignItems: 'center', gap: 6 }}
            onClick={pickFolder}
          >
            <FolderOpen size={13} /> Select Folder
          </button>
          <button
            className="modal-btn cancel"
            style={{ flex: 1, justifyContent: 'center', display: 'flex', alignItems: 'center', gap: 6 }}
            onClick={() => fileRef.current && fileRef.current.click()}
          >
            <FileCode size={13} /> Select Files
          </button>
          <button
            className="modal-btn cancel"
            style={{ justifyContent: 'center', display: 'flex', alignItems: 'center', gap: 6, fontSize: 11 }}
            onClick={() => setShowManualPath(!showManualPath)}
            title="Type a local path"
          >
            {'\u2328'}
          </button>
        </div>

        {/* Hidden inputs */}
        <input
          ref={fileRef}
          type="file"
          multiple
          style={{ display: 'none' }}
          onChange={(e) => {
            setFiles((p) => [...p, ...Array.from(e.target.files)]);
          }}
        />
        <input
          ref={folderRef}
          type="file"
          webkitdirectory=""
          multiple
          style={{ display: 'none' }}
          onChange={handleFolderInput}
        />

        {/* Manual path input (toggle) */}
        {showManualPath && (
          <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
            <input
              type="text"
              placeholder="D:\Projects\MyApp"
              value={dirPath}
              onChange={(e) => setDirPath(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') ingestByPath();
              }}
              style={{ flex: 1 }}
            />
            <button
              className="modal-btn confirm"
              onClick={ingestByPath}
              disabled={loading || !dirPath.trim()}
              style={{ whiteSpace: 'nowrap' }}
            >
              {loading ? '...' : 'Index Path'}
            </button>
          </div>
        )}

        {/* File summary */}
        {files.length > 0 && (
          <div
            style={{
              background: 'var(--surface)',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border)',
              padding: 12,
              marginBottom: 12,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--accent)' }}>
                {files.length} files {dirCount > 1 ? `across ${dirCount} folders` : ''}
              </span>
              <button
                onClick={clearFiles}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-3)',
                  cursor: 'pointer',
                  fontSize: 11,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                }}
              >
                <Trash2 size={11} /> Clear
              </button>
            </div>
            {/* Extension breakdown */}
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
              {ext.map(([e, c]) => (
                <span
                  key={e}
                  style={{
                    fontSize: 10,
                    padding: '2px 8px',
                    borderRadius: 12,
                    background: 'var(--accent-tint)',
                    color: 'var(--accent)',
                    fontFamily: 'var(--font-mono)',
                    border: '1px solid var(--border)',
                  }}
                >
                  .{e} ({c})
                </span>
              ))}
            </div>
            {/* Folder list */}
            <div style={{ maxHeight: 120, overflowY: 'auto' }}>
              {Object.entries(fileTree).map(([dir, items]) => (
                <div
                  key={dir}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                    padding: '3px 0',
                    fontSize: 11,
                    color: 'var(--text-2)',
                    fontFamily: 'var(--font-mono)',
                  }}
                >
                  <FolderOpen size={11} style={{ color: 'var(--accent)', flexShrink: 0 }} />
                  <span style={{ color: 'var(--text-1)' }}>{dir}/</span>
                  <span style={{ color: 'var(--text-3)' }}>{items.length} files</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Progress */}
        {loading && (
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: progress + '%' }} />
          </div>
        )}
        {loading && (
          <div style={{ fontSize: 11, color: 'var(--text-3)', textAlign: 'center', fontFamily: 'var(--font-mono)' }}>
            {progress}% uploaded
          </div>
        )}

        {/* Result */}
        {result && (
          <div className={'result-banner ' + result.type}>
            {result.type === 'success' ? <CheckCircle2 size={12} /> : <AlertCircle size={12} />} {result.message}
          </div>
        )}

        {/* Actions */}
        <div className="modal-actions">
          <button className="modal-btn cancel" onClick={onClose}>
            Close
          </button>
          {files.length > 0 && (
            <button className="modal-btn confirm" onClick={uploadFiles} disabled={loading}>
              {loading ? 'Indexing...' : `Index ${files.length} files`}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default React.memo(IngestModal);

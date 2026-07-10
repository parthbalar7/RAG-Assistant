import React, { useState, useRef } from 'react';
import { FileText, Upload, CheckCircle2, AlertCircle } from 'lucide-react';
import { API, api, extractError } from '../utils/api';
import { useAuth } from '../contexts/AuthContext';

function PiUploadModal({ onClose, onToast, onDocAdded }) {
  const { token } = useAuth();
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState(null);
  const fileRef = useRef(null);

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setStatus(null);
    try {
      const f = new FormData();
      f.append('file', file);
      const h = token ? { Authorization: 'Bearer ' + token } : {};
      const r = await fetch(API + '/api/pageindex/upload', { method: 'POST', body: f, headers: h });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        throw new Error(extractError(d, 'Upload failed'));
      }
      const data = await r.json();
      setStatus({ type: 'success', msg: `Uploaded! Doc ID: ${data.doc_id}. Processing...` });
      onDocAdded({ doc_id: data.doc_id, filename: file.name, status: 'processing' });
      onToast('success', `PDF "${file.name}" submitted`);
      const pollId = setInterval(async () => {
        try {
          const s = await api.get('/api/pageindex/document/' + data.doc_id, token);
          if (s.status === 'completed') {
            clearInterval(pollId);
            setStatus({ type: 'success', msg: 'Tree index built! Ready for queries.' });
            onDocAdded({ doc_id: data.doc_id, filename: file.name, status: 'completed' });
            onToast('success', `"${file.name}" ready`);
          }
        } catch (e) {
          /* keep polling */
        }
      }, 5000);
      setTimeout(() => clearInterval(pollId), 300000);
    } catch (e) {
      setStatus({ type: 'error', msg: e.message });
      onToast('error', e.message);
    }
    setUploading(false);
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>Upload PDF to Tree Index</h2>
        <p>Builds a hierarchical tree index using Claude, then uses LLM reasoning for retrieval. Runs locally.</p>
        <div className="upload-zone" onClick={() => fileRef.current && fileRef.current.click()}>
          <input ref={fileRef} type="file" accept=".pdf" hidden onChange={(e) => setFile(e.target.files[0])} />
          {file ? (
            <span style={{ fontSize: 13, color: 'var(--accent)' }}>
              <FileText size={14} style={{ verticalAlign: -2 }} /> {file.name}
            </span>
          ) : (
            <span style={{ fontSize: 12, color: 'var(--text-3)' }}>
              <Upload size={18} style={{ display: 'block', margin: '0 auto 6px' }} /> Click to select a PDF
            </span>
          )}
        </div>
        {uploading && (
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: '70%' }} />
          </div>
        )}
        {status && (
          <div className={`result-banner ${status.type}`}>
            {status.type === 'success' ? <CheckCircle2 size={14} /> : <AlertCircle size={14} />} {status.msg}
          </div>
        )}
        <div className="modal-actions">
          <button className="modal-btn cancel" onClick={onClose}>
            Close
          </button>
          <button className="modal-btn confirm" onClick={handleUpload} disabled={!file || uploading}>
            {uploading ? 'Building tree...' : 'Upload & Index'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default React.memo(PiUploadModal);

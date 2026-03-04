import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import {
  Send, Menu, FolderOpen, Upload, Trash2, ChevronDown, ChevronRight, FileCode, Copy, Check,
  X, AlertCircle, CheckCircle2, Info, Plus, MessageSquare, Sun, Moon,
  PanelRightOpen, PanelRightClose, BarChart3, FolderTree, Settings, Bot, Route, LogIn, LogOut,
  Mic, MicOff, Eye, EyeOff, FileText, BookOpen, Sparkles, Brain, Search, PlusCircle
} from 'lucide-react';

const API = process.env.REACT_APP_API_URL || '';

/* Helper to extract error message from FastAPI response */
function extractError(d, fallback) {
  if (!d) return fallback || 'Unknown error';
  if (typeof d.detail === 'string') return d.detail;
  if (Array.isArray(d.detail)) return d.detail.map(e => e.msg || JSON.stringify(e)).join('; ');
  if (typeof d.detail === 'object') return d.detail.msg || d.detail.message || JSON.stringify(d.detail);
  if (typeof d.message === 'string') return d.message;
  if (typeof d.error === 'string') return d.error;
  return fallback || 'Unknown error';
}

/* ── API helpers ── */
const api = {
  get: async (path, token) => {
    const h = token ? { Authorization: 'Bearer ' + token } : {};
    const r = await fetch(API + path, { headers: h });
    if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(extractError(d, r.statusText)); }
    return r.json();
  },
  post: async (path, body, token) => {
    const h = { 'Content-Type': 'application/json' };
    if (token) h['Authorization'] = 'Bearer ' + token;
    const r = await fetch(API + path, { method: 'POST', headers: h, body: JSON.stringify(body) });
    if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(extractError(d, r.statusText)); }
    return r.json();
  },
  put: async (path, body, token) => {
    const h = { 'Content-Type': 'application/json' };
    if (token) h['Authorization'] = 'Bearer ' + token;
    const r = await fetch(API + path, { method: 'PUT', headers: h, body: JSON.stringify(body) });
    return r.json();
  },
  del: async (path, token) => {
    const h = token ? { Authorization: 'Bearer ' + token } : {};
    const r = await fetch(API + path, { method: 'DELETE', headers: h });
    return r.json();
  },
  upload: async (files) => {
    const f = new FormData();
    files.forEach(function(x) { f.append('files', x); });
    const r = await fetch(API + '/api/upload', { method: 'POST', body: f });
    if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(extractError(d, 'Upload failed')); }
    return r.json();
  },
};

async function* streamQuery(query, history, opts, token) {
  const h = { 'Content-Type': 'application/json' };
  if (token) h['Authorization'] = 'Bearer ' + token;
  const r = await fetch(API + '/api/query/stream', { method: 'POST', headers: h, body: JSON.stringify({ query, conversation_history: history, ...opts }) });
  if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(extractError(d, r.statusText)); }
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split('\n');
    buf = lines.pop() || '';
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try { yield JSON.parse(line.slice(6)); } catch (e) { /* skip */ }
      }
    }
  }
}

/* ── Particles Background ── */
function ParticlesBackground() {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animId;
    let mouse = { x: 0, y: 0 };
    const particles = [];
    const count = 60;

    const resize = () => { canvas.width = canvas.offsetWidth; canvas.height = canvas.offsetHeight; };
    resize();
    window.addEventListener('resize', resize);

    for (let i = 0; i < count; i++) {
      particles.push({
        x: Math.random() * canvas.width, y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.3, vy: (Math.random() - 0.5) * 0.3,
        r: Math.random() * 1.5 + 0.5,
        color: i % 3 === 0 ? '#00f0ff' : i % 3 === 1 ? '#a855f7' : '#f43f9e',
      });
    }

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      particles.forEach((p, i) => {
        p.x += p.vx; p.y += p.vy;
        // Mouse attraction
        const dx = mouse.x - p.x, dy = mouse.y - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 200) { p.vx += dx * 0.00005; p.vy += dy * 0.00005; }
        // Bounce
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
        // Draw particle
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = p.color; ctx.globalAlpha = 0.6; ctx.fill();
        // Draw connections
        particles.forEach((p2, j) => {
          if (j <= i) return;
          const d = Math.sqrt((p.x - p2.x) ** 2 + (p.y - p2.y) ** 2);
          if (d < 120) {
            ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = p.color; ctx.globalAlpha = (1 - d / 120) * 0.15;
            ctx.lineWidth = 0.5; ctx.stroke();
          }
        });
      });
      ctx.globalAlpha = 1;
      animId = requestAnimationFrame(draw);
    };

    const handleMouse = (e) => {
      const rect = canvas.getBoundingClientRect();
      mouse.x = e.clientX - rect.left; mouse.y = e.clientY - rect.top;
    };
    canvas.addEventListener('mousemove', handleMouse);
    draw();

    return () => { cancelAnimationFrame(animId); window.removeEventListener('resize', resize); canvas.removeEventListener('mousemove', handleMouse); };
  }, []);

  return <canvas ref={canvasRef} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'all', zIndex: 0 }} />;
}

/* ── Code Block ── */
function CodeBlock({ language, children }) {
  const [copied, setCopied] = useState(false);
  const code = String(children).replace(/\n$/, '');
  return (
    <div className="code-block">
      <div className="code-header">
        <span className="code-lang">{language || 'text'}</span>
        <button className="code-copy" onClick={() => { navigator.clipboard.writeText(code); setCopied(true); setTimeout(() => setCopied(false), 2000); }}>
          {copied ? <><Check size={10} /> copied</> : <><Copy size={10} /> copy</>}
        </button>
      </div>
      <SyntaxHighlighter language={language || 'text'} style={oneDark}
        customStyle={{ margin: 0, padding: '14px', background: 'var(--code-bg)', fontSize: '11.5px', lineHeight: '1.65' }}>
        {code}
      </SyntaxHighlighter>
    </div>
  );
}

/* ── Collapsible Sources with Previews ── */
function SourcesPanel({ sources, onViewPdf }) {
  const [open, setOpen] = useState(false);
  const [expanded, setExpanded] = useState({});
  if (!sources || sources.length === 0) return null;

  const toggle = (i) => setExpanded(p => ({ ...p, [i]: !p[i] }));

  return (
    <div className="sources-panel">
      <button className="sources-toggle" onClick={() => setOpen(!open)}>
        {open ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
        <BookOpen size={11} /> {sources.length} source{sources.length !== 1 ? 's' : ''} referenced
      </button>
      {open && <div className="sources-list">
        {sources.map((s, i) => (
          <div key={i} className={'source-card' + (expanded[i] ? ' expanded' : '')} onClick={() => toggle(i)}>
            <div className="source-card-header">
              <FileCode size={12} style={{ color: 'var(--neon-purple)', flexShrink: 0 }} />
              <span className="source-card-file">{s.file}</span>
              {s.page && <span className="source-card-page" onClick={e => { e.stopPropagation(); onViewPdf && onViewPdf(s); }}>p.{s.page}</span>}
              <span className="source-card-score">{(s.score * 100).toFixed(0)}%</span>
              {s.search_type && <span style={{ fontSize: 9, color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>{s.search_type}</span>}
            </div>
            <div className="source-card-preview">
              {s.preview || `Lines: ${s.lines || '?'} · Language: ${s.language || '?'}`}
            </div>
          </div>
        ))}
      </div>}
    </div>
  );
}

/* ── Toast ── */
function Toasts({ toasts, onDismiss }) {
  return <div className="toasts">{toasts.map(function(t) { return (
    <div key={t.id} className={'toast ' + t.type}>
      {t.type === 'success' && <CheckCircle2 size={14} />}
      {t.type === 'error' && <AlertCircle size={14} />}
      {t.type === 'info' && <Info size={14} />}
      <span style={{ flex: 1 }}>{t.message}</span>
      <button className="toast-close" onClick={() => onDismiss(t.id)}><X size={12} /></button>
    </div>
  ); })}</div>;
}

/* ── Auth Modal ── */
function AuthModal({ onClose, onAuth }) {
  const [tab, setTab] = useState('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    setError(''); setLoading(true);
    try {
      const endpoint = tab === 'login' ? '/api/auth/login' : '/api/auth/register';
      const body = tab === 'login' ? { username, password } : { username, password, display_name: displayName || username };
      const res = await api.post(endpoint, body);
      onAuth(res.token, res.user); onClose();
    } catch (e) { setError(e.message); }
    setLoading(false);
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()} style={{ width: 400 }}>
        <h2>{tab === 'login' ? 'Welcome Back' : 'Create Account'}</h2>
        <p>Sign in to save your chat history and preferences.</p>
        <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
          <button className={'modal-btn ' + (tab === 'login' ? 'confirm' : 'cancel')} style={{ flex: 1 }} onClick={() => setTab('login')}>Login</button>
          <button className={'modal-btn ' + (tab === 'register' ? 'confirm' : 'cancel')} style={{ flex: 1 }} onClick={() => setTab('register')}>Register</button>
        </div>
        <input type="text" placeholder="Username" value={username} onChange={e => setUsername(e.target.value)} />
        <input type="password" placeholder="Password" value={password}
          onChange={e => setPassword(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') submit(); }} />
        {tab === 'register' && <input type="text" placeholder="Display name (optional)" value={displayName} onChange={e => setDisplayName(e.target.value)} />}
        {error && <div className="result-banner error"><AlertCircle size={12} /> {error}</div>}
        <div className="modal-actions">
          <button className="modal-btn cancel" onClick={onClose}>Cancel</button>
          <button className="modal-btn confirm" onClick={submit} disabled={loading || !username || !password}>
            {loading ? '...' : tab === 'login' ? 'Login' : 'Register'}
          </button>
        </div>
      </div>
    </div>
  );
}

/* ── Ingest Modal (Drag & Drop folders + files) ── */
function IngestModal({ onClose, onToast, onRefresh }) {
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
        entry.file((f) => {
          // Preserve relative path
          const path = entry.fullPath.startsWith('/') ? entry.fullPath.slice(1) : entry.fullPath;
          const fileWithPath = new File([f], path, { type: f.type });
          resolve([fileWithPath]);
        }, () => resolve([]));
      } else if (entry.isDirectory) {
        const reader = entry.createReader();
        const allEntries = [];
        const readBatch = () => {
          reader.readEntries((entries) => {
            if (entries.length === 0) {
              Promise.all(allEntries.map(readEntryRecursive)).then((results) => resolve(results.flat()));
            } else {
              allEntries.push(...entries);
              readBatch(); // Keep reading (readEntries returns max 100)
            }
          }, () => resolve([]));
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
      setFiles(prev => [...prev, ...allFiles]);
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
            } catch { /* skip unreadable */ }
          } else if (entry.kind === 'directory') {
            // Skip common junk directories
            const skip = ['node_modules', '__pycache__', '.git', '.next', 'dist', 'build', '.venv', 'venv', '.idea', 'target', 'bin', 'obj'];
            if (!skip.includes(name) && !name.startsWith('.')) {
              await readDir(entry, fullPath);
            }
          }
        }
      };

      await readDir(dirHandle);
      if (collected.length > 0) {
        setFiles(prev => [...prev, ...collected]);
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
    const newFiles = Array.from(e.target.files).map(f => {
      const path = f.webkitRelativePath || f.name;
      return new File([f], path, { type: f.type });
    });
    setFiles(prev => [...prev, ...newFiles]);
    if (newFiles.length > 0) onToast('info', `Found ${newFiles.length} files`);
  };

  // Manual path ingest (original method)
  const ingestByPath = async () => {
    if (!dirPath.trim()) return;
    setLoading(true); setResult(null);
    try {
      const r = await api.post('/api/ingest', { directory: dirPath.trim() });
      setResult({ type: 'success', message: `Indexed ${r.chunks_indexed} chunks from ${r.documents_processed} files` });
      onToast('success', `Ingested ${r.documents_processed} files`); onRefresh();
    } catch (e) { setResult({ type: 'error', message: e.message }); }
    setLoading(false);
  };

  // Upload collected files
  const uploadFiles = async () => {
    if (files.length === 0) return;
    setLoading(true); setResult(null); setProgress(0);

    const BATCH_SIZE = 50;
    let totalChunks = 0, totalFiles = [];

    for (let i = 0; i < files.length; i += BATCH_SIZE) {
      const batch = files.slice(i, i + BATCH_SIZE);
      const formData = new FormData();
      batch.forEach(f => formData.append('files', f, f.name));

      try {
        const r = await fetch(API + '/api/upload', { method: 'POST', body: formData });
        if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(extractError(d, 'Upload failed')); }
        const data = await r.json();
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

  const clearFiles = () => { setFiles([]); setResult(null); setProgress(0); };

  // Group files by top-level directory for display
  const fileTree = useMemo(() => {
    const tree = {};
    files.forEach(f => {
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
    files.forEach(f => { const e = f.name.split('.').pop(); m[e] = (m[e] || 0) + 1; });
    return Object.entries(m).sort((a, b) => b[1] - a[1]).slice(0, 8);
  }, [files]);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 520 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2>Index Documents</h2>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--text-tertiary)', cursor: 'pointer' }}><X size={16} /></button>
        </div>
        <p>Drag & drop a folder, or use the buttons below. Supports 80+ file types.</p>

        {/* Drop Zone */}
        <div
          className={'upload-zone' + (dragging ? ' dragging' : '')}
          onDragEnter={(e) => { e.preventDefault(); setDragging(true); }}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          style={dragging ? { borderColor: 'var(--neon-cyan)', background: 'rgba(0,240,255,0.06)', boxShadow: 'inset 0 0 40px rgba(0,240,255,0.05)' } : {}}
        >
          <FolderOpen size={28} style={{ color: dragging ? 'var(--neon-cyan)' : 'var(--text-tertiary)', marginBottom: 8, transition: '0.2s' }} />
          <p style={{ fontSize: 13, color: dragging ? 'var(--neon-cyan)' : 'var(--text-secondary)', fontWeight: 500 }}>
            {dragging ? 'Drop folder or files here' : 'Drag & drop a project folder here'}
          </p>
          <p style={{ fontSize: 10, color: 'var(--text-tertiary)', marginTop: 4 }}>
            .py .js .ts .java .kt .cs .go .rs .html .css .sql .md .json .xml and 60+ more
          </p>
        </div>

        {/* Action Buttons */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
          <button className="modal-btn cancel" style={{ flex: 1, justifyContent: 'center', display: 'flex', alignItems: 'center', gap: 6 }} onClick={pickFolder}>
            <FolderOpen size={13} /> Select Folder
          </button>
          <button className="modal-btn cancel" style={{ flex: 1, justifyContent: 'center', display: 'flex', alignItems: 'center', gap: 6 }}
            onClick={() => fileRef.current && fileRef.current.click()}>
            <FileCode size={13} /> Select Files
          </button>
          <button className="modal-btn cancel" style={{ justifyContent: 'center', display: 'flex', alignItems: 'center', gap: 6, fontSize: 11 }}
            onClick={() => setShowManualPath(!showManualPath)} title="Type a local path">
            ⌨
          </button>
        </div>

        {/* Hidden inputs */}
        <input ref={fileRef} type="file" multiple style={{ display: 'none' }}
          onChange={e => { setFiles(p => [...p, ...Array.from(e.target.files)]); }} />
        <input ref={folderRef} type="file" webkitdirectory="" multiple style={{ display: 'none' }}
          onChange={handleFolderInput} />

        {/* Manual path input (toggle) */}
        {showManualPath && (
          <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
            <input type="text" placeholder="D:\Projects\MyApp" value={dirPath}
              onChange={e => setDirPath(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') ingestByPath(); }}
              style={{ flex: 1 }} />
            <button className="modal-btn confirm" onClick={ingestByPath} disabled={loading || !dirPath.trim()} style={{ whiteSpace: 'nowrap' }}>
              {loading ? '...' : 'Index Path'}
            </button>
          </div>
        )}

        {/* File summary */}
        {files.length > 0 && (
          <div style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', padding: 12, marginBottom: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--neon-cyan)' }}>
                {files.length} files {dirCount > 1 ? `across ${dirCount} folders` : ''}
              </span>
              <button onClick={clearFiles} style={{ background: 'none', border: 'none', color: 'var(--text-tertiary)', cursor: 'pointer', fontSize: 11, display: 'flex', alignItems: 'center', gap: 4 }}>
                <Trash2 size={11} /> Clear
              </button>
            </div>
            {/* Extension breakdown */}
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
              {ext.map(([e, c]) => (
                <span key={e} style={{ fontSize: 10, padding: '2px 8px', borderRadius: 12, background: 'var(--accent-soft)', color: 'var(--neon-cyan)', fontFamily: 'var(--font-mono)', border: '1px solid var(--border-neon)' }}>
                  .{e} ({c})
                </span>
              ))}
            </div>
            {/* Folder list */}
            <div style={{ maxHeight: 120, overflowY: 'auto' }}>
              {Object.entries(fileTree).map(([dir, items]) => (
                <div key={dir} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '3px 0', fontSize: 11, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                  <FolderOpen size={11} style={{ color: 'var(--neon-purple)', flexShrink: 0 }} />
                  <span style={{ color: 'var(--text-primary)' }}>{dir}/</span>
                  <span style={{ color: 'var(--text-tertiary)' }}>{items.length} files</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Progress */}
        {loading && <div className="progress-bar"><div className="progress-fill" style={{ width: progress + '%' }} /></div>}
        {loading && <div style={{ fontSize: 11, color: 'var(--text-tertiary)', textAlign: 'center', fontFamily: 'var(--font-mono)' }}>{progress}% uploaded</div>}

        {/* Result */}
        {result && <div className={'result-banner ' + result.type}>{result.type === 'success' ? <CheckCircle2 size={12} /> : <AlertCircle size={12} />} {result.message}</div>}

        {/* Actions */}
        <div className="modal-actions">
          <button className="modal-btn cancel" onClick={onClose}>Close</button>
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

/* ── Analytics Panel ── */
function AnalyticsPanel() {
  const [data, setData] = useState(null);
  useEffect(() => { api.get('/api/analytics?days=7').then(setData).catch(() => {}); }, []);
  if (!data) return <div style={{ padding: 16, fontSize: 12, color: 'var(--text-tertiary)' }}>Loading analytics...</div>;
  const COLORS = ['#00f0ff', '#a855f7', '#f43f9e', '#22f5a0', '#f59e0b', '#3b82f6'];
  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 14 }}>
        <div className="stat-card"><div className="stat-label">Total Queries</div><div className="stat-value">{data.total_queries}</div></div>
        <div className="stat-card"><div className="stat-label">Avg Latency</div><div className="stat-value">{(data.avg_latency_ms / 1000).toFixed(1)}s</div></div>
        <div className="stat-card"><div className="stat-label">Tokens Used</div><div className="stat-value">{(data.total_tokens / 1000).toFixed(0)}k</div></div>
        <div className="stat-card"><div className="stat-label">Categories</div><div className="stat-value">{data.categories ? data.categories.length : 0}</div></div>
      </div>
      {data.daily && data.daily.length > 0 && <>
        <div className="pr-section-title">Queries / Day</div>
        <div style={{ height: 140, marginBottom: 14 }}>
          <ResponsiveContainer>
            <BarChart data={data.daily}>
              <XAxis dataKey="date" tick={{ fontSize: 9, fill: 'var(--text-tertiary)' }} tickFormatter={d => d.slice(5)} />
              <YAxis tick={{ fontSize: 9, fill: 'var(--text-tertiary)' }} width={30} />
              <Tooltip contentStyle={{ background: 'var(--glass-strong)', border: '1px solid var(--border-neon)', borderRadius: 8, fontSize: 11, backdropFilter: 'blur(12px)' }} />
              <Bar dataKey="queries" fill="url(#barGradient)" radius={[4, 4, 0, 0]} />
              <defs><linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#00f0ff" /><stop offset="100%" stopColor="#a855f7" /></linearGradient></defs>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </>}
      {data.categories && data.categories.length > 0 && <>
        <div className="pr-section-title">Categories</div>
        <div style={{ height: 160 }}>
          <ResponsiveContainer>
            <PieChart>
              <Pie data={data.categories} dataKey="count" nameKey="name" cx="50%" cy="50%" innerRadius={35} outerRadius={60}
                label={({ name, percent }) => name + ' ' + (percent * 100).toFixed(0) + '%'}
                labelLine={false} style={{ fontSize: 9 }}>
                {data.categories.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        </div>
      </>}
    </div>
  );
}

/* ── PageIndex Upload Modal ── */
function PiUploadModal({ onClose, onToast, onDocAdded }) {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState(null);
  const fileRef = useRef(null);

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true); setStatus(null);
    try {
      const f = new FormData(); f.append('file', file);
      const r = await fetch(API + '/api/pageindex/upload', { method: 'POST', body: f });
      if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(extractError(d, 'Upload failed')); }
      const data = await r.json();
      setStatus({ type: 'success', msg: `Uploaded! Doc ID: ${data.doc_id}. Processing...` });
      onDocAdded({ doc_id: data.doc_id, filename: file.name, status: 'processing' });
      onToast('success', `PDF "${file.name}" submitted`);
      const pollId = setInterval(async () => {
        try {
          const s = await fetch(API + '/api/pageindex/document/' + data.doc_id).then(r => r.json());
          if (s.status === 'completed') {
            clearInterval(pollId);
            setStatus({ type: 'success', msg: 'Tree index built! Ready for queries.' });
            onDocAdded({ doc_id: data.doc_id, filename: file.name, status: 'completed' });
            onToast('success', `"${file.name}" ready`);
          }
        } catch (e) { /* keep polling */ }
      }, 5000);
      setTimeout(() => clearInterval(pollId), 300000);
    } catch (e) { setStatus({ type: 'error', msg: e.message }); onToast('error', e.message); }
    setUploading(false);
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <h2>Upload PDF to Tree Index</h2>
        <p>Builds a hierarchical tree index using Claude, then uses LLM reasoning for retrieval. Runs locally.</p>
        <div className="upload-zone" onClick={() => fileRef.current && fileRef.current.click()}>
          <input ref={fileRef} type="file" accept=".pdf" hidden onChange={e => setFile(e.target.files[0])} />
          {file ? <span style={{ fontSize: 13, color: 'var(--neon-cyan)' }}><FileText size={14} style={{ verticalAlign: -2 }} /> {file.name}</span>
                : <span style={{ fontSize: 12, color: 'var(--text-tertiary)' }}><Upload size={18} style={{ display: 'block', margin: '0 auto 6px' }} /> Click to select a PDF</span>}
        </div>
        {uploading && <div className="progress-bar"><div className="progress-fill" style={{ width: '70%' }} /></div>}
        {status && <div className={`result-banner ${status.type}`}>{status.type === 'success' ? <CheckCircle2 size={14} /> : <AlertCircle size={14} />} {status.msg}</div>}
        <div className="modal-actions">
          <button className="modal-btn cancel" onClick={onClose}>Close</button>
          <button className="modal-btn confirm" onClick={handleUpload} disabled={!file || uploading}>{uploading ? 'Building tree...' : 'Upload & Index'}</button>
        </div>
      </div>
    </div>
  );
}

/* ── PDF Viewer Panel ── */
function PdfViewerPanel({ source, onClose }) {
  if (!source) return null;
  return (
    <div className="pdf-viewer-panel">
      <div className="pdf-viewer-header">
        <span><FileText size={12} style={{ verticalAlign: -2 }} /> {source.file} — Page {source.page || '?'}</span>
        <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--text-tertiary)', cursor: 'pointer' }}><X size={16} /></button>
      </div>
      <div className="pdf-viewer-content">
        <div className="pdf-page-preview">{source.preview || 'No preview available for this page.'}</div>
      </div>
    </div>
  );
}

/* ── File Tree ── */

function IntegrityRadarPanel({ token, addToast, isReady }) {
  const [scan, setScan] = useState(null);
  const [history, setHistory] = useState([]);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState(null);

  const runScan = async () => {
    try {
      setRunning(true);
      setErr(null);

      const res = await api.post('/api/integrity/scan', { persist: true }, token);
      setScan(res);

      const hist = await api.get('/api/integrity/history?days=30&limit=20', token);
      setHistory(hist.scans || []);

      addToast('success', `Integrity scan complete · Health score: ${res.health?.score ?? '—'}`);
    } catch (e) {
      setErr(e.message || 'Scan failed');
      addToast('error', e.message || 'Integrity scan failed');
    } finally {
      setRunning(false);
    }
  };

  useEffect(() => {
    if (!isReady) return;

    (async () => {
      try {
        const hist = await api.get('/api/integrity/history?days=30&limit=20', token);
        setHistory(hist.scans || []);
      } catch {}
    })();
  }, [isReady, token]);

  const score = scan?.health?.score ?? null;
  const band = scan?.health?.band ?? '';
  const counts = scan?.health?.counts || {};
  const issues = scan?.issues || [];
  const recs = scan?.recommendations || [];

  const badgeClass = (sev) => {
    if (sev === 'critical') return 'sev critical';
    if (sev === 'high') return 'sev high';
    if (sev === 'medium') return 'sev medium';
    return 'sev low';
  };

  return (
    <div className="radar">
      <div className="radar-header">
        <div>
          <div className="pr-section-title" style={{ marginBottom: 6 }}>
            Knowledge Integrity & Risk Radar
          </div>
          <div className="radar-sub">
            Detect contradictions, blind spots, resilience gaps, and documentation drift.
          </div>
        </div>
        <button
          className={'radar-scan-btn ' + (running ? 'loading' : '')}
          onClick={runScan}
          disabled={running || !isReady}
        >
          <Sparkles size={14} /> {running ? 'Scanning…' : 'Run scan'}
        </button>
      </div>

      {!isReady && (
        <div className="radar-warn">
          <AlertCircle size={14} /> Index documents first to enable integrity scans.
        </div>
      )}

      {err && (
        <div className="radar-warn">
          <AlertCircle size={14} /> {err}
        </div>
      )}

      {isReady && (
        <>
          <div className="radar-top">
            <div className="radar-card">
              <div className="radar-card-title">Health</div>
              <div className="gauge">
                <div
                  className="gauge-ring"
                  style={score == null ? {} : { '--p': score }}
                />
                <div className="gauge-center">
                  <div className="gauge-score">
                    {score == null ? '—' : score}
                  </div>
                  <div className="gauge-band">{band || '—'}</div>
                </div>
              </div>
              <div className="radar-meta">
                <div>
                  <span className="k">Sampled</span>
                  <span className="v">{scan?.sampled_chunks ?? '—'}</span>
                </div>
                <div>
                  <span className="k">Total</span>
                  <span className="v">{scan?.total_chunks ?? '—'}</span>
                </div>
                <div>
                  <span className="k">Time</span>
                  <span className="v">
                    {scan?.duration_ms ? `${scan.duration_ms}ms` : '—'}
                  </span>
                </div>
              </div>
            </div>

            <div className="radar-card">
              <div className="radar-card-title">Signals</div>
              <div className="radar-signals">
                <div className="sig"><span>Contradictions</span><b>{counts.contradiction || 0}</b></div>
                <div className="sig"><span>Blind spots</span><b>{counts.blind_spot || 0}</b></div>
                <div className="sig"><span>Resilience</span><b>{counts.resilience_gap || 0}</b></div>
                <div className="sig"><span>Drift</span><b>{counts.drift || 0}</b></div>
              </div>
            </div>
          </div>

          <div className="radar-card" style={{ marginTop: 10 }}>
            <div className="radar-card-title">Top recommendations</div>
            {recs.length === 0 && <div className="radar-muted">Run a scan to get recommendations.</div>}
            {recs.map((r, i) => (
              <div key={i} className="rec">
                <CheckCircle2 size={14} /> {r}
              </div>
            ))}
          </div>

          <div className="radar-card" style={{ marginTop: 10 }}>
            <div className="radar-card-title">Issues</div>
            {issues.length === 0 && <div className="radar-muted">No issues to show.</div>}
            {issues.map((iss, i) => (
              <details key={i} className="issue">
                <summary>
                  <span className={badgeClass(iss.severity)}>{iss.severity}</span>
                  <span className="issue-title">{iss.title}</span>
                </summary>
                <div className="issue-body">
                  <div className="issue-desc">{iss.description}</div>
                </div>
              </details>
            ))}
          </div>
        </>
      )}
    </div>
  );
}


function FileTreePanel() {
  const [files, setFiles] = useState([]);
  useEffect(() => { api.get('/api/files').then(d => setFiles(d.files || [])).catch(() => {}); }, []);
  const tree = {};
  files.forEach(function(f) {
    const parts = f.path.split('/');
    const dir = parts.length > 1 ? parts.slice(0, -1).join('/') : '.';
    if (!tree[dir]) tree[dir] = [];
    tree[dir].push(f);
  });
  if (files.length === 0) return <div style={{ padding: 16, fontSize: 12, color: 'var(--text-tertiary)' }}>No files indexed yet.</div>;
  return (
    <div>
      <div style={{ fontSize: 11, color: 'var(--text-tertiary)', padding: '4px 8px', marginBottom: 4, fontFamily: 'var(--font-mono)' }}>{files.length} files indexed</div>
      {Object.entries(tree).map(([dir, items]) => (
        <div key={dir}>
          <div className="pr-section-title" style={{ paddingLeft: 8 }}>{dir}</div>
          {items.map((f, i) => (
            <div key={i} className="file-item">
              <FileCode size={12} style={{ color: 'var(--neon-cyan)', flexShrink: 0 }} />
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.path.split('/').pop()}</span>
              <span className="file-lang">{f.language}</span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

/* ── Voice Input Hook ── */
function useVoiceInput(onResult) {
  const [recording, setRecording] = useState(false);
  const recogRef = useRef(null);

  const toggle = useCallback(() => {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
      onResult(null, 'Speech recognition not supported in this browser');
      return;
    }
    if (recording && recogRef.current) {
      recogRef.current.stop(); setRecording(false); return;
    }
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recog = new SR();
    recog.continuous = false; recog.interimResults = false; recog.lang = 'en-US';
    recog.onresult = (e) => { const t = e.results[0][0].transcript; onResult(t); setRecording(false); };
    recog.onerror = () => setRecording(false);
    recog.onend = () => setRecording(false);
    recogRef.current = recog;
    recog.start(); setRecording(true);
  }, [recording, onResult]);

  return { recording, toggle };
}

/* ══════════════════════════════════════════════ */
/* Main App                                       */
/* ══════════════════════════════════════════════ */

/* ── Memory Panel ── */
function MemoryPanel({ token, onToast }) {
  const [memories, setMemories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState(null);
  const [searching, setSearching] = useState(false);
  const [showAdd, setShowAdd] = useState(false);
  const [expandedId, setExpandedId] = useState(null);
  const [stats, setStats] = useState({ total: 0, types: {} });

  const fetchMemories = useCallback(async () => {
    setLoading(true);
    try {
      const h = token ? { Authorization: 'Bearer ' + token } : {};
      const r = await fetch(API + '/api/memory', { headers: h });
      if (r.ok) {
        const data = await r.json();
        setMemories(data.fragments || []);
        const types = {};
        (data.fragments || []).forEach(f => { types[f.memory_type] = (types[f.memory_type] || 0) + 1; });
        setStats({ total: (data.fragments || []).length, types });
      }
    } catch (e) { /* silently fail */ }
    setLoading(false);
  }, [token]);

  useEffect(() => { fetchMemories(); }, [fetchMemories]);

  const handleSearch = async () => {
    if (!searchQuery.trim()) { setSearchResults(null); return; }
    setSearching(true);
    try {
      const h = token ? { Authorization: 'Bearer ' + token } : {};
      const r = await fetch(API + '/api/memory/search?q=' + encodeURIComponent(searchQuery) + '&top_k=8', { headers: h });
      if (r.ok) { const data = await r.json(); setSearchResults(data.results || []); }
    } catch (e) { onToast('error', 'Memory search failed'); }
    setSearching(false);
  };

  const handleDelete = async (id) => {
    try {
      const h = token ? { Authorization: 'Bearer ' + token } : {};
      await fetch(API + '/api/memory/' + id, { method: 'DELETE', headers: h });
      setMemories(p => p.filter(m => m.fragment_id !== id));
      setStats(p => ({ ...p, total: p.total - 1 }));
      onToast('info', 'Memory deleted');
    } catch (e) { onToast('error', 'Delete failed'); }
  };

  const handleClear = async () => {
    if (!window.confirm('Clear ALL memories? This cannot be undone.')) return;
    try {
      const h = token ? { Authorization: 'Bearer ' + token } : {};
      await fetch(API + '/api/memory', { method: 'DELETE', headers: h });
      setMemories([]); setStats({ total: 0, types: {} });
      onToast('info', 'All memories cleared');
    } catch (e) { onToast('error', 'Clear failed'); }
  };

  const handleAdd = async (content, memType, importance) => {
    try {
      const h = { 'Content-Type': 'application/json' };
      if (token) h['Authorization'] = 'Bearer ' + token;
      const r = await fetch(API + '/api/memory', {
        method: 'POST', headers: h,
        body: JSON.stringify({ content, memory_type: memType, importance }),
      });
      if (r.ok) { fetchMemories(); onToast('success', 'Memory stored'); setShowAdd(false); }
    } catch (e) { onToast('error', 'Add failed'); }
  };

  const typeColors = {
    fact: { bg: 'rgba(0,240,255,0.08)', color: 'var(--neon-cyan)', border: 'rgba(0,240,255,0.2)' },
    key_fact: { bg: 'rgba(0,240,255,0.08)', color: 'var(--neon-cyan)', border: 'rgba(0,240,255,0.2)' },
    pref: { bg: 'rgba(168,85,247,0.08)', color: 'var(--neon-purple)', border: 'rgba(168,85,247,0.2)' },
    user_preference: { bg: 'rgba(168,85,247,0.08)', color: 'var(--neon-purple)', border: 'rgba(168,85,247,0.2)' },
    decision: { bg: 'rgba(34,245,160,0.08)', color: 'var(--neon-green)', border: 'rgba(34,245,160,0.2)' },
    insight: { bg: 'rgba(245,158,11,0.08)', color: 'var(--warm)', border: 'rgba(245,158,11,0.2)' },
    summary: { bg: 'rgba(244,63,158,0.08)', color: 'var(--neon-pink)', border: 'rgba(244,63,158,0.2)' },
    conversation_summary: { bg: 'rgba(244,63,158,0.08)', color: 'var(--neon-pink)', border: 'rgba(244,63,158,0.2)' },
  };
  const typeLabel = (t) => (t || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  const displayList = searchResults !== null ? searchResults : memories;

  if (loading) return <div style={{ padding: 16, fontSize: 12, color: 'var(--text-tertiary)' }}>Loading memories...</div>;

  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 14 }}>
        <div className="stat-card" style={{ padding: 10 }}><div className="stat-label">Memories</div><div className="stat-value" style={{ fontSize: 18 }}>{stats.total}</div></div>
        <div className="stat-card" style={{ padding: 10 }}><div className="stat-label">Types</div><div className="stat-value" style={{ fontSize: 18 }}>{Object.keys(stats.types).length}</div></div>
      </div>
      {stats.total > 0 && <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 12 }}>
        {Object.entries(stats.types).map(([type, count]) => {
          const c = typeColors[type] || typeColors.fact;
          return <span key={type} style={{ fontSize: 9, padding: '2px 7px', borderRadius: 10, background: c.bg, color: c.color, border: '1px solid ' + c.border, fontFamily: 'var(--font-mono)' }}>{typeLabel(type)} ({count})</span>;
        })}
      </div>}
      <div style={{ display: 'flex', gap: 6, marginBottom: 10 }}>
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)', padding: '0 8px' }}>
          <Search size={12} style={{ color: 'var(--text-tertiary)', flexShrink: 0 }} />
          <input type="text" value={searchQuery} onChange={e => setSearchQuery(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') handleSearch(); if (e.key === 'Escape') { setSearchResults(null); setSearchQuery(''); } }}
            placeholder="Search memories..."
            style={{ flex: 1, border: 'none', background: 'transparent', color: 'var(--text-primary)', fontSize: 11, padding: '7px 6px', outline: 'none', fontFamily: 'var(--font-body)' }} />
          {searchResults !== null && <button onClick={() => { setSearchResults(null); setSearchQuery(''); }} style={{ background: 'none', border: 'none', color: 'var(--text-tertiary)', cursor: 'pointer', padding: 2 }}><X size={10} /></button>}
        </div>
        <button onClick={handleSearch} disabled={searching || !searchQuery.trim()} style={{ padding: '0 10px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-neon)', background: 'var(--accent-soft)', color: 'var(--neon-cyan)', cursor: 'pointer', fontSize: 11 }}>{searching ? '...' : 'Go'}</button>
      </div>
      {searchResults !== null && <div style={{ fontSize: 10, color: 'var(--text-tertiary)', marginBottom: 8, fontFamily: 'var(--font-mono)' }}>{searchResults.length} result{searchResults.length !== 1 ? 's' : ''} for "{searchQuery}"</div>}
      <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
        <button className="sl-footer-btn" style={{ flex: 1, justifyContent: 'center' }} onClick={() => setShowAdd(true)}><PlusCircle size={11} /> Add Memory</button>
        {stats.total > 0 && <button className="sl-footer-btn danger" style={{ justifyContent: 'center' }} onClick={handleClear}><Trash2 size={11} /></button>}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {displayList.length === 0 && <div style={{ textAlign: 'center', padding: 24, color: 'var(--text-tertiary)', fontSize: 12 }}>
          <Brain size={28} style={{ display: 'block', margin: '0 auto 8px', opacity: 0.3 }} />
          {stats.total === 0 ? 'No memories yet. Chat with the assistant and memories will be extracted automatically.' : 'No results found.'}
        </div>}
        {displayList.map((mem, i) => {
          const c = typeColors[mem.memory_type] || typeColors.fact;
          const isExp = expandedId === (mem.fragment_id || i);
          return (
            <div key={mem.fragment_id || i} className={'source-card' + (isExp ? ' expanded' : '')} style={{ cursor: 'pointer' }}
              onClick={() => setExpandedId(isExp ? null : (mem.fragment_id || i))}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4, flexWrap: 'wrap' }}>
                <span style={{ fontSize: 9, padding: '1px 6px', borderRadius: 8, background: c.bg, color: c.color, border: '1px solid ' + c.border, fontFamily: 'var(--font-mono)' }}>{typeLabel(mem.memory_type)}</span>
                {mem.similarity !== undefined && <span style={{ fontSize: 9, color: 'var(--neon-cyan)', fontFamily: 'var(--font-mono)' }}>{(mem.similarity * 100).toFixed(0)}%</span>}
                <span style={{ fontSize: 9, color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)', marginLeft: 'auto' }}>{'★'.repeat(Math.round((mem.importance || 0.5) * 5))}{'☆'.repeat(5 - Math.round((mem.importance || 0.5) * 5))}</span>
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-primary)', lineHeight: 1.55 }}>{mem.content}</div>
              {isExp && <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid var(--border)', animation: 'fadeUp 0.2s ease' }}>
                {mem.tags && (typeof mem.tags === 'string' ? JSON.parse(mem.tags || '[]') : mem.tags).length > 0 && <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 6 }}>
                  {(typeof mem.tags === 'string' ? JSON.parse(mem.tags) : mem.tags).map((tag, j) => <span key={j} style={{ fontSize: 9, padding: '1px 5px', borderRadius: 6, background: 'var(--bg-subtle)', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>#{tag}</span>)}
                </div>}
                {mem.source_query && <div style={{ fontSize: 10, color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)', marginBottom: 4 }}>From: "{mem.source_query.slice(0, 60)}{mem.source_query.length > 60 ? '...' : ''}"</div>}
                {mem.created_at > 0 && <div style={{ fontSize: 10, color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)', marginBottom: 6 }}>{new Date(mem.created_at * 1000).toLocaleDateString()} {new Date(mem.created_at * 1000).toLocaleTimeString()}</div>}
                <button onClick={(e) => { e.stopPropagation(); handleDelete(mem.fragment_id); }} style={{ fontSize: 10, padding: '3px 8px', borderRadius: 6, border: '1px solid rgba(239,68,68,0.2)', background: 'rgba(239,68,68,0.06)', color: 'var(--danger)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}><Trash2 size={10} /> Delete</button>
              </div>}
            </div>
          );
        })}
      </div>
      {showAdd && <AddMemoryModal onClose={() => setShowAdd(false)} onAdd={handleAdd} />}
    </div>
  );
}

function AddMemoryModal({ onClose, onAdd }) {
  const [content, setContent] = useState('');
  const [memType, setMemType] = useState('fact');
  const [importance, setImportance] = useState(0.7);
  const types = ['fact', 'pref', 'decision', 'insight'];
  return (
    <div className="modal-overlay" onClick={onClose} style={{ zIndex: 150 }}>
      <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 440 }}>
        <h2>Add Memory</h2>
        <p>Manually teach the assistant a fact, preference, or insight to remember.</p>
        <textarea value={content} onChange={e => setContent(e.target.value)} placeholder="e.g., The project uses FastAPI with PostgreSQL..."
          style={{ width: '100%', minHeight: 80, padding: 12, borderRadius: 'var(--radius-md)', border: '1px solid var(--border-neon)', background: 'var(--bg-surface)', color: 'var(--text-primary)', fontSize: 13, fontFamily: 'var(--font-body)', resize: 'vertical', outline: 'none', boxSizing: 'border-box' }} />
        <div style={{ display: 'flex', gap: 6, margin: '12px 0', flexWrap: 'wrap' }}>
          {types.map(t => <button key={t} onClick={() => setMemType(t)} className={memType === t ? 'modal-btn confirm' : 'modal-btn cancel'} style={{ fontSize: 11, padding: '5px 12px' }}>{t.replace(/_/g, ' ')}</button>)}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
          <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>Importance:</span>
          <input type="range" min="0" max="1" step="0.1" value={importance} onChange={e => setImportance(parseFloat(e.target.value))} style={{ flex: 1, accentColor: 'var(--neon-cyan)' }} />
          <span style={{ fontSize: 11, color: 'var(--neon-cyan)', fontFamily: 'var(--font-mono)', minWidth: 30 }}>{(importance * 100).toFixed(0)}%</span>
        </div>
        <div className="modal-actions">
          <button className="modal-btn cancel" onClick={onClose}>Cancel</button>
          <button className="modal-btn confirm" onClick={() => onAdd(content, memType, importance)} disabled={!content.trim()}><Brain size={12} style={{ verticalAlign: -2 }} /> Store</button>
        </div>
      </div>
    </div>
  );
}


export default function App() {
  const [theme, setTheme] = useState(() => { try { return localStorage.getItem('rag-theme') || 'dark'; } catch { return 'dark'; } });
  useEffect(() => { document.documentElement.setAttribute('data-theme', theme); try { localStorage.setItem('rag-theme', theme); } catch {} }, [theme]);

  const [token, setToken] = useState(() => { try { return localStorage.getItem('rag-token'); } catch { return null; } });
  const [user, setUser] = useState(null);
  const [showAuth, setShowAuth] = useState(false);

  useEffect(() => {
    if (token) { api.get('/api/auth/me', token).then(d => { if (d.user) setUser(d.user); else { setToken(null); try { localStorage.removeItem('rag-token'); } catch {} } }).catch(() => {}); }
  }, [token]);

  const handleAuth = (t, u) => { setToken(t); setUser(u); try { localStorage.setItem('rag-token', t); } catch {} };
  const handleLogout = () => { setToken(null); setUser(null); try { localStorage.removeItem('rag-token'); } catch {} };

  // Layout
  const [leftOpen, setLeftOpen] = useState(true);
  const [rightOpen, setRightOpen] = useState(false);
  const [rightTab, setRightTab] = useState('files');
  const [leftWidth, setLeftWidth] = useState(280);
  const [rightWidth, setRightWidth] = useState(320);
  const resizingRef = useRef(null);

  // Chat
  const [sessions, setSessions] = useState([]);
  const [activeSession, setActiveSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [stats, setStats] = useState(null);
  const [showIngest, setShowIngest] = useState(false);
  const [toasts, setToasts] = useState([]);
  const toastId = useRef(0);

  // Settings
  const [useReranking, setUseReranking] = useState(true);
  const [useStreaming, setUseStreaming] = useState(true);
  const [useHybrid, setUseHybrid] = useState(true);
  const [useRouting, setUseRouting] = useState(true);
  const [useAgent, setUseAgent] = useState(false);
  const [usePageIndex, setUsePageIndex] = useState(false);
  const [useMemory, setUseMemory] = useState(true);
  const [piDocs, setPiDocs] = useState([]);
  const [piActiveDoc, setPiActiveDoc] = useState(null);
  const [showPiUpload, setShowPiUpload] = useState(false);

  // New features
  const [showMdPreview, setShowMdPreview] = useState(false);
  const [pdfSource, setPdfSource] = useState(null);

  const chatEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Voice
  const voiceCallback = useCallback((text, err) => {
    if (err) { addToast('error', err); return; }
    if (text) setInput(p => p + (p ? ' ' : '') + text);
  }, []);
  const { recording, toggle: toggleVoice } = useVoiceInput(voiceCallback);

  const fetchStats = useCallback(() => { api.get('/api/stats').then(setStats).catch(() => setStats(null)); }, []);
  useEffect(() => { fetchStats(); const i = setInterval(fetchStats, 20000); return () => clearInterval(i); }, [fetchStats]);

  const fetchSessions = useCallback(() => { api.get('/api/sessions', token).then(d => setSessions(d.sessions || [])).catch(() => {}); }, [token]);
  useEffect(() => { fetchSessions(); }, [fetchSessions]);

  useEffect(() => {
    if (activeSession) {
      api.get('/api/sessions/' + activeSession + '/messages').then(d => {
        setMessages((d.messages || []).map(m => ({ role: m.role, content: m.content, sources: m.sources, metadata: m.metadata })));
      }).catch(() => {});
    }
  }, [activeSession]);

  useEffect(() => { if (chatEndRef.current) chatEndRef.current.scrollIntoView({ behavior: 'smooth' }); }, [messages, streaming]);
  useEffect(() => { if (textareaRef.current) { textareaRef.current.style.height = 'auto'; textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 140) + 'px'; } }, [input]);

  const addToast = useCallback((type, message) => { const id = ++toastId.current; setToasts(p => [...p, { id, type, message }]); setTimeout(() => setToasts(p => p.filter(t => t.id !== id)), 4000); }, []);
  const getHistory = () => messages.filter(m => m.role === 'user' || m.role === 'assistant').map(m => ({ role: m.role, content: m.content }));

  // Resize handlers
  const startResize = (side) => (e) => {
    e.preventDefault(); resizingRef.current = side;
    const handleMove = (e) => {
      if (resizingRef.current === 'left') setLeftWidth(Math.max(200, Math.min(500, e.clientX)));
      else if (resizingRef.current === 'right') setRightWidth(Math.max(240, Math.min(600, window.innerWidth - e.clientX)));
    };
    const handleUp = () => { resizingRef.current = null; document.removeEventListener('mousemove', handleMove); document.removeEventListener('mouseup', handleUp); };
    document.addEventListener('mousemove', handleMove); document.addEventListener('mouseup', handleUp);
  };

  const newSession = async () => {
    try { const s = await api.post('/api/sessions', {}, token); setSessions(p => [s, ...p]); setActiveSession(s.id); setMessages([]); }
    catch (e) { setActiveSession(null); setMessages([]); }
  };

  const handleSend = async () => {
    const q = input.trim();
    if (!q || loading || streaming) return;
    setInput('');
    let sid = activeSession;
    if (!sid) {
      try { const s = await api.post('/api/sessions', {}, token); setSessions(p => [s, ...p]); sid = s.id; setActiveSession(s.id); } catch (e) { /* ok */ }
    }
    setMessages(p => [...p, { role: 'user', content: q }]);
    const opts = { use_reranking: useReranking, use_hybrid: useHybrid, use_routing: useRouting, use_agent: useAgent, use_pageindex: !!(usePageIndex && piActiveDoc), pageindex_doc_id: piActiveDoc || null, use_memory: useMemory };

    if (useStreaming) {
      setStreaming(true);
      let msg = { role: 'assistant', content: '', sources: [], route: null, memoriesUsed: 0 };
      setMessages(p => [...p, msg]);
      try {
        for await (const ev of streamQuery(q, getHistory(), opts, token)) {
          if (ev.type === 'sources') msg = { ...msg, sources: ev.sources };
          else if (ev.type === 'route') msg = { ...msg, route: ev.route };
          else if (ev.type === 'memories') msg = { ...msg, memoriesUsed: ev.count };
          else if (ev.type === 'token') msg = { ...msg, content: msg.content + ev.token };
          setMessages(p => [...p.slice(0, -1), { ...msg }]);
        }
      } catch (e) { addToast('error', e.message); msg = { ...msg, content: msg.content + '\n\n**Error:** ' + e.message }; setMessages(p => [...p.slice(0, -1), { ...msg }]); }
      setStreaming(false);
    } else {
      setLoading(true);
      try {
        const r = await api.post('/api/query', { query: q, conversation_history: getHistory(), ...opts }, token);
        setMessages(p => [...p, { role: 'assistant', content: r.answer, sources: r.sources, meta: { model: r.model, latency: r.latency_ms, usage: r.usage }, route: r.route, memoriesUsed: r.memories_used || 0 }]);
        if (sid && messages.length === 0) {
          const title = q.slice(0, 40) + (q.length > 40 ? '...' : '');
          api.put('/api/sessions/' + sid, { title: title }, token).catch(() => {});
          fetchSessions();
        }
      } catch (e) { addToast('error', e.message); setMessages(p => [...p, { role: 'assistant', content: '**Error:** ' + e.message }]); }
      setLoading(false);
    }
    fetchSessions();
  };

  const isReady = stats && stats.document_count > 0;
  const prompts = ["How is the project structured?", "What API endpoints exist?", "Explain the config options", "Show error handling patterns"];

  return (
    <div className="app-layout">
      {/* ── Left Sidebar ── */}
      <aside className={'sidebar-left ' + (leftOpen ? '' : 'collapsed')} style={leftOpen ? { width: leftWidth, minWidth: leftWidth } : {}}>
        <div className="sl-header">
          <div className="sl-logo"><Sparkles size={16} /></div>
          <div className="sl-title">RAG Assistant</div>
        </div>
        <button className="sl-new-btn" onClick={newSession}><Plus size={14} /> New Chat</button>
        <div className="sl-sessions">
          {sessions.map(s => (
            <div key={s.id} className={'sl-session ' + (activeSession === s.id ? 'active' : '')} onClick={() => setActiveSession(s.id)}>
              <MessageSquare size={12} />
              <span className="sl-session-title">{s.title || 'New Chat'}</span>
              <button className="sl-session-del" onClick={e => { e.stopPropagation(); api.del('/api/sessions/' + s.id, token); setSessions(p => p.filter(x => x.id !== s.id)); if (activeSession === s.id) { setActiveSession(null); setMessages([]); } }}><X size={12} /></button>
            </div>
          ))}
        </div>
        <div className="sl-footer">
          <button className="sl-footer-btn" onClick={() => setShowIngest(true)}><FolderOpen size={12} /> Index Documents</button>
          <button className="sl-footer-btn danger" onClick={async () => { if (window.confirm('Clear all indexed docs?')) { await api.del('/api/collection', token); fetchStats(); addToast('info', 'Collection cleared'); } }}><Trash2 size={12} /> Clear Collection</button>
          {user
            ? <button className="sl-footer-btn" onClick={handleLogout}><LogOut size={12} /> {user.display_name}</button>
            : <button className="sl-footer-btn" onClick={() => setShowAuth(true)}><LogIn size={12} /> Sign In</button>}
        </div>
      </aside>

      {leftOpen && <div className="resize-handle" onMouseDown={startResize('left')} />}

      {/* ── Main Content ── */}
      <main className="main-content">
        <div className="topbar">
          <button className="topbar-btn" onClick={() => setLeftOpen(!leftOpen)}><Menu size={16} /></button>
          <span className="topbar-title">{isReady ? stats.document_count + ' chunks indexed' : 'Index documents to start'}</span>
          {useAgent && <span className="topbar-badge"><Bot size={10} /> AGENT</span>}
          {useMemory && <span className="topbar-badge" style={{ borderColor: 'rgba(168,85,247,0.3)', color: 'var(--neon-purple)', background: 'rgba(168,85,247,0.06)' }}><Brain size={10} /> MEMORY</span>}
          {usePageIndex && <span className="topbar-badge" style={{ borderColor: 'var(--border-glow)', color: 'var(--neon-purple)' }}>TREE SEARCH</span>}
          {useHybrid && !usePageIndex && <span className="topbar-badge">HYBRID</span>}
          <div className="topbar-right">
            <button className="topbar-btn" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>{theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}</button>
            <button className="topbar-btn" onClick={() => setRightOpen(!rightOpen)}>{rightOpen ? <PanelRightClose size={14} /> : <PanelRightOpen size={14} />}</button>
          </div>
        </div>

        <div className="chat-area">
          {messages.length === 0 ? (
            <div className="welcome">
              <ParticlesBackground />
              <div style={{ position: 'relative', zIndex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <div className="welcome-hero">
                  <div className="welcome-orb"><div className="welcome-orb-inner"><Sparkles size={20} color="#fff" /></div></div>
                </div>
                <h1>Documentation Assistant</h1>
                <p>{isReady ? "Ask me anything about your codebase. I'll search, reason, and cite every claim." : 'Click "Index Documents" in the sidebar to get started.'}</p>
                {isReady && <div className="welcome-chips">{prompts.map((p, i) => <button key={i} className="welcome-chip" onClick={() => { setInput(p); if (textareaRef.current) textareaRef.current.focus(); }}>{p}</button>)}</div>}
              </div>
            </div>
          ) : messages.map((msg, i) => (
            <div key={i} className="message">
              {msg.role === 'user' ? (
                <div className="msg-user"><div className="msg-user-bubble">{msg.content}</div></div>
              ) : (
                <div className="msg-assistant">
                  <div className="msg-label">
                    <div className="msg-dot"><Sparkles size={11} /></div>
                    <span className="msg-name">RAG Assistant</span>
                    {msg.meta && msg.meta.latency && <span style={{ fontSize: 9, color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>{(msg.meta.latency / 1000).toFixed(1)}s</span>}
                    {msg.route && <span className="msg-route"><Route size={8} /> {msg.route.category}</span>}
                    {msg.memoriesUsed > 0 && <span className="msg-route" style={{ borderColor: 'rgba(168,85,247,0.3)', color: 'var(--neon-purple)', background: 'rgba(168,85,247,0.08)' }}><Brain size={8} /> {msg.memoriesUsed} memories</span>}
                  </div>
                  <div className="msg-body">
                    <ReactMarkdown components={{
                      code({ node, inline, className, children, ...props }) {
                        const m = /language-(\w+)/.exec(className || '');
                        return !inline && m ? <CodeBlock language={m[1]}>{children}</CodeBlock> : <code className={className} {...props}>{children}</code>;
                      },
                    }}>{msg.content}</ReactMarkdown>
                    {streaming && i === messages.length - 1 && <span className="streaming-cursor" />}
                  </div>
                  {msg.route && msg.route.steps && <div className="agent-steps">
                    {Array.from({ length: msg.route.steps }, (_, j) => <span key={j} className="agent-step"><span className="step-icon" /> step {j + 1}</span>)}
                  </div>}
                  <SourcesPanel sources={msg.sources} onViewPdf={setPdfSource} />
                </div>
              )}
            </div>
          ))}
          {loading && <div className="message"><div className="msg-assistant"><div className="msg-label"><div className="msg-dot"><Sparkles size={11} /></div><span className="msg-name">RAG Assistant</span></div><div className="loading-dots"><span /><span /><span /></div></div></div>}
          <div ref={chatEndRef} />
        </div>

        <div className="input-area">
          {showMdPreview && input.trim() && (
            <div className="md-preview">
              <ReactMarkdown components={{ code({ className, children, ...props }) { return <code className={className} {...props}>{children}</code>; } }}>{input}</ReactMarkdown>
            </div>
          )}
          <div className="input-wrapper">
            <div className="input-box">
              <textarea ref={textareaRef} value={input} onChange={e => setInput(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
                placeholder={isReady ? "Ask about your codebase..." : "Index documents first..."} disabled={loading || streaming || !isReady} rows={1} />
              <div className="input-toolbar">
                <button className={'voice-btn' + (recording ? ' recording' : '')} onClick={toggleVoice} title="Voice input">
                  {recording ? <MicOff size={14} /> : <Mic size={14} />}
                </button>
                <button className={'preview-btn' + (showMdPreview ? ' active' : '')} onClick={() => setShowMdPreview(!showMdPreview)} title="Markdown preview">
                  {showMdPreview ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
                <button className="send-btn" onClick={handleSend} disabled={!input.trim() || loading || streaming || !isReady}><Send size={14} /></button>
              </div>
            </div>
            <div className="input-hint">Enter to send · Shift+Enter newline · {useAgent ? 'Agent' : useStreaming ? 'Stream' : 'Standard'} mode{recording ? ' · 🎙 Listening...' : ''}</div>
          </div>
        </div>
      </main>

      {rightOpen && <div className="resize-handle" onMouseDown={startResize('right')} />}

      {/* ── Right Panel ── */}
      <aside className={'panel-right ' + (rightOpen ? '' : 'collapsed')} style={rightOpen ? { width: rightWidth, minWidth: rightWidth } : {}}>
        <div className="pr-tabs">
          <button className={'pr-tab ' + (rightTab === 'files' ? 'active' : '')} onClick={() => setRightTab('files')}><FolderTree size={12} /> Files</button>
          <button className={'pr-tab ' + (rightTab === 'memory' ? 'active' : '')} onClick={() => setRightTab('memory')}><Brain size={12} /> Memory</button>
          <button className={'pr-tab ' + (rightTab === 'analytics' ? 'active' : '')} onClick={() => setRightTab('analytics')}><BarChart3 size={12} /> Analytics</button>
          <button className={'pr-tab ' + (rightTab === 'radar' ? 'active' : '')} onClick={() => setRightTab('radar')}><Sparkles size={12} /> Radar</button>
          <button className={'pr-tab ' + (rightTab === 'settings' ? 'active' : '')} onClick={() => setRightTab('settings')}><Settings size={12} /> Settings</button>
        </div>
        <div className="pr-content">
          {rightTab === 'files' && <FileTreePanel />}
          {rightTab === 'memory' && <MemoryPanel token={token} onToast={addToast} />}
          {rightTab === 'analytics' && <AnalyticsPanel />}
          {rightTab === 'radar' && <IntegrityRadarPanel token={token} addToast={addToast} isReady={isReady} />}
          {rightTab === 'settings' && <>
            <div className="pr-section-title">Retrieval</div>
            <div className="setting-row"><span>Hybrid search</span><div className={'toggle ' + (useHybrid ? 'on' : '')} onClick={() => setUseHybrid(!useHybrid)} /></div>
            <div className="setting-row"><span>Reranking</span><div className={'toggle ' + (useReranking ? 'on' : '')} onClick={() => setUseReranking(!useReranking)} /></div>
            <div className="setting-row"><span>Query routing</span><div className={'toggle ' + (useRouting ? 'on' : '')} onClick={() => setUseRouting(!useRouting)} /></div>
            <div className="pr-section-title">Generation</div>
            <div className="setting-row"><span>Stream responses</span><div className={'toggle ' + (useStreaming ? 'on' : '')} onClick={() => setUseStreaming(!useStreaming)} /></div>
            <div className="setting-row"><span>Agent mode</span><div className={'toggle ' + (useAgent ? 'on' : '')} onClick={() => setUseAgent(!useAgent)} /></div>
            <div className="pr-section-title">Memory</div>
            <div className="setting-row"><span><Brain size={12} style={{ verticalAlign: -2 }} /> Long-term memory</span><div className={'toggle ' + (useMemory ? 'on' : '')} onClick={() => setUseMemory(!useMemory)} /></div>
            {useMemory && <div style={{ fontSize: 10, color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)', marginBottom: 8, lineHeight: 1.6 }}>
              Extracts facts & preferences from conversations. Retrieved via embeddings before each response.
            </div>}
            <div className="pr-section-title">PageIndex (PDF)</div>
            <div className="setting-row"><span>Enable tree search</span><div className={'toggle ' + (usePageIndex ? 'on' : '')} onClick={() => setUsePageIndex(!usePageIndex)} /></div>
            {usePageIndex && <>
              <div style={{ fontSize: 10, color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)', marginBottom: 8, lineHeight: 1.6 }}>
                Local reasoning-based RAG for PDFs. No external API needed.
              </div>
              <button className="sl-footer-btn" style={{ marginBottom: 8 }} onClick={() => setShowPiUpload(true)}>
                <Upload size={12} /> Upload PDF
              </button>
              {piDocs.map((d, i) => (
                <div key={i} className="file-item" style={{ cursor: 'pointer', background: piActiveDoc === d.doc_id ? 'var(--accent-soft)' : undefined, borderRadius: 'var(--radius-sm)', border: piActiveDoc === d.doc_id ? '1px solid var(--border-neon)' : '1px solid transparent' }}
                  onClick={() => setPiActiveDoc(piActiveDoc === d.doc_id ? null : d.doc_id)}>
                  <FileText size={12} style={{ color: piActiveDoc === d.doc_id ? 'var(--neon-cyan)' : 'var(--text-tertiary)' }} />
                  <span style={{ fontSize: 11, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{d.filename}</span>
                  <span className="file-lang" style={{ background: d.status === 'completed' ? 'rgba(34,245,160,0.1)' : 'rgba(245,158,11,0.1)', color: d.status === 'completed' ? 'var(--neon-green)' : 'var(--warm)' }}>{d.status === 'completed' ? 'ready' : d.status}</span>
                </div>
              ))}
              {usePageIndex && !piActiveDoc && piDocs.length > 0 && <div style={{ fontSize: 10, color: 'var(--warm)', marginTop: 6, fontFamily: 'var(--font-mono)' }}>↑ Select a document to query</div>}
            </>}
            <div className="pr-section-title">System</div>
            <div style={{ fontSize: 10, color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)', lineHeight: 2 }}>
              Model: {stats ? stats.llm_model : '—'}<br />
              Embeddings: {stats ? stats.embedding_model : '—'}<br />
              Chunks: {stats ? stats.document_count : 0}<br />
              BM25: {stats ? stats.bm25_weight : 0.3} / Vector: {stats ? stats.vector_weight : 0.7}
            </div>
          </>}
        </div>
      </aside>

      {/* ── PDF Viewer Overlay ── */}
      {pdfSource && <PdfViewerPanel source={pdfSource} onClose={() => setPdfSource(null)} />}

      {/* ── Modals ── */}
      {showAuth && <AuthModal onClose={() => setShowAuth(false)} onAuth={handleAuth} />}
      {showIngest && <IngestModal onClose={() => setShowIngest(false)} onToast={addToast} onRefresh={fetchStats} />}
      {showPiUpload && <PiUploadModal onClose={() => setShowPiUpload(false)} onToast={addToast} onDocAdded={doc => setPiDocs(p => [...p.filter(d => d.doc_id !== doc.doc_id), doc])} />}
      <Toasts toasts={toasts} onDismiss={id => setToasts(p => p.filter(t => t.id !== id))} />
    </div>
  );
}
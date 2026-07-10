import React, { useState, useRef, useEffect, useCallback, Suspense, lazy, memo } from 'react';
import {
  Send,
  Menu,
  FolderOpen,
  Trash2,
  X,
  Plus,
  MessageSquare,
  Sun,
  Moon,
  PanelRightOpen,
  PanelRightClose,
  FolderTree,
  Settings,
  Bot,
  Route,
  LogIn,
  LogOut,
  Mic,
  MicOff,
  Eye,
  EyeOff,
  FileText,
  Sparkles,
  Brain,
  Network,
  Upload,
} from 'lucide-react';

import { api, streamQuery } from './utils/api';
import { useAuth } from './contexts/AuthContext';
import { useToast } from './contexts/ToastContext';
import { useSettings } from './contexts/SettingsContext';
import useVoiceInput from './hooks/useVoiceInput';

import ParticlesBackground from './components/ParticlesBackground';
import SourcesPanel from './components/SourcesPanel';
import DecomposedBadge from './components/DecomposedBadge';
import GraphPathBadge from './components/GraphPathBadge';
import GapPrompt from './components/GapPrompt';
import ProvenanceBadge from './components/ProvenanceBadge';
import Toasts from './components/Toasts';
import AuthModal from './components/AuthModal';
import ModelPicker from './components/ModelPicker';

const ReactMarkdown = lazy(() => import('react-markdown'));
const CodeBlock = lazy(() => import('./components/CodeBlock'));
const IngestModal = lazy(() => import('./components/IngestModal'));
const PdfViewerPanel = lazy(() => import('./components/PdfViewerPanel'));
const IntegrityRadarPanel = lazy(() => import('./components/IntegrityRadarPanel'));
const FileTreePanel = lazy(() => import('./components/FileTreePanel'));
const MemoryPanel = lazy(() => import('./components/MemoryPanel'));
const GraphPanel = lazy(() => import('./components/GraphPanel'));
const PiUploadModal = lazy(() => import('./components/PiUploadModal'));

function PanelFallback() {
  return <div style={{ padding: 16, fontSize: 12, color: 'var(--text-tertiary)' }}>Loading...</div>;
}

function MarkdownFallback({ children }) {
  return <div style={{ whiteSpace: 'pre-wrap' }}>{children}</div>;
}

const MarkdownContent = memo(function MarkdownContent({ children, codeBlocks = true }) {
  const components = codeBlocks
    ? {
        code({ node: _node, inline, className, children: codeChildren, ...props }) {
          const m = /language-(\w+)/.exec(className || '');
          return !inline && m ? (
            <Suspense
              fallback={
                <pre className="code-block">
                  <code>{codeChildren}</code>
                </pre>
              }
            >
              <CodeBlock language={m[1]}>{codeChildren}</CodeBlock>
            </Suspense>
          ) : (
            <code className={className} {...props}>
              {codeChildren}
            </code>
          );
        },
      }
    : {
        code({ className, children: codeChildren, ...props }) {
          return (
            <code className={className} {...props}>
              {codeChildren}
            </code>
          );
        },
      };

  return (
    <Suspense fallback={<MarkdownFallback>{children}</MarkdownFallback>}>
      <ReactMarkdown components={components}>{children}</ReactMarkdown>
    </Suspense>
  );
});

const MessageRow = memo(
  function MessageRow({ msg, isStreamingLast, token, setPdfSource, addToast, onRefreshEvent, gapContextRef }) {
    if (msg.role === 'user') {
      return (
        <div className="message">
          <div className="msg-user">
            <div className="msg-user-bubble">{msg.content}</div>
          </div>
        </div>
      );
    }

    return (
      <div className="message">
        <div className="msg-assistant">
          <div className="msg-label">
            <div className="msg-dot">
              <Sparkles size={11} />
            </div>
            <span className="msg-name">RAG Assistant</span>
            {msg.meta && msg.meta.latency && (
              <span style={{ fontSize: 9, color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>
                {(msg.meta.latency / 1000).toFixed(1)}s
              </span>
            )}
            {msg.route && (
              <span className="msg-route">
                <Route size={8} /> {msg.route.category}
              </span>
            )}
            {msg.memoriesUsed > 0 && (
              <span
                className="msg-route"
                style={{
                  borderColor: 'rgba(168,85,247,0.3)',
                  color: 'var(--neon-purple)',
                  background: 'rgba(168,85,247,0.08)',
                }}
              >
                <Brain size={8} /> {msg.memoriesUsed} memories
              </span>
            )}
          </div>
          <div className="msg-body">
            <MarkdownContent>{msg.content}</MarkdownContent>
            {isStreamingLast && <span className="streaming-cursor" />}
          </div>
          {msg.route && msg.route.steps && (
            <div className="agent-steps">
              {Array.from({ length: msg.route.steps }, (_, j) => (
                <span key={j} className="agent-step">
                  <span className="step-icon" /> step {j + 1}
                </span>
              ))}
            </div>
          )}
          <SourcesPanel sources={msg.sources} onViewPdf={setPdfSource} />
          <DecomposedBadge subQueries={msg.subQueries} />
          <GraphPathBadge graphPath={msg.graphPath} />
          <ProvenanceBadge provenance={msg.provenance} />
          <GapPrompt
            gap={msg.gap}
            query={msg.query || ''}
            token={token}
            contextRef={gapContextRef}
            onIngested={() => addToast('success', 'Web content ingested - refreshing answer...')}
            onRefreshEvent={onRefreshEvent}
          />
        </div>
      </div>
    );
  },
  (prev, next) => prev.msg === next.msg && prev.isStreamingLast === next.isStreamingLast && prev.token === next.token,
);

export default function App() {
  const { token, user, showAuth, setShowAuth, handleAuth, handleLogout } = useAuth();
  const { toasts, addToast, dismissToast } = useToast();
  const {
    useReranking,
    setUseReranking,
    useStreaming,
    setUseStreaming,
    useHybrid,
    setUseHybrid,
    useRouting,
    setUseRouting,
    useAgent,
    setUseAgent,
    usePageIndex,
    setUsePageIndex,
    useMemory,
    setUseMemory,
    useGraph,
    setUseGraph,
    useHyDE,
    setUseHyDE,
    useSplade,
    setUseSplade,
    useMultiQuery,
    setUseMultiQuery,
    useParentExpand,
    setUseParentExpand,
  } = useSettings();

  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem('rag-theme') || 'dark';
    } catch {
      return 'dark';
    }
  });
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('rag-theme', theme);
    } catch {}
  }, [theme]);

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
  const [filesRefreshKey, setFilesRefreshKey] = useState(0);

  // LLM backend
  const [llmStatus, setLlmStatus] = useState(null);
  const [ollamaModels, setOllamaModels] = useState([]);

  const [piDocs, setPiDocs] = useState([]);
  const [piActiveDoc, setPiActiveDoc] = useState(null);
  const [showPiUpload, setShowPiUpload] = useState(false);

  // UI
  const [showMdPreview, setShowMdPreview] = useState(false);
  const [pdfSource, setPdfSource] = useState(null);

  const wsRef = useRef(null);
  const refreshRef = useRef(null);
  const streamingMainRef = useRef(false);
  const gapContextRef = useRef({});
  const chatEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Voice
  const voiceCallback = useCallback(
    (text, err) => {
      if (err) {
        addToast('error', err);
        return;
      }
      if (text) setInput((p) => p + (p ? ' ' : '') + text);
    },
    [addToast],
  );
  const { recording, toggle: toggleVoice } = useVoiceInput(voiceCallback);

  const fetchStats = useCallback(() => {
    api
      .get('/api/stats', token)
      .then(setStats)
      .catch(() => setStats(null));
  }, [token]);
  useEffect(() => {
    fetchStats();
    const i = setInterval(fetchStats, 20000);
    return () => clearInterval(i);
  }, [fetchStats]);

  const fetchLlmStatus = useCallback(() => {
    api
      .get('/api/llm/status', token)
      .then((d) => {
        setLlmStatus(d);
      })
      .catch(() => {});
  }, [token]);
  useEffect(() => {
    if (token) fetchLlmStatus();
  }, [fetchLlmStatus, token]);

  const fetchOllamaModels = useCallback(() => {
    api
      .get('/api/llm/models', token)
      .then((d) => setOllamaModels(d.models || []))
      .catch(() => {});
  }, [token]);

  const switchLlm = useCallback(
    async (backend, model) => {
      try {
        await api.post('/api/llm/switch', { backend, model: model || null }, token);
        fetchLlmStatus();
      } catch (e) {
        addToast('error', 'Switch failed: ' + (e.message || e));
      }
    },
    [token, fetchLlmStatus, addToast],
  );

  const fetchSessions = useCallback(() => {
    api
      .get('/api/sessions', token)
      .then((d) => setSessions(d.sessions || []))
      .catch(() => {});
  }, [token]);
  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  useEffect(() => {
    if (activeSession) {
      api
        .get('/api/sessions/' + activeSession + '/messages', token)
        .then((d) => {
          setMessages(
            (d.messages || []).map((m) => ({
              role: m.role,
              content: m.content,
              sources: m.sources,
              metadata: m.metadata,
            })),
          );
        })
        .catch(() => {});
    }
  }, [activeSession, token]);

  useEffect(() => {
    if (chatEndRef.current) chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streaming]);
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 140) + 'px';
    }
  }, [input]);

  const getHistory = () =>
    messages
      .filter((m) => m.role === 'user' || m.role === 'assistant')
      .map((m) => ({ role: m.role, content: m.content }));

  // Resize handlers
  const startResize = (side) => (e) => {
    e.preventDefault();
    resizingRef.current = side;
    const handleMove = (e) => {
      if (resizingRef.current === 'left') setLeftWidth(Math.max(200, Math.min(500, e.clientX)));
      else if (resizingRef.current === 'right')
        setRightWidth(Math.max(240, Math.min(600, window.innerWidth - e.clientX)));
    };
    const handleUp = () => {
      resizingRef.current = null;
      document.removeEventListener('mousemove', handleMove);
      document.removeEventListener('mouseup', handleUp);
    };
    document.addEventListener('mousemove', handleMove);
    document.addEventListener('mouseup', handleUp);
  };

  const newSession = async () => {
    try {
      const s = await api.post('/api/sessions', {}, token);
      setSessions((p) => [s, ...p]);
      setActiveSession(s.id);
      setMessages([]);
    } catch (e) {
      setActiveSession(null);
      setMessages([]);
    }
  };

  const handleSend = async (queryOverride) => {
    const q = (queryOverride !== undefined ? queryOverride : input).trim();
    if (!q || loading || streaming) return;
    if (queryOverride === undefined) setInput('');
    let sid = activeSession;
    if (!sid) {
      try {
        const s = await api.post('/api/sessions', {}, token);
        setSessions((p) => [s, ...p]);
        sid = s.id;
        setActiveSession(s.id);
      } catch (e) {
        /* ok */
      }
    }
    setMessages((p) => [...p, { role: 'user', content: q }]);
    const opts = {
      use_reranking: useReranking,
      use_hybrid: useHybrid,
      use_routing: useRouting,
      use_hyde: useHyDE,
      use_splade: useSplade,
      use_multiquery: useMultiQuery,
      use_parent_expand: useParentExpand,
      use_agent: useAgent,
      use_pageindex: !!(usePageIndex && piActiveDoc),
      pageindex_doc_id: piActiveDoc || null,
      use_memory: useMemory,
      use_graph: useGraph,
    };

    if (useStreaming) {
      setStreaming(true);
      streamingMainRef.current = true;
      // Identity key: a refresh message appended mid-stream (research loop)
      // must never be clobbered by index-based writes to the last slot.
      const streamId = 'stream-' + Date.now();
      let msg = {
        role: 'assistant',
        content: '',
        sources: [],
        route: null,
        memoriesUsed: 0,
        provenance: null,
        graphPath: null,
        subQueries: null,
        gap: null,
        query: q,
        streamId,
      };
      setMessages((p) => [...p, msg]);
      const isFirstMsg = messages.length === 0;
      let flushTimer = null;
      const writeMsg = () => setMessages((p) => p.map((m) => (m.streamId === streamId ? { ...msg } : m)));
      const flushMessage = () => {
        if (flushTimer) {
          clearTimeout(flushTimer);
          flushTimer = null;
        }
        writeMsg();
      };
      const scheduleFlush = () => {
        if (flushTimer) return;
        flushTimer = setTimeout(() => {
          flushTimer = null;
          writeMsg();
        }, 50);
      };
      try {
        for await (const ev of streamQuery(q, getHistory(), { ...opts, session_id: sid }, token, wsRef)) {
          if (ev.type === 'sources') msg = { ...msg, sources: ev.sources };
          else if (ev.type === 'route') msg = { ...msg, route: ev.route };
          else if (ev.type === 'memories') msg = { ...msg, memoriesUsed: ev.count };
          else if (ev.type === 'provenance') msg = { ...msg, provenance: ev.map };
          else if (ev.type === 'graph_path') msg = { ...msg, graphPath: ev.traversal };
          else if (ev.type === 'decomposed') msg = { ...msg, subQueries: ev.sub_queries };
          else if (ev.type === 'gap_detected')
            msg = { ...msg, gap: { topic: ev.topic, reason: ev.reason, top_score: ev.top_score } };
          else if (ev.type === 'research_iteration')
            addToast('info', `Refining web search (round ${ev.iteration || 2}): ${ev.query || ev.topic || ''}`);
          else if (ev.type === 'token') msg = { ...msg, content: msg.content + ev.token };
          else if (ev.type === 'session_renamed') {
            setSessions((p) => p.map((s) => (s.id === sid ? { ...s, title: ev.title } : s)));
          } else if (ev.type === 'error') {
            addToast('error', ev.message);
            msg = { ...msg, content: msg.content + (msg.content ? '\n\n' : '') + '**Error:** ' + ev.message };
            flushMessage();
            break;
          }
          if (ev.type === 'token') scheduleFlush();
          else if (ev.type !== 'session_renamed' && ev.type !== 'research_iteration') flushMessage();
        }
        flushMessage();
      } catch (e) {
        addToast('error', e.message);
        msg = { ...msg, content: msg.content + '\n\n**Error:** ' + e.message };
        flushMessage();
      }
      // Frontend fallback: name session from first message if backend didn't emit session_renamed
      if (isFirstMsg && sid) {
        const title = q.slice(0, 50) + (q.length > 50 ? '...' : '');
        setSessions((p) => p.map((s) => (s.id === sid && (s.title === 'New Chat' || !s.title) ? { ...s, title } : s)));
      }
      streamingMainRef.current = false;
      // A refresh stream started mid-answer now owns the streaming flag.
      if (!refreshRef.current) setStreaming(false);
    } else {
      setLoading(true);
      try {
        const r = await api.post(
          '/api/query',
          { query: q, conversation_history: getHistory(), session_id: sid, ...opts },
          token,
        );
        setMessages((p) => [
          ...p,
          {
            role: 'assistant',
            content: r.answer,
            sources: r.sources,
            meta: { model: r.model, latency: r.latency_ms, usage: r.usage },
            route: r.route,
            memoriesUsed: r.memories_used || 0,
          },
        ]);
        if (sid && messages.length === 0) {
          const title = q.slice(0, 50) + (q.length > 50 ? '...' : '');
          api.put('/api/sessions/' + sid, { title }, token).catch(() => {});
          setSessions((p) => p.map((s) => (s.id === sid ? { ...s, title } : s)));
        }
      } catch (e) {
        addToast('error', e.message);
        setMessages((p) => [...p, { role: 'assistant', content: '**Error:** ' + e.message }]);
      }
      setLoading(false);
    }
    fetchSessions();
  };
  // After an approved web augment the backend regenerates the answer itself
  // (research loop) and streams it on GapPrompt's own socket — GapPrompt
  // forwards those events here so the refreshed answer lands in the chat as a
  // fresh assistant message, without a client-side requery.
  const handleRefreshEvent = useCallback(
    (ev, query) => {
      if (ev.type === 'research_iteration') {
        addToast('info', `Refining web search (round ${ev.iteration || 2}): ${ev.query || ev.topic || ''}`);
        return;
      }
      let r = refreshRef.current;
      if (!r && (ev.type === 'token' || ev.type === 'sources')) {
        const id = 'refresh-' + Date.now();
        r = refreshRef.current = { id, buf: '', timer: null };
        setStreaming(true);
        setMessages((p) => [...p, { role: 'assistant', content: '', sources: [], gap: null, query, refreshId: id }]);
      }
      if (!r) return;
      const id = r.id;
      const flush = () => {
        if (r.timer) clearTimeout(r.timer);
        r.timer = null;
        const text = r.buf;
        r.buf = '';
        if (!text) return;
        setMessages((p) => p.map((m) => (m.refreshId === id ? { ...m, content: m.content + text } : m)));
      };
      if (ev.type === 'token') {
        r.buf += ev.token;
        if (!r.timer) r.timer = setTimeout(flush, 50);
      } else if (ev.type === 'sources') {
        setMessages((p) => p.map((m) => (m.refreshId === id ? { ...m, sources: ev.sources } : m)));
      } else if (ev.type === 'done' || ev.type === 'error') {
        flush();
        refreshRef.current = null;
        if (!streamingMainRef.current) setStreaming(false);
        if (ev.type === 'error' && ev.message) addToast('error', ev.message);
      }
    },
    [addToast],
  );

  const hasIndexedDocs = stats && stats.document_count > 0;
  const isReady = hasIndexedDocs || (usePageIndex && !!piActiveDoc);
  // Always-fresh context for GapPrompt's research socket (stable ref identity
  // survives MessageRow's memo; .current is read at click time).
  gapContextRef.current = {
    sessionId: activeSession,
    history: getHistory(),
    opts: {
      use_reranking: useReranking,
      use_hybrid: useHybrid,
      use_hyde: useHyDE,
      use_splade: useSplade,
      use_multiquery: useMultiQuery,
    },
  };
  const prompts = [
    'How is the project structured?',
    'What API endpoints exist?',
    'Explain the config options',
    'Show error handling patterns',
  ];

  // Auth gate
  if (!token) {
    return (
      <>
        <AuthModal required onClose={() => {}} onAuth={handleAuth} />
        <Toasts toasts={toasts} onDismiss={dismissToast} />
      </>
    );
  }

  return (
    <div className="app-layout">
      {/* ── Left Sidebar ── */}
      <aside
        className={'sidebar-left ' + (leftOpen ? '' : 'collapsed')}
        style={leftOpen ? { width: leftWidth, minWidth: leftWidth } : {}}
      >
        <div className="sl-header">
          <div className="sl-logo">
            <Sparkles size={16} />
          </div>
          <div className="sl-title">Parth's RAG Assistant</div>
        </div>
        <button className="sl-new-btn" onClick={newSession}>
          <Plus size={14} /> New Chat
        </button>
        <div className="sl-sessions">
          {sessions.map((s) => (
            <div
              key={s.id}
              className={'sl-session ' + (activeSession === s.id ? 'active' : '')}
              onClick={() => setActiveSession(s.id)}
            >
              <MessageSquare size={12} />
              <span className="sl-session-title">{s.title || 'New Chat'}</span>
              <button
                className="sl-session-del"
                onClick={(e) => {
                  e.stopPropagation();
                  api.del('/api/sessions/' + s.id, token);
                  setSessions((p) => p.filter((x) => x.id !== s.id));
                  if (activeSession === s.id) {
                    setActiveSession(null);
                    setMessages([]);
                  }
                }}
              >
                <X size={12} />
              </button>
            </div>
          ))}
        </div>
        <div className="sl-footer">
          <button className="sl-footer-btn" onClick={() => setShowIngest(true)}>
            <FolderOpen size={12} /> Index Documents
          </button>
          <button
            className="sl-footer-btn danger"
            onClick={async () => {
              if (window.confirm('Clear all indexed docs?')) {
                await api.del('/api/collection', token);
                fetchStats();
                setFilesRefreshKey((k) => k + 1);
                addToast('info', 'Collection cleared');
              }
            }}
          >
            <Trash2 size={12} /> Clear Collection
          </button>
          {user ? (
            <button className="sl-footer-btn" onClick={handleLogout}>
              <LogOut size={12} /> {user.display_name}
            </button>
          ) : (
            <button className="sl-footer-btn" onClick={() => setShowAuth(true)}>
              <LogIn size={12} /> Sign In
            </button>
          )}
        </div>
      </aside>

      {leftOpen && <div className="resize-handle" onMouseDown={startResize('left')} />}

      {/* ── Main Content ── */}
      <main className="main-content">
        <div className="topbar">
          <button className="topbar-btn" onClick={() => setLeftOpen(!leftOpen)}>
            <Menu size={16} />
          </button>
          <span className="topbar-title">
            {hasIndexedDocs
              ? stats.document_count + ' chunks indexed'
              : usePageIndex && piActiveDoc
                ? 'Tree search active'
                : 'Index documents to start'}
          </span>
          {useAgent && (
            <span className="topbar-badge">
              <Bot size={10} /> AGENT
            </span>
          )}
          {useMemory && (
            <span
              className="topbar-badge"
              style={{
                borderColor: 'rgba(168,85,247,0.3)',
                color: 'var(--neon-purple)',
                background: 'rgba(168,85,247,0.06)',
              }}
            >
              <Brain size={10} /> MEMORY
            </span>
          )}
          {usePageIndex && (
            <span className="topbar-badge" style={{ borderColor: 'var(--border-glow)', color: 'var(--neon-purple)' }}>
              TREE SEARCH
            </span>
          )}
          {useHybrid && !usePageIndex && <span className="topbar-badge">HYBRID</span>}
          <div className="topbar-right">
            <button className="topbar-btn" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
              {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
            </button>
            <button className="topbar-btn" onClick={() => setRightOpen(!rightOpen)}>
              {rightOpen ? <PanelRightClose size={14} /> : <PanelRightOpen size={14} />}
            </button>
          </div>
        </div>

        <div className="chat-area">
          {messages.length === 0 ? (
            <div className="welcome">
              <ParticlesBackground />
              <div
                style={{
                  position: 'relative',
                  zIndex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                }}
              >
                <div className="welcome-hero">
                  <div className="welcome-orb">
                    <div className="welcome-orb-inner">
                      <Sparkles size={20} color="#fff" />
                    </div>
                  </div>
                </div>
                <h1>Documentation Assistant</h1>
                <p>
                  {isReady
                    ? "Ask me anything about your codebase. I'll search, reason, and cite every claim."
                    : 'Click "Index Documents" in the sidebar to get started.'}
                </p>
                {isReady && (
                  <div className="welcome-chips">
                    {prompts.map((p, i) => (
                      <button
                        key={i}
                        className="welcome-chip"
                        onClick={() => {
                          setInput(p);
                          if (textareaRef.current) textareaRef.current.focus();
                        }}
                      >
                        {p}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ) : (
            messages.map((msg, i) => (
              <MessageRow
                key={i}
                msg={msg}
                isStreamingLast={streaming && i === messages.length - 1}
                token={token}
                setPdfSource={setPdfSource}
                addToast={addToast}
                onRefreshEvent={handleRefreshEvent}
                gapContextRef={gapContextRef}
              />
            ))
          )}
          {loading && (
            <div className="message">
              <div className="msg-assistant">
                <div className="msg-label">
                  <div className="msg-dot">
                    <Sparkles size={11} />
                  </div>
                  <span className="msg-name">RAG Assistant</span>
                </div>
                <div className="loading-dots">
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="input-area">
          {showMdPreview && input.trim() && (
            <div className="md-preview">
              <MarkdownContent codeBlocks={false}>{input}</MarkdownContent>
            </div>
          )}
          <div className="input-wrapper">
            <div className="input-box">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder={isReady ? 'Ask about your codebase...' : 'No documents indexed — chat uses memory only...'}
                disabled={loading || streaming}
                rows={1}
              />
              <div className="input-toolbar">
                <button
                  className={'voice-btn' + (recording ? ' recording' : '')}
                  onClick={toggleVoice}
                  title="Voice input"
                >
                  {recording ? <MicOff size={14} /> : <Mic size={14} />}
                </button>
                <button
                  className={'preview-btn' + (showMdPreview ? ' active' : '')}
                  onClick={() => setShowMdPreview(!showMdPreview)}
                  title="Markdown preview"
                >
                  {showMdPreview ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
                <ModelPicker
                  llmStatus={llmStatus}
                  ollamaModels={ollamaModels}
                  onFetchModels={fetchOllamaModels}
                  onSwitch={switchLlm}
                />
                {streaming ? (
                  <button
                    className="send-btn stop-btn"
                    onClick={() => {
                      if (wsRef.current) {
                        wsRef.current.close();
                        wsRef.current = null;
                      }
                    }}
                    title="Stop generation"
                  >
                    <X size={14} />
                  </button>
                ) : (
                  <button className="send-btn" onClick={() => handleSend()} disabled={!input.trim() || loading}>
                    <Send size={14} />
                  </button>
                )}
              </div>
            </div>
            <div className="input-hint">
              Enter to send · Shift+Enter newline · {useAgent ? 'Agent' : useStreaming ? 'Stream' : 'Standard'} mode
              {recording ? ' · 🎙 Listening...' : ''}
            </div>
          </div>
        </div>
      </main>

      {rightOpen && <div className="resize-handle" onMouseDown={startResize('right')} />}

      {/* ── Right Panel ── */}
      <aside
        className={'panel-right ' + (rightOpen ? '' : 'collapsed')}
        style={rightOpen ? { width: rightWidth, minWidth: rightWidth } : {}}
      >
        <div className="pr-tabs">
          <button
            className={'pr-tab ' + (rightTab === 'files' ? 'active' : '')}
            onClick={() => setRightTab('files')}
            title="Files"
          >
            <FolderTree size={15} />
          </button>
          <button
            className={'pr-tab ' + (rightTab === 'memory' ? 'active' : '')}
            onClick={() => setRightTab('memory')}
            title="Memory"
          >
            <Brain size={15} />
          </button>
          <button
            className={'pr-tab ' + (rightTab === 'radar' ? 'active' : '')}
            onClick={() => setRightTab('radar')}
            title="Knowledge Radar"
          >
            <Sparkles size={15} />
          </button>
          <button
            className={'pr-tab ' + (rightTab === 'graph' ? 'active' : '')}
            onClick={() => setRightTab('graph')}
            title="Knowledge Graph"
          >
            <Network size={15} />
          </button>
          <button
            className={'pr-tab ' + (rightTab === 'settings' ? 'active' : '')}
            onClick={() => setRightTab('settings')}
            title="Settings"
          >
            <Settings size={15} />
          </button>
        </div>
        <div className="pr-content">
          <Suspense fallback={<PanelFallback />}>
            {rightTab === 'files' && <FileTreePanel refreshKey={filesRefreshKey} token={token} onToast={addToast} />}
            {rightTab === 'memory' && <MemoryPanel token={token} onToast={addToast} />}
            {rightTab === 'radar' && <IntegrityRadarPanel token={token} addToast={addToast} isReady={isReady} />}
            {rightTab === 'graph' && <GraphPanel token={token} onToast={addToast} isReady={isReady} />}
          </Suspense>
          {rightTab === 'settings' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div className="settings-card">
                <div className="settings-card-title">Retrieval</div>
                <div className="setting-row">
                  <span>Hybrid search</span>
                  <div className={'toggle ' + (useHybrid ? 'on' : '')} onClick={() => setUseHybrid(!useHybrid)} />
                </div>
                <div className="setting-row">
                  <span>Reranking</span>
                  <div
                    className={'toggle ' + (useReranking ? 'on' : '')}
                    onClick={() => setUseReranking(!useReranking)}
                  />
                </div>
                <div className="setting-row">
                  <span>Query routing</span>
                  <div className={'toggle ' + (useRouting ? 'on' : '')} onClick={() => setUseRouting(!useRouting)} />
                </div>
                <div className="setting-row">
                  <span>HyDE query expansion</span>
                  <div className={'toggle ' + (useHyDE ? 'on' : '')} onClick={() => setUseHyDE(!useHyDE)} />
                </div>
                {useHyDE && (
                  <div className="settings-hint">
                    Generates a hypothetical answer to improve vector search relevance. Adds ~300-800ms per query.
                  </div>
                )}
                <div className="setting-row">
                  <span>Multi-query expansion</span>
                  <div
                    className={'toggle ' + (useMultiQuery ? 'on' : '')}
                    onClick={() => setUseMultiQuery(!useMultiQuery)}
                  />
                </div>
                {useMultiQuery && (
                  <div className="settings-hint">
                    Generates paraphrase variants and fuses their results (RAG-Fusion). Takes precedence over HyDE when
                    both are on.
                  </div>
                )}
                <div className="setting-row">
                  <span>SPLADE sparse retrieval</span>
                  <div className={'toggle ' + (useSplade ? 'on' : '')} onClick={() => setUseSplade(!useSplade)} />
                </div>
                {useSplade && (
                  <div className="settings-hint">
                    Learned vocabulary expansion replaces BM25. Requires RAG_SPLADE_ENABLED=true in .env to take effect.
                  </div>
                )}
                <div className="setting-row">
                  <span>Parent context expansion</span>
                  <div
                    className={'toggle ' + (useParentExpand ? 'on' : '')}
                    onClick={() => setUseParentExpand(!useParentExpand)}
                  />
                </div>
                {useParentExpand && (
                  <div className="settings-hint">
                    Expands top-ranked chunks with neighboring text from the same section after reranking (small-to-big
                    retrieval).
                  </div>
                )}
              </div>

              <div className="settings-card">
                <div className="settings-card-title">Generation</div>
                <div className="setting-row">
                  <span>Stream responses</span>
                  <div
                    className={'toggle ' + (useStreaming ? 'on' : '')}
                    onClick={() => setUseStreaming(!useStreaming)}
                  />
                </div>
                <div className="setting-row">
                  <span>Agent mode</span>
                  <div className={'toggle ' + (useAgent ? 'on' : '')} onClick={() => setUseAgent(!useAgent)} />
                </div>
              </div>

              <div className="settings-card">
                <div className="settings-card-title">Memory</div>
                <div className="setting-row">
                  <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                    <Brain size={12} /> Long-term memory
                  </span>
                  <div className={'toggle ' + (useMemory ? 'on' : '')} onClick={() => setUseMemory(!useMemory)} />
                </div>
                {useMemory && (
                  <div className="settings-hint" style={{ marginTop: 8 }}>
                    Extracts facts &amp; preferences from conversations, retrieved via embeddings.
                  </div>
                )}
                <div className="setting-row" style={{ marginTop: 8 }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                    <Network size={12} /> Graph traversal
                  </span>
                  <div className={'toggle ' + (useGraph ? 'on' : '')} onClick={() => setUseGraph(!useGraph)} />
                </div>
                {useGraph && (
                  <div className="settings-hint" style={{ marginTop: 6 }}>
                    Augments vector search with graph pathfinding. Build the graph first in the Graph tab.
                  </div>
                )}
              </div>

              <div className="settings-card">
                <div className="settings-card-title">PageIndex · PDF</div>
                <div className="setting-row">
                  <span>Tree search</span>
                  <div
                    className={'toggle ' + (usePageIndex ? 'on' : '')}
                    onClick={() => setUsePageIndex(!usePageIndex)}
                  />
                </div>
                {usePageIndex && (
                  <>
                    <div className="settings-hint" style={{ marginTop: 8, marginBottom: 8 }}>
                      Local reasoning-based RAG for PDFs. No external API needed.
                    </div>
                    <button
                      className="sl-footer-btn"
                      style={{ width: '100%', justifyContent: 'center', marginBottom: 8 }}
                      onClick={() => setShowPiUpload(true)}
                    >
                      <Upload size={12} /> Upload PDF
                    </button>
                    {piDocs.map((d, i) => (
                      <div
                        key={i}
                        className="file-item"
                        style={{
                          cursor: 'pointer',
                          background: piActiveDoc === d.doc_id ? 'var(--accent-soft)' : undefined,
                          borderRadius: 'var(--radius-sm)',
                          border: piActiveDoc === d.doc_id ? '1px solid var(--border-neon)' : '1px solid transparent',
                        }}
                        onClick={() => setPiActiveDoc(piActiveDoc === d.doc_id ? null : d.doc_id)}
                      >
                        <FileText
                          size={12}
                          style={{ color: piActiveDoc === d.doc_id ? 'var(--neon-cyan)' : 'var(--text-tertiary)' }}
                        />
                        <span
                          style={{ fontSize: 11, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                        >
                          {d.filename}
                        </span>
                        <span
                          className="file-lang"
                          style={{
                            background: d.status === 'completed' ? 'rgba(34,245,160,0.1)' : 'rgba(245,158,11,0.1)',
                            color: d.status === 'completed' ? 'var(--neon-green)' : 'var(--warm)',
                          }}
                        >
                          {d.status === 'completed' ? 'ready' : d.status}
                        </span>
                      </div>
                    ))}
                    {!piActiveDoc && piDocs.length > 0 && (
                      <div className="settings-hint" style={{ color: 'var(--warm)', marginTop: 6 }}>
                        Select a document to query
                      </div>
                    )}
                  </>
                )}
              </div>

              <div className="settings-card">
                <div className="settings-card-title">LLM Backend</div>
                {llmStatus ? (
                  <div
                    style={{
                      fontSize: 11,
                      fontFamily: 'var(--font-mono)',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 5,
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ color: 'var(--text-tertiary)' }}>Backend</span>
                      <span
                        style={{ color: llmStatus.backend === 'ollama' ? 'var(--neon-cyan)' : 'var(--neon-green)' }}
                      >
                        {llmStatus.backend}
                      </span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
                      <span style={{ color: 'var(--text-tertiary)', flexShrink: 0 }}>Model</span>
                      <span
                        style={{
                          color: 'var(--text-secondary)',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          textAlign: 'right',
                        }}
                      >
                        {llmStatus.model}
                      </span>
                    </div>
                    {llmStatus.backend === 'ollama' && (
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ color: 'var(--text-tertiary)' }}>Ollama</span>
                        <span style={{ color: llmStatus.ollama_reachable ? 'var(--neon-green)' : 'var(--warm)' }}>
                          {llmStatus.ollama_reachable ? '\u25CF reachable' : '\u25CB unreachable'}
                        </span>
                      </div>
                    )}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
                      <span style={{ color: 'var(--text-tertiary)', flexShrink: 0 }}>Memory</span>
                      <span
                        style={{
                          color: 'var(--text-secondary)',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          textAlign: 'right',
                        }}
                      >
                        {llmStatus.memory_model}
                      </span>
                    </div>
                  </div>
                ) : (
                  <span style={{ fontSize: 11, color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>
                    {'\u2014'}
                  </span>
                )}
                <div className="settings-hint" style={{ marginTop: 8 }}>
                  Switch model from the chat input bar
                </div>
              </div>

              <div className="settings-card">
                <div className="settings-card-title">System</div>
                <div
                  style={{
                    fontSize: 11,
                    fontFamily: 'var(--font-mono)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 5,
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
                    <span style={{ color: 'var(--text-tertiary)', flexShrink: 0 }}>Embeddings</span>
                    <span
                      style={{
                        color: 'var(--text-secondary)',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                        textAlign: 'right',
                      }}
                    >
                      {stats ? stats.embedding_model : '\u2014'}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ color: 'var(--text-tertiary)' }}>Chunks</span>
                    <span style={{ color: 'var(--text-secondary)' }}>{stats ? stats.document_count : 0}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ color: 'var(--text-tertiary)' }}>BM25 / Vector</span>
                    <span style={{ color: 'var(--text-secondary)' }}>
                      {stats ? stats.bm25_weight : 0.3} / {stats ? stats.vector_weight : 0.7}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </aside>

      {/* ── PDF Viewer Overlay ── */}
      {pdfSource && (
        <Suspense fallback={<PanelFallback />}>
          <PdfViewerPanel source={pdfSource} onClose={() => setPdfSource(null)} />
        </Suspense>
      )}

      {/* ── Modals ── */}
      {showAuth && <AuthModal onClose={() => setShowAuth(false)} onAuth={handleAuth} />}
      <Suspense fallback={null}>
        {showIngest && (
          <IngestModal
            onClose={() => setShowIngest(false)}
            onToast={addToast}
            onRefresh={() => {
              fetchStats();
              setFilesRefreshKey((k) => k + 1);
            }}
            token={token}
          />
        )}
        {showPiUpload && (
          <PiUploadModal
            onClose={() => setShowPiUpload(false)}
            onToast={addToast}
            onDocAdded={(doc) => setPiDocs((p) => [...p.filter((d) => d.doc_id !== doc.doc_id), doc])}
          />
        )}
      </Suspense>
      <Toasts toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}

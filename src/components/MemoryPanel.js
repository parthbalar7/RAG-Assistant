import React, { useState, useEffect, useCallback } from 'react';
import { Search, X, PlusCircle, Sparkles, Trash2, Brain } from 'lucide-react';
import { API } from '../utils/api';
import AddMemoryModal from './AddMemoryModal';

function MemoryPanel({ token, onToast }) {
  const [memories, setMemories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState(null);
  const [searching, setSearching] = useState(false);
  const [showAdd, setShowAdd] = useState(false);
  const [expandedId, setExpandedId] = useState(null);
  const [stats, setStats] = useState({ total: 0, types: {} });
  const [consolidating, setConsolidating] = useState(false);
  const [consolidateResult, setConsolidateResult] = useState(null);

  const fetchMemories = useCallback(async () => {
    setLoading(true);
    try {
      const h = token ? { Authorization: 'Bearer ' + token } : {};
      const r = await fetch(API + '/api/memory', { headers: h });
      if (r.ok) {
        const data = await r.json();
        setMemories(data.fragments || []);
        const types = {};
        (data.fragments || []).forEach((f) => {
          types[f.memory_type] = (types[f.memory_type] || 0) + 1;
        });
        setStats({ total: (data.fragments || []).length, types });
      }
    } catch (e) {
      /* silently fail */
    }
    setLoading(false);
  }, [token]);

  useEffect(() => {
    fetchMemories();
  }, [fetchMemories]);

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      setSearchResults(null);
      return;
    }
    setSearching(true);
    try {
      const h = token ? { Authorization: 'Bearer ' + token } : {};
      const r = await fetch(API + '/api/memory/search?q=' + encodeURIComponent(searchQuery) + '&top_k=8', {
        headers: h,
      });
      if (r.ok) {
        const data = await r.json();
        setSearchResults(data.results || []);
      }
    } catch (e) {
      onToast('error', 'Memory search failed');
    }
    setSearching(false);
  };

  const handleDelete = async (id) => {
    try {
      const h = token ? { Authorization: 'Bearer ' + token } : {};
      await fetch(API + '/api/memory/' + id, { method: 'DELETE', headers: h });
      setMemories((p) => p.filter((m) => m.fragment_id !== id));
      setStats((p) => ({ ...p, total: p.total - 1 }));
      onToast('info', 'Memory deleted');
    } catch (e) {
      onToast('error', 'Delete failed');
    }
  };

  const handleClear = async () => {
    if (!window.confirm('Clear ALL memories? This cannot be undone.')) return;
    try {
      const h = token ? { Authorization: 'Bearer ' + token } : {};
      await fetch(API + '/api/memory', { method: 'DELETE', headers: h });
      setMemories([]);
      setStats({ total: 0, types: {} });
      onToast('info', 'All memories cleared');
    } catch (e) {
      onToast('error', 'Clear failed');
    }
  };

  const handleConsolidate = async () => {
    setConsolidating(true);
    setConsolidateResult(null);
    try {
      const h = { 'Content-Type': 'application/json' };
      if (token) h['Authorization'] = 'Bearer ' + token;
      const r = await fetch(API + '/api/memory/consolidate', { method: 'POST', headers: h });
      const data = await r.json();
      setConsolidateResult(data);
      if (data.merged > 0) {
        onToast('success', data.message);
        fetchMemories();
      } else {
        onToast('info', data.message || 'Nothing to consolidate');
      }
    } catch (e) {
      onToast('error', 'Consolidation failed');
    }
    setConsolidating(false);
  };

  const handleAdd = async (content, memType, importance) => {
    try {
      const h = { 'Content-Type': 'application/json' };
      if (token) h['Authorization'] = 'Bearer ' + token;
      const r = await fetch(API + '/api/memory', {
        method: 'POST',
        headers: h,
        body: JSON.stringify({ content, memory_type: memType, importance }),
      });
      if (r.ok) {
        fetchMemories();
        onToast('success', 'Memory stored');
        setShowAdd(false);
      }
    } catch (e) {
      onToast('error', 'Add failed');
    }
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
  const typeLabel = (t) => (t || '').replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
  const displayList = searchResults !== null ? searchResults : memories;

  if (loading)
    return <div style={{ padding: 16, fontSize: 12, color: 'var(--text-tertiary)' }}>Loading memories...</div>;

  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 14 }}>
        <div className="stat-card" style={{ padding: 10 }}>
          <div className="stat-label">Memories</div>
          <div className="stat-value" style={{ fontSize: 18 }}>
            {stats.total}
          </div>
        </div>
        <div className="stat-card" style={{ padding: 10 }}>
          <div className="stat-label">Types</div>
          <div className="stat-value" style={{ fontSize: 18 }}>
            {Object.keys(stats.types).length}
          </div>
        </div>
      </div>
      {stats.total > 0 && (
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 12 }}>
          {Object.entries(stats.types).map(([type, count]) => {
            const c = typeColors[type] || typeColors.fact;
            return (
              <span
                key={type}
                style={{
                  fontSize: 9,
                  padding: '2px 7px',
                  borderRadius: 10,
                  background: c.bg,
                  color: c.color,
                  border: '1px solid ' + c.border,
                  fontFamily: 'var(--font-mono)',
                }}
              >
                {typeLabel(type)} ({count})
              </span>
            );
          })}
        </div>
      )}
      <div style={{ display: 'flex', gap: 6, marginBottom: 10 }}>
        <div
          style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            background: 'var(--bg-surface)',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border)',
            padding: '0 8px',
          }}
        >
          <Search size={12} style={{ color: 'var(--text-tertiary)', flexShrink: 0 }} />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleSearch();
              if (e.key === 'Escape') {
                setSearchResults(null);
                setSearchQuery('');
              }
            }}
            placeholder="Search memories..."
            style={{
              flex: 1,
              border: 'none',
              background: 'transparent',
              color: 'var(--text-primary)',
              fontSize: 11,
              padding: '7px 6px',
              outline: 'none',
              fontFamily: 'var(--font-body)',
            }}
          />
          {searchResults !== null && (
            <button
              onClick={() => {
                setSearchResults(null);
                setSearchQuery('');
              }}
              style={{
                background: 'none',
                border: 'none',
                color: 'var(--text-tertiary)',
                cursor: 'pointer',
                padding: 2,
              }}
            >
              <X size={10} />
            </button>
          )}
        </div>
        <button
          onClick={handleSearch}
          disabled={searching || !searchQuery.trim()}
          style={{
            padding: '0 10px',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border-neon)',
            background: 'var(--accent-soft)',
            color: 'var(--neon-cyan)',
            cursor: 'pointer',
            fontSize: 11,
          }}
        >
          {searching ? '...' : 'Go'}
        </button>
      </div>
      {searchResults !== null && (
        <div style={{ fontSize: 10, color: 'var(--text-tertiary)', marginBottom: 8, fontFamily: 'var(--font-mono)' }}>
          {searchResults.length} result{searchResults.length !== 1 ? 's' : ''} for "{searchQuery}"
        </div>
      )}
      <div style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
        <button
          className="sl-footer-btn"
          style={{ flex: 1, justifyContent: 'center' }}
          onClick={() => setShowAdd(true)}
        >
          <PlusCircle size={11} /> Add Memory
        </button>
        {stats.total > 0 && (
          <>
            <button
              className="sl-footer-btn"
              style={{ justifyContent: 'center', borderColor: 'var(--border-neon)', color: 'var(--neon-purple)' }}
              onClick={handleConsolidate}
              disabled={consolidating}
              title="Merge semantically related memories into fewer, richer ones"
            >
              <Sparkles size={11} /> {consolidating ? '\u2026' : 'Consolidate'}
            </button>
            <button className="sl-footer-btn danger" style={{ justifyContent: 'center' }} onClick={handleClear}>
              <Trash2 size={11} />
            </button>
          </>
        )}
      </div>
      {consolidateResult && (
        <div
          style={{
            fontSize: 10,
            fontFamily: 'var(--font-mono)',
            color: consolidateResult.merged > 0 ? 'var(--neon-green)' : 'var(--text-tertiary)',
            marginBottom: 10,
            padding: '6px 8px',
            borderRadius: 'var(--radius-sm)',
            background: consolidateResult.merged > 0 ? 'rgba(34,245,160,0.06)' : 'var(--bg-surface)',
            border: '1px solid var(--border)',
          }}
        >
          {consolidateResult.message}
        </div>
      )}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {displayList.length === 0 && (
          <div style={{ textAlign: 'center', padding: 24, color: 'var(--text-tertiary)', fontSize: 12 }}>
            <Brain size={28} style={{ display: 'block', margin: '0 auto 8px', opacity: 0.3 }} />
            {stats.total === 0
              ? 'No memories yet. Chat with the assistant and memories will be extracted automatically.'
              : 'No results found.'}
          </div>
        )}
        {displayList.map((mem, i) => {
          const c = typeColors[mem.memory_type] || typeColors.fact;
          const isExp = expandedId === (mem.fragment_id || i);
          return (
            <div
              key={mem.fragment_id || i}
              className={'source-card' + (isExp ? ' expanded' : '')}
              style={{ cursor: 'pointer' }}
              onClick={() => setExpandedId(isExp ? null : mem.fragment_id || i)}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4, flexWrap: 'wrap' }}>
                <span
                  style={{
                    fontSize: 9,
                    padding: '1px 6px',
                    borderRadius: 8,
                    background: c.bg,
                    color: c.color,
                    border: '1px solid ' + c.border,
                    fontFamily: 'var(--font-mono)',
                  }}
                >
                  {typeLabel(mem.memory_type)}
                </span>
                {mem.similarity !== undefined && (
                  <span style={{ fontSize: 9, color: 'var(--neon-cyan)', fontFamily: 'var(--font-mono)' }}>
                    {(mem.similarity * 100).toFixed(0)}%
                  </span>
                )}
                <span
                  style={{
                    fontSize: 9,
                    color: 'var(--text-tertiary)',
                    fontFamily: 'var(--font-mono)',
                    marginLeft: 'auto',
                  }}
                >
                  {'\u2605'.repeat(Math.round((mem.importance || 0.5) * 5))}
                  {'\u2606'.repeat(5 - Math.round((mem.importance || 0.5) * 5))}
                </span>
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-primary)', lineHeight: 1.55 }}>{mem.content}</div>
              {isExp && (
                <div
                  style={{
                    marginTop: 8,
                    paddingTop: 8,
                    borderTop: '1px solid var(--border)',
                    animation: 'fadeUp 0.2s ease',
                  }}
                >
                  {mem.tags && (typeof mem.tags === 'string' ? JSON.parse(mem.tags || '[]') : mem.tags).length > 0 && (
                    <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 6 }}>
                      {(typeof mem.tags === 'string' ? JSON.parse(mem.tags) : mem.tags).map((tag, j) => (
                        <span
                          key={j}
                          style={{
                            fontSize: 9,
                            padding: '1px 5px',
                            borderRadius: 6,
                            background: 'var(--bg-subtle)',
                            color: 'var(--text-secondary)',
                            fontFamily: 'var(--font-mono)',
                          }}
                        >
                          #{tag}
                        </span>
                      ))}
                    </div>
                  )}
                  {mem.source_query && (
                    <div
                      style={{
                        fontSize: 10,
                        color: 'var(--text-tertiary)',
                        fontFamily: 'var(--font-mono)',
                        marginBottom: 4,
                      }}
                    >
                      From: "{mem.source_query.slice(0, 60)}
                      {mem.source_query.length > 60 ? '...' : ''}"
                    </div>
                  )}
                  {mem.created_at > 0 && (
                    <div
                      style={{
                        fontSize: 10,
                        color: 'var(--text-tertiary)',
                        fontFamily: 'var(--font-mono)',
                        marginBottom: 6,
                      }}
                    >
                      {new Date(mem.created_at * 1000).toLocaleDateString()}{' '}
                      {new Date(mem.created_at * 1000).toLocaleTimeString()}
                    </div>
                  )}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(mem.fragment_id);
                    }}
                    style={{
                      fontSize: 10,
                      padding: '3px 8px',
                      borderRadius: 6,
                      border: '1px solid rgba(239,68,68,0.2)',
                      background: 'rgba(239,68,68,0.06)',
                      color: 'var(--danger)',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 4,
                    }}
                  >
                    <Trash2 size={10} /> Delete
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>
      {showAdd && <AddMemoryModal onClose={() => setShowAdd(false)} onAdd={handleAdd} />}
    </div>
  );
}

export default React.memo(MemoryPanel);

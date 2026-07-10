import React, { useState } from 'react';
import { Search, RefreshCw, CheckCircle2, AlertCircle } from 'lucide-react';
import { WS_URL } from '../utils/api';

function GapPrompt({ gap, query, token, contextRef, onIngested, onRefreshEvent }) {
  const [state, setState] = useState('idle'); // idle | searching | refreshing | done | dismissed | error
  const [result, setResult] = useState(null);
  const [round, setRound] = useState(1);

  if (!gap || state === 'dismissed') return null;

  const approve = () => {
    setState('searching');
    const ws = new WebSocket(WS_URL);
    let ingested = false;
    let finalized = false;
    // The refresh stream in the chat must always be closed out exactly once,
    // even when the socket drops without an explicit done event.
    const finish = () => {
      if (finalized) return;
      finalized = true;
      if (ingested && onRefreshEvent) onRefreshEvent({ type: 'done' }, query);
    };
    ws.onopen = () => {
      // Session + history + retrieval toggles let the backend persist the
      // regenerated answer and verify the gap under the user's real config.
      const ctx = contextRef?.current || {};
      ws.send(
        JSON.stringify({
          type: 'web_search_approved',
          token,
          topic: gap.topic,
          query,
          session_id: ctx.sessionId || null,
          conversation_history: ctx.history || [],
          opts: ctx.opts || {},
        }),
      );
    };
    ws.onmessage = (e) => {
      let ev;
      try {
        ev = JSON.parse(e.data);
      } catch {
        return;
      }
      if (ev.type === 'web_search_started') setState('searching');
      else if (ev.type === 'research_iteration') {
        setRound(ev.iteration || 2);
        if (onRefreshEvent) onRefreshEvent(ev, query);
      } else if (ev.type === 'web_ingested') {
        setResult(ev);
        if (ev.error && !ev.chunks_added && !ingested) {
          // Terminal only when nothing was ever ingested — a failed later
          // round still regenerates from earlier rounds on this socket.
          setState('error');
          ws.close();
        } else if (ev.error && !ev.chunks_added) {
          setState('refreshing');
        } else {
          // The backend auto-regenerates the answer after ingest and streams
          // it on this same socket — no client-side requery.
          ingested = true;
          setState('refreshing');
          if (ev.chunks_added > 0 && onIngested) onIngested(ev);
        }
      } else if (ev.type === 'done') {
        if (ingested) setState('done');
        finish();
        ws.close();
      } else if (ev.type === 'error' && !ingested) {
        setState('error');
        setResult({ error: ev.message || 'Web search failed.' });
      } else if (ingested && onRefreshEvent) {
        // Forward the regenerated answer's stream (sources/tokens/errors) to the chat.
        onRefreshEvent(ev, query);
      }
    };
    ws.onclose = () => {
      setState((s) => (s === 'refreshing' ? 'done' : s));
      finish();
    };
    ws.onerror = () => {
      setState('error');
      setResult({ error: 'Connection failed' });
    };
  };

  return (
    <div className="gap-prompt">
      {state === 'idle' && (
        <>
          <div className="gap-prompt-body">
            <Search size={11} style={{ color: 'var(--neon-cyan)', flexShrink: 0 }} />
            <span>
              <strong>Limited coverage</strong> — {gap.reason} Search the web for <em>"{gap.topic}"</em> and add it to
              your index?
            </span>
          </div>
          <div className="gap-prompt-actions">
            <button className="gap-btn gap-btn-yes" onClick={approve}>
              Search web
            </button>
            <button className="gap-btn gap-btn-no" onClick={() => setState('dismissed')}>
              Dismiss
            </button>
          </div>
        </>
      )}
      {state === 'searching' && (
        <div className="gap-prompt-body">
          <RefreshCw
            size={11}
            style={{ color: 'var(--neon-cyan)', animation: 'spin 1s linear infinite', flexShrink: 0 }}
          />
          <span>
            Searching the web for <em>"{gap.topic}"</em>...{round > 1 ? ` (round ${round})` : ''}
          </span>
        </div>
      )}
      {state === 'refreshing' && result && (
        <div className="gap-prompt-body">
          <RefreshCw
            size={11}
            style={{ color: 'var(--neon-cyan)', animation: 'spin 1s linear infinite', flexShrink: 0 }}
          />
          <span>
            Added <strong>{result.chunks_added} chunks</strong> — refreshing answer from web results...
          </span>
        </div>
      )}
      {state === 'done' && result && (
        <div className="gap-prompt-body">
          <CheckCircle2 size={11} style={{ color: 'var(--neon-green)', flexShrink: 0 }} />
          <span>
            <strong>Answer refreshed from web results</strong> — {result.chunks_added} chunks from{' '}
            {result.urls?.length || 0} source(s).
          </span>
        </div>
      )}
      {state === 'error' && (
        <div className="gap-prompt-body">
          <AlertCircle size={11} style={{ color: 'var(--neon-red, #ff4d4d)', flexShrink: 0 }} />
          <span>{result?.error || 'Web search failed.'}</span>
        </div>
      )}
    </div>
  );
}

export default React.memo(GapPrompt);

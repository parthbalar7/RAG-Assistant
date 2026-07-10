export const API = process.env.REACT_APP_API_URL || '';
export const WS_URL = (process.env.REACT_APP_API_URL || window.location.origin).replace(/^http/, 'ws') + '/api/ws';

/* Helper to extract error message from FastAPI response */
export function extractError(d, fallback) {
  if (!d) return fallback || 'Unknown error';
  if (typeof d.detail === 'string') return d.detail;
  if (Array.isArray(d.detail)) return d.detail.map((e) => e.msg || JSON.stringify(e)).join('; ');
  if (typeof d.detail === 'object') return d.detail.msg || d.detail.message || JSON.stringify(d.detail);
  if (typeof d.message === 'string') return d.message;
  if (typeof d.error === 'string') return d.error;
  return fallback || 'Unknown error';
}

/* ── API helpers ── */
export const api = {
  get: async (path, token) => {
    const h = token ? { Authorization: 'Bearer ' + token } : {};
    const r = await fetch(API + path, { headers: h });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error(extractError(d, r.statusText));
    }
    return r.json();
  },
  post: async (path, body, token) => {
    const h = { 'Content-Type': 'application/json' };
    if (token) h['Authorization'] = 'Bearer ' + token;
    const r = await fetch(API + path, { method: 'POST', headers: h, body: JSON.stringify(body) });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error(extractError(d, r.statusText));
    }
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
  upload: async (files, token) => {
    const f = new FormData();
    files.forEach(function (x) {
      f.append('files', x, x.name);
    });
    const h = token ? { Authorization: 'Bearer ' + token } : {};
    const r = await fetch(API + '/api/upload', { method: 'POST', body: f, headers: h });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error(extractError(d, 'Upload failed'));
    }
    return r.json();
  },
};

export async function* streamQuery(query, history, opts, token, wsRef) {
  const ws = new WebSocket(WS_URL);
  if (wsRef) wsRef.current = ws;
  const evQueue = [];
  let notify = null;
  let wsError = null;

  const enqueue = (item) => {
    evQueue.push(item);
    if (notify) {
      notify();
      notify = null;
    }
  };

  ws.onopen = () => ws.send(JSON.stringify({ token, query_data: { query, conversation_history: history, ...opts } }));
  ws.onmessage = (e) => {
    try {
      enqueue({ value: JSON.parse(e.data) });
    } catch {}
  };
  ws.onclose = () => enqueue({ done: true });
  ws.onerror = () => {
    wsError = new Error('WebSocket connection failed');
    enqueue({ done: true });
  };

  const wait = () =>
    new Promise((r) => {
      if (evQueue.length > 0) r();
      else notify = r;
    });

  try {
    while (true) {
      await wait();
      while (evQueue.length > 0) {
        const item = evQueue.shift();
        if (item.done) {
          if (wsError) throw wsError;
          return;
        }
        yield item.value;
        if (item.value.type === 'done') return;
      }
    }
  } finally {
    if (wsRef) wsRef.current = null;
    if (ws.readyState < 2) ws.close();
  }
}

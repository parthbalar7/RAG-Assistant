import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Network, RefreshCw, ZoomIn, X } from 'lucide-react';
import { api } from '../utils/api';

// Community palette from docs/UI_DESIGN.md §6 — six desaturated hues that read
// on both themes. Nodes without a community fall back to --accent.
const COMMUNITY_PALETTE = ['#7E93B8', '#8FAE93', '#C2AC77', '#B08A9E', '#7FA6A8', '#A08D7B'];

// Canvas colour: needs a concrete value (canvas cannot consume var() strings).
const communityColor = (n, accent) =>
  typeof n.community === 'number' && n.community >= 0
    ? COMMUNITY_PALETTE[n.community % COMMUNITY_PALETTE.length]
    : accent;

// CSS colour for JSX (legend dots, selected-node popover).
const communityColorCss = (n) =>
  typeof n.community === 'number' && n.community >= 0
    ? COMMUNITY_PALETTE[n.community % COMMUNITY_PALETTE.length]
    : 'var(--accent)';

// Resolve theme tokens to concrete values for canvas drawing. Re-read whenever
// data-theme changes (a MutationObserver invalidates the cache).
const readTokens = () => {
  const root = document.documentElement;
  const cs = getComputedStyle(root);
  const get = (name, fallback) => cs.getPropertyValue(name).trim() || fallback;
  return {
    bg: get('--bg', '#f7f6f2'),
    surface: get('--surface', '#ffffff'),
    text2: get('--text-2', '#5d5b54'),
    text3: get('--text-3', '#98958b'),
    accent: get('--accent', '#4e5f7e'),
    fontBody: get('--font-body', 'sans-serif'),
    edgeAlpha: root.dataset.theme === 'dark' ? 0.4 : 0.3,
  };
};

function GraphPanel({ token, onToast, isReady }) {
  const canvasRef = useRef(null);
  const simRef = useRef({ nodes: [], edges: [], alpha: 0, raf: null, tick: null, nodeMap: {} });
  const viewRef = useRef({ x: 0, y: 0, scale: 1 });
  const dragRef = useRef(null);
  const hoveredRef = useRef(null); // tracks hovered node without re-render
  const tokensRef = useRef(null); // resolved theme tokens for canvas drawing
  const [graphData, setGraphData] = useState(null);
  const [stats, setStats] = useState(null);
  const [building, setBuilding] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [docLegend, setDocLegend] = useState([]);

  const fetchStats = useCallback(async () => {
    try {
      const s = await api.get('/api/graph/stats', token);
      setStats(s);
    } catch (_) {}
  }, [token]);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  // Re-resolve canvas tokens and repaint when the theme flips.
  useEffect(() => {
    const obs = new MutationObserver(() => {
      tokensRef.current = null;
      const sim = simRef.current;
      if (!sim.raf && sim.tick) sim.raf = requestAnimationFrame(sim.tick);
    });
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => obs.disconnect();
  }, []);

  const handleBuild = async () => {
    if (!isReady) {
      onToast('error', 'Index documents first');
      return;
    }
    setBuilding(true);
    try {
      const r = await api.post('/api/graph/build', {}, token);
      onToast('success', `Graph built: ${r.nodes} nodes, ${r.edges} edges in ${(r.ms / 1000).toFixed(1)}s`);
      fetchStats();
      handleLoad();
    } catch (e) {
      onToast('error', e.message);
    } finally {
      setBuilding(false);
    }
  };

  const handleLoad = async () => {
    try {
      const d = await api.get('/api/graph?max_nodes=200', token);
      setGraphData(d);
      setStats(d.stats);
    } catch (e) {
      onToast('error', e.message);
    }
  };

  // ── Force simulation ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!graphData || !graphData.nodes.length) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Size canvas to its CSS-rendered dimensions
    const rect = canvas.getBoundingClientRect();
    const W = rect.width || 400;
    const H = rect.height || 500;
    canvas.width = W;
    canvas.height = H;

    // Legend: Louvain communities (themes) when present
    const commCounts = {};
    graphData.nodes.forEach((n) => {
      if (typeof n.community === 'number' && n.community >= 0)
        commCounts[n.community] = (commCounts[n.community] || 0) + 1;
    });
    if (Object.keys(commCounts).length > 0) {
      setDocLegend(
        Object.entries(commCounts)
          .sort((a, b) => b[1] - a[1])
          .slice(0, COMMUNITY_PALETTE.length)
          .map(([cid, count]) => ({
            label: `theme ${Number(cid) + 1} (${count})`,
            color: COMMUNITY_PALETTE[Number(cid) % COMMUNITY_PALETTE.length],
          })),
      );
    } else {
      setDocLegend([]); // no communities → all nodes accent, nothing to distinguish
    }

    // Labels are reserved for top-degree nodes (roughly the ten best-connected).
    const degrees = graphData.nodes.map((n) => n.degree || 0).sort((a, b) => b - a);
    const labelThreshold = Math.max(2, degrees[Math.min(9, degrees.length - 1)] || 0);

    const MARGIN = 60;
    const nodeMap = {};
    const simNodes = graphData.nodes.map((n, i) => {
      // Grid-based start positions — avoids circular corner clustering
      const cols = Math.ceil(Math.sqrt(graphData.nodes.length * (W / H)));
      const col = i % cols;
      const row = Math.floor(i / cols);
      const cellW = (W - MARGIN * 2) / cols;
      const cellH = (H - MARGIN * 2) / Math.ceil(graphData.nodes.length / cols);
      const node = {
        ...n,
        x: MARGIN + col * cellW + cellW / 2 + (Math.random() - 0.5) * cellW * 0.5,
        y: MARGIN + row * cellH + cellH / 2 + (Math.random() - 0.5) * cellH * 0.5,
        vx: 0,
        vy: 0,
        radius: Math.max(4, Math.min(11, 4 + n.degree)), // 4–11px by degree
      };
      nodeMap[n.id] = node;
      return node;
    });

    const simEdges = graphData.edges
      .map((e) => ({
        ...e,
        source: nodeMap[e.source],
        target: nodeMap[e.target],
      }))
      .filter((e) => e.source && e.target);

    const sim = simRef.current;
    sim.nodes = simNodes;
    sim.edges = simEdges;
    sim.alpha = 1.0;
    sim.nodeMap = nodeMap;
    if (sim.raf) cancelAnimationFrame(sim.raf);

    const ctx = canvas.getContext('2d');
    const IDEAL_DIST = 90;

    const tick = () => {
      sim.alpha *= 0.975;
      if (sim.alpha < 0.002) sim.alpha = 0;

      if (sim.alpha > 0) {
        const a = sim.alpha;
        const cx = W / 2,
          cy = H / 2;

        for (let i = 0; i < simNodes.length; i++) {
          const ni = simNodes[i];

          // Node-node repulsion
          for (let j = i + 1; j < simNodes.length; j++) {
            const nj = simNodes[j];
            const dx = nj.x - ni.x,
              dy = nj.y - ni.y;
            const dist2 = dx * dx + dy * dy || 0.01;
            const dist = Math.sqrt(dist2);
            const force = (1600 / dist2) * a;
            const fx = (dx / dist) * force,
              fy = (dy / dist) * force;
            ni.vx -= fx;
            ni.vy -= fy;
            nj.vx += fx;
            nj.vy += fy;
          }

          // Soft boundary repulsion
          const bForce = 6 * a;
          if (ni.x < MARGIN) ni.vx += (bForce * (MARGIN - ni.x)) / MARGIN;
          if (ni.x > W - MARGIN) ni.vx -= (bForce * (ni.x - (W - MARGIN))) / MARGIN;
          if (ni.y < MARGIN) ni.vy += (bForce * (MARGIN - ni.y)) / MARGIN;
          if (ni.y > H - MARGIN) ni.vy -= (bForce * (ni.y - (H - MARGIN))) / MARGIN;

          // Weak center gravity
          ni.vx += (cx - ni.x) * 0.008 * a;
          ni.vy += (cy - ni.y) * 0.008 * a;
        }

        // Edge springs
        for (const e of simEdges) {
          const dx = e.target.x - e.source.x,
            dy = e.target.y - e.source.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 0.1;
          const force = (dist - IDEAL_DIST) * 0.05 * a;
          const fx = (dx / dist) * force,
            fy = (dy / dist) * force;
          e.source.vx += fx;
          e.source.vy += fy;
          e.target.vx -= fx;
          e.target.vy -= fy;
        }

        // Integrate + soft clamp
        for (const n of simNodes) {
          n.vx *= 0.82;
          n.vy *= 0.82;
          n.x += n.vx;
          n.y += n.vy;
          n.x = Math.max(n.radius + 2, Math.min(W - n.radius - 2, n.x));
          n.y = Math.max(n.radius + 2, Math.min(H - n.radius - 2, n.y));
        }
      }

      // ── Draw ────────────────────────────────────────────────────────────────
      const T = tokensRef.current || (tokensRef.current = readTokens());
      const { x: ox, y: oy, scale } = viewRef.current;
      const hov = hoveredRef.current;
      const sel = simRef.current._selected;
      const focus = hov || sel;
      const focusId = focus ? focus.id : null;

      ctx.fillStyle = T.bg;
      ctx.fillRect(0, 0, W, H);
      ctx.save();
      ctx.translate(ox, oy);
      ctx.scale(scale, scale);

      // Edges: 1px hairlines, text-3 at 30% (40% dark); the focused node's
      // edges draw in full accent, the rest dim to 15%.
      ctx.lineWidth = 1;
      for (const e of simEdges) {
        const connected = focus && (e.source.id === focusId || e.target.id === focusId);
        ctx.strokeStyle = connected ? T.accent : T.text3;
        ctx.globalAlpha = connected ? 1 : focus ? 0.15 : T.edgeAlpha;
        ctx.beginPath();
        ctx.moveTo(e.source.x, e.source.y);
        ctx.lineTo(e.target.x, e.target.y);
        ctx.stroke();

        // Relation label on the focused node's edges only
        if (connected && e.rel) {
          const mx = (e.source.x + e.target.x) / 2;
          const my = (e.source.y + e.target.y) / 2;
          const relLabel = e.rel.replace(/_/g, ' ');
          ctx.font = `10px ${T.fontBody}`;
          const tw = ctx.measureText(relLabel).width;
          ctx.fillStyle = T.bg;
          ctx.fillRect(mx - tw / 2 - 3, my - 8, tw + 6, 14);
          ctx.fillStyle = T.text2;
          ctx.fillText(relLabel, mx - tw / 2, my + 3);
        }
      }
      ctx.globalAlpha = 1;

      // Nodes: filled circles, 1.5px surface ring; selection/hover = 2px accent ring
      for (const n of simNodes) {
        const color = communityColor(n, T.accent);
        const isFocus = focus && n.id === focusId;
        ctx.globalAlpha = focus && !isFocus ? 0.15 : 1;

        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = T.surface;
        ctx.lineWidth = 1.5;
        ctx.stroke();

        if (isFocus) {
          ctx.beginPath();
          ctx.arc(n.x, n.y, n.radius + 2.5, 0, Math.PI * 2);
          ctx.strokeStyle = T.accent;
          ctx.lineWidth = 2;
          ctx.stroke();
        }

        // Labels: top-degree nodes and the hovered/selected node only
        if (isFocus || n.degree >= labelThreshold) {
          const label = n.label.length > 24 ? n.label.slice(0, 23) + '…' : n.label;
          ctx.font = `11px ${T.fontBody}`;
          ctx.fillStyle = T.text2;
          ctx.fillText(label, n.x + n.radius + 6, n.y + 4);
        }
      }
      ctx.globalAlpha = 1;

      ctx.restore();
      if (sim.alpha > 0 || hoveredRef.current || dragRef.current) {
        sim.raf = requestAnimationFrame(tick);
      } else {
        sim.raf = null;
      }
    };

    sim.tick = tick;
    sim.raf = requestAnimationFrame(tick);
    return () => {
      if (sim.raf) cancelAnimationFrame(sim.raf);
      sim.raf = null;
      sim.tick = null;
    };
  }, [graphData]); // eslint-disable-line

  // ── Mouse helpers ──────────────────────────────────────────────────────────
  const restartLoop = () => {
    const sim = simRef.current;
    if (!sim.raf && sim.tick) sim.raf = requestAnimationFrame(sim.tick);
  };

  const getCanvasPos = (e) => {
    const r = canvasRef.current.getBoundingClientRect();
    const { x: ox, y: oy, scale } = viewRef.current;
    return { x: (e.clientX - r.left - ox) / scale, y: (e.clientY - r.top - oy) / scale };
  };

  const findNodeAt = ({ x, y }) => {
    for (const n of simRef.current.nodes) if (Math.hypot(n.x - x, n.y - y) < n.radius + 8) return n;
    return null;
  };

  const onMouseDown = (e) => {
    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      ox: viewRef.current.x,
      oy: viewRef.current.y,
      moved: false,
    };
  };

  const onMouseMove = (e) => {
    const pos = getCanvasPos(e);
    hoveredRef.current = findNodeAt(pos);
    canvasRef.current.style.cursor = hoveredRef.current ? 'pointer' : dragRef.current ? 'grabbing' : 'grab';
    restartLoop();
    if (!dragRef.current) return;
    const dx = e.clientX - dragRef.current.startX;
    const dy = e.clientY - dragRef.current.startY;
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) dragRef.current.moved = true;
    viewRef.current.x = dragRef.current.ox + dx;
    viewRef.current.y = dragRef.current.oy + dy;
    restartLoop();
  };

  const onMouseLeave = () => {
    hoveredRef.current = null;
    restartLoop();
  };

  const onMouseUp = (e) => {
    if (!dragRef.current) return;
    if (!dragRef.current.moved) {
      const hit = findNodeAt(getCanvasPos(e));
      simRef.current._selected = hit;
      setSelectedNode(hit);
    }
    dragRef.current = null;
    restartLoop();
  };

  const onWheel = (e) => {
    e.preventDefault();
    viewRef.current.scale = Math.max(0.2, Math.min(4, viewRef.current.scale * (e.deltaY < 0 ? 1.12 : 0.9)));
    restartLoop();
  };

  const resetView = () => {
    viewRef.current = { x: 0, y: 0, scale: 1 };
    restartLoop();
  };

  const hasGraph = stats && stats.nodes > 0;
  const chipBtnStyle = { height: 24, padding: '0 8px', fontSize: 11, gap: 5 };

  return (
    <div className="panel-content" style={{ display: 'flex', flexDirection: 'column', gap: 8, height: '100%' }}>
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <Network size={13} style={{ color: 'var(--accent)' }} />
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-1)' }}>Knowledge Graph</span>
        {hasGraph && (
          <span style={{ fontSize: 11, color: 'var(--text-3)', marginLeft: 'auto' }}>
            {stats.nodes} nodes · {stats.edges} edges · {stats.documents} docs
          </span>
        )}
      </div>

      {graphData && graphData.nodes.length > 0 ? (
        <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div style={{ flex: 1, position: 'relative', minHeight: 0 }}>
            <canvas
              ref={canvasRef}
              style={{
                width: '100%',
                height: '100%',
                display: 'block',
                background: 'var(--bg)',
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border)',
              }}
              onMouseDown={onMouseDown}
              onMouseMove={onMouseMove}
              onMouseUp={onMouseUp}
              onMouseLeave={onMouseLeave}
              onWheel={onWheel}
            />

            {/* Controls — compact ghost buttons in a top-right surface chip row */}
            <div
              style={{
                position: 'absolute',
                top: 8,
                right: 8,
                display: 'flex',
                gap: 2,
                padding: 2,
                background: 'var(--surface)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-md)',
              }}
            >
              <button className="sl-footer-btn" style={chipBtnStyle} onClick={handleBuild} disabled={building}>
                <RefreshCw size={11} style={building ? { animation: 'spin 0.8s linear infinite' } : undefined} />
                {building ? 'Building…' : 'Rebuild'}
              </button>
              <button className="sl-footer-btn" style={chipBtnStyle} onClick={resetView} title="Reset pan/zoom">
                <ZoomIn size={11} /> Fit
              </button>
            </div>

            {/* Selected node — popover */}
            {selectedNode && (
              <div
                style={{
                  position: 'absolute',
                  left: 8,
                  right: 8,
                  bottom: 8,
                  background: 'var(--surface)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-md)',
                  boxShadow: 'var(--shadow-pop)',
                  padding: '8px 12px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 3 }}>
                  <span
                    style={{
                      width: 9,
                      height: 9,
                      borderRadius: '50%',
                      background: communityColorCss(selectedNode),
                      flexShrink: 0,
                    }}
                  />
                  <span
                    style={{
                      fontSize: 12,
                      fontWeight: 600,
                      color: 'var(--text-1)',
                      flex: 1,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {selectedNode.label}
                  </span>
                  {selectedNode.type && (
                    <span
                      style={{
                        fontSize: 11,
                        color: 'var(--text-3)',
                        background: 'var(--surface-2)',
                        padding: '1px 6px',
                        borderRadius: 'var(--radius-sm)',
                        flexShrink: 0,
                      }}
                    >
                      {selectedNode.type}
                    </span>
                  )}
                  <button
                    style={{
                      background: 'none',
                      border: 'none',
                      cursor: 'pointer',
                      color: 'var(--text-3)',
                      padding: 0,
                      flexShrink: 0,
                      display: 'flex',
                    }}
                    onClick={() => {
                      simRef.current._selected = null;
                      setSelectedNode(null);
                    }}
                  >
                    <X size={12} />
                  </button>
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-3)' }}>
                  {selectedNode.degree} connection{selectedNode.degree !== 1 ? 's' : ''} · {selectedNode.chunk_count}{' '}
                  chunk{selectedNode.chunk_count !== 1 ? 's' : ''}
                </div>
                {selectedNode.doc && (
                  <div
                    style={{
                      fontSize: 11,
                      color: 'var(--text-3)',
                      marginTop: 3,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {selectedNode.doc}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Community legend */}
          {docLegend.length > 1 && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 10px', padding: '0 2px' }}>
              {docLegend.map(({ label, color }) => (
                <span
                  key={label}
                  style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, color: 'var(--text-3)' }}
                >
                  <span style={{ width: 8, height: 8, borderRadius: '50%', background: color, flexShrink: 0 }} />
                  {label}
                </span>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div
          style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 8,
            textAlign: 'center',
            padding: 16,
          }}
        >
          <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-2)' }}>
            {hasGraph ? 'Graph ready' : 'No knowledge graph yet'}
          </div>
          <div style={{ fontSize: 13, color: 'var(--text-3)' }}>
            Maps concepts, functions and relationships
            <br />
            across your indexed documents
          </div>
          <button
            className="modal-btn cancel"
            style={{ marginTop: 8 }}
            onClick={hasGraph ? handleLoad : handleBuild}
            disabled={building}
          >
            {building ? (
              <>
                <RefreshCw size={13} style={{ animation: 'spin 0.8s linear infinite' }} />
                Building{'…'}
              </>
            ) : hasGraph ? (
              'View graph'
            ) : (
              'Build graph'
            )}
          </button>
        </div>
      )}
    </div>
  );
}

export default React.memo(GraphPanel);

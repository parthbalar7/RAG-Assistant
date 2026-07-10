import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Network, RefreshCw, ZoomIn, X } from 'lucide-react';
import { api } from '../utils/api';

const GRAPH_COLORS = [
  '#00f0ff',
  '#a855f7',
  '#f59e0b',
  '#34d399',
  '#f87171',
  '#60a5fa',
  '#fb923c',
  '#e879f9',
  '#a3e635',
  '#38bdf8',
  '#f472b6',
  '#4ade80',
];

// Colour by Louvain community when the backend provides one (community >= 0);
// fall back to the per-document colour for graphs built before communities.
const nodeColor = (n) =>
  GRAPH_COLORS[(typeof n.community === 'number' && n.community >= 0 ? n.community : n.color_idx) % GRAPH_COLORS.length];

function GraphPanel({ token, onToast, isReady }) {
  const canvasRef = useRef(null);
  const simRef = useRef({ nodes: [], edges: [], alpha: 0, raf: null, tick: null, nodeMap: {} });
  const viewRef = useRef({ x: 0, y: 0, scale: 1 });
  const dragRef = useRef(null);
  const hoveredRef = useRef(null); // tracks hovered node without re-render
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

    // Legend: Louvain communities (themes) when present, else document -> color
    const commCounts = {};
    graphData.nodes.forEach((n) => {
      if (typeof n.community === 'number' && n.community >= 0)
        commCounts[n.community] = (commCounts[n.community] || 0) + 1;
    });
    if (Object.keys(commCounts).length > 0) {
      setDocLegend(
        Object.entries(commCounts)
          .sort((a, b) => b[1] - a[1])
          .slice(0, GRAPH_COLORS.length)
          .map(([cid, count]) => ({
            label: `theme ${Number(cid) + 1} (${count})`,
            color: GRAPH_COLORS[Number(cid) % GRAPH_COLORS.length],
          })),
      );
    } else {
      const docMap = {};
      graphData.nodes.forEach((n) => {
        if (n.doc && !(n.doc in docMap)) docMap[n.doc] = n.color_idx;
      });
      setDocLegend(
        Object.entries(docMap).map(([doc, idx]) => ({
          label: doc.split(/[\\/]/).pop(), // just filename
          color: GRAPH_COLORS[idx % GRAPH_COLORS.length],
        })),
      );
    }

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
        radius: Math.max(5, Math.min(14, 5 + n.degree * 1.2)),
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
      const { x: ox, y: oy, scale } = viewRef.current;
      const hov = hoveredRef.current;
      const sel = simRef.current._selected;

      ctx.clearRect(0, 0, W, H);
      ctx.save();
      ctx.translate(ox, oy);
      ctx.scale(scale, scale);

      // Edges
      for (const e of simEdges) {
        const connected = hov && (e.source === hov || e.target === hov);
        const baseAlpha = connected ? 0.9 : hov ? 0.08 : 0.35;
        ctx.strokeStyle = connected ? `rgba(255,255,255,${baseAlpha})` : `rgba(120,130,150,${baseAlpha})`;
        ctx.lineWidth = connected ? 1.5 : 0.8;
        ctx.beginPath();
        ctx.moveTo(e.source.x, e.source.y);
        ctx.lineTo(e.target.x, e.target.y);
        ctx.stroke();

        // Relation label on connected edges only
        if (connected && e.rel) {
          const mx = (e.source.x + e.target.x) / 2;
          const my = (e.source.y + e.target.y) / 2;
          const relLabel = e.rel.replace(/_/g, ' ');
          ctx.font = '9px monospace';
          const tw = ctx.measureText(relLabel).width;
          ctx.fillStyle = 'rgba(10,14,24,0.85)';
          ctx.fillRect(mx - tw / 2 - 3, my - 8, tw + 6, 13);
          ctx.fillStyle = 'rgba(200,210,230,0.9)';
          ctx.fillText(relLabel, mx - tw / 2, my + 2);
        }
      }

      // Nodes
      for (const n of simNodes) {
        const color = nodeColor(n);
        const isHov = hov === n;
        const isSel = sel && sel.id === n.id;
        const dimmed = hov && !isHov;

        ctx.globalAlpha = dimmed ? 0.25 : 1;

        // Outer glow ring for hovered/selected
        if (isHov || isSel) {
          ctx.beginPath();
          ctx.arc(n.x, n.y, n.radius + 5, 0, Math.PI * 2);
          ctx.strokeStyle = color + '55';
          ctx.lineWidth = 3;
          ctx.stroke();
        }

        // Node fill
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
        ctx.fillStyle = color + (isHov || isSel ? 'ee' : '88');
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = isHov || isSel ? 2 : 1;
        ctx.stroke();

        ctx.globalAlpha = 1;

        // Label: always show for hovered/selected, or degree >= 2
        const showLabel = isHov || isSel || (n.degree >= 2 && !dimmed);
        if (showLabel) {
          const label = n.label.length > 22 ? n.label.slice(0, 21) + '\u2026' : n.label;
          const fontSize = isHov || isSel ? 11 : Math.max(9, Math.min(10, 8 + n.degree * 0.4));
          ctx.font = `${isHov || isSel ? 'bold ' : ''}${fontSize}px monospace`;
          const tw = ctx.measureText(label).width;
          const lx = n.x + n.radius + 5;
          const ly = n.y + 4;

          // Label background pill
          ctx.fillStyle = 'rgba(8, 12, 22, 0.88)';
          ctx.beginPath();
          const pad = 3;
          ctx.roundRect
            ? ctx.roundRect(lx - pad, ly - fontSize, tw + pad * 2, fontSize + 4, 3)
            : ctx.rect(lx - pad, ly - fontSize, tw + pad * 2, fontSize + 4);
          ctx.fill();

          // Label border
          ctx.strokeStyle = color + '44';
          ctx.lineWidth = 0.5;
          ctx.stroke();

          // Label text
          ctx.fillStyle = isHov || isSel ? color : 'rgba(210,220,235,0.95)';
          ctx.fillText(label, lx, ly);
        }

        // Type badge for hovered/selected
        if ((isHov || isSel) && n.type) {
          ctx.font = '8px monospace';
          const badge = n.type;
          const bw = ctx.measureText(badge).width;
          ctx.fillStyle = 'rgba(8,12,22,0.9)';
          ctx.fillRect(n.x - bw / 2 - 4, n.y + n.radius + 3, bw + 8, 12);
          ctx.fillStyle = color + 'cc';
          ctx.fillText(badge, n.x - bw / 2, n.y + n.radius + 12);
        }
      }

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

  return (
    <div className="panel-content" style={{ display: 'flex', flexDirection: 'column', gap: 8, height: '100%' }}>
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <Network size={13} style={{ color: 'var(--neon-cyan)' }} />
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>Knowledge Graph</span>
        {stats && stats.nodes > 0 && (
          <span
            style={{ fontSize: 9, color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)', marginLeft: 'auto' }}
          >
            {stats.nodes}N · {stats.edges}E · {stats.documents}D
          </span>
        )}
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', gap: 6 }}>
        <button
          className="sl-footer-btn"
          style={{
            flex: 1,
            justifyContent: 'center',
            color: building ? 'var(--text-tertiary)' : 'var(--neon-cyan)',
            borderColor: 'var(--border-neon)',
          }}
          onClick={handleBuild}
          disabled={building}
        >
          <RefreshCw size={11} style={{ animation: building ? 'spin 1s linear infinite' : 'none' }} />
          {building ? 'Building\u2026' : stats && stats.nodes > 0 ? 'Rebuild' : 'Build Graph'}
        </button>
        {stats && stats.nodes > 0 && !graphData && (
          <button className="sl-footer-btn" style={{ flex: 1, justifyContent: 'center' }} onClick={handleLoad}>
            <Network size={11} /> View
          </button>
        )}
        {graphData && (
          <button className="sl-footer-btn" onClick={resetView} title="Reset pan/zoom">
            <ZoomIn size={11} />
          </button>
        )}
      </div>

      {/* Canvas */}
      {graphData && graphData.nodes.length > 0 ? (
        <div style={{ flex: 1, position: 'relative', minHeight: 0, display: 'flex', flexDirection: 'column', gap: 6 }}>
          <canvas
            ref={canvasRef}
            style={{
              flex: 1,
              width: '100%',
              display: 'block',
              background: '#080c16',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border)',
              minHeight: 0,
            }}
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={onMouseUp}
            onMouseLeave={onMouseLeave}
            onWheel={onWheel}
          />

          {/* Document colour legend */}
          {docLegend.length > 1 && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 10px', padding: '4px 2px' }}>
              {docLegend.map(({ label, color }) => (
                <span
                  key={label}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 4,
                    fontSize: 9,
                    color: 'var(--text-tertiary)',
                    fontFamily: 'var(--font-mono)',
                  }}
                >
                  <span style={{ width: 8, height: 8, borderRadius: '50%', background: color, flexShrink: 0 }} />
                  {label.length > 24 ? '\u2026' + label.slice(-22) : label}
                </span>
              ))}
            </div>
          )}

          {/* Selected node card */}
          {selectedNode && (
            <div
              style={{
                background: 'var(--bg-surface)',
                border: `1px solid ${nodeColor(selectedNode)}55`,
                borderRadius: 'var(--radius-md)',
                padding: '8px 12px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 3 }}>
                <span
                  style={{
                    width: 9,
                    height: 9,
                    borderRadius: '50%',
                    background: nodeColor(selectedNode),
                    flexShrink: 0,
                  }}
                />
                <span
                  style={{
                    fontSize: 12,
                    fontWeight: 600,
                    color: 'var(--text-primary)',
                    fontFamily: 'var(--font-mono)',
                    flex: 1,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {selectedNode.label}
                </span>
                <span
                  style={{
                    fontSize: 9,
                    color: 'var(--text-tertiary)',
                    background: 'var(--glass)',
                    padding: '1px 6px',
                    borderRadius: 6,
                    flexShrink: 0,
                  }}
                >
                  {selectedNode.type}
                </span>
                <button
                  style={{
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    color: 'var(--text-tertiary)',
                    padding: 0,
                    flexShrink: 0,
                  }}
                  onClick={() => {
                    simRef.current._selected = null;
                    setSelectedNode(null);
                  }}
                >
                  <X size={12} />
                </button>
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>
                {selectedNode.degree} connection{selectedNode.degree !== 1 ? 's' : ''} · {selectedNode.chunk_count}{' '}
                chunk{selectedNode.chunk_count !== 1 ? 's' : ''}
              </div>
              {selectedNode.doc && (
                <div
                  style={{
                    fontSize: 9,
                    color: 'var(--text-tertiary)',
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
      ) : (
        <div
          style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 8,
            color: 'var(--text-tertiary)',
            fontSize: 12,
            textAlign: 'center',
          }}
        >
          <Network size={32} style={{ opacity: 0.2 }} />
          <div>
            Click <strong>Build Graph</strong> to extract entities
          </div>
          <div style={{ fontSize: 10, opacity: 0.7 }}>
            Maps concepts, functions &amp; relationships
            <br />
            across your indexed documents
          </div>
        </div>
      )}
    </div>
  );
}

export default React.memo(GraphPanel);

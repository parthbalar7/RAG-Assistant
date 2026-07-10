import React, { useState, useEffect } from 'react';
import { Sparkles, AlertCircle, CheckCircle2 } from 'lucide-react';
import { api } from '../utils/api';

function IntegrityRadarPanel({ token, addToast, isReady }) {
  const [scan, setScan] = useState(null);
  const [history, setHistory] = useState([]); // eslint-disable-line no-unused-vars
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

      addToast('success', `Integrity scan complete \u00B7 Health score: ${res.health?.score ?? '\u2014'}`);
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
          <div className="radar-sub">Detect contradictions, blind spots, resilience gaps, and documentation drift.</div>
        </div>
        <button
          className={'radar-scan-btn ' + (running ? 'loading' : '')}
          onClick={runScan}
          disabled={running || !isReady}
        >
          <Sparkles size={14} /> {running ? 'Scanning\u2026' : 'Run scan'}
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
                <div className="gauge-ring" style={score == null ? {} : { '--p': score }} />
                <div className="gauge-center">
                  <div className="gauge-score">{score == null ? '\u2014' : score}</div>
                  <div className="gauge-band">{band || '\u2014'}</div>
                </div>
              </div>
              <div className="radar-meta">
                <div>
                  <span className="k">Sampled</span>
                  <span className="v">{scan?.sampled_chunks ?? '\u2014'}</span>
                </div>
                <div>
                  <span className="k">Total</span>
                  <span className="v">{scan?.total_chunks ?? '\u2014'}</span>
                </div>
                <div>
                  <span className="k">Time</span>
                  <span className="v">{scan?.duration_ms ? `${scan.duration_ms}ms` : '\u2014'}</span>
                </div>
              </div>
            </div>

            <div className="radar-card">
              <div className="radar-card-title">Signals</div>
              <div className="radar-signals">
                <div className="sig">
                  <span>Contradictions</span>
                  <b>{counts.contradiction || 0}</b>
                </div>
                <div className="sig">
                  <span>Blind spots</span>
                  <b>{counts.blind_spot || 0}</b>
                </div>
                <div className="sig">
                  <span>Resilience</span>
                  <b>{counts.resilience_gap || 0}</b>
                </div>
                <div className="sig">
                  <span>Drift</span>
                  <b>{counts.drift || 0}</b>
                </div>
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

export default React.memo(IntegrityRadarPanel);

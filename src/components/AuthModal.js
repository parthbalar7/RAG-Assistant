import React, { useState } from 'react';
import { AlertCircle } from 'lucide-react';
import { api } from '../utils/api';

function AuthModal({ onClose, onAuth, required }) {
  const [tab, setTab] = useState('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    setError('');
    setLoading(true);
    try {
      const endpoint = tab === 'login' ? '/api/auth/login' : '/api/auth/register';
      const body =
        tab === 'login' ? { username, password } : { username, password, display_name: displayName || username };
      const res = await api.post(endpoint, body);
      onAuth(res.token, res.user);
      onClose();
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  };

  return (
    <div className="modal-overlay" onClick={required ? undefined : onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()} style={{ width: 400 }}>
        <h2>{tab === 'login' ? 'Welcome Back' : 'Create Account'}</h2>
        <p>
          {required
            ? 'Sign in or register to use RAG Assistant.'
            : 'Sign in to save your chat history and preferences.'}
        </p>
        <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
          <button
            className={'modal-btn ' + (tab === 'login' ? 'confirm' : 'cancel')}
            style={{ flex: 1 }}
            onClick={() => setTab('login')}
          >
            Login
          </button>
          <button
            className={'modal-btn ' + (tab === 'register' ? 'confirm' : 'cancel')}
            style={{ flex: 1 }}
            onClick={() => setTab('register')}
          >
            Register
          </button>
        </div>
        <input type="text" placeholder="Username" value={username} onChange={(e) => setUsername(e.target.value)} />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') submit();
          }}
        />
        {tab === 'register' && (
          <input
            type="text"
            placeholder="Display name (optional)"
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
          />
        )}
        {error && (
          <div className="result-banner error">
            <AlertCircle size={12} /> {error}
          </div>
        )}
        <div className="modal-actions">
          {!required && (
            <button className="modal-btn cancel" onClick={onClose}>
              Cancel
            </button>
          )}
          <button className="modal-btn confirm" onClick={submit} disabled={loading || !username || !password}>
            {loading ? '...' : tab === 'login' ? 'Login' : 'Register'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default React.memo(AuthModal);

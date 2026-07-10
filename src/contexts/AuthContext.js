import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { api } from '../utils/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => {
    try {
      return localStorage.getItem('rag-token');
    } catch {
      return null;
    }
  });
  const [user, setUser] = useState(null);
  const [showAuth, setShowAuth] = useState(false);

  useEffect(() => {
    if (token) {
      api
        .get('/api/auth/me', token)
        .then((d) => {
          if (d.user) setUser(d.user);
          else {
            setToken(null);
            try {
              localStorage.removeItem('rag-token');
            } catch {}
          }
        })
        .catch(() => {});
    }
  }, [token]);

  const handleAuth = useCallback((t, u) => {
    setToken(t);
    setUser(u);
    try {
      localStorage.setItem('rag-token', t);
    } catch {}
  }, []);

  const handleLogout = useCallback(() => {
    setToken(null);
    setUser(null);
    try {
      localStorage.removeItem('rag-token');
    } catch {}
  }, []);

  const value = { token, user, showAuth, setShowAuth, handleAuth, handleLogout };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}

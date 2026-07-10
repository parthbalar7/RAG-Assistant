import React, { createContext, useContext, useState } from 'react';

const SettingsContext = createContext(null);

export function SettingsProvider({ children }) {
  const [useReranking, setUseReranking] = useState(true);
  const [useStreaming, setUseStreaming] = useState(true);
  const [useHybrid, setUseHybrid] = useState(true);
  const [useRouting, setUseRouting] = useState(true);
  const [useAgent, setUseAgent] = useState(false);
  const [usePageIndex, setUsePageIndex] = useState(false);
  const [useMemory, setUseMemory] = useState(true);
  const [useGraph, setUseGraph] = useState(false);
  const [useHyDE, setUseHyDE] = useState(false);
  const [useSplade, setUseSplade] = useState(false);
  const [useMultiQuery, setUseMultiQuery] = useState(false);
  const [useParentExpand, setUseParentExpand] = useState(false);

  const value = {
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
  };

  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

export function useSettings() {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error('useSettings must be used within SettingsProvider');
  return ctx;
}

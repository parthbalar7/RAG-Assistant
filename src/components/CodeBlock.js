import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy, Check } from 'lucide-react';

function CodeBlock({ language, children }) {
  const [copied, setCopied] = useState(false);
  const code = String(children).replace(/\n$/, '');
  return (
    <div className="code-block">
      <div className="code-header">
        <span className="code-lang">{language || 'text'}</span>
        <button
          className="code-copy"
          onClick={() => {
            navigator.clipboard.writeText(code);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
          }}
        >
          {copied ? (
            <>
              <Check size={10} /> copied
            </>
          ) : (
            <>
              <Copy size={10} /> copy
            </>
          )}
        </button>
      </div>
      <SyntaxHighlighter
        language={language || 'text'}
        style={oneDark}
        customStyle={{
          margin: 0,
          padding: '14px',
          background: 'var(--surface-2)',
          fontSize: '11.5px',
          lineHeight: '1.65',
        }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}

export default React.memo(CodeBlock);

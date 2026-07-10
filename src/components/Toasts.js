import React from 'react';
import { CheckCircle2, AlertCircle, Info, X } from 'lucide-react';

function Toasts({ toasts, onDismiss }) {
  return (
    <div className="toasts">
      {toasts.map(function (t) {
        return (
          <div key={t.id} className={'toast ' + t.type}>
            {t.type === 'success' && <CheckCircle2 size={14} />}
            {t.type === 'error' && <AlertCircle size={14} />}
            {t.type === 'info' && <Info size={14} />}
            <span style={{ flex: 1 }}>{t.message}</span>
            <button className="toast-close" onClick={() => onDismiss(t.id)}>
              <X size={12} />
            </button>
          </div>
        );
      })}
    </div>
  );
}

export default React.memo(Toasts);

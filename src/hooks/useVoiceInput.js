import { useState, useRef, useCallback } from 'react';

export default function useVoiceInput(onResult) {
  const [recording, setRecording] = useState(false);
  const recogRef = useRef(null);

  const toggle = useCallback(() => {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
      onResult(null, 'Speech recognition not supported in this browser');
      return;
    }
    if (recording && recogRef.current) {
      recogRef.current.stop();
      setRecording(false);
      return;
    }
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recog = new SR();
    recog.continuous = false;
    recog.interimResults = false;
    recog.lang = 'en-US';
    recog.onresult = (e) => {
      const t = e.results[0][0].transcript;
      onResult(t);
      setRecording(false);
    };
    recog.onerror = () => setRecording(false);
    recog.onend = () => setRecording(false);
    recogRef.current = recog;
    recog.start();
    setRecording(true);
  }, [recording, onResult]);

  return { recording, toggle };
}

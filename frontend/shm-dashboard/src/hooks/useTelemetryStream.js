/**
 * useTelemetryStream.js
 * Custom React hook managing real-time WebSocket telemetry from FastAPI.
 * Dynamically discovers channels from incoming payloads, handles auto-reconnect
 * with exponential backoff, circular MSE buffer (100 frames), and structure switching.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { WS_BASE_URL } from '../config';

const HISTORY_SIZE = 100;

export function useTelemetryStream(urlOrStructureId = 'kw51') {
  const url = (typeof urlOrStructureId === 'string' && (urlOrStructureId.startsWith('ws://') || urlOrStructureId.startsWith('wss://')))
    ? urlOrStructureId
    : `${WS_BASE_URL}/ws/${urlOrStructureId || 'kw51'}/telemetry`;

  const [isConnected, setIsConnected] = useState(false);
  const [payload, setPayload] = useState(null);
  const [nodeHistory, setNodeHistory] = useState({});

  const wsRef = useRef(null);
  const reconnectDelay = useRef(1000);
  const reconnectTimer = useRef(null);
  const mounted = useRef(true);
  const pingTimer = useRef(null);

  const clearTimers = useCallback(() => {
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    if (pingTimer.current) clearInterval(pingTimer.current);
  }, []);

  // Reset telemetry history when switching structure/URL
  useEffect(() => {
    setPayload(null);
    setNodeHistory({});
  }, [url]);

  const connect = useCallback(() => {
    if (!mounted.current) return;

    // Close any previous open socket
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.close();
    }

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!mounted.current) return;
      setIsConnected(true);
      reconnectDelay.current = 1000;

      // Heartbeat ping every 15s
      pingTimer.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send('ping');
      }, 15000);
    };

    ws.onmessage = (evt) => {
      if (!mounted.current) return;
      if (evt.data === 'pong') return;
      try {
        const data = JSON.parse(evt.data);
        setPayload(data);

        // Dynamically discover all channels present in payload.nodes
        if (data.nodes && typeof data.nodes === 'object') {
          const channelNames = Object.keys(data.nodes);
          setNodeHistory(prev => {
            const next = { ...prev };
            channelNames.forEach(ch => {
              const mse = data.nodes[ch]?.mse ?? 0;
              const tau = data.nodes[ch]?.threshold ?? 0;
              const entry = { t: data.window_index ?? Date.now(), mse, tau };
              const prevArr = prev[ch] || [];
              const arr = [...prevArr, entry];
              next[ch] = arr.length > HISTORY_SIZE ? arr.slice(-HISTORY_SIZE) : arr;
            });
            return next;
          });
        }
      } catch {
        // ignore parse errors
      }
    };

    ws.onclose = () => {
      if (!mounted.current) return;
      setIsConnected(false);
      clearTimers();
      // Exponential backoff: 1s → 2s → 4s → 8s (max)
      reconnectDelay.current = Math.min(reconnectDelay.current * 2, 8000);
      reconnectTimer.current = setTimeout(connect, reconnectDelay.current);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [url, clearTimers]);

  useEffect(() => {
    mounted.current = true;
    connect();
    return () => {
      mounted.current = false;
      clearTimers();
      wsRef.current?.close();
    };
  }, [connect, clearTimers]);

  const disconnect = useCallback(() => {
    mounted.current = false;
    clearTimers();
    wsRef.current?.close();
  }, [clearTimers]);

  return { isConnected, payload, nodeHistory, disconnect };
}


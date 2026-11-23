/**
 * WebSocketClient.js
 * 
 * Subscribes to real-time FastAPI WebSocket telemetry stream at ws://localhost:8000/ws/telemetry.
 * Parses incoming JSON payloads and dispatches node health updates directly to SceneManager.
 */

export class TelemetryWebSocketClient {
    /**
     * @param {string} url - WebSocket server endpoint (default: ws://localhost:8000/ws/telemetry)
     * @param {Function} onTelemetryCallback - Callback receiving (nodeId, status, mseScore, payload)
     */
    constructor(url = "ws://localhost:8000/ws/telemetry", onTelemetryCallback = null) {
        this.url = url;
        this.onTelemetryCallback = onTelemetryCallback;
        this.ws = null;
        this.isReconnecting = false;
        this.reconnectInterval = 3000;
        this.pingTimer = null;
    }

    connect() {
        console.log(`[WebSocketClient] Connecting to ${this.url}...`);
        try {
            this.ws = new WebSocket(this.url);

            this.ws.onopen = () => {
                console.log("[WebSocketClient] Connected to live SHM Digital Twin Telemetry Stream.");
                this.isReconnecting = false;
                
                // Start 15s ping heartbeat
                this.pingTimer = setInterval(() => {
                    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                        this.ws.send("ping");
                    }
                }, 15000);
            };

            this.ws.onmessage = (event) => {
                if (event.data === "pong") return;
                try {
                    const payload = JSON.parse(event.data);
                    this._handleTelemetryPayload(payload);
                } catch (err) {
                    console.error("[WebSocketClient] Error parsing JSON telemetry payload:", err);
                }
            };

            this.ws.onclose = () => {
                console.warn("[WebSocketClient] Connection closed. Attempting reconnect...");
                this._cleanup();
                this._scheduleReconnect();
            };

            this.ws.onerror = (err) => {
                console.error("[WebSocketClient] WebSocket error:", err);
            };
        } catch (err) {
            console.error("[WebSocketClient] Connection attempt failed:", err);
            this._scheduleReconnect();
        }
    }

    _handleTelemetryPayload(payload) {
        if (!payload || !payload.nodes) return;

        // Extract system status and per-node health
        const systemStatus = payload.system_status || "NORMAL";
        const defectNode = payload.localized_defect_node;

        Object.entries(payload.nodes).forEach(([nodeId, nodeData]) => {
            if (this.onTelemetryCallback) {
                this.onTelemetryCallback(nodeId, nodeData.status, nodeData.mse, payload);
            }
        });
    }

    _scheduleReconnect() {
        if (this.isReconnecting) return;
        this.isReconnecting = true;
        setTimeout(() => {
            this.connect();
        }, this.reconnectInterval);
    }

    _cleanup() {
        if (this.pingTimer) {
            clearInterval(this.pingTimer);
            this.pingTimer = null;
        }
    }

    disconnect() {
        this.isReconnecting = false;
        this._cleanup();
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}

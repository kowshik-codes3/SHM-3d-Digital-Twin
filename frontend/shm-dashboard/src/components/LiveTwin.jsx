/**
 * LiveTwin.jsx
 * Stitch Screen: "StructureIQ - Live 3D Digital Twin"
 * SVG bridge + sensor panel + real-time telemetry charts + fault injection controls.
 */

import { useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import DigitalTwinCanvas from './DigitalTwinCanvas';
import TelemetryChart from './TelemetryChart';
import { API_BASE_URL } from '../config';

const FAULT_PRESETS = [
  { id: 'structural_degradation', label: 'Structural Degradation', desc: 'Injects 65% stiffness loss on MidSpan_Girder', icon: 'crisis_alert', color: '#ef4444' },
  { id: 'foundation_settlement',  label: 'Foundation Settlement',  desc: 'Injects 50% subsidence baseline drift',         icon: 'terrain',       color: '#f59e0b' },
  { id: 'dynamic_impulse',        label: 'Transient Dynamic Impulse', desc: 'High-G impact transient event',              icon: 'bolt',          color: '#a78bfa' },
];

function LiveDetectorCard({ detectorId, data }) {
  const state = data?.state ?? 'NORMAL';
  const score = data?.score ?? 0;
  const cadence = data?.update_cadence_seconds ?? 1.0;
  const reason = data?.reason;
  const cleanName = detectorId.replace(/_/g, ' ').toUpperCase();

  const badgeColor = {
    NORMAL: '#68f5b8',
    WARNING: '#f59e0b',
    CRITICAL: '#ef4444',
    LOW_CONFIDENCE: '#38bdf8',
    SENSOR_FAULT: '#c084fc',
  }[state] ?? '#68f5b8';

  return (
    <div className="px-3 py-2 border-b" style={{ borderColor: 'var(--outline-variant)', background: 'rgba(255,255,255,0.015)' }}>
      <div className="flex items-center justify-between mb-1">
        <span className="font-telemetry-sm text-[11px] font-semibold truncate text-slate-200" title={cleanName}>
          {cleanName}
        </span>
        <span
          className="font-label-caps text-[9px] px-1.5 py-0.2 rounded"
          style={{ color: badgeColor, border: `1px solid ${badgeColor}40`, background: `${badgeColor}15` }}
        >
          {state}
        </span>
      </div>
      <div className="flex items-center justify-between font-label-caps text-[10px]" style={{ color: 'var(--on-surface-variant)' }}>
        <span>Cadence: {cadence >= 60 ? `${cadence / 60}m` : `${cadence}s`}</span>
        <span>Score: {typeof score === 'number' ? score.toFixed(3) : '-'}</span>
      </div>
      {reason && (
        <div className="mt-1 font-telemetry-sm text-[9px] p-1 rounded truncate text-sky-400" style={{ background: 'rgba(56,189,248,0.1)', border: '1px solid rgba(56,189,248,0.2)' }} title={reason}>
          {reason}
        </div>
      )}
    </div>
  );
}

function NodeCard({ channelName, nodeData, sensorMeta }) {
  const status = sensorMeta?.status ?? nodeData?.status ?? 'NORMAL';
  const mse = nodeData?.mse ?? 0;
  const ratio = nodeData?.ratio ?? 0;
  const tau = nodeData?.threshold ?? 1;
  const qScore = sensorMeta?.quality_score;
  const faultReason = sensorMeta?.reasons?.[0];
  const barPct = Math.min((ratio / 3) * 100, 100);

  const statusStyle = {
    NORMAL:         { badge: 'badge-normal',   dot: '#68f5b8', label: 'NOMINAL' },
    DEGRADED:       { badge: 'badge-warning',  dot: '#94a3b8', label: 'DEGRADED' },
    WARNING:        { badge: 'badge-warning',  dot: '#f59e0b', label: 'WARNING' },
    CRITICAL:       { badge: 'badge-critical', dot: '#ef4444', label: 'CRITICAL' },
    LOW_CONFIDENCE: { badge: 'badge-warning',  dot: '#38bdf8', label: 'LOW CONF' },
    SENSOR_FAULT:   { badge: 'badge-warning',  dot: '#9ca3af', label: 'FAULT' },
    OFFLINE:        { badge: 'badge-warning',  dot: '#475569', label: 'OFFLINE' },
  }[status] ?? { badge: 'badge-normal', dot: '#68f5b8', label: 'NOMINAL' };

  const barColor = status === 'CRITICAL' ? '#ef4444' : status === 'WARNING' ? '#f59e0b' : status === 'LOW_CONFIDENCE' ? '#38bdf8' : (status === 'OFFLINE' || status === 'SENSOR_FAULT') ? '#64748b' : '#22d3ee';

  return (
    <div className="px-3 py-2.5 border-b" style={{ borderColor: 'var(--outline-variant)' }}>
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: statusStyle.dot }} />
          <span className="font-telemetry-sm text-telemetry-sm truncate max-w-[100px]" style={{ color: 'var(--on-surface)' }}>
            {channelName.replace('Sensor_', '').replace(/_/g, ' ')}
          </span>
        </div>
        <span className={`${statusStyle.badge} px-1.5 py-0.5 rounded font-label-caps text-label-caps`}>
          {statusStyle.label}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-1 mb-1.5">
        <div>
          <div className="font-label-caps text-label-caps uppercase mb-0.5" style={{ color: 'var(--on-surface-variant)' }}>MSE</div>
          <div className="font-telemetry-sm text-telemetry-sm" style={{ color: 'var(--primary)' }}>{Number(mse).toFixed(5)}</div>
        </div>
        <div>
          <div className="font-label-caps text-label-caps uppercase mb-0.5" style={{ color: 'var(--on-surface-variant)' }}>UCL (τ)</div>
          <div className="font-telemetry-sm text-telemetry-sm" style={{ color: 'var(--on-surface-variant)' }}>{Number(tau).toFixed(5)}</div>
        </div>
      </div>

      {/* Residual energy bar */}
      <div className="w-full h-1 rounded-full overflow-hidden" style={{ background: 'var(--surface-container)' }}>
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${barPct}%`, backgroundColor: barColor, boxShadow: status !== 'NORMAL' ? `0 0 6px ${barColor}` : 'none' }}
        />
      </div>
      <div className="flex items-center justify-between font-label-caps text-label-caps mt-0.5" style={{ color: 'var(--on-surface-variant)' }}>
        <span>{qScore !== undefined ? `Q ${Math.round(qScore * 100)}%` : ''}</span>
        <span>{Number(ratio).toFixed(2)}× UCL</span>
      </div>
      {faultReason && (
        <div className="mt-1 font-telemetry-sm text-[9px] text-slate-400 truncate" title={faultReason}>
          {faultReason}
        </div>
      )}
    </div>
  );
}

export default function LiveTwin({
  payload, nodeHistory, isConnected, structures,
  activeStructureId, onOpenSensorEditor, onOpenAddModal
}) {
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [faultLoading, setFaultLoading] = useState(null);
  const [isRunning, setIsRunning] = useState(true);

  const systemStatus = payload?.system_status ?? 'NORMAL';
  const defectNode = payload?.localized_defect_node ?? null;

  const channels = payload?.nodes ? Object.keys(payload.nodes) : [];

  const handleFaultInject = async (faultId) => {
    setFaultLoading(faultId);
    const targetMap = {
      structural_degradation: { node_id: channels[0] || 'sensor_0', fault_type: 'stiffness_loss', severity: 0.65 },
      foundation_settlement:  { node_id: channels[1] || channels[0] || 'sensor_0', fault_type: 'settlement', severity: 0.50 },
      dynamic_impulse:        { node_id: channels[0] || 'sensor_0', fault_type: 'spike', severity: 0.85 },
    };
    const body = targetMap[faultId] || { node_id: channels[0] || 'sensor_0', fault_type: faultId, severity: 0.65 };

    try {
      await fetch(`${API_BASE_URL}/api/v1/structures/${activeStructureId}/simulate-fault`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
    } catch (e) {
      console.error('Fault inject error:', e);
    } finally {
      setFaultLoading(null);
    }
  };

  const handleToggleStream = async () => {
    const endpoint = isRunning ? 'stop' : 'start';
    try {
      await fetch(`${API_BASE_URL}/api/v1/structures/${activeStructureId}/${endpoint}`, { method: 'POST' });
      setIsRunning(!isRunning);
    } catch (e) { console.error(e); }
  };

  const handleReset = async () => {
    try {
      await fetch(`${API_BASE_URL}/api/v1/structures/${activeStructureId}/reset-faults`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
    } catch (e) { console.error('Reset fault error:', e); }
  };

  return (
    <div className="flex w-full h-full overflow-hidden">

      {/* ── Left Sidebar: Sensor Status ── */}
      <div className="w-56 flex-shrink-0 flex flex-col border-r overflow-hidden" style={{ borderColor: 'var(--outline-variant)', background: 'var(--surface-container-low)' }}>
        {/* Header */}
        <div className="px-3 py-2.5 border-b flex items-center justify-between" style={{ borderColor: 'var(--outline-variant)' }}>
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-[16px]" style={{ color: 'var(--primary-container)' }}>sensors</span>
            <span className="font-body-sm text-body-sm font-semibold" style={{ color: 'var(--on-surface)' }}>
              SENSORS ({channels.length})
            </span>
          </div>
          <span className="font-telemetry-sm text-telemetry-sm" style={{ color: 'var(--on-surface-variant)' }}>
            {payload?.fs ?? 50} Hz
          </span>
        </div>

        {/* Active Detectors Section */}
        {payload?.detectors && Object.keys(payload.detectors).length > 0 && (
          <div className="border-b" style={{ borderColor: 'var(--outline-variant)' }}>
            <div className="px-3 py-1.5 flex items-center justify-between" style={{ background: 'rgba(0,0,0,0.1)' }}>
              <span className="font-label-caps text-[9px] uppercase tracking-wider text-slate-400">
                Detectors ({Object.keys(payload.detectors).length})
              </span>
            </div>
            {Object.entries(payload.detectors).map(([dId, dData]) => (
              <LiveDetectorCard key={dId} detectorId={dId} data={dData} />
            ))}
          </div>
        )}

        {/* Sensor cards */}
        <div className="flex-1 overflow-y-auto">
          {channels.length > 0 ? channels.map(ch => {
            const sensorMeta = payload?.sensors?.find(s => s.name === ch || s.id === ch);
            return (
              <NodeCard key={ch} channelName={ch} nodeData={payload?.nodes?.[ch]} sensorMeta={sensorMeta} />
            );
          }) : (
            <div className="flex flex-col items-center gap-2 py-10 opacity-40">
              <span className="material-symbols-outlined text-[32px]">sensors_off</span>
              <span className="font-body-sm text-body-sm text-center px-4" style={{ color: 'var(--on-surface-variant)' }}>
                Awaiting telemetry stream...
              </span>
            </div>
          )}
        </div>

        {/* Sensor Placement Editor button */}
        <div className="p-3 border-t" style={{ borderColor: 'var(--outline-variant)' }}>
          <button
            onClick={onOpenSensorEditor}
            className="w-full flex items-center gap-2 px-3 py-2 rounded font-body-sm text-body-sm transition-colors cursor-pointer"
            style={{ background: 'var(--surface-container)', color: 'var(--on-surface-variant)' }}
          >
            <span className="material-symbols-outlined text-[16px]">edit_location</span>
            <span>Sensor Placement</span>
          </button>
        </div>
      </div>

      {/* ── Center: SVG Digital Twin + Chart ── */}
      <div className="flex-1 flex flex-col min-w-0 min-h-0">

        {/* Critical alert banner */}
        <AnimatePresence>
          {systemStatus === 'CRITICAL' && defectNode && (
            <motion.div
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              className="flex items-center justify-between px-4 py-2 flex-shrink-0"
              style={{ background: 'rgba(239,68,68,0.08)', borderBottom: '1px solid rgba(239,68,68,0.25)' }}
            >
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined animate-pulse-error text-[18px]" style={{ color: '#ef4444' }}>crisis_alert</span>
                <span className="font-body-sm text-body-sm font-semibold" style={{ color: '#ef4444' }}>
                  STRUCTURAL ANOMALY DETECTED — LIMIT STATE EXCEEDANCE
                </span>
                <span className="font-telemetry-sm px-1.5 py-0.5 rounded" style={{ color: '#fca5a5', background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.3)' }}>
                  {defectNode.replace('Sensor_', '').replace(/_/g, '-')}
                </span>
              </div>
              <span className="font-label-caps text-label-caps" style={{ color: 'rgba(239,68,68,0.6)' }}>
                τ UCL BREACHED
              </span>
            </motion.div>
          )}
        </AnimatePresence>

        {/* 3D Canvas viewport — takes 65% of center */}
        <div
          className="flex-1 relative overflow-hidden"
          style={{ minHeight: '55%', background: 'radial-gradient(ellipse at 50% 60%, rgba(14,30,60,0.6) 0%, var(--background) 70%)' }}
        >
          {/* Node detail overlay when selected */}
          {selectedNodeId && payload?.nodes?.[selectedNodeId] && (() => {
            const nd = payload.nodes[selectedNodeId];
            return (
              <div
                className="absolute top-3 right-3 z-10 p-3 rounded-lg"
                style={{ background: 'rgba(28,31,42,0.92)', border: '1px solid var(--outline-variant)', backdropFilter: 'blur(12px)', minWidth: 180 }}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-telemetry-sm font-semibold" style={{ color: 'var(--primary-container)' }}>
                    {selectedNodeId.replace('Sensor_', '').replace(/_/g, '-')}
                  </span>
                  <button onClick={() => setSelectedNodeId(null)} className="material-symbols-outlined text-[14px] cursor-pointer" style={{ color: 'var(--on-surface-variant)' }}>close</button>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div><div className="font-label-caps" style={{ color: 'var(--on-surface-variant)' }}>MSE</div><div className="font-telemetry-sm" style={{ color: 'var(--primary)' }}>{Number(nd.mse).toFixed(5)}</div></div>
                  <div><div className="font-label-caps" style={{ color: 'var(--on-surface-variant)' }}>STATUS</div><div className={`font-label-caps badge-${(nd.status||'NORMAL').toLowerCase()} px-1.5 py-0.5 rounded inline-block`}>{nd.status || 'NORMAL'}</div></div>
                </div>
              </div>
            );
          })()}

          {/* Interactive 3D Digital Twin (React Three Fiber / Three.js) */}
          <DigitalTwinCanvas
            payload={payload}
            structure={
              structures?.find(s => s.structure_id === activeStructureId) || {
                structure_id: activeStructureId,
                name: activeStructureId,
                channels: channels,
              }
            }
          />

          {/* Bottom status line */}
          <div className="absolute bottom-2 left-3 right-3 flex items-center justify-between pointer-events-none z-10">
            <span className="font-label-caps text-label-caps uppercase" style={{ color: 'var(--on-surface-variant)' }}>
              CANTILEVER TRUSS 3D MODEL — SHM-v2.4
            </span>
            <div className="flex items-center gap-2">
              <span className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
              <span className="font-label-caps text-label-caps" style={{ color: 'var(--on-surface-variant)' }}>
                {isConnected ? 'WS LIVE' : 'DISCONNECTED'}
              </span>
            </div>
          </div>
        </div>

        {/* ── Live MSE Residuals Chart ── */}
        <div
          className="flex-shrink-0 border-t"
          style={{ height: '38%', minHeight: 140, borderColor: 'var(--outline-variant)', background: 'var(--surface-container-lowest)' }}
        >
          <TelemetryChart nodeHistory={nodeHistory} payload={payload} />
        </div>
      </div>

      {/* ── Right Panel: Ingestion + Fault Controls ── */}
      <div className="w-56 flex-shrink-0 flex flex-col border-l overflow-y-auto" style={{ borderColor: 'var(--outline-variant)', background: 'var(--surface-container-low)' }}>

        {/* Ingestion control */}
        <div className="px-3 py-3 border-b" style={{ borderColor: 'var(--outline-variant)' }}>
          <div className="flex items-center justify-between mb-2.5">
            <span className="font-label-caps text-label-caps uppercase" style={{ color: 'var(--on-surface-variant)' }}>
              INGESTION PIPELINE
            </span>
            <span className={`font-label-caps text-label-caps uppercase px-1.5 py-0.5 rounded ${isRunning ? 'badge-normal' : 'badge-warning'}`}>
              {isRunning ? 'ACTIVE' : 'PAUSED'}
            </span>
          </div>
          <button
            onClick={handleToggleStream}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded font-body-sm text-body-sm font-semibold transition-all cursor-pointer"
            style={{
              background: isRunning ? 'rgba(245,158,11,0.1)' : 'rgba(104,245,184,0.1)',
              color: isRunning ? '#f59e0b' : '#68f5b8',
              border: `1px solid ${isRunning ? 'rgba(245,158,11,0.3)' : 'rgba(104,245,184,0.3)'}`,
            }}
          >
            <span className="material-symbols-outlined text-[16px]">{isRunning ? 'pause' : 'play_arrow'}</span>
            <span>{isRunning ? 'Pause Stream' : 'Resume Stream'}</span>
          </button>
        </div>

        {/* Data Quality & Ingestion Health Panel */}
        <div className="px-3 py-2.5 border-b" style={{ borderColor: 'var(--outline-variant)' }}>
          <div className="flex items-center justify-between mb-1.5">
            <span className="font-label-caps text-[10px] uppercase" style={{ color: 'var(--on-surface-variant)' }}>
              DATA QUALITY
            </span>
            <span className="font-telemetry-sm text-[10px] font-semibold" style={{
              color: payload?.quality?.overall_status === 'OK' ? '#68f5b8'
                : payload?.quality?.overall_status === 'DEGRADED' ? '#f59e0b'
                : payload?.quality?.overall_status ? '#9ca3af' : '#68f5b8'
            }}>
              {payload?.quality ? `${Math.round((payload.quality.overall_quality ?? 1) * 100)}% ${payload.quality.overall_status}` : 'OK'}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-x-2 gap-y-1 text-[10px] font-telemetry-sm" style={{ color: 'var(--on-surface-variant)' }}>
            <span>Source: {payload?.ingestion?.source_status ?? (isRunning ? 'UP' : 'IDLE')}</span>
            <span>Rate: {(payload?.ingestion?.effective_frames_per_sec ?? 0).toFixed(1)} fps</span>
            <span>Lag: {(payload?.ingestion?.processing_lag_sec ?? 0).toFixed(2)}s</span>
            <span>Dropped: {payload?.ingestion?.dropped_frames_total ?? 0}</span>
          </div>
          {payload?.ingestion?.last_data_timestamp && (
            <div className="mt-1 text-[9px] font-telemetry-sm text-slate-400">
              Last packet: {new Date(payload.ingestion.last_data_timestamp * 1000).toLocaleTimeString()}
            </div>
          )}
          {payload?.primary_reason && systemStatus !== 'NORMAL' && (
            <div className="mt-1.5 p-1.5 rounded text-[9px] font-telemetry-sm text-sky-300 break-words" style={{ background: 'rgba(56,189,248,0.08)', border: '1px solid rgba(56,189,248,0.2)' }}>
              {payload.primary_reason}
            </div>
          )}
        </div>

        {/* Fault injection */}
        <div className="px-3 py-3 border-b flex-1" style={{ borderColor: 'var(--outline-variant)' }}>
          <div className="flex items-center gap-1.5 mb-3">
            <span className="material-symbols-outlined text-[16px]" style={{ color: '#f59e0b' }}>bolt</span>
            <span className="font-label-caps text-label-caps uppercase" style={{ color: 'var(--on-surface-variant)' }}>
              SIMULATE STRESS
            </span>
          </div>

          <div className="flex flex-col gap-2">
            {FAULT_PRESETS.map(f => (
              <button
                key={f.id}
                onClick={() => handleFaultInject(f.id)}
                disabled={!!faultLoading}
                className="text-left p-2.5 rounded cursor-pointer transition-all"
                style={{ background: 'var(--surface-container)', border: `1px solid ${f.color}22` }}
              >
                <div className="flex items-center gap-1.5 mb-0.5">
                  <span className="material-symbols-outlined text-[14px]" style={{ color: f.color }}>{f.icon}</span>
                  <span className="font-body-sm text-body-sm font-semibold" style={{ color: f.color }}>
                    {f.label}
                  </span>
                </div>
                <p className="font-body-sm text-[10px] leading-tight" style={{ color: 'var(--on-surface-variant)' }}>
                  {f.desc}
                </p>
              </button>
            ))}
          </div>
        </div>

        {/* Reset */}
        <div className="p-3">
          <button
            onClick={handleReset}
            className="w-full flex items-center justify-center gap-1.5 px-3 py-2 rounded font-body-sm text-body-sm font-semibold cursor-pointer transition-all"
            style={{ background: 'var(--surface-container)', color: 'var(--on-surface-variant)', border: '1px solid var(--outline-variant)' }}
          >
            <span className="material-symbols-outlined text-[16px]">restart_alt</span>
            <span>Reset to Healthy Baseline</span>
          </button>
        </div>
      </div>
    </div>
  );
}

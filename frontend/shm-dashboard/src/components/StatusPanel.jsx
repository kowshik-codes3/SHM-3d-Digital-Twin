/**
 * StatusPanel.jsx
 * StructureIQ Navigation Bar, Dynamic Sensor Telemetry Left Sidebar,
 * and Ingestion & Fault Injection Control Panel.
 */

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity, AlertTriangle, CheckCircle2, ChevronDown, Cpu, Layers,
  Play, Pause, Plus, Radio, RotateCcw, ShieldAlert, Wifi, WifiOff, Zap
} from 'lucide-react';
import { API_BASE_URL } from '../config';

export function StatusBadge({ status, reason }) {
  const styles = {
    NORMAL:         { bg: 'bg-emerald-500/10', text: 'text-emerald-400', border: 'border-emerald-500/30', icon: <CheckCircle2 size={12} />, pulse: '' },
    WARNING:        { bg: 'bg-amber-500/10',   text: 'text-amber-400',   border: 'border-amber-500/30',   icon: <AlertTriangle size={12} />, pulse: 'pulse-warning' },
    CRITICAL:       { bg: 'bg-red-500/10',     text: 'text-red-400',     border: 'border-red-500/30',     icon: <Zap size={12} />,           pulse: 'pulse-critical' },
    LOW_CONFIDENCE: { bg: 'bg-sky-500/10',     text: 'text-sky-400',     border: 'border-sky-500/30',     icon: <Activity size={12} />,      pulse: '' },
    SENSOR_FAULT:   { bg: 'bg-purple-500/10',  text: 'text-purple-400',  border: 'border-purple-500/30',  icon: <ShieldAlert size={12} />,   pulse: 'pulse-warning' },
  };
  const s = styles[status] ?? styles.NORMAL;
  return (
    <span
      title={reason || status}
      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold border ${s.bg} ${s.text} ${s.border} ${s.pulse}`}
    >
      {s.icon} {status}
    </span>
  );
}

function DetectorCard({ detectorId, data }) {
  const state = data?.state ?? 'NORMAL';
  const score = data?.score ?? 0;
  const tau = data?.tau ?? 0;
  const cadence = data?.update_cadence_seconds ?? 1.0;
  const reason = data?.reason;

  const cleanName = detectorId.replace(/_/g, ' ').toUpperCase();

  return (
    <motion.div layout className="glass-panel p-2.5 border border-white/10 mb-2">
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-1.5 truncate">
          <Cpu size={12} className="text-cyan-400 shrink-0" />
          <span className="text-[11px] font-semibold text-slate-200 truncate">{cleanName}</span>
        </div>
        <StatusBadge status={state} reason={reason} />
      </div>

      <div className="grid grid-cols-2 gap-1 text-[10px] text-slate-400 mb-1">
        <div>Cadence: <span className="font-mono text-slate-200">{cadence >= 60 ? `${cadence / 60}m` : `${cadence}s`}</span></div>
        <div className="text-right">Score: <span className="font-mono text-slate-200">{typeof score === 'number' ? score.toFixed(3) : '-'}</span></div>
      </div>

      {reason && (
        <div className="text-[10px] text-sky-400/90 font-mono bg-sky-950/30 p-1 rounded border border-sky-500/20 truncate" title={reason}>
          ℹ {reason}
        </div>
      )}
    </motion.div>
  );
}

function NodeCard({ channelName, nodeData }) {
  const status = nodeData?.status ?? 'NORMAL';
  const mse = nodeData?.mse ?? 0;
  const tau = nodeData?.threshold ?? 1;
  const ratio = nodeData?.ratio ?? 0;
  const barPct = Math.min((ratio / 3) * 100, 100);

  const barColor = status === 'CRITICAL' ? 'bg-red-500' : status === 'WARNING' ? 'bg-amber-500' : status === 'LOW_CONFIDENCE' ? 'bg-sky-500' : 'bg-emerald-500';
  const borderColor = status === 'CRITICAL' ? 'border-red-500/40 shadow-red-950/20 shadow-lg' : status === 'WARNING' ? 'border-amber-500/40' : 'border-white/5';
  const cleanLabel = channelName.replace(/^Sensor_/, '').replace(/_/g, ' ');

  return (
    <motion.div
      layout
      className={`glass-panel p-3 border ${borderColor} transition-colors duration-300`}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2 truncate">
          <span className="w-2 h-2 rounded-full bg-cyan-400/80 shrink-0" />
          <span className="text-xs font-medium text-slate-200 truncate">{cleanLabel}</span>
        </div>
        <StatusBadge status={status} />
      </div>
      <div className="grid grid-cols-2 gap-x-2 gap-y-1 text-xs mb-2">
        <div>
          <span className="text-slate-500 text-[10px] uppercase tracking-wider">MSE</span>
          <p className="mono font-semibold text-slate-200">{Number(mse).toFixed(5)}</p>
        </div>
        <div>
          <span className="text-slate-500 text-[10px] uppercase tracking-wider">UCL (τ)</span>
          <p className="mono font-semibold text-slate-200">{Number(tau).toFixed(5)}</p>
        </div>
      </div>
      {/* Risk bar */}
      <div className="h-1.5 rounded-full bg-white/5 w-full overflow-hidden">
        <motion.div
          className={`h-full rounded-full ${barColor}`}
          animate={{ width: `${barPct}%` }}
          transition={{ duration: 0.2 }}
        />
      </div>
      <div className="flex justify-between mt-1 text-[11px]">
        <span className="text-slate-500">Residual Ratio</span>
        <span className={`mono font-semibold ${status === 'CRITICAL' ? 'text-red-400 font-bold' : 'text-slate-400'}`}>
          {Number(ratio).toFixed(2)}×
        </span>
      </div>
    </motion.div>
  );
}

// ─── Top Navigation Bar ───────────────────────────────────────────────────────

export function TopNavBar({
  systemStatus,
  isConnected,
  defectNode,
  currentView,
  onViewChange,
  structures = [],
  activeStructureId,
  onSelectStructure,
  onOpenAddModal,
  criticalAlertCount = 0,
}) {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const activeStructure = structures.find(s => s.structure_id === activeStructureId) || {
    structure_id: activeStructureId,
    name: activeStructureId,
  };

  return (
    <header className="flex items-center justify-between px-5 py-2.5 border-b border-white/5 bg-[#090d16]/90 backdrop-blur-xl z-30">
      {/* Left: Branding & View Tabs */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2.5 cursor-pointer" onClick={() => onViewChange('twin')}>
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Activity size={18} className="text-white" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="font-bold text-sm tracking-wide text-white">StructureIQ</span>
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-cyan-500/20 text-cyan-400 font-mono">v2.0</span>
            </div>
            <p className="text-[10px] text-slate-500 hidden sm:block">Digital Twin SHM Platform</p>
          </div>
        </div>

        {/* View Navigation Tabs */}
        <nav className="flex items-center gap-1 bg-white/[0.03] p-1 rounded-lg border border-white/5">
          <button
            onClick={() => onViewChange('twin')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
              currentView === 'twin'
                ? 'bg-blue-600/30 text-white border border-blue-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-white/[0.04]'
            }`}
          >
            <Activity size={13} className={currentView === 'twin' ? 'text-blue-400' : ''} />
            <span>Live Twin</span>
          </button>

          <button
            onClick={() => onViewChange('fleet')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
              currentView === 'fleet'
                ? 'bg-blue-600/30 text-white border border-blue-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-white/[0.04]'
            }`}
          >
            <Layers size={13} className={currentView === 'fleet' ? 'text-blue-400' : ''} />
            <span>Fleet Grid</span>
            <span className="text-[10px] px-1 rounded bg-white/10 text-slate-300 font-mono">
              {structures.length}
            </span>
          </button>

          <button
            onClick={() => onViewChange('alerts')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
              currentView === 'alerts'
                ? 'bg-blue-600/30 text-white border border-blue-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-white/[0.04]'
            }`}
          >
            <ShieldAlert size={13} className={currentView === 'alerts' ? 'text-amber-400' : ''} />
            <span>Alerts Console</span>
            {criticalAlertCount > 0 && (
              <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-red-500/30 text-red-400 border border-red-500/40 font-mono font-bold animate-pulse">
                {criticalAlertCount}
              </span>
            )}
          </button>
        </nav>
      </div>

      {/* Center/Right: Structure Selector & Commissioning Action */}
      <div className="flex items-center gap-3">
        {/* Dynamic Structure Dropdown */}
        <div className="relative">
          <button
            onClick={() => setDropdownOpen(!dropdownOpen)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-white/10 bg-white/[0.04] hover:bg-white/[0.08] text-xs text-slate-200 transition-colors"
          >
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping inline-block" />
            <span className="font-semibold">{activeStructure.name || activeStructure.structure_id}</span>
            <ChevronDown size={14} className="text-slate-400" />
          </button>

          {dropdownOpen && (
            <div className="absolute right-0 mt-2 w-64 rounded-xl border border-white/10 bg-[#0d1322] shadow-2xl p-1 z-50">
              <div className="px-3 py-2 text-[10px] uppercase font-bold text-slate-500 tracking-wider">
                Select Asset Twin
              </div>
              {structures.map(s => (
                <button
                  key={s.structure_id}
                  onClick={() => {
                    onSelectStructure(s.structure_id);
                    setDropdownOpen(false);
                  }}
                  className={`w-full text-left px-3 py-2 rounded-lg text-xs flex items-center justify-between transition-colors ${
                    s.structure_id === activeStructureId
                      ? 'bg-blue-600/20 text-white border border-blue-500/30'
                      : 'text-slate-300 hover:bg-white/[0.05]'
                  }`}
                >
                  <div className="truncate mr-2">
                    <p className="font-semibold truncate">{s.name}</p>
                    <p className="text-[10px] text-slate-500 uppercase">{s.structure_type} · {s.num_channels} ch</p>
                  </div>
                  {s.structure_id === activeStructureId && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-400 shrink-0">ACTIVE</span>
                  )}
                </button>
              ))}
              <div className="border-t border-white/5 my-1" />
              <button
                onClick={() => {
                  setDropdownOpen(false);
                  onOpenAddModal();
                }}
                className="w-full text-left px-3 py-2 rounded-lg text-xs flex items-center gap-2 text-cyan-400 hover:bg-cyan-500/10 transition-colors font-medium"
              >
                <Plus size={14} /> Add New Asset Twin...
              </button>
            </div>
          )}
        </div>

        {/* Add Asset Quick Button */}
        <button
          onClick={onOpenAddModal}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white text-xs font-semibold shadow-md shadow-cyan-600/20 transition-all cursor-pointer"
        >
          <Plus size={14} />
          <span className="hidden md:inline">Add Asset</span>
        </button>

        {/* Defect Node indicator */}
        {defectNode && systemStatus === 'CRITICAL' && (
          <AnimatePresence>
            <motion.span
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0 }}
              className="text-xs text-red-400 mono hidden lg:block px-2.5 py-1 rounded bg-red-950/40 border border-red-500/30"
            >
              ⚡ Defect: {defectNode.replace('Sensor_', '')}
            </motion.span>
          </AnimatePresence>
        )}

        <StatusBadge status={systemStatus ?? 'NORMAL'} />

        {/* WebSocket Connection Indicator */}
        <div className={`flex items-center gap-1.5 text-xs px-2 py-1 rounded-md bg-white/[0.02] border border-white/5 ${isConnected ? 'text-emerald-400' : 'text-slate-500'}`}>
          {isConnected ? <Wifi size={13} /> : <WifiOff size={13} />}
          <span className="hidden sm:block">{isConnected ? 'Live' : 'Connecting…'}</span>
        </div>
      </div>
    </header>
  );
}

// ─── Left Sidebar ─────────────────────────────────────────────────────────────

export function LeftSidebar({ payload, structure }) {
  // Dynamically get channel names from payload nodes or structure config
  const channelNames = payload?.nodes
    ? Object.keys(payload.nodes)
    : structure?.channels || [];

  const detectors = payload?.detectors || {};
  const detectorIds = Object.keys(detectors);

  return (
    <aside className="flex flex-col gap-2 p-3 overflow-y-auto w-full">
      {/* Pluggable Detectors Multi-Cadence Section */}
      {detectorIds.length > 0 && (
        <div className="mb-2">
          <div className="flex items-center gap-2 mb-1 px-1">
            <Cpu size={12} className="text-cyan-400" />
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-widest">
              Active Detectors ({detectorIds.length})
            </span>
          </div>
          {detectorIds.map(dId => (
            <DetectorCard key={dId} detectorId={dId} data={detectors[dId]} />
          ))}
          <div className="border-t border-white/5 my-2" />
        </div>
      )}

      {/* Dynamic Sensor Nodes Section */}
      <div className="flex items-center justify-between mb-1 px-1">
        <div className="flex items-center gap-2">
          <Radio size={12} className="text-cyan-400" />
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-widest">
            Sensors ({channelNames.length})
          </span>
        </div>
        <span className="text-[10px] text-slate-500 font-mono">
          {structure?.fs ? `${structure.fs} Hz` : '50 Hz'}
        </span>
      </div>
      {channelNames.map(ch => (
        <NodeCard key={ch} channelName={ch} nodeData={payload?.nodes?.[ch]} />
      ))}
      {channelNames.length === 0 && (
        <div className="text-center py-8 text-xs text-slate-500">
          Waiting for telemetry frames...
        </div>
      )}
    </aside>
  );
}

// ─── Right Control Panel ──────────────────────────────────────────────────────

export function RightControlPanel({ structureId = 'kw51', structure, onRefreshStructures }) {
  const [actionLoading, setActionLoading] = useState(false);
  const [ingestionRunning, setIngestionRunning] = useState(structure?.is_running ?? true);

  const channelNames = structure?.channels || (
    structure?.sensors ? structure.sensors.map(s => s.name || s.id) : []
  );

  const targetNode = channelNames[0] || 'sensor_0';

  const handleFault = async (faultType, severity) => {
    setActionLoading(true);
    try {
      await fetch(`${API_BASE_URL}/api/v1/structures/${structureId}/simulate-fault`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ node_id: targetNode, fault_type: faultType, severity }),
      });
    } catch (e) {
      // fallback to legacy route
      await fetch(`${API_BASE_URL}/api/v1/simulate-fault`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ node_id: targetNode, fault_type: faultType, severity }),
      }).catch(() => {});
    } finally {
      setActionLoading(false);
    }
  };

  const handleReset = async () => {
    setActionLoading(true);
    try {
      await fetch(`${API_BASE_URL}/api/v1/structures/${structureId}/reset-faults`, { method: 'POST' });
    } catch {
      await fetch(`${API_BASE_URL}/api/v1/reset-faults`, { method: 'POST' }).catch(() => {});
    } finally {
      setActionLoading(false);
    }
  };

  const toggleIngestion = async () => {
    setActionLoading(true);
    const endpoint = ingestionRunning ? 'stop' : 'start';
    try {
      await fetch(`${API_BASE_URL}/api/v1/structures/${structureId}/${endpoint}`, { method: 'POST' });
      setIngestionRunning(!ingestionRunning);
      if (onRefreshStructures) onRefreshStructures();
    } catch (err) {
      console.error(err);
    } finally {
      setActionLoading(false);
    }
  };

  return (
    <aside className="flex flex-col gap-4 p-3">
      {/* Ingestion Stream Toggle */}
      <div className="glass-panel p-3 border border-white/5">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-widest">
            Ingestion Pipeline
          </span>
          <span className={`text-[10px] px-2 py-0.5 rounded-full font-mono ${ingestionRunning ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-700 text-slate-400'}`}>
            {ingestionRunning ? 'ACTIVE' : 'PAUSED'}
          </span>
        </div>
        <button
          onClick={toggleIngestion}
          disabled={actionLoading}
          className={`w-full flex items-center justify-center gap-2 text-xs font-semibold py-2 px-3 rounded-lg border transition-all cursor-pointer ${
            ingestionRunning
              ? 'border-amber-500/40 text-amber-300 hover:bg-amber-950/30'
              : 'border-emerald-500/40 text-emerald-300 hover:bg-emerald-950/30'
          }`}
        >
          {ingestionRunning ? <Pause size={13} /> : <Play size={13} />}
          {ingestionRunning ? 'Pause Stream' : 'Resume Stream'}
        </button>
      </div>

      {/* Fault Injection Panel */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2 mb-1">
          <Zap size={12} className="text-amber-400" />
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-widest">
            Simulate Stress
          </span>
        </div>

        <button
          onClick={() => handleFault('stiffness_loss', 0.65)}
          disabled={actionLoading}
          className="w-full text-left text-xs px-3 py-2.5 rounded-lg border border-red-500/30 bg-red-950/10 hover:bg-red-950/30 text-slate-200 cursor-pointer transition-all"
        >
          <span className="font-semibold block text-red-400">Structural Degradation</span>
          <span className="text-slate-400 text-[11px]">Injects 65% stiffness loss on {targetNode.replace('Sensor_', '')}</span>
        </button>

        <button
          onClick={() => handleFault('settlement', 0.50)}
          disabled={actionLoading}
          className="w-full text-left text-xs px-3 py-2.5 rounded-lg border border-orange-500/30 bg-orange-950/10 hover:bg-orange-950/30 text-slate-200 cursor-pointer transition-all"
        >
          <span className="font-semibold block text-orange-400">Foundation Settlement</span>
          <span className="text-slate-400 text-[11px]">Injects 50% subsidence baseline drift</span>
        </button>

        <button
          onClick={() => handleFault('spike', 0.75)}
          disabled={actionLoading}
          className="w-full text-left text-xs px-3 py-2.5 rounded-lg border border-yellow-500/30 bg-yellow-950/10 hover:bg-yellow-950/30 text-slate-200 cursor-pointer transition-all"
        >
          <span className="font-semibold block text-yellow-400">Transient Dynamic Impulse</span>
          <span className="text-slate-400 text-[11px]">High-G impact transient event</span>
        </button>

        <button
          onClick={handleReset}
          disabled={actionLoading}
          className="w-full flex items-center justify-center gap-2 text-xs px-3 py-2.5 rounded-lg border border-emerald-500/30 bg-emerald-950/10 text-emerald-400 hover:bg-emerald-950/30 cursor-pointer transition-all mt-1 font-medium"
        >
          <RotateCcw size={12} /> Reset to Healthy Baseline
        </button>
      </div>
    </aside>
  );
}


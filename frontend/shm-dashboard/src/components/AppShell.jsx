/**
 * AppShell.jsx
 * Universal Stitch header and navigation shell for StructureIQ SHM Platform.
 */

import React, { useState } from 'react';

export default function AppShell({
  currentView,
  onNavigate,
  structures = [],
  activeStructureId,
  onSelectStructure,
  systemStatus = 'NORMAL',
  isConnected = false,
  alertCount = 0,
  onOpenAddModal,
  children
}) {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const activeStructure = structures.find(s => s.structure_id === activeStructureId) || structures[0];

  const statusStyle = {
    NORMAL:         { badge: 'badge-normal',   text: 'text-tertiary',  dot: 'bg-tertiary',  label: 'NORMAL' },
    WARNING:        { badge: 'badge-warning',  text: 'text-warning',   dot: 'bg-warning',   label: 'WARNING' },
    CRITICAL:       { badge: 'badge-critical', text: 'text-error',     dot: 'bg-error',     label: 'CRITICAL' },
    LOW_CONFIDENCE: { badge: 'badge-warning',  text: 'text-cyan-400',  dot: 'bg-cyan-400',  label: 'LOW CONFIDENCE' },
    SENSOR_FAULT:   { badge: 'badge-warning',  text: 'text-purple-400',dot: 'bg-purple-400',label: 'SENSOR FAULT' },
  }[systemStatus] ?? { badge: 'badge-normal', text: 'text-tertiary', dot: 'bg-tertiary', label: 'NORMAL' };

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-background text-on-surface">
      {/* ── Stitch Top App Bar ── */}
      <header className="app-header justify-between select-none">
        {/* Left: Brand + Navigation tabs */}
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2 cursor-pointer" onClick={() => onNavigate('twin')}>
            <div className="w-7 h-7 rounded bg-primary-container flex items-center justify-center text-on-primary">
              <span className="material-symbols-outlined text-[18px]">polyline</span>
            </div>
            <div className="flex flex-col leading-none">
              <div className="flex items-center gap-1.5">
                <span className="font-headline-sm text-sm font-bold tracking-tight text-on-surface">StructureIQ</span>
                <span className="font-telemetry-sm text-[10px] px-1 py-0.2 rounded bg-surface-container-highest text-primary font-mono">v2.4</span>
              </div>
              <span className="font-label-caps text-[9px] text-on-surface-variant uppercase tracking-wider">Digital Twin SHM</span>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="flex items-center gap-1">
            <button
              onClick={() => onNavigate('twin')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded font-body-sm text-xs font-medium cursor-pointer transition-all ${
                currentView === 'twin'
                  ? 'bg-surface-container-high text-primary border border-primary/20 shadow-sm'
                  : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container'
              }`}
            >
              <span className="material-symbols-outlined text-[16px]">show_chart</span>
              <span>Live Twin</span>
            </button>

            <button
              onClick={() => onNavigate('fleet')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded font-body-sm text-xs font-medium cursor-pointer transition-all ${
                currentView === 'fleet'
                  ? 'bg-surface-container-high text-primary border border-primary/20 shadow-sm'
                  : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container'
              }`}
            >
              <span className="material-symbols-outlined text-[16px]">domain</span>
              <span>Fleet Grid</span>
              <span className="font-telemetry-sm text-[10px] px-1 rounded bg-surface-container-highest text-on-surface-variant">
                {structures.length}
              </span>
            </button>

            <button
              onClick={() => onNavigate('alerts')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded font-body-sm text-xs font-medium cursor-pointer transition-all ${
                currentView === 'alerts'
                  ? 'bg-surface-container-high text-primary border border-primary/20 shadow-sm'
                  : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container'
              }`}
            >
              <span className="material-symbols-outlined text-[16px]">notifications_active</span>
              <span>Alerts Console</span>
              {alertCount > 0 && (
                <span className="font-telemetry-sm text-[10px] px-1 rounded bg-error/20 text-error font-bold">
                  {alertCount}
                </span>
              )}
            </button>

            <button
              onClick={() => onNavigate('settings')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded font-body-sm text-xs font-medium cursor-pointer transition-all ${
                currentView === 'settings'
                  ? 'bg-surface-container-high text-primary border border-primary/20 shadow-sm'
                  : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container'
              }`}
            >
              <span className="material-symbols-outlined text-[16px]">settings</span>
              <span>Settings</span>
            </button>
          </nav>
        </div>

        {/* Right: Structure Selector + Status Pill + Add Button */}
        <div className="flex items-center gap-3">
          {/* Active Structure Selector */}
          <div className="relative">
            <button
              onClick={() => setDropdownOpen(!dropdownOpen)}
              className="flex items-center gap-2 px-3 py-1.5 rounded bg-surface-container hover:bg-surface-container-high border border-outline-variant font-body-sm text-xs cursor-pointer transition-colors"
            >
              <span className="w-2 h-2 rounded-full bg-primary" />
              <span className="font-medium text-on-surface max-w-[160px] truncate">
                {activeStructure?.name || 'Select Asset'}
              </span>
              <span className="material-symbols-outlined text-[16px] text-on-surface-variant">
                {dropdownOpen ? 'arrow_drop_up' : 'arrow_drop_down'}
              </span>
            </button>

            {dropdownOpen && (
              <div className="absolute right-0 mt-1 w-64 rounded-lg bg-surface-container-high border border-outline-variant shadow-xl z-50 py-1 overflow-hidden">
                <div className="px-3 py-1.5 font-label-caps text-[10px] text-on-surface-variant uppercase border-b border-outline-variant">
                  Monitored Assets ({structures.length})
                </div>
                {structures.map(s => (
                  <button
                    key={s.structure_id}
                    onClick={() => {
                      onSelectStructure(s.structure_id);
                      setDropdownOpen(false);
                    }}
                    className={`w-full text-left px-3 py-2 flex items-center justify-between hover:bg-surface-container text-xs cursor-pointer ${
                      s.structure_id === activeStructureId ? 'text-primary bg-primary/10' : 'text-on-surface'
                    }`}
                  >
                    <div>
                      <div className="font-medium">{s.name}</div>
                      <div className="font-telemetry-sm text-[10px] text-on-surface-variant font-mono">{s.structure_id}</div>
                    </div>
                    <span className="font-label-caps text-[10px] px-1 rounded bg-surface-container">
                      {s.num_channels} ch
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Add Asset Button */}
          <button
            onClick={onOpenAddModal}
            className="btn-primary"
          >
            <span className="material-symbols-outlined text-[16px]">add_circle</span>
            <span>+ Add Asset</span>
          </button>

          {/* System Status Pill */}
          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full font-label-caps text-[11px] font-semibold border ${statusStyle.badge}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${statusStyle.dot} ${systemStatus !== 'NORMAL' ? 'animate-ping' : ''}`} />
            <span>{statusStyle.label}</span>
          </div>

          {/* Live Telemetry Ping */}
          <div className="flex items-center gap-1.5 font-telemetry-sm text-[11px] text-on-surface-variant px-2 py-1 rounded bg-surface-container">
            <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-tertiary animate-pulse' : 'bg-outline'}`} />
            <span className={isConnected ? 'text-tertiary font-semibold' : 'text-outline'}>
              {isConnected ? 'LIVE' : 'OFFLINE'}
            </span>
          </div>
        </div>
      </header>

      {/* Main View Container */}
      <main className="flex-1 overflow-hidden relative">
        {children}
      </main>
    </div>
  );
}

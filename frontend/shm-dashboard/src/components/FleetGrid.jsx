/**
 * FleetGrid.jsx
 * Stitch Screen: "StructureIQ - Structures Overview Fleet Grid" (1a94a80d...)
 * Multi-structure civil asset fleet management with KPIs, filter/sort toolbar,
 * real-time health indexes, ingestion toggles, and commissioning action.
 */

import React, { useState } from 'react';
import { API_BASE_URL } from '../config';

const TYPE_ICONS = {
  bridge: 'merge_type',
  building: 'apartment',
  dam: 'water_damage',
  tower: 'cell_tower',
  tunnel: 'tunnel',
};

export default function FleetGrid({
  structures = [],
  onSelectStructure,
  onNavigate,
  onOpenAddModal,
  onRefresh
}) {
  const [searchQuery, setSearchQuery] = useState('');
  const [typeFilter, setTypeFilter] = useState('ALL');
  const [viewMode, setViewMode] = useState('grid'); // 'grid' | 'list'
  const [actionLoading, setActionLoading] = useState(null);

  // Compute live KPIs
  const totalAssets = structures.length;
  const runningCount = structures.filter(s => s.is_running).length;
  const totalChannels = structures.reduce((acc, s) => acc + (s.num_channels || 0), 0);

  // Filter structures
  const filtered = structures.filter(s => {
    const matchesSearch = !searchQuery ||
      s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.structure_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (s.structure_type || '').toLowerCase().includes(searchQuery.toLowerCase());
    const matchesType = typeFilter === 'ALL' || (s.structure_type || 'bridge').toUpperCase() === typeFilter;
    return matchesSearch && matchesType;
  });

  const handleToggleRunning = async (structureId, isRunning) => {
    setActionLoading(structureId);
    const endpoint = isRunning ? 'stop' : 'start';
    try {
      await fetch(`${API_BASE_URL}/api/v1/structures/${structureId}/${endpoint}`, { method: 'POST' });
      onRefresh && onRefresh();
    } catch (e) {
      console.error(e);
    } finally {
      setActionLoading(null);
    }
  };

  const handleDelete = async (structureId) => {
    if (structureId === 'bridge_demo_01') {
      alert('The primary bridge demo structure is protected and cannot be deleted.');
      return;
    }
    if (!window.confirm(`Are you sure you want to decommission and delete asset '${structureId}'?`)) {
      return;
    }
    setActionLoading(structureId);
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/structures/${structureId}`, { method: 'DELETE' });
      if (res.ok) {
        onRefresh && onRefresh();
      }
    } catch (e) {
      console.error(e);
    } finally {
      setActionLoading(null);
    }
  };

  return (
    <div className="h-full overflow-y-auto p-6 flex flex-col gap-6 max-w-[1720px] mx-auto w-full">

      {/* ── Page Header ── */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-primary font-label-caps text-label-caps uppercase tracking-wider mb-1">
            <span className="material-symbols-outlined text-[16px]">domain_verification</span>
            <span>Infrastructure Asset Intelligence</span>
          </div>
          <h1 className="font-headline-lg text-headline-lg tracking-tight text-on-surface">Structural Asset Fleet</h1>
          <p className="font-body-sm text-body-sm text-on-surface-variant mt-0.5">
            {totalAssets} Monitored Civil Assets • Real-time Synchronous PyTorch Autoencoder Telemetry
          </p>
        </div>

        <button
          onClick={onOpenAddModal}
          className="btn-primary"
        >
          <span className="material-symbols-outlined text-[18px]">add_circle</span>
          <span>+ Add Structure</span>
        </button>
      </div>

      {/* ── Top KPI Summary Bar ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {/* Total Assets */}
        <div className="p-4 rounded-xl bg-surface-container-low border border-outline-variant/30 flex flex-col justify-between">
          <div className="flex items-center justify-between text-on-surface-variant font-label-caps text-label-caps uppercase">
            <span>Total Assets</span>
            <span className="material-symbols-outlined text-primary text-[18px]">domain_verification</span>
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="font-telemetry-lg text-telemetry-lg font-bold text-on-surface">
              {String(totalAssets).padStart(2, '0')}
            </span>
            <span className="font-body-sm text-body-sm text-on-surface-variant">/ Civil Class I</span>
          </div>
          <div className="mt-2 font-telemetry-sm text-telemetry-sm text-tertiary flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-tertiary" />
            <span>{runningCount} Online • {totalAssets - runningCount} Standby</span>
          </div>
        </div>

        {/* Fleet Health Index */}
        <div className="p-4 rounded-xl bg-surface-container-low border border-outline-variant/30 flex flex-col justify-between">
          <div className="flex items-center justify-between text-on-surface-variant font-label-caps text-label-caps uppercase">
            <span>Fleet Health Index</span>
            <span className="material-symbols-outlined text-tertiary text-[18px]">health_and_safety</span>
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="font-telemetry-lg text-telemetry-lg font-bold text-on-surface">96.4%</span>
            <span className="font-body-sm text-body-sm text-outline">Mean FHI</span>
          </div>
          <div className="mt-2 font-telemetry-sm text-telemetry-sm flex items-center justify-between">
            <span className="text-tertiary">All Nominals In-Spec</span>
            <span className="text-primary font-medium">UCL Guard Active</span>
          </div>
        </div>

        {/* Ingested Channels */}
        <div className="p-4 rounded-xl bg-surface-container-low border border-outline-variant/30 flex flex-col justify-between">
          <div className="flex items-center justify-between text-on-surface-variant font-label-caps text-label-caps uppercase">
            <span>Ingested Channels</span>
            <span className="material-symbols-outlined text-secondary text-[18px]">sensors</span>
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="font-telemetry-lg text-telemetry-lg font-bold text-on-surface">{totalChannels}</span>
            <span className="font-telemetry-sm text-telemetry-sm text-on-surface-variant">Active DAQ Points</span>
          </div>
          <div className="mt-2 font-telemetry-sm text-telemetry-sm text-on-surface-variant flex items-center justify-between">
            <span>Sampling: 50 Hz</span>
            <span className="text-tertiary font-medium">100% Ingress Rate</span>
          </div>
        </div>

        {/* Inference Load */}
        <div className="p-4 rounded-xl bg-surface-container-low border border-outline-variant/30 flex flex-col justify-between">
          <div className="flex items-center justify-between text-on-surface-variant font-label-caps text-label-caps uppercase">
            <span>Inference Executions</span>
            <span className="material-symbols-outlined text-primary text-[18px]">memory</span>
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="font-telemetry-lg text-telemetry-lg font-bold text-on-surface">
              {structures.reduce((acc, s) => acc + (s.inferences_executed || 0), 0).toLocaleString()}
            </span>
          </div>
          <div className="mt-2 font-telemetry-sm text-telemetry-sm text-tertiary flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-tertiary" />
            <span>PyTorch LSTM Autoencoders Active</span>
          </div>
        </div>
      </div>

      {/* ── Filter & Search Toolbar ── */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-xl bg-surface-container-low border border-outline-variant/30">
        {/* Search */}
        <div className="flex items-center gap-2 flex-1 min-w-[240px] px-3 py-1.5 rounded-lg bg-surface-container border border-outline-variant/40">
          <span className="material-symbols-outlined text-[18px] text-on-surface-variant">search</span>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Filter by asset name, type, id..."
            className="w-full bg-transparent text-xs text-on-surface outline-none placeholder:text-on-surface-variant/60 font-body-sm"
          />
          {searchQuery && (
            <button onClick={() => setSearchQuery('')} className="material-symbols-outlined text-[16px] text-on-surface-variant cursor-pointer">
              close
            </button>
          )}
        </div>

        {/* Type Filter Pills */}
        <div className="flex items-center gap-1 overflow-x-auto">
          {['ALL', 'BRIDGE', 'BUILDING', 'DAM', 'TOWER'].map(t => (
            <button
              key={t}
              onClick={() => setTypeFilter(t)}
              className={`px-3 py-1.5 rounded-lg font-label-caps text-label-caps font-semibold uppercase cursor-pointer transition-all ${
                typeFilter === t
                  ? 'bg-primary text-on-primary shadow-sm'
                  : 'bg-surface-container text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high'
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        {/* View Toggle */}
        <div className="flex items-center gap-1 bg-surface-container p-0.5 rounded-lg">
          <button
            onClick={() => setViewMode('grid')}
            className={`p-1 rounded cursor-pointer ${viewMode === 'grid' ? 'bg-surface-container-highest text-primary' : 'text-on-surface-variant'}`}
            title="Grid View"
          >
            <span className="material-symbols-outlined text-[18px]">grid_view</span>
          </button>
          <button
            onClick={() => setViewMode('list')}
            className={`p-1 rounded cursor-pointer ${viewMode === 'list' ? 'bg-surface-container-highest text-primary' : 'text-on-surface-variant'}`}
            title="List View"
          >
            <span className="material-symbols-outlined text-[18px]">view_list</span>
          </button>
        </div>
      </div>

      {/* ── Asset Grid ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filtered.map(structure => {
          const typeIcon = TYPE_ICONS[structure.structure_type] || 'apartment';
          const isRunning = structure.is_running;

          return (
            <div
              key={structure.structure_id}
              className="p-5 rounded-xl bg-surface-container-low border border-outline-variant/30 hover:border-primary/40 transition-all flex flex-col justify-between group shadow-sm relative overflow-hidden"
            >
              {/* Card top */}
              <div>
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded-lg bg-surface-container flex items-center justify-center text-primary border border-outline-variant/30">
                      <span className="material-symbols-outlined text-[18px]">{typeIcon}</span>
                    </div>
                    <div>
                      <span className="font-label-caps text-[9px] text-on-surface-variant uppercase tracking-wider block">
                        {structure.structure_type?.toUpperCase() || 'CIVIL ASSET'}
                      </span>
                      <h3 className="font-headline-sm text-sm font-semibold text-on-surface group-hover:text-primary transition-colors">
                        {structure.name}
                      </h3>
                    </div>
                  </div>

                  <span className={`px-2 py-0.5 rounded font-label-caps text-[10px] font-semibold border ${isRunning ? 'badge-normal' : 'badge-warning'}`}>
                    {isRunning ? 'STREAMING' : 'PAUSED'}
                  </span>
                </div>

                {/* ID & channel stats */}
                <div className="font-telemetry-sm text-[11px] text-on-surface-variant font-mono mb-4 flex items-center gap-2">
                  <span className="px-1.5 py-0.5 rounded bg-surface-container">ID: {structure.structure_id}</span>
                  <span>•</span>
                  <span>{structure.num_channels} Sensor Channels</span>
                </div>

                {/* Channel pills */}
                <div className="flex flex-wrap gap-1 mb-4">
                  {(structure.channels || []).slice(0, 4).map(ch => (
                    <span
                      key={ch}
                      className="font-telemetry-sm text-[10px] px-2 py-0.5 rounded bg-surface-container text-on-surface-variant border border-outline-variant/20"
                    >
                      {ch.replace('Sensor_', '').replace(/_/g, ' ')}
                    </span>
                  ))}
                  {(structure.channels || []).length > 4 && (
                    <span className="font-telemetry-sm text-[10px] px-1.5 py-0.5 rounded bg-surface-container text-primary">
                      +{structure.channels.length - 4} more
                    </span>
                  )}
                </div>

                {/* Telemetry Stats */}
                <div className="p-3 rounded-lg bg-surface-container/60 border border-outline-variant/20 mb-4 grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="font-label-caps text-[9px] text-on-surface-variant uppercase block">Inferences</span>
                    <span className="font-telemetry-sm text-on-surface font-semibold font-mono">
                      {(structure.inferences_executed || 0).toLocaleString()}
                    </span>
                  </div>
                  <div>
                    <span className="font-label-caps text-[9px] text-on-surface-variant uppercase block">Health State</span>
                    <span className="font-telemetry-sm text-tertiary font-semibold flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-tertiary" />
                      Nominal
                    </span>
                  </div>
                </div>
              </div>

              {/* Card Actions */}
              <div className="pt-3 border-t border-outline-variant/20 flex items-center justify-between gap-2">
                <button
                  onClick={() => {
                    onSelectStructure(structure.structure_id);
                    onNavigate('twin');
                  }}
                  className="btn-ghost-cyan flex-1 justify-center text-xs py-1.5"
                >
                  <span className="material-symbols-outlined text-[16px]">visibility</span>
                  <span>Inspect Twin</span>
                </button>

                <button
                  onClick={() => handleToggleRunning(structure.structure_id, isRunning)}
                  disabled={actionLoading === structure.structure_id}
                  className="p-1.5 rounded hover:bg-surface-container text-on-surface-variant hover:text-on-surface border border-outline-variant/30 cursor-pointer"
                  title={isRunning ? 'Pause Ingestion' : 'Resume Ingestion'}
                >
                  <span className="material-symbols-outlined text-[18px]">
                    {isRunning ? 'pause' : 'play_arrow'}
                  </span>
                </button>

                {structure.structure_id !== 'bridge_demo_01' && (
                  <button
                    onClick={() => handleDelete(structure.structure_id)}
                    disabled={actionLoading === structure.structure_id}
                    className="p-1.5 rounded hover:bg-error/10 text-outline hover:text-error border border-outline-variant/30 cursor-pointer"
                    title="Decommission Structure"
                  >
                    <span className="material-symbols-outlined text-[18px]">delete</span>
                  </button>
                )}
              </div>
            </div>
          );
        })}

        {/* ── Commission New Asset Card ── */}
        <div
          onClick={onOpenAddModal}
          className="p-6 rounded-xl border-2 border-dashed border-outline-variant/40 hover:border-primary/60 bg-surface-container/20 hover:bg-surface-container/40 transition-all flex flex-col items-center justify-center text-center cursor-pointer min-h-[260px] group"
        >
          <div className="w-12 h-12 rounded-full bg-surface-container-high group-hover:bg-primary/20 flex items-center justify-center text-on-surface-variant group-hover:text-primary mb-3 transition-colors">
            <span className="material-symbols-outlined text-[24px]">add</span>
          </div>
          <h4 className="font-headline-sm text-sm font-semibold text-on-surface group-hover:text-primary transition-colors mb-1">
            Commission New Civil Asset
          </h4>
          <p className="font-body-sm text-[11px] text-on-surface-variant max-w-[200px]">
            Deploy new bridge, dam, or building with automated LSTM artifact synthesis
          </p>
        </div>
      </div>
    </div>
  );
}

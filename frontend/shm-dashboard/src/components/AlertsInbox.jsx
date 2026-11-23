/**
 * AlertsInbox.jsx
 * Stitch Screen: "StructureIQ - Alerts Inbox & Incident Console" (f4bca74d...)
 * Enterprise incident console with Defcon-2 counters, severity filter tabs,
 * real-time audit log table, operator acknowledgement, and AI Diagnosis drawer trigger.
 */

import React, { useState } from 'react';

export default function AlertsInbox({
  alerts = [],
  alertStats = { total: 0, critical: 0, warning: 0, unacknowledged: 0 },
  onAcknowledgeAlert,
  onOpenAlertDetail,
  onRefresh
}) {
  const [filterSeverity, setFilterSeverity] = useState('ALL');
  const [searchQuery, setSearchQuery] = useState('');
  const [ackLoading, setAckLoading] = useState(null);

  // Filter alerts
  const filtered = alerts.filter(a => {
    const matchesSev = filterSeverity === 'ALL' || a.severity === filterSeverity ||
      (filterSeverity === 'UNACK' && !a.acknowledged);
    const matchesSearch = !searchQuery ||
      a.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      a.structure_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (a.channel_name || '').toLowerCase().includes(searchQuery.toLowerCase());
    return matchesSev && matchesSearch;
  });

  const handleAck = async (alertId) => {
    setAckLoading(alertId);
    try {
      await onAcknowledgeAlert(alertId);
    } finally {
      setAckLoading(null);
    }
  };

  return (
    <div className="h-full overflow-y-auto p-6 flex flex-col gap-6 max-w-[1720px] mx-auto w-full">

      {/* ── Page Header ── */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-error font-label-caps text-label-caps uppercase tracking-wider mb-1">
            <span className="material-symbols-outlined text-[16px]">crisis_alert</span>
            <span>Real-Time Incident Triage & Defcon Protocol</span>
          </div>
          <h1 className="font-headline-lg text-headline-lg tracking-tight text-on-surface">Incident Console</h1>
          <p className="font-body-sm text-body-sm text-on-surface-variant mt-0.5">
            Audit logging of structural anomaly events and operator acknowledgements
          </p>
        </div>

        <button
          onClick={onRefresh}
          className="btn-ghost-cyan"
        >
          <span className="material-symbols-outlined text-[16px]">refresh</span>
          <span>Refresh Feed</span>
        </button>
      </div>

      {/* ── Top KPI Stat Cards ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {/* Total Incidents */}
        <div className="p-4 rounded-xl bg-surface-container-low border border-outline-variant/30 flex flex-col justify-between">
          <div className="flex items-center justify-between text-on-surface-variant font-label-caps text-label-caps uppercase">
            <span>Total Events</span>
            <span className="material-symbols-outlined text-primary text-[18px]">receipt_long</span>
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="font-telemetry-lg text-telemetry-lg font-bold text-on-surface">
              {String(alertStats.total || alerts.length).padStart(2, '0')}
            </span>
            <span className="font-body-sm text-body-sm text-on-surface-variant">/ All Time</span>
          </div>
          <div className="mt-2 font-telemetry-sm text-telemetry-sm text-on-surface-variant">
            Persistent WAL SQLite Log
          </div>
        </div>

        {/* Critical Alarms */}
        <div className="p-4 rounded-xl bg-surface-container-low border border-error/30 flex flex-col justify-between">
          <div className="flex items-center justify-between text-error font-label-caps text-label-caps uppercase">
            <span>Critical Limit State</span>
            <span className="material-symbols-outlined text-error text-[18px]">emergency</span>
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="font-telemetry-lg text-telemetry-lg font-bold text-error">
              {String(alertStats.critical || alerts.filter(a => a.severity === 'CRITICAL').length).padStart(2, '0')}
            </span>
            <span className="font-body-sm text-body-sm text-error/80">Active</span>
          </div>
          <div className="mt-2 font-telemetry-sm text-telemetry-sm text-error/80 flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-error animate-ping" />
            <span>Exceeded 1.5× τ UCL</span>
          </div>
        </div>

        {/* Warning Flares */}
        <div className="p-4 rounded-xl bg-surface-container-low border border-warning/30 flex flex-col justify-between">
          <div className="flex items-center justify-between text-warning font-label-caps text-label-caps uppercase">
            <span>Warning Flares</span>
            <span className="material-symbols-outlined text-warning text-[18px]">warning</span>
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="font-telemetry-lg text-telemetry-lg font-bold text-warning">
              {String(alertStats.warning || alerts.filter(a => a.severity === 'WARNING').length).padStart(2, '0')}
            </span>
            <span className="font-body-sm text-body-sm text-warning/80">Moderate</span>
          </div>
          <div className="mt-2 font-telemetry-sm text-telemetry-sm text-warning/80">
            Persistent Deviation Buffer
          </div>
        </div>

        {/* Unacknowledged */}
        <div className="p-4 rounded-xl bg-surface-container-low border border-primary/30 flex flex-col justify-between">
          <div className="flex items-center justify-between text-primary font-label-caps text-label-caps uppercase">
            <span>Action Required</span>
            <span className="material-symbols-outlined text-primary text-[18px]">pending_actions</span>
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="font-telemetry-lg text-telemetry-lg font-bold text-primary">
              {String(alertStats.unacknowledged || alerts.filter(a => !a.acknowledged).length).padStart(2, '0')}
            </span>
            <span className="font-body-sm text-body-sm text-primary/80">Unacknowledged</span>
          </div>
          <div className="mt-2 font-telemetry-sm text-telemetry-sm text-on-surface-variant">
            Operator Sign-off Pending
          </div>
        </div>
      </div>

      {/* ── Toolbar: Filter Tabs & Search ── */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-xl bg-surface-container-low border border-outline-variant/30">
        {/* Severity Filter Tabs */}
        <div className="flex items-center gap-1">
          {[
            { id: 'ALL', label: 'All Incidents' },
            { id: 'UNACK', label: 'Unacknowledged' },
            { id: 'CRITICAL', label: 'Critical Only' },
            { id: 'WARNING', label: 'Warnings Only' },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setFilterSeverity(tab.id)}
              className={`px-3 py-1.5 rounded-lg font-label-caps text-label-caps uppercase font-semibold cursor-pointer transition-all ${
                filterSeverity === tab.id
                  ? 'bg-primary text-on-primary shadow-sm'
                  : 'bg-surface-container text-on-surface-variant hover:text-on-surface'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Search */}
        <div className="flex items-center gap-2 min-w-[220px] px-3 py-1.5 rounded-lg bg-surface-container border border-outline-variant/40">
          <span className="material-symbols-outlined text-[16px] text-on-surface-variant">search</span>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search incident ID, asset..."
            className="w-full bg-transparent text-xs text-on-surface outline-none placeholder:text-on-surface-variant/60 font-body-sm"
          />
        </div>
      </div>

      {/* ── Incidents Table ── */}
      <div className="rounded-xl bg-surface-container-low border border-outline-variant/30 overflow-hidden shadow-sm">
        <table className="data-table">
          <thead>
            <tr>
              <th>Incident Ref</th>
              <th>Asset / Structure</th>
              <th>Trigger Channel</th>
              <th>Severity</th>
              <th>Peak MSE / Ratio</th>
              <th>Status</th>
              <th>Timestamp</th>
              <th className="text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length > 0 ? (
              filtered.map(alert => {
                const isCrit = alert.severity === 'CRITICAL';
                return (
                  <tr key={alert.id} className="hover:bg-surface-container/40">
                    {/* ID */}
                    <td className="font-telemetry-sm font-semibold font-mono text-primary">
                      {alert.id}
                    </td>

                    {/* Structure */}
                    <td>
                      <div className="font-medium text-on-surface">{alert.structure_id}</div>
                    </td>

                    {/* Channel */}
                    <td className="font-mono text-xs text-on-surface-variant">
                      {alert.channel_name ? alert.channel_name.replace('Sensor_', '').replace(/_/g, ' ') : 'System Global'}
                    </td>

                    {/* Severity Badge */}
                    <td>
                      <span className={`px-2 py-0.5 rounded font-label-caps text-[10px] font-semibold border ${isCrit ? 'badge-critical' : 'badge-warning'}`}>
                        {alert.severity}
                      </span>
                    </td>

                    {/* MSE */}
                    <td className="font-mono text-xs">
                      <span className="text-on-surface">{Number(alert.mse_value || 0).toFixed(5)}</span>
                      {alert.ratio && (
                        <span className="text-on-surface-variant text-[10px] ml-1.5 font-semibold">
                          ({Number(alert.ratio).toFixed(1)}× UCL)
                        </span>
                      )}
                    </td>

                    {/* Status */}
                    <td>
                      {alert.acknowledged ? (
                        <span className="inline-flex items-center gap-1 text-[11px] text-tertiary font-medium">
                          <span className="material-symbols-outlined text-[14px]">done_all</span>
                          Signed-off
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-[11px] text-error font-medium">
                          <span className="w-1.5 h-1.5 rounded-full bg-error animate-ping" />
                          Pending
                        </span>
                      )}
                    </td>

                    {/* Timestamp */}
                    <td className="font-telemetry-sm text-[11px] text-on-surface-variant font-mono">
                      {alert.timestamp ? new Date(alert.timestamp).toLocaleTimeString() : 'Recent'}
                    </td>

                    {/* Actions */}
                    <td className="text-right">
                      <div className="flex items-center justify-end gap-1.5">
                        <button
                          onClick={() => onOpenAlertDetail(alert)}
                          className="px-2.5 py-1 rounded bg-surface-container hover:bg-surface-container-high text-primary font-body-sm text-xs font-medium cursor-pointer transition-colors flex items-center gap-1"
                          title="View AI Diagnosis"
                        >
                          <span className="material-symbols-outlined text-[14px]">psychology</span>
                          <span>AI Diagnosis</span>
                        </button>

                        {!alert.acknowledged && (
                          <button
                            onClick={() => handleAck(alert.id)}
                            disabled={ackLoading === alert.id}
                            className="px-2.5 py-1 rounded bg-tertiary/10 hover:bg-tertiary/20 text-tertiary border border-tertiary/30 font-body-sm text-xs font-semibold cursor-pointer transition-colors"
                          >
                            {ackLoading === alert.id ? 'Saving...' : 'Acknowledge'}
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan={8} className="py-12 text-center text-on-surface-variant font-body-sm">
                  <div className="flex flex-col items-center gap-2 opacity-50">
                    <span className="material-symbols-outlined text-[36px]">verified</span>
                    <span>No active incident alerts matching current criteria. All nominal.</span>
                  </div>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

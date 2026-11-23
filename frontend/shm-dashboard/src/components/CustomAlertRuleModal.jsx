/**
 * CustomAlertRuleModal.jsx
 * Stitch Screen: "StructureIQ - Custom Alert Rule Modal" (118a0d09...)
 * Rule builder modal for defining custom structural limit states,
 * MSE threshold triggers, and automated telemetry actions.
 */

import React, { useState } from 'react';

export default function CustomAlertRuleModal({
  isOpen,
  onClose,
  onSaveRule,
  channels = ['Sensor_MidSpan_Girder', 'Sensor_Pier_Left', 'Sensor_Pier_Right', 'Sensor_Abutment_Left', 'Sensor_Ambient_Temp']
}) {
  const [ruleName, setRuleName] = useState('');
  const [channel, setChannel] = useState(channels[0] || 'Sensor_MidSpan_Girder');
  const [operator, setOperator] = useState('>');
  const [threshold, setThreshold] = useState('0.00350');
  const [persistence, setPersistence] = useState('5');
  const [severity, setSeverity] = useState('CRITICAL');
  const [action, setAction] = useState('DISPATCH_DEFCON');

  if (!isOpen) return null;

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!ruleName.trim()) {
      alert('Please enter a rule name.');
      return;
    }
    onSaveRule && onSaveRule({
      id: `RULE-${Date.now().toString().slice(-4)}`,
      name: ruleName,
      channel,
      condition: `${channel} ${operator} ${threshold} for ${persistence} frames`,
      severity,
      action,
      enabled: true,
    });
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
      <div className="w-full max-w-xl bg-surface-container-low rounded-2xl border border-outline-variant/40 shadow-2xl overflow-hidden flex flex-col animate-in fade-in zoom-in-95 duration-200">

        {/* ── Modal Header ── */}
        <div className="p-6 border-b border-outline-variant/30 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-primary/10 text-primary flex items-center justify-center">
              <span className="material-symbols-outlined text-[20px]">add_alert</span>
            </div>
            <div>
              <h2 className="font-headline-sm text-base font-bold text-on-surface">Add Custom Alert Rule</h2>
              <span className="font-label-caps text-[10px] text-on-surface-variant uppercase">Define Custom Limit State & Ingress Triggers</span>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-surface-container text-on-surface-variant hover:text-on-surface cursor-pointer"
          >
            <span className="material-symbols-outlined text-[20px]">close</span>
          </button>
        </div>

        {/* ── Modal Form ── */}
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {/* Rule Name */}
          <div>
            <label className="font-label-caps text-[10px] uppercase text-on-surface-variant block mb-1">Rule Name</label>
            <input
              type="text"
              required
              value={ruleName}
              onChange={(e) => setRuleName(e.target.value)}
              placeholder="e.g., Mid-Span Peak Deflection Exceedance"
              className="w-full px-3 py-2 rounded-lg bg-surface-container border border-outline-variant/40 text-on-surface font-body-sm text-xs outline-none focus:border-primary"
            />
          </div>

          {/* Channel & Severity */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="font-label-caps text-[10px] uppercase text-on-surface-variant block mb-1">Monitored Channel</label>
              <select
                value={channel}
                onChange={(e) => setChannel(e.target.value)}
                className="w-full px-3 py-2 rounded-lg bg-surface-container border border-outline-variant/40 text-on-surface font-body-sm text-xs outline-none focus:border-primary"
              >
                {channels.map(ch => (
                  <option key={ch} value={ch}>
                    {ch.replace('Sensor_', '').replace(/_/g, ' ')}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="font-label-caps text-[10px] uppercase text-on-surface-variant block mb-1">Trigger Severity</label>
              <select
                value={severity}
                onChange={(e) => setSeverity(e.target.value)}
                className="w-full px-3 py-2 rounded-lg bg-surface-container border border-outline-variant/40 text-on-surface font-body-sm text-xs outline-none focus:border-primary"
              >
                <option value="CRITICAL">CRITICAL (Defcon-2)</option>
                <option value="WARNING">WARNING (Flare)</option>
                <option value="INFO">INFORMATIONAL</option>
              </select>
            </div>
          </div>

          {/* Condition Builder */}
          <div className="p-3.5 rounded-xl bg-surface-container border border-outline-variant/20 space-y-3">
            <span className="font-label-caps text-[10px] uppercase text-primary font-semibold block">Condition Logic</span>

            <div className="grid grid-cols-3 gap-2">
              <div>
                <label className="font-label-caps text-[9px] uppercase text-on-surface-variant block mb-1">Operator</label>
                <select
                  value={operator}
                  onChange={(e) => setOperator(e.target.value)}
                  className="w-full px-2.5 py-1.5 rounded bg-surface-container-high border border-outline-variant/30 text-xs text-on-surface"
                >
                  <option value=">">Greater than (&gt;)</option>
                  <option value="<">Less than (&lt;)</option>
                  <option value=">=">Greater or Equal (&ge;)</option>
                </select>
              </div>

              <div>
                <label className="font-label-caps text-[9px] uppercase text-on-surface-variant block mb-1">Threshold (MSE)</label>
                <input
                  type="number"
                  step="0.0001"
                  value={threshold}
                  onChange={(e) => setThreshold(e.target.value)}
                  className="w-full px-2.5 py-1.5 rounded bg-surface-container-high border border-outline-variant/30 text-xs text-on-surface font-mono"
                />
              </div>

              <div>
                <label className="font-label-caps text-[9px] uppercase text-on-surface-variant block mb-1">Persistence</label>
                <input
                  type="number"
                  min="1"
                  max="100"
                  value={persistence}
                  onChange={(e) => setPersistence(e.target.value)}
                  className="w-full px-2.5 py-1.5 rounded bg-surface-container-high border border-outline-variant/30 text-xs text-on-surface font-mono"
                  placeholder="5 frames"
                />
              </div>
            </div>
          </div>

          {/* Automated Action */}
          <div>
            <label className="font-label-caps text-[10px] uppercase text-on-surface-variant block mb-1">Dispatched Action</label>
            <select
              value={action}
              onChange={(e) => setAction(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-surface-container border border-outline-variant/40 text-on-surface font-body-sm text-xs outline-none focus:border-primary"
            >
              <option value="DISPATCH_DEFCON">Trigger Incident & Defcon-2 Alarm</option>
              <option value="LOG_AUDIT">Log in Persistent SQLite WAL Audit</option>
              <option value="PAUSE_STREAM">Auto-Pause Telemetry Ingestion</option>
            </select>
          </div>

          {/* ── Footer ── */}
          <div className="pt-4 border-t border-outline-variant/30 flex items-center justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded bg-surface-container hover:bg-surface-container-high text-on-surface font-body-sm text-xs font-semibold cursor-pointer"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="btn-primary text-xs"
            >
              <span className="material-symbols-outlined text-[16px]">save</span>
              <span>Create Rule</span>
            </button>
          </div>
        </form>

      </div>
    </div>
  );
}

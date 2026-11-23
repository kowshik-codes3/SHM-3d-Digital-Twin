/**
 * SettingsPage.jsx
 * Stitch Screen: "StructureIQ - Settings & System Configuration" (9e45e7f1...)
 * Configuration panel for MQTT/Kafka brokers, PyTorch LSTM autoencoder hyperparameters,
 * alert rules engine, and sensor calibration baselines.
 */

import React, { useState } from 'react';
import CustomAlertRuleModal from './CustomAlertRuleModal';

export default function SettingsPage() {
  const [mqttBroker, setMqttBroker] = useState('mqtt://telemetry.shm-grid.internal:1883');
  const [kafkaTopic, setKafkaTopic] = useState('shm.civil.sensor-stream.v1');
  const [tlsEnabled, setTlsEnabled] = useState(true);
  const [sampleRate, setSampleRate] = useState('50');

  // Autoencoder Hyperparameters
  const [hiddenUnits, setHiddenUnits] = useState('64');
  const [latentDim, setLatentDim] = useState('16');
  const [sequenceLength, setSequenceLength] = useState('100');
  const [uclMultiplier, setUclMultiplier] = useState('1.5');

  // Alert Rules
  const [rules, setRules] = useState([
    { id: 'RULE-01', name: 'Mid-Span Critical Deflection', channel: 'Sensor_MidSpan_Girder', condition: 'MSE > 0.0035 for 5 frames', severity: 'CRITICAL', enabled: true },
    { id: 'RULE-02', name: 'Pier Foundation Subsidence', channel: 'Sensor_Pier_Left', condition: 'MSE > 0.0028 for 8 frames', severity: 'CRITICAL', enabled: true },
    { id: 'RULE-03', name: 'Abutment Thermal Drift', channel: 'Sensor_Abutment_Left', condition: 'MSE > 0.0020 for 15 frames', severity: 'WARNING', enabled: false },
  ]);

  const [ruleModalOpen, setRuleModalOpen] = useState(false);
  const [saveToast, setSaveToast] = useState(false);

  const handleSaveRule = (newRule) => {
    setRules([...rules, newRule]);
  };

  const toggleRule = (ruleId) => {
    setRules(rules.map(r => r.id === ruleId ? { ...r, enabled: !r.enabled } : r));
  };

  const deleteRule = (ruleId) => {
    setRules(rules.filter(r => r.id !== ruleId));
  };

  const handleSaveConfig = () => {
    setSaveToast(true);
    setTimeout(() => setSaveToast(false), 3000);
  };

  return (
    <div className="h-full overflow-y-auto p-6 max-w-[1720px] mx-auto w-full space-y-6">

      {/* ── Page Header ── */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-primary font-label-caps text-label-caps uppercase tracking-wider mb-1">
            <span className="material-symbols-outlined text-[16px]">settings</span>
            <span>Platform Topology & Engine Controls</span>
          </div>
          <h1 className="font-headline-lg text-headline-lg tracking-tight text-on-surface">System Configuration</h1>
          <p className="font-body-sm text-body-sm text-on-surface-variant mt-0.5">
            Manage real-time telemetry brokers, PyTorch autoencoder hyperparameters, and alert triage rules
          </p>
        </div>

        <button
          onClick={handleSaveConfig}
          className="btn-primary"
        >
          <span className="material-symbols-outlined text-[18px]">save</span>
          <span>Save Configuration</span>
        </button>
      </div>

      {saveToast && (
        <div className="p-3 rounded-lg bg-tertiary/15 border border-tertiary/30 text-tertiary font-body-sm text-xs flex items-center gap-2">
          <span className="material-symbols-outlined text-[16px]">check_circle</span>
          <span>Configuration saved successfully. PyTorch inference runtime updated.</span>
        </div>
      )}

      {/* ── Two Column Configuration Layout ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* Column 1: Telemetry Ingress & Brokers */}
        <div className="space-y-6">
          <div className="p-5 rounded-xl bg-surface-container-low border border-outline-variant/30 space-y-4">
            <div className="flex items-center gap-2 pb-2 border-b border-outline-variant/20">
              <span className="material-symbols-outlined text-[20px] text-primary">router</span>
              <h2 className="font-headline-sm text-sm font-semibold text-on-surface">Telemetry Ingress & Edge Brokers</h2>
            </div>

            <div>
              <label className="font-label-caps text-[10px] uppercase text-on-surface-variant block mb-1">MQTT Gateway Endpoint</label>
              <input
                type="text"
                value={mqttBroker}
                onChange={(e) => setMqttBroker(e.target.value)}
                className="w-full px-3 py-2 rounded-lg bg-surface-container border border-outline-variant/40 text-on-surface font-telemetry-sm text-xs outline-none focus:border-primary"
              />
            </div>

            <div>
              <label className="font-label-caps text-[10px] uppercase text-on-surface-variant block mb-1">Kafka Stream Topic</label>
              <input
                type="text"
                value={kafkaTopic}
                onChange={(e) => setKafkaTopic(e.target.value)}
                className="w-full px-3 py-2 rounded-lg bg-surface-container border border-outline-variant/40 text-on-surface font-telemetry-sm text-xs outline-none focus:border-primary"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="font-label-caps text-[10px] uppercase text-on-surface-variant block mb-1">Sampling Rate (Hz)</label>
                <input
                  type="number"
                  value={sampleRate}
                  onChange={(e) => setSampleRate(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-surface-container border border-outline-variant/40 text-on-surface font-telemetry-sm text-xs outline-none focus:border-primary"
                />
              </div>

              <div className="flex flex-col justify-end">
                <label className="flex items-center gap-2 p-2 rounded-lg bg-surface-container cursor-pointer">
                  <input
                    type="checkbox"
                    checked={tlsEnabled}
                    onChange={(e) => setTlsEnabled(e.target.checked)}
                    className="rounded text-primary"
                  />
                  <span className="font-body-sm text-xs text-on-surface">Enforce mTLS v1.3</span>
                </label>
              </div>
            </div>
          </div>

          {/* PyTorch Autoencoder Hyperparameters */}
          <div className="p-5 rounded-xl bg-surface-container-low border border-outline-variant/30 space-y-4">
            <div className="flex items-center gap-2 pb-2 border-b border-outline-variant/20">
              <span className="material-symbols-outlined text-[20px] text-tertiary">neurology</span>
              <h2 className="font-headline-sm text-sm font-semibold text-on-surface">PyTorch LSTM Anomaly Model Tuning</h2>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="font-label-caps text-[10px] uppercase text-on-surface-variant block mb-1">Hidden Units</label>
                <input
                  type="number"
                  value={hiddenUnits}
                  onChange={(e) => setHiddenUnits(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-surface-container border border-outline-variant/40 text-on-surface font-telemetry-sm text-xs outline-none"
                />
              </div>

              <div>
                <label className="font-label-caps text-[10px] uppercase text-on-surface-variant block mb-1">Latent Vector Dimension</label>
                <input
                  type="number"
                  value={latentDim}
                  onChange={(e) => setLatentDim(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-surface-container border border-outline-variant/40 text-on-surface font-telemetry-sm text-xs outline-none"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="font-label-caps text-[10px] uppercase text-on-surface-variant block mb-1">Sliding Window Length</label>
                <input
                  type="number"
                  value={sequenceLength}
                  onChange={(e) => setSequenceLength(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-surface-container border border-outline-variant/40 text-on-surface font-telemetry-sm text-xs outline-none"
                />
              </div>

              <div>
                <label className="font-label-caps text-[10px] uppercase text-on-surface-variant block mb-1">UCL Alpha Multiplier (τ)</label>
                <input
                  type="number"
                  step="0.1"
                  value={uclMultiplier}
                  onChange={(e) => setUclMultiplier(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-surface-container border border-outline-variant/40 text-on-surface font-telemetry-sm text-xs outline-none"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Column 2: Alert Rules Engine Table */}
        <div className="p-5 rounded-xl bg-surface-container-low border border-outline-variant/30 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-outline-variant/20 mb-4">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-[20px] text-warning">rule</span>
                <h2 className="font-headline-sm text-sm font-semibold text-on-surface">Active Alert Rules Engine</h2>
              </div>

              <button
                onClick={() => setRuleModalOpen(true)}
                className="btn-ghost-cyan text-xs py-1"
              >
                <span className="material-symbols-outlined text-[16px]">add</span>
                <span>Add Custom Rule</span>
              </button>
            </div>

            {/* Rules list */}
            <div className="space-y-3">
              {rules.map(rule => (
                <div
                  key={rule.id}
                  className="p-3.5 rounded-lg bg-surface-container border border-outline-variant/20 flex items-center justify-between gap-3"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`px-1.5 py-0.2 rounded font-label-caps text-[9px] font-bold ${rule.severity === 'CRITICAL' ? 'badge-critical' : 'badge-warning'}`}>
                        {rule.severity}
                      </span>
                      <h4 className="font-body-sm text-xs font-semibold text-on-surface truncate">
                        {rule.name}
                      </h4>
                    </div>
                    <div className="font-telemetry-sm text-[11px] text-on-surface-variant font-mono truncate">
                      {rule.condition}
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => toggleRule(rule.id)}
                      className={`px-2 py-1 rounded font-label-caps text-[10px] cursor-pointer ${rule.enabled ? 'bg-tertiary/20 text-tertiary' : 'bg-surface-container-highest text-outline'}`}
                    >
                      {rule.enabled ? 'ACTIVE' : 'DISABLED'}
                    </button>
                    <button
                      onClick={() => deleteRule(rule.id)}
                      className="p-1 rounded text-outline hover:text-error cursor-pointer"
                    >
                      <span className="material-symbols-outlined text-[16px]">delete</span>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="pt-4 border-t border-outline-variant/20 text-on-surface-variant font-body-sm text-[11px]">
            Rules evaluate dynamically against incoming WebSocket telemetry packets before dispatching to the incident audit WAL.
          </div>
        </div>

      </div>

      {/* Modal */}
      <CustomAlertRuleModal
        isOpen={ruleModalOpen}
        onClose={() => setRuleModalOpen(false)}
        onSaveRule={handleSaveRule}
      />
    </div>
  );
}

/**
 * AlertDetailModal.jsx
 * Stitch Screen: "StructureIQ - Alert Detail & AI Diagnosis (#EV-4092)" (532a00a4...)
 * Detailed incident dossier with Defcon-2 banner, physics-informed AI diagnosis,
 * strain divergence analysis, and emergency mitigation protocols.
 */

import React from 'react';

export default function AlertDetailModal({
  alert,
  onClose,
  onAcknowledge
}) {
  if (!alert) return null;

  const isCritical = alert.severity === 'CRITICAL';
  const alertId = alert.id || '#EV-4092';
  const channel = alert.channel_name ? alert.channel_name.replace('Sensor_', '').replace(/_/g, ' ') : 'MidSpan Girder';
  const ratio = alert.ratio ? Number(alert.ratio).toFixed(2) : '3.82';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
      <div className="w-full max-w-4xl max-h-[90vh] bg-surface-container-low rounded-2xl border border-outline-variant/40 shadow-2xl overflow-hidden flex flex-col animate-in fade-in zoom-in-95 duration-200">

        {/* ── Header Banner (Defcon) ── */}
        <div className={`p-6 border-b relative overflow-hidden ${isCritical ? 'bg-error-container/20 border-error/30' : 'bg-warning/10 border-warning/30'}`}>
          <div className="flex items-center justify-between gap-4 mb-2">
            <div className="flex items-center gap-2">
              <span className={`px-2.5 py-0.5 rounded font-label-caps text-label-caps uppercase font-bold flex items-center gap-1.5 ${isCritical ? 'badge-critical' : 'badge-warning'}`}>
                <span className="material-symbols-outlined text-[14px]">crisis_alert</span>
                <span>{isCritical ? 'CRITICAL DEFCON-2 RECORD' : 'WARNING FLARE RECORD'}</span>
              </span>
              <span className="font-telemetry-sm text-telemetry-sm text-primary font-mono bg-surface-container-high px-2 py-0.5 rounded">
                {alertId}
              </span>
            </div>

            <button
              onClick={onClose}
              className="p-1 rounded hover:bg-surface-container text-on-surface-variant hover:text-on-surface cursor-pointer"
            >
              <span className="material-symbols-outlined text-[20px]">close</span>
            </button>
          </div>

          <h2 className="font-headline-lg text-lg sm:text-xl font-bold text-on-surface tracking-tight">
            {channel} Structural Anomaly Divergence
          </h2>

          <div className="flex flex-wrap items-center gap-x-6 gap-y-1 mt-2 font-body-sm text-xs text-on-surface-variant">
            <span className="flex items-center gap-1 text-on-surface">
              <span className="material-symbols-outlined text-[16px] text-primary">pin_drop</span>
              <span>Asset: {alert.structure_id}</span>
            </span>
            <span className="flex items-center gap-1">
              <span className="material-symbols-outlined text-[16px] text-error">alarm</span>
              <span>Timestamp: {alert.timestamp ? new Date(alert.timestamp).toUTCString() : 'Active Alert'}</span>
            </span>
            <span className="flex items-center gap-1 text-error font-semibold">
              <span className="material-symbols-outlined text-[16px]">trending_up</span>
              <span>Residual Ratio: {ratio}× UCL</span>
            </span>
          </div>
        </div>

        {/* ── Scrollable Dossier Body ── */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">

          {/* AI Diagnosis Card */}
          <div className="p-5 rounded-xl bg-surface-container border border-primary/20 relative overflow-hidden">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <div className="w-7 h-7 rounded bg-primary/10 text-primary flex items-center justify-center">
                  <span className="material-symbols-outlined text-[18px]">psychology</span>
                </div>
                <div>
                  <h3 className="font-headline-sm text-sm font-semibold text-on-surface">Physics-Informed ML Diagnosis</h3>
                  <span className="font-label-caps text-[10px] text-on-surface-variant uppercase">Bi-directional LSTM Autoencoder Analysis</span>
                </div>
              </div>
              <span className="font-telemetry-sm text-xs text-tertiary bg-tertiary/10 px-2 py-0.5 rounded font-mono">
                98.4% Confidence
              </span>
            </div>

            <p className="font-body-sm text-xs text-on-surface leading-relaxed mb-3">
              Telemetry residuals indicate localized stiffness degradation. Structural energy redistribution detected across adjacent spatial coordinates via Gaussian graph diffusion. Reconstructed MSE exceeds safe operational baseline by {ratio}×.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 p-3 rounded-lg bg-surface-container-lowest/80 text-xs">
              <div>
                <span className="font-label-caps text-[9px] text-on-surface-variant uppercase block">Anomaly Type</span>
                <span className="font-medium text-error flex items-center gap-1 mt-0.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-error" />
                  Stiffness Degradation
                </span>
              </div>
              <div>
                <span className="font-label-caps text-[9px] text-on-surface-variant uppercase block">Spatial Localization</span>
                <span className="font-medium text-primary mt-0.5 block">{channel}</span>
              </div>
              <div>
                <span className="font-label-caps text-[9px] text-on-surface-variant uppercase block">Thermal Decoupling</span>
                <span className="font-medium text-tertiary mt-0.5 block">Baseline Compensated</span>
              </div>
            </div>
          </div>

          {/* Recommended Mitigation Protocols */}
          <div className="p-5 rounded-xl bg-surface-container border border-outline-variant/30">
            <div className="flex items-center gap-2 mb-3">
              <span className="material-symbols-outlined text-[18px] text-warning">rule</span>
              <h3 className="font-headline-sm text-sm font-semibold text-on-surface">Recommended Field Protocols</h3>
            </div>

            <div className="space-y-2.5">
              <div className="flex items-start gap-3 p-3 rounded-lg bg-surface-container-low border border-outline-variant/20">
                <span className="w-5 h-5 rounded-full bg-error/20 text-error flex items-center justify-center font-bold text-xs flex-shrink-0">1</span>
                <div>
                  <h4 className="font-body-sm text-xs font-semibold text-on-surface">Dispatch Emergency NDT Ultrasonic Inspection</h4>
                  <p className="font-body-sm text-[11px] text-on-surface-variant mt-0.5">
                    Target physical inspection within 2 meters of {channel} coordinates. Verify weld joints and bearing seats.
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3 p-3 rounded-lg bg-surface-container-low border border-outline-variant/20">
                <span className="w-5 h-5 rounded-full bg-warning/20 text-warning flex items-center justify-center font-bold text-xs flex-shrink-0">2</span>
                <div>
                  <h4 className="font-body-sm text-xs font-semibold text-on-surface">Implement Dynamic Load Restriction</h4>
                  <p className="font-body-sm text-[11px] text-on-surface-variant mt-0.5">
                    Divert heavy freight transit over span until stress distribution reaches stable baseline equilibrium.
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3 p-3 rounded-lg bg-surface-container-low border border-outline-variant/20">
                <span className="w-5 h-5 rounded-full bg-primary/20 text-primary flex items-center justify-center font-bold text-xs flex-shrink-0">3</span>
                <div>
                  <h4 className="font-body-sm text-xs font-semibold text-on-surface">Increase Telemetry DAQ Ingress Rate</h4>
                  <p className="font-body-sm text-[11px] text-on-surface-variant mt-0.5">
                    Stream at full 240 Hz without temporal batching to capture transient high-frequency micro-cracking events.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ── Footer Actions ── */}
        <div className="p-4 border-t border-outline-variant/30 bg-surface-container-low flex items-center justify-between gap-3">
          <button
            onClick={() => {
              alert(`Exporting Incident Dossier for ${alertId} as PDF/CSV.`);
            }}
            className="btn-ghost-cyan text-xs"
          >
            <span className="material-symbols-outlined text-[16px]">download</span>
            <span>Export Dossier (CSV/PDF)</span>
          </button>

          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="px-4 py-2 rounded bg-surface-container hover:bg-surface-container-high text-on-surface font-body-sm text-xs font-semibold cursor-pointer"
            >
              Dismiss
            </button>

            {!alert.acknowledged && (
              <button
                onClick={() => {
                  onAcknowledge && onAcknowledge(alert.id);
                  onClose();
                }}
                className="btn-primary text-xs"
              >
                <span className="material-symbols-outlined text-[16px]">verified</span>
                <span>Sign-off & Acknowledge</span>
              </button>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}

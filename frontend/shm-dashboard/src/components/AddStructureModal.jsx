/**
 * AddStructureModal.jsx
 * Add Structure Wizard matching Stitch UI design (Screen f75c8ff5...).
 * Interactive modal to commission new infrastructure assets, configure 3D sensor nodes,
 * and select ingestion protocols (Simulator, MQTT, Kafka, HTTP).
 */

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Check, Cpu, HardHat, Layers, Plus, Radio, Trash2, X, Zap
} from 'lucide-react';
import { API_BASE_URL } from '../config';

const STRUCTURE_TYPES = [
  { value: 'BRIDGE', label: 'Suspension / Girder Bridge', icon: '🌉' },
  { value: 'BUILDING', label: 'Commercial High-Rise', icon: '🏢' },
  { value: 'DAM', label: 'Hydraulic Gravity Dam', icon: '🌊' },
  { value: 'TOWER', label: 'Lattice Communications Tower', icon: '🗼' },
];

const INGESTION_TYPES = [
  {
    value: 'simulator',
    label: 'Synthetic Vibration Simulator',
    desc: 'Generates multi-harmonic structural noise, traffic transients & thermal drift.',
    defaultParams: { noise_level: 0.05, base_temp: 24.0 },
  },
  {
    value: 'http',
    label: 'HTTP Direct Push API',
    desc: 'Receives POST JSON telemetry frames from edge DAQ / field gateways.',
    defaultParams: { port: 8000 },
  },
  {
    value: 'mqtt',
    label: 'MQTT Industrial Broker',
    desc: 'Subscribes to sensor telemetry topics over paho-mqtt QoS 0/1.',
    defaultParams: { broker_host: 'localhost', broker_port: 1883, topic: 'shm/telemetry' },
  },
  {
    value: 'kafka',
    label: 'Apache Kafka Event Stream',
    desc: 'High-throughput partitioned event consumer with automated offset tracking.',
    defaultParams: { bootstrap_servers: 'localhost:9092', topic: 'shm.telemetry.v1' },
  },
];

export default function AddStructureModal({ isOpen, onClose, onCreated }) {
  const [step, setStep] = useState(1);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  // Form state
  const [structureId, setStructureId] = useState('');
  const [name, setName] = useState('');
  const [structureType, setStructureType] = useState('BRIDGE');
  const [fs, setFs] = useState(50);
  const [ingestionType, setIngestionType] = useState('simulator');
  const [ingestionParams, setIngestionParams] = useState({ noise_level: 0.05, base_temp: 24.0 });

  const [sensors, setSensors] = useState([
    { channel_name: 'Sensor_North_Pier', sensor_type: 'ACCELEROMETER', coordinates_3d: [-10.0, 0.0, 0.0], threshold_tau: 0.04 },
    { channel_name: 'Sensor_Center_Arch', sensor_type: 'ACCELEROMETER', coordinates_3d: [0.0, 15.0, 0.0], threshold_tau: 0.04 },
    { channel_name: 'Sensor_South_Pier', sensor_type: 'ACCELEROMETER', coordinates_3d: [10.0, 0.0, 0.0], threshold_tau: 0.04 },
  ]);

  if (!isOpen) return null;

  const handleAddSensor = () => {
    const idx = sensors.length + 1;
    setSensors([
      ...sensors,
      {
        channel_name: `Sensor_Node_${idx}`,
        sensor_type: 'ACCELEROMETER',
        coordinates_3d: [idx * 5.0, 0.0, 0.0],
        threshold_tau: 0.04,
      },
    ]);
  };

  const handleRemoveSensor = (index) => {
    if (sensors.length <= 1) {
      alert('A structure must have at least 1 sensor node.');
      return;
    }
    setSensors(sensors.filter((_, i) => i !== index));
  };

  const handleSensorChange = (index, field, value) => {
    const updated = [...sensors];
    updated[index][field] = value;
    setSensors(updated);
  };

  const handleCoordinateChange = (index, axisIdx, value) => {
    const updated = [...sensors];
    const coords = [...updated[index].coordinates_3d];
    coords[axisIdx] = parseFloat(value) || 0;
    updated[index].coordinates_3d = coords;
    setSensors(updated);
  };

  const handleSelectIngestion = (typeObj) => {
    setIngestionType(typeObj.value);
    setIngestionParams(typeObj.defaultParams);
  };

  const handleSubmit = async () => {
    setError(null);
    setSubmitting(true);

    const payload = {
      structure_id: structureId.trim(),
      name: name.trim() || structureId.trim(),
      structure_type: structureType,
      fs: Number(fs) || 50,
      sensors: sensors.map(s => ({
        channel_name: s.channel_name.trim(),
        sensor_type: s.sensor_type,
        coordinates_3d: s.coordinates_3d,
        threshold_tau: s.threshold_tau || 0.04,
      })),
      ingestion: {
        type: ingestionType,
        params: ingestionParams,
      },
    };

    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/structures`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || 'Failed to register structure');
      }

      const created = await res.json();
      onCreated(created.structure_id);
      onClose();
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-md">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className="w-full max-w-2xl bg-[#0c1220] border border-white/10 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/5 bg-[#101728]">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-cyan-500/20 text-cyan-400 flex items-center justify-center border border-cyan-500/30">
              <Plus size={18} />
            </div>
            <div>
              <h2 className="text-sm font-bold text-white">Commission New Digital Twin</h2>
              <p className="text-[11px] text-slate-400">Step {step} of 3: {step === 1 ? 'Asset Metadata' : step === 2 ? 'Sensor Placement' : 'Ingestion Pipeline'}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-colors cursor-pointer"
          >
            <X size={18} />
          </button>
        </div>

        {/* Stepper Progress Bar */}
        <div className="grid grid-cols-3 border-b border-white/5 text-center text-[11px] font-medium bg-black/20">
          <div className={`py-2 border-b-2 transition-colors ${step === 1 ? 'border-cyan-400 text-cyan-300 font-semibold' : 'border-transparent text-slate-500'}`}>
            1. Asset Metadata
          </div>
          <div className={`py-2 border-b-2 transition-colors ${step === 2 ? 'border-cyan-400 text-cyan-300 font-semibold' : 'border-transparent text-slate-500'}`}>
            2. Sensor Topology
          </div>
          <div className={`py-2 border-b-2 transition-colors ${step === 3 ? 'border-cyan-400 text-cyan-300 font-semibold' : 'border-transparent text-slate-500'}`}>
            3. Ingestion & Protocol
          </div>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto flex-1">
          {error && (
            <div className="mb-4 p-3 rounded-lg bg-red-950/40 border border-red-500/40 text-red-300 text-xs flex items-center gap-2">
              <Zap size={14} className="shrink-0 text-red-400" />
              <span>{error}</span>
            </div>
          )}

          {/* STEP 1: Metadata */}
          {step === 1 && (
            <div className="flex flex-col gap-4">
              <div>
                <label className="text-xs font-semibold text-slate-300 block mb-1">
                  Structure Unique Identifier <span className="text-red-400">*</span>
                </label>
                <input
                  type="text"
                  placeholder="e.g. dam_oroville_01"
                  value={structureId}
                  onChange={(e) => setStructureId(e.target.value.toLowerCase().replace(/\s+/g, '_'))}
                  className="w-full px-3 py-2 rounded-lg bg-white/[0.04] border border-white/10 text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
                />
                <span className="text-[10px] text-slate-500 mt-1 block">Used for routing WebSocket streams & REST endpoints.</span>
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-300 block mb-1">
                  Display Asset Name
                </label>
                <input
                  type="text"
                  placeholder="e.g. Oroville Spillway Section A"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-white/[0.04] border border-white/10 text-xs text-white focus:outline-none focus:border-cyan-500"
                />
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-300 block mb-2">
                  Structural Typology
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {STRUCTURE_TYPES.map((t) => (
                    <button
                      key={t.value}
                      type="button"
                      onClick={() => setStructureType(t.value)}
                      className={`p-3 rounded-xl border text-left flex items-center gap-3 transition-all cursor-pointer ${
                        structureType === t.value
                          ? 'border-cyan-500/60 bg-cyan-950/20 text-white'
                          : 'border-white/5 bg-white/[0.02] text-slate-400 hover:border-white/10'
                      }`}
                    >
                      <span className="text-2xl">{t.icon}</span>
                      <div>
                        <div className="text-xs font-semibold">{t.value}</div>
                        <div className="text-[10px] text-slate-500">{t.label}</div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-300 block mb-1">
                  Sampling Frequency (Hz)
                </label>
                <input
                  type="number"
                  value={fs}
                  onChange={(e) => setFs(parseInt(e.target.value) || 50)}
                  className="w-full px-3 py-2 rounded-lg bg-white/[0.04] border border-white/10 text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
                />
              </div>
            </div>
          )}

          {/* STEP 2: Sensors */}
          {step === 2 && (
            <div className="flex flex-col gap-3">
              <div className="flex items-center justify-between mb-1">
                <div>
                  <h3 className="text-xs font-bold text-white uppercase tracking-wider">
                    Instrumented Channels ({sensors.length})
                  </h3>
                  <p className="text-[11px] text-slate-400">Specify DAQ channels and physical 3D telemetry spatial coordinates.</p>
                </div>
                <button
                  type="button"
                  onClick={handleAddSensor}
                  className="flex items-center gap-1 px-2.5 py-1 rounded bg-cyan-500/20 text-cyan-300 text-xs font-semibold border border-cyan-500/30 hover:bg-cyan-500/30 cursor-pointer"
                >
                  <Plus size={12} /> Add Sensor
                </button>
              </div>

              <div className="flex flex-col gap-2 max-h-72 overflow-y-auto pr-1">
                {sensors.map((s, idx) => (
                  <div
                    key={idx}
                    className="p-3 rounded-xl bg-white/[0.02] border border-white/5 flex flex-col gap-2"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex-1">
                        <label className="text-[10px] text-slate-500 uppercase font-mono block">Channel ID</label>
                        <input
                          type="text"
                          value={s.channel_name}
                          onChange={(e) => handleSensorChange(idx, 'channel_name', e.target.value)}
                          className="w-full px-2 py-1 rounded bg-white/[0.04] border border-white/10 text-xs text-white font-mono"
                        />
                      </div>
                      <div className="w-36">
                        <label className="text-[10px] text-slate-500 uppercase font-mono block">Type</label>
                        <select
                          value={s.sensor_type}
                          onChange={(e) => handleSensorChange(idx, 'sensor_type', e.target.value)}
                          className="w-full px-2 py-1 rounded bg-[#101728] border border-white/10 text-xs text-white"
                        >
                          <option value="ACCELEROMETER">Accelerometer</option>
                          <option value="STRAIN_GAUGE">Strain Gauge</option>
                          <option value="INCLINOMETER">Inclinometer</option>
                          <option value="TEMPERATURE">Temperature</option>
                        </select>
                      </div>
                      <button
                        type="button"
                        onClick={() => handleRemoveSensor(idx)}
                        className="p-1.5 mt-3.5 rounded text-red-400 hover:bg-red-950/30 cursor-pointer"
                      >
                        <Trash2 size={13} />
                      </button>
                    </div>

                    <div className="grid grid-cols-3 gap-2">
                      <div>
                        <label className="text-[9px] text-slate-500 uppercase font-mono block">X (m)</label>
                        <input
                          type="number"
                          step="0.5"
                          value={s.coordinates_3d[0]}
                          onChange={(e) => handleCoordinateChange(idx, 0, e.target.value)}
                          className="w-full px-2 py-1 rounded bg-white/[0.03] border border-white/10 text-xs text-white font-mono"
                        />
                      </div>
                      <div>
                        <label className="text-[9px] text-slate-500 uppercase font-mono block">Y (m)</label>
                        <input
                          type="number"
                          step="0.5"
                          value={s.coordinates_3d[1]}
                          onChange={(e) => handleCoordinateChange(idx, 1, e.target.value)}
                          className="w-full px-2 py-1 rounded bg-white/[0.03] border border-white/10 text-xs text-white font-mono"
                        />
                      </div>
                      <div>
                        <label className="text-[9px] text-slate-500 uppercase font-mono block">Z (m)</label>
                        <input
                          type="number"
                          step="0.5"
                          value={s.coordinates_3d[2]}
                          onChange={(e) => handleCoordinateChange(idx, 2, e.target.value)}
                          className="w-full px-2 py-1 rounded bg-white/[0.03] border border-white/10 text-xs text-white font-mono"
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* STEP 3: Ingestion */}
          {step === 3 && (
            <div className="flex flex-col gap-4">
              <div>
                <label className="text-xs font-semibold text-slate-300 block mb-2">
                  Select Ingestion Adapter
                </label>
                <div className="grid grid-cols-1 gap-2">
                  {INGESTION_TYPES.map((t) => (
                    <button
                      key={t.value}
                      type="button"
                      onClick={() => handleSelectIngestion(t)}
                      className={`p-3 rounded-xl border text-left flex items-start justify-between transition-all cursor-pointer ${
                        ingestionType === t.value
                          ? 'border-cyan-500/60 bg-cyan-950/20 text-white'
                          : 'border-white/5 bg-white/[0.02] text-slate-400 hover:border-white/10'
                      }`}
                    >
                      <div>
                        <div className="text-xs font-semibold flex items-center gap-2">
                          <Cpu size={14} className={ingestionType === t.value ? 'text-cyan-400' : ''} />
                          <span>{t.label}</span>
                        </div>
                        <p className="text-[11px] text-slate-400 mt-1">{t.desc}</p>
                      </div>
                      {ingestionType === t.value && (
                        <span className="w-5 h-5 rounded-full bg-cyan-400 text-black flex items-center justify-center shrink-0 mt-0.5">
                          <Check size={12} />
                        </span>
                      )}
                    </button>
                  ))}
                </div>
              </div>

              {/* Ingestion Parameters config */}
              <div className="p-3 rounded-xl bg-white/[0.02] border border-white/5">
                <span className="text-[10px] text-slate-500 font-mono uppercase tracking-wider block mb-2">
                  Adapter Configuration Parameters
                </span>
                <pre className="text-[11px] font-mono text-cyan-300 bg-black/40 p-2.5 rounded-lg border border-white/5 overflow-x-auto">
                  {JSON.stringify(ingestionParams, null, 2)}
                </pre>
                <p className="text-[10px] text-slate-500 mt-2">
                  * Dynamic auto-synthesis will instantly initialize baseline PyTorch LSTM weights, scaler, and UCL thresholds.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-white/5 bg-[#101728]">
          {step > 1 ? (
            <button
              type="button"
              onClick={() => setStep(step - 1)}
              className="px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-xs font-semibold text-slate-300 cursor-pointer transition-colors"
            >
              Back
            </button>
          ) : (
            <div />
          )}

          {step < 3 ? (
            <button
              type="button"
              disabled={step === 1 && !structureId.trim()}
              onClick={() => setStep(step + 1)}
              className="px-5 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white text-xs font-semibold shadow-md shadow-cyan-600/20 cursor-pointer transition-all"
            >
              Next Step
            </button>
          ) : (
            <button
              type="button"
              disabled={submitting}
              onClick={handleSubmit}
              className="px-6 py-2 rounded-lg bg-gradient-to-r from-emerald-600 to-cyan-600 hover:from-emerald-500 hover:to-cyan-500 text-white text-xs font-bold shadow-lg shadow-emerald-600/20 cursor-pointer transition-all flex items-center gap-2"
            >
              {submitting ? 'Commissioning Asset...' : 'Commission Asset Twin'}
            </button>
          )}
        </div>
      </motion.div>
    </div>
  );
}

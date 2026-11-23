/**
 * SensorPlacementEditor.jsx
 * StructureIQ - Interactive 3D Spatial Sensor Placement Editor.
 * Matches Stitch UI Design:
 * - 3D Bridge spatial canvas with real-time orbit controls and camera view projections (Top, Front, Side, Isometric).
 * - Interactive 3D sensor nodes placed directly on physical bridge coordinates (X, Y, Z meters).
 * - Real-time 3D coordinate modification, deck snapping, node deletion, and new channel addition.
 * - Saves directly to backend PUT /api/v1/structures/{id}/sensors with instant 3D model synchronization.
 */

import { useState, useRef, useMemo, useCallback, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Text } from '@react-three/drei';
import * as THREE from 'three';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Box, Check, Crosshair, Edit3, Layers, Maximize2,
  Move, Plus, RotateCcw, Save, Settings, Trash2, X, Zap
} from 'lucide-react';
import StructureGLBModel from './StructureGLBModel';
import { API_BASE_URL } from '../config';

const SENSOR_TYPES = ['ACCELEROMETER', 'STRAIN_GAUGE', 'INCLINOMETER', 'TEMPERATURE'];

const TYPE_COLORS = {
  ACCELEROMETER: '#22d3ee',
  STRAIN_GAUGE: '#3b82f6',
  INCLINOMETER: '#f59e0b',
  TEMPERATURE: '#10b981',
};

const CAMERA_PRESETS = {
  top:   { pos: [0, 32, 0.001], target: [0, 4, 0] },
  front: { pos: [0, 6, 28], target: [0, 4, 0] },
  side:  { pos: [30, 6, 0], target: [0, 4, 0] },
  iso:   { pos: [18, 16, 18], target: [0, 4, 0] },
};

// ─── 3D Bridge Geometry (matches main canvas) ──────────────────────────────────

function EditorBridgeGeometry({ onDeckClick }) {
  const concreteMat = (
    <meshStandardMaterial color="#374151" roughness={0.85} metalness={0.05} />
  );
  const steelMat = (
    <meshStandardMaterial color="#4b5563" roughness={0.4} metalness={0.7} />
  );

  return (
    <group>
      {/* Bridge Deck - Clickable to position sensor */}
      <mesh
        position={[0, 4.0, 0]}
        castShadow
        receiveShadow
        onClick={(e) => {
          e.stopPropagation();
          if (onDeckClick) {
            onDeckClick([
              Math.round(e.point.x * 2) / 2,
              Math.round(e.point.y * 10) / 10,
              Math.round(e.point.z * 2) / 2
            ]);
          }
        }}
      >
        <boxGeometry args={[28, 0.4, 5]} />
        {concreteMat}
      </mesh>

      {/* Road surface */}
      <mesh position={[0, 4.22, 0]} receiveShadow>
        <boxGeometry args={[27.5, 0.05, 3.5]} />
        <meshStandardMaterial color="#1f2937" roughness={0.95} />
      </mesh>

      {/* Lane markings */}
      {[-4, 0, 4].map((x, i) => (
        <mesh key={i} position={[x, 4.28, 0]}>
          <boxGeometry args={[4, 0.01, 0.15]} />
          <meshStandardMaterial color="#d1d5db" roughness={0.95} />
        </mesh>
      ))}

      {/* Left Abutment */}
      <mesh position={[-13, 1.5, 0]} castShadow receiveShadow>
        <boxGeometry args={[2.5, 3, 6]} />
        {concreteMat}
      </mesh>

      {/* Right Abutment */}
      <mesh position={[13, 1.5, 0]} castShadow receiveShadow>
        <boxGeometry args={[2.5, 3, 6]} />
        {concreteMat}
      </mesh>

      {/* Left Pier */}
      <mesh position={[-4.5, 1.8, 0]} castShadow receiveShadow>
        <boxGeometry args={[1.4, 3.6, 4]} />
        {concreteMat}
      </mesh>
      <mesh position={[-4.5, 3.8, 0]}>
        <boxGeometry args={[2.2, 0.4, 5.5]} />
        {concreteMat}
      </mesh>

      {/* Right Pier */}
      <mesh position={[4.5, 1.8, 0]} castShadow receiveShadow>
        <boxGeometry args={[1.4, 3.6, 4]} />
        {concreteMat}
      </mesh>
      <mesh position={[4.5, 3.8, 0]}>
        <boxGeometry args={[2.2, 0.4, 5.5]} />
        {concreteMat}
      </mesh>

      {/* Steel Girders */}
      {[-1.5, 0, 1.5].map((z, i) => (
        <mesh key={i} position={[0, 3.7, z]} castShadow>
          <boxGeometry args={[27, 0.3, 0.25]} />
          {steelMat}
        </mesh>
      ))}

      {/* Cross beams */}
      {[-10, -5, 0, 5, 10].map((x, i) => (
        <mesh key={i} position={[x, 3.7, 0]}>
          <boxGeometry args={[0.25, 0.3, 4]} />
          {steelMat}
        </mesh>
      ))}

      {/* Guardrails */}
      {[-1, 1].map((s, i) => (
        <mesh key={i} position={[0, 4.6, s * 2.1]}>
          <boxGeometry args={[27.5, 0.3, 0.1]} />
          <meshStandardMaterial color="#6b7280" roughness={0.4} metalness={0.6} />
        </mesh>
      ))}
    </group>
  );
}

// ─── 3D Sensor Node in Editor ──────────────────────────────────────────────────

function EditorSensorNode3D({ sensor, isSelected, onSelect }) {
  const meshRef = useRef();
  const color = TYPE_COLORS[sensor.sensor_type] || '#22d3ee';
  const coords = sensor.coordinates_3d || [0, 4.3, 0];

  useFrame(({ clock }) => {
    if (!meshRef.current) return;
    if (isSelected) {
      const pulse = 1 + 0.15 * Math.sin(clock.getElapsedTime() * 5);
      meshRef.current.scale.set(pulse, pulse, pulse);
    } else {
      meshRef.current.scale.set(1, 1, 1);
    }
  });

  return (
    <group
      position={coords}
      onClick={(e) => {
        e.stopPropagation();
        onSelect(sensor.channel_name);
      }}
    >
      {/* Core Sphere */}
      <mesh ref={meshRef} castShadow>
        <sphereGeometry args={[0.42, 24, 24]} />
        <meshStandardMaterial
          color={color}
          emissive={isSelected ? color : '#000000'}
          emissiveIntensity={isSelected ? 0.9 : 0.2}
          roughness={0.2}
          metalness={0.6}
        />
      </mesh>

      {/* Selection Halo Ring */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.65, 0.05, 8, 32]} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={isSelected ? 0.9 : 0.4}
        />
      </mesh>

      {/* Floating 3D Label */}
      <Text
        position={[0, 0.9, 0]}
        fontSize={0.4}
        color={isSelected ? '#ffffff' : color}
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.05}
        outlineColor="#090d16"
      >
        {sensor.channel_name.replace('Sensor_', '').replace(/_/g, ' ')}
      </Text>
    </group>
  );
}

// ─── Main 3D Sensor Placement Component ───────────────────────────────────────

export default function SensorPlacementEditor({ structure, onSave, onClose }) {
  const sid = structure?.structure_id || 'bridge_demo_01';

  // Normalize initial sensors from localStorage, structure.sensors, or defaults
  const [sensors, setSensors] = useState(() => {
    // 1. Try reading from persistent localStorage
    try {
      const stored = localStorage.getItem(`structureiq_sensors_${sid}`);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed) && parsed.length > 0) {
          return parsed.map((s, i) => ({
            channel_name: s.channel_name || s.name || `Sensor_Node_${i + 1}`,
            sensor_type: s.sensor_type || s.type || 'ACCELEROMETER',
            coordinates_3d: s.coordinates_3d || s.position_xyz || [0, 4.3, 0],
            threshold_tau: s.threshold_tau || 0.04,
          }));
        }
      }
    } catch (e) {
      // ignore
    }

    // 2. Try structure.sensors
    if (structure?.sensors && structure.sensors.length > 0) {
      return structure.sensors.map((s, i) => {
        const coords = s.coordinates_3d || s.position_xyz || [
          ((i / Math.max(structure.sensors.length - 1, 1)) - 0.5) * 24,
          4.3,
          0
        ];
        return {
          channel_name: s.channel_name || s.name || `Sensor_Node_${i + 1}`,
          sensor_type: s.sensor_type || s.type || 'ACCELEROMETER',
          coordinates_3d: coords,
          threshold_tau: s.threshold_tau || 0.04,
        };
      });
    }

    const defaultChannels = structure?.channels || [
      'Sensor_Abutment_Left',
      'Sensor_Pier_Left',
      'Sensor_MidSpan_Girder',
      'Sensor_Pier_Right',
      'Sensor_Ambient_Temp',
    ];

    const defaultPositions = [
      [-12, 4.3, 0],
      [-4.5, 4.3, 0],
      [0, 4.5, 0],
      [4.5, 4.3, 0],
      [12, 4.3, 1.5],
    ];

    return defaultChannels.map((ch, i) => ({
      channel_name: ch,
      sensor_type: ch.includes('Temp') ? 'TEMPERATURE' : 'ACCELEROMETER',
      coordinates_3d: defaultPositions[i] || [((i / (defaultChannels.length - 1)) - 0.5) * 24, 4.3, 0],
      threshold_tau: 0.04,
    }));
  });

  const [selectedId, setSelectedId] = useState(
    sensors[0]?.channel_name || null
  );
  const [view, setView] = useState('iso'); // 'top', 'front', 'side', 'iso'
  const [snapToSurface, setSnapToSurface] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const orbitRef = useRef();

  const selectedSensor = sensors.find(s => s.channel_name === selectedId);

  // Smooth camera preset transition
  const setCameraView = useCallback((preset) => {
    setView(preset);
    const ctrl = orbitRef.current;
    if (!ctrl) return;
    const cam = ctrl.object;
    const p = CAMERA_PRESETS[preset];
    if (!p) return;
    cam.position.set(...p.pos);
    ctrl.target.set(...p.target);
    cam.updateProjectionMatrix();
    ctrl.update();
  }, []);

  // Update a field on a sensor
  const updateSensor = (id, field, value) => {
    setSensors(prev =>
      prev.map(s => (s.channel_name === id ? { ...s, [field]: value } : s))
    );
  };

  // Update coordinate along axis
  const updateCoord = (id, axisIdx, val) => {
    setSensors(prev =>
      prev.map(s => {
        if (s.channel_name !== id) return s;
        const coords = [...(s.coordinates_3d || [0, 4.3, 0])];
        coords[axisIdx] = parseFloat(val) || 0;
        return { ...s, coordinates_3d: coords };
      })
    );
  };

  // Add new sensor
  const addSensor = () => {
    const idx = sensors.length + 1;
    const newSensor = {
      channel_name: `Sensor_Custom_${idx}`,
      sensor_type: 'ACCELEROMETER',
      coordinates_3d: [0, 4.3, 0],
      threshold_tau: 0.04,
    };
    setSensors(prev => [...prev, newSensor]);
    setSelectedId(newSensor.channel_name);
  };

  // Remove sensor
  const removeSensor = (id) => {
    if (sensors.length <= 1) {
      alert('The structure requires at least 1 sensor node.');
      return;
    }
    setSensors(prev => prev.filter(s => s.channel_name !== id));
    if (selectedId === id) {
      const remaining = sensors.filter(s => s.channel_name !== id);
      setSelectedId(remaining[0]?.channel_name || null);
    }
  };

  // Save handler
  const handleSave = async () => {
    setIsSaving(true);
    setSaveSuccess(false);

    try {
      const payload = sensors.map(s => ({
        channel_name: s.channel_name,
        name: s.channel_name,
        sensor_type: s.sensor_type,
        type: s.sensor_type,
        coordinates_3d: s.coordinates_3d,
        position_xyz: s.coordinates_3d,
        mesh_name: s.mesh_name || null,
        threshold_tau: s.threshold_tau || 0.04,
      }));

      // Immediate localStorage persistence
      try {
        localStorage.setItem(`structureiq_sensors_${sid}`, JSON.stringify(payload));
        window.dispatchEvent(new Event('structureiq_sensors_updated'));
      } catch (e) {
        console.warn('LocalStorage error:', e);
      }

      // Save each sensor position to backend structure config
      for (const s of sensors) {
        const sensorId = s.id || s.channel_name;
        if (sensorId && s.coordinates_3d) {
          try {
            await fetch(`${API_BASE_URL}/api/v1/structures/${sid}/sensors/${sensorId}/position`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                position_xyz: s.coordinates_3d,
                mesh_name: s.mesh_name || null,
              }),
            });
          } catch (e) {
            // non-fatal per-sensor error
          }
        }
      }

      // Call parent onSave which hits backend and updates state
      if (onSave) {
        await onSave(payload);
      }

      setSaveSuccess(true);
      setTimeout(() => {
        setSaveSuccess(false);
        if (onClose) onClose();
      }, 1200);
    } catch (err) {
      console.error('Failed to save sensor layout:', err);
      alert('Error saving sensors: ' + err.message);
    } finally {
      setIsSaving(false);
    }
  };

  // Click on 3D model surface relocates selected sensor and binds mesh part name
  const handleDeckClick = (point, meshName) => {
    if (!selectedId) return;
    const finalPoint = snapToSurface
      ? [Math.round(point[0] * 2) / 2, Math.round(point[1] * 10) / 10, Math.round(point[2] * 2) / 2]
      : point;
    updateSensor(selectedId, 'coordinates_3d', finalPoint);
    if (meshName) {
      updateSensor(selectedId, 'mesh_name', meshName);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-stretch bg-black/80 backdrop-blur-md">
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="flex w-full h-full overflow-hidden"
      >
        {/* ── Left Sidebar: Sensor Nodes Catalog ── */}
        <div
          className="w-72 flex-shrink-0 flex flex-col overflow-hidden border-r"
          style={{ borderColor: 'var(--outline-variant)', background: 'var(--surface-container-low)' }}
        >
          {/* Header */}
          <div
            className="px-4 py-3 border-b flex items-center justify-between"
            style={{ borderColor: 'var(--outline-variant)', background: 'var(--surface-container)' }}
          >
            <div className="flex items-center gap-2">
              <Crosshair size={16} className="text-cyan-400" />
              <span className="font-body-sm text-body-sm font-bold text-white">3D Sensor Placement</span>
            </div>
            <button
              onClick={onClose}
              className="p-1 rounded hover:bg-white/5 text-slate-400 hover:text-white transition-colors cursor-pointer"
            >
              <X size={16} />
            </button>
          </div>

          {/* Active Asset Info */}
          <div className="px-4 py-2.5 border-b" style={{ borderColor: 'var(--outline-variant)' }}>
            <div className="font-label-caps text-label-caps uppercase text-slate-500 mb-0.5">Active Asset</div>
            <div className="text-xs font-semibold text-white">{structure?.name || 'Suspension Bridge'}</div>
            <div className="font-telemetry-sm text-[10px] text-cyan-400 mt-0.5">{structure?.structure_id || 'bridge_demo_01'}</div>
          </div>

          {/* 3D Camera Projection Toggles */}
          <div className="px-4 py-2.5 border-b" style={{ borderColor: 'var(--outline-variant)' }}>
            <div className="font-label-caps text-label-caps uppercase text-slate-400 mb-1.5 flex items-center justify-between">
              <span>Projection View</span>
              <span className="text-[9px] text-cyan-400/80 font-mono">3D CAMERA</span>
            </div>
            <div className="grid grid-cols-4 gap-1">
              {[
                { id: 'top', label: 'Top' },
                { id: 'front', label: 'Front' },
                { id: 'side', label: 'Side' },
                { id: 'iso', label: '3D Iso' },
              ].map(v => (
                <button
                  key={v.id}
                  onClick={() => setCameraView(v.id)}
                  className={`py-1.5 text-xs font-semibold rounded cursor-pointer transition-all ${
                    view === v.id
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm shadow-cyan-500/20'
                      : 'text-slate-400 hover:text-white bg-white/[0.02] border border-white/5'
                  }`}
                >
                  {v.label}
                </button>
              ))}
            </div>
          </div>

          {/* Sensors List */}
          <div className="flex-1 overflow-y-auto px-3 py-2 flex flex-col gap-1.5">
            <div className="flex items-center justify-between px-1 mb-1">
              <span className="font-label-caps text-label-caps text-slate-400 uppercase">
                Instrumented Nodes ({sensors.length})
              </span>
              <button
                onClick={addSensor}
                className="text-[10px] font-semibold text-cyan-400 hover:text-cyan-300 flex items-center gap-1 cursor-pointer"
              >
                <Plus size={11} /> Add Node
              </button>
            </div>

            {sensors.map(s => {
              const isSel = selectedId === s.channel_name;
              const col = TYPE_COLORS[s.sensor_type] || '#22d3ee';
              return (
                <div
                  key={s.channel_name}
                  onClick={() => setSelectedId(s.channel_name)}
                  className={`p-2 rounded-lg cursor-pointer border transition-all ${
                    isSel
                      ? 'border-cyan-500/50 bg-cyan-950/25 shadow-sm shadow-cyan-950/40'
                      : 'border-white/5 hover:border-white/20 bg-white/[0.02]'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: col }} />
                      <span className="text-xs font-semibold text-slate-200 truncate max-w-[140px]">
                        {s.channel_name.replace('Sensor_', '').replace(/_/g, ' ')}
                      </span>
                    </div>
                    <button
                      onClick={(e) => { e.stopPropagation(); removeSensor(s.channel_name); }}
                      className="p-1 rounded text-red-400/40 hover:text-red-400 cursor-pointer transition-colors"
                      title="Remove Sensor"
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                  <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono mt-1 pl-4">
                    <span>{s.sensor_type}</span>
                    <span className="text-slate-500">
                      [{s.coordinates_3d?.map(n => Number(n).toFixed(1)).join(', ')}]
                    </span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Bottom Save Action */}
          <div className="p-3 border-t flex flex-col gap-2" style={{ borderColor: 'var(--outline-variant)' }}>
            {saveSuccess && (
              <div className="p-2 rounded bg-emerald-950/40 border border-emerald-500/40 text-emerald-300 text-xs flex items-center gap-1.5">
                <Check size={14} className="text-emerald-400" />
                <span>Saved successfully to 3D Twin!</span>
              </div>
            )}
            <button
              onClick={handleSave}
              disabled={isSaving}
              className="w-full py-2.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold flex items-center justify-center gap-2 shadow-lg shadow-cyan-600/20 cursor-pointer transition-all disabled:opacity-50"
            >
              <Save size={14} />
              <span>{isSaving ? 'Syncing with Twin...' : 'Save Sensor Layout'}</span>
            </button>
          </div>
        </div>

        {/* ── Center: Interactive 3D Canvas ── */}
        <div className="flex-1 relative overflow-hidden" style={{ background: '#080d16' }}>
          {/* Top Canvas Bar */}
          <div className="absolute top-3 left-3 z-10 flex items-center gap-2 pointer-events-none">
            <div className="px-3 py-1.5 rounded-lg bg-black/60 border border-white/10 backdrop-blur-md text-xs text-slate-200 flex items-center gap-2 pointer-events-auto">
              <Box size={14} className="text-cyan-400" />
              <span className="font-mono font-semibold uppercase">{view} 3D PROJECTION</span>
            </div>
            <div className="flex items-center gap-2 pointer-events-auto">
              <button
                type="button"
                onClick={() => setSnapToSurface(!snapToSurface)}
                className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium border transition backdrop-blur-md cursor-pointer flex items-center gap-1.5 ${
                  snapToSurface
                    ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
                    : 'bg-black/60 text-slate-400 border-white/10 hover:text-white'
                }`}
              >
                <Crosshair size={13} /> Snap to Surface: {snapToSurface ? 'ON' : 'OFF'}
              </button>
              <div className="px-3 py-1.5 rounded-lg bg-black/60 border border-white/10 backdrop-blur-md text-xs text-slate-300">
                Click 3D surface or adjust X, Y, Z coordinates to move sensor
              </div>
            </div>
          </div>

          {/* 3D Scene */}
          <Canvas
            shadows
            camera={{ position: [18, 16, 18], fov: 50, near: 0.1, far: 500 }}
            style={{ width: '100%', height: '100%' }}
          >
            {/* Lighting */}
            <ambientLight intensity={0.4} />
            <directionalLight position={[20, 30, 10]} intensity={1.2} castShadow />
            <directionalLight position={[-15, 20, -10]} intensity={0.4} color="#6366f1" />
            <pointLight position={[0, 8, 0]} intensity={0.5} color="#22d3ee" />

            {/* Ground Grid */}
            <Grid
              args={[60, 60]}
              position={[0, 0, 0]}
              cellSize={2}
              cellThickness={0.3}
              cellColor="#1e3a5f"
              sectionSize={10}
              sectionThickness={0.8}
              sectionColor="#1d4ed8"
              fadeDistance={60}
              fadeStrength={1}
              infiniteGrid
            />

            {/* Dynamic 3D CAD/GLB Model with Fallback to Procedural Bridge */}
            <Suspense fallback={<EditorBridgeGeometry onDeckClick={handleDeckClick} />}>
              <StructureGLBModel
                modelUrl={`${API_BASE_URL}/api/v1/structures/${sid}/model`}
                sensors={sensors}
                highlightMeshName={selectedSensor?.mesh_name}
                onSurfaceClick={(point, meshName) => handleDeckClick(point, meshName)}
              />
            </Suspense>

            {/* 3D Sensor Nodes */}
            {sensors.map(s => (
              <EditorSensorNode3D
                key={s.channel_name}
                sensor={s}
                isSelected={selectedId === s.channel_name}
                onSelect={setSelectedId}
              />
            ))}

            <OrbitControls
              ref={orbitRef}
              enableDamping
              dampingFactor={0.06}
              minDistance={4}
              maxDistance={90}
              target={[0, 4, 0]}
            />
          </Canvas>
        </div>

        {/* ── Right Panel: Selected 3D Node Properties ── */}
        <AnimatePresence>
          {selectedSensor && (
            <motion.div
              initial={{ x: 30, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: 30, opacity: 0 }}
              className="w-72 flex-shrink-0 flex flex-col overflow-hidden border-l"
              style={{ borderColor: 'var(--outline-variant)', background: 'var(--surface-container-low)' }}
            >
              <div
                className="px-4 py-3 border-b"
                style={{ borderColor: 'var(--outline-variant)', background: 'var(--surface-container)' }}
              >
                <div className="text-xs font-bold text-white flex items-center justify-between">
                  <span>Selected Node</span>
                  <span
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ backgroundColor: TYPE_COLORS[selectedSensor.sensor_type] || '#22d3ee' }}
                  />
                </div>
                <div className="text-[11px] text-cyan-400 font-mono mt-0.5 truncate">{selectedSensor.channel_name}</div>
              </div>

              <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3.5 text-xs">
                {/* Channel Name */}
                <div>
                  <label className="font-label-caps text-label-caps uppercase text-slate-400 block mb-1">Channel Name</label>
                  <input
                    type="text"
                    value={selectedSensor.channel_name}
                    onChange={e => updateSensor(selectedSensor.channel_name, 'channel_name', e.target.value)}
                    className="w-full px-2.5 py-1.5 rounded bg-white/[0.04] border border-white/10 text-white text-xs font-mono focus:border-cyan-500 focus:outline-none"
                  />
                </div>

                {/* Sensor Type */}
                <div>
                  <label className="font-label-caps text-label-caps uppercase text-slate-400 block mb-1">Sensor Typology</label>
                  <select
                    value={selectedSensor.sensor_type}
                    onChange={e => updateSensor(selectedSensor.channel_name, 'sensor_type', e.target.value)}
                    className="w-full px-2.5 py-1.5 rounded bg-[#101728] border border-white/10 text-white text-xs focus:border-cyan-500 focus:outline-none"
                  >
                    {SENSOR_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
                  </select>
                </div>

                {/* Threshold Tau */}
                <div>
                  <label className="font-label-caps text-label-caps uppercase text-slate-400 block mb-1">
                    UCL Threshold (τ)
                  </label>
                  <input
                    type="number"
                    step="0.005"
                    value={selectedSensor.threshold_tau}
                    onChange={e => updateSensor(selectedSensor.channel_name, 'threshold_tau', parseFloat(e.target.value) || 0.04)}
                    className="w-full px-2.5 py-1.5 rounded bg-white/[0.04] border border-white/10 text-white text-xs font-mono focus:border-cyan-500 focus:outline-none"
                  />
                </div>

                {/* Assigned 3D Mesh Part */}
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="font-label-caps text-label-caps uppercase text-slate-400 block">
                      Assigned 3D Mesh Part
                    </label>
                    {selectedSensor.mesh_name && (
                      <span className="text-[10px] text-cyan-400 font-mono">Bound</span>
                    )}
                  </div>
                  <input
                    type="text"
                    placeholder="Click 3D surface or type part name"
                    value={selectedSensor.mesh_name || ''}
                    onChange={e => updateSensor(selectedSensor.channel_name, 'mesh_name', e.target.value)}
                    className="w-full px-2.5 py-1.5 rounded bg-white/[0.04] border border-white/10 text-white text-xs font-mono focus:border-cyan-500 focus:outline-none"
                  />
                  <p className="text-[10px] text-slate-500 mt-1">
                    Clicking the 3D model surface will raycast coordinates and bind the clicked mesh part.
                  </p>
                </div>

                {/* 3D Spatial Coordinates (X, Y, Z in meters) */}
                <div className="pt-2 border-t border-white/5">
                  <div className="flex items-center justify-between mb-2">
                    <label className="font-label-caps text-label-caps uppercase text-cyan-300 font-semibold block">
                      3D Spatial Coordinates (m)
                    </label>
                    <span className="text-[10px] text-slate-500 font-mono">X, Y, Z</span>
                  </div>

                  <div className="flex flex-col gap-2">
                    {[
                      { axis: 'X (Span Longitudinal)', idx: 0, min: -14, max: 14, step: 0.5 },
                      { axis: 'Y (Elevation Vertical)', idx: 1, min: 0, max: 10, step: 0.2 },
                      { axis: 'Z (Transverse Lateral)', idx: 2, min: -3, max: 3, step: 0.5 },
                    ].map(({ axis, idx, min, max, step }) => {
                      const curVal = selectedSensor.coordinates_3d?.[idx] ?? 0;
                      return (
                        <div key={axis} className="p-2 rounded bg-white/[0.02] border border-white/5 flex flex-col gap-1">
                          <div className="flex items-center justify-between">
                            <span className="text-[10px] text-slate-400 font-mono">{axis}</span>
                            <span className="text-xs font-mono font-bold text-cyan-300">
                              {Number(curVal).toFixed(2)} m
                            </span>
                          </div>
                          <div className="flex items-center gap-2">
                            <input
                              type="range"
                              min={min}
                              max={max}
                              step={step}
                              value={curVal}
                              onChange={e => updateCoord(selectedSensor.channel_name, idx, e.target.value)}
                              className="flex-1 accent-cyan-400 cursor-pointer"
                            />
                            <input
                              type="number"
                              step={step}
                              value={curVal}
                              onChange={e => updateCoord(selectedSensor.channel_name, idx, e.target.value)}
                              className="w-16 px-1.5 py-0.5 rounded bg-black/40 border border-white/10 text-right text-xs font-mono text-white"
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Snap presets */}
                <div className="pt-2 border-t border-white/5 flex flex-col gap-1.5">
                  <span className="font-label-caps text-label-caps uppercase text-slate-500">Quick Snapping</span>
                  <div className="grid grid-cols-2 gap-1.5">
                    <button
                      onClick={() => updateCoord(selectedSensor.channel_name, 1, 4.3)}
                      className="px-2 py-1 rounded bg-white/[0.03] hover:bg-white/[0.08] text-[10px] text-slate-300 font-mono border border-white/5 cursor-pointer"
                    >
                      Snap to Deck (Y=4.3)
                    </button>
                    <button
                      onClick={() => updateCoord(selectedSensor.channel_name, 0, 0)}
                      className="px-2 py-1 rounded bg-white/[0.03] hover:bg-white/[0.08] text-[10px] text-slate-300 font-mono border border-white/5 cursor-pointer"
                    >
                      Center Span (X=0)
                    </button>
                    <button
                      onClick={() => updateCoord(selectedSensor.channel_name, 0, -4.5)}
                      className="px-2 py-1 rounded bg-white/[0.03] hover:bg-white/[0.08] text-[10px] text-slate-300 font-mono border border-white/5 cursor-pointer"
                    >
                      Left Pier (X=-4.5)
                    </button>
                    <button
                      onClick={() => updateCoord(selectedSensor.channel_name, 0, 4.5)}
                      className="px-2 py-1 rounded bg-white/[0.03] hover:bg-white/[0.08] text-[10px] text-slate-300 font-mono border border-white/5 cursor-pointer"
                    >
                      Right Pier (X=4.5)
                    </button>
                  </div>
                </div>
              </div>

              <div className="p-3 border-t" style={{ borderColor: 'var(--outline-variant)' }}>
                <button
                  onClick={() => removeSensor(selectedSensor.channel_name)}
                  className="w-full py-1.5 rounded-lg border border-red-500/30 text-red-400 text-xs font-semibold hover:bg-red-950/30 cursor-pointer transition-colors flex items-center justify-center gap-1.5"
                >
                  <Trash2 size={13} /> Remove Node
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
}

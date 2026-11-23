/**
 * DigitalTwinCanvas.jsx
 * React Three Fiber 3D procedural bridge scene with dynamic material mutations.
 * Structural nodes pulse red (CRITICAL), glow amber (WARNING), or stay grey (NORMAL).
 */

import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Environment, Text } from '@react-three/drei';
import { useRef, useMemo, useCallback, useState, useEffect, Suspense } from 'react';
import * as THREE from 'three';
import { Upload } from 'lucide-react';
import StructureGLBModel from './StructureGLBModel';
import ModelUploadModal from './ModelUploadModal';
import { API_BASE_URL } from '../config';

const CAMERA_PRESETS = {
  front: { pos: [0, 6, 26], target: [0, 3, 0] },
  top:   { pos: [0, 28, 0.01], target: [0, 3, 0] },
  iso:   { pos: [18, 18, 18], target: [0, 3, 0] },
};

const NODE_POSITIONS = {
  Sensor_Abutment_Left:  [-12, 2, 0],
  Sensor_Pier_Left:      [-4.5, 3.5, 0],
  Sensor_MidSpan_Girder: [0, 4.6, 0],
  Sensor_Pier_Right:     [4.5, 3.5, 0],
  Sensor_Ambient_Temp:   [12, 1, 2],
};

const STATUS_COLORS = {
  NORMAL:   { color: '#64748b', emissive: '#000000', emissiveIntensity: 0 },
  WARNING:  { color: '#f59e0b', emissive: '#331c00', emissiveIntensity: 0.6 },
  CRITICAL: { color: '#ef4444', emissive: '#7f1d1d', emissiveIntensity: 1.2 },
  LOW_CONFIDENCE: { color: '#38bdf8', emissive: '#0c4a6e', emissiveIntensity: 0.8 },
  SENSOR_FAULT: { color: '#6b7280', emissive: '#1e293b', emissiveIntensity: 0.25 },
  FAULT:    { color: '#6b7280', emissive: '#1e293b', emissiveIntensity: 0.25 },
  OFFLINE:  { color: '#475569', emissive: '#000000', emissiveIntensity: 0 },
  DEGRADED: { color: '#94a3b8', emissive: '#334155', emissiveIntensity: 0.3 },
};

// ─── Individual Structural Sensor Node ────────────────────────────────────────

function StructuralNode({ nodeId, status = 'NORMAL', mse = 0, position = [0, 0, 0], radius = 0.35 }) {
  const meshRef = useRef();
  const matRef = useRef();
  const phaseRef = useRef(Math.random() * Math.PI * 2);
  const pos = position || NODE_POSITIONS[nodeId] || [0, 0, 0];

  const targetColor = useMemo(() => new THREE.Color(STATUS_COLORS[status]?.color ?? '#64748b'), [status]);
  const targetEmissive = useMemo(() => new THREE.Color(STATUS_COLORS[status]?.emissive ?? '#000000'), [status]);
  const targetEmissiveIntensity = STATUS_COLORS[status]?.emissiveIntensity ?? 0;

  useFrame(({ clock }) => {
    if (!matRef.current) return;
    const t = clock.getElapsedTime();

    // Smooth color interpolation
    matRef.current.color.lerp(targetColor, 0.06);
    matRef.current.emissive.lerp(targetEmissive, 0.06);

    // Pulse emissive for CRITICAL (2 Hz sine wave)
    if (status === 'CRITICAL') {
      const pulse = 0.5 + 0.5 * Math.sin(2 * Math.PI * 2 * t + phaseRef.current);
      matRef.current.emissiveIntensity = targetEmissiveIntensity * pulse;
      // Scale pulse
      if (meshRef.current) {
        const s = 1 + 0.12 * pulse;
        meshRef.current.scale.setScalar(s);
      }
    } else {
      matRef.current.emissiveIntensity += (targetEmissiveIntensity - matRef.current.emissiveIntensity) * 0.08;
      if (meshRef.current) {
        meshRef.current.scale.lerp(new THREE.Vector3(1, 1, 1), 0.1);
      }
    }
  });

  const isFaultOrOffline = status === 'OFFLINE' || status === 'SENSOR_FAULT' || status === 'FAULT';

  return (
    <group position={pos}>
      <mesh ref={meshRef} castShadow={!isFaultOrOffline}>
        <sphereGeometry args={[radius, isFaultOrOffline ? 12 : 24, isFaultOrOffline ? 12 : 24]} />
        <meshStandardMaterial
          ref={matRef}
          color={STATUS_COLORS[status]?.color ?? '#64748b'}
          emissive={isFaultOrOffline ? '#000000' : (STATUS_COLORS[status]?.emissive ?? '#000000')}
          emissiveIntensity={0}
          roughness={isFaultOrOffline ? 0.9 : 0.3}
          metalness={isFaultOrOffline ? 0.1 : 0.6}
          wireframe={isFaultOrOffline}
          transparent={isFaultOrOffline}
          opacity={isFaultOrOffline ? 0.65 : 1.0}
        />
      </mesh>
      {/* Outer ring indicator (dashed / wireframe when OFFLINE or SENSOR_FAULT) */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[radius * 1.5, radius * 0.12, 8, isFaultOrOffline ? 12 : 32]} />
        <meshStandardMaterial
          color={STATUS_COLORS[status]?.color ?? '#64748b'}
          transparent
          wireframe={isFaultOrOffline}
          opacity={isFaultOrOffline ? 0.45 : 0.35}
        />
      </mesh>
    </group>
  );
}

// ─── Procedural Bridge Geometry ───────────────────────────────────────────────

function BridgeGeometry() {
  const concreteMat = (
    <meshStandardMaterial color="#374151" roughness={0.85} metalness={0.05} />
  );
  const steelMat = (
    <meshStandardMaterial color="#4b5563" roughness={0.4} metalness={0.7} />
  );

  return (
    <group>
      {/* Bridge Deck */}
      <mesh position={[0, 4.0, 0]} castShadow receiveShadow>
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
      {/* Left Pier cap */}
      <mesh position={[-4.5, 3.8, 0]}>
        <boxGeometry args={[2.2, 0.4, 5.5]} />
        {concreteMat}
      </mesh>

      {/* Right Pier */}
      <mesh position={[4.5, 1.8, 0]} castShadow receiveShadow>
        <boxGeometry args={[1.4, 3.6, 4]} />
        {concreteMat}
      </mesh>
      {/* Right Pier cap */}
      <mesh position={[4.5, 3.8, 0]}>
        <boxGeometry args={[2.2, 0.4, 5.5]} />
        {concreteMat}
      </mesh>

      {/* Steel Girders (3 longitudinal) */}
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

// ─── Camera Preset Controls ───────────────────────────────────────────────────

function CameraButton({ children, onClick, className = '' }) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 text-xs rounded-lg border border-white/10 bg-black/40 text-slate-300 hover:bg-white/10 hover:text-white cursor-pointer transition-all duration-150 backdrop-blur-sm ${className}`}
    >
      {children}
    </button>
  );
}

// ─── Main Canvas Component ────────────────────────────────────────────────────

export default function DigitalTwinCanvas({ payload, structure }) {
  const orbitRef = useRef();
  const [activePreset, setActivePreset] = useState('front');
  const [updateTick, setUpdateTick] = useState(0);
  const [modelKey, setModelKey] = useState(0);
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [modelBounds, setModelBounds] = useState(null);

  const structureId = structure?.structure_id || 'kw51';

  // Proportional sphere radius based on bounding diameter
  const dynamicSphereRadius = useMemo(() => {
    if (!modelBounds?.radius) return 0.35;
    return Math.max(0.18, Math.min(1.2, modelBounds.radius * 0.025));
  }, [modelBounds]);

  // Listen for instant sensor placement updates
  useEffect(() => {
    const handleUpdate = () => setUpdateTick(t => t + 1);
    window.addEventListener('structureiq_sensors_updated', handleUpdate);
    return () => window.removeEventListener('structureiq_sensors_updated', handleUpdate);
  }, []);

  const nodesToRender = useMemo(() => {
    // Collect active channel names
    const activeChannels = payload?.nodes
      ? Object.keys(payload.nodes)
      : structure?.channels || Object.keys(NODE_POSITIONS);

    // Build map of sensor coords: apply backend defaults first, then user's custom placement
    const sensorCoordMap = {};

    // 1. Overlay structure.sensors from backend if passed
    if (structure?.sensors && Array.isArray(structure.sensors)) {
      structure.sensors.forEach(s => {
        const chName = s.channel_name || s.name || s.id;
        const coords = s.coordinates_3d || s.position_xyz;
        if (chName && coords && Array.isArray(coords)) {
          sensorCoordMap[chName] = coords;
          if (s.id) sensorCoordMap[s.id] = coords;
        }
      });
    }

    // 2. User's saved custom placement in localStorage takes HIGHEST priority!
    const sid = structure?.structure_id || 'kw51';
    try {
      const stored = localStorage.getItem(`structureiq_sensors_${sid}`);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed) && parsed.length > 0) {
          parsed.forEach(s => {
            const chName = s.channel_name || s.name || s.id;
            const coords = s.coordinates_3d || s.position_xyz;
            if (chName && coords && Array.isArray(coords)) {
              sensorCoordMap[chName] = coords;
              if (s.id) sensorCoordMap[s.id] = coords;
            }
          });
        }
      }
    } catch (e) {
      // ignore
    }

    return activeChannels.map((ch, idx) => {
      let pos = sensorCoordMap[ch] || NODE_POSITIONS[ch];
      if (!pos) {
        // Procedurally distribute across bridge span
        const total = activeChannels.length;
        const x = total > 1 ? ((idx / (total - 1)) - 0.5) * 24 : 0;
        pos = [x, 4.2, 0];
      }
      return {
        id: ch,
        position: pos,
        status: payload?.nodes?.[ch]?.status ?? 'NORMAL',
        mse: payload?.nodes?.[ch]?.mse ?? 0,
      };
    });
  }, [payload, structure, updateTick]);

  const setCameraPreset = useCallback((preset) => {
    const ctrl = orbitRef.current;
    if (!ctrl) return;
    const cam = ctrl.object;
    const p = CAMERA_PRESETS[preset];
    if (!p) return;
    setActivePreset(preset);
    cam.position.set(...p.pos);
    ctrl.target.set(...p.target);
    cam.updateProjectionMatrix();
    ctrl.update();
  }, []);

  const sensorStatuses = useMemo(() => {
    const map = {};
    if (payload?.nodes) {
      Object.entries(payload.nodes).forEach(([k, v]) => {
        map[k] = v.status || 'NORMAL';
      });
    }
    return map;
  }, [payload]);

  return (
    <div className="relative w-full h-full">
      <Canvas
        shadows
        camera={{ position: [18, 14, 18], fov: 50, near: 0.1, far: 500 }}
        style={{ background: 'transparent' }}
      >
        {/* Lighting */}
        <ambientLight intensity={0.4} />
        <directionalLight
          position={[20, 30, 10]}
          intensity={1.2}
          castShadow
          shadow-mapSize={[2048, 2048]}
        />
        <directionalLight position={[-10, 15, -10]} intensity={0.4} color="#6366f1" />
        <pointLight position={[0, 10, 0]} intensity={0.3} color="#3b82f6" />

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
          followCamera={false}
          infiniteGrid
        />

        {/* Dynamic 3D CAD/GLB Model with Fallback to Procedural Bridge */}
        <Suspense fallback={<BridgeGeometry />}>
          <StructureGLBModel
            key={`${structureId}_${modelKey}`}
            modelUrl={`${API_BASE_URL}/api/v1/structures/${structureId}/model`}
            sensors={structure?.sensors || []}
            sensorStatuses={sensorStatuses}
            onBoundsComputed={setModelBounds}
          />
        </Suspense>

        {/* Structural Sensor Nodes with Proportional Sizing */}
        {nodesToRender.map(node => (
          <StructuralNode
            key={node.id}
            nodeId={node.id}
            position={node.position}
            status={node.status}
            mse={node.mse}
            radius={dynamicSphereRadius}
          />
        ))}

        {/* Fog for atmosphere */}
        <fog attach="fog" args={['#0a0f1e', 40, 100]} />

        <OrbitControls
          ref={orbitRef}
          enableDamping
          dampingFactor={0.06}
          minDistance={3}
          maxDistance={120}
          target={[0, 3, 0]}
        />
      </Canvas>

      {/* Toolbar Controls Overlay */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2 z-10">
        <CameraButton onClick={() => setCameraPreset('front')}>Front</CameraButton>
        <CameraButton onClick={() => setCameraPreset('top')}>Top</CameraButton>
        <CameraButton onClick={() => setCameraPreset('iso')}>Isometric</CameraButton>
        <div className="h-4 w-px bg-white/20 mx-1" />
        <CameraButton
          onClick={() => setIsUploadOpen(true)}
          className="flex items-center gap-1.5 text-cyan-400 border-cyan-500/30 bg-cyan-950/50 hover:bg-cyan-900/60"
        >
          <Upload className="w-3.5 h-3.5" /> Import CAD
        </CameraButton>
      </div>

      {/* CAD Upload Modal */}
      <ModelUploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        structureId={structureId}
        onModelUpdated={() => {
          setModelKey(k => k + 1);
          setUpdateTick(t => t + 1);
        }}
      />
    </div>
  );
}

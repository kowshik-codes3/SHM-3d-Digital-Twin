/**
 * BridgeSVG.jsx  
 * Procedural SVG suspension bridge that matches the Stitch "Live 3D Digital Twin" design.
 * Sensor nodes pulse/glow based on live WebSocket payload status.
 */

import { useMemo } from 'react';

const NODE_DEFS = [
  { id: 'Sensor_Abutment_Left',  label: 'SN-01 • ABUTMENT LEFT',  x: 60,  y: 280, leg: 'L' },
  { id: 'Sensor_Pier_Left',      label: 'SN-02 • PIER LEFT',       x: 202, y: 340, leg: 'L' },
  { id: 'Sensor_MidSpan_Girder', label: 'SN-03 • MID-SPAN GIRDER', x: 390, y: 280, leg: 'C' },
  { id: 'Sensor_Pier_Right',     label: 'SN-04 • PIER RIGHT',      x: 578, y: 340, leg: 'R' },
  { id: 'Sensor_Ambient_Temp',   label: 'SN-05 • ABUTMENT RIGHT',  x: 720, y: 280, leg: 'R' },
];

const STATUS_STYLE = {
  NORMAL:   { fill: '#3b82f6', glow: 'rgba(59,130,246,0.4)',  ring: '#3b82f6', pulse: false },
  WARNING:  { fill: '#f59e0b', glow: 'rgba(245,158,11,0.5)',  ring: '#f59e0b', pulse: true  },
  CRITICAL: { fill: '#ef4444', glow: 'rgba(239,68,68,0.6)',   ring: '#ef4444', pulse: true  },
  SELECTED: { fill: '#22d3ee', glow: 'rgba(34,211,238,0.5)',  ring: '#22d3ee', pulse: false },
};

function SensorNode({ def, nodeData, selectedId, onSelect }) {
  const status = nodeData?.status ?? 'NORMAL';
  const mse = nodeData?.mse ?? 0;
  const isSelected = selectedId === def.id;
  const style = isSelected ? STATUS_STYLE.SELECTED : (STATUS_STYLE[status] ?? STATUS_STYLE.NORMAL);

  return (
    <g
      className="cursor-pointer"
      onClick={() => onSelect(def.id)}
      style={{ filter: `drop-shadow(0 0 6px ${style.glow})` }}
    >
      {/* Outer pulsing ring (warning/critical only) */}
      {style.pulse && (
        <circle
          cx={def.x} cy={def.y} r="10"
          fill="none"
          stroke={style.ring}
          strokeWidth="1"
          opacity="0"
          className="animate-ping-slow"
          style={{ transformOrigin: `${def.x}px ${def.y}px` }}
        />
      )}
      {/* Inner ring */}
      <circle cx={def.x} cy={def.y} r="7" fill="none" stroke={style.ring} strokeWidth="1.5" opacity="0.6" />
      {/* Core dot */}
      <circle cx={def.x} cy={def.y} r="4" fill={style.fill} />

      {/* Tooltip on hover */}
      <title>{def.label} — {status} | MSE: {Number(mse).toFixed(5)}</title>
    </g>
  );
}

export default function BridgeSVG({ payload, nodes: customNodes, selectedNodeId, onSelectNode }) {
  const nodes = customNodes || NODE_DEFS;

  // Build dynamic position map for custom sensors
  const dynamicNodes = useMemo(() => {
    if (customNodes) return customNodes;
    return NODE_DEFS;
  }, [customNodes]);

  return (
    <svg
      viewBox="0 0 780 500"
      className="w-full h-full"
      style={{ maxHeight: '100%' }}
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        {/* Gradient for deck */}
        <linearGradient id="deckGrad" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#1e293b" />
          <stop offset="100%" stopColor="#0f172a" />
        </linearGradient>
        {/* Glow filter */}
        <filter id="cyanGlow">
          <feGaussianBlur in="SourceGraphic" stdDeviation="3" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
        {/* Grid pattern */}
        <pattern id="microGrid" width="20" height="20" patternUnits="userSpaceOnUse">
          <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(255,255,255,0.03)" strokeWidth="0.5" />
        </pattern>
      </defs>

      {/* Background grid */}
      <rect width="780" height="500" fill="url(#microGrid)" />

      {/* Horizon ellipse (depth glow) */}
      <ellipse cx="390" cy="390" rx="320" ry="40" fill="none" stroke="rgba(34,211,238,0.05)" strokeWidth="1" />
      <ellipse cx="390" cy="440" rx="240" ry="30" fill="none" stroke="rgba(34,211,238,0.03)" strokeWidth="1" />

      {/* Waterline reference */}
      <line x1="40" y1="440" x2="740" y2="440" stroke="#004e5a" strokeDasharray="4,8" strokeWidth="1" opacity="0.5" />
      <text x="45" y="436" fill="#22d3ee" fontSize="8" fontFamily="JetBrains Mono" opacity="0.4">WATERLINE / DEPTH REF</text>

      {/* ── Pier Foundations ── */}
      <polygon points="170,460 190,330 215,330 235,460" fill="#171b26" stroke="#22d3ee" strokeWidth="0.8" opacity="0.7" />
      <polygon points="545,460 565,330 590,330 610,460" fill="#171b26" stroke="#22d3ee" strokeWidth="0.8" opacity="0.7" />

      {/* ── Pylon Towers ── */}
      <line x1="202" y1="330" x2="202" y2="110" stroke="#8aebff" strokeWidth="2.2" opacity="0.85" />
      <line x1="202" y1="110" x2="190" y2="70" stroke="#8aebff" strokeWidth="1.8" opacity="0.7" />
      <line x1="578" y1="330" x2="578" y2="110" stroke="#8aebff" strokeWidth="2.2" opacity="0.85" />
      <line x1="578" y1="110" x2="590" y2="70" stroke="#8aebff" strokeWidth="1.8" opacity="0.7" />

      {/* Tower bracings */}
      <line x1="185" y1="170" x2="220" y2="230" stroke="#22d3ee" strokeWidth="0.9" opacity="0.45" />
      <line x1="220" y1="170" x2="185" y2="230" stroke="#22d3ee" strokeWidth="0.9" opacity="0.45" />
      <line x1="560" y1="170" x2="595" y2="230" stroke="#22d3ee" strokeWidth="0.9" opacity="0.45" />
      <line x1="595" y1="170" x2="560" y2="230" stroke="#22d3ee" strokeWidth="0.9" opacity="0.45" />

      {/* Tower caps */}
      <rect x="185" y="67" width="34" height="8" fill="#22d3ee" opacity="0.4" rx="1" />
      <rect x="560" y="67" width="34" height="8" fill="#22d3ee" opacity="0.4" rx="1" />

      {/* ── Main Suspension Cables ── */}
      <path d="M 60 260 Q 202 110 390 270" fill="none" stroke="#adc6ff" strokeWidth="1.5" opacity="0.7" />
      <path d="M 390 270 Q 578 110 720 260" fill="none" stroke="#adc6ff" strokeWidth="1.5" opacity="0.7" />

      {/* ── Stay Cables (left pylon) ── */}
      {[[90, 270], [140, 270], [260, 275], [320, 275]].map(([x2, y2], i) => (
        <line key={`lc-${i}`} x1="190" y1="70" x2={x2} y2={y2} stroke="#8aebff" strokeWidth="0.7" opacity={i < 2 ? 0.45 : 0.55} />
      ))}
      {/* ── Stay Cables (right pylon) ── */}
      {[[460, 275], [520, 275], [640, 270], [690, 270]].map(([x2, y2], i) => (
        <line key={`rc-${i}`} x1="590" y1="70" x2={x2} y2={y2} stroke="#8aebff" strokeWidth="0.7" opacity={i < 2 ? 0.55 : 0.45} />
      ))}

      {/* ── Road Deck / Girder ── */}
      <rect x="40" y="272" width="700" height="16" rx="2" fill="url(#deckGrad)" stroke="#22d3ee" strokeWidth="0.6" opacity="0.9" />

      {/* Deck cross-members */}
      {[120,200,280,390,480,560,640].map(x => (
        <line key={`xm-${x}`} x1={x} y1="272" x2={x} y2="288" stroke="#22d3ee" strokeWidth="0.4" opacity="0.3" />
      ))}

      {/* Abutments */}
      <rect x="30" y="275" width="30" height="30" rx="2" fill="#1e293b" stroke="#22d3ee" strokeWidth="0.8" opacity="0.8" />
      <rect x="720" y="275" width="30" height="30" rx="2" fill="#1e293b" stroke="#22d3ee" strokeWidth="0.8" opacity="0.8" />

      {/* ── Sensor Nodes ── */}
      {dynamicNodes.map(def => (
        <SensorNode
          key={def.id}
          def={def}
          nodeData={payload?.nodes?.[def.id]}
          selectedId={selectedNodeId}
          onSelect={onSelectNode}
        />
      ))}

      {/* Scale bar */}
      <g transform="translate(40, 475)" opacity="0.4">
        <line x1="0" y1="0" x2="60" y2="0" stroke="#22d3ee" strokeWidth="1" />
        <line x1="0" y1="-3" x2="0" y2="3" stroke="#22d3ee" strokeWidth="1" />
        <line x1="60" y1="-3" x2="60" y2="3" stroke="#22d3ee" strokeWidth="1" />
        <text x="30" y="-5" textAnchor="middle" fill="#22d3ee" fontSize="7" fontFamily="JetBrains Mono">50 m</text>
      </g>

      {/* Coordinate axes */}
      <g transform="translate(720, 60)" opacity="0.35">
        <line x1="0" y1="0" x2="0" y2="-20" stroke="#22d3ee" strokeWidth="1" />
        <line x1="0" y1="0" x2="20" y2="0" stroke="#ef4444" strokeWidth="1" />
        <text x="22" y="4" fill="#ef4444" fontSize="7" fontFamily="JetBrains Mono">X</text>
        <text x="-3" y="-22" fill="#22d3ee" fontSize="7" fontFamily="JetBrains Mono">Z</text>
      </g>
    </svg>
  );
}

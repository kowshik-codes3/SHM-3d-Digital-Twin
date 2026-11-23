/**
 * TelemetryChart.jsx
 * Real-time multi-channel MSE residual line charts powered by Recharts.
 * Dynamically renders arbitrary sensor channels and UCL threshold reference lines.
 */

import { useMemo } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const PALETTE = [
  '#22d3ee', // Cyan
  '#818cf8', // Indigo
  '#f43f5e', // Rose
  '#fb923c', // Orange
  '#10b981', // Emerald
  '#a855f7', // Purple
  '#38bdf8', // Light Blue
  '#eab308', // Amber
];

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass-panel p-2 text-xs border border-white/10" style={{ background: 'rgba(15, 19, 29, 0.9)' }}>
      <p className="text-slate-400 mb-1 font-mono">Frame #{label}</p>
      {payload.map(p => (
        <p key={p.dataKey} style={{ color: p.color }} className="flex justify-between gap-3">
          <span>{p.name}:</span>
          <span className="mono font-semibold">{Number(p.value).toFixed(5)}</span>
        </p>
      ))}
    </div>
  );
};

export default function TelemetryChart({ nodeHistory = {}, payload }) {
  // Dynamically determine active channels
  const channels = useMemo(() => {
    const keys = Object.keys(nodeHistory);
    if (keys.length > 0) return keys;
    if (payload?.nodes && typeof payload.nodes === 'object') {
      return Object.keys(payload.nodes);
    }
    return [];
  }, [nodeHistory, payload]);

  const channelConfig = useMemo(() => {
    return channels.map((key, idx) => ({
      key,
      label: key.replace(/^Sensor_/, '').replace(/_/g, ' '),
      color: PALETTE[idx % PALETTE.length],
    }));
  }, [channels]);

  // Build flat array for recharts: [{t, [ch1], [ch2], ...}]
  const chartData = useMemo(() => {
    if (channelConfig.length === 0) return [];
    const firstKey = channelConfig[0].key;
    const firstChannel = nodeHistory[firstKey] || [];
    return firstChannel.map((entry, i) => {
      const pt = { t: entry.t };
      channelConfig.forEach(({ key }) => {
        pt[key] = nodeHistory[key]?.[i]?.mse ?? 0;
      });
      return pt;
    });
  }, [nodeHistory, channelConfig]);

  // Latest UCL per channel from most recent payload
  const taus = useMemo(() => {
    const t = {};
    channelConfig.forEach(({ key }) => {
      t[key] = payload?.nodes?.[key]?.threshold ?? null;
    });
    return t;
  }, [payload, channelConfig]);

  return (
    <div className="w-full h-full flex flex-col gap-2 p-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
            Live MSE Residuals
          </span>
          <span className="text-xs text-slate-500 hidden sm:inline">— threshold lines (UCL τ)</span>
        </div>
        <span className="text-xs text-cyan-400/80 font-mono">
          {channelConfig.length} Active Channels
        </span>
      </div>
      <div style={{ flex: 1, minHeight: 0 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 4, right: 16, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis
              dataKey="t"
              tick={{ fill: '#64748b', fontSize: 9 }}
              tickLine={false}
              axisLine={false}
              tickFormatter={v => `#${v}`}
            />
            <YAxis
              tick={{ fill: '#64748b', fontSize: 9 }}
              tickLine={false}
              axisLine={false}
              tickFormatter={v => (typeof v === 'number' ? v.toFixed(3) : v)}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: '10px', color: '#94a3b8', paddingTop: 2 }}
              formatter={(val, entry) => (
                <span style={{ color: entry.color }} className="font-mono text-xs">{val}</span>
              )}
            />
            {channelConfig.map(({ key, label, color }) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                name={label}
                stroke={color}
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
              />
            ))}
            {/* UCL threshold reference lines */}
            {channelConfig.map(({ key, color }) =>
              taus[key] != null ? (
                <ReferenceLine
                  key={`tau-${key}`}
                  y={taus[key]}
                  stroke={color}
                  strokeDasharray="4 4"
                  strokeOpacity={0.4}
                  strokeWidth={1}
                />
              ) : null
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}


/**
 * App.jsx
 * StructureIQ - Unified Digital Twin Platform (Stitch Design Implementation)
 * Connects all 8 Stitch screens with real-time FastAPI REST endpoints & WebSocket telemetry:
 * 1. Live 3D Digital Twin (BridgeSVG + Sensor Telemetry + Residuals Chart + Fault Controls)
 * 2. Structures Overview Fleet Grid (Fleet KPIs + Multi-Asset Cards + Lifecycle actions)
 * 3. Alerts Inbox & Incident Console (Defcon-2 KPIs + Real-time Incident Audit Log)
 * 4. Add Structure Wizard (3-step asset commissioning modal)
 * 5. Sensor Placement 3D Editor (Spatial canvas node configuration)
 * 6. Alert Detail & AI Diagnosis (Defcon-2 modal with physics-informed diagnosis)
 * 7. Settings & System Configuration (Brokers, LSTM hyperparameters, rule builder)
 * 8. Custom Alert Rule Modal (Trigger logic condition builder)
 */

import React, { useState, useEffect, useCallback } from 'react';
import AppShell from './components/AppShell';
import LiveTwin from './components/LiveTwin';
import FleetGrid from './components/FleetGrid';
import AlertsInbox from './components/AlertsInbox';
import AlertDetailModal from './components/AlertDetailModal';
import SettingsPage from './components/SettingsPage';
import SensorPlacementEditor from './components/SensorPlacementEditor';
import AddStructureModal from './components/AddStructureModal';
import { useTelemetryStream } from './hooks/useTelemetryStream';
import { API_BASE_URL } from './config';
import './index.css';

export default function App() {
  const [activeView, setActiveView] = useState('twin'); // 'twin' | 'fleet' | 'alerts' | 'settings'
  const [activeStructureId, setActiveStructureId] = useState('kw51');
  const [structures, setStructures] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [selectedAlertForDetail, setSelectedAlertForDetail] = useState(null);

  // Modals
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isSensorEditorOpen, setIsSensorEditorOpen] = useState(false);

  // Real-time WebSocket Telemetry Stream isolated by activeStructureId
  const { isConnected, payload, nodeHistory } = useTelemetryStream(activeStructureId);

  // ── Fetch Structures from Backend ──
  const fetchStructures = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/structures`);
      if (res.ok) {
        const data = await res.json();
        const rawList = Array.isArray(data) ? data : (data.structures || []);
        const merged = rawList.map(s => {
          try {
            const stored = localStorage.getItem(`structureiq_sensors_${s.structure_id}`);
            if (stored) {
              const customSensors = JSON.parse(stored);
              if (Array.isArray(customSensors) && customSensors.length > 0) {
                return { ...s, sensors: customSensors };
              }
            }
          } catch (e) {}
          return s;
        });
        setStructures(merged);
      }
    } catch (err) {
      console.warn('Backend /structures fetch notice:', err.message);
    }
  }, []);

  // ── Fetch Alerts Audit Log from Backend ──
  const fetchAlerts = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/alerts`);
      if (res.ok) {
        const data = await res.json();
        setAlerts(Array.isArray(data) ? data : (data.alerts || []));
      }
    } catch (err) {
      console.warn('Backend /alerts fetch notice:', err.message);
    }
  }, []);

  useEffect(() => {
    fetchStructures();
    fetchAlerts();
    const sInt = setInterval(fetchStructures, 5000);
    const aInt = setInterval(fetchAlerts, 4000);
    return () => {
      clearInterval(sInt);
      clearInterval(aInt);
    };
  }, [fetchStructures, fetchAlerts]);

  // Acknowledge alert handler
  const handleAcknowledgeAlert = async (alertId) => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/alerts/${alertId}/acknowledge`, {
        method: 'POST',
      });
      if (res.ok) {
        fetchAlerts();
      }
    } catch (err) {
      console.error('Failed to acknowledge alert:', err);
    }
  };

  const systemStatus = payload?.system_status ?? 'NORMAL';
  const unackAlertCount = alerts.filter(a => !a.acknowledged).length;
  const currentStructure = structures.find(s => s.structure_id === activeStructureId);

  return (
    <AppShell
      currentView={activeView}
      onNavigate={setActiveView}
      structures={structures}
      activeStructureId={activeStructureId}
      onSelectStructure={setActiveStructureId}
      systemStatus={systemStatus}
      isConnected={isConnected}
      alertCount={unackAlertCount}
      onOpenAddModal={() => setIsAddModalOpen(true)}
    >
      {/* ── View 1: Live 3D Digital Twin ── */}
      {activeView === 'twin' && (
        <LiveTwin
          payload={payload}
          nodeHistory={nodeHistory}
          isConnected={isConnected}
          structures={structures}
          activeStructureId={activeStructureId}
          onOpenSensorEditor={() => setIsSensorEditorOpen(true)}
          onOpenAddModal={() => setIsAddModalOpen(true)}
        />
      )}

      {/* ── View 2: Structures Overview Fleet Grid ── */}
      {activeView === 'fleet' && (
        <FleetGrid
          structures={structures}
          onSelectStructure={(id) => {
            setActiveStructureId(id);
            setActiveView('twin');
          }}
          onNavigate={setActiveView}
          onOpenAddModal={() => setIsAddModalOpen(true)}
          onRefresh={fetchStructures}
        />
      )}

      {/* ── View 3: Alerts Inbox & Incident Console ── */}
      {activeView === 'alerts' && (
        <AlertsInbox
          alerts={alerts}
          alertStats={{
            total: alerts.length,
            critical: alerts.filter(a => a.severity === 'CRITICAL').length,
            warning: alerts.filter(a => a.severity === 'WARNING').length,
            unacknowledged: unackAlertCount,
          }}
          onAcknowledgeAlert={handleAcknowledgeAlert}
          onOpenAlertDetail={setSelectedAlertForDetail}
          onRefresh={fetchAlerts}
        />
      )}

      {/* ── View 4: Settings & System Configuration ── */}
      {activeView === 'settings' && (
        <SettingsPage />
      )}

      {/* ── Modal: Add Structure Commissioning Wizard ── */}
      <AddStructureModal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        onCreated={(newStructureId) => {
          setActiveStructureId(newStructureId);
          setActiveView('twin');
          fetchStructures();
        }}
      />

      {/* ── Modal / Drawer: Sensor Placement 3D Editor ── */}
      {isSensorEditorOpen && (
        <SensorPlacementEditor
          structure={currentStructure}
          onClose={() => setIsSensorEditorOpen(false)}
          onSave={async (updatedSensors) => {
            // Save to localStorage for instant client-side persistence
            try {
              localStorage.setItem(`structureiq_sensors_${activeStructureId}`, JSON.stringify(updatedSensors));
            } catch (e) {
              console.warn('localStorage save warning:', e);
            }

            // Optimistic update so 3D digital twin renders new positions instantly
            setStructures(prev => {
              const found = prev.some(s => s.structure_id === activeStructureId);
              if (found) {
                return prev.map(s =>
                  s.structure_id === activeStructureId ? { ...s, sensors: updatedSensors } : s
                );
              }
              return [
                ...prev,
                {
                  structure_id: activeStructureId,
                  name: 'Suspension Bridge Prototype',
                  sensors: updatedSensors,
                },
              ];
            });

            try {
              await fetch(`${API_BASE_URL}/api/v1/structures/${activeStructureId}/sensors`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(updatedSensors),
              });
              fetchStructures();
            } catch (err) {
              console.error('Save sensors backend error:', err);
            }
          }}
        />
      )}

      {/* ── Modal / Drawer: Alert Detail & AI Diagnosis (#EV-4092) ── */}
      {selectedAlertForDetail && (
        <AlertDetailModal
          alert={selectedAlertForDetail}
          onClose={() => setSelectedAlertForDetail(null)}
          onAcknowledge={handleAcknowledgeAlert}
        />
      )}
    </AppShell>
  );
}

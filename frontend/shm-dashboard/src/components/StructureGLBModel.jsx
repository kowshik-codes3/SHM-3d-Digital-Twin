/**
 * StructureGLBModel.jsx
 * Dynamic GLB model loader for Digital Twin canvas and sensor placement editor.
 * 
 * Features:
 * - Loads model from /api/v1/structures/{id}/model using Drei's useGLTF
 * - Traverses meshes, preserves semantic names
 * - Tints parts dynamically by sensor health (assigned mesh_name or nearest sensor)
 * - Raycasts surface clicks for click-to-place sensor placement
 * - Computes structure bounding box and scales sensor sphere markers proportionally
 */

import React, { useMemo, useEffect, useRef } from 'react';
import { useGLTF } from '@react-three/drei';
import * as THREE from 'three';

const STATUS_COLORS = {
  NORMAL:   { color: '#475569', emissive: '#000000', emissiveIntensity: 0 },
  WARNING:  { color: '#f59e0b', emissive: '#b45309', emissiveIntensity: 0.5 },
  CRITICAL: { color: '#ef4444', emissive: '#b91c1c', emissiveIntensity: 0.9 },
  LOW_CONFIDENCE: { color: '#38bdf8', emissive: '#0284c7', emissiveIntensity: 0.6 },
  SENSOR_FAULT:   { color: '#c084fc', emissive: '#7e22ce', emissiveIntensity: 0.7 },
};

export default function StructureGLBModel({
  modelUrl,
  sensorStatuses = {}, // { [sensorId or channelName]: 'NORMAL' | 'WARNING' | 'CRITICAL' }
  sensors = [],        // [ { id, name, position_xyz, mesh_name }, ... ]
  onSurfaceClick,      // (point: [x, y, z], meshName: string) => void
  onBoundsComputed,    // (bounds: { min, max, dimensions, radius }) => void
  highlightMeshName,
}) {
  const { scene } = useGLTF(modelUrl);
  const clonedScene = useMemo(() => scene.clone(true), [scene]);
  const meshMapRef = useRef(new Map());

  // Compute scene bounding box once
  useEffect(() => {
    if (!clonedScene) return;

    const box = new THREE.Box3().setFromObject(clonedScene);
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    const min = [box.min.x, box.min.y, box.min.z];
    const max = [box.max.x, box.max.y, box.max.z];
    const dimensions = [box.max.x - box.min.x, box.max.y - box.min.y, box.max.z - box.min.z];

    if (onBoundsComputed) {
      onBoundsComputed({
        min,
        max,
        dimensions,
        radius: sphere.radius || 10.0,
      });
    }
  }, [clonedScene, onBoundsComputed]);

  // Map sensor health statuses to meshes
  useEffect(() => {
    if (!clonedScene) return;

    // Build sensor lookup by mesh_name and by position
    const meshToSensor = new Map();
    const sensorPositions = [];

    sensors.forEach((s) => {
      const status = sensorStatuses[s.id] || sensorStatuses[s.name] || 'NORMAL';
      if (s.mesh_name) {
        meshToSensor.set(s.mesh_name.toLowerCase(), status);
      }
      const pos = s.position_xyz || s.coordinates_3d;
      if (pos && Array.isArray(pos) && pos.length === 3) {
        sensorPositions.push({
          pos: new THREE.Vector3(pos[0], pos[1], pos[2]),
          status,
        });
      }
    });

    clonedScene.traverse((child) => {
      if (!child.isMesh) return;

      const meshName = (child.name || child.userData?.name || '').toLowerCase();
      let partStatus = 'NORMAL';

      // 1. Direct assignment match
      if (meshToSensor.has(meshName)) {
        partStatus = meshToSensor.get(meshName);
      } else if (sensorPositions.length > 0) {
        // 2. Nearest sensor match (spatial proximity)
        const meshCenter = new THREE.Vector3();
        if (child.geometry?.boundingSphere) {
          meshCenter.copy(child.geometry.boundingSphere.center);
        } else {
          child.geometry?.computeBoundingSphere();
          if (child.geometry?.boundingSphere) {
            meshCenter.copy(child.geometry.boundingSphere.center);
          }
        }
        child.localToWorld(meshCenter);

        let nearestDist = Infinity;
        sensorPositions.forEach((sp) => {
          const dist = meshCenter.distanceTo(sp.pos);
          if (dist < nearestDist) {
            nearestDist = dist;
            partStatus = sp.status;
          }
        });
      }

      // Check if this mesh is highlighted in sensor editor
      const isHighlighted = highlightMeshName && (
        child.name === highlightMeshName || meshName === highlightMeshName.toLowerCase()
      );

      // Mutate material
      const config = STATUS_COLORS[partStatus] || STATUS_COLORS.NORMAL;
      if (!child.material || child.material.isCloned !== true) {
        child.material = child.material ? child.material.clone() : new THREE.MeshStandardMaterial();
        child.material.isCloned = true;
      }

      if (isHighlighted) {
        child.material.color = new THREE.Color('#38bdf8');
        child.material.emissive = new THREE.Color('#0284c7');
        child.material.emissiveIntensity = 0.8;
      } else if (partStatus !== 'NORMAL') {
        child.material.color = new THREE.Color(config.color);
        child.material.emissive = new THREE.Color(config.emissive);
        child.material.emissiveIntensity = config.emissiveIntensity;
      } else {
        child.material.color = new THREE.Color('#374151');
        child.material.emissive = new THREE.Color('#000000');
        child.material.emissiveIntensity = 0;
      }
    });
  }, [clonedScene, sensors, sensorStatuses, highlightMeshName]);

  const handleClick = (e) => {
    e.stopPropagation();
    if (!onSurfaceClick) return;

    const clickedPoint = [
      Math.round(e.point.x * 100) / 100,
      Math.round(e.point.y * 100) / 100,
      Math.round(e.point.z * 100) / 100,
    ];
    const meshName = e.object?.name || 'Unnamed_Part';
    onSurfaceClick(clickedPoint, meshName);
  };

  return (
    <primitive
      object={clonedScene}
      onClick={handleClick}
      castShadow
      receiveShadow
    />
  );
}

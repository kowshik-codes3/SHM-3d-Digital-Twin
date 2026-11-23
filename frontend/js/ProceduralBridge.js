/**
 * ProceduralBridge.js
 * 
 * Programmatic 3D Bridge Generator & Dynamic Health Shader Controller for Three.js.
 * 
 * Spans X ∈ [-12m, +12m], Z ∈ [0, 5m], Y ∈ [-2m, +2m].
 * Tags structural nodes 1:1 with ML sensor channel IDs:
 *   - Sensor_Abutment_Left
 *   - Sensor_Pier_Left
 *   - Sensor_MidSpan_Girder
 *   - Sensor_Pier_Right
 *   - Sensor_Abutment_Right
 *   - Sensor_Ambient_Temp
 */

export class ProceduralBridgeBuilder {
    /**
     * @param {typeof THREE} THREE - Three.js library instance
     */
    constructor(THREE) {
        this.THREE = THREE;
        this.group = new THREE.Group();
        this.group.name = "BridgeStructure";

        // Map sensor_id -> { mesh, markerMesh, material, baselineColor, baselineEmissive, status, pulseTime }
        this.taggedNodes = new Map();
        
        // Sensor channel definitions & coordinates
        this.SENSOR_DEFINITIONS = [
            {
                id: "Sensor_Abutment_Left",
                pos: [-12.0, 2.0, 0.0],
                type: "abutment",
                label: "Left Abutment"
            },
            {
                id: "Sensor_Pier_Left",
                pos: [-4.0, 2.0, 0.0],
                type: "pier",
                label: "Left Pier"
            },
            {
                id: "Sensor_MidSpan_Girder",
                pos: [0.0, 4.3, 0.0],
                type: "girder",
                label: "Mid-Span Girder"
            },
            {
                id: "Sensor_Pier_Right",
                pos: [4.0, 2.0, 0.0],
                type: "pier",
                label: "Right Pier"
            },
            {
                id: "Sensor_Abutment_Right",
                pos: [12.0, 2.0, 0.0],
                type: "abutment",
                label: "Right Abutment"
            },
            {
                id: "Sensor_Ambient_Temp",
                pos: [-12.0, 4.8, 0.0],
                type: "weather_station",
                label: "Weather Station"
            }
        ];

        // Color definitions for status states
        this.STATUS_COLORS = {
            NORMAL: {
                color: null, // Will use baseline
                emissive: 0x000000,
                intensity: 0.0
            },
            WARNING: {
                color: 0xed8936, // Amber / Orange
                emissive: 0x331c00,
                intensity: 0.4
            },
            CRITICAL: {
                color: 0xe53e3e, // Crimson Red
                emissive: 0x660000,
                intensity: 0.8
            }
        };

        this._buildBridge();
    }

    /**
     * Constructs the full 3D bridge assembly programmatically.
     */
    _buildBridge() {
        const THREE = this.THREE;

        // Base materials
        const concreteBaseMat = new THREE.MeshStandardMaterial({
            color: 0x70757a,
            roughness: 0.7,
            metalness: 0.1
        });

        const steelDeckMat = new THREE.MeshStandardMaterial({
            color: 0x4a5568,
            roughness: 0.4,
            metalness: 0.6
        });

        // 1. Foundation Ground Pads
        const padGeo = new THREE.BoxGeometry(26, 0.4, 6);
        const padMesh = new THREE.Mesh(padGeo, concreteBaseMat.clone());
        padMesh.position.set(0, -0.2, 0);
        padMesh.receiveShadow = true;
        this.group.add(padMesh);

        // 2. Concrete Piers & Abutments
        const pierLocations = [
            { id: "Sensor_Abutment_Left", x: -12.0, w: 2.2, h: 4.0, d: 4.4 },
            { id: "Sensor_Pier_Left",     x: -4.0,  w: 1.6, h: 4.0, d: 4.0 },
            { id: "Sensor_Pier_Right",    x: 4.0,   w: 1.6, h: 4.0, d: 4.0 },
            { id: "Sensor_Abutment_Right",x: 12.0,  w: 2.2, h: 4.0, d: 4.4 }
        ];

        pierLocations.forEach(p => {
            const mat = new THREE.MeshStandardMaterial({
                color: 0x70757a,
                roughness: 0.7,
                metalness: 0.1
            });

            const geo = new THREE.BoxGeometry(p.w, p.h, p.d);
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.set(p.x, p.h / 2, 0);
            mesh.name = p.id;
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            this.group.add(mesh);

            // Register in taggedNodes
            this.taggedNodes.set(p.id, {
                mesh: mesh,
                markerMesh: null,
                material: mat,
                baselineColor: mat.color.clone(),
                baselineEmissive: new THREE.Color(0x000000),
                status: "NORMAL",
                pulseTime: 0
            });
        });

        // 3. Steel Deck Assembly (Girders + Slab)
        const mainGirderMat = new THREE.MeshStandardMaterial({
            color: 0x4a5568,
            roughness: 0.4,
            metalness: 0.6
        });

        // Main Longitudinal Girder - Tagged as Sensor_MidSpan_Girder
        const girderGeo = new THREE.BoxGeometry(26.0, 0.6, 3.6);
        const girderMesh = new THREE.Mesh(girderGeo, mainGirderMat);
        girderMesh.position.set(0, 4.3, 0);
        girderMesh.name = "Sensor_MidSpan_Girder";
        girderMesh.castShadow = true;
        girderMesh.receiveShadow = true;
        this.group.add(girderMesh);

        this.taggedNodes.set("Sensor_MidSpan_Girder", {
            mesh: girderMesh,
            markerMesh: null,
            material: mainGirderMat,
            baselineColor: mainGirderMat.color.clone(),
            baselineEmissive: new THREE.Color(0x000000),
            status: "NORMAL",
            pulseTime: 0
        });

        // Road Deck Slab on top of girders
        const deckMat = new THREE.MeshStandardMaterial({
            color: 0x2d3748,
            roughness: 0.8,
            metalness: 0.2
        });
        const slabGeo = new THREE.BoxGeometry(26.4, 0.2, 4.2);
        const slabMesh = new THREE.Mesh(slabGeo, deckMat);
        slabMesh.position.set(0, 4.7, 0);
        slabMesh.castShadow = true;
        slabMesh.receiveShadow = true;
        this.group.add(slabMesh);

        // Guardrails & Markings
        const railGeo = new THREE.BoxGeometry(26.4, 0.4, 0.1);
        const railMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, metalness: 0.8 });
        const leftRail = new THREE.Mesh(railGeo, railMat);
        leftRail.position.set(0, 5.0, 2.05);
        const rightRail = new THREE.Mesh(railGeo, railMat);
        rightRail.position.set(0, 5.0, -2.05);
        this.group.add(leftRail);
        this.group.add(rightRail);

        // 4. Weather Station Node (Sensor_Ambient_Temp)
        const weatherMat = new THREE.MeshStandardMaterial({
            color: 0x319795,
            roughness: 0.3,
            metalness: 0.5
        });
        const poleGeo = new THREE.CylinderGeometry(0.08, 0.08, 1.2, 12);
        const poleMesh = new THREE.Mesh(poleGeo, weatherMat);
        poleMesh.position.set(-12.0, 4.6, 0.0);
        poleMesh.castShadow = true;
        this.group.add(poleMesh);

        const stationHeadGeo = new THREE.SphereGeometry(0.25, 16, 16);
        const stationMesh = new THREE.Mesh(stationHeadGeo, weatherMat);
        stationMesh.position.set(-12.0, 5.2, 0.0);
        stationMesh.name = "Sensor_Ambient_Temp";
        stationMesh.castShadow = true;
        this.group.add(stationMesh);

        this.taggedNodes.set("Sensor_Ambient_Temp", {
            mesh: stationMesh,
            markerMesh: null,
            material: weatherMat,
            baselineColor: weatherMat.color.clone(),
            baselineEmissive: new THREE.Color(0x000000),
            status: "NORMAL",
            pulseTime: 0
        });

        // 5. Attach Glowing Sensor Spheres/Markers to each node
        this.SENSOR_DEFINITIONS.forEach(def => {
            const markerMat = new THREE.MeshStandardMaterial({
                color: 0x3182ce,
                emissive: 0x1a365d,
                emissiveIntensity: 0.5,
                roughness: 0.2,
                metalness: 0.8
            });
            const markerGeo = new THREE.SphereGeometry(0.35, 24, 24);
            const markerMesh = new THREE.Mesh(markerGeo, markerMat);
            markerMesh.position.set(def.pos[0], def.pos[1], def.pos[2] + 2.2);
            markerMesh.name = `${def.id}_Marker`;
            markerMesh.castShadow = true;
            this.group.add(markerMesh);

            const nodeData = this.taggedNodes.get(def.id);
            if (nodeData) {
                nodeData.markerMesh = markerMesh;
                nodeData.markerMat = markerMat;
                nodeData.baselineMarkerColor = markerMat.color.clone();
                nodeData.baselineMarkerEmissive = markerMat.emissive.clone();
            }
        });
    }

    /**
     * Updates the structural health status of a tagged node.
     * 
     * @param {string} nodeId - Sensor channel ID (e.g. "Sensor_MidSpan_Girder")
     * @param {"NORMAL"|"WARNING"|"CRITICAL"} status - Health status
     * @param {number} [mseScore=0.0] - Optional MSE reconstruction error
     */
    updateNodeHealth(nodeId, status, mseScore = 0.0) {
        const THREE = this.THREE;
        const node = this.taggedNodes.get(nodeId);
        if (!node) {
            console.warn(`[ProceduralBridge] Node '${nodeId}' not found in tagged hierarchy.`);
            return;
        }

        node.status = status;
        node.mseScore = mseScore;

        const targetConfig = this.STATUS_COLORS[status] || this.STATUS_COLORS.NORMAL;

        if (status === "NORMAL") {
            node.material.color.copy(node.baselineColor);
            node.material.emissive.setHex(0x000000);
            node.material.emissiveIntensity = 0.0;

            if (node.markerMat) {
                node.markerMat.color.copy(node.baselineMarkerColor);
                node.markerMat.emissive.copy(node.baselineMarkerEmissive);
                node.markerMat.emissiveIntensity = 0.5;
            }
        } else {
            const col = new THREE.Color(targetConfig.color);
            const em = new THREE.Color(targetConfig.emissive);

            node.material.color.copy(col);
            node.material.emissive.copy(em);
            node.material.emissiveIntensity = targetConfig.intensity;

            if (node.markerMat) {
                node.markerMat.color.copy(col);
                node.markerMat.emissive.copy(em);
                node.markerMat.emissiveIntensity = targetConfig.intensity * 1.5;
            }
        }
    }

    /**
     * Animates dynamic pulsing effects for nodes in CRITICAL status.
     * Call this inside the main WebGL animation loop.
     * 
     * @param {number} deltaTime - Time elapsed since last frame in seconds
     */
    tick(deltaTime) {
        const THREE = this.THREE;
        this.taggedNodes.forEach((node) => {
            if (node.status === "CRITICAL") {
                node.pulseTime += deltaTime;
                // 2 Hz pulse frequency (2 * 2π ≈ 12.56 rad/s)
                const pulse = (Math.sin(node.pulseTime * 12.56) + 1.0) / 2.0; // [0.0, 1.0]

                // Modulate emissive intensity between 0.3 and 1.5
                const intensity = 0.3 + pulse * 1.2;
                node.material.emissiveIntensity = intensity;

                if (node.markerMat) {
                    node.markerMat.emissiveIntensity = intensity * 1.8;
                    // Subtle scale pulse on sensor marker sphere
                    const scale = 1.0 + pulse * 0.25;
                    node.markerMesh.scale.set(scale, scale, scale);
                }
            } else if (node.markerMesh && node.markerMesh.scale.x !== 1.0) {
                node.markerMesh.scale.set(1.0, 1.0, 1.0);
            }
        });
    }

    /**
     * Returns the root Three.js Group containing the bridge scene tree.
     * @returns {THREE.Group}
     */
    getSceneObject() {
        return this.group;
    }

    /**
     * Clean up geometries & materials to prevent WebGL memory leaks.
     */
    dispose() {
        this.group.traverse((child) => {
            if (child.isMesh) {
                if (child.geometry) child.geometry.dispose();
                if (child.material) {
                    if (Array.isArray(child.material)) {
                        child.material.forEach(m => m.dispose());
                    } else {
                        child.material.dispose();
                    }
                }
            }
        });
        this.taggedNodes.clear();
    }
}

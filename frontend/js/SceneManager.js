/**
 * SceneManager.js
 * 
 * Three.js WebGL Scene Manager & Canvas Host.
 * Manages Perspective Camera, OrbitControls, Directional Lighting, Shadow Maps,
 * Dark-mode Grid, Animation Ticker, and Window Resize handling.
 */

import { ProceduralBridgeBuilder } from './ProceduralBridge.js';

export class SceneManager {
    /**
     * @param {HTMLCanvasElement|HTMLElement} container - DOM element or canvas host
     * @param {typeof THREE} THREE - Three.js library instance
     * @param {typeof OrbitControls} OrbitControls - Three.js OrbitControls class
     */
    constructor(container, THREE, OrbitControls) {
        this.container = container;
        this.THREE = THREE;
        this.OrbitControls = OrbitControls;

        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.bridgeBuilder = null;

        this.clock = new THREE.Clock();
        this.animFrameId = null;
        this.isDestroyed = false;

        this._initScene();
    }

    _initScene() {
        const THREE = this.THREE;

        // 1. Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a202c); // Dark mode Slate (#1a202c)
        this.scene.fog = new THREE.FogExp2(0x1a202c, 0.015);

        // 2. Camera setup
        const aspect = this._getAspect();
        this.camera = new THREE.PerspectiveCamera(50, aspect, 0.1, 1000);
        this.camera.position.set(18, 12, 22);

        // 3. Renderer setup
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: false,
            powerPreference: "high-performance"
        });
        this.renderer.setSize(this._getWidth(), this._getHeight());
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        if (THREE.SRGBColorSpace) {
            this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        }

        // Attach canvas element
        if (this.container instanceof HTMLCanvasElement) {
            // Container is canvas itself
        } else {
            this.container.appendChild(this.renderer.domElement);
        }

        // 4. OrbitControls setup
        if (this.OrbitControls) {
            this.controls = new this.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.05;
            this.controls.target.set(0, 3.5, 0);
            this.controls.minDistance = 5;
            this.controls.maxDistance = 80;
            this.controls.maxPolarAngle = Math.PI / 2 + 0.05; // Ground collision limit
            this.controls.update();
        }

        // 5. Lighting Setup
        // Primary Directional Sun Light
        const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
        dirLight.position.set(20, 30, 15);
        dirLight.castShadow = true;
        dirLight.shadow.mapSize.width = 2048;
        dirLight.shadow.mapSize.height = 2048;
        dirLight.shadow.camera.near = 0.5;
        dirLight.shadow.camera.far = 80;
        dirLight.shadow.camera.left = -20;
        dirLight.shadow.camera.right = 20;
        dirLight.shadow.camera.top = 15;
        dirLight.shadow.camera.bottom = -15;
        dirLight.shadow.bias = -0.0005;
        this.scene.add(dirLight);

        // Soft Ambient Fill Light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
        this.scene.add(ambientLight);

        // Hemisphere Sky/Ground Light
        const hemiLight = new THREE.HemisphereLight(0x87ceeb, 0x2d3748, 0.6);
        hemiLight.position.set(0, 50, 0);
        this.scene.add(hemiLight);

        // 6. Dark Mode Structural Grid Helper
        const gridHelper = new THREE.GridHelper(60, 60, 0x4a5568, 0x2d3748);
        gridHelper.position.set(0, -0.01, 0);
        this.scene.add(gridHelper);

        // 7. Instantiate Procedural Bridge Geometry
        this.bridgeBuilder = new ProceduralBridgeBuilder(THREE);
        this.scene.add(this.bridgeBuilder.getSceneObject());

        // 8. Event Listeners & Start Loop
        this._onResize = this._onResize.bind(this);
        window.addEventListener('resize', this._onResize);

        this._animate = this._animate.bind(this);
        this.clock.start();
        this._animate();
    }

    _getWidth() {
        return this.container.clientWidth || window.innerWidth;
    }

    _getHeight() {
        return this.container.clientHeight || window.innerHeight;
    }

    _getAspect() {
        return this._getWidth() / Math.max(this._getHeight(), 1);
    }

    _onResize() {
        if (this.isDestroyed || !this.renderer || !this.camera) return;
        const width = this._getWidth();
        const height = this._getHeight();

        this.camera.aspect = width / Math.max(height, 1);
        this.camera.updateProjectionMatrix();

        this.renderer.setSize(width, height);
    }

    _animate() {
        if (this.isDestroyed) return;

        this.animFrameId = requestAnimationFrame(this._animate);

        const delta = this.clock.getDelta();

        if (this.controls) {
            this.controls.update();
        }

        if (this.bridgeBuilder) {
            this.bridgeBuilder.tick(delta);
        }

        this.renderer.render(this.scene, this.camera);
    }

    /**
     * Updates structural health status of a tagged node in the 3D scene.
     * 
     * @param {string} nodeId - Sensor channel ID (e.g. "Sensor_MidSpan_Girder")
     * @param {"NORMAL"|"WARNING"|"CRITICAL"} status - Health status
     * @param {number} [mseScore=0.0] - Optional MSE score
     */
    updateNodeHealth(nodeId, status, mseScore = 0.0) {
        if (this.bridgeBuilder) {
            this.bridgeBuilder.updateNodeHealth(nodeId, status, mseScore);
        }
    }

    /**
     * Resets all nodes in the 3D bridge to NORMAL status.
     */
    resetAllNodes() {
        if (this.bridgeBuilder) {
            this.bridgeBuilder.taggedNodes.forEach((_, nodeId) => {
                this.bridgeBuilder.updateNodeHealth(nodeId, "NORMAL", 0.0);
            });
        }
    }

    /**
     * Retrieves sub-mesh or marker by ML sensor channel ID.
     * @param {string} nodeId 
     * @returns {THREE.Object3D|null}
     */
    getNodeObject(nodeId) {
        return this.scene.getObjectByName(nodeId);
    }

    /**
     * Clean up WebGL resources on unmount.
     */
    dispose() {
        this.isDestroyed = true;
        if (this.animFrameId) {
            cancelAnimationFrame(this.animFrameId);
        }

        window.removeEventListener('resize', this._onResize);

        if (this.bridgeBuilder) {
            this.bridgeBuilder.dispose();
        }

        if (this.renderer) {
            this.renderer.dispose();
            if (this.renderer.domElement && this.renderer.domElement.parentNode) {
                this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
            }
        }
    }
}

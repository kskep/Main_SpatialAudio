import { Room, RoomConfig } from './room/room';
import { Surface } from './room/types';
import { Camera } from './camera/camera';
import { vec3 } from 'gl-matrix';
import * as dat from 'dat.gui';
import { Sphere } from './objects/sphere';
import { SphereRenderer } from './objects/sphere-renderer';
import { RayTracer } from './raytracer/raytracer';
import { AudioProcessorModified } from './sound/audio-processor_modified'; // Use modified version
import { WaveformRenderer } from './visualization/waveform-renderer';

export class Main {
    private canvas: HTMLCanvasElement;
    private soundFiles: { [key: string]: string } = {
        'Snare': '/src/soundfile/snare.wav',
        'Loop': '/src/soundfile/loop.wav',
        'Top Loop': '/src/soundfile/top_loop.wav'
    };
    private currentAudioBuffer: AudioBuffer | null = null;
    private device: GPUDevice;
    private audioCtx: AudioContext; // Add AudioContext property
    private context: GPUCanvasContext;
    private room: Room;
    private camera: Camera;
    private depthTexture!: GPUTexture;
    private keys: { [key: string]: boolean } = {};
    private roomConfig: RoomConfig;
    private gui!: dat.GUI; // Add definite assignment assertion
    private sphere: Sphere;
    private sphereRenderer: SphereRenderer;
    private sourceControllers!: { // Add definite assignment assertion
        x: dat.GUIController;
        y: dat.GUIController;
        z: dat.GUIController;
    };
    private rayTracer: RayTracer;
    private audioProcessor: AudioProcessorModified; // Use modified version
    private waveformRenderer: WaveformRenderer;
    private sourceParams = {
        sourcePower: 0
    };

    constructor(canvas: HTMLCanvasElement, device: GPUDevice) {
        this.canvas = canvas;
        this.device = device;
        this.context = canvas.getContext('webgpu') as GPUCanvasContext;
        this.audioCtx = new AudioContext(); // Create AudioContext

        // Configure the canvas context
        this.context.configure({
            device: this.device,
            format: navigator.gpu.getPreferredCanvasFormat(),
            alphaMode: 'premultiplied',
        });

        // Initialize room config
        this.roomConfig = {
            dimensions: { width: 8, height: 3, depth: 5 },
            materials: {
                walls: {
                    absorption125Hz: 0.10,
                    absorption250Hz: 0.10,
                    absorption500Hz: 0.10,
                    absorption1kHz: 0.10,
                    absorption2kHz: 0.10,
                    absorption4kHz: 0.10,
                    absorption8kHz: 0.10,
                    absorption16kHz: 0.10,
                    scattering125Hz: 0.10,
                    scattering250Hz: 0.10,
                    scattering500Hz: 0.10,
                    scattering1kHz: 0.10,
                    scattering2kHz: 0.10,
                    scattering4kHz: 0.10,
                    scattering8kHz: 0.10,
                    scattering16kHz: 0.10,
                    roughness: 0.1,
                    phaseShift: 0,
                    phaseRandomization: 0.1
                },
                ceiling: {
                    absorption125Hz: 0.15,
                    absorption250Hz: 0.15,
                    absorption500Hz: 0.15,
                    absorption1kHz: 0.15,
                    absorption2kHz: 0.15,
                    absorption4kHz: 0.15,
                    absorption8kHz: 0.15,
                    absorption16kHz: 0.15,
                    scattering125Hz: 0.10,
                    scattering250Hz: 0.10,
                    scattering500Hz: 0.10,
                    scattering1kHz: 0.10,
                    scattering2kHz: 0.10,
                    scattering4kHz: 0.10,
                    scattering8kHz: 0.10,
                    scattering16kHz: 0.10,
                    roughness: 0.1,
                    phaseShift: 0,
                    phaseRandomization: 0.1
                },
                floor: {
                    absorption125Hz: 0.05,
                    absorption250Hz: 0.05,
                    absorption500Hz: 0.05,
                    absorption1kHz: 0.05,
                    absorption2kHz: 0.05,
                    absorption4kHz: 0.05,
                    absorption8kHz: 0.05,
                    absorption16kHz: 0.05,
                    scattering125Hz: 0.20,
                    scattering250Hz: 0.20,
                    scattering500Hz: 0.20,
                    scattering1kHz: 0.20,
                    scattering2kHz: 0.20,
                    scattering4kHz: 0.20,
                    scattering8kHz: 0.20,
                    scattering16kHz: 0.20,
                    roughness: 0.2,
                    phaseShift: 0,
                    phaseRandomization: 0.2
                }
            }
        };

        // Initialize room
        this.room = new Room(device, this.roomConfig);

        // Initialize camera near the back wall of the room
        this.camera = new Camera(
            vec3.fromValues(0, 1.7, 3), // Start near back wall (Z=3 is inside Z=4 boundary)
            -90,  // Looking toward the center
            0     // Level view
        );

        // Initialize sphere in the middle of the room
        this.sphere = new Sphere(
            vec3.fromValues(0, 1.7, 0), // Center of room, eye level
            0.2                         // Smaller radius for sound source
        );

        // Initialize sphere renderer
        this.sphereRenderer = new SphereRenderer(device);

        // Initialize ray tracer
        this.rayTracer = new RayTracer(device, this.sphere, this.room, this.camera);
        this.audioProcessor = new AudioProcessorModified(this.audioCtx, this.room, this.camera, this.audioCtx.sampleRate); // Pass audioCtx and sampleRate

        // Create waveform canvas element and style it to appear at the bottom of the screen.
        const waveformCanvas = document.createElement('canvas');
        waveformCanvas.id = "waveform-canvas";
        waveformCanvas.style.position = "fixed";
        waveformCanvas.style.bottom = "0";
        waveformCanvas.style.left = "0";
        waveformCanvas.style.width = "100%";
        waveformCanvas.style.height = "150px";
        waveformCanvas.style.backgroundColor = "rgba(0, 0, 0, 0.5)"; // Semi-transparent background
        document.body.appendChild(waveformCanvas);

        // Initialize waveform renderer
        this.waveformRenderer = new WaveformRenderer(waveformCanvas);

        // Setup debug UI
        this.setupDebugUI();

        // Setup input handlers
        this.setupInputHandlers();

        // Create depth texture
        this.createDepthTexture();
    }

    private setupDebugUI(): void {
        this.gui = new dat.GUI();

        const roomFolder = this.gui.addFolder('Room Dimensions');
        roomFolder.add(this.roomConfig.dimensions, 'width', 2, 20).onChange(() => this.updateRoom());
        roomFolder.add(this.roomConfig.dimensions, 'height', 2, 10).onChange(() => this.updateRoom());
        roomFolder.add(this.roomConfig.dimensions, 'depth', 2, 20).onChange(() => this.updateRoom());

        const materialsFolder = this.gui.addFolder('Materials');
        
        // Add frequency-dependent absorption controls for walls
        const wallsFolder = materialsFolder.addFolder('Walls');
        wallsFolder.add(this.roomConfig.materials.walls, 'absorption125Hz', 0, 1).name('125Hz Absorption');
        wallsFolder.add(this.roomConfig.materials.walls, 'absorption1kHz', 0, 1).name('1kHz Absorption');
        wallsFolder.add(this.roomConfig.materials.walls, 'absorption8kHz', 0, 1).name('8kHz Absorption');
        
        // Add frequency-dependent absorption controls for ceiling
        const ceilingFolder = materialsFolder.addFolder('Ceiling');
        ceilingFolder.add(this.roomConfig.materials.ceiling, 'absorption125Hz', 0, 1).name('125Hz Absorption');
        ceilingFolder.add(this.roomConfig.materials.ceiling, 'absorption1kHz', 0, 1).name('1kHz Absorption');
        ceilingFolder.add(this.roomConfig.materials.ceiling, 'absorption8kHz', 0, 1).name('8kHz Absorption');
        
        // Add frequency-dependent absorption controls for floor
        const floorFolder = materialsFolder.addFolder('Floor');
        floorFolder.add(this.roomConfig.materials.floor, 'absorption125Hz', 0, 1).name('125Hz Absorption');
        floorFolder.add(this.roomConfig.materials.floor, 'absorption1kHz', 0, 1).name('1kHz Absorption');
        floorFolder.add(this.roomConfig.materials.floor, 'absorption8kHz', 0, 1).name('8kHz Absorption');

        // Create a data object for the sound source position
        const sourcePosition = {
            x: 0,
            y: 1.7,
            z: 0
        };

        // Update sphere controls to allow movement
        const sourceFolder = this.gui.addFolder('Sound Source');

        // Store controller references for updating ranges
        this.sourceControllers = {
            x: sourceFolder.add(sourcePosition, 'x', -this.roomConfig.dimensions.width/2, this.roomConfig.dimensions.width/2)
                .onChange((value: number) => {
                    const pos = this.sphere.getPosition();
                    pos[0] = value;
                    this.sphere.setPosition(pos);
                }),
            y: sourceFolder.add(sourcePosition, 'y', 0, this.roomConfig.dimensions.height)
                .onChange((value: number) => {
                    const pos = this.sphere.getPosition();
                    pos[1] = value;
                    this.sphere.setPosition(pos);
                }),
            z: sourceFolder.add(sourcePosition, 'z', -this.roomConfig.dimensions.depth/2, this.roomConfig.dimensions.depth/2)
                .onChange((value: number) => {
                    const pos = this.sphere.getPosition();
                    pos[2] = value;
                    this.sphere.setPosition(pos);
                })
        };

        // Add source power control
        sourceFolder.add(this.sourceParams, 'sourcePower', -60, 20)
            .name('Power (dB)')
            .onChange((value: number) => {
                this.sourceParams = {
                    ...this.sourceParams,
                    sourcePower: value
                };
                // Recalculate IR when power changes
                this.calculateIR();
            });

        // Add ray tracing controls
        const rayTracingFolder = this.gui.addFolder('Ray Tracing');
        const rayTracingControls = {
            calculateIR: async () => {
                await this.calculateIR();
            }
        };
        rayTracingFolder.add(rayTracingControls, 'calculateIR').name('Calculate IR');

        // Removed old Audio Debug folder as playback is handled by new controls

        roomFolder.open();
        sourceFolder.open();
        rayTracingFolder.open();

        // Add sound file playback controls
        const audioFolder = this.gui.addFolder('Sound Playback');
        const audioControls = {
            selectedFile: 'Snare', // Default selection
            play: async () => {
                if (!audioControls.selectedFile) return;
                try {
                    // Load the audio file if not already loaded or if selection changed
                    const fileUrl = this.soundFiles[audioControls.selectedFile];
                    this.currentAudioBuffer = await this.audioProcessor.loadAudioFile(fileUrl);
                    await this.audioProcessor.playAudioWithIR(this.currentAudioBuffer);
                } catch (error) {
                    console.error('Error playing audio:', error);
                }
            },
            stop: () => {
                this.audioProcessor.stopAllSounds();
            }
        };

        // Add dropdown for sound file selection
        audioFolder.add(audioControls, 'selectedFile', Object.keys(this.soundFiles))
            .name('Sound File');
        
        // Add play/stop buttons
        audioFolder.add(audioControls, 'play').name('Play Sound');
        audioFolder.add(audioControls, 'stop').name('Stop Sound');
        
        audioFolder.open();
    }

    private updateRoom(): void {
        // Recreate room with new dimensions
        this.room = new Room(this.device, this.roomConfig);

        // Ensure camera stays within room bounds
        this.constrainCamera();

        // Update source position slider ranges
        this.sourceControllers.x.min(-this.roomConfig.dimensions.width/2)
            .max(this.roomConfig.dimensions.width/2);
        this.sourceControllers.y.min(0).max(this.roomConfig.dimensions.height);
        this.sourceControllers.z.min(-this.roomConfig.dimensions.depth/2)
            .max(this.roomConfig.dimensions.depth/2);

        // Keep sphere at current position unless it's outside new bounds
        const currentPos = this.sphere.getPosition();
        const validPos = this.room.getClosestValidPosition([currentPos[0], currentPos[1], currentPos[2]]); // Convert vec3 to tuple
        this.sphere.setPosition(validPos);
    }

    private constrainCamera(): void {
        const pos = this.camera.getPosition();
        const { width, height, depth } = this.roomConfig.dimensions;
        const halfWidth = width / 2;
        const halfDepth = depth / 2;
        const margin = 0.5; // Keep camera slightly away from walls

        // Constrain position
        const newPos = vec3.fromValues(
            Math.max(-halfWidth + margin, Math.min(halfWidth - margin, pos[0])),
            Math.max(margin, Math.min(height - margin, pos[1])),
            Math.max(-halfDepth + margin, Math.min(halfDepth - margin, pos[2]))
        );

        this.camera.setPosition(newPos);
    }

    private setupInputHandlers(): void {
        // Keyboard controls
        window.addEventListener('keydown', (e) => this.keys[e.key.toLowerCase()] = true);
        window.addEventListener('keyup', (e) => this.keys[e.key.toLowerCase()] = false);

        // New keyboard rotation controls
        window.addEventListener('keydown', (event) => {
            switch (event.key) {
                case 'ArrowLeft':
                    this.camera.rotateWithKeyboard('left');
                    break;
                case 'ArrowRight':
                    this.camera.rotateWithKeyboard('right');
                    break;
                case 'ArrowUp':
                    this.camera.rotateWithKeyboard('up');
                    break;
                case 'ArrowDown':
                    this.camera.rotateWithKeyboard('down');
                    break;
            }
        });
    }

    private createDepthTexture(): void {
        this.depthTexture = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
    }

    private handleInput(deltaTime: number): void {
        // Apply movement
        if (this.keys['w']) this.camera.moveForward(deltaTime);
        if (this.keys['s']) this.camera.moveForward(-deltaTime);
        if (this.keys['a']) this.camera.moveRight(-deltaTime);
        if (this.keys['d']) this.camera.moveRight(deltaTime);
        if (this.keys[' ']) this.camera.moveUp(deltaTime);
        if (this.keys['shift']) this.camera.moveUp(-deltaTime);
    }

    public render(deltaTime: number): void {
        // Handle input
        this.handleInput(deltaTime);

        // Update room's view projection with camera
        const aspect = this.canvas.width / this.canvas.height;
        const viewProjection = this.camera.getViewProjection(aspect);
        this.device.queue.writeBuffer(
            this.room['uniformBuffer'], // Accessing private member, might need to add a public method
            0,
            viewProjection as Float32Array
        );

        // Begin render pass
        const commandEncoder = this.device.createCommandEncoder();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
            depthStencilAttachment: {
                view: this.depthTexture.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            }
        });

        // Render room
        this.room.render(renderPass);

        // Render sphere
        this.sphereRenderer.render(
            renderPass,
            viewProjection as Float32Array,
            this.sphere.getPosition(),
            this.sphere.getRadius()
        );

        // Render rays
        this.rayTracer.render(renderPass, viewProjection as Float32Array);

        // End render pass and submit
        renderPass.end();
        this.device.queue.submit([commandEncoder.finish()]);
    }

    public resize(): void {
        if (this.canvas.width !== this.canvas.clientWidth ||
            this.canvas.height !== this.canvas.clientHeight) {

            this.canvas.width = this.canvas.clientWidth;
            this.canvas.height = this.canvas.clientHeight;

            // Recreate depth texture with new size
            this.depthTexture.destroy();
            this.createDepthTexture();
        }
    }

    private async calculateIR(): Promise<void> {
        try {
            await this.rayTracer.calculateRayPaths();
            const hits = this.rayTracer.getRayHits();
            
            if (!hits || hits.length === 0) {
                console.warn("No valid ray hits to process");
                return;
            }
            
            await this.audioProcessor.processRayHits(hits);

            // Visualize both waveform and FFT if we have valid hits
            await this.audioProcessor.visualizeImpulseResponse(this.waveformRenderer);
        } catch (error) {
            console.error("Error calculating impulse response:", error);
        }
    }
}

// Animation loop
let lastTime = 0;
function animate(main: Main, time: number) {
    const deltaTime = (time - lastTime) / 1000; // Convert to seconds
    lastTime = time;

    main.resize();
    main.render(deltaTime);
    requestAnimationFrame((time) => animate(main, time));
}

// Initialize and start
async function init() {
    const canvas = document.querySelector('canvas');
    if (!canvas) throw new Error('No canvas element found');

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No GPU adapter found');

    const device = await adapter.requestDevice();
    const main = new Main(canvas, device);

    // Start animation loop
    requestAnimationFrame((time) => animate(main, time));
}

init().catch(console.error);
import { vec3 } from 'gl-matrix';
import { Ray, FrequencyBands } from './ray';
import { Room } from '../room/room';
import { Sphere } from '../objects/sphere';
import { RayRenderer } from './ray-renderer';
import { WallMaterial, MATERIAL_PRESETS } from '../room/room-materials';

export interface RayTracerConfig {
    numRays: number;
    maxBounces: number;
    minEnergy: number;
}

export interface RayHit {
    position: vec3;
    energies: FrequencyBands;
    time: number;      // Arrival time
    phase: number;     // Phase at hit
    frequency: number; // Frequency of the ray
    dopplerShift: number; // Doppler shift at this point
}

export interface RayPathPoint extends RayHit {
    bounceNumber: number;  // Which bounce this point represents
    rayIndex: number;     // Which ray this point belongs to
}

export interface ImpulseResponse {
    time: Float32Array;     // Time points
    amplitude: Float32Array; // Amplitude values
    sampleRate: number;     // Sample rate of the impulse response
    frequencies: Float32Array; // Frequency content at each time point
}

interface ImageSource {
    position: vec3;
    order: number;
    reflectionPath: vec3[];
}

interface RayPath {
    points: vec3[];
    energies: FrequencyBands;
    totalDistance: number;
}

const MAX_POINTS_PER_RAY = 1000;

export class RayTracer {
    private device: GPUDevice;
    private soundSource: Sphere;
    private room: Room;
    private config: RayTracerConfig;
    private rays: Ray[] = [];
    private hits: RayHit[] = [];
    private rayPaths: { origin: vec3, direction: vec3, energies: FrequencyBands }[] = [];
    private rayPathPoints: RayPathPoint[] = [];
    private rayRenderer: RayRenderer;
    private readonly VISIBILITY_THRESHOLD = 0.05;
    private readonly SPEED_OF_SOUND = 343.0;
    private readonly AIR_TEMPERATURE = 20.0;
    private rayPointsBuffer: GPUBuffer;

    constructor(
        device: GPUDevice,
        soundSource: Sphere,
        room: Room,
        config: RayTracerConfig = {
            numRays: 1000,
            maxBounces: 50,
            minEnergy: 0.05
        }
    ) {
        this.device = device;
        this.soundSource = soundSource;
        this.room = room;
        this.config = config;
        this.rayRenderer = new RayRenderer(device);

        // Create rayPoints buffer with enough space for all rays
        const rayPointsBufferSize = 
            config.numRays * // Number of rays
            MAX_POINTS_PER_RAY * // Points per ray
            (4 * 4 + 4 + 4 + 4 + 4); // Size of RayPoint struct (vec3f + 4 floats)

        this.rayPointsBuffer = device.createBuffer({
            size: rayPointsBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
    }

    private generateRays(): void {
        this.rays = [];
        const sourcePos = this.soundSource.getPosition();
        const sphereRadius = this.soundSource.getRadius();

        // Define frequency bands
        const frequencies = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]; // Hz

        console.log('Starting ray generation with:', {
            numRays: this.config.numRays,
            sourcePosition: Array.from(sourcePos),
            sphereRadius
        });

        for (let i = 0; i < this.config.numRays; i++) {
            const theta = 2 * Math.PI * Math.random();
            const phi = Math.acos(2 * Math.random() - 1);

            const direction = vec3.fromValues(
                Math.sin(phi) * Math.cos(theta),
                Math.sin(phi) * Math.sin(theta),
                Math.cos(phi)
            );

            const rayOrigin = vec3.create();
            vec3.scale(rayOrigin, direction, sphereRadius);
            vec3.add(rayOrigin, rayOrigin, sourcePos);

            // Select which frequency band to use for this ray
            const frequency = frequencies[i % frequencies.length];
            
            // Create ray with initial energy for all frequency bands
            const ray = new Ray(rayOrigin, direction, 1.0, frequency);
            
                       
            ray.setEnergies({
                energy125Hz: 1.0,
                energy250Hz: 1.0,
                energy500Hz: 1.0,
                energy1kHz: 1.0,
                energy2kHz: 1.0,
                energy4kHz: 1.0,
                energy8kHz: 1.0,
                energy16kHz: 1.0
            });
            
            
            this.rays.push(ray);
        }
    }

    public async calculateRayPaths(): Promise<void> {
        this.hits = [];
        this.rays = [];
        this.rayPaths = [];
        this.rayPathPoints = [];

        this.generateRays();

        // Store initial ray paths
        for (const ray of this.rays) {
            this.rayPaths.push({
                origin: ray.getOrigin(),
                direction: ray.getDirection(),
                energies: ray.getEnergies()
            });
        }

        await this.calculateLateReflections(); // Simplified: using only late reflections for now

        // Log summary of ray tracing results
        const activeRays = this.rayPaths.filter(path =>
            this.calculateAverageEnergy(path.energies) > this.config.minEnergy
        );

        console.log('Ray tracing summary:', {
            totalRays: this.rays.length,
            activeRayPaths: this.rayPaths.length,
            activeHits: this.hits.length,
            averageEnergy: activeRays.length > 0
                ? activeRays.reduce((sum, path) => sum + this.calculateAverageEnergy(path.energies), 0) / activeRays.length
                : 0
        });

        console.log(`Completed ray tracing`);
    }

    private calculateDopplerShift(rayDirection: vec3, surfaceNormal: vec3): number {
        const relativeVelocity = vec3.dot(rayDirection, surfaceNormal) * this.SPEED_OF_SOUND;
        return this.SPEED_OF_SOUND / (this.SPEED_OF_SOUND - relativeVelocity);
    }

    private async calculateLateReflections(): Promise<void> {
        const { width, height, depth } = this.room.config.dimensions;
        const halfWidth = width / 2;
        const halfDepth = depth / 2;

        console.log('Starting late reflections calculation');
        let activeRayCount = 0;

        // Define room planes
        const planes = [
            { normal: vec3.fromValues(1, 0, 0), d: halfWidth, material: { ...MATERIAL_PRESETS.CONCRETE } },
            { normal: vec3.fromValues(-1, 0, 0), d: halfWidth, material: { ...MATERIAL_PRESETS.CONCRETE } },
            { normal: vec3.fromValues(0, 1, 0), d: 0, material: { ...MATERIAL_PRESETS.CONCRETE } },
            { normal: vec3.fromValues(0, -1, 0), d: height, material: { ...MATERIAL_PRESETS.CONCRETE } },
            { normal: vec3.fromValues(0, 0, 1), d: halfDepth, material: { ...MATERIAL_PRESETS.CONCRETE } },
            { normal: vec3.fromValues(0, 0, -1), d: halfDepth, material: { ...MATERIAL_PRESETS.CONCRETE } }
        ];

        // Process each ray
        for (let rayIndex = 0; rayIndex < this.rays.length; rayIndex++) {
            const ray = this.rays[rayIndex];
            let bounces = 0;
            let currentTime = 0;
            let totalPoints = 0;

            while (ray.isRayActive() &&
                   bounces < this.config.maxBounces &&
                   this.calculateAverageEnergy(ray.getEnergies()) > this.config.minEnergy) {
                let closestT = Infinity;
                let closestPlane = null;
                const origin = ray.getOrigin();
                const direction = ray.getDirection();

                for (const plane of planes) {
                    const denom = vec3.dot(direction, plane.normal);
                    if (Math.abs(denom) > 0.0001) {
                        const t = -(vec3.dot(origin, plane.normal) + plane.d) / denom;
                        if (t > 0.0001 && t < closestT) {
                            closestT = t;
                            closestPlane = plane;
                        }
                    }
                }

                if (closestPlane && closestPlane.material) {
                    const hitPoint = vec3.scaleAndAdd(vec3.create(), origin, direction, closestT - 0.0001);
                    const distanceTraveled = vec3.distance(origin, hitPoint);

                    // Sample points along the ray path before hit
                    const sampleDistance = 0.1; // Sample every 10cm
                    const numSamples = Math.floor(distanceTraveled / sampleDistance);

                    for (let i = 0; i < numSamples; i++) {
                        // Check if we've exceeded points per ray
                        if (totalPoints >= MAX_POINTS_PER_RAY) {
                            console.warn(`Ray ${rayIndex} exceeded maximum points (${MAX_POINTS_PER_RAY})`);
                            break;
                        }

                        const t = i * sampleDistance;
                        const pointPosition = vec3.scaleAndAdd(vec3.create(), origin, direction, t);
                        const pointTime = currentTime + (t / this.SPEED_OF_SOUND);

                        // Calculate wave properties at this point
                        const wavelength = this.SPEED_OF_SOUND / ray.getFrequency();
                        const phaseAtPoint = (ray.getPhase() + (2 * Math.PI * t) / wavelength) % (2 * Math.PI);

                        // Store point data
                        this.rayPathPoints.push({
                            position: pointPosition,
                            energies: ray.getEnergies(),
                            time: pointTime,
                            phase: phaseAtPoint,
                            frequency: ray.getFrequency(),
                            dopplerShift: 1.0,
                            bounceNumber: bounces,
                            rayIndex: rayIndex
                        });

                        totalPoints++;
                    }

                    const travelTime = distanceTraveled / this.SPEED_OF_SOUND;
                    currentTime += travelTime;

                    // Calculate reflection
                    const dot = vec3.dot(direction, closestPlane.normal);
                    const reflected = vec3.create();
                    vec3.scale(reflected, closestPlane.normal, -2 * dot);
                    vec3.add(reflected, direction, reflected);
                    vec3.normalize(reflected, reflected);

                    // Get energy before update
                    const energyBefore = this.calculateAverageEnergy(ray.getEnergies());
                    const materialType = Object.entries(this.room.config.materials).find(
                        ([_, material]) => material === closestPlane.material
                    )?.[0] || 'unknown';

                    // Update ray position and properties
                    const newOrigin = vec3.scaleAndAdd(vec3.create(), hitPoint, closestPlane.normal, 0.0001);
                    ray.updateRay(
                        newOrigin,
                        reflected,
                        closestPlane.material,
                        distanceTraveled,
                        this.AIR_TEMPERATURE,
                        50 // Default humidity
                    );

                    // Get energy after update
                    const energyAfter = this.calculateAverageEnergy(ray.getEnergies());

                    // Log energy state for every 100th ray
                    if (rayIndex % 100 === 0) {
                        console.log(`Ray ${rayIndex}, Bounce ${bounces}:`, {
                            energyBefore,
                            energyAfter,
                            energyLoss: energyBefore - energyAfter,
                            material: materialType,
                            absorption: closestPlane.material.absorption1kHz,
                            distanceTraveled
                        });
                    }

                    if (energyAfter <= this.config.minEnergy) {
                        if (rayIndex % 100 === 0) {
                            console.log(`Ray ${rayIndex} deactivated: energy ${energyAfter} below minimum ${this.config.minEnergy}`);
                        }
                        ray.deactivate();
                    }

                    const wavelength = this.SPEED_OF_SOUND / ray.getFrequency();
                    const phaseChange = (2 * Math.PI * distanceTraveled) / wavelength;
                    const newPhase = (ray.getPhase() + phaseChange) % (2 * Math.PI);

                    const dopplerShift = this.calculateDopplerShift(direction, closestPlane.normal);

                    // Add hit point to ray path points
                    this.rayPathPoints.push({
                        position: vec3.clone(hitPoint),
                        energies: ray.getEnergies(),
                        time: currentTime,
                        phase: newPhase,
                        frequency: ray.getFrequency(),
                        dopplerShift,
                        bounceNumber: bounces,
                        rayIndex: rayIndex
                    });

                    // Store path segment if energy is above threshold
                    if (this.calculateAverageEnergy(ray.getEnergies()) > this.VISIBILITY_THRESHOLD) {
                        this.rayPaths.push({
                            origin: vec3.clone(newOrigin),
                            direction: vec3.clone(reflected),
                            energies: ray.getEnergies()
                        });
                    }

                    // Add to hits array
                    this.hits.push({
                        position: vec3.clone(hitPoint),
                        energies: ray.getEnergies(),
                        time: currentTime,
                        phase: newPhase,
                        frequency: ray.getFrequency(),
                        dopplerShift
                    });

                    bounces++;
                } else {
                    ray.deactivate();
                }
            }
        }
    }

    private calculateAverageEnergy(energies: FrequencyBands): number {
        const values = Object.values(energies);
        return values.reduce((sum, energy) => sum + energy, 0) / values.length;
    }

    public getRayHits(): RayHit[] {
        return this.hits;
    }

    public render(pass: GPURenderPassEncoder, viewProjection: Float32Array): void {
        this.rayRenderer.render(
            pass,
            viewProjection,
            this.rayPaths,
            this.room.config.dimensions
        );
    }

    public recalculateRays(): void {
        // Reset the renderer state
        this.rayRenderer.resetRender();
        // Recalculate ray paths
        this.calculateRayPaths();
    }
}
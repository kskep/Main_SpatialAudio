import { vec3 } from 'gl-matrix';
import { Ray, FrequencyBands } from './ray';
import { Room } from '../room/room';
import { Sphere } from '../objects/sphere';
import { RayRenderer } from './ray-renderer';
import { Camera } from '../camera/camera'; // Assuming Camera class is defined in this file

// New interface for edge detection
interface Edge {
    start: vec3;
    end: vec3;
    adjacentSurfaces: number[]; // Indices of surfaces sharing this edge
}

interface ImageSource {
    position: vec3;
    order: number;
    reflectionPath: vec3[];
    surfaces: number[]; // Surfaces encountered in the reflection path
}

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
    bounces: number;   // Number of bounces
    distance: number;  // Distance from listener
    direction: vec3;   // Direction from listener (normalized vector from listener to hit)
}

export interface RayPathPoint {
    position: vec3; // Added position
    energies: FrequencyBands; // Added energies
    time: number; // Added time
    phase: number; // Added phase
    frequency: number; // Added frequency
    dopplerShift: number; // Added doppler shift
    bounces: number; // Added bounces (renamed from bounceNumber)
    distance: number; // Added distance
    direction: vec3; // Added direction
    rayIndex: number;     // Which ray this point belongs to
}

export interface ImpulseResponse {
    time: Float32Array;     // Time points
    amplitude: Float32Array; // Amplitude values
    sampleRate: number;     // Sample rate of the impulse response
    frequencies: Float32Array; // Frequency content at each time point
}

const MAX_POINTS_PER_RAY = 1000;

export class RayTracer {
    private soundSource: Sphere;
    private room: Room;
    private camera: Camera;
    private config: RayTracerConfig;
    private rays: Ray[] = [];
    private hits: RayHit[] = [];
    private rayPaths: { origin: vec3, direction: vec3, energies: FrequencyBands }[] = [];
    private rayPathPoints: RayPathPoint[] = [];
    private rayRenderer: RayRenderer;
    private readonly VISIBILITY_THRESHOLD = 0.05;
    private readonly SPEED_OF_SOUND = 343.0;
    private readonly AIR_TEMPERATURE = 20.0;
    private edges: Edge[] = [];
    private imageSources: ImageSource[] = [];

    constructor(
        device: GPUDevice,
        soundSource: Sphere,
        room: Room,
        camera: Camera,
        config: RayTracerConfig = {
            numRays: 1000,
            maxBounces: 50,
            minEnergy: 0.05
        }
    ) {
        this.soundSource = soundSource;
        this.room = room;
        this.camera = camera;
        this.config = config;
        this.rayRenderer = new RayRenderer(device);
    }

    private generateRays(): void {
        this.rays = [];
        this.hits = []; // Clear previous hits

        // Use actual camera position for listener
        // const listenerPos = this.camera.getPosition(); // Removed unused variable
        const sourcePos = this.soundSource.getPosition();
        // const directDistance = vec3.distance(listenerPos, sourcePos); // Removed unused variable
        // const directTime = directDistance / this.SPEED_OF_SOUND; // Ensure this is commented out or removed

        // Define frequency bands
        const frequencies = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]; // Hz

        console.log('Starting ray generation with:', {
            numRays: this.config.numRays,
            sourcePosition: Array.from(sourcePos),
            sphereRadius: this.soundSource.getRadius()
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
            vec3.scale(rayOrigin, direction, this.soundSource.getRadius());
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

        // Use actual camera position for listener
        const listenerPos = this.camera.getPosition();
        const sourcePos = this.soundSource.getPosition();
        // const directDistance = vec3.distance(listenerPos, sourcePos); // Removed unused variable
        // const directTime = directDistance / this.SPEED_OF_SOUND; // Ensure this is commented out or removed

        // Add direct sound as a strong hit
        this.hits.push(
            this.createListenerRelativeHit(
                sourcePos,
                {
                    energy125Hz: 5.0,  // Strong direct sound
                    energy250Hz: 5.0,
                    energy500Hz: 5.0,
                    energy1kHz: 5.0,
                    energy2kHz: 5.0,
                    energy4kHz: 5.0,
                    energy8kHz: 5.0,
                    energy16kHz: 5.0
                },
                0, // Initial time
                0, // Initial phase
                1000, // Reference frequency
                1.0, // No Doppler shift
                0 // No bounces (direct sound)
            )
        );

        // Generate image sources for early reflections
        this.generateImageSources(2); // Up to 2nd order reflections
        
        // Calculate early reflections using image sources
        await this.calculateEarlyReflections();
        
        // Detect edges and generate rays for late reflections
        this.detectEdges();
        this.generateRays();
        
        // Store initial ray paths
        for (const ray of this.rays) {
            this.rayPaths.push({
                origin: ray.getOrigin(),
                direction: ray.getDirection(),
                energies: ray.getEnergies()
            });
        }

        // Calculate late reflections using ray tracing
        await this.calculateLateReflections();
        
        // Log summary
        console.log('Ray tracing summary:', {
            imageSourcePaths: this.imageSources.length - 1, // Subtract original source
            totalRays: this.rays.length,
            activeRayPaths: this.rayPaths.length,
            activeHits: this.hits.length
        });
    }

    private generateImageSources(maxOrder: number = 2): void {
        this.imageSources = [];
        const sourcePos = this.soundSource.getPosition();
        
        // Add original source (order 0)
        this.imageSources.push({
            position: vec3.clone(sourcePos),
            order: 0,
            reflectionPath: [vec3.clone(sourcePos)],
            surfaces: []
        });
        
        // Define room planes with their indices
        const { width, height, depth } = this.room.config.dimensions;
        const halfWidth = width / 2;
        const halfDepth = depth / 2;
        
        const planes = [
            { normal: vec3.fromValues(1, 0, 0), d: halfWidth, index: 0 },  // +X (right)
            { normal: vec3.fromValues(-1, 0, 0), d: halfWidth, index: 1 }, // -X (left)
            { normal: vec3.fromValues(0, 1, 0), d: 0, index: 2 },          // +Y (floor)
            { normal: vec3.fromValues(0, -1, 0), d: height, index: 3 },    // -Y (ceiling)
            { normal: vec3.fromValues(0, 0, 1), d: halfDepth, index: 4 },  // +Z (back)
            { normal: vec3.fromValues(0, 0, -1), d: halfDepth, index: 5 }  // -Z (front)
        ];
        
        // Generate image sources up to maxOrder
        let currentSources = [...this.imageSources];
        
        for (let order = 1; order <= maxOrder; order++) {
            const newSources: ImageSource[] = [];
            
            for (const source of currentSources) {
                if (source.order === order - 1) {
                    for (let i = 0; i < planes.length; i++) {
                        const plane = planes[i];
                        
                        // Skip if this surface was just reflected from
                        if (source.surfaces.length > 0 && source.surfaces[source.surfaces.length - 1] === plane.index) {
                            continue;
                        }
                        
                        // Calculate reflection
                        const sourcePos = source.position;
                        const distance = 2 * (vec3.dot(sourcePos, plane.normal) - plane.d);
                        const imagePos = vec3.create();
                        vec3.scale(imagePos, plane.normal, distance);
                        vec3.subtract(imagePos, sourcePos, imagePos);
                        
                        // Calculate reflection point on surface
                        const reflectionPoint = vec3.create();
                        vec3.add(reflectionPoint, sourcePos, imagePos);
                        vec3.scale(reflectionPoint, reflectionPoint, 0.5);
                        
                        // Create new reflection path
                        const newPath = [...source.reflectionPath, vec3.clone(reflectionPoint)];
                        const newSurfaces = [...source.surfaces, plane.index];
                        
                        newSources.push({
                            position: imagePos,
                            order: order,
                            reflectionPath: newPath,
                            surfaces: newSurfaces
                        });
                    }
                }
            }
            
            this.imageSources.push(...newSources);
            currentSources = [...this.imageSources];
        }
        
        console.log(`Generated ${this.imageSources.length} image sources up to order ${maxOrder}`);
    }

    private async calculateEarlyReflections(): Promise<void> {
        const listenerPos = this.camera.getPosition();
        
        // Get room materials mapping for surface indices
        const materials = [
            this.room.config.materials.walls,   // index 0: right wall (+X) -> Use 'walls' for all walls
            this.room.config.materials.walls,    // index 1: left wall (-X) -> Use 'walls'
            this.room.config.materials.floor,   // index 2: floor (+Y)
            this.room.config.materials.ceiling, // index 3: ceiling (-Y)
            this.room.config.materials.walls,    // index 4: back wall (+Z) -> Use 'walls'
            this.room.config.materials.walls    // index 5: front wall (-Z) -> Use 'walls'
        ];
        
        for (const source of this.imageSources) {
            if (source.order === 0) continue; // Skip original source
            
            // Calculate direct path from image source to listener
            const imageToListener = vec3.create();
            vec3.subtract(imageToListener, listenerPos, source.position);
            const distance = vec3.length(imageToListener);
            vec3.normalize(imageToListener, imageToListener);
            
            // Calculate time of arrival
            const timeOfArrival = distance / this.SPEED_OF_SOUND;
            
            // Calculate energy based on distance
            const energyDecay = 1.0 / (distance * distance);
            
            // Apply surface absorption for each reflection
            let energies: FrequencyBands = {
                energy125Hz: 1.0 * energyDecay,
                energy250Hz: 1.0 * energyDecay,
                energy500Hz: 1.0 * energyDecay,
                energy1kHz: 1.0 * energyDecay,
                energy2kHz: 1.0 * energyDecay,
                energy4kHz: 1.0 * energyDecay,
                energy8kHz: 1.0 * energyDecay,
                energy16kHz: 1.0 * energyDecay
            };
            
            // Apply frequency-dependent absorption for each reflection
            for (let i = 0; i < source.surfaces.length; i++) {
                const surfaceIndex = source.surfaces[i];
                
                // Safety check to prevent undefined access
                if (surfaceIndex >= 0 && surfaceIndex < materials.length) {
                    const material = materials[surfaceIndex];
                    
                    // Check if material exists before using it
                    if (material) {
                        // Apply absorption (multiply by (1-absorption) for each reflection)
                        energies.energy125Hz *= (1.0 - material.absorption125Hz);
                        energies.energy250Hz *= (1.0 - material.absorption250Hz);
                        energies.energy500Hz *= (1.0 - material.absorption500Hz);
                        energies.energy1kHz *= (1.0 - material.absorption1kHz);
                        energies.energy2kHz *= (1.0 - material.absorption2kHz);
                        energies.energy4kHz *= (1.0 - material.absorption4kHz);
                        energies.energy8kHz *= (1.0 - material.absorption8kHz);
                        energies.energy16kHz *= (1.0 - material.absorption16kHz);
                    }
                }
            }
            
            // Boost early reflections (stronger boost for earlier reflections)
            const reflectionBoost = 2.0 / (source.order + 1);
            energies.energy125Hz *= reflectionBoost;
            energies.energy250Hz *= reflectionBoost;
            energies.energy500Hz *= reflectionBoost;
            energies.energy1kHz *= reflectionBoost;
            energies.energy2kHz *= reflectionBoost;
            energies.energy4kHz *= reflectionBoost;
            energies.energy8kHz *= reflectionBoost;
            energies.energy16kHz *= reflectionBoost;
            
            // Create ray path segments for visualization
            for (let i = 0; i < source.reflectionPath.length - 1; i++) {
                const start = source.reflectionPath[i];
                const end = source.reflectionPath[i + 1];
                
                const direction = vec3.create();
                vec3.subtract(direction, end, start);
                vec3.normalize(direction, direction);
                
                this.rayPaths.push({
                    origin: vec3.clone(start),
                    direction: vec3.clone(direction),
                    energies: { ...energies }
                });
            }
            
            // Add final path segment to listener
            const lastPoint = source.reflectionPath[source.reflectionPath.length - 1];
            
            this.rayPaths.push({
                origin: vec3.clone(lastPoint),
                direction: vec3.clone(imageToListener), // Use calculated direction
                energies: { ...energies }
            });
            
            // Add hit point at listener position
            const hitDirection = vec3.create();
            vec3.subtract(hitDirection, source.position, listenerPos); // Direction from listener to image source (approximates arrival direction)
            vec3.normalize(hitDirection, hitDirection);

            this.hits.push({
                position: vec3.clone(listenerPos),
                energies: { ...energies },
                time: timeOfArrival,
                phase: 2 * Math.PI * 1000 * timeOfArrival, // 1kHz reference
                frequency: 1000, 
                dopplerShift: 1.0,
                bounces: source.order,
                distance: distance, // Add distance
                direction: hitDirection // Add direction
            });
        }
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
        
        // Define room planes with materials from room config
        const roomMaterials = this.room.config.materials;
        const planes = [
            { normal: vec3.fromValues(1, 0, 0), d: halfWidth, material: roomMaterials.walls }, // Use 'walls'
            { normal: vec3.fromValues(-1, 0, 0), d: halfWidth, material: roomMaterials.walls }, // Use 'walls'
            { normal: vec3.fromValues(0, 1, 0), d: 0, material: roomMaterials.floor },
            { normal: vec3.fromValues(0, -1, 0), d: height, material: roomMaterials.ceiling },
            { normal: vec3.fromValues(0, 0, 1), d: halfDepth, material: roomMaterials.walls }, // Use 'walls'
            { normal: vec3.fromValues(0, 0, -1), d: halfDepth, material: roomMaterials.walls }  // Use 'walls'
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

                        // Store point data - ensure all RayPathPoint fields are present
                        const pointDirection = vec3.create(); // Calculate direction for the point
                        vec3.subtract(pointDirection, pointPosition, this.camera.getPosition());
                        const pointDistance = vec3.length(pointDirection);
                        vec3.normalize(pointDirection, pointDirection);

                        this.rayPathPoints.push({
                            position: pointPosition,
                            energies: ray.getEnergies(),
                            time: pointTime,
                            phase: phaseAtPoint,
                            frequency: ray.getFrequency(),
                            dopplerShift: 1.0, // Assuming no doppler for path points for now
                            bounces: bounces, // Use current bounce count
                            distance: pointDistance, // Add distance
                            direction: pointDirection, // Add direction
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

                    // Create hit with listener-relative parameters
                    this.hits.push(
                        this.createListenerRelativeHit(
                            hitPoint,
                            ray.getEnergies(),
                            currentTime,
                            newPhase,
                            ray.getFrequency(),
                            dopplerShift,
                            bounces + 1
                        )
                    );

                    // Store path segment if energy is above threshold
                    if (this.calculateAverageEnergy(ray.getEnergies()) > this.VISIBILITY_THRESHOLD) {
                        this.rayPaths.push({
                            origin: vec3.clone(newOrigin),
                            direction: vec3.clone(reflected),
                            energies: ray.getEnergies()
                        });
                    }

                    bounces++;
                } else {
                    ray.deactivate();
                }
            }
        }
    }

    private createListenerRelativeHit(
        position: vec3,
        energies: FrequencyBands,
        time: number,
        phase: number,
        frequency: number,
        dopplerShift: number,
        bounces: number
    ): RayHit {
        // Get listener position
        const listenerPos = this.camera.getPosition();
        
        // Calculate distance and direction to listener
        const toListener = vec3.create();
        vec3.subtract(toListener, listenerPos, position);
        const distance = vec3.length(toListener);
        
        // Calculate time to reach listener
        const travelTime = distance / this.SPEED_OF_SOUND;
        const totalTime = time + travelTime;
        
        // Apply distance attenuation
        const attenuatedEnergies = { ...energies };
        const distanceAttenuation = 1.0 / (distance * distance);
        
        // Scale energies by inverse square law
        Object.keys(attenuatedEnergies).forEach(key => {
            // Type assertion to assure TypeScript 'key' is a key of FrequencyBands
            const bandKey = key as keyof FrequencyBands;
            if (attenuatedEnergies[bandKey] !== undefined) { 
                 attenuatedEnergies[bandKey] *= distanceAttenuation;
            }
        });
        
        // Create hit with proper listener-relative parameters
        return {
            position: vec3.clone(position),
            energies: attenuatedEnergies,
            time: totalTime,
            phase: phase,
            frequency: frequency,
            dopplerShift: dopplerShift,
            bounces: bounces,
            distance: distance,
            direction: vec3.normalize(vec3.create(), toListener)
        };
    }

    private calculateAverageEnergy(energies: FrequencyBands): number {
        const values = Object.values(energies);
        return values.reduce((sum, energy) => sum + energy, 0) / values.length;
    }

    public getRayHits(): RayHit[] {
        if (!this.hits || !Array.isArray(this.hits)) {
            console.warn('No valid ray hits available');
            return [];
        }
        return this.hits.filter(hit => hit && hit.position && hit.energies);
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

    // Method to detect edges from room geometry
    private detectEdges(): void {
        const { width, height, depth } = this.room.config.dimensions;
        const halfWidth = width / 2;
        const halfDepth = depth / 2;
        
        // Define room corners
        const corners = [
            vec3.fromValues(-halfWidth, 0, -halfDepth),
            vec3.fromValues(halfWidth, 0, -halfDepth),
            vec3.fromValues(halfWidth, 0, halfDepth),
            vec3.fromValues(-halfWidth, 0, halfDepth),
            vec3.fromValues(-halfWidth, height, -halfDepth),
            vec3.fromValues(halfWidth, height, -halfDepth),
            vec3.fromValues(halfWidth, height, halfDepth),
            vec3.fromValues(-halfWidth, height, halfDepth),
        ];
        
        // Define edges (12 edges for a rectangular room)
        // For each edge, store the two adjacent surface indices
        this.edges = [
            // Bottom edges
            { start: corners[0], end: corners[1], adjacentSurfaces: [2, 4] },
            { start: corners[1], end: corners[2], adjacentSurfaces: [2, 0] },
            { start: corners[2], end: corners[3], adjacentSurfaces: [2, 5] },
            { start: corners[3], end: corners[0], adjacentSurfaces: [2, 1] },
            
            // Top edges
            { start: corners[4], end: corners[5], adjacentSurfaces: [3, 4] },
            { start: corners[5], end: corners[6], adjacentSurfaces: [3, 0] },
            { start: corners[6], end: corners[7], adjacentSurfaces: [3, 5] },
            { start: corners[7], end: corners[4], adjacentSurfaces: [3, 1] },
            
            // Vertical edges
            { start: corners[0], end: corners[4], adjacentSurfaces: [1, 4] },
            { start: corners[1], end: corners[5], adjacentSurfaces: [0, 4] },
            { start: corners[2], end: corners[6], adjacentSurfaces: [0, 5] },
            { start: corners[3], end: corners[7], adjacentSurfaces: [1, 5] },
        ];

        console.log('Edge detection completed:', {
            numEdges: this.edges.length,
            roomDimensions: { width, height, depth }
        });
    }
}
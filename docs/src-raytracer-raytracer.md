# Ray Tracer Documentation

## File: `src/raytracer/raytracer.ts`

This file implements a ray tracing system for simulating sound propagation in a 3D environment. It models how sound waves travel from a source, bounce off surfaces, and reach the listener.

## Interfaces

### `RayTracerConfig`
Configuration parameters for the ray tracer.
- `numRays: number` - Number of rays to emit from the sound source
- `maxBounces: number` - Maximum number of reflections per ray
- `minEnergy: number` - Minimum energy threshold for ray termination

### `RayHit`
Represents a point where a ray hits a surface or the listener.
- `position: vec3` - 3D position of the hit
- `energies: FrequencyBands` - Energy levels across frequency bands
- `time: number` - Arrival time of the sound at this point
- `phase: number` - Phase of the sound wave at this point
- `frequency: number` - Frequency of the ray
- `dopplerShift: number` - Doppler shift at this point

### `RayPathPoint`
Extends `RayHit` with additional information about the ray's path.
- `bounceNumber: number` - Which reflection this point represents
- `rayIndex: number` - Which ray this point belongs to

### `ImpulseResponse`
Represents the calculated impulse response from the ray tracing.
- `time: Float32Array` - Time points
- `amplitude: Float32Array` - Amplitude values
- `sampleRate: number` - Sample rate of the impulse response
- `frequencies: Float32Array` - Frequency content at each time point

### `ImageSource` (private)
Represents a virtual sound source created by reflections.
- `position: vec3` - Position of the image source
- `order: number` - Reflection order (number of bounces)
- `reflectionPath: vec3[]` - Path of reflections

### `RayPath` (private)
Represents the complete path of a ray.
- `points: vec3[]` - Points along the ray's path
- `energies: FrequencyBands` - Energy levels at the end of the path
- `totalDistance: number` - Total distance traveled by the ray

## Class: `RayTracer`

The main class that implements the ray tracing algorithm for sound propagation.

### Properties

- `device: GPUDevice` - WebGPU device for rendering
- `soundSource: Sphere` - The sound source object
- `room: Room` - The room environment
- `config: RayTracerConfig` - Configuration parameters
- `rays: Ray[]` - Array of rays emitted from the source
- `hits: RayHit[]` - Array of ray hit points
- `rayPaths` - Array of ray paths for visualization
- `rayPathPoints: RayPathPoint[]` - Detailed points along ray paths
- `rayRenderer: RayRenderer` - Renderer for visualizing rays
- `VISIBILITY_THRESHOLD: number` - Threshold for ray visibility (0.05)
- `SPEED_OF_SOUND: number` - Speed of sound in air (343.0 m/s)
- `AIR_TEMPERATURE: number` - Air temperature (20.0°C)
- `rayPointsBuffer: GPUBuffer` - Buffer for ray points data

### Methods

#### `constructor(device: GPUDevice, soundSource: Sphere, room: Room, config?: RayTracerConfig)`
- Initializes the ray tracer with the given parameters
- Sets up default configuration if not provided
- Creates the ray renderer and buffer for ray points

#### `private generateRays(): void`
- Creates rays emanating from the sound source in random directions
- Distributes rays evenly across the sphere using spherical coordinates
- Assigns different frequency bands to rays
- Initializes each ray with full energy across all frequency bands

#### `public async calculateRayPaths(): Promise<void>`
- Main method to calculate ray paths through the environment
- Clears previous calculation results
- Generates new rays and traces their paths
- Handles reflections and energy attenuation
- Collects hit points for impulse response generation

#### `private calculateDopplerShift(rayDirection: vec3, surfaceNormal: vec3): number`
- Calculates the Doppler shift for a ray reflecting off a surface
- Takes into account the ray direction and surface normal

#### `private async calculateLateReflections(): Promise<void>`
- Calculates late reflections (higher-order bounces)
- Handles energy attenuation based on material properties
- Accounts for air absorption and distance attenuation
- Tracks ray paths for visualization

#### `private calculateAverageEnergy(energies: FrequencyBands): number`
- Calculates the average energy across all frequency bands
- Used to determine if a ray should continue bouncing

#### `public getRayHits(): RayHit[]`
- Returns the array of ray hit points
- Used by the audio processor to generate the impulse response

#### `public render(pass: GPURenderPassEncoder, viewProjection: Float32Array): void`
- Renders the ray paths using the ray renderer
- Visualizes the sound propagation in the 3D environment

#### `public recalculateRays(): void`
- Recalculates ray paths with the current configuration
- Used when parameters change

## Relationships

This module is imported by:
- `main.ts` - Creates a ray tracer instance and uses it to calculate impulse responses

This module imports:
- `gl-matrix` - For vector operations
- `./ray` - For the Ray class and FrequencyBands interface
- `../room/room` - For the Room environment
- `../objects/sphere` - For the sound source
- `./ray-renderer` - For visualizing rays
- `../room/room-materials` - For material properties affecting sound reflection 
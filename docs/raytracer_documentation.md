# Raytracer Module Documentation

This document provides a detailed explanation of the raytracer module located in the `src/raytracer/` directory. The module implements an acoustic ray tracing system for spatial audio simulation.

## Table of Contents

1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [Ray Class (`ray.ts`)](#ray-class-rayts)
4. [RayTracer Class (`raytracer.ts`)](#raytracer-class-raytracerts)
5. [RayRenderer Class (`ray-renderer.ts`)](#rayrenderer-class-ray-rendererts)
6. [WebGPU Shaders (`shaders/raytracer.wgsl`)](#webgpu-shaders-shadersraytracerwgsl)
7. [Key Concepts and Algorithms](#key-concepts-and-algorithms)
8. [Relationships with Other Modules](#relationships-with-other-modules)

## Overview

The raytracer module simulates how sound propagates in a 3D environment by tracing rays from a sound source and calculating their reflections, energy loss, and arrival times at a listener position. It uses both image-source method for early reflections and stochastic ray tracing for late reflections, implementing a physically-based acoustic simulation.

The system accounts for:
- Frequency-dependent material absorption
- Air absorption based on temperature and humidity
- Doppler shift
- Phase changes
- Energy attenuation over distance

## File Structure

The raytracer module consists of the following files:

- `ray.ts`: Defines the `Ray` class representing a single sound ray
- `raytracer.ts`: Implements the main `RayTracer` class that manages the simulation
- `ray-renderer.ts`: Handles visualization of ray paths using WebGPU
- `shaders/raytracer.wgsl`: WebGPU shader code for GPU-accelerated ray calculations

## Ray Class (`ray.ts`)

### Purpose
Represents a single sound ray in the acoustic simulation with properties for tracking its path, energy, and wave characteristics.

### Key Components

#### FrequencyBands Interface
```typescript
export interface FrequencyBands {
    energy125Hz: number;
    energy250Hz: number;
    energy500Hz: number;
    energy1kHz: number;
    energy2kHz: number;
    energy4kHz: number;
    energy8kHz: number;
    energy16kHz: number;
}
```
Tracks energy levels across 8 frequency bands from 125Hz to 16kHz.

#### Ray Class Properties
- `origin`: 3D position vector where the ray is currently located
- `direction`: Normalized 3D vector indicating ray direction
- `energies`: Object containing energy values for each frequency band
- `pathLength`: Total distance the ray has traveled
- `bounces`: Number of reflections the ray has undergone
- `isActive`: Boolean flag indicating if the ray is still active
- `time`: Time elapsed since ray emission
- `phase`: Current phase of the wave
- `frequency`: Frequency of the ray (for wave simulation)

#### Important Methods

- **Constructor**: Initializes a ray with origin, direction, initial energy, and frequency
  ```typescript
  constructor(origin: vec3, direction: vec3, initialEnergy: number = 1.0, frequency: number = 1000)
  ```

- **updateRay**: Updates ray properties after a reflection
  ```typescript
  updateRay(
      newOrigin: vec3,
      newDirection: vec3,
      energyLoss: { /* absorption coefficients */ },
      distance: number,
      temperature: number = 20,
      humidity: number = 50
  )
  ```
  This method:
  1. Updates position and direction
  2. Applies material absorption based on reflection coefficients
  3. Applies frequency-dependent air absorption
  4. Updates path length, bounce count, time, and phase

- **calculateAirAbsorption**: Calculates frequency-dependent air absorption based on distance, temperature, and humidity
  ```typescript
  private calculateAirAbsorption(
      distance: number,
      temperature: number,
      humidity: number
  ): { /* absorption values */ }
  ```

- **Getters/Setters**: Methods to access and modify ray properties

## RayTracer Class (`raytracer.ts`)

### Purpose
Manages the overall ray tracing simulation, including ray generation, reflection calculations, and hit detection.

### Key Components

#### Interfaces
- `RayTracerConfig`: Configuration for the ray tracer
- `RayHit`: Information about a ray hitting the listener
- `RayPathPoint`: Point along a ray path with associated properties
- `Edge`: Represents an edge between surfaces for diffraction
- `ImageSource`: Represents a virtual sound source created by reflections

#### RayTracer Class Properties
- `device`: WebGPU device for GPU-accelerated calculations
- `soundSource`: The sound source object (Sphere)
- `room`: The room object containing surfaces and materials
- `camera`: The camera/listener position
- `config`: Configuration parameters
- `rays`: Array of active Ray objects
- `hits`: Array of detected ray hits at the listener
- `rayPaths`: Array of ray path segments for visualization
- `rayPathPoints`: Array of points along ray paths
- `edges`: Array of detected edges in the room
- `imageSources`: Array of virtual sound sources

#### Important Methods

- **calculateRayPaths**: Main method that orchestrates the simulation
  ```typescript
  public async calculateRayPaths(): Promise<void>
  ```
  This method:
  1. Resets simulation state
  2. Generates image sources for early reflections
  3. Calculates early reflections using the image source method
  4. Detects edges for diffraction
  5. Generates rays for late reflections
  6. Calculates late reflections using stochastic ray tracing

- **generateImageSources**: Creates virtual sound sources for early reflections
  ```typescript
  private generateImageSources(maxOrder: number = 2): void
  ```
  Implements the image source method up to the specified reflection order.

- **calculateEarlyReflections**: Calculates early reflections using image sources
  ```typescript
  private async calculateEarlyReflections(): Promise<void>
  ```
  Traces paths from image sources to the listener, calculating energy, time, and phase.

- **calculateLateReflections**: Calculates late reflections using stochastic ray tracing
  ```typescript
  private async calculateLateReflections(): Promise<void>
  ```
  Traces rays through multiple bounces, applying material and air absorption.

- **detectEdges**: Identifies edges in the room for diffraction calculations
  ```typescript
  private detectEdges(): void
  ```

- **generateRays**: Creates rays emanating from the sound source
  ```typescript
  private generateRays(): void
  ```
  Distributes rays uniformly around the sound source using spherical coordinates.

- **render**: Renders ray paths for visualization
  ```typescript
  public render(pass: GPURenderPassEncoder, viewProjection: Float32Array): void
  ```
  Delegates to RayRenderer to visualize ray paths.

## RayRenderer Class (`ray-renderer.ts`)

### Purpose
Handles the visualization of ray paths using WebGPU for efficient rendering.

### Key Components

#### RayRenderer Class Properties
- `device`: WebGPU device
- `pipeline`: WebGPU render pipeline
- `vertexBuffer`: Buffer for ray path vertices
- `uniformBuffer`: Buffer for uniform data (view projection matrix)
- `uniformBindGroup`: Bind group for shader uniforms

#### Important Methods

- **Constructor**: Sets up WebGPU resources
  ```typescript
  constructor(device: GPUDevice)
  ```
  Creates shader modules, pipeline, buffers, and bind groups.

- **render**: Renders ray paths
  ```typescript
  public render(
      pass: GPURenderPassEncoder,
      viewProjection: Float32Array,
      rays: { origin: vec3, direction: vec3, energies: FrequencyBands }[],
      roomDimensions: { width: number, height: number, depth: number }
  ): void
  ```
  This method:
  1. Updates uniform buffer with view projection matrix
  2. Creates vertex data for ray paths
  3. Calculates ray intersections with room boundaries
  4. Updates vertex buffer
  5. Issues draw commands

- **calculateFrequencyWeightedEnergy**: Calculates energy distribution across frequency bands
  ```typescript
  private calculateFrequencyWeightedEnergy(energies: FrequencyBands): {value: number, bandDistribution: number[]}
  ```
  Weights and averages energy values across low, mid, and high frequency bands.

- **resetRender**: Resets rendering state
  ```typescript
  public resetRender(): void
  ```

## WebGPU Shaders (`shaders/raytracer.wgsl`)

### Purpose
Provides GPU-accelerated ray tracing calculations using WebGPU compute shaders.

### Key Components

#### Structures
- `Ray`: Represents a ray with position, direction, energy values, and wave properties
- `Surface`: Represents a surface with normal, position, and material properties
- `Edge`: Represents an edge between surfaces
- `RayHit`: Represents a ray hit point with associated properties

#### Main Compute Shader
The main compute shader processes rays in parallel, calculating reflections and energy loss.

Key operations:
1. Ray-surface intersection tests
2. Reflection calculations
3. Energy attenuation
4. Phase updates

## Key Concepts and Algorithms

### Image Source Method
Used for early reflections (up to 2nd order), this method:
1. Creates virtual sound sources by mirroring the original source across room surfaces
2. Traces direct paths from these virtual sources to the listener
3. Validates paths by checking if they pass through the corresponding surfaces

### Stochastic Ray Tracing
Used for late reflections, this method:
1. Generates rays in random directions from the sound source
2. Traces each ray through multiple bounces
3. Applies material absorption at each reflection
4. Applies air absorption based on distance traveled
5. Records hits when rays pass near the listener

### Energy Calculations
The system tracks energy across 8 frequency bands (125Hz to 16kHz):
1. Material absorption is applied at each reflection based on material properties
2. Air absorption is calculated based on distance, temperature, and humidity
3. Distance attenuation follows the inverse square law

### Wave Properties
The system simulates wave properties:
1. Phase is updated based on distance traveled and frequency
2. Doppler shift is calculated at reflections
3. Time of arrival is tracked for impulse response generation

## Relationships with Other Modules

The raytracer module interacts with several other components:

- **Room Module**: Provides the geometry and materials for ray tracing
  - Uses `Room` class for room dimensions and surface properties
  - Uses `WallMaterial` for material absorption coefficients

- **Objects Module**: Provides the sound source
  - Uses `Sphere` class to represent the sound source position

- **Camera Module**: Provides the listener position
  - Uses `Camera` class to get the listener position and orientation

- **Sound Module**: Consumes ray tracing results
  - Ray hits are used to generate impulse responses for audio processing

- **Visualization Module**: Displays ray tracing results
  - Ray paths are rendered for visual feedback

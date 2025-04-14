# Room Module Documentation

This document provides a detailed explanation of the room module located in the `src/room/` directory. The module implements the virtual environment for the spatial audio simulation, defining both the geometric and acoustic properties of the space.

## Table of Contents

1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [Room Class (`room.ts`)](#room-class-roomts)
4. [Room Materials (`room-materials.ts`)](#room-materials-room-materialsts)
5. [Room Types (`types.ts`)](#room-types-typests)
6. [Key Concepts and Techniques](#key-concepts-and-techniques)
7. [Relationships with Other Modules](#relationships-with-other-modules)

## Overview

The room module defines the virtual environment in which the spatial audio simulation takes place. It provides both the geometric representation of the room (dimensions, surfaces) and the acoustic properties of its materials (absorption, scattering). The room is modeled as a rectangular box with six surfaces (floor, ceiling, and four walls), each with its own acoustic properties that affect how sound waves interact with them.

The system provides:
- A configurable rectangular room with adjustable dimensions
- Frequency-dependent acoustic material properties for each surface
- Visual representation of the room boundaries
- Physical properties like volume and surface area for acoustic calculations
- Environmental parameters like temperature and humidity

## File Structure

The room module consists of the following files:

- `room.ts`: Implements the Room class representing the virtual environment
- `room-materials.ts`: Defines material properties and presets for room surfaces
- `types.ts`: Contains type definitions and enumerations for room components

## Room Class (`room.ts`)

### Purpose
Represents the virtual environment for the spatial audio simulation, defining its geometry and providing methods for rendering and acoustic calculations.

### Key Components

#### RoomConfig Interface
```typescript
export interface RoomConfig {
    dimensions: {
        width: number;
        height: number;
        depth: number;
    };
    materials: {
        walls: WallMaterial;
        ceiling: WallMaterial;
        floor: WallMaterial;
    };
}
```
Defines the configuration for a room, including dimensions and material properties.

#### Room Class Properties
- `config`: Configuration object containing room dimensions and materials
- `device`: WebGPU device for rendering
- `uniformBuffer`: Buffer for uniform data (transformation matrices)
- `vertexBuffer`: Buffer containing room vertex data
- `pipeline`: WebGPU render pipeline for room rendering
- `uniformBindGroup`: Bind group for shader uniforms

#### Important Methods

- **Constructor**: Initializes a room with specified dimensions and materials
  ```typescript
  constructor(device: GPUDevice, config: RoomConfig)
  ```
  Sets up WebGPU resources for room rendering and stores the configuration.

- **createVertexBuffer**: Creates a vertex buffer for room rendering
  ```typescript
  private createVertexBuffer(): GPUBuffer
  ```
  Generates vertices for the room boundaries as line segments.

- **createRenderPipeline**: Creates the WebGPU render pipeline
  ```typescript
  private createRenderPipeline(): GPURenderPipeline
  ```
  Sets up the rendering pipeline with appropriate shaders and states.

- **render**: Renders the room boundaries
  ```typescript
  public render(pass: GPURenderPassEncoder): void
  ```
  Draws the room as a wireframe box.

- **getVolume**: Calculates the room's volume
  ```typescript
  public getVolume(): number
  ```
  Returns the volume in cubic meters.

- **getSurfaceArea**: Calculates the room's total surface area
  ```typescript
  public getSurfaceArea(): number
  ```
  Returns the total surface area in square meters.

- **getTemperature** and **getHumidity**: Provide environmental parameters
  ```typescript
  public getTemperature(): number
  public getHumidity(): number
  ```
  Return default values for temperature (20°C) and humidity (50%).

- **getClosestValidPosition**: Constrains a position to be within the room
  ```typescript
  public getClosestValidPosition(position: [number, number, number]): [number, number, number]
  ```
  Ensures a position is inside the room boundaries with a margin.

## Room Materials (`room-materials.ts`)

### Purpose
Defines the acoustic properties of materials used for room surfaces, affecting how sound waves interact with them.

### Key Components

#### WallMaterial Interface
```typescript
export interface WallMaterial {
    // Absorption coefficients for 8 frequency bands
    absorption125Hz: number;  // 125 Hz
    absorption250Hz: number;  // 250 Hz
    absorption500Hz: number;  // 500 Hz
    absorption1kHz: number;   // 1000 Hz
    absorption2kHz: number;   // 2000 Hz
    absorption4kHz: number;   // 4000 Hz
    absorption8kHz: number;   // 8000 Hz
    absorption16kHz: number;  // 16000 Hz

    // Scattering coefficients for 8 frequency bands
    scattering125Hz: number;
    scattering250Hz: number;
    scattering500Hz: number;
    scattering1kHz: number;
    scattering2kHz: number;
    scattering4kHz: number;
    scattering8kHz: number;
    scattering16kHz: number;

    roughness: number;       // 0-1, affects reflection pattern
    phaseShift: number;      // Fixed phase shift on reflection (radians)
    phaseRandomization: number; // Max random phase variation (radians)
}
```
Defines the acoustic properties of a material across frequency bands.

#### RoomMaterials Interface
```typescript
export interface RoomMaterials {
    left:   WallMaterial;
    right:  WallMaterial;
    top:    WallMaterial;
    bottom: WallMaterial;
    front:  WallMaterial;
    back:   WallMaterial;
}
```
Defines materials for each of the six surfaces of the room.

#### Material Presets
The module provides predefined material presets with realistic acoustic properties:
- `CONCRETE`: High-density material with low absorption and scattering
- `WOOD`: Medium-density material with frequency-dependent absorption and moderate scattering

Each preset includes detailed frequency-dependent absorption and scattering coefficients based on real-world measurements.

## Room Types (`types.ts`)

### Purpose
Provides type definitions and enumerations for room components.

### Key Components

#### Surface Enumeration
```typescript
export enum Surface {
    FLOOR = 0,
    CEILING = 1,
    WALL_FRONT_BACK = 2,
    WALL_LEFT_RIGHT = 3
}
```
Enumerates the different surface types in the room.

#### Interface Definitions
- `RoomConfig`: Configuration for room dimensions and materials
- `RoomDimensions`: Dimensions of the room (width, height, depth)
- `RoomMaterials`: Materials for different room surfaces

## Key Concepts and Techniques

### Acoustic Modeling
The room module implements several acoustic concepts:
1. **Frequency-Dependent Absorption**: Materials absorb different amounts of energy across frequency bands
2. **Scattering Coefficients**: Define how sound is scattered rather than reflected specularly
3. **Surface Roughness**: Affects the pattern of reflections
4. **Phase Effects**: Materials can introduce phase shifts and randomization

### WebGPU Rendering
The room visualization demonstrates several WebGPU concepts:
1. **Line Rendering**: Draws the room as wireframe lines
2. **Vertex Buffer Management**: Creates and manages vertex data
3. **Shader Implementation**: Uses simple shaders for wireframe rendering
4. **Pipeline Configuration**: Sets up the render pipeline with appropriate states

### Geometric Calculations
The module provides methods for acoustic calculations:
1. **Volume Calculation**: Used for reverb time calculations
2. **Surface Area Calculation**: Used for energy decay calculations
3. **Position Constraints**: Ensures objects remain within the room boundaries

## Relationships with Other Modules

The room module interacts with several other components:

- **Main Application**: Initializes and configures the room
  - Creates the Room instance with specified dimensions and materials
  - Provides UI controls for adjusting room properties

- **Raytracer Module**: Uses the room for ray-surface intersections
  - Gets room dimensions to define the boundaries for ray tracing
  - Uses material properties to calculate energy loss at reflections
  - Defines planes based on room surfaces for intersection tests

- **Sound Module**: Uses room properties for acoustic calculations
  - Uses room volume and surface area for reverb time calculations
  - Uses material absorption coefficients for frequency-dependent decay
  - Uses environmental parameters like temperature and humidity

- **Objects Module**: Provides the environment context
  - The sound source and listener exist within the room boundaries
  - Room dimensions constrain object positions

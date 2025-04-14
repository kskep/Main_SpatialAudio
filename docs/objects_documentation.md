# Objects Module Documentation

This document provides a detailed explanation of the objects module located in the `src/objects/` directory. The module implements 3D objects used in the spatial audio simulation, particularly the sound source representation.

## Table of Contents

1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [Sphere Class (`sphere.ts`)](#sphere-class-spherets)
4. [SphereRenderer Class (`sphere-renderer.ts`)](#sphererenderer-class-sphere-rendererts)
5. [Key Concepts and Techniques](#key-concepts-and-techniques)
6. [Relationships with Other Modules](#relationships-with-other-modules)

## Overview

The objects module provides the implementation of 3D objects used in the spatial audio simulation. Currently, it focuses on the sound source representation, implemented as a sphere that can be positioned within the virtual environment. The module separates the logical representation of objects (data and behavior) from their visual representation (rendering), following good software design principles.

The system provides:
- A simple, efficient representation of the sound source
- WebGPU-based rendering of the sound source as a visible sphere
- Position control for placing the sound source in the virtual environment

## File Structure

The objects module consists of the following files:

- `sphere.ts`: Implements the Sphere class representing the sound source
- `sphere-renderer.ts`: Implements the SphereRenderer class for visualizing the sphere using WebGPU

## Sphere Class (`sphere.ts`)

### Purpose
Represents a sound source in the spatial audio simulation as a sphere with a position and radius.

### Key Components

#### Properties
- `position`: 3D vector (vec3) representing the sphere's position in world space
- `radius`: Number representing the sphere's radius
- `DEFAULT_RADIUS`: Constant defining the default radius for sound sources

#### Important Methods

- **Constructor**: Initializes a sphere with a position and radius
  ```typescript
  constructor(position: vec3 = vec3.fromValues(0, 1.7, 0), radius: number = 0.2)
  ```
  Creates a sphere with a default position at head height (1.7m) and a small radius (0.2m).

- **update**: Updates the sphere's state over time
  ```typescript
  public update(deltaTime: number, roomDimensions: { width: number, height: number, depth: number }): void
  ```
  Currently a placeholder as the sound source is static, but could be extended for moving sound sources.

- **setPosition**: Sets the sphere's position
  ```typescript
  public setPosition(position: vec3): void
  ```
  Updates the sphere's position to the provided 3D coordinates.

- **getPosition**: Gets the sphere's position
  ```typescript
  public getPosition(): vec3
  ```
  Returns a copy of the sphere's position vector.

- **getRadius**: Gets the sphere's radius
  ```typescript
  public getRadius(): number
  ```
  Returns the sphere's radius.

## SphereRenderer Class (`sphere-renderer.ts`)

### Purpose
Renders a sphere using WebGPU, providing visual representation of the sound source in the 3D environment.

### Key Components

#### Properties
- `device`: WebGPU device for rendering
- `pipeline`: WebGPU render pipeline for sphere rendering
- `vertexBuffer`: Buffer containing sphere vertex data
- `indexBuffer`: Buffer containing sphere index data
- `uniformBuffer`: Buffer for uniform data (transformation matrices)
- `uniformBindGroup`: Bind group for shader uniforms
- `indexCount`: Number of indices in the sphere mesh

#### Important Methods

- **Constructor**: Sets up WebGPU resources for sphere rendering
  ```typescript
  constructor(device: GPUDevice)
  ```
  This method:
  1. Creates sphere mesh data (vertices and indices)
  2. Sets up WebGPU buffers for the mesh
  3. Creates a uniform buffer for transformation matrices
  4. Initializes the render pipeline and bind group

- **createSphereData**: Generates sphere mesh data
  ```typescript
  private createSphereData(radius: number = 1, segments: number = 16): { vertices: Float32Array, indices: Uint16Array }
  ```
  This method:
  1. Generates sphere vertices using spherical coordinates
  2. Creates triangle indices for the sphere mesh
  3. Returns the vertex and index data as typed arrays

- **createPipeline**: Creates the WebGPU render pipeline
  ```typescript
  private createPipeline(): GPURenderPipeline
  ```
  This method:
  1. Defines vertex and fragment shaders for sphere rendering
  2. Configures the pipeline with appropriate vertex attributes and render states
  3. Sets up depth testing and backface culling

- **render**: Renders the sphere at a specified position and radius
  ```typescript
  public render(
      renderPass: GPURenderPassEncoder,
      viewProjectionMatrix: Float32Array,
      position: vec3,
      radius: number
  ): void
  ```
  This method:
  1. Creates a model matrix based on the sphere's position and radius
  2. Combines it with the view-projection matrix in the uniform buffer
  3. Sets up the render state and draws the sphere

### Shader Implementation

The sphere renderer uses a simple shader that:
1. Transforms vertices using model, view, and projection matrices
2. Uses the normalized position as the surface normal (valid for unit spheres)
3. Applies basic diffuse lighting for a yellow sphere
4. Supports depth testing for proper occlusion

```wgsl
struct Uniforms {
    viewProjectionMatrix: mat4x4f,
    modelMatrix: mat4x4f,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
};

@vertex
fn vertexMain(@location(0) position: vec3f) -> VertexOutput {
    var output: VertexOutput;
    output.position = uniforms.viewProjectionMatrix * uniforms.modelMatrix * vec4f(position, 1.0);
    output.normal = position.xyz;  // For a unit sphere, position is the same as normal
    return output;
}

@fragment
fn fragmentMain(@location(0) normal: vec3f) -> @location(0) vec4f {
    let N = normalize(normal);
    let L = normalize(vec3f(1.0, 1.0, 1.0));
    let diffuse = max(dot(N, L), 0.0);
    return vec4f(vec3f(1.0, 1.0, 0.0) * (0.5 + 0.5 * diffuse), 1.0);
}
```

## Key Concepts and Techniques

### Object-Oriented Design
The module follows object-oriented design principles:
1. **Separation of Concerns**: Logical representation (Sphere) is separate from visual representation (SphereRenderer)
2. **Encapsulation**: Internal details are hidden behind public interfaces
3. **Single Responsibility**: Each class has a clear, focused purpose

### WebGPU Rendering
The sphere renderer demonstrates several WebGPU concepts:
1. **Mesh Generation**: Programmatically creates a sphere mesh
2. **Buffer Management**: Creates and manages vertex, index, and uniform buffers
3. **Shader Implementation**: Defines WGSL shaders for rendering
4. **Pipeline Configuration**: Sets up the render pipeline with appropriate states
5. **Uniform Handling**: Updates transformation matrices for rendering

### 3D Mathematics
The module leverages gl-matrix for 3D math operations:
1. **Vector Operations**: Uses vec3 for positions and directions
2. **Matrix Transformations**: Applies translation and scaling to position and size the sphere
3. **Matrix Composition**: Combines model, view, and projection matrices for rendering

## Relationships with Other Modules

The objects module interacts with several other components:

- **Main Application**: Initializes and manages the objects
  - Creates the Sphere and SphereRenderer instances
  - Updates the sphere's position based on user input
  - Triggers rendering during the render loop

- **Raytracer Module**: Uses the sound source for ray generation
  - Gets the sphere's position as the origin for ray tracing
  - Uses the sphere's radius to offset ray origins from the center

- **Room Module**: Provides the environment context
  - The sphere exists within the room boundaries
  - Room dimensions could be used for collision detection (if implemented)

- **Camera Module**: Provides the view for rendering
  - The view-projection matrix is used to render the sphere from the camera's perspective
  - The camera position is used as the listener position in the spatial audio simulation

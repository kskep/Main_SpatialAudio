# Sphere Renderer Documentation

## File: `src/objects/sphere-renderer.ts`

This file implements a WebGPU renderer for spheres, used to visualize the sound source in the spatial audio simulation.

## Class: `SphereRenderer`

The `SphereRenderer` class handles the creation and rendering of a sphere mesh using WebGPU.

### Properties

- `device: GPUDevice` - The WebGPU device for GPU operations
- `pipeline: GPURenderPipeline` - The WebGPU render pipeline for the sphere
- `vertexBuffer: GPUBuffer` - Buffer containing sphere vertex data
- `indexBuffer: GPUBuffer` - Buffer containing sphere index data
- `uniformBuffer: GPUBuffer` - Buffer for transformation matrices
- `uniformBindGroup: GPUBindGroup` - Bind group for the uniform buffer
- `indexCount: number` - Number of indices in the sphere mesh

### Methods

#### `constructor(device: GPUDevice)`
- Initializes the sphere renderer with the given WebGPU device
- Creates the sphere mesh data (vertices and indices)
- Sets up the vertex and index buffers
- Creates the uniform buffer for transformation matrices
- Creates the render pipeline and bind group

#### `private createSphereData(radius: number, segments: number): { vertices: Float32Array, indices: Uint16Array }`
- Generates the vertex and index data for a sphere
- Parameters:
  - `radius` - The radius of the sphere (default: 1)
  - `segments` - The number of segments for sphere tessellation (default: 16)
- Returns an object containing:
  - `vertices` - Float32Array of vertex positions
  - `indices` - Uint16Array of triangle indices

#### `private createPipeline(): GPURenderPipeline`
- Creates a WebGPU render pipeline for the sphere
- Defines the vertex and fragment shaders:
  - Vertex shader transforms the sphere vertices using view-projection and model matrices
  - Fragment shader applies simple lighting to the sphere with a yellow color
- Configures the pipeline with appropriate vertex attributes and render states

#### `public render(renderPass: GPURenderPassEncoder, viewProjectionMatrix: Float32Array, position: vec3, radius: number): void`
- Renders the sphere with the given parameters
- Parameters:
  - `renderPass` - The current WebGPU render pass
  - `viewProjectionMatrix` - The combined view-projection matrix
  - `position` - The position of the sphere in world space
  - `radius` - The radius of the sphere
- Creates a model matrix based on the position and radius
- Updates the uniform buffer with the matrices
- Sets the pipeline, vertex buffer, index buffer, and bind group
- Issues the draw call to render the sphere

## Relationships

This module is imported by:
- `main.ts` - Creates a sphere renderer instance and uses it to render the sound source

This module depends on:
- `gl-matrix` - For vector and matrix operations
- `objects/sphere.ts` - Uses the sphere's position and radius for rendering 
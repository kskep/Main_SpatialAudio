# Raytracer Module Documentation (`src/raytracer`)

This document provides an overview of the files within the `src/raytracer` directory, which implements the core acoustic ray tracing simulation for the spatial audio system.

## Files

### `ray.ts`

*   **Purpose:** Defines the `Ray` class, representing a single sound ray used in the acoustic simulation.
*   **Key Features:**
    *   Stores the ray's origin (`vec3`) and normalized direction (`vec3`).
    *   Tracks energy across multiple frequency bands (`FrequencyBands` interface) - 125Hz to 16kHz.
    *   Maintains state variables: `pathLength`, `bounces`, `isActive`, `time`, `phase`, and `frequency`.
    *   Provides methods to get the ray's current properties (`getOrigin`, `getDirection`, `getEnergies`, etc.).
    *   Includes an `updateRay` method:
        *   Updates the ray's position and direction after a reflection.
        *   Applies frequency-dependent energy loss based on material absorption coefficients.
        *   Applies frequency-dependent air absorption based on distance traveled, temperature, and humidity (using ISO 9613-1 standard).
        *   Updates the ray's travel time and phase based on distance and speed of sound.
    *   Defines the `FrequencyBands` interface for consistent energy tracking.

### `ray-renderer.ts`

*   **Purpose:** Defines the `RayRenderer` class, responsible for visualizing the traced rays using WebGPU.
*   **Key Features:**
    *   Manages WebGPU resources: `GPUDevice`, `GPURenderPipeline`, `GPUBuffer` (for vertices and uniforms), `GPUBindGroup`.
    *   Contains WGSL shader code (vertex and fragment) inline:
        *   **Vertex Shader:** Transforms ray endpoint vertices using the view-projection matrix. Passes energy, frequency band distribution, and phase data to the fragment shader.
        *   **Fragment Shader:** Renders rays as lines. Colors lines based on frequency content (mapping low/mid/high bands to RGB). Modulates color intensity and alpha based on the ray's energy and phase to simulate wave propagation visually. Uses alpha blending for transparency.
    *   `render` method:
        *   Takes the render pass encoder, view-projection matrix, ray data (positions, energies, phases), and room dimensions.
        *   Prepares vertex data based on ray start and end points (or segments).
        *   Updates the uniform buffer (view-projection matrix).
        *   Sets the pipeline, bind group, and vertex buffer.
        *   Issues the draw call (`draw(rays.length * 2)` for line list).
        *   Includes an optimization (`lastViewProjection`, `hasRendered`) to skip rendering if the camera view hasn't changed, improving performance.
    *   `resetRender` method: Resets the render state, forcing a re-render on the next frame.

### `raytracer.ts`

*   **Purpose:** Defines the main `RayTracer` class, which orchestrates the entire acoustic ray tracing process. This is the central coordinator for the simulation.
*   **Key Features:**
    *   **Initialization:** Takes `GPUDevice`, `SoundSource` (as `Sphere`), `Room`, `Camera` (as listener), and `RayTracerConfig` (number of rays, max bounces, min energy threshold).
    *   **Ray Generation (`generateRays`):** Creates an initial set of `Ray` objects originating from the sound source surface with randomized directions and associated frequencies.
    *   **Simulation Loop (`calculateRayPaths` - likely involves private helper methods):**
        *   Traces each ray through the `Room` geometry.
        *   Performs intersection tests between rays and room surfaces (walls, potentially objects).
        *   Handles reflections: When a ray hits a surface, it calculates the new direction (specular reflection, potentially diffuse in future) and updates the ray's energy using `ray.updateRay` based on the surface's material properties (`WallMaterial`) and air absorption.
        *   Tracks ray paths and bounces, terminating rays when energy drops below `minEnergy` or `maxBounces` is reached.
    *   **Listener Interaction:** Detects when rays intersect with a region around the listener (`Camera` position).
    *   **Hit Collection (`hits`, `RayHit` interface):** Stores detailed information about rays reaching the listener, including arrival time, energy per band, incoming direction, phase, Doppler shift (calculated based on relative motion), and bounce count.
    *   **Image Source Method (`generateImageSources`, `calculateEarlyReflections`):** Implements the image source technique to precisely calculate early reflections up to a certain order. This complements the stochastic ray tracing for late reverberation.
    *   **Edge Detection/Diffraction (`detectEdges`, `Edge` interface):** Contains logic (possibly preliminary) to identify geometric edges in the room, which is a prerequisite for simulating sound diffraction effects.
    *   **Data Structures:** Defines interfaces like `RayHit`, `RayPathPoint`, `ImpulseResponse`, `ImageSource`, `Edge` to structure the simulation data.
    *   **Visualization Preparation:** Collects ray path data (`rayPaths`, `rayPathPoints`) formatted for use by the `RayRenderer`.
    *   **GPU Acceleration:** Potentially offloads parts of the computation (like intersection testing) to the GPU via compute shaders (see `shaders/raytracer.wgsl`). Manages relevant GPU buffers (`rayPointsBuffer`).

### `shaders/` (Directory)

*   **Purpose:** Contains WGSL (WebGPU Shading Language) files for GPU computations.
*   **Files:**
    *   **`raytracer.wgsl`:** Expected to contain the compute shader(s) for accelerating the core ray tracing algorithm (e.g., ray-triangle intersection, BVH traversal if used). This allows massively parallel processing of rays.
    *   **`spatial_audio.wgsl`:** May contain other GPU compute shaders related to spatial audio processing tasks that benefit from parallelization, such as HRTF convolution, reverb generation, or audio effect processing.

## Summary

The `src/raytracer` module forms the foundation of the acoustic simulation. `raytracer.ts` manages the simulation, `ray.ts` defines the fundamental ray object, `ray-renderer.ts` handles visualization, and the `shaders/` directory contains GPU code for acceleration. Together, these components simulate how sound propagates and reflects within the defined `Room` to generate data used for spatial audio rendering.

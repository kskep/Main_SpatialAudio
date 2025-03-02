# Project File Overview

This document provides a detailed description of each file (and folder) in the `src/` directory. It is intended to help developers and users understand the role, key aspects, and important properties of every part of the project. The project combines WebGPU-based 3D rendering with advanced spatial audio simulation and analysis.

---

## src/main.ts

**Purpose:**  
The entry point and core orchestrator of the application.

**Key Responsibilities:**
- Initializes the WebGPU context and configures the canvas.
- Sets up core components including the room, camera, sound source (sphere), and ray tracer.
- Integrates and coordinates the rendering loop, audio processing, and user interaction (using dat.GUI for debugging/configuration).
- Handles window resizing and triggers re-rendering of the scene.
- Starts the animation loop that continuously updates and renders the scene.

**Important Aspects:**
- Holds references to global objects such as the Camera, Room, Sphere, RayTracer, AudioProcessor, and WaveformRenderer.
- Manages the canvas, GPU device, and context, ensuring proper resource allocation and cleanup.
- Acts as the central integration point for user input and system state management.

---

## src/camera/camera.ts

**Purpose:**  
Implements a 3D camera system using the `gl-matrix` library.

**Key Responsibilities:**
- Maintains the camera's position, orientation (yaw and pitch), and view direction.
- Offers methods to rotate the camera using both mouse (smooth, sensitive rotation) and keyboard input (fixed speed rotation).
- Provides movement methods (`moveForward`, `moveRight`, `moveUp`) that alter the position based on the camera's current heading.
- Computes the view and view-projection matrices needed for rendering.
- Contains debug functionality to retrieve and display current camera parameters.

**Important Aspects:**
- Properties include the camera's position vector, orientation angles (yaw and pitch), and derived vectors (front, up, and right).
- Calculates essential matrices for transforming 3D world coordinates into camera-space for rendering.

---

## src/objects/sphere.ts

**Purpose:**  
Represents a sound source in the scene.

**Key Responsibilities:**
- Encapsulates the position and radius of the sphere, which visually represents the sound source.
- Provides methods to update or set the sphere's position.
- Returns its radius for use during rendering or collision calculations.
- Note: The `update` method is a placeholder, as the sound source is static by design.

**Important Aspects:**
- Holds fundamental properties such as a position vector and a radius value.
- Serves as the origin point for the acoustic simulation (i.e., where sound rays originate).

---

## src/objects/sphere-renderer.ts

**Purpose:**  
Handles the visual rendering of the sphere (sound source) within the 3D scene.

**Key Responsibilities:**
- Generates sphere mesh data (vertices and indices) for a visually smooth object.
- Creates GPU buffers for vertex data, index data, and transformation uniforms.
- Sets up a WebGPU render pipeline (vertex & fragment shaders) to draw the sphere.
- Updates the sphere's model matrix (translation and scaling) and combines it with the view-projection matrix before drawing.
- Issues WebGPU commands to draw the sphere based on the current scene state.

**Important Aspects:**
- Manages mesh construction and GPU buffer allocation.
- Applies transformation data so that the rendered sphere accurately represents its position and scale in the scene.

---

## src/raytracer/ray.ts

**Purpose:**  
Defines the `Ray` class used in the ray tracing simulation for sound propagation.

**Key Responsibilities:**
- Stores the ray's origin, normalized direction, and energy levels across eight frequency bands.
- Tracks the cumulative path length, number of bounces, time of travel, and phase information.
- Provides methods to update the ray's state upon interactions with room surfaces (e.g., energy loss, phase change on reflection).
- Implements calculations for Doppler effects and air absorption effects.
- Serves as the basic unit for the overall ray tracing audio simulation.

**Important Aspects:**
- **Properties:**
  - **Origin:** A vector representing the starting position of the ray.
  - **Direction:** A normalized vector showing the ray's travel direction.
  - **Energies:** An array or object storing energy values for eight frequency bands (e.g., 125Hz, 250Hz, 500Hz, 1kHz, 2kHz, 4kHz, 8kHz, 16kHz).
  - **Cumulative Path Length:** A float tracking the distance traveled by the ray.
  - **Number of Bounces:** An integer that counts how many surfaces the ray has interacted with.
  - **Time and Phase:** Values used to compute wave contributions and Doppler effects.
- This file lays the groundwork for passing detailed acoustic simulation data to other parts of the project.

---

## src/raytracer/raytracer.ts

**Purpose:**  
Performs and manages the ray tracing simulation for sound propagation.

**Key Responsibilities:**
- Initializes and manages a collection of `Ray` instances from the sound source.
- Interfaces with room geometry and material properties to compute reflections and energy decay.
- Utilizes the `RayRenderer` to visualize calculated ray paths.
- Collects "ray hit" data (position, energy, time, phase, Doppler shift) for subsequent audio processing.
- Offers functions to trigger recalculation and re-rendering of rays as scene parameters change.

**Important Aspects:**
- Maintains collections such as arrays of rays, ray hits, and ray path points.
- Configurable parameters (e.g., number of rays, maximum bounces) determine the simulation fidelity.
- Acts as the simulation engine, updating ray states based on interactions with room surfaces and materials.

---

## src/raytracer/ray-renderer.ts

**Purpose:**  
Visualizes the paths of rays computed by the ray tracing simulation.

**Key Responsibilities:**
- Creates GPU buffers to store vertex and uniform data corresponding to the computed ray paths.
- Sets up a dedicated render pipeline for drawing lines that represent the rays.
- Updates and writes transformation matrices, including view-projection and model matrices for each ray segment.
- Integrates frequency band data (energy, phase) into the vertex attributes.
- Supports resetting the render state when the camera moves or when a new ray calculation is performed.

**Important Aspects:**
- Manages visualization-specific properties such as vertex buffer content and color coding based on ray energy.
- Ensures that the dynamic nature of the simulation is promptly reflected in the rendered output.

---

## src/raytracer/shaders/raytracer.wgsl

**Purpose:**  
Defines the WGSL shader code for the ray tracing simulation.

**Key Responsibilities:**
- Declares data structures for rays, surfaces, intersections, and ray hits.
- Implements logic to calculate intersections of rays with room surfaces.
- Computes critical simulation data such as ray path length, energy loss (based on material absorption), and phase accumulation.
- Updates ray state for multiple bounces and writes hit information used later by the audio processor.
- Plays a central role in simulating the physics of sound propagation within the scene.

**Important Aspects:**
- Uses parallel GPU computation to process multiple rays simultaneously.
- Embeds physical simulation parameters directly into the shader logic (e.g., absorption coefficients, bounce calculations).

---

## src/raytracer/shaders/spatial_audio.wgsl

**Purpose:**  
Provides WGSL shader code for spatial audio processing.

**Key Responsibilities:**
- Processes ray hit data to calculate an audio impulse response.
- Structures the listener's data (position, orientation vectors) and integrates it with ray hit properties.
- Computes frequency-dependent attenuation, air absorption, and directional effects.
- Outputs left and right audio channel contributions along with frequency and timing information.
- Serves as the GPU-accelerated component for generating spatially aware audio from the simulation.

**Important Aspects:**
- Converts sparse ray sampling data into per-sample contributions for a dense impulse response.
- Handles the transformation of physical simulation parameters (like phase and energy) into audible effects.

---

## src/room/room.ts

**Purpose:**  
Implements the room environment in which the simulation occurs.

**Key Responsibilities:**
- Manages the 3D geometry (dimensions) of the room and its rendering.
- Creates GPU buffers and sets up a render pipeline for drawing the room boundaries (walls, floor, ceiling).
- Provides utility methods to compute room volume, surface area, and fetch environmental parameters (e.g., temperature, humidity).
- Offers a method to ensure objects remain within valid positions inside the room.
- Uses material properties (from room-materials) to help inform acoustic simulations.

**Important Aspects:**
- Properties include the room's dimensions, physical boundaries, and environmental parameters.
- Plays a vital role in influencing ray behavior through geometry constraints and material interactions.

---

## src/room/room-materials.ts

**Purpose:**  
Defines the material properties for the room's surfaces.

**Key Responsibilities:**
- Declares the `WallMaterial` interface for encapsulating absorption and scattering coefficients across eight frequency bands.
- Declares the `RoomMaterials` interface to group materials for all six faces of the room.
- Provides standard material presets (e.g., CONCRETE, WOOD) with detailed frequency responses.
- Supplies values for roughness, phase shift, and phase randomization to simulate realistic reflections and diffractions.

**Important Aspects:**
- Sets the physical parameters that directly affect how sound interacts with surfaces.
- Material properties defined here are used throughout the simulation to calculate energy decay and reflection behavior.

---

## src/room/types.ts

**Purpose:**  
Contains type definitions for room configuration.

**Key Responsibilities:**
- Defines interfaces and types for room dimensions and material properties.
- Enumerates different surface types (e.g., FLOOR, CEILING, walls) for use in rendering and simulation.
- Enforces type safety in room modules and integration with other systems.

**Important Aspects:**
- Provides a robust type system ensuring consistent use of physical properties and dimensions.
- Acts as a central reference for all room-related data structures within the project.

---

## src/sound/audio-processor.ts

**Purpose:**  
Processes the computed ray hit data to generate an impulse response (IR) for the audio simulation.

**Key Responsibilities:**
- Integrates with the WebAudio API to create and configure audio buffers.
- Uses computed ray hit data to form an IR, accounting for energy decay, reflections, and frequency-dependent effects.
- Normalizes and applies an envelope to the generated IR to ensure a smooth audio output.
- Interfaces with the `SpatialAudioProcessor` to convert 3D simulation data into audio signals.
- Provides logging and error handling to ensure robust processing of audio data.

**Important Aspects:**
- Converts physical simulation results into an impulse response suitable for convolution-based audio rendering.
- Balances computations between CPU and GPU processing to achieve real-time audio synthesis.

---

## src/sound/spatial-audio-processor.ts

**Purpose:**  
Performs GPU-accelerated spatial audio processing.

**Key Responsibilities:**
- Sets up a compute pipeline in WebGPU to process ray hit information and transform it into an audio impulse response.
- Allocates GPU buffers for listener data, ray hit information, wave properties, and the output spatial IR.
- Dispatches compute shaders to compute per-sample contributions for left and right audio channels.
- Reads back and validates processed audio data from the GPU.
- Ensures efficient spatial audio simulation by leveraging parallel computation on the GPU.

**Important Aspects:**
- Key properties include GPU buffers and pipeline configurations that manage ray hit data and wave properties.
- Implements time binning to convert sparse ray contributions into a densely sampled impulse response array.
- Essential for producing spatially accurate audio output from geometric and material simulations.

---

## src/visualization/waveform-renderer.ts

**Purpose:**  
Renders real-time visualizations (waveforms and FFT analysis) of the audio impulse response.

**Key Responsibilities:**
- Draws the waveform for left and right audio channels on an HTML canvas.
- Contains an FFT implementation to analyze the frequency content of the audio signal.
- Renders additional visual aids such as zero lines and time markers to help interpret the audio signal.
- Scales and smooths the audio data so that the waveform fills the canvas and is visually appealing.
- Useful for debugging and understanding the spatial audio's temporal and spectral behavior.

**Important Aspects:**
- Uses canvas rendering techniques to convert audio data into graphical representations.
- Provides real-time feedback that aids in tuning both the audio processing and the overall simulation parameters.

---

## src/webgpu.d.ts

**Purpose:**  
Offers TypeScript definitions for the experimental WebGPU API.

**Key Responsibilities:**
- Declares types, interfaces, and global objects (e.g., `GPUDevice`, `GPUAdapter`, `GPUTexture`, etc.) required to work with WebGPU.
- Ensures strong type checking for WebGPU-related code in the project.
- Serves as a bridge between experimental WebGPU features and TypeScript's static type system.

**Important Aspects:**
- Acts as the foundational type definition layer ensuring that GPU programming errors are caught at compile time.
- Provides crucial support for the overall stability and compatibility of the WebGPU-dependent modules.

---

## Future Enhancements

- **Hybrid Acoustic Simulation:** Combine image-source methods with the current ray tracing approach to model early reflections more accurately.
- **Performance Optimizations:** Investigate GPU memory optimizations and adaptive sampling techniques to improve processing speed.
- **Extended Material Models:** Incorporate more complex material properties and dynamic environmental factors (e.g., humidity, temperature) to refine acoustic simulation fidelity.
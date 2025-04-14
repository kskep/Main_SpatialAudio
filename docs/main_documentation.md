# Main Module Documentation

This document provides a detailed explanation of the main module located in the `src/main.ts` file. The module serves as the entry point and central coordinator for the spatial audio simulation application.

## Table of Contents

1. [Overview](#overview)
2. [Main Class](#main-class)
3. [Initialization](#initialization)
4. [User Interface](#user-interface)
5. [Input Handling](#input-handling)
6. [Rendering](#rendering)
7. [Audio Processing](#audio-processing)
8. [Animation Loop](#animation-loop)
9. [Application Initialization](#application-initialization)
10. [Key Concepts and Techniques](#key-concepts-and-techniques)
11. [Relationships with Other Modules](#relationships-with-other-modules)

## Overview

The main module serves as the entry point for the spatial audio simulation application. It initializes the WebGPU context, sets up the 3D environment, manages user interaction, coordinates the rendering loop, and orchestrates the various components of the system. It acts as the central hub that connects the room, camera, sound source, ray tracer, and audio processor modules.

## Main Class

The `Main` class is the primary component of the module, responsible for managing the application state and coordinating all subsystems.

### Properties

- **Rendering Properties**:
  - `canvas`: HTML canvas element for rendering
  - `device`: WebGPU device for GPU operations
  - `context`: WebGPU canvas context
  - `depthTexture`: Texture for depth testing

- **Scene Objects**:
  - `room`: Room object representing the virtual environment
  - `camera`: Camera object representing the listener's position and orientation
  - `sphere`: Sphere object representing the sound source
  - `roomConfig`: Configuration for room dimensions and materials

- **Renderers**:
  - `sphereRenderer`: Renderer for the sound source sphere
  - `waveformRenderer`: Renderer for audio visualization

- **Audio and Simulation**:
  - `rayTracer`: Ray tracer for acoustic simulation
  - `audioProcessor`: Instance of `AudioProcessorModified` for audio generation and playback.

- **User Interface**:
  - `gui`: dat.GUI instance for the control panel
  - `sourceControllers`: References to UI controllers for source position
  - `sourceParams`: Parameters for the sound source

- **Input Handling**:
  - `keys`: Object tracking keyboard state

## Initialization

The constructor of the `Main` class performs the following initialization steps:

1. **WebGPU Setup**:
   - Stores references to the canvas and device
   - Configures the WebGPU context

2. **Room Configuration**:
   - Initializes room dimensions (8m × 3m × 5m)
   - Sets up material properties for walls, ceiling, and floor with frequency-dependent absorption and scattering

3. **Scene Objects**:
   - Creates the room with the specified configuration
   - Initializes the camera near the back wall at eye level (1.7m)
   - Creates the sound source sphere in the center of the room

4. **Renderers**:
   - Initializes the sphere renderer for the sound source
   - Creates and styles a canvas for waveform visualization
   - Initializes the waveform renderer

5. **Simulation Components**:
   - Sets up the ray tracer with references to the room, source, and listener
   - Initializes the modified audio processor (`AudioProcessorModified`).

6. **User Interface and Input**:
   - Calls `setupDebugUI()` to create the control panel
   - Calls `setupInputHandlers()` to set up keyboard controls
   - Creates the depth texture for 3D rendering

## User Interface

The `setupDebugUI()` method creates a comprehensive control panel using dat.GUI with the following sections:

1. **Room Dimensions**:
   - Sliders for width, height, and depth
   - Updates the room when values change

2. **Materials**:
   - Controls for absorption coefficients at different frequencies
   - Separate folders for walls, ceiling, and floor

3. **Sound Source**:
   - Position controls (X, Y, Z) with ranges based on room dimensions
   - Power control in decibels

4. **Ray Tracing**:
   - Button to calculate impulse response
   - Button to play convolved sound

5. **Audio Debug**:
   - Various buttons for testing audio playback
   - Options for sine waves, clicks, and noise with impulse response

The `updateRoom()` method is called when room dimensions change, which:
- Recreates the room with new dimensions
- Ensures the camera stays within bounds
- Updates source position slider ranges
- Keeps the sphere at a valid position

## Input Handling

The `setupInputHandlers()` method sets up keyboard controls:

1. **Movement Controls**:
   - WASD keys for horizontal movement
   - Space/Shift for vertical movement

2. **Rotation Controls**:
   - Arrow keys for camera rotation

The `handleInput()` method processes keyboard input each frame:
- Moves the camera based on which keys are pressed
- Applies movement relative to the camera's orientation
- Scales movement by delta time for consistent speed

The `constrainCamera()` method ensures the camera stays within the room boundaries.

## Rendering

The `render()` method performs the rendering for each frame:

1. **Input Processing**:
   - Calls `handleInput()` to process keyboard input

2. **Camera Update**:
   - Gets the view-projection matrix from the camera
   - Updates the room's uniform buffer

3. **Render Pass Setup**:
   - Creates a command encoder
   - Begins a render pass with color and depth attachments

4. **Scene Rendering**:
   - Renders the room wireframe
   - Renders the sound source sphere
   - Renders ray paths if available

5. **Command Submission**:
   - Ends the render pass
   - Submits the command buffer

The `resize()` method handles canvas resizing:
- Updates canvas dimensions to match the display size
- Recreates the depth texture with the new dimensions

## Audio Processing

The `calculateIR()` method coordinates the acoustic simulation and audio processing:

1. **Ray Tracing**:
   - Calls `rayTracer.calculateRayPaths()` to perform the acoustic simulation
   - Retrieves ray hits from the ray tracer

2. **Audio Processing**:
   - Passes ray hits to the audio processor
   - Generates the impulse response

3. **Visualization**:
   - Updates the waveform and spectrogram visualizations

## Animation Loop

The animation loop is implemented in the `animate()` function:

1. **Time Calculation**:
   - Calculates delta time between frames

2. **Frame Update**:
   - Calls `main.resize()` to handle any size changes
   - Calls `main.render()` to render the frame

3. **Loop Continuation**:
   - Uses `requestAnimationFrame()` to schedule the next frame

## Application Initialization

The `init()` function initializes the application:

1. **Canvas Retrieval**:
   - Gets the canvas element from the DOM

2. **WebGPU Setup**:
   - Requests a WebGPU adapter
   - Creates a WebGPU device

3. **Application Creation**:
   - Creates an instance of the `Main` class

4. **Animation Start**:
   - Starts the animation loop

## Key Concepts and Techniques

### WebGPU Integration
The main module demonstrates several WebGPU concepts:
1. **Device and Context Setup**: Initializes WebGPU for rendering
2. **Render Pass Management**: Creates and manages render passes
3. **Command Encoding**: Encodes rendering commands
4. **Resource Management**: Creates and updates buffers and textures

### Application Architecture
The module implements a well-structured application architecture:
1. **Component-Based Design**: Separates functionality into specialized components
2. **Centralized Coordination**: Main class orchestrates all subsystems
3. **Event-Driven Updates**: Responds to user input and UI changes

### User Interface Design
The control panel provides a comprehensive interface:
1. **Hierarchical Organization**: Groups controls by function
2. **Dynamic Constraints**: Updates control ranges based on room dimensions
3. **Real-Time Feedback**: Immediate updates when parameters change

### Animation and Timing
The animation loop implements proper timing:
1. **Delta Time Calculation**: Ensures consistent movement speed
2. **Frame Scheduling**: Uses requestAnimationFrame for efficient rendering
3. **Resize Handling**: Adapts to window size changes

## Relationships with Other Modules

The main module interacts with all other components of the system:

- **Room Module**: Creates and configures the virtual environment
  - Initializes the room with dimensions and materials
  - Updates room properties based on user input

- **Camera Module**: Manages the listener's position and view
  - Initializes the camera at a specific position
  - Processes input to move and rotate the camera

- **Objects Module**: Handles the sound source
  - Creates the sphere representing the sound source
  - Updates its position based on user input

- **Raytracer Module**: Performs the acoustic simulation
  - Initializes the ray tracer with scene objects
  - Triggers ray path calculation
  - Retrieves ray hits for audio processing

- **Sound Module**: Processes audio based on simulation results using `AudioProcessorModified`.
  - Initializes the modified audio processor.
  - Passes ray hits for impulse response generation
  - Triggers audio playback

- **Visualization Module**: Displays audio data
  - Creates the visualization canvas
  - Updates visualizations when the impulse response changes

# Main Application Documentation

## File: `src/main.ts`

This file serves as the entry point for the spatial audio simulation application. It initializes the WebGPU context, sets up the 3D environment, handles user input, and coordinates the rendering and audio processing.

## Class: `Main`

The `Main` class is the central controller for the application, integrating all the components of the spatial audio simulation.

### Properties

- `canvas: HTMLCanvasElement` - The canvas element for rendering
- `device: GPUDevice` - The WebGPU device for GPU operations
- `context: GPUCanvasContext` - The WebGPU canvas context
- `room: Room` - The 3D room environment
- `camera: Camera` - The first-person camera for navigation
- `depthTexture: GPUTexture` - Depth texture for 3D rendering
- `keys: { [key: string]: boolean }` - Object tracking keyboard input state
- `roomConfig: RoomConfig` - Configuration for the room dimensions and materials
- `gui: dat.GUI` - Debug UI for controlling parameters
- `sphere: Sphere` - The sound source object
- `sphereRenderer: SphereRenderer` - Renderer for the sound source
- `sourceControllers` - GUI controllers for the sound source position
- `rayTracer: RayTracer` - Ray tracing system for sound propagation
- `audioProcessor: AudioProcessor` - Audio processing system
- `waveformRenderer: WaveformRenderer` - Visualization for audio waveforms
- `sourceParams` - Parameters for the sound source, including power level

### Methods

#### `constructor(canvas: HTMLCanvasElement, device: GPUDevice)`
- Initializes the application with the provided canvas and WebGPU device
- Configures the WebGPU context
- Sets up the room, camera, sphere, renderers, and audio processor
- Creates a waveform visualization canvas
- Sets up the debug UI and input handlers

#### `private setupDebugUI(): void`
- Creates a dat.GUI interface with controls for:
  - Room dimensions
  - Material properties
  - Sound source position
  - Source power
  - Ray tracing controls
  - Audio debug controls

#### `private updateRoom(): void`
- Recreates the room with updated dimensions
- Constrains the camera to stay within room bounds
- Updates source position slider ranges
- Keeps the sphere at a valid position

#### `private constrainCamera(): void`
- Ensures the camera stays within the room boundaries
- Applies a margin to keep the camera slightly away from walls

#### `private setupInputHandlers(): void`
- Sets up keyboard event listeners for movement and rotation
- Maps WASD keys to movement and arrow keys to rotation

#### `private createDepthTexture(): void`
- Creates a depth texture for 3D rendering with the canvas dimensions

#### `private handleInput(deltaTime: number): void`
- Processes keyboard input for camera movement
- Applies movement based on the keys being pressed

#### `public render(deltaTime: number): void`
- Handles input processing
- Updates the view projection matrix
- Creates a render pass
- Renders the room, sphere, and ray visualization
- Submits the command buffer to the GPU

#### `public resize(): void`
- Resizes the canvas and depth texture when the window size changes

#### `private async calculateIR(): Promise<void>`
- Calculates ray paths for sound propagation
- Processes ray hits to generate an impulse response
- Visualizes the impulse response waveform

### Functions

#### `animate(main: Main, time: number)`
- Animation loop function that:
  - Calculates delta time
  - Calls resize and render methods
  - Requests the next animation frame

#### `async init()`
- Initializes the application:
  - Gets the canvas element
  - Requests a WebGPU adapter and device
  - Creates a Main instance
  - Starts the animation loop

## Relationships

This file imports and integrates multiple modules:
- `room/room.ts` - For creating the 3D room environment
- `room/types.ts` - For room configuration types
- `camera/camera.ts` - For first-person navigation
- `objects/sphere.ts` - For the sound source object
- `objects/sphere-renderer.ts` - For rendering the sound source
- `raytracer/raytracer.ts` - For sound propagation simulation
- `sound/audio-processor.ts` - For audio processing
- `visualization/waveform-renderer.ts` - For audio visualization

The Main class serves as the central coordinator, connecting all these components together to create a complete spatial audio simulation system. 
# Camera Module Documentation

## File: `src/camera/camera.ts`

This file implements a first-person camera system for navigating the 3D environment.

## Class: `Camera`

The `Camera` class provides functionality for a first-person camera that can move and rotate within the 3D space.

### Properties

- `position: vec3` - The 3D position of the camera in world space
- `yaw: number` - The horizontal rotation angle in degrees (around the Y axis)
- `pitch: number` - The vertical rotation angle in degrees (around the X axis)
- `front: vec3` - The normalized direction vector the camera is facing
- `up: vec3` - The up vector for the camera, typically (0, 1, 0)
- `PITCH_LIMIT: number` - Constant limiting the pitch angle to prevent camera flipping (89 degrees)
- `MOVEMENT_SPEED: number` - Constant controlling how fast the camera moves (5.0 units per second)
- `MOUSE_SENSITIVITY: number` - Constant controlling how sensitive the camera is to mouse movement (0.1)
- `KEYBOARD_ROTATION_SPEED: number` - Constant controlling how fast the camera rotates with keyboard input (2.0 degrees)

### Methods

#### `constructor(position: vec3, yaw: number, pitch: number)`
- Initializes a new camera with the given position and rotation angles
- Default values: position at (0, 1.7, 3), yaw at -90 degrees, pitch at 0 degrees
- Initializes the front and up vectors and calls `updateVectors()`

#### `private updateVectors(): void`
- Recalculates the front vector based on the current yaw and pitch angles
- Uses trigonometry to convert angles to a direction vector
- Normalizes the front vector

#### `rotate(deltaX: number, deltaY: number): void`
- Rotates the camera based on mouse movement
- Applies mouse sensitivity to the input deltas
- Updates yaw (horizontal) and pitch (vertical) angles
- Constrains pitch to prevent camera flipping
- Calls `updateVectors()` to recalculate the front vector

#### `moveForward(distance: number): void`
- Moves the camera forward or backward along the front vector
- Scales movement by the movement speed and the provided distance

#### `moveRight(distance: number): void`
- Moves the camera left or right perpendicular to the front vector
- Calculates the right vector using cross product of front and up vectors
- Scales movement by the movement speed and the provided distance

#### `moveUp(distance: number): void`
- Moves the camera up or down along the world up vector
- Scales movement by the movement speed and the provided distance

#### `getViewMatrix(): mat4`
- Returns the view matrix for the camera
- Creates a lookAt matrix from the camera position, target (position + front), and up vector

#### `getViewProjection(aspect: number): mat4`
- Returns the combined view-projection matrix
- Creates a perspective projection matrix with 45-degree FOV, the given aspect ratio, and near/far planes at 0.1 and 100.0
- Multiplies the projection matrix by the view matrix

#### `getPosition(): vec3`
- Returns a copy of the camera's position vector

#### `getFront(): vec3`
- Returns a copy of the camera's front direction vector

#### `getUp(): vec3`
- Returns a copy of the camera's up vector

#### `getDebugInfo(): { position: vec3, yaw: number, pitch: number }`
- Returns an object with debug information about the camera's state
- Includes position, yaw, and pitch

#### `rotateWithKeyboard(direction: 'left' | 'right' | 'up' | 'down'): void`
- Rotates the camera based on keyboard input
- Updates yaw for left/right and pitch for up/down
- Constrains pitch to prevent camera flipping
- Calls `updateVectors()` to recalculate the front vector

#### `setPosition(position: vec3): void`
- Sets the camera's position to a copy of the provided vector

## Relationships

This module is imported by:
- `main.ts` - Uses the Camera class for navigation in the 3D environment
- `raytracer.ts` - Uses the camera position for ray calculations
- `audio-processor.ts` - Uses the camera position for spatial audio processing 
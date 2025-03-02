# Sphere Object Documentation

## File: `src/objects/sphere.ts`

This file implements a simple sphere object that represents a sound source in the spatial audio simulation.

## Class: `Sphere`

The `Sphere` class represents a spherical sound source that can be positioned in the 3D environment.

### Properties

- `position: vec3` - The 3D position of the sphere in world space
- `radius: number` - The radius of the sphere
- `DEFAULT_RADIUS: number` - Constant defining the default radius (0.2 units)

### Methods

#### `constructor(position: vec3, radius: number)`
- Initializes a new sphere with the given position and radius
- Default values: position at (0, 1.7, 0), radius of 0.2 units
- Clones the position vector to ensure the sphere has its own copy

#### `update(deltaTime: number, roomDimensions: { width: number, height: number, depth: number }): void`
- Method for updating the sphere's state over time
- Currently does nothing as the sound source is static
- Included for potential future animations or dynamic behavior

#### `setPosition(position: vec3): void`
- Sets the sphere's position to the provided vector
- Uses vec3.copy to update the internal position

#### `getPosition(): vec3`
- Returns a copy of the sphere's position vector

#### `getRadius(): number`
- Returns the sphere's radius

## Relationships

This module is imported by:
- `main.ts` - Creates a sphere instance as the sound source
- `raytracer.ts` - Uses the sphere as the origin for ray tracing
- `sphere-renderer.ts` - Uses the sphere's properties for rendering 
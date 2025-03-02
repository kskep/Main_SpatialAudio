# Room Types Documentation

## File: `src/room/types.ts`

This file defines the core data structures and types used for room configuration and surface identification in the spatial audio simulation.

## Interfaces

### `RoomConfig`

Defines the configuration for a room, including its dimensions and material properties.

- `dimensions` - Object containing the room's width, height, and depth
- `materials` - Object containing absorption properties for different surfaces

### `RoomDimensions`

Defines the dimensions of a room.

- `width: number` - Width of the room (X-axis)
- `height: number` - Height of the room (Y-axis)
- `depth: number` - Depth of the room (Z-axis)

### `RoomMaterials`

Defines the material properties for different surfaces in the room.

- `walls: { absorption: number }` - Absorption coefficient for walls
- `ceiling: { absorption: number }` - Absorption coefficient for ceiling
- `floor: { absorption: number }` - Absorption coefficient for floor

## Enums

### `Surface`

Enumeration identifying different surface types in the room.

- `FLOOR = 0` - The floor surface
- `CEILING = 1` - The ceiling surface
- `WALL_FRONT_BACK = 2` - The front and back walls
- `WALL_LEFT_RIGHT = 3` - The left and right walls

## Relationships

This module is imported by:
- `room.ts` - Uses these types to define the room's properties
- `main.ts` - Uses the RoomConfig interface to configure the room
- `raytracer.ts` - Uses the Surface enum to identify which surface a ray hits 
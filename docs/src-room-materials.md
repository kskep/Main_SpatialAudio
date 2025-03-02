# Room Materials Documentation

## File: `src/room/room-materials.ts`

This file defines the acoustic material properties used in the spatial audio simulation, including detailed frequency-dependent absorption and scattering coefficients.

## Interfaces

### `RoomMaterials`

Defines the material properties for each surface of the room.

- `left: WallMaterial` - Material properties for the left wall
- `right: WallMaterial` - Material properties for the right wall
- `top: WallMaterial` - Material properties for the ceiling
- `bottom: WallMaterial` - Material properties for the floor
- `front: WallMaterial` - Material properties for the front wall
- `back: WallMaterial` - Material properties for the back wall

### `WallMaterial`

Defines the acoustic properties of a surface with frequency-dependent characteristics.

#### Absorption Coefficients
- `absorption125Hz: number` - Absorption coefficient at 125 Hz
- `absorption250Hz: number` - Absorption coefficient at 250 Hz
- `absorption500Hz: number` - Absorption coefficient at 500 Hz
- `absorption1kHz: number` - Absorption coefficient at 1000 Hz
- `absorption2kHz: number` - Absorption coefficient at 2000 Hz
- `absorption4kHz: number` - Absorption coefficient at 4000 Hz
- `absorption8kHz: number` - Absorption coefficient at 8000 Hz
- `absorption16kHz: number` - Absorption coefficient at 16000 Hz

#### Scattering Coefficients
- `scattering125Hz: number` - Scattering coefficient at 125 Hz
- `scattering250Hz: number` - Scattering coefficient at 250 Hz
- `scattering500Hz: number` - Scattering coefficient at 500 Hz
- `scattering1kHz: number` - Scattering coefficient at 1000 Hz
- `scattering2kHz: number` - Scattering coefficient at 2000 Hz
- `scattering4kHz: number` - Scattering coefficient at 4000 Hz
- `scattering8kHz: number` - Scattering coefficient at 8000 Hz
- `scattering16kHz: number` - Scattering coefficient at 16000 Hz

#### Other Properties
- `roughness: number` - Surface roughness (0-1), affects reflection pattern
- `phaseShift: number` - Fixed phase shift on reflection (radians)
- `phaseRandomization: number` - Maximum random phase variation (radians)

## Constants

### `MATERIAL_PRESETS`

A collection of predefined material presets with realistic acoustic properties.

#### `CONCRETE`
Preset for concrete surfaces with:
- Low absorption (0.01-0.06 across frequency bands)
- Moderate scattering (0.10-0.25 across frequency bands)
- Low roughness (0.1)
- No phase shift
- Low phase randomization (0.1 radians)

#### `WOOD`
Preset for wooden surfaces with:
- Moderate absorption (0.06-0.15 across frequency bands, higher at low frequencies)
- Higher scattering (0.15-0.50 across frequency bands, increasing with frequency)
- Moderate roughness (0.3)
- No phase shift
- Moderate phase randomization (0.2 radians)

## Relationships

This module is imported by:
- `room.ts` - Uses these material definitions to configure the acoustic properties of the room
- `raytracer.ts` - Uses the material properties to calculate sound reflection characteristics
- `audio-processor.ts` - Uses the absorption and scattering coefficients for impulse response generation 
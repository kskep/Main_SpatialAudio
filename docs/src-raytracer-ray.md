# Ray Class Documentation

## File: `src/raytracer/ray.ts`

This file implements the Ray class, which represents a single sound ray in the acoustic ray tracing simulation. It tracks the ray's path, energy levels across frequency bands, and other properties as it propagates through the environment.

## Interfaces

### `FrequencyBands`
Represents energy levels across different frequency bands.
- `energy125Hz: number` - Energy at 125 Hz
- `energy250Hz: number` - Energy at 250 Hz
- `energy500Hz: number` - Energy at 500 Hz
- `energy1kHz: number` - Energy at 1 kHz
- `energy2kHz: number` - Energy at 2 kHz
- `energy4kHz: number` - Energy at 4 kHz
- `energy8kHz: number` - Energy at 8 kHz
- `energy16kHz: number` - Energy at 16 kHz

## Class: `Ray`

The `Ray` class represents a single sound ray used in acoustic ray tracing.

### Properties

- `origin: vec3` - The current position of the ray
- `direction: vec3` - The normalized direction vector of the ray
- `energies: FrequencyBands` - Energy levels across frequency bands
- `pathLength: number` - Total distance the ray has traveled
- `bounces: number` - Number of reflections the ray has undergone
- `isActive: boolean` - Whether the ray is still active in the simulation
- `time: number` - Time elapsed since the ray was emitted
- `phase: number` - Current phase of the sound wave
- `frequency: number` - Frequency of the ray in Hz

### Methods

#### `constructor(origin: vec3, direction: vec3, initialEnergy: number = 1.0, frequency: number = 1000)`
- Initializes a new ray with the given origin, direction, initial energy, and frequency
- Normalizes the direction vector
- Sets all frequency bands to the same initial energy
- Initializes path length, bounces, time, and phase to zero
- Sets the ray as active

#### `getOrigin(): vec3`
- Returns a copy of the ray's current origin position

#### `getDirection(): vec3`
- Returns a copy of the ray's current direction vector

#### `getEnergies(): FrequencyBands`
- Returns a copy of the ray's energy levels across frequency bands

#### `getAverageEnergy(): number`
- Calculates and returns the average energy across all frequency bands

#### `getBounces(): number`
- Returns the number of reflections the ray has undergone

#### `isRayActive(): boolean`
- Returns whether the ray is still active in the simulation

#### `getTime(): number`
- Returns the time elapsed since the ray was emitted

#### `getPhase(): number`
- Returns the current phase of the sound wave

#### `getFrequency(): number`
- Returns the frequency of the ray

#### `updateTime(newTime: number): void`
- Updates the ray's time to the specified value

#### `updatePhase(newPhase: number): void`
- Updates the ray's phase to the specified value

#### `updateRay(newOrigin: vec3, newDirection: vec3, energyLoss: {...}, distance: number, temperature: number = 20, humidity: number = 50): void`
- Updates the ray after a reflection or propagation
- Parameters:
  - `newOrigin` - New position of the ray
  - `newDirection` - New direction of the ray
  - `energyLoss` - Material absorption coefficients across frequency bands
  - `distance` - Distance traveled in this step
  - `temperature` - Air temperature in Celsius (default: 20°C)
  - `humidity` - Relative humidity percentage (default: 50%)
- Updates the ray's origin and direction
- Calculates air absorption based on distance, temperature, and humidity
- Applies both material absorption and air absorption to the ray's energy
- Updates path length, bounces, and checks if the ray should be deactivated

#### `private calculateAirAbsorption(distance: number, temperature: number, humidity: number): {...}`
- Calculates frequency-dependent air absorption
- Uses empirical formulas based on ISO 9613-1
- Returns absorption coefficients for each frequency band

#### `deactivate(): void`
- Marks the ray as inactive, removing it from further simulation

#### `setEnergies(energies: FrequencyBands): void`
- Sets the ray's energy levels across frequency bands to the specified values

## Relationships

This module is imported by:
- `raytracer.ts` - Creates and manages rays for the acoustic simulation

The Ray class is a fundamental component of the acoustic ray tracing system, representing the individual sound rays that propagate through the environment and interact with surfaces. 
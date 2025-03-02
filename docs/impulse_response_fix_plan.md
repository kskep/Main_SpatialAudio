# Impulse Response Fix Implementation Plan

## 1. Room Property Calculations Update
```typescript
// In RayTracer.ts calculateRoomProperties()
const materials = this.room.config.materials;
const surfaceAreas = {
    walls: 2 * (width * height + width * depth),
    floor: width * depth,
    ceiling: width * depth
};

const totalAbsorption =
    surfaceAreas.walls * materials.walls.absorptionMid +
    surfaceAreas.floor * materials.floor.absorptionMid +
    surfaceAreas.ceiling * materials.ceiling.absorptionMid;

const avgAbsorption = totalAbsorption / (surfaceAreas.walls + surfaceAreas.floor + surfaceAreas.ceiling);
this.roomProperties.reverbTime = 0.161 * volume / (avgAbsorption * surfaceArea);
```

## 2. Dynamic Time Resolution
```wgsl
// In raytracer.wgsl
let timePerSlot = f32(room.dimensions.x + room.dimensions.y + room.dimensions.z) / (343.0 * 20.0);
let slotIndex = u32(time / timePerSlot) % TIME_SLOTS;
```

## 3. Frequency-Dependent Phase
```wgsl
struct Ray {
    ...
    phaseLow: f32,
    phaseMid: f32,
    phaseHigh: f32
};

fn updatePhase(ray: Ray, distance: f32) -> Ray {
    ray.phaseLow = fract(ray.phaseLow + distance/0.343);
    ray.phaseMid = fract(ray.phaseMid + distance/0.343);
    ray.phaseHigh = fract(ray.phaseHigh + distance/0.08575);
    return ray;
}
```

## 4. Scaled Reflection Randomness
```wgsl
// In main() after line 155
let roomVolume = room.dimensions.x * room.dimensions.y * room.dimensions.z;
let sizeFactor = 1.0 - smoothstep(50.0, 500.0, roomVolume);
let randomness = ... * sizeFactor;
```

## Implementation Steps

1. Update RoomProperties struct to include material absorption data
2. Modify mean free path and reverb time calculations
3. Add frequency-specific phase tracking in Ray struct
4. Implement dynamic time slot calculation
5. Add room size-based randomness scaling
6. Update shader energy accumulation with phase-aware summation

## Validation Tests

1. Small room (5x5x5m) should show distinct early reflections <50ms
2. Large room (20x20x20m) should have reverb tail >500ms
3. High absorption materials should reduce late energy
4. Phase differences should create comb filtering effects

## User made notes

1. Core Improvements
1.1 - Room-Dependent Time Resolution

// In constructor
this.TIME_SLOTS = Math.max(1024, Math.floor(roomSize * 100)); // More slots for larger rooms
this.timeResolution = Math.min(0.01, roomSize / 1000); // Finer resolution for smaller rooms

// In processTemporalAccumulation()
const time = i * this.timeResolution; // Use variable resolution
1.2 Proper Energy Decay

// In processTemporalAccumulation()
const decay = Math.exp(-6.0 * time / this.roomProperties.reverbTime); // More realistic decay
const roomDependent = decay * (1.0 - time / (roomSize / 343.0)); // Room-size dependent factor
1.3 Material Absorption

// In calculateLateReflections()
const energyLoss = {
    low: material.absorptionLow * 0.8,    // More realistic absorption
    mid: material.absorptionMid * 0.8,
    high: material.absorptionHigh * 0.8
};
2. : Advanced Features
2.1 Wave Phenomena Implementation

// Add to RayTracer class
private calculateWaveEffects(ray: Ray, distance: number): void {
    const wavelength = 343.0 / frequency; // Speed of sound / frequency
    const waveNumber = 2 * Math.PI / wavelength;

    // Implement wavefront curvature effects
    const curvature = 1.0 / (distance + 0.1);
    ray.energy *= Math.exp(-curvature * waveNumber);
}
2.2 Modal Analysis

// Add to RoomProperties
modalFrequencies: number[]; // Track room modes

// In calculateRoomProperties()
this.roomProperties.modalFrequencies = this.calculateRoomModes();
2.3 Diffraction Effects

// Enhance edge diffraction
const diffraction = this.calculateUTDDiffraction(
    edgePoint,
    edgeNormal,
    diffractionAngle,
    frequency
);
3. Optimization
3.1 Performance Improvements

// Optimize time slot processing
const processTimeSlots = (timeSlots: Float32Array) => {
    // Use parallel processing for energy accumulation
    return this.device.queue.parallelFor(
        this.TIME_SLOTS,
        (index) => {
            // Process each time slot in parallel
            const baseIdx = index * 4;
            // ... energy calculations ...
        }
    );
};
3.2 Memory Optimization

// Use compressed storage for time slots
this.timeSlotsBuffer = this.device.createBuffer({
    size: this.TIME_SLOTS * 16, // 4 floats per slot
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true
});

## Validation Process

- Verify impulse response differences
- Measure reverb time variations
- Check reflection density changes
- Source Position Testing
- Test corner positions
- Test center positions
- Test near-wall positions
- Verify unique impulse responses
- Material Property Testing
- Test different absorption coefficients
- Verify frequency-dependent effects
- Check energy decay patterns
- Validate material interactions
- Future Enhancements
- Advanced Features
- Air absorption modeling
- Temperature effects
- Humidity influence
- Non-linear acoustic effects
- Performance Optimizations
- Dynamic resolution adjustment
- Adaptive sampling
- Multi-threaded processing
- GPU memory optimization
- Quality Improvements
- Higher frequency resolution
- More accurate material models
- Better diffraction handling
- Enhanced wave phenomena simulation
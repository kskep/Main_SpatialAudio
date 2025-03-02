# Creating an Impulse Response using WebGPU Ray Tracing

This document guides you through the process of capturing per-ray data (energy, frequency, phase, time) and using these values to generate an impulse response via summing sine wave contributions from each ray sample point.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step 1: Understanding the Concept](#step-1-understanding-the-concept)
4. [Step 2: Modify Your Shader (WGSL)](#step-2-modify-your-shader-wgsl)
5. [Step 3: Setup GPU Buffers and Read-Back](#step-3-setup-gpu-buffers-and-read-back)
6. [Step 4: Processing and Synthesizing the Impulse Response](#step-4-processing-and-synthesizing-the-impulse-response)
7. [Step 5: Testing and Debugging](#step-5-testing-and-debugging)
8. [Step 6: Further Improvements](#step-6-further-improvements)
9. [Advanced Features](#advanced-features)

---

## Overview

Your goal is to use WebGPU raytracing to simulate an acoustic impulse response. Rays are thrown once, and at every hit point (or along the ray's path) you capture:

| Parameter    | Description |
|-------------|-------------|
| **Energy**  | Amplitude contribution |
| **Frequency** | Wave frequency |
| **Phase**   | Wave phase |
| **Time**    | Ray's energy contribution at receiver |

Later on, you will reconstruct the waveform by generating sine waves from each ray sample (using the captured frequency and phase) and summing them based on their arrival time (i.e., performing time-aligned summation).

> 💡 **Key Concept**: This process is similar to wave interference where multiple sine waves add up (AM-like summation, aligned with physical principles).

---

## Prerequisites

Before starting, ensure you have:

- ✓ Solid understanding of WebGPU and WGSL
- ✓ Familiarity with TypeScript/JavaScript for pipelines and buffers
- ✓ Basic digital signal processing knowledge (sine wave synthesis and superposition)

---

## Step 1: Understanding the Concept

### Ray Sampling
In the raytracing shader, you sample points along each ray. Each sample records:
- Spatial location
- Energy
- Time of arrival
- Phase

### Wave Synthesis
On the CPU (or separate GPU compute pass), each sample point becomes a sine wave generator where:
- Amplitude is derived from **energy**
- **Frequency** and **phase** generate the sine wave
- Time-aligned samples are summed for interference effects

> 🎯 **Objective**: Create a discrete impulse response (time series) for acoustic simulation via convolution.

---

## Step 2: Modify Your Shader (WGSL)

Update your WGSL shader to sample along the ray's path and store a `RayPoint` at regular intervals.

```wgsl
struct RayPoint {
    position: vec3f,
    energy: f32,
    time: f32,
    phase: f32,
    frequency: f32
};
```

### Key Computations

The shader samples points every 0.1 meters along a ray path:

1. Use `t` for distance along ray
2. Calculate travel time: `travel_time = t / speed_of_sound`
3. Update phase: `phase = ray.phase + 2.0 * PI * ray.frequency * travel_time`

> ⚠️ **Important**: Verify your `main` function in *raytracer.wgsl* captures all required sample points.

---

## Step 3: Setup GPU Buffers and Read-Back

### Buffer Setup

1. Create Storage Buffer:
   ```typescript
   // Allocate buffer for rayPoints
   const rayPointsBuffer = device.createBuffer({
     size: BUFFER_SIZE,
     usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
   });
   ```

2. Update Bind Groups:
   - Match shader bind groups with TypeScript code
   - Use `pipeline.getBindGroupLayout(index)` for automatic layouts

### Data Read-Back

```typescript
// After compute shader dispatch
await rayPointsBuffer.mapAsync(GPUMapMode.READ);
const arrayBuffer = rayPointsBuffer.getMappedRange();
const rayPointsData = new Float32Array(arrayBuffer);
// Process rayPointsData
rayPointsBuffer.unmap();
```

---

## Step 4: Processing and Synthesizing the Impulse Response

### 1. Time Binning

Group `RayPoint` samples by time value using a small epsilon window:

```javascript
const EPSILON = 0.0005; // Time window for grouping
const timeStep = 0.001; // Time resolution (seconds)
const maxTime = 2.0;    // Duration (seconds)
const numSamples = Math.floor(maxTime / timeStep);
```

### 2. Sine Wave Generation

For each sample:
```javascript
const sineValue = energy * Math.sin(2 * Math.PI * frequency * t + phase);
```

### 3. Summation Process

```javascript
const impulseResponse = new Float32Array(numSamples).fill(0);

rayPoints.forEach(point => {
    const sampleIndex = Math.floor(point.time / timeStep);
    if (sampleIndex < numSamples) {
        const sineValue = point.energy * 
            Math.sin(2 * Math.PI * point.frequency * point.time + point.phase);
        impulseResponse[sampleIndex] += sineValue;
    }
});
```

---

## Step 5: Testing and Debugging

### Visual Verification
- 📊 Plot impulse response waveform
- 🔍 Inspect phase evolution
- 📉 Verify energy decay

### Test Cases
1. Single ray setup
2. Multiple ray interference
3. Edge cases:
   - No surface hits
   - Simultaneous contributions
   - Boundary conditions

### Parameter Optimization
- 📏 Sampling distance (0.1m default)
- ⏱️ Time resolution
- 🎯 Accuracy vs. Performance balance

---

## Step 6: Further Improvements

### Performance Optimization
- [ ] Profile compute shader
- [ ] Optimize CPU post-processing
- [ ] Memory usage analysis

### Quality Enhancements
- [ ] Anti-aliasing for high frequencies
- [ ] Multiple reflection support
- [ ] Validation against measured data

---

## Advanced Features

### 1. Hybrid Reverberation Model
- Early reflections via ray-tracing
- Algorithmic late reverberation
- Seamless blend between stages

### 2. Frequency-Dependent Materials

```wgsl
struct SurfaceMaterial {
    absorptionLow: f32,
    absorptionMid: f32,
    absorptionHigh: f32
};
```

Example material properties:
```javascript
const materials = {
    concrete: { 
        absorption: [0.1, 0.3, 0.6], 
        roughness: 0.25 
    },
    glass: { 
        absorption: [0.05, 0.1, 0.4], 
        roughness: 0.05 
    },
    fabric: { 
        absorption: [0.6, 0.8, 0.9], 
        roughness: 0.8 
    }
};
```

### 3. Real-time Features
- Doppler effect implementation
- Dynamic source/listener positioning
- Adaptive sampling based on scene complexity

---

## Implementation Details

### WGSL Material System
```wgsl
// Updated SurfaceMaterial struct with frequency-dependent absorption
struct SurfaceMaterial {
    absorptionLow: f32;    // 125-500 Hz range
    absorptionMid: f32;    // 500-2000 Hz range
    absorptionHigh: f32;   // 2000+ Hz range
    roughness: f32;        // Diffuse reflection factor
    @align(16) _padding: vec3<f32>;
};

// Enhanced collision handling with material properties
fn handleCollision(ray: Ray, hit: HitInfo, material: SurfaceMaterial) {
    let frequencyFactor = smoothstep(500.0, 2000.0, ray.frequency);
    let absorption = mix(
        mix(material.absorptionLow, material.absorptionMid, frequencyFactor),
        material.absorptionHigh,
        saturate((ray.frequency - 2000.0) / 8000.0)
    );
    
    ray.energy *= 1.0 - (absorption + material.roughness * 0.5);
    
    if (material.roughness > 0.01) {
        ray.direction = normalize(ray.direction + material.roughness * randomVec3());
    }
}
```

### TypeScript Material Buffer
```typescript
interface MaterialBuffer {
    absorption: [number, number, number]; // [low, mid, high] coefficients
    roughness: number;                    // Surface diffusion factor
}

// GPU buffer initialization for material properties
const materialBuffer = this.device.createBuffer({
    size: MATERIAL_BUFFER_SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
});
```

---

## Conclusion

By capturing ray sample points in your WGSL shader, reading back the detailed data, and then processing it to sum sine waves based on time, frequency, phase, and energy, you can accurately simulate an impulse response in a physically plausible way. Experiment with parameters, verify results with test cases, and refine the approach to suit the particular acoustic scenario you wish to model.

---

## Advanced Reverberation Techniques

### Frequency Band Decomposition
```typescript
// ray-renderer.ts band configuration
const FREQUENCY_BANDS = [
    { center: 125,  bandwidth: 100 },  // Low frequencies
    { center: 500,  bandwidth: 300 },  // Mid frequencies
    { center: 2000, bandwidth: 1000 }, // High frequencies
];

// GPU buffer initialization
const bandBuffer = device.createBuffer({
    size: FREQUENCY_BANDS.length * 2 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
});
```

### Parallel Band Processing
```wgsl
// raytracer.wgsl kernel modification
@group(0) @binding(2) var<storage> bands: array<vec2<f32>>;

@compute @workgroup_size(64)
fn traceRays(@builtin(global_invocation_id) gid: vec3<u32>) {
    let band_idx = gid.x % 3;
    let center = bands[band_idx].x;
    let bandwidth = bands[band_idx].y;
    
    // Generate frequency within band using golden ratio distribution
    let freq = center + bandwidth * fract(gid.x * 0.618);
    traceRay(freq);
}
```

### GPU Accelerated Late Reverberation
```wgsl
// late-reverb.wgsl compute shader
@group(0) @binding(0) var<storage, read_write> impulse: array<f32>;
@group(0) @binding(1) var<storage, read_write> fdn: array<vec4<f32>>;

const DELAYS = vec4u(47, 73, 103, 137); // Prime number delays
const DECAY = exp(-1.0 / (0.5 * 48000.0));

@compute @workgroup_size(64)
fn computeLateReverb(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    
    // Feedback Delay Network (FDN) processing
    var reverb = vec4(impulse[idx]);
    for (var i = 0u; i < 4u; i++) {
        let delay = DELAYS[i];
        let prev_idx = idx >= delay ? idx - delay : 0u;
        reverb[i] += DECAY * fdn[prev_idx][i];
    }
    
    fdn[idx] = reverb;
    impulse[idx] += dot(reverb, vec4(0.25));
}

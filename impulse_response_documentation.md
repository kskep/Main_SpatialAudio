# Impulse Response Generation Documentation

This document provides a detailed explanation of the components responsible for creating and processing impulse responses in the spatial audio simulation system. The impulse response is a critical element that captures how sound propagates and reflects in the virtual environment, enabling realistic spatial audio rendering.

## Table of Contents

1. [Overview](#overview)
2. [Key Files and Components](#key-files-and-components)
3. [Impulse Response Generation Process](#impulse-response-generation-process)
4. [Early Reflections Processing](#early-reflections-processing)
5. [Late Reverberation Processing](#late-reverberation-processing)
6. [Binaural Spatialization](#binaural-spatialization)
7. [Audio Playback and Testing](#audio-playback-and-testing)
8. [Key Algorithms and Techniques](#key-algorithms-and-techniques)

## Overview

The impulse response (IR) represents how sound propagates from a source to a listener in a specific environment. In this spatial audio simulation, the impulse response is generated by combining ray tracing data with acoustic modeling techniques. The system creates a binaural impulse response that captures both the directional characteristics of early reflections and the diffuse nature of late reverberation, resulting in a realistic spatial audio experience.

## Key Files and Components

The impulse response generation involves several key files and components:

### Main Coordinator
- **`main.ts`**: Initiates the impulse response calculation process through the `calculateIR()` method, which coordinates the ray tracing and audio processing steps.

### Ray Tracing
- **`raytracer.ts`**: Performs the acoustic simulation by tracing rays from the sound source and calculating their reflections in the room. Provides ray hit data that forms the basis for the impulse response.

### Audio Processing
- **`audio-processor.ts`**: Central component that processes ray hit data into a binaural impulse response. Coordinates early reflections, late reverberation, and final impulse response assembly.

### Specialized Components
- **`spatial-audio-processor.ts`**: Handles 3D spatialization using Head-Related Transfer Functions (HRTF) for directional audio cues.
- **`feedback-delay-network.ts`**: Implements a Feedback Delay Network (FDN) for generating natural-sounding late reverberation.
- **`diffuse-field-model.ts`**: Models the diffuse sound field that develops after early reflections, creating realistic late reverberation.

### Visualization
- **`waveform-renderer.ts`**: Visualizes the generated impulse response in both time domain (waveform) and frequency domain (spectrogram).

## Impulse Response Generation Process

The impulse response generation follows these main steps:

1. **Ray Tracing Simulation**:
   - The `calculateIR()` method in `main.ts` initiates the process by calling `rayTracer.calculateRayPaths()`.
   - The ray tracer simulates sound propagation by tracing rays from the source and calculating reflections.
   - Ray hit data is collected, including position, time, energy, and direction information.

2. **Audio Processing**:
   - Ray hit data is passed to `audioProcessor.processRayHits()`.
   - The `processRayHitsInternal()` method processes the hits into a stereo impulse response.
   - The impulse response is divided into early reflections and late reverberation.

3. **Impulse Response Assembly**:
   - Early reflections and late reverberation are combined with proper crossfading.
   - The final impulse response is stored in an AudioBuffer for convolution.
   - The impulse response is visualized using the waveform renderer.

4. **Audio Playback**:
   - The impulse response can be tested by convolving it with various test signals (sine waves, clicks, noise).
   - The convolved audio is played back through the Web Audio API.

## Early Reflections Processing

Early reflections (typically the first 100ms) are processed using the following approach:

1. **Sorting by Arrival Time**:
   ```typescript
   const sortedHits = [...hits].sort((a, b) => a.time - b.time);
   const earlyHits = sortedHits.filter(hit => hit.time < 0.1); // First 100ms
   ```

2. **HRTF Spatialization**:
   ```typescript
   const [leftGain, rightGain] = this.spatialProcessor.calculateImprovedHRTF(
       hit.position,
       this.camera.getPosition(),
       this.camera.getFront(),
       this.camera.getRight(),
       this.camera.getUp()
   );
   ```

3. **Temporal Spreading**:
   ```typescript
   const spreadSamples = Math.min(0.005 * this.sampleRate, 50); // 5ms spread
   for (let j = -spreadSamples; j <= spreadSamples; j++) {
       const idx = sampleIndex + j;
       if (idx >= 0 && idx < irLength) {
           const spread = Math.exp(-Math.abs(j) / (spreadSamples / 2));
           leftIR[idx] += amplitude * leftGain * spread;
           rightIR[idx] += amplitude * rightGain * spread;
       }
   }
   ```

4. **Energy Calculation**:
   ```typescript
   const totalEnergy = Object.values(hit.energies).reduce((sum, e) => sum + (typeof e === 'number' ? e : 0), 0);
   const amplitude = Math.sqrt(totalEnergy) * Math.exp(-hit.bounces * 0.5);
   ```

## Late Reverberation Processing

Late reverberation (after 100ms) is modeled using two complementary approaches:

### Feedback Delay Network (FDN)

The FDN in `feedback-delay-network.ts` creates a dense, natural-sounding reverb tail:

1. **Delay Lines with Prime Numbers**:
   ```typescript
   const primes = [743, 769, 797, 823, 853, 877, 907, 937, 
                   967, 997, 1021, 1049, 1087, 1117, 1151, 1181];
   
   for (let i = 0; i < numDelays; i++) {
       const delayTime = primes[i % primes.length];
       this.delays.push(new DelayLine(audioCtx, delayTime));
   }
   ```

2. **Hadamard Feedback Matrix**:
   ```typescript
   private initializeHadamardMatrix(size: number): void {
       // ... matrix generation code ...
       
       this.feedbackMatrix = [];
       const norm = Math.sqrt(size);
       for (let i = 0; i < size; i++) {
           this.feedbackMatrix.push(new Float32Array(size));
           for (let j = 0; j < size; j++) {
               this.feedbackMatrix[i][j] = matrix[i][j] / norm;
           }
       }
   }
   ```

3. **Frequency-Dependent Decay**:
   ```typescript
   public setRT60(rt60Values: {[frequency: string]: number}): void {
       const rt60_1k = rt60Values['1000'] || 1.0;
       const rt60_low = rt60Values['125'] || rt60_1k * 1.2;
       const rt60_high = rt60Values['8000'] || rt60_1k * 0.8;
       
       for (let i = 0; i < this.delays.length; i++) {
           const delayTimeInSeconds = this.delays[i].getDelayTime() / this.audioCtx.sampleRate;
           this.gains[i] = Math.pow(10, -3 * delayTimeInSeconds / rt60_1k);
           
           if (highRatio < 1.0) {
               this.filters[i].frequency.value = 8000 * highRatio;
           }
       }
   }
   ```

### Diffuse Field Model

The diffuse field model in `diffuse-field-model.ts` complements the FDN with statistical modeling:

1. **Velvet Noise Generation**:
   ```typescript
   private generateVelvetNoise(
       buffer: Float32Array, 
       rt60: number, 
       echoDensity: number, 
       meanTimeGap: number,
       diffusion: number
   ): void {
       const td = 1 / echoDensity; // Average time between impulses (seconds)
       const totalPulses = Math.floor(buffer.length / (td * this.sampleRate));
       
       for (let i = 0; i < totalPulses; i++) {
           const position = Math.floor(i * td * this.sampleRate + 
               (Math.random() - 0.5) * 2 * diffusion * td * this.sampleRate);
           
           if (position < 0 || position >= buffer.length) continue;
           
           const polarity = Math.random() > 0.5 ? 1 : -1;
           const time = position / this.sampleRate;
           const amplitude = Math.exp(-6.91 * time / rt60);
           
           buffer[position] += polarity * amplitude;
       }
   }
   ```

2. **Exponential Decay Envelope**:
   ```typescript
   private applyDecayEnvelope(buffer: Float32Array, rt60: number): void {
       for (let i = 0; i < buffer.length; i++) {
           const time = i / this.sampleRate;
           buffer[i] *= Math.exp(-6.91 * time / rt60);
       }
   }
   ```

3. **Frequency-Dependent Filtering**:
   ```typescript
   public applyFrequencyFiltering(
       impulseResponses: Map<string, Float32Array>
   ): Float32Array {
       // ... code to combine frequency bands ...
       
       for (const [freq, ir] of impulseResponses.entries()) {
           let bandGain = 1.0;
           
           switch (freq) {
               case '125': bandGain = 0.9; break;
               case '250': bandGain = 0.95; break;
               // ... other frequency bands ...
           }
           
           for (let i = 0; i < Math.min(ir.length, totalLength); i++) {
               outputIR[i] += ir[i] * bandGain;
           }
       }
   }
   ```

## Binaural Spatialization

Binaural spatialization is achieved using Head-Related Transfer Functions (HRTF) in `spatial-audio-processor.ts`:

1. **HRTF Calculation**:
   ```typescript
   public calculateImprovedHRTF(
       sourcePos: vec3,
       listenerPos: vec3,
       listenerFront: vec3,
       listenerRight: vec3,
       listenerUp: vec3
   ): [number, number] {
       // Calculate direction vector from listener to source
       const direction = vec3.create();
       vec3.subtract(direction, sourcePos, listenerPos);
       const distance = vec3.length(direction);
       vec3.normalize(direction, direction);
       
       // Calculate azimuth and elevation
       const dotRight = vec3.dot(direction, listenerRight);
       const dotFront = vec3.dot(direction, listenerFront);
       const azimuth = Math.atan2(dotRight, dotFront);
       
       const dotUp = vec3.dot(direction, listenerUp);
       const elevation = Math.asin(Math.max(-1, Math.min(1, dotUp)));
       
       // Calculate gains based on spherical head model
       let leftGain = 0.5, rightGain = 0.5;
       
       if (azimuth < 0) { // Source is to the left
           leftGain = 0.9 - 0.4 * azimuth/Math.PI;
           rightGain = 0.4 + 0.5 * (1 + azimuth/Math.PI);
       } else { // Source is to the right
           leftGain = 0.4 + 0.5 * (1 - azimuth/Math.PI);
           rightGain = 0.9 + 0.4 * azimuth/Math.PI;
       }
       
       // Apply elevation and distance effects
       const elevationFactor = 1.0 - Math.abs(elevation) / (Math.PI/2) * 0.3;
       const distanceAtten = 1.0 / Math.max(1, distance);
       
       leftGain *= elevationFactor * distanceAtten;
       rightGain *= elevationFactor * distanceAtten;
       
       return [leftGain, rightGain];
   }
   ```

2. **Advanced HRTF Processing**:
   The system also includes more advanced HRTF processing in `hrtf-processor.ts`, which can load and apply measured HRTF datasets for more accurate binaural rendering.

## Audio Playback and Testing

The impulse response can be tested using various methods in `audio-processor.ts`:

1. **Convolution Setup**:
   ```typescript
   private async setupImpulseResponseBuffer(leftIR: Float32Array, rightIR: Float32Array): Promise<void> {
       const length = leftIR.length;
       
       this.impulseResponseBuffer = this.audioCtx.createBuffer(2, length, this.audioCtx.sampleRate);
       
       const leftChannel = this.impulseResponseBuffer.getChannelData(0);
       const rightChannel = this.impulseResponseBuffer.getChannelData(1);
       
       leftChannel.set(leftIR);
       rightChannel.set(rightIR);
   }
   ```

2. **Test Sound Playback**:
   ```typescript
   public playConvolvedSound(): void {
       if (!this.impulseResponseBuffer) {
           console.warn('No impulse response buffer available');
           return;
       }
       
       const convolver = this.audioCtx.createConvolver();
       convolver.buffer = this.impulseResponseBuffer;
       
       const source = this.audioCtx.createBufferSource();
       source.buffer = clickBuffer;
       source.connect(convolver);
       convolver.connect(this.audioCtx.destination);
       source.start();
   }
   ```

## Key Algorithms and Techniques

The impulse response generation employs several sophisticated algorithms and techniques:

### 1. Image Source Method
Used for early reflections, this method creates virtual sound sources by mirroring the original source across room surfaces, providing accurate early reflection paths.

### 2. Stochastic Ray Tracing
Used for late reflections, this method traces rays in random directions from the sound source, calculating their reflections and energy loss to model complex sound propagation.

### 3. Feedback Delay Network (FDN)
A specialized reverb algorithm that creates natural-sounding reverberation using multiple delay lines with a lossless mixing matrix, providing a dense and smooth reverb tail.

### 4. Velvet Noise Algorithm
An efficient method for generating high-quality diffuse reverberation using sparse impulse patterns that sound perceptually smooth while requiring less computation than convolution reverb.

### 5. Head-Related Transfer Functions (HRTF)
Simulates how sounds are filtered by the head, pinnae, and torso to create convincing 3D audio localization cues, enabling realistic binaural rendering.

### 6. Frequency-Dependent Processing
All acoustic phenomena (absorption, scattering, air attenuation) are modeled with frequency dependence across multiple bands, matching how real materials and air affect different frequencies differently.

### 7. Temporal Spreading
Applies natural spreading to impulses to avoid artificial-sounding "clicks" in the impulse response, creating more realistic sound propagation.

### 8. Exponential Decay Envelopes
Models the natural energy decay in rooms using exponential functions based on reverberation time (RT60) values calculated from room properties and material absorption.

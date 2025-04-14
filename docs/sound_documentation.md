# Sound Module Documentation

This document provides a detailed explanation of the sound module located in the `src/sound/` directory. The module implements spatial audio processing for acoustic simulations, creating realistic 3D sound experiences based on ray tracing data.

## Table of Contents

1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [AudioProcessorModified Class (`audio-processor_modified.ts`)](#audioprocessormodified-class-audio-processor_modifiedts)
4. [FeedbackDelayNetwork Class (`feedback-delay-network.ts`)](#feedbackdelaynetwork-class-feedback-delay-networkts)
5. [DiffuseFieldModelModified Class (`diffuse-field-model_modified.ts`)](#diffusefieldmodelmodified-class-diffuse-field-model_modifiedts)
6. [Helper Spatialization Functions](#helper-spatialization-functions)
7. ~~[SpatialAudioProcessor Class (`spatial-audio-processor.ts`)]~~ (No longer used for main IR spatialization)
8. ~~[HRTFProcessor Class (`hrtf-processor.ts`)]~~ (No longer used for main IR spatialization)
8. [Key Concepts and Algorithms](#key-concepts-and-algorithms)
9. [Relationships with Other Modules](#relationships-with-other-modules)

## Overview

The sound module transforms ray tracing data into spatial audio by generating and processing impulse responses that simulate how sound propagates in a 3D environment. It handles both early reflections and late reverberation, applies simplified spatialization (Gain + ITD) for directional cues, and provides tools for audio playback and testing.

The system accounts for:
- Frequency-dependent material absorption
- Room acoustics (via late reverb model)
- Binaural spatialization using Gain + Interaural Time Delay (ITD)
- Early reflections vs. diffuse late reverberation
- Real-time audio processing and convolution

## File Structure

The sound module consists of the following files:

- `audio-processor_modified.ts`: Main class that orchestrates the audio processing pipeline using simplified spatialization.
- `feedback-delay-network.ts`: Implements a feedback delay network (currently unused in main IR path but available).
- `diffuse-field-model_modified.ts`: Models diffuse sound fields for late reverberation with improved stereo decorrelation.
- Helper functions in `audio-processor_modified.ts` (`calculateBalancedSpatialGains`, `calculateITDsamples`): Provide the Gain+ITD spatialization logic.
- ~~`spatial-audio-processor.ts`~~: No longer used for main IR spatialization.
- ~~`hrtf-processor.ts`~~: No longer used for main IR spatialization.

## AudioProcessorModified Class (`audio-processor_modified.ts`)

### Purpose
Serves as the main entry point for the audio processing pipeline, coordinating the generation of impulse responses from ray tracing data and handling audio playback.

### Key Components

#### Properties
- `audioCtx`: Web Audio API context for audio processing
- `sampleRate`: Sample rate for audio processing
- `impulseResponseBuffer`: AudioBuffer containing the generated impulse response
- `room`: Reference to the Room object for acoustic properties
- `camera`: Reference to the Camera object for listener position
- `fdn`: Feedback Delay Network instance (potentially unused in main IR path).
- `diffuseFieldModel`: Instance of `DiffuseFieldModelModified` for late reverberation.
- Helper functions (`calculateBalancedSpatialGains`, `calculateITDsamples`) defined within the file implement the spatialization logic.

#### Important Methods

- **Constructor**: Initializes the audio processing components
  ```typescript
  constructor(device: GPUDevice, room: Room, camera: Camera, sampleRate: number = 44100)
  ```

- **processRayHits**: Processes ray hit data to generate an impulse response
  ```typescript
  async processRayHits(
      rayHits: RayHit[],
      maxTime: number = 0.5,
      params = { /* configuration parameters */ }
  ): Promise<void>
  ```
  This method:
  1. Validates and filters ray hit data
  2. Calls `processRayHitsInternal` to generate left and right channel IRs
  3. Sets up the final impulse response buffer

- **processRayHitsInternal**: Internal method that processes ray hits into stereo impulse responses
  ```typescript
  private processRayHitsInternal(rayHits: RayHit[]): [Float32Array, Float32Array]
  ```
  This method:
  1. Sorts ray hits by arrival time
  2. Processes early reflections (first 100ms) using a Gain + ITD model with temporal spreading.
  3. Processes late reverberation using the `DiffuseFieldModelModified`.
  4. Combines early and late parts with proper crossfading

- **setupImpulseResponseBuffer**: Creates an AudioBuffer from processed impulse response data
  ```typescript
  private async setupImpulseResponseBuffer(leftIR: Float32Array, rightIR: Float32Array): Promise<void>
  ```

- **Debug and Test Methods**:
  - `debugPlaySineWave()`: Plays a simple sine wave for testing
  - `playConvolvedSineWave()`: Plays a sine wave convolved with the impulse response
  - `playConvolvedClick()`: Plays a generated click sound convolved with the impulse response.
  - `playNoiseWithIR()`: Plays white noise convolved with the impulse response

## FeedbackDelayNetwork Class (`feedback-delay-network.ts`)

*(Content moved up, original section on SpatialAudioProcessor removed/commented)*

### Purpose
Implements a Feedback Delay Network (FDN) for generating high-quality, natural-sounding late reverberation. (Note: Currently initialized but not used in the primary IR generation path of `AudioProcessorModified`).

### Key Components

#### Properties
- `delays`: Array of DelayLine objects for creating echo patterns
- `feedbackMatrix`: Matrix controlling how delay lines feed back into each other
- `gains`: Gain values for each delay line
- `filters`: Biquad filters for frequency-dependent decay
- `dryGain` and `wetGain`: Mix levels for dry (direct) and wet (processed) signals

#### Important Methods

- **Constructor**: Sets up the FDN with specified number of delay lines
  ```typescript
  constructor(audioCtx: AudioContext, numDelays: number = 16)
  ```
  Uses prime-number delay times to avoid modal resonances and initializes a Hadamard feedback matrix.

- **initializeHadamardMatrix**: Creates a Hadamard matrix for lossless mixing
  ```typescript
  private initializeHadamardMatrix(size: number): void
  ```
  Generates a matrix that ensures energy preservation and diffuse mixing.

- **setRT60**: Sets reverberation time (RT60) for different frequency bands
  ```typescript
  public setRT60(rt60Values: {[frequency: string]: number}): void
  ```
  Adjusts gain and filter parameters to achieve the desired decay times.

- **process**: Processes a mono audio buffer through the FDN
  ```typescript
  public process(input: Float32Array): Float32Array
  ```
  Applies the FDN algorithm to create a reverberant output.

- **processStereo**: Processes stereo audio through the FDN
  ```typescript
  public processStereo(inputLeft: Float32Array, inputRight: Float32Array): [Float32Array, Float32Array]
  ```
  Creates decorrelated stereo reverberation for more realistic spatial impression.

#### DelayLine Class
An internal helper class that implements a simple delay line with read and write operations.

## DiffuseFieldModelModified Class (`diffuse-field-model_modified.ts`)

### Purpose
Models the diffuse sound field that develops in a room after early reflections, creating realistic late reverberation using statistical methods (Velvet Noise) and improved stereo decorrelation.

### Key Components

#### Properties
- `sampleRate`: Sample rate for audio processing
- `roomVolume`: Volume of the room in cubic meters
- `surfaceArea`: Total surface area of the room
- `meanAbsorption`: Average absorption coefficients across frequency bands
- `diffusionCoefficients`: Diffusion coefficients for different frequencies

#### Important Methods

- **generateDiffuseField**: Creates a diffuse field reverb tail for each frequency band using Velvet Noise.
  ```typescript
  public generateDiffuseField(
      duration: number,
      rt60Values: { [freq: string]: number }
  ): Map<string, Float32Array>
  ```

- **generateVelvetNoise**: Creates efficient sparse FIR for diffuse reverb modeling, including amplitude decay based on RT60.
  ```typescript
  private generateVelvetNoise(
      buffer: Float32Array,
      rt60: number,
      echoDensity: number,
      meanTimeGap: number,
      diffusion: number
  ): void
  ```

- **applyFrequencyFiltering**: Combines the frequency band impulse responses into a single mono IR, applying frequency-dependent gains.
  ```typescript
  public applyFrequencyFiltering(
      impulseResponses: Map<string, Float32Array>
  ): Float32Array
  ```

- **processLateReverberation**: Generates the final stereo late reverberation tail. Calculates RT60, generates the diffuse field per band, combines bands, and applies stereo decorrelation (using small delays and filtering).
  ```typescript
  public processLateReverberation(
      lateHits: any[],
      camera: any,
      roomConfig: any,
      sampleRate: number
  ): [Float32Array, Float32Array]
  ```

- **calculateRT60Values**: Calculates frequency-dependent RT60 values based on room properties using Sabine's formula with empirical adjustments.
  ```typescript
  private calculateRT60Values(lateHits: any[], roomConfig: any): { [freq: string]: number }
  ```

## Helper Spatialization Functions

Defined within `audio-processor_modified.ts`.

### `calculateBalancedSpatialGains`
Calculates left/right gain factors based on azimuth, elevation, and distance using a balanced panning approach (sine-based modulation). Includes distance attenuation and front-back reduction.

### `calculateITDsamples`
Calculates the Interaural Time Delay (ITD) in samples based on azimuth using Woodworth's formula approximation, clamped to a maximum value.

## ~~HRTFProcessor Class (`hrtf-processor.ts`)~~

*(This class exists but is no longer used in the primary IR generation path of `AudioProcessorModified`. It remains available for potential future use with measured HRTF datasets.)*

### Purpose
Provides advanced HRTF processing capabilities for high-quality binaural audio rendering using external datasets.

### Key Components

#### HRTFDataset Interface
```typescript
interface HRTFDataset { /* ... */ }
```

#### Properties
- `hrtfDatasets`: Map of available HRTF datasets
- `audioCtx`: Web Audio API context
- `currentDataset`: Currently active HRTF dataset

#### Important Methods

- **loadHRTFDataset**: Loads an HRTF dataset from a URL (JSON or potentially SOFA).
- **getHRTFForDirection**: Gets the closest HRTF filter for a specific direction.
- **applyHRTFToBuffer**: Convolves an input buffer with an HRTF filter.
- **applyMultiDirectionalHRTF**: Handles time-varying HRTF for moving sources.

## Key Concepts and Algorithms

### Impulse Response Generation
The system generates binaural room impulse responses (BRIRs) in two stages:
1. **Early Reflections**: Processed using ray tracing data and a simplified spatialization model.
   - Each ray hit is spatialized using calculated Interaural Level Differences (ILD) and Interaural Time Differences (ITD) based on arrival direction and distance.
   - Energy is scaled based on ray energy and bounce count.
   - Temporal spreading is applied to smooth the impulses.
   - Typically includes direct sound and early reflections (first 100ms).

2. **Late Reverberation**: Modeled using statistical methods via `DiffuseFieldModelModified`.
   - Uses the Velvet Noise algorithm to generate a diffuse reverb tail for multiple frequency bands based on room properties (volume, absorption) and calculated RT60 values.
   - Applies frequency-dependent filtering/gain when combining bands.
   - Creates a stereo output using decorrelation techniques (small delays/filtering).

### Simplified Binaural Spatialization (Gain + ITD)
Used for early reflections, this model applies:
1. **Interaural Level Difference (ILD):** Gain differences between left and right channels calculated based on azimuth, elevation, and distance (`calculateBalancedSpatialGains`).
2. **Interaural Time Difference (ITD):** Small time delays applied to left or right channel based on azimuth (`calculateITDsamples`), using linear interpolation for fractional sample delays.
3. **Temporal Spreading:** Each reflection impulse is spread over a short duration (e.g., 20ms) with an exponential decay to avoid sharp clicks.

### ~~HRTF Processing~~
*(Full HRTF processing using external datasets is available via `HRTFProcessor` but not currently enabled in the main IR path).*

### Feedback Delay Network
*(Available via `FeedbackDelayNetwork` class but not currently enabled in the main IR path).*

### Velvet Noise Algorithm
Used in `DiffuseFieldModelModified` for generating the late reverberation tail efficiently.

## Relationships with Other Modules

The sound module interacts with several other components:

- **Raytracer Module**: Provides ray hit data used to generate impulse responses
  - Uses `RayHit` interface to receive reflection data
  - Processes frequency-dependent energy information from ray tracing

- **Room Module**: Provides room geometry and material properties
  - Uses room dimensions to calculate acoustic parameters for late reverb
  - Uses material absorption coefficients for frequency-dependent decay (RT60 calculation)

- **Camera Module**: Provides listener position and orientation
  - Uses camera position as listener position for spatialization calculations
  - Uses camera orientation to determine relative source directions

- **Visualization Module**: Displays audio-related visualizations
  - Can visualize impulse responses and frequency content
  - Provides feedback on audio processing results

- **Web Audio API**: Handles audio playback and processing
  - Uses AudioContext, AudioBuffer, and ConvolverNode
  - Implements real-time audio processing and convolution
## FeedbackDelayNetwork Class (`feedback-delay-network.ts`)

### Purpose
Implements a Feedback Delay Network (FDN) for generating high-quality, natural-sounding late reverberation.

### Key Components

#### Properties
- `delays`: Array of DelayLine objects for creating echo patterns
- `feedbackMatrix`: Matrix controlling how delay lines feed back into each other
- `gains`: Gain values for each delay line
- `filters`: Biquad filters for frequency-dependent decay
- `dryGain` and `wetGain`: Mix levels for dry (direct) and wet (processed) signals

#### Important Methods

- **Constructor**: Sets up the FDN with specified number of delay lines
  ```typescript
  constructor(audioCtx: AudioContext, numDelays: number = 16)
  ```
  Uses prime-number delay times to avoid modal resonances and initializes a Hadamard feedback matrix.

- **initializeHadamardMatrix**: Creates a Hadamard matrix for lossless mixing
  ```typescript
  private initializeHadamardMatrix(size: number): void
  ```
  Generates a matrix that ensures energy preservation and diffuse mixing.

- **setRT60**: Sets reverberation time (RT60) for different frequency bands
  ```typescript
  public setRT60(rt60Values: {[frequency: string]: number}): void
  ```
  Adjusts gain and filter parameters to achieve the desired decay times.

- **process**: Processes a mono audio buffer through the FDN
  ```typescript
  public process(input: Float32Array): Float32Array
  ```
  Applies the FDN algorithm to create a reverberant output.

- **processStereo**: Processes stereo audio through the FDN
  ```typescript
  public processStereo(inputLeft: Float32Array, inputRight: Float32Array): [Float32Array, Float32Array]
  ```
  Creates decorrelated stereo reverberation for more realistic spatial impression.

#### DelayLine Class
An internal helper class that implements a simple delay line with read and write operations.

## DiffuseFieldModel Class (`diffuse-field-model.ts`)

### Purpose
Models the diffuse sound field that develops in a room after early reflections, creating realistic late reverberation.

### Key Components

#### Properties
- `sampleRate`: Sample rate for audio processing
- `roomVolume`: Volume of the room in cubic meters
- `surfaceArea`: Total surface area of the room
- `meanAbsorption`: Average absorption coefficients across frequency bands
- `diffusionCoefficients`: Diffusion coefficients for different frequencies

#### Important Methods

- **generateDiffuseField**: Creates a diffuse field reverb tail for each frequency band
  ```typescript
  public generateDiffuseField(
      duration: number, 
      rt60Values: { [freq: string]: number }
  ): Map<string, Float32Array>
  ```
  Generates frequency-dependent reverb tails based on RT60 values.

- **generateVelvetNoise**: Creates efficient sparse FIR for diffuse reverb modeling
  ```typescript
  private generateVelvetNoise(
      buffer: Float32Array, 
      rt60: number, 
      echoDensity: number, 
      meanTimeGap: number,
      diffusion: number
  ): void
  ```
  Implements the velvet noise algorithm for efficient and natural-sounding reverberation.

- **applyDecayEnvelope**: Applies exponential decay envelope to the reverb tail
  ```typescript
  private applyDecayEnvelope(buffer: Float32Array, rt60: number): void
  ```

- **applyFrequencyFiltering**: Applies shelving filters to model frequency-dependent decay
  ```typescript
  public applyFrequencyFiltering(
      impulseResponses: Map<string, Float32Array>
  ): Float32Array
  ```

- **processLateReverberation**: Processes late reverberation for a set of ray hits
  ```typescript
  public processLateReverberation(
      lateHits: any[],
      camera: any,
      roomConfig: any,
      sampleRate: number
  ): [Float32Array, Float32Array]
  ```
  Creates stereo late reverberation based on room properties and ray hit statistics.

- **combineWithEarlyReflections**: Combines early reflections with diffuse field
  ```typescript
  public combineWithEarlyReflections(
      earlyReflections: Float32Array,
      diffuseField: Float32Array,
      crossoverTime: number
  ): Float32Array
  ```
  Applies crossfading between early reflections and late reverberation.

## HRTFProcessor Class (`hrtf-processor.ts`)

### Purpose
Provides advanced HRTF processing capabilities for high-quality binaural audio rendering.

### Key Components

#### HRTFDataset Interface
```typescript
interface HRTFDataset {
    name: string;
    elevations: number[];   // Available elevation angles (degrees)
    azimuths: number[];     // Available azimuth angles (degrees)
    filters: Map<string, AudioBuffer>; // HRTF filters indexed by "elevation_azimuth"
}
```

#### Properties
- `hrtfDatasets`: Map of available HRTF datasets
- `audioCtx`: Web Audio API context
- `currentDataset`: Currently active HRTF dataset

#### Important Methods

- **loadHRTFDataset**: Loads an HRTF dataset from a URL
  ```typescript
  async loadHRTFDataset(url: string, name: string): Promise<void>
  ```
  Loads and parses HRTF data from a JSON or SOFA format file.

- **getHRTFForDirection**: Gets the HRTF filter for a specific direction
  ```typescript
  getHRTFForDirection(azimuthDegrees: number, elevationDegrees: number): AudioBuffer | null
  ```
  Finds the closest available HRTF filter for the specified direction.

- **applyHRTFToBuffer**: Applies HRTF filtering to an audio buffer
  ```typescript
  applyHRTFToBuffer(
      inputBuffer: AudioBuffer, 
      azimuthDegrees: number, 
      elevationDegrees: number
  ): AudioBuffer
  ```
  Convolves the input audio with the appropriate HRTF filter.

- **applyMultiDirectionalHRTF**: Applies HRTF for moving sound sources
  ```typescript
  applyMultiDirectionalHRTF(
      inputBuffer: AudioBuffer,
      directionData: { azimuth: number, elevation: number, time: number }[]
  ): AudioBuffer
  ```
  Handles time-varying HRTF processing for moving sound sources.

## Key Concepts and Algorithms

### Impulse Response Generation
The system generates binaural room impulse responses (BRIRs) in two stages:
1. **Early Reflections**: Processed using ray tracing data and HRTF
   - Each ray hit is spatialized using HRTF based on its arrival direction
   - Energy, time, and phase information are preserved
   - Typically includes direct sound and first-order reflections (first 100ms)

2. **Late Reverberation**: Modeled using statistical methods
   - Feedback Delay Network creates a dense, natural-sounding reverb tail
   - Diffuse Field Model ensures proper energy decay and frequency response
   - Parameters derived from room properties and ray hit statistics

### HRTF Processing
Head-Related Transfer Functions simulate how sounds are filtered by the head, pinnae, and torso:
1. Simple HRTF implementation uses basic head shadow and interaural time/level differences
2. Advanced implementation can load measured HRTF datasets in SOFA format
3. Interpolation between measurements provides smooth transitions for moving sources

### Feedback Delay Network
A specialized reverb algorithm that creates natural-sounding reverberation:
1. Multiple delay lines with prime-number lengths avoid modal resonances
2. Hadamard matrix provides lossless mixing between delay lines
3. Frequency-dependent feedback gains model air and material absorption
4. Decorrelation techniques create convincing stereo imaging

### Velvet Noise Algorithm
An efficient method for generating high-quality diffuse reverberation:
1. Creates sparse impulse patterns that sound perceptually smooth
2. Requires significantly less computation than convolution reverb
3. Parameters controlled by echo density, RT60, and diffusion values

## Relationships with Other Modules

The sound module interacts with several other components:

- **Raytracer Module**: Provides ray hit data used to generate impulse responses
  - Uses `RayHit` interface to receive reflection data
  - Processes frequency-dependent energy information from ray tracing

- **Room Module**: Provides room geometry and material properties
  - Uses room dimensions to calculate acoustic parameters
  - Uses material absorption coefficients for frequency-dependent decay

- **Camera Module**: Provides listener position and orientation
  - Uses camera position as listener position for HRTF calculations
  - Uses camera orientation to determine relative source directions

- **Visualization Module**: Displays audio-related visualizations
  - Can visualize impulse responses and frequency content
  - Provides feedback on audio processing results

- **Web Audio API**: Handles audio playback and processing
  - Uses AudioContext, AudioBuffer, and ConvolverNode
  - Implements real-time audio processing and convolution

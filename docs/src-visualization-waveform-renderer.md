# Waveform Renderer Documentation

## File: `src/visualization/waveform-renderer.ts`

This file implements visualization tools for audio data, including waveform and frequency spectrum (FFT) displays.

## Class: `FFT`

A helper class that implements the Fast Fourier Transform algorithm for converting time-domain audio data to frequency-domain data.

### Properties

- `size: number` - The size of the FFT (number of samples)
- `real: Float32Array` - Array for the real part of the complex numbers
- `imag: Float32Array` - Array for the imaginary part of the complex numbers
- `cosTable: Float32Array` - Precomputed cosine values for optimization
- `sinTable: Float32Array` - Precomputed sine values for optimization

### Methods

#### `constructor(size: number)`
- Initializes the FFT with the specified size
- Allocates arrays for real and imaginary components
- Precomputes cosine and sine tables for efficiency

#### `transform(input: Float32Array): { magnitudes: Float32Array, phases: Float32Array }`
- Performs the Fast Fourier Transform on the input data
- Implements the Cooley-Tukey FFT algorithm
- Returns an object containing:
  - `magnitudes` - Array of frequency magnitudes
  - `phases` - Array of phase angles

#### `private reverseBits(x: number, n: number): number`
- Helper method for the FFT algorithm
- Reverses the bits of a number for the butterfly operations

## Class: `WaveformRenderer`

The main class for visualizing audio data on HTML canvases.

### Properties

- `canvas: HTMLCanvasElement` - The canvas for drawing the waveform
- `ctx: CanvasRenderingContext2D` - The 2D rendering context for the waveform canvas
- `fftCanvas: HTMLCanvasElement` - The canvas for drawing the frequency spectrum
- `fftCtx: CanvasRenderingContext2D` - The 2D rendering context for the FFT canvas
- `analyser: AnalyserNode` - Web Audio API analyzer node for real-time analysis
- `audioCtx: AudioContext` - Web Audio API context

### Methods

#### `constructor(canvas: HTMLCanvasElement)`
- Initializes the renderer with the provided canvas
- Creates a second canvas for the FFT display
- Sets up the Web Audio API context and analyzer
- Configures canvas styles and positions
- Sets up a resize event listener

#### `private resize(): void`
- Resizes both canvases to match their client dimensions
- Called on window resize events

#### `public async drawWaveformWithFFT(stereoData: Float32Array): Promise<void>`
- Draws both the waveform and frequency spectrum for the provided audio data
- Extracts the left channel for FFT analysis
- Applies windowing to the data for better frequency resolution
- Computes the FFT and draws the frequency spectrum
- Includes frequency labels and grid lines

#### `public drawWaveform(stereoData: Float32Array): void`
- Draws the time-domain waveform for the provided audio data
- Supports stereo data (left and right channels)
- Scales the waveform to fit the canvas
- Draws time markers and grid lines
- Uses different colors for left and right channels

## Relationships

This module is imported by:
- `main.ts` - Creates a waveform renderer for visualizing the impulse response
- `audio-processor.ts` - Uses the waveform renderer to visualize processed audio data

The WaveformRenderer provides visual feedback about the audio characteristics of the spatial simulation, helping users understand both the time-domain impulse response and its frequency content. 
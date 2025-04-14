/**
 * WaveformRenderer
 *
 * This module draws a waveform on a given canvas element.
 * It accepts a Float32Array representing the impulse response data
 * (assumed to be in the range of ~[-1, 1]) and scales it to fill the canvas.
 */

// FFT helper class
class FFT {
    private size: number;
    private real: Float32Array;
    private imag: Float32Array;
    private cosTable: Float32Array;
    private sinTable: Float32Array;

    constructor(size: number) {
        this.size = size;
        this.real = new Float32Array(size);
        this.imag = new Float32Array(size);
        this.cosTable = new Float32Array(size);
        this.sinTable = new Float32Array(size);

        // Precompute tables
        for (let i = 0; i < size; i++) {
            const angle = -2 * Math.PI * i / size;
            this.cosTable[i] = Math.cos(angle);
            this.sinTable[i] = Math.sin(angle);
        }
    }

    transform(input: Float32Array): { magnitudes: Float32Array, phases: Float32Array } {
        // Copy input to real array, clear imaginary
        this.real.set(input);
        this.imag.fill(0);

        // Perform FFT
        const n = this.size;
        for (let i = 0; i < n; i++) {
            if (i < this.reverseBits(i, n)) {
                // Swap elements
                [this.real[i], this.real[this.reverseBits(i, n)]] =
                [this.real[this.reverseBits(i, n)], this.real[i]];
            }
        }

        for (let blockSize = 2; blockSize <= n; blockSize *= 2) {
            const halfBlock = blockSize / 2;

            for (let blockStart = 0; blockStart < n; blockStart += blockSize) {
                for (let i = 0; i < halfBlock; i++) {
                    const angle = -2 * Math.PI * i / blockSize;
                    const cos = Math.cos(angle);
                    const sin = Math.sin(angle);

                    const a = blockStart + i;
                    const b = blockStart + i + halfBlock;

                    const tr = this.real[b] * cos - this.imag[b] * sin;
                    const ti = this.real[b] * sin + this.imag[b] * cos;

                    this.real[b] = this.real[a] - tr;
                    this.imag[b] = this.imag[a] - ti;
                    this.real[a] = this.real[a] + tr;
                    this.imag[a] = this.imag[a] + ti;
                }
            }
        }

        // Calculate magnitudes and phases
        const magnitudes = new Float32Array(n / 2);
        const phases = new Float32Array(n / 2);
        for (let i = 0; i < n / 2; i++) {
            magnitudes[i] = Math.sqrt(this.real[i] * this.real[i] + this.imag[i] * this.imag[i]);
            phases[i] = Math.atan2(this.imag[i], this.real[i]);
        }

        return { magnitudes, phases };
    }

    private reverseBits(x: number, n: number): number {
        let result = 0;
        let power = Math.log2(n);

        for (let i = 0; i < power; i++) {
            result = (result << 1) + (x & 1);
            x >>= 1;
        }

        return result;
    }
}

export class WaveformRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private fftCanvas: HTMLCanvasElement;
  private fftCtx: CanvasRenderingContext2D;
  private analyser: AnalyserNode;
  private audioCtx: AudioContext;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const context = canvas.getContext('2d');
    if (!context) {
      throw new Error('Unable to get 2D context from canvas.');
    }
    this.ctx = context;

    // Create FFT canvas
    this.fftCanvas = document.createElement('canvas');
    this.fftCanvas.id = "fft-canvas";
    this.fftCanvas.style.position = "fixed";
    this.fftCanvas.style.bottom = "150px"; // Position above waveform
    this.fftCanvas.style.left = "0";
    this.fftCanvas.style.width = "100%";
    this.fftCanvas.style.height = "150px";
    this.fftCanvas.style.backgroundColor = "rgba(0, 0, 0, 0.5)";
    this.fftCanvas.style.zIndex = "1000"; // Ensure it's visible on top
    document.body.appendChild(this.fftCanvas);

    // Set explicit dimensions for main canvas
    this.canvas.style.width = "100%";
    this.canvas.style.height = "150px";
    this.canvas.style.backgroundColor = "rgba(0, 0, 0, 0.5)";
    this.canvas.style.position = "fixed";
    this.canvas.style.bottom = "0";
    this.canvas.style.left = "0";
    this.canvas.style.zIndex = "1000"; // Ensure it's visible on top

    const fftContext = this.fftCanvas.getContext('2d');
    if (!fftContext) {
      throw new Error('Unable to get 2D context from FFT canvas.');
    }
    this.fftCtx = fftContext;

    // Initialize Web Audio
    this.audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
    this.analyser = this.audioCtx.createAnalyser();
    this.analyser.fftSize = 2048;
    this.analyser.smoothingTimeConstant = 0.85;

    window.addEventListener('resize', () => this.resize());
    this.resize();
  }

  private resize(): void {
    // Get the window dimensions
    const width = window.innerWidth;
    const height = 150; // Fixed height for both canvases

    // Set dimensions for both canvases
    this.canvas.width = width;
    this.canvas.height = height;
    this.fftCanvas.width = width;
    this.fftCanvas.height = height;

    // Clear both canvases
    this.ctx.clearRect(0, 0, width, height);
    this.fftCtx.clearRect(0, 0, width, height);
  }

  public async drawWaveformWithFFT(stereoData: Float32Array): Promise<void> {
    this.drawWaveform(stereoData);

    // Extract left channel for FFT analysis
    const leftChannel = new Float32Array(stereoData.length / 2);
    for (let i = 0; i < stereoData.length / 2; i++) {
      leftChannel[i] = stereoData[i * 2];
    }

    const spectrogramWidth = this.fftCanvas.width;
    const spectrogramHeight = this.fftCanvas.height;
    const imageData = this.fftCtx.createImageData(spectrogramWidth, spectrogramHeight);

    // FFT parameters
    const windowSize = Math.min(1024, Math.pow(2, Math.floor(Math.log2(leftChannel.length))));
    const hopSize = windowSize / 8; // Smaller hop size for more overlap
    const numWindows = Math.max(1, Math.floor((leftChannel.length - windowSize) / hopSize) + 1);

    // Create FFT processor
    const fft = new FFT(windowSize);
    const window = new Float32Array(windowSize);

    // Validate window configuration
    if (numWindows <= 0 || windowSize <= 0) {
        console.error('Invalid FFT configuration:', { dataLength: leftChannel.length, windowSize, numWindows });
        return;
    }

    // Hann window
    for (let i = 0; i < windowSize; i++) {
        window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / windowSize));
    }

    // Process each time window
    const spectrogramData = new Array(numWindows);

    // First pass: compute all FFTs
    for (let windowIndex = 0; windowIndex < numWindows; windowIndex++) {
        const startIndex = Math.floor(windowIndex * hopSize);
        const segment = new Float32Array(windowSize);

        // Apply window
        for (let i = 0; i < windowSize; i++) {
            segment[i] = leftChannel[startIndex + i] * window[i];
        }

        // Perform FFT
        spectrogramData[windowIndex] = fft.transform(segment).magnitudes;
    }

    // Second pass: draw with interpolation
    for (let x = 0; x < spectrogramWidth; x++) {
        // Map x coordinate back to window index with bounds checking
        const windowPos = Math.max(0, Math.min((x / spectrogramWidth) * (numWindows - 1), numWindows - 1));
        const windowIndex1 = Math.floor(windowPos);
        const windowIndex2 = Math.min(windowIndex1 + 1, numWindows - 1);
        const windowAlpha = windowPos - windowIndex1;

        for (let y = 0; y < spectrogramHeight; y++) {
            // Use logarithmic frequency mapping with bounds checking
            const logFreq = Math.min(Math.exp(Math.log(1 + windowSize/2) * (y / spectrogramHeight)) - 1, windowSize/2 - 1);
            const freqIndex = Math.min(Math.floor(logFreq), windowSize/2 - 1);

            // Interpolate between frequency bins
            const alpha = logFreq - freqIndex;
            const nextFreqIndex = Math.min(freqIndex + 1, windowSize/2 - 1);

            // Get magnitudes from both windows with bounds checking
            const getMagnitude = (windowIndex: number) => {
                const magnitudes = spectrogramData[windowIndex];
                if (!magnitudes) return 0;

                const mag1 = freqIndex < magnitudes.length ? magnitudes[freqIndex] : 0;
                const mag2 = nextFreqIndex < magnitudes.length ? magnitudes[nextFreqIndex] : 0;
                return mag1 * (1 - alpha) + mag2 * alpha;
            };

            // Interpolate between windows
            const magnitude1 = getMagnitude(windowIndex1);
            const magnitude2 = getMagnitude(windowIndex2);
            const magnitude = magnitude1 * (1 - windowAlpha) + magnitude2 * windowAlpha;

            // Convert magnitude to dB and normalize with adjusted range
            const db = 20 * Math.log10(magnitude + 1e-6);
            const normalizedDb = (db + 120) / 120; // Wider dynamic range
            const value = Math.max(0, Math.min(1, normalizedDb));

            // Enhanced color gradient for better visibility
            const r = Math.floor(value * 255);
            const g = Math.floor((1 - Math.pow(1 - value, 2)) * 255);
            const b = Math.floor((1 - value) * 255);

            const pixelIndex = (y * spectrogramWidth + x) * 4;
            imageData.data[pixelIndex] = r;
            imageData.data[pixelIndex + 1] = g;
            imageData.data[pixelIndex + 2] = b;
            imageData.data[pixelIndex + 3] = 255;
        }
    }

    // Draw the spectrogram
    this.fftCtx.putImageData(imageData, 0, 0);

    // Draw frequency axis (logarithmic scale)
    this.fftCtx.fillStyle = '#ffffff';
    this.fftCtx.font = '12px Arial';
    this.fftCtx.textAlign = 'left';

    const freqLabels = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];
    freqLabels.forEach(freq => {
        const logFreq = Math.log2(freq / 20) / Math.log2(20000 / 20);
        const y = spectrogramHeight * (1 - Math.sqrt(logFreq));
        this.fftCtx.fillText(`${freq >= 1000 ? (freq/1000) + 'k' : freq}Hz`, 5, y);
    });

    // Draw time axis
    const timeScale = leftChannel.length / this.audioCtx.sampleRate;
    const timeLabels = [0, timeScale/4, timeScale/2, timeScale*3/4, timeScale];
    timeLabels.forEach(time => {
        const x = (time / timeScale) * spectrogramWidth;
        this.fftCtx.fillText(`${time.toFixed(3)}s`, x, spectrogramHeight - 5);
    });
  }

  /**
   * Draws the waveform using the provided stereo impulse response data.
   *
   * @param stereoData - The stereo impulse response data as an interleaved Float32Array.
   */
  public drawWaveform(stereoData: Float32Array): void {
    if (!stereoData || stereoData.length === 0) {
        console.warn('No data to draw waveform');
        return;
    }

    // Split stereo channels and calculate max values
    const leftChannel = new Float32Array(stereoData.length / 2);
    const rightChannel = new Float32Array(stereoData.length / 2);
    for (let i = 0; i < stereoData.length / 2; i++) {
        leftChannel[i] = stereoData[i * 2];
        rightChannel[i] = stereoData[i * 2 + 1];
    }

    // Calculate max value from both channels - FIXED VERSION
    // Instead of using Math.max with spread operator which can cause a stack overflow,
    // manually loop through the values to find the maximum
    let maxValue = 0;
    for (let i = 0; i < leftChannel.length; i++) {
        const leftAbs = Math.abs(leftChannel[i]);
        if (!isNaN(leftAbs) && isFinite(leftAbs) && leftAbs > maxValue) {
            maxValue = leftAbs;
        }
        
        const rightAbs = Math.abs(rightChannel[i]);
        if (!isNaN(rightAbs) && isFinite(rightAbs) && rightAbs > maxValue) {
            maxValue = rightAbs;
        }
    }

    console.log(`Drawing waveform with ${stereoData.length / 2} samples per channel, max value: ${maxValue}`);

    const { width, height } = this.canvas;
    const ctx = this.ctx;

    // Clear the canvas
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.fillRect(0, 0, width, height);

    // Use the validated max value for scaling
    const scale = maxValue > 0 ? 1 / maxValue : 1;

    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;

    // Vertical grid lines (time divisions)
    for (let x = 0; x < width; x += width / 10) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Horizontal grid lines (amplitude divisions)
    for (let y = 0; y < height; y += height / 8) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw zero line
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Draw waveform
    ctx.beginPath();
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;

    // Draw left channel (green)
    ctx.strokeStyle = 'rgba(0, 255, 0, 0.8)';
    const stepLeft = Math.max(1, Math.floor(leftChannel.length / width));
    let lastYLeft = 0;

    ctx.beginPath();
    for (let x = 0; x < width; x++) {
      const index = Math.min(x * stepLeft, leftChannel.length - 1);
      const value = leftChannel[index] * scale;
      const y = (0.4 - value * 0.3) * height; // Draw in upper half

      if (x === 0) {
        ctx.moveTo(x, y);
      } else {
        const cpx = (x + (x - 1)) / 2;
        ctx.quadraticCurveTo(cpx, lastYLeft, x, y);
      }
      lastYLeft = y;
    }
    ctx.stroke();

    // Draw right channel (blue)
    ctx.strokeStyle = 'rgba(0, 128, 255, 0.8)';
    const stepRight = Math.max(1, Math.floor(rightChannel.length / width));
    let lastYRight = 0;

    ctx.beginPath();
    for (let x = 0; x < width; x++) {
      const index = Math.min(x * stepRight, rightChannel.length - 1);
      const value = rightChannel[index] * scale;
      const y = (0.6 - value * 0.3) * height; // Draw in lower half

      if (x === 0) {
        ctx.moveTo(x, y);
      } else {
        const cpx = (x + (x - 1)) / 2;
        ctx.quadraticCurveTo(cpx, lastYRight, x, y);
      }
      lastYRight = y;
    }
    ctx.stroke();

    // Draw time markers
    ctx.fillStyle = '#ffffff';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    for (let i = 0; i <= 10; i++) {
      const x = (width * i) / 10;
      const time = (i * (stereoData.length / 2) * 1000 / this.canvas.width).toFixed(0);
      ctx.fillText(`${time}ms`, x, height - 5);
    }
  }
}
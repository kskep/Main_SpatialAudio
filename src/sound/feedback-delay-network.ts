import { FrequencyBands } from '../raytracer/ray';

// A Feedback Delay Network implementation for smoother late reverberation
export class FeedbackDelayNetwork {
    private delays: DelayLine[];
    private feedbackMatrix: Float32Array[];
    private audioCtx: AudioContext;
    private gains: Float32Array;
    // private filters: BiquadFilterNode[]; // Remove BiquadFilterNode usage
    private filterAlphas: Float32Array; // Store alpha coefficient for each filter
    private filterStates: Float32Array; // Store y[n-1] state for each filter
    private dryGain: number = 0.2;
    private wetGain: number = 0.8;
    
    constructor(audioCtx: AudioContext, numDelays: number = 16) {
        this.audioCtx = audioCtx;
        this.delays = [];
        this.feedbackMatrix = [];
        this.gains = new Float32Array(numDelays);
        this.filters = [];
        
        const primes = [743, 769, 797, 823, 853, 877, 907, 937, 
                         967, 997, 1021, 1049, 1087, 1117, 1151, 1181];
        
        for (let i = 0; i < numDelays; i++) {
            const delayTime = primes[i % primes.length];
            this.delays.push(new DelayLine(audioCtx, delayTime));
            
            // Initialize filter state
            this.filterAlphas[i] = 0.98; // Default alpha (adjust in setRT60)
            this.filterStates[i] = 0;
            this.gains[i] = 0.7;
        }
        
        this.initializeHadamardMatrix(numDelays);
    }
    
    private initializeHadamardMatrix(size: number): void {
        const base = [[1]];
        
        let matrix = base;
        while (matrix.length < size) {
            const n = matrix.length;
            const newMatrix = Array(n * 2).fill(0).map(() => Array(n * 2).fill(0));
            
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    newMatrix[i][j] = matrix[i][j];
                    newMatrix[i][j + n] = matrix[i][j];
                    newMatrix[i + n][j] = matrix[i][j];
                    newMatrix[i + n][j + n] = -matrix[i][j];
                }
            }
            
            matrix = newMatrix;
        }
        
        this.feedbackMatrix = [];
        const norm = Math.sqrt(size);
        for (let i = 0; i < size; i++) {
            this.feedbackMatrix.push(new Float32Array(size));
            for (let j = 0; j < size; j++) {
                this.feedbackMatrix[i][j] = matrix[i][j] / norm;
            }
        }
        
        console.log(`Initialized ${size}x${size} Hadamard matrix for FDN`);
    }
    
    public setRT60(rt60Values: {[frequency: string]: number}): void {
        const rt60_1k = rt60Values['1000'] || 1.0;
        const rt60_low = rt60Values['125'] || rt60_1k * 1.2;
        const rt60_high = rt60Values['8000'] || rt60_1k * 0.8;
        
        console.log(`Setting FDN RT60: low=${rt60_low}s, mid=${rt60_1k}s, high=${rt60_high}s`);
        
        for (let i = 0; i < this.delays.length; i++) {
            const delayTimeInSeconds = this.delays[i].getDelayTime() / this.audioCtx.sampleRate;
            this.gains[i] = Math.pow(10, -3 * delayTimeInSeconds / rt60_1k);
            
            const lowRatio = rt60_low / rt60_1k;
            const highRatio = rt60_high / rt60_1k;
            
            // Calculate alpha for one-pole low-pass based on high-frequency decay
            // This is an approximation: relates cutoff frequency to RT60 decay time constant
            const cutoffFreq = 5000 * highRatio; // Target cutoff frequency based on RT60 ratio
            const timeConstant = 1 / (2 * Math.PI * cutoffFreq);
            // Alpha calculation for one-pole filter: alpha = dt / (T + dt) where dt=1/sampleRate, T=timeConstant
            const dt = 1 / this.audioCtx.sampleRate;
            this.filterAlphas[i] = dt / (timeConstant + dt);
            // Clamp alpha to prevent instability or no filtering
            this.filterAlphas[i] = Math.max(0.1, Math.min(0.995, this.filterAlphas[i]));
        }
        console.log(`[Debug FDN] Filter alphas set (example): ${this.filterAlphas[0].toFixed(4)}`);
    }
    
    public setDryWetMix(dryLevel: number, wetLevel: number): void {
        this.dryGain = Math.max(0, Math.min(1, dryLevel));
        this.wetGain = Math.max(0, Math.min(1, wetLevel));
    }
    
    public process(input: Float32Array): Float32Array {
        const output = new Float32Array(input.length);
        const numDelays = this.delays.length;
        
        let delayOutputs = new Array(numDelays).fill(0).map(() => new Float32Array(input.length));
        
        for (let i = 0; i < input.length; i++) {
            for (let d = 0; d < numDelays; d++) {
                delayOutputs[d][i] = this.delays[d].read();
            }
            
            const delayInputs = new Float32Array(numDelays);
            for (let d = 0; d < numDelays; d++) {
                let sum = 0;
                for (let j = 0; j < numDelays; j++) {
                    sum += this.feedbackMatrix[d][j] * delayOutputs[j][i] * this.gains[j];
                }
                delayInputs[d] = sum;
            }
            
            for (let d = 0; d < numDelays; d++) {
                const inputContribution = input[i] * (1 / numDelays);
                // Apply one-pole low-pass filter: y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
                const alpha = this.filterAlphas[d];
                const filteredFeedback = alpha * delayInputs[d] + (1 - alpha) * this.filterStates[d];
                this.filterStates[d] = filteredFeedback; // Update state for next sample
                this.delays[d].write(inputContribution + filteredFeedback);
            }
            
            const drySignal = input[i] * this.dryGain;
            const wetSignal = delayOutputs.reduce((sum, delayOutput) => sum + delayOutput[i], 0) / numDelays * this.wetGain;
            
            output[i] = drySignal + wetSignal;
        }
        
        return output;
    }
    
    public processStereo(inputLeft: Float32Array, inputRight: Float32Array): [Float32Array, Float32Array] {
        const outputLeft = new Float32Array(inputLeft.length);
        const outputRight = new Float32Array(inputRight.length);
        const numDelays = this.delays.length;
        
        const leftDelays = Math.floor(numDelays * 0.6);
        const rightDelays = numDelays - Math.floor(numDelays * 0.4);
        
        let delayOutputs = new Array(numDelays).fill(0).map(() => new Float32Array(inputLeft.length));
        
        for (let i = 0; i < inputLeft.length; i++) {
            for (let d = 0; d < numDelays; d++) {
                delayOutputs[d][i] = this.delays[d].read();
            }
            
            const delayInputs = new Float32Array(numDelays);
            
            for (let d = 0; d < leftDelays; d++) {
                let sum = 0;
                for (let j = 0; j < leftDelays; j++) {
                    sum += this.feedbackMatrix[d][j] * delayOutputs[j][i] * this.gains[j];
                }
                delayInputs[d] = sum;
            }
            
            for (let d = rightDelays; d < numDelays; d++) {
                let sum = 0;
                for (let j = rightDelays; j < numDelays; j++) {
                    sum += this.feedbackMatrix[d - rightDelays][j - rightDelays] * 
                           delayOutputs[j][i] * this.gains[j];
                }
                delayInputs[d] = sum;
            }
            
            for (let d = 0; d < leftDelays; d++) {
                const inputContribution = inputLeft[i] * (1 / leftDelays);
                // Apply one-pole low-pass filter (Left part)
                const alphaL = this.filterAlphas[d];
                const filteredFeedbackL = alphaL * delayInputs[d] + (1 - alphaL) * this.filterStates[d];
                this.filterStates[d] = filteredFeedbackL;
                this.delays[d].write(inputContribution + filteredFeedbackL);
            }
            
            for (let d = rightDelays; d < numDelays; d++) {
                const inputContribution = inputRight[i] * (1 / (numDelays - rightDelays));
                // Apply one-pole low-pass filter (Right part)
                const alphaR = this.filterAlphas[d];
                const filteredFeedbackR = alphaR * delayInputs[d] + (1 - alphaR) * this.filterStates[d];
                this.filterStates[d] = filteredFeedbackR;
                this.delays[d].write(inputContribution + filteredFeedbackR);
            }
            
            let leftWet = 0;
            for (let d = 0; d < leftDelays; d++) {
                leftWet += delayOutputs[d][i];
            }
            leftWet /= leftDelays;
            
            let rightWet = 0;
            for (let d = rightDelays; d < numDelays; d++) {
                rightWet += delayOutputs[d][i];
            }
            rightWet /= (numDelays - rightDelays);
            
            outputLeft[i] = inputLeft[i] * this.dryGain + leftWet * this.wetGain;
            outputRight[i] = inputRight[i] * this.dryGain + rightWet * this.wetGain;
        }
        
        return [outputLeft, outputRight];
    }
    
    // Removed incorrect filterSample method
}

class DelayLine {
    private buffer: Float32Array;
    private writeIndex: number = 0;
    private size: number;
    
    constructor(audioCtx: AudioContext, delaySamples: number) {
        this.size = delaySamples;
        this.buffer = new Float32Array(delaySamples);
        this.buffer.fill(0);
    }
    
    public write(sample: number): void {
        this.buffer[this.writeIndex] = sample;
        this.writeIndex = (this.writeIndex + 1) % this.size;
    }
    
    public read(): number {
        // Read from the position exactly 'size' samples behind the write index
        const readIndex = (this.writeIndex - this.size + this.size) % this.size; // Correct read index for full delay
        return this.buffer[readIndex];
    }
    
    public getDelayTime(): number {
        return this.size;
    }
    
    public clear(): void {
        this.buffer.fill(0);
        this.writeIndex = 0;
    }
}

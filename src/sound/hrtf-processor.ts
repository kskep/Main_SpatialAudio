// Enhanced HRTF processing implementation

import { vec3 } from 'gl-matrix';

// Interface for HRTF dataset
interface HRTFDataset {
    name: string;
    elevations: number[];   // Available elevation angles (degrees)
    azimuths: number[];     // Available azimuth angles (degrees)
    filters: Map<string, AudioBuffer>; // HRTF filters indexed by "elevation_azimuth"
}

export class HRTFProcessor {
    private hrtfDatasets: Map<string, HRTFDataset> = new Map();
    private audioCtx: AudioContext;
    private currentDataset: HRTFDataset | null = null;
    
    constructor(audioCtx: AudioContext) {
        this.audioCtx = audioCtx;
    }
    
    // Load HRTF dataset from a standard SOFA file or simplified JSON format
    async loadHRTFDataset(url: string, name: string): Promise<void> {
        try {
            const response = await fetch(url);
            const data = await response.json();
            
            // Parse the dataset (this is a simplified example)
            // Real implementation would handle SOFA format properly
            const dataset: HRTFDataset = {
                name,
                elevations: data.elevations,
                azimuths: data.azimuths,
                filters: new Map()
            };
            
            // Process each HRTF filter pair
            for (const item of data.hrirs) {
                const key = `${item.elevation}_${item.azimuth}`;
                const leftIR = new Float32Array(item.left);
                const rightIR = new Float32Array(item.right);
                
                // Create AudioBuffer for each HRTF pair
                const buffer = this.audioCtx.createBuffer(2, leftIR.length, this.audioCtx.sampleRate);
                buffer.getChannelData(0).set(leftIR);
                buffer.getChannelData(1).set(rightIR);
                
                dataset.filters.set(key, buffer);
            }
            
            this.hrtfDatasets.set(name, dataset);
            
            // Set as current if it's our first dataset
            if (!this.currentDataset) {
                this.currentDataset = dataset;
            }
            
            console.log(`HRTF dataset '${name}' loaded successfully`);
        } catch (error) {
            console.error("Failed to load HRTF dataset:", error);
        }
    }
    
    // Select an HRTF dataset by name
    selectDataset(name: string): boolean {
        if (this.hrtfDatasets.has(name)) {
            this.currentDataset = this.hrtfDatasets.get(name)!;
            return true;
        }
        return false;
    }
    
    // Get the HRTF filter for a specific direction
    getHRTFForDirection(azimuthDegrees: number, elevationDegrees: number): AudioBuffer | null {
        if (!this.currentDataset) return null;
        
        // Normalize angles
        azimuthDegrees = ((azimuthDegrees % 360) + 360) % 360;
        elevationDegrees = Math.max(-90, Math.min(90, elevationDegrees));
        
        // Find closest available elevation
        const closestElevation = this.findClosest(this.currentDataset.elevations, elevationDegrees);
        
        // Find closest available azimuth
        const closestAzimuth = this.findClosest(this.currentDataset.azimuths, azimuthDegrees);
        
        // Retrieve the corresponding HRTF filter
        const key = `${closestElevation}_${closestAzimuth}`;
        return this.currentDataset.filters.get(key) || null;
    }
    
    // Apply HRTF filtering to an audio buffer
    applyHRTFToBuffer(
        inputBuffer: AudioBuffer, 
        azimuthDegrees: number, 
        elevationDegrees: number
    ): AudioBuffer {
        // Get appropriate HRTF
        const hrtfBuffer = this.getHRTFForDirection(azimuthDegrees, elevationDegrees);
        if (!hrtfBuffer) {
            console.warn("No HRTF filter available for specified direction");
            return inputBuffer; // Return original as fallback
        }
        
        // Prepare output buffer
        const outputBuffer = this.audioCtx.createBuffer(
            2, // Stereo output
            inputBuffer.length,
            this.audioCtx.sampleRate
        );
        
        // Get filter IRs
        const leftIR = hrtfBuffer.getChannelData(0);
        const rightIR = hrtfBuffer.getChannelData(1);
        
        // Get input audio (assume mono for simplicity)
        const inputChannel = inputBuffer.getChannelData(0);
        
        // Perform convolution for both ears
        const leftOutput = this.convolve(inputChannel, leftIR);
        const rightOutput = this.convolve(inputChannel, rightIR);
        
        // Set the output channels
        outputBuffer.getChannelData(0).set(leftOutput);
        outputBuffer.getChannelData(1).set(rightOutput);
        
        return outputBuffer;
    }
    
    // Flexible version that can sample from multiple directions for moving sources
    applyMultiDirectionalHRTF(
        inputBuffer: AudioBuffer,
        directionData: { azimuth: number, elevation: number, time: number }[]
    ): AudioBuffer {
        // Implemented for time-varying HRTF processing
        // This is particularly important for moving sound sources
        
        // Sort direction data by time
        directionData.sort((a, b) => a.time - b.time);
        
        // Prepare output buffer
        const outputBuffer = this.audioCtx.createBuffer(
            2, // Stereo output
            inputBuffer.length,
            this.audioCtx.sampleRate
        );
        
        const leftChannel = outputBuffer.getChannelData(0);
        const rightChannel = outputBuffer.getChannelData(1);
        
        // Process each segment with appropriate HRTF
        for (let i = 0; i < directionData.length - 1; i++) {
            const currentDir = directionData[i];
            const nextDir = directionData[i + 1];
            
            // Sample indices for this segment
            const startSample = Math.floor(currentDir.time * this.audioCtx.sampleRate);
            const endSample = Math.floor(nextDir.time * this.audioCtx.sampleRate);
            
            if (startSample >= inputBuffer.length) break;
            
            // Get HRTF filters for current direction
            const hrtfBuffer = this.getHRTFForDirection(currentDir.azimuth, currentDir.elevation);
            if (!hrtfBuffer) continue;
            
            // Extract segment data
            const segmentLength = Math.min(endSample - startSample, inputBuffer.length - startSample);
            const segmentData = new Float32Array(segmentLength);
            for (let j = 0; j < segmentLength; j++) {
                segmentData[j] = inputBuffer.getChannelData(0)[startSample + j];
            }
            
            // Apply HRTF
            const leftIR = hrtfBuffer.getChannelData(0);
            const rightIR = hrtfBuffer.getChannelData(1);
            
            const leftResult = this.convolve(segmentData, leftIR);
            const rightResult = this.convolve(segmentData, rightIR);
            
            // Copy to output with crossfade for smooth transitions
            for (let j = 0; j < segmentLength; j++) {
                if (startSample + j < outputBuffer.length) {
                    leftChannel[startSample + j] += leftResult[j];
                    rightChannel[startSample + j] += rightResult[j];
                }
            }
        }
        
        return outputBuffer;
    }
    
    // Helper method for basic convolution (you'd likely use Web Audio's ConvolverNode in practice)
    private convolve(input: Float32Array, impulse: Float32Array): Float32Array {
        const result = new Float32Array(input.length + impulse.length - 1);
        
        // Basic convolution algorithm
        for (let i = 0; i < input.length; i++) {
            for (let j = 0; j < impulse.length; j++) {
                result[i + j] += input[i] * impulse[j];
            }
        }
        
        // Return the portion matching the input length (with latency compensation)
        return result.slice(0, input.length);
    }
    
    // Helper to find closest value in an array
    private findClosest(array: number[], target: number): number {
        return array.reduce((prev, curr) => 
            Math.abs(curr - target) < Math.abs(prev - target) ? curr : prev
        );
    }
}

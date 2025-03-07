// Enhanced diffuse field modeling

export class DiffuseFieldModel {
    private sampleRate: number;
    private roomVolume: number;
    private surfaceArea: number;
    private meanAbsorption: { [freq: string]: number };
    private diffusionCoefficients: { [freq: string]: number };
    
    constructor(sampleRate: number, roomConfig: any) {
        this.sampleRate = sampleRate;
        
        // Extract room properties
        const { width, height, depth } = roomConfig.dimensions;
        this.roomVolume = width * height * depth;
        this.surfaceArea = 2 * (width * height + width * depth + height * depth);
        
        // Calculate mean absorption coefficients across all surfaces
        this.meanAbsorption = this.calculateMeanAbsorption(roomConfig.materials);
        
        // Initialize diffusion coefficients (could be loaded from measurements)
        this.diffusionCoefficients = {
            '125': 0.1,
            '250': 0.2,
            '500': 0.3,
            '1000': 0.4,
            '2000': 0.5,
            '4000': 0.6,
            '8000': 0.6,
            '16000': 0.7
        };
    }
    
    // Calculate frequency-dependent mean absorption
    private calculateMeanAbsorption(materials: any): { [freq: string]: number } {
        const result: { [freq: string]: number } = {
            '125': 0,
            '250': 0,
            '500': 0,
            '1000': 0,
            '2000': 0,
            '4000': 0,
            '8000': 0,
            '16000': 0
        };
        
        let surfaceCount = 0;
        
        // Process each surface's material
        for (const [surface, material] of Object.entries(materials)) {
            if (material) {
                result['125'] += material.absorption125Hz || 0.1;
                result['250'] += material.absorption250Hz || 0.1;
                result['500'] += material.absorption500Hz || 0.1;
                result['1000'] += material.absorption1kHz || 0.1;
                result['2000'] += material.absorption2kHz || 0.1;
                result['4000'] += material.absorption4kHz || 0.1;
                result['8000'] += material.absorption8kHz || 0.1;
                result['16000'] += material.absorption16kHz || 0.1;
                surfaceCount++;
            }
        }
        
        // Calculate mean if we have surfaces
        if (surfaceCount > 0) {
            for (const freq in result) {
                result[freq] /= surfaceCount;
            }
        }
        
        return result;
    }
    
    // Create a diffuse field reverb tail for each frequency band
    public generateDiffuseField(
        duration: number, 
        rt60Values: { [freq: string]: number }
    ): Map<string, Float32Array> {
        const result = new Map<string, Float32Array>();
        const frequencies = ['125', '250', '500', '1000', '2000', '4000', '8000', '16000'];
        
        // For each frequency band
        for (const freq of frequencies) {
            // Calculate sample count
            const sampleCount = Math.ceil(duration * this.sampleRate);
            const buffer = new Float32Array(sampleCount);
            
            // Get RT60 for this frequency band
            const rt60 = rt60Values[freq] || 1.0;
            
            // Get diffusion coefficient for this frequency
            const diffusion = this.diffusionCoefficients[freq] || 0.3;
            
            // Parameters for statistical model
            const echoDensity = 1000 * (this.roomVolume / 100) * (1 + diffusion); // Echoes per second
            const meanFreePath = 4 * this.roomVolume / this.surfaceArea; // Mean free path in meters
            const speedOfSound = 343; // m/s
            const meanTimeGap = meanFreePath / speedOfSound; // Time between reflections
            
            // Generate statistical reverb model
            this.generateVelvetNoise(buffer, rt60, echoDensity, meanTimeGap, diffusion);
            
            // Apply frequency-dependent envelope
            this.applyDecayEnvelope(buffer, rt60);
            
            // Store the buffer for this frequency band
            result.set(freq, buffer);
        }
        
        return result;
    }
    
    // Create velvet noise (efficient sparse FIR) for diffuse reverb modeling
    private generateVelvetNoise(
        buffer: Float32Array, 
        rt60: number, 
        echoDensity: number, 
        meanTimeGap: number,
        diffusion: number
    ): void {
        // Calculate required properties
        const td = 1 / echoDensity; // Average time between impulses (seconds)
        const totalPulses = Math.floor(buffer.length / (td * this.sampleRate));
        
        // Generate positions and polarities for the pulses
        for (let i = 0; i < totalPulses; i++) {
            // Randomized position with higher density over time
            const position = Math.floor(i * td * this.sampleRate + 
                (Math.random() - 0.5) * 2 * diffusion * td * this.sampleRate);
            
            // Skip if position is outside buffer
            if (position < 0 || position >= buffer.length) continue;
            
            // Random polarity (white noise)
            const polarity = Math.random() > 0.5 ? 1 : -1;
            
            // Calculate amplitude based on decay curve
            const time = position / this.sampleRate;
            const amplitude = Math.exp(-6.91 * time / rt60);
            
            // Add pulse to buffer
            buffer[position] += polarity * amplitude;
        }
    }
    
    // Apply exponential decay envelope
    private applyDecayEnvelope(buffer: Float32Array, rt60: number): void {
        for (let i = 0; i < buffer.length; i++) {
            const time = i / this.sampleRate;
            buffer[i] *= Math.exp(-6.91 * time / rt60);
        }
    }
    
    // Apply shelving filters to model frequency-dependent decay
    public applyFrequencyFiltering(
        impulseResponses: Map<string, Float32Array>
    ): Float32Array {
        // Total length based on any of the band IRs
        const anyIR = impulseResponses.values().next().value;
        const totalLength = anyIR ? anyIR.length : 0;
        
        // Create output buffer
        const outputIR = new Float32Array(totalLength);
        
        // Simple approach: sum all frequency bands
        for (const [freq, ir] of impulseResponses.entries()) {
            // Apply gain for each frequency band based on typical RT60 patterns
            let bandGain = 1.0;
            
            switch (freq) {
                case '125': bandGain = 0.9; break;
                case '250': bandGain = 0.95; break;
                case '500': bandGain = 1.0; break;
                case '1000': bandGain = 1.0; break;
                case '2000': bandGain = 0.9; break;
                case '4000': bandGain = 0.8; break;
                case '8000': bandGain = 0.7; break;
                case '16000': bandGain = 0.6; break;
                default: bandGain = 1.0;
            }
            
            // Sum into output with appropriate gain
            for (let i = 0; i < Math.min(ir.length, totalLength); i++) {
                outputIR[i] += ir[i] * bandGain;
            }
        }
        
        // Normalize the output
        const maxAmp = Math.max(...Array.from(outputIR).map(Math.abs));
        if (maxAmp > 0) {
            for (let i = 0; i < outputIR.length; i++) {
                outputIR[i] /= maxAmp;
            }
        }
        
        return outputIR;
    }
    
    // Process late reverberation for a set of ray hits
    public processLateReverberation(
        lateHits: any[],
        camera: any,
        roomConfig: any,
        sampleRate: number
    ): [Float32Array, Float32Array] {
        // Calculate RT60 values based on room properties and hit statistics
        const rt60Values = this.calculateRT60Values(lateHits, roomConfig);
        
        // Generate diffuse field responses for each frequency band
        const diffuseResponses = this.generateDiffuseField(2.0, rt60Values); // 2 seconds of reverb
        
        // Apply frequency-dependent filtering
        const monoIR = this.applyFrequencyFiltering(diffuseResponses);
        
        // Create stereo output with spatial variation
        const leftIR = new Float32Array(monoIR.length);
        const rightIR = new Float32Array(monoIR.length);
        
        // Add subtle stereo decorrelation
        for (let i = 0; i < monoIR.length; i++) {
            const phase = Math.random() * 2 * Math.PI;
            leftIR[i] = monoIR[i] * (1 + 0.1 * Math.sin(phase));
            rightIR[i] = monoIR[i] * (1 + 0.1 * Math.cos(phase));
        }
        
        return [leftIR, rightIR];
    }
    
    private calculateRT60Values(lateHits: any[], roomConfig: any): { [freq: string]: number } {
        // Base RT60 calculation using Sabine's formula
        const frequencies = ['125', '250', '500', '1000', '2000', '4000', '8000', '16000'];
        const rt60Values: { [freq: string]: number } = {};
        
        for (const freq of frequencies) {
            // Get absorption coefficient for this frequency
            const absorption = this.meanAbsorption[freq];
            
            // Sabine's formula: RT60 = 0.161 * V / (A * α)
            // where V is room volume, A is surface area, α is absorption coefficient
            let rt60 = 0.161 * this.roomVolume / (this.surfaceArea * absorption);
            
            // Adjust based on frequency (empirical adjustments)
            if (parseInt(freq) < 500) {
                rt60 *= 1.2; // Longer decay for low frequencies
            } else if (parseInt(freq) > 2000) {
                rt60 *= 0.8; // Shorter decay for high frequencies
            }
            
            // Clamp to reasonable values
            rt60Values[freq] = Math.min(Math.max(rt60, 0.1), 3.0);
        }
        
        return rt60Values;
    }
    
    // Combine early reflections with diffuse field for complete IR
    public combineWithEarlyReflections(
        earlyReflections: Float32Array,
        diffuseField: Float32Array,
        crossoverTime: number
    ): Float32Array {
        const earlyLength = earlyReflections.length;
        const diffuseLength = diffuseField.length;
        const totalLength = Math.max(earlyLength, diffuseLength);
        
        // Create output buffer
        const output = new Float32Array(totalLength);
        
        // Calculate crossover sample
        const crossoverSample = Math.floor(crossoverTime * this.sampleRate);
        
        // Apply crossfade window
        const fadeLength = Math.floor(0.01 * this.sampleRate); // 10ms fade
        
        // Copy early reflections
        for (let i = 0; i < earlyLength; i++) {
            if (i < crossoverSample - fadeLength) {
                // Fully early reflections
                output[i] = earlyReflections[i];
            } else if (i < crossoverSample + fadeLength) {
                // Crossfade region
                const fadePos = (i - (crossoverSample - fadeLength)) / (fadeLength * 2);
                const earlyGain = 1 - fadePos;
                const diffuseGain = fadePos;
                
                output[i] = earlyReflections[i] * earlyGain;
                if (i < diffuseLength) {
                    output[i] += diffuseField[i] * diffuseGain;
                }
            }
        }
        
        // Copy diffuse field for remainder
        for (let i = crossoverSample + fadeLength; i < totalLength; i++) {
            if (i < diffuseLength) {
                output[i] = diffuseField[i];
            }
        }
        
        return output;
    }
}

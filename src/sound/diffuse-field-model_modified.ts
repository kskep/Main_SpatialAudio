// src/sound/diffuse-field-model_modified.ts
// Enhanced diffuse field modeling with improved stereo decorrelation

export class DiffuseFieldModelModified { // Renamed class
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
            '125': 0.1, '250': 0.2, '500': 0.3, '1000': 0.4,
            '2000': 0.5, '4000': 0.6, '8000': 0.6, '16000': 0.7
        };
    }

    // Calculate frequency-dependent mean absorption
    private calculateMeanAbsorption(materials: any): { [freq: string]: number } {
        const result: { [freq: string]: number } = {
            '125': 0, '250': 0, '500': 0, '1000': 0,
            '2000': 0, '4000': 0, '8000': 0, '16000': 0
        };
        let surfaceCount = 0;
        for (const [surface, material] of Object.entries(materials)) {
            if (material) {
                result['125'] += (material as any).absorption125Hz || 0.1;
                result['250'] += (material as any).absorption250Hz || 0.1;
                result['500'] += (material as any).absorption500Hz || 0.1;
                result['1000'] += (material as any).absorption1kHz || 0.1;
                result['2000'] += (material as any).absorption2kHz || 0.1;
                result['4000'] += (material as any).absorption4kHz || 0.1;
                result['8000'] += (material as any).absorption8kHz || 0.1;
                result['16000'] += (material as any).absorption16kHz || 0.1;
                surfaceCount++;
            }
        }
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

        // Ensure we have a valid duration
        const safeDuration = Math.max(duration, 0.5); // Minimum 500ms
        console.log(`[Debug] Generating diffuse field with duration: ${safeDuration}s`);

        for (const freq of frequencies) {
            // Calculate sample count with a minimum value
            const sampleCount = Math.max(Math.ceil(safeDuration * this.sampleRate), 1000); // Minimum 1000 samples

            // Create buffer with validated length
            const buffer = new Float32Array(sampleCount);

            // Get RT60 with fallback
            const rt60 = rt60Values[freq] || 1.0;

            // Calculate diffusion parameters with safety checks
            const diffusion = this.diffusionCoefficients[freq] || 0.3;

            // Ensure room volume and surface area are positive
            const safeRoomVolume = Math.max(this.roomVolume, 10); // Minimum 10 cubic meters
            const safeSurfaceArea = Math.max(this.surfaceArea, 10); // Minimum 10 square meters

            // Calculate echo density with safety checks
            const echoDensity = Math.max(100, 1000 * (safeRoomVolume / 100) * (1 + diffusion));

            // Calculate mean free path and time gap
            const meanFreePath = 4 * safeRoomVolume / safeSurfaceArea;
            const speedOfSound = 343;
            const meanTimeGap = meanFreePath / speedOfSound;

            if (freq === '16000') {
                console.log(`[Debug 16k Params] Freq=${freq}, RT60=${rt60}, EchoDensity=${echoDensity}, MeanTimeGap=${meanTimeGap}, Diffusion=${diffusion}, BufferLen=${buffer.length}`);
            }

            // Generate the velvet noise
            this.generateVelvetNoise(buffer, rt60, echoDensity, meanTimeGap, diffusion);

            // Store the result
            result.set(freq, buffer);

            console.log(`[Debug] Generated ${freq}Hz band: ${sampleCount} samples, RT60=${rt60.toFixed(2)}s`);
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
        // Ensure we have a reasonable echo density
        const safeEchoDensity = Math.max(echoDensity, 100); // Minimum 100 echoes per second
        const td = 1 / safeEchoDensity;

        // Calculate total pulses, ensuring at least 100 pulses for any buffer
        let totalPulses = Math.floor(buffer.length / (td * this.sampleRate));
        totalPulses = Math.max(totalPulses, 100); // Ensure minimum number of pulses

        console.log(`[Debug] Velvet Noise: Buffer length=${buffer.length}, RT60=${rt60.toFixed(2)}s, EchoDensity=${safeEchoDensity}, TotalPulses=${totalPulses}`);

        for (let i = 0; i < totalPulses; i++) {
            // Calculate position with jitter based on diffusion
            const basePosition = i * buffer.length / totalPulses; // Distribute evenly across buffer
            const jitter = (Math.random() - 0.5) * 2 * diffusion * buffer.length / totalPulses;
            const position = Math.floor(basePosition + jitter);

            if (position < 0 || position >= buffer.length) continue;
            const polarity = Math.random() > 0.5 ? 1 : -1;
            const time = position / this.sampleRate;

            const decayTerm = -6.91 * time / rt60;
            // console.log(`[Debug Velvet Calc] i=${i}, pos=${position}, time=${time.toFixed(4)}, rt60=${rt60.toFixed(4)}, decayTerm=${decayTerm.toFixed(4)}`);

            // Amplitude calculation already includes decay based on RT60
            const amplitude = Math.exp(decayTerm);

            if (isNaN(amplitude)) {
                console.error(`[Debug Velvet NaN] NaN detected! i=${i}, pos=${position}, time=${time}, rt60=${rt60}, decayTerm=${decayTerm}`);
                // Optionally handle the NaN case, e.g., set amplitude to 0
                // amplitude = 0;
            }

            buffer[position] += polarity * amplitude;
        }
    }

    // Apply exponential decay envelope (Redundant if generateVelvetNoise includes decay)
    /*
    private applyDecayEnvelope(buffer: Float32Array, rt60: number): void {
        for (let i = 0; i < buffer.length; i++) {
            const time = i / this.sampleRate;
            buffer[i] *= Math.exp(-6.91 * time / rt60);
        }
    }
    */

    // Apply shelving filters to model frequency-dependent decay
    public applyFrequencyFiltering(
        impulseResponses: Map<string, Float32Array>
    ): Float32Array {
        const anyIR = impulseResponses.values().next().value;
        const totalLength = anyIR ? anyIR.length : 0;
        if (totalLength === 0) return new Float32Array(0);

        const outputIR = new Float32Array(totalLength);
        for (const [freq, ir] of impulseResponses.entries()) {
            let bandGain = 1.0;
            switch (freq) {
                case '125': bandGain = 0.9; break;
                case '250': bandGain = 0.95; break;
                case '500': bandGain = 1.0; break;
                case '1000': bandGain = 1.0; break;
                case '2000': bandGain = 0.8; break;  
                case '4000': bandGain = 0.5; break;  
                case '8000': bandGain = 0.3; break;  
                case '16000': bandGain = 0.15; break;
            }
            for (let i = 0; i < Math.min(ir.length, totalLength); i++) {
                outputIR[i] += ir[i] * bandGain;
            }
        }

        // Normalize the output
        let maxAmp = 0;
        for (let i = 0; i < outputIR.length; i++) {
            maxAmp = Math.max(maxAmp, Math.abs(outputIR[i]));
        }
        if (maxAmp > 0) {
            const scale = 1.0 / maxAmp;
            for (let i = 0; i < outputIR.length; i++) {
                outputIR[i] *= scale;
            }
        }
        return outputIR;
    }

    // Process late reverberation for a set of ray hits
    public processLateReverberation(
        lateHits: any[],
        camera: any, // Assuming Camera type
        roomConfig: any,
        sampleRate: number
    ): [Float32Array, Float32Array] {
        // Validate input
        if (!lateHits || lateHits.length === 0) {
            console.warn("No late hits provided for reverberation processing");
            // Return minimal valid IRs instead of empty ones
            const minLength = Math.ceil(0.5 * sampleRate); // 500ms minimum
            const defaultIR = new Float32Array(minLength);
            // Add a small impulse to avoid completely silent IRs
            defaultIR[0] = 0.1;
            return [defaultIR.slice(), defaultIR.slice()];
        }

        console.log(`[Debug] Processing late reverberation with ${lateHits.length} hits`);

        const rt60Values = this.calculateRT60Values(lateHits, roomConfig);
        console.log(`[Debug] RT60 Values: 500Hz=${rt60Values['500']?.toFixed(2)}s, 1kHz=${rt60Values['1000']?.toFixed(2)}s`);

        const diffuseResponses = this.generateDiffuseField(2.0, rt60Values); // 2 seconds reverb
        const monoIR = this.applyFrequencyFiltering(diffuseResponses);

        // Validate monoIR has content
        if (!monoIR || monoIR.length === 0) {
            console.error("Failed to generate valid mono IR");
            // Return minimal valid IRs
            const minLength = Math.ceil(0.5 * sampleRate); // 500ms minimum
            const defaultIR = new Float32Array(minLength);
            defaultIR[0] = 0.1;
            return [defaultIR.slice(), defaultIR.slice()];
        }

        // Check if monoIR has any non-zero values
        let hasContent = false;
        for (let i = 0; i < monoIR.length; i++) {
            if (Math.abs(monoIR[i]) > 1e-10) {
                hasContent = true;
                break;
            }
        }

        if (!hasContent) {
            console.warn("Generated mono IR has no significant content");
            // Add minimal content to avoid silent IRs
            monoIR[0] = 0.1;
            monoIR[Math.floor(monoIR.length * 0.1)] = 0.05;
            monoIR[Math.floor(monoIR.length * 0.2)] = 0.025;
        }

        console.log(`[Debug] Generated mono IR length: ${monoIR.length}`);

        const leftIR = new Float32Array(monoIR.length);
        const rightIR = new Float32Array(monoIR.length);

        // --- MODIFICATION START: Improved Stereo Decorrelation ---
        // Use a small, randomized delay (e.g., up to ~1ms) and slight filtering difference
        // const maxDelaySamples = Math.floor(0.001 * this.sampleRate); // Max 1ms delay
        // const delaySamples = Math.floor(Math.random() * maxDelaySamples); // Remove fixed delay

        // Simple low-pass filter coefficients (example)
        const alpha = 0.98; // Controls filter cutoff - adjust as needed
        let prevL = 0; // Low-pass state L
        let prevR = 0; // Low-pass state R

        // All-pass filter states for L/R channels
        const allPassG_L = 0.5; // Coefficient for Left
        const allPassD_L = 3;   // Delay for Left (samples)
        let allPassX_L: number[] = new Array(allPassD_L + 1).fill(0);
        let allPassY_L: number[] = new Array(allPassD_L + 1).fill(0);

        const allPassG_R = 0.45; // Slightly different Coefficient for Right
        const allPassD_R = 5;   // Slightly different Delay for Right (samples)
        let allPassX_R: number[] = new Array(allPassD_R + 1).fill(0); // Input buffer for right channel (Use _R)
        let allPassY_R: number[] = new Array(allPassD_R + 1).fill(0); // Output buffer for right channel (Use _R)

        for (let i = 0; i < monoIR.length; i++) {
            const currentSample = monoIR[i];

            // Left channel (slightly filtered)
            const filteredSampleL = alpha * currentSample + (1 - alpha) * prevL;
            prevL = filteredSampleL;

            // Apply All-Pass Filter to Left Channel
            for (let k = allPassD_L; k > 0; k--) {
                allPassX_L[k] = allPassX_L[k-1];
                allPassY_L[k] = allPassY_L[k-1];
            }
            allPassX_L[0] = filteredSampleL;
            const x_n_minus_D_L = allPassX_L[allPassD_L];
            const y_n_minus_D_L = allPassY_L[allPassD_L];
            const allPassOutputL = allPassG_L * filteredSampleL + x_n_minus_D_L - allPassG_L * y_n_minus_D_L;
            allPassY_L[0] = allPassOutputL;
            leftIR[i] = allPassOutputL;

            // Right channel (delayed and slightly differently filtered)
            // const delayedIndex = i - delaySamples; // No longer needed
            const delayedSample = currentSample; // Use the current sample for right channel processing before filtering
            // Use the same alpha for both channels for balanced filtering
            // const alphaR = alpha * 0.99; // Remove differing alpha
            const filteredDelayedSample = alpha * delayedSample + (1 - alpha) * prevR; // Apply low-pass first
            prevR = filteredDelayedSample;

            // Apply All-Pass Filter to the filtered+delayed right channel signal
            // Shift buffers
            for (let k = allPassD_R; k > 0; k--) { // Use _R
                allPassX_R[k] = allPassX_R[k-1];
                allPassY_R[k] = allPassY_R[k-1];
            }
            allPassX_R[0] = filteredDelayedSample;

            // Calculate all-pass output: y[n] = G*x[n] + x[n-D] - G*y[n-D]
            const x_n_minus_D_R = allPassX_R[allPassD_R]; // Use _R
            const y_n_minus_D_R = allPassY_R[allPassD_R]; // Use _R
            const allPassOutputR = allPassG_R * filteredDelayedSample + x_n_minus_D_R - allPassG_R * y_n_minus_D_R; // Use filteredDelayedSample

            allPassY_R[0] = allPassOutputR; // Store output for next iteration (Use _R)
            rightIR[i] = allPassOutputR;    // Assign final output to right IR (Use _R)
        }
        // --- MODIFICATION END ---

        // Final validation
        console.log(`[Debug] Final IR lengths - Left: ${leftIR.length}, Right: ${rightIR.length}`);

        return [leftIR, rightIR];
    }

    private calculateRT60Values(lateHits: any[], roomConfig: any): { [freq: string]: number } {
        const frequencies = ['125', '250', '500', '1000', '2000', '4000', '8000', '16000'];
        const rt60Values: { [freq: string]: number } = {};
        for (const freq of frequencies) {
            const absorption = this.meanAbsorption[freq];
            // Ensure absorption is positive and non-zero to avoid division by zero or negative RT60
            const effectiveAbsorption = Math.max(absorption, 0.01);
            let rt60 = 0.161 * this.roomVolume / (this.surfaceArea * effectiveAbsorption);

            // Adjust based on frequency (empirical adjustments)
            if (parseInt(freq) < 500) rt60 *= 1.2;
            else if (parseInt(freq) > 2000) rt60 *= 0.8;

            // Clamp to reasonable values
            rt60Values[freq] = Math.min(Math.max(rt60, 0.1), 5.0); // Increased max RT60 slightly
        }
        return rt60Values;
    }

    // Combine early reflections with diffuse field for complete IR
    // (This function might be better placed in AudioProcessor, but kept here for reference)
    public combineWithEarlyReflections(
        earlyReflections: Float32Array,
        diffuseField: Float32Array,
        crossoverTime: number
    ): Float32Array {
        const earlyLength = earlyReflections.length;
        const diffuseLength = diffuseField.length;
        const totalLength = Math.max(earlyLength, diffuseLength);
        const output = new Float32Array(totalLength);
        const crossoverSample = Math.floor(crossoverTime * this.sampleRate);
        const fadeLength = Math.floor(0.01 * this.sampleRate); // 10ms fade

        for (let i = 0; i < totalLength; i++) {
            const earlyVal = (i < earlyLength) ? earlyReflections[i] : 0;
            const diffuseVal = (i < diffuseLength) ? diffuseField[i] : 0;

            if (i < crossoverSample - fadeLength) {
                output[i] = earlyVal; // Fully early
            } else if (i < crossoverSample + fadeLength) {
                // Crossfade region
                const fadePos = (i - (crossoverSample - fadeLength)) / (fadeLength * 2);
                const earlyGain = 0.5 * (1 + Math.cos(fadePos * Math.PI)); // Cosine fade
                const diffuseGain = 0.5 * (1 - Math.cos(fadePos * Math.PI)); // Cosine fade
                output[i] = earlyVal * earlyGain + diffuseVal * diffuseGain;
            } else {
                output[i] = diffuseVal; // Fully diffuse
            }
        }
        return output;
    }
}
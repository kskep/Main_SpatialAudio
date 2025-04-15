// src/sound/audio-processor_modified.ts
// Integrates HRTFProcessor and DiffuseFieldModelModified for improved realism

import { Camera } from '../camera/camera';
import { Room } from '../room/room';
import { WaveformRenderer } from '../visualization/waveform-renderer'; // Corrected path
import { RayHit } from '../raytracer/raytracer'; // Assume RayHit is here
import { DiffuseFieldModelModified } from './diffuse-field-model_modified'; // Corrected filename
import { vec3 } from 'gl-matrix';

export class AudioProcessorModified { // Renamed class
    private audioCtx: AudioContext;
    private room: Room; // Added type annotation
    private camera: Camera; // Added type annotation
    private diffuseFieldModel: DiffuseFieldModelModified;
    private impulseResponseBuffer: AudioBuffer | null = null; // Store the final IR buffer
    private lastImpulseData: Float32Array | null = null; // Store interleaved stereo data for visualization
    private sampleRate: number;
    private lastRayHits: RayHit[] = []; // Store last processed hits for potential reuse/debugging
    private currentSourceNode: AudioBufferSourceNode | null = null; // Track the playing source

    constructor(audioCtx: AudioContext, room: Room, camera: Camera, sampleRate: number) { // Added types to params
        this.audioCtx = audioCtx;
        this.room = room;
        this.camera = camera;
        this.sampleRate = sampleRate;

        // Initialize modified diffuse field model
        // Ensure room dimensions and materials are correctly passed
        const roomConfigForModel = {
             dimensions: {
                 width: room.config.dimensions.width || 10, // Access via config
                 height: room.config.dimensions.height || 3, // Access via config
                 depth: room.config.dimensions.depth || 10 // Access via config
             },
             materials: room.config.materials || { // Access via config
                 walls: { absorption125Hz: 0.1, absorption250Hz: 0.1, absorption500Hz: 0.1, absorption1kHz: 0.1, absorption2kHz: 0.1, absorption4kHz: 0.1, absorption8kHz: 0.1, absorption16kHz: 0.1, scattering125Hz: 0.1, scattering250Hz: 0.2, scattering500Hz: 0.3, scattering1kHz: 0.4, scattering2kHz: 0.5, scattering4kHz: 0.6, scattering8kHz: 0.6, scattering16kHz: 0.7, roughness: 0.5, phaseShift: 0, phaseRandomization: 0 },
                 ceiling: { absorption125Hz: 0.15, absorption250Hz: 0.15, absorption500Hz: 0.15, absorption1kHz: 0.15, absorption2kHz: 0.15, absorption4kHz: 0.15, absorption8kHz: 0.15, absorption16kHz: 0.15, scattering125Hz: 0.1, scattering250Hz: 0.2, scattering500Hz: 0.3, scattering1kHz: 0.4, scattering2kHz: 0.5, scattering4kHz: 0.6, scattering8kHz: 0.6, scattering16kHz: 0.7, roughness: 0.5, phaseShift: 0, phaseRandomization: 0 },
                 floor: { absorption125Hz: 0.05, absorption250Hz: 0.05, absorption500Hz: 0.05, absorption1kHz: 0.05, absorption2kHz: 0.05, absorption4kHz: 0.05, absorption8kHz: 0.05, absorption16kHz: 0.05, scattering125Hz: 0.1, scattering250Hz: 0.2, scattering500Hz: 0.3, scattering1kHz: 0.4, scattering2kHz: 0.5, scattering4kHz: 0.6, scattering8kHz: 0.6, scattering16kHz: 0.7, roughness: 0.5, phaseShift: 0, phaseRandomization: 0 }
             }
         };
        this.diffuseFieldModel = new DiffuseFieldModelModified(this.sampleRate, roomConfigForModel);
    }

    async processRayHits(
        rayHits: RayHit[],
        // Removed unused params
    ): Promise<void> {
        try {
            if (!rayHits || !Array.isArray(rayHits) || rayHits.length === 0) {
                console.warn('No valid ray hits to process'); return;
            }
            // Update check for initialized components
            if (!this.diffuseFieldModel) {
                 console.error('Audio components not initialized'); return;
            }

            const validHits = rayHits.filter(hit => hit && hit.position && hit.energies && isFinite(hit.time));
            if (validHits.length === 0) {
                console.warn('No valid ray hits after filtering');
                return;
            }

            this.lastRayHits = validHits; // Store the valid hits

            const [leftIR, rightIR] = this.processRayHitsInternal(validHits);

            // Create interleaved stereo data for visualization
            const stereoData = new Float32Array(leftIR.length * 2);
            for (let i = 0; i < leftIR.length; i++) {
                stereoData[i * 2] = leftIR[i];
                stereoData[i * 2 + 1] = rightIR[i];
            }
            this.lastImpulseData = stereoData;

            // Set up the final impulse response buffer
            await this.setupImpulseResponseBuffer(leftIR, rightIR); // Restore call
        } catch (error) {
            console.error('Error processing ray hits:', error);
            throw error; // Re-throw to allow caller to handle
        }
    }

    private processRayHitsInternal(hits: RayHit[]): [Float32Array, Float32Array] {
        // Ensure we have a valid IR length (2 seconds)
        const irLength = Math.max(Math.ceil(this.sampleRate * 2), 1000); // At least 2 seconds or 1000 samples
        console.log(`[Debug] Creating IR buffers with length: ${irLength} samples (${irLength/this.sampleRate}s)`);

        const leftIR = new Float32Array(irLength);
        const rightIR = new Float32Array(irLength);

        try { // Restore try block for the entire method
            // Sort hits by time for temporal coherence
            const sortedHits = [...hits].sort((a, b) => a.time - b.time);

            // Process Early Reflections
            const earlyReflectionCutoff = 0.1; // 100ms
            const earlyHits = sortedHits.filter(hit => hit.time < earlyReflectionCutoff);
            for (const hit of earlyHits) {
                const sampleIndex = Math.floor(hit.time * this.sampleRate);
                // Validate sampleIndex *before* complex calculations
                if (sampleIndex < 0 || sampleIndex >= irLength || !isFinite(sampleIndex)) {
                     console.warn(`[Debug] Invalid sampleIndex ${sampleIndex} for hit time ${hit.time}`);
                     continue;
                }

                // 1. Calculate Hit Amplitude (with adjusted bounce attenuation)
                const totalEnergy = Object.values(hit.energies).reduce((sum: number, e) => sum + (typeof e === 'number' ? e : 0), 0); // Explicitly type sum
                // Adjusted bounce factor (less attenuation)
                // Apply an additional scaling factor to control overall loudness
                const loudnessScale = 0.15; // Reduced loudness again to avoid potential clipping
                let amplitude = Math.sqrt(Math.max(0, totalEnergy)) * Math.exp(-hit.bounces * 0.2) * loudnessScale;

                // Clamp and validate amplitude immediately
                amplitude = Math.max(0, Math.min(1.0, amplitude)); // Clamp between 0 and 1
                if (!isFinite(amplitude)) {
                    console.warn(`[Debug] Amplitude became non-finite after calculation/clamping for hit time ${hit.time}. Setting to 0.`);
                    amplitude = 0;
                }
                
                console.log(`[Debug] Early Hit Time: ${hit.time.toFixed(4)}, Bounces: ${hit.bounces}, Amplitude: ${amplitude.toExponential(3)} (Scale: ${loudnessScale})`); // Log amplitude
                if (amplitude < 1e-6) {
                    // console.warn(`[Debug] Negligible amplitude ${amplitude.toExponential(3)} for hit time ${hit.time}`);
                    continue; // Skip negligible energy hits
                }
                // 2. Calculate Direction
                const listenerPos = this.camera.getPosition();
                const direction = vec3.create();
                vec3.subtract(direction, hit.position, listenerPos);
                const distance = vec3.length(direction); // Re-add distance calculation
                vec3.normalize(direction, direction); // Direction from listener to hit arrival point

                const listenerRight = this.camera.getRight();
                const listenerFront = this.camera.getFront();
                const listenerUp = this.camera.getUp();

                const dotRight = vec3.dot(direction, listenerRight);
                const dotFront = vec3.dot(direction, listenerFront);
                const dotUp = vec3.dot(direction, listenerUp);

                const azimuthRad = Math.atan2(dotRight, dotFront);
                const elevationRad = Math.asin(Math.max(-1, Math.min(1, dotUp)));
                // Removed unused degree conversions

                // --- Apply Gain + ITD Model ---

                // 3. Calculate Gain based on direction/distance using helper
                let [leftGain, rightGain] = this.calculateBalancedSpatialGains(azimuthRad, elevationRad, distance); // Use new balanced gain function
                if (!isFinite(leftGain) || !isFinite(rightGain)) {
                    console.warn(`[Debug] Invalid spatial gains L=${leftGain}, R=${rightGain} for hit time ${hit.time}`);
                    leftGain = 0; rightGain = 0; // Default to zero gain if invalid
                }

                // 4. Calculate ITD using helper
                let itd_samples = this.calculateITDsamples(azimuthRad, this.sampleRate);
                if (!isFinite(itd_samples)) {
                    console.warn(`[Debug] Invalid ITD ${itd_samples} for hit time ${hit.time}`);
                    itd_samples = 0; // Default to zero ITD if invalid
                }

                // 5. Determine delay per ear
                // --- DEBUG: Temporarily disable ITD ---
                let leftDelaySamples = 0;
                let rightDelaySamples = 0;
                // if (itd_samples > 0) { // Sound arrives at left ear first, delay right ear
                //     rightDelaySamples = itd_samples;
                // } else if (itd_samples < 0) { // Sound arrives at right ear first, delay left ear
                //     leftDelaySamples = -itd_samples;
                // }
                // --- END DEBUG ---

                // 6. Apply gain-adjusted amplitude with delay
                // Removed unused indices (using baseLeftIndex/baseRightIndex below instead)

                // --- Re-introduce Temporal Spreading (Increased Duration Again) ---
                const spreadDurationMs = 5; // Reduced spread significantly to 5ms
                // --- Apply impulse directly without temporal spread ---
                const baseLeftIndex = sampleIndex + leftDelaySamples;
                const baseRightIndex = sampleIndex + rightDelaySamples;

                // Apply impulse directly to integer sample index (no interpolation)
                const currentAmplitudeL = amplitude * leftGain;
                const currentAmplitudeR = amplitude * rightGain;

                const targetIndexL = Math.floor(baseLeftIndex);
                if (targetIndexL >= 0 && targetIndexL < irLength) {
                    leftIR[targetIndexL] += currentAmplitudeL;
                }

                const targetIndexR = Math.floor(baseRightIndex);
                 if (targetIndexR >= 0 && targetIndexR < irLength) {
                    rightIR[targetIndexR] += currentAmplitudeR;
                }
                // --- End direct impulse application ---
                // --- End Gain + ITD Model ---
            }
            // --- Debug Log Start ---
            let earlyEnergyL_dbg = 0;
            let earlyEnergyR_dbg = 0;
            for(let i=0; i<earlyReflectionCutoff*this.sampleRate && i < leftIR.length; i++) { // Rough energy check for early part
                 earlyEnergyL_dbg += leftIR[i]*leftIR[i];
                 earlyEnergyR_dbg += rightIR[i]*rightIR[i];
            }
            console.log(`[Debug] Total Energy After Early Hits (approx) - L: ${earlyEnergyL_dbg.toExponential(2)}, R: ${earlyEnergyR_dbg.toExponential(2)}`);
            // --- Debug Log End ---

            // --- Process Late Reflections using Modified Diffuse Field Model ---
            const lateHits = sortedHits.filter(hit => hit.time >= earlyReflectionCutoff);
            let lateDiffuseL = new Float32Array(irLength); // Initialize empty
            let lateDiffuseR = new Float32Array(irLength);

            if (lateHits.length > 0 && this.diffuseFieldModel) {
                 const roomConfig = { // Pass current room config
                     dimensions: { width: this.room.config.dimensions.width || 10, height: this.room.config.dimensions.height || 3, depth: this.room.config.dimensions.depth || 10 }, // Access via config
                     materials: this.room.config.materials // Access via config
                 };
                 try {
                    // Process and get the late reverb IRs
                    const [generatedLateL, generatedLateR] = this.diffuseFieldModel.processLateReverberation(
                        lateHits, this.camera, roomConfig, this.sampleRate
                    );
                    // Copy into the main late buffers, respecting length
                    const copyLength = Math.min(irLength, generatedLateL.length);
                    lateDiffuseL.set(generatedLateL.slice(0, copyLength));
                    lateDiffuseR.set(generatedLateR.slice(0, copyLength));

                    console.log(`[Debug] Generated Late Reverb Length: ${generatedLateL.length}, IR Length: ${irLength}`);
                } catch (e) {
                    console.error("Error generating late reverberation:", e);
                }
            }

            // --- Combine Early Reflections and Late Reverberation with Crossfade ---
            // Ensure valid crossfade points
            const crossfadeStartSample = Math.floor(0.08 * this.sampleRate); // Start fade at 80ms
            const crossfadeEndSample = Math.floor(0.12 * this.sampleRate); // End fade at 120ms
            const crossfadeDuration = Math.max(1, crossfadeEndSample - crossfadeStartSample); // Ensure positive duration
            const lateReverbGain = 0.005; // Further reduced gain for late reverb

            console.log(`[Debug] Combining... Crossfade: ${crossfadeStartSample}-${crossfadeEndSample} samples. Late Gain: ${lateReverbGain}`);

            // Validate late reverb buffers
            if (lateDiffuseL.length === 0 || lateDiffuseR.length === 0) {
                console.warn("Late reverb buffers are empty, skipping late reverb");
                // Skip late reverb processing, just keep early reflections
            } else {
                for (let i = 0; i < irLength; i++) {
                    const lateIdx = i; // Late reverb starts from index 0 in its own buffer
                    const lateL = (lateIdx >= 0 && lateIdx < lateDiffuseL.length) ? lateDiffuseL[lateIdx] * lateReverbGain : 0;
                    const lateR = (lateIdx >= 0 && lateIdx < lateDiffuseR.length) ? lateDiffuseR[lateIdx] * lateReverbGain : 0;

                    if (i < crossfadeStartSample) {
                        // Fully early part - already in leftIR/rightIR
                        continue; // No change needed
                    } else if (i >= crossfadeStartSample && i < crossfadeEndSample) {
                        // Crossfade region
                        const fadePos = (i - crossfadeStartSample) / crossfadeDuration;
                        const earlyGain = 0.5 * (1 + Math.cos(fadePos * Math.PI)); // Cosine fade out for early
                        const diffuseGain = 0.5 * (1 - Math.cos(fadePos * Math.PI)); // Cosine fade in for late
                        leftIR[i] = leftIR[i] * earlyGain + lateL * diffuseGain;
                        rightIR[i] = rightIR[i] * earlyGain + lateR * diffuseGain;
                    } else {
                        // Fully late part
                        leftIR[i] = lateL;
                        rightIR[i] = lateR;
                    }
                }
            }

            this.sanitizeIRBuffers(leftIR, rightIR); // Restore call

            // --- Debug Log Start ---
            let finalEnergyL_dbg = 0;
            let finalEnergyR_dbg = 0;
            for(let i=0; i<leftIR.length; i++) finalEnergyL_dbg += leftIR[i]*leftIR[i];
            for(let i=0; i<rightIR.length; i++) finalEnergyR_dbg += rightIR[i]*rightIR[i];
            console.log(`[Debug] Total Energy After Combine/Sanitize - L: ${finalEnergyL_dbg.toExponential(2)}, R: ${finalEnergyR_dbg.toExponential(2)}`);
            // --- Debug Log End ---

            return [leftIR, rightIR]; // Restore return statement within try block

        } catch (error) {
            console.error('Error in processRayHitsInternal:', error);
            return [new Float32Array(irLength), new Float32Array(irLength)]; // Return empty on error
        }
    }

    // --- Helper methods (sanitizeIRBuffers, setupImpulseResponseBuffer, etc.) ---
    // Restore these methods

    private sanitizeIRBuffers(leftIR: Float32Array, rightIR: Float32Array): void {
        // Fix NaN/infinity
        for (let i = 0; i < leftIR.length; i++) {
            if (!isFinite(leftIR[i])) leftIR[i] = 0;
            if (!isFinite(rightIR[i])) rightIR[i] = 0;
        }

        // Find max value for normalization
        let maxValue = 0;
        for (let i = 0; i < leftIR.length; i++) {
            maxValue = Math.max(maxValue, Math.abs(leftIR[i]), Math.abs(rightIR[i]));
        }
        const targetPeak = 0.9;
        // Normalize if peak is too high or too low (adjust threshold if needed)
        // if (maxValue > 1.0 || (maxValue > 0 && maxValue < 0.01)) {
        if (maxValue > targetPeak) {
            const targetPeak = 0.9; // Target 90% peak
            const gainFactor = (maxValue > 0) ? targetPeak / maxValue : 1.0;
            console.log(`[Debug] Sanitize IR: MaxValue = ${maxValue.toExponential(3)}, TargetPeak = ${targetPeak}, GainFactor = ${gainFactor.toFixed(3)}`); // Log normalization values
            if (gainFactor !== 1.0) {
                 for (let i = 0; i < leftIR.length; i++) {
                     leftIR[i] *= gainFactor;
                     rightIR[i] *= gainFactor;
                 }
                 console.log(`Normalized IR buffers by factor ${gainFactor.toFixed(3)}, peak was ${maxValue.toFixed(3)}`);
            }
        }

        // Zero out the first sample to prevent potential DC offset clicks
        if (leftIR.length > 0) leftIR[0] = 0;
        if (rightIR.length > 0) rightIR[0] = 0;

        // Apply gentle fade-in/out (e.g., 5ms)
        const fadeSamples = Math.min(Math.floor(this.sampleRate * 0.005), 50); // Reduced fade-in/out back to 5ms
        if (fadeSamples > 0 && leftIR.length > fadeSamples * 2) {
            for (let i = 0; i < fadeSamples; i++) {
                const fadeGain = i / fadeSamples;
                leftIR[i] *= fadeGain;
                rightIR[i] *= fadeGain;
                const endIdx = leftIR.length - 1 - i;
                leftIR[endIdx] *= fadeGain;
                rightIR[endIdx] *= fadeGain;
            }
        }
    }

     private async setupImpulseResponseBuffer(leftIR: Float32Array, rightIR: Float32Array): Promise<void> {
        try {
            // Enhanced validation
            if (!leftIR || !rightIR) {
                console.error('IR buffers are null or undefined');
                this.impulseResponseBuffer = null;
                return;
            }

            if (leftIR.length === 0 || rightIR.length === 0) {
                console.error('IR buffers have zero length');
                this.impulseResponseBuffer = null;
                return;
            }

            if (leftIR.length !== rightIR.length) {
                console.error(`IR buffer length mismatch: L=${leftIR.length}, R=${rightIR.length}`);
                // Try to fix by padding the shorter one
                const maxLength = Math.max(leftIR.length, rightIR.length);
                if (leftIR.length < maxLength) {
                    const newLeftIR = new Float32Array(maxLength);
                    newLeftIR.set(leftIR);
                    leftIR = newLeftIR;
                    console.log('Left IR buffer padded to match right buffer length');
                } else {
                    const newRightIR = new Float32Array(maxLength);
                    newRightIR.set(rightIR);
                    rightIR = newRightIR;
                    console.log('Right IR buffer padded to match left buffer length');
                }
            }

            // Check for NaN or Infinity values
            let hasInvalidValues = false;
            for (let i = 0; i < leftIR.length; i++) {
                if (!isFinite(leftIR[i])) {
                    console.warn(`Invalid value in left IR at index ${i}: ${leftIR[i]}`);
                    leftIR[i] = 0;
                    hasInvalidValues = true;
                }
                if (!isFinite(rightIR[i])) {
                    console.warn(`Invalid value in right IR at index ${i}: ${rightIR[i]}`);
                    rightIR[i] = 0;
                    hasInvalidValues = true;
                }
            }

            if (hasInvalidValues) {
                console.warn('Fixed invalid values in IR buffers');
            }

            // Check if buffers have any non-zero content
            let hasContent = false;
            for (let i = 0; i < leftIR.length; i++) {
                if (Math.abs(leftIR[i]) > 1e-10 || Math.abs(rightIR[i]) > 1e-10) {
                    hasContent = true;
                    break;
                }
            }

            if (!hasContent) {
                console.warn('IR buffers contain only zeros or very small values, adding minimal impulse');
                // Add a minimal impulse to avoid silent IRs
                leftIR[0] = 0.1;
                rightIR[0] = 0.1;
            }

            const length = leftIR.length;
            // Check if context is running, resume if suspended
            if (this.audioCtx.state === 'suspended') {
                await this.audioCtx.resume();
            }

            this.impulseResponseBuffer = this.audioCtx.createBuffer(2, length, this.audioCtx.sampleRate);

            // Use copyToChannel for potentially better performance/safety
            this.impulseResponseBuffer.copyToChannel(leftIR, 0);
            this.impulseResponseBuffer.copyToChannel(rightIR, 1);

            console.log(`Impulse response buffer created/updated, length: ${length / this.sampleRate}s`);

        } catch (error) {
            console.error('Error setting up impulse response buffer:', error);
            this.impulseResponseBuffer = null; // Invalidate buffer on error

            // Create a minimal valid buffer as fallback
            try {
                const minLength = Math.ceil(0.5 * this.sampleRate); // 500ms minimum
                const fallbackBuffer = this.audioCtx.createBuffer(2, minLength, this.audioCtx.sampleRate);
                const leftChannel = fallbackBuffer.getChannelData(0);
                const rightChannel = fallbackBuffer.getChannelData(1);

                // Add a minimal impulse
                leftChannel[0] = 0.1;
                rightChannel[0] = 0.1;

                this.impulseResponseBuffer = fallbackBuffer;
                console.log('Created fallback impulse response buffer');
            } catch (fallbackError) {
                console.error('Failed to create fallback buffer:', fallbackError);
            }
        }
    }

    // Getter for the current IR buffer
    public getImpulseResponseBuffer(): AudioBuffer | null {
        return this.impulseResponseBuffer;
    }


    // --- Debug and Playback Methods ---
    // Restore these methods

    public async visualizeImpulseResponse(renderer: WaveformRenderer): Promise<void> {
        if (this.lastImpulseData) {
            await renderer.drawWaveformWithFFT(this.lastImpulseData);
        } else {
            console.warn('No impulse data available to visualize.');
        }
    }

    // Creates the necessary nodes for convolution but doesn't connect to destination or start
    // Assumes dryWetMix = 1.0 (fully wet) for simplicity now
    public createConvolvedSource(
        audioBufferToConvolve: AudioBuffer,
        impulseResponseBuffer: AudioBuffer // Expect IR buffer as argument
    ): { source: AudioBufferSourceNode, convolver: ConvolverNode, wetGain: GainNode } | null {
        if (this.audioCtx.state === 'suspended') {
             // Attempt to resume context if needed, but don't await here
             this.audioCtx.resume().catch(err => console.error("Error resuming audio context:", err));
        }

        try {
            const convolver = this.audioCtx.createConvolver();
            convolver.normalize = false; // Normalization is handled in sanitizeIRBuffers
            convolver.buffer = impulseResponseBuffer; // Use provided IR buffer

            const source = this.audioCtx.createBufferSource();
            source.buffer = audioBufferToConvolve;

            const wetGain = this.audioCtx.createGain();
            wetGain.gain.value = 1.0; // Fully wet signal

            // Connect source -> convolver -> wetGain
            source.connect(convolver);
            convolver.connect(wetGain);

            return { source, convolver, wetGain };

        } catch (error) {
            console.error('Error creating convolved source nodes:', error);
            return null;
        }
    }

    // Load an audio file and return its AudioBuffer
    public async loadAudioFile(url: string): Promise<AudioBuffer> {
        try {
            const response = await fetch(url);
            const arrayBuffer = await response.arrayBuffer();
            return await this.audioCtx.decodeAudioData(arrayBuffer);
        } catch (error) {
            console.error('Error loading audio file:', error);
            throw error;
        }
    }

    // Play an audio buffer with the current IR applied
    public async playAudioWithIR(audioBuffer: AudioBuffer): Promise<void> {
        // Stop any currently playing sound first
        this.stopAllSounds();

        if (!this.impulseResponseBuffer) {
            console.warn('No impulse response buffer available');
            return;
        }

        try {
            const nodes = this.createConvolvedSource(audioBuffer, this.impulseResponseBuffer);
            if (!nodes) {
                console.error('Failed to create audio nodes');
                return;
            }

            const { source, wetGain } = nodes;
            wetGain.connect(this.audioCtx.destination);

            // Track the source node and clear it when ended
            this.currentSourceNode = source;
            source.onended = () => {
                if (this.currentSourceNode === source) { // Ensure it's the same node
                    this.currentSourceNode = null;
                }
                // Disconnect nodes to allow garbage collection
                try {
                    wetGain.disconnect();
                    nodes.convolver.disconnect();
                    source.disconnect();
                } catch (e) { /* Ignore errors if already disconnected */ }
            };

            source.start(0);

        } catch (error) {
            console.error('Error playing audio with IR:', error);
            this.currentSourceNode = null; // Clear tracking on error
        }
    }

    // Stop the currently playing sound source
    public stopAllSounds(): void {
        if (this.currentSourceNode) {
            try {
                this.currentSourceNode.stop();
                // onended handler will disconnect and clear the node
            } catch (error) {
                console.error('Error stopping sound source:', error);
                // Force clear tracking if stop fails
                this.currentSourceNode = null;
            }
        }
    }


    // Removed playConvolvedClick as Main class now handles playback control


    // Removed playNoiseWithIR as it's replaced by GUI controls

    // --- Copied Debug/Helper Methods from Original AudioProcessor ---

    // Note: calculateDecayCurve uses this.lastRayHits now stored by processRayHits
    private calculateDecayCurve(numSamples: number): Float32Array {
        const decayCurve = new Float32Array(numSamples);
        const sampleRate = this.audioCtx.sampleRate;

        // Use lastRayHits if available, otherwise provide a default decay
        if (!this.lastRayHits || this.lastRayHits.length === 0) {
            console.warn('No ray hits available for decay curve calculation, using default.');
            // Simple exponential decay as fallback (e.g., 1 second RT60)
            const rt60 = 1.0;
            for (let i = 0; i < numSamples; i++) {
                const time = i / sampleRate;
                decayCurve[i] = Math.exp(-6.91 * time / rt60);
            }
            return decayCurve;
        }

        // Limit the number of ray hits to process for the curve
        const maxHits = Math.min(this.lastRayHits.length, 1000);
        const sortedHits = [...this.lastRayHits]
            .sort((a, b) => a.time - b.time)
            .slice(0, maxHits);

        if (sortedHits.length === 0) return decayCurve.fill(1.0); // Should not happen if lastRayHits is not empty

        const startTime = sortedHits[0].time;
        const endTime = sortedHits[sortedHits.length - 1].time;
        const timeRange = Math.max(endTime - startTime, 0.1); // Ensure minimum time range

        // Reduce number of bins for faster calculations
        const numBins = Math.min(100, numSamples);
        const binDuration = timeRange / numBins;
        const energyBins = new Array(numBins).fill(0);

        let maxBinEnergy = 0;

        // Process hits to fill energy bins
        for (const hit of sortedHits) {
             const binIndex = Math.floor((hit.time - startTime) / binDuration);
             if (binIndex >= 0 && binIndex < numBins) {
                 // Use a simplified energy calculation for performance
                 const totalEnergy = Object.values(hit.energies).reduce((sum: number, e) => sum + (typeof e === 'number' ? e : 0), 0);
                 energyBins[binIndex] += totalEnergy;
                 maxBinEnergy = Math.max(maxBinEnergy, energyBins[binIndex]);
             }
        }


        // Normalize and smooth the energy bins (simple moving average)
        const smoothedBins = new Array(numBins).fill(0);
        if (maxBinEnergy > 0) {
            const scale = 1.0 / maxBinEnergy;
            for (let i = 0; i < numBins; i++) {
                // Apply smoothing (e.g., average of 3 bins)
                const prev = energyBins[i - 1] || energyBins[i];
                const next = energyBins[i + 1] || energyBins[i];
                smoothedBins[i] = Math.sqrt(((prev + energyBins[i] + next) / 3) * scale); // Use sqrt for perceptual loudness
            }
        } else {
             // If no energy, return flat curve (or decay)
             return this.calculateDecayCurve(numSamples); // Recurse with fallback
        }


        // Apply the smoothed curve to the output buffer
        for (let i = 0; i < numSamples; i++) {
            const timeInSeconds = i / sampleRate;
            const relativeBinPosition = (timeInSeconds - startTime) / binDuration;
            const binIndex = Math.floor(relativeBinPosition);

            if (binIndex < 0) {
                decayCurve[i] = smoothedBins[0] || 1.0; // Start with first bin value
            } else if (binIndex >= numBins - 1) {
                decayCurve[i] = smoothedBins[numBins - 1] || 0; // End with last bin value
            } else {
                // Linear interpolation between bins
                const fraction = relativeBinPosition - binIndex;
                decayCurve[i] = smoothedBins[binIndex] * (1 - fraction) + smoothedBins[binIndex + 1] * fraction;
            }
        }

        return decayCurve;
    }


    public async debugPlaySineWave(): Promise<void> {
        try {
            if (this.audioCtx.state === 'suspended') { await this.audioCtx.resume(); }
            const duration = 2;
            const frequency = 440;
            const sampleRate = this.audioCtx.sampleRate;
            const samples = duration * sampleRate;

            const buffer = this.audioCtx.createBuffer(1, samples, sampleRate);
            const channelData = buffer.getChannelData(0);

            for (let i = 0; i < samples; i++) {
                channelData[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate) * 0.5; // Reduce amplitude slightly
            }

            const source = this.audioCtx.createBufferSource();
            source.buffer = buffer;
            source.connect(this.audioCtx.destination);
            source.start();
        } catch (error) {
            console.error("Error playing sine wave:", error);
        }
    }

    // Removed playConvolvedSineWave as it's replaced by GUI controls

    // Add other necessary methods from original if needed...

    // --- Helper Functions for Gain + ITD Model ---
    // Moved inside class as private methods

    // New Balanced Gain Function
    private calculateBalancedSpatialGains(
        azimuthRad: number,
        elevationRad: number,
        distance: number
    ): [number, number] {
        const pi = Math.PI;
        const piOver2 = pi / 2;

        // Base gain using sine relationship for smooth panning
        // Ensure azimuth is in [-PI, PI]
        const clampedAzimuth = Math.max(-pi, Math.min(pi, azimuthRad));
        const sinAz = Math.sin(clampedAzimuth);

        // Scale gains based on sine, aiming for roughly constant power feel
        const baseGain = 0.707; // sqrt(0.5)
        let leftGain = baseGain * (1 - sinAz * 0.8); // Modulate around base gain
        let rightGain = baseGain * (1 + sinAz * 0.8);

        // Apply elevation effects
        const elevationFactor = 1.0 - Math.abs(elevationRad) / piOver2 * 0.3;
        leftGain *= elevationFactor;
        rightGain *= elevationFactor;

        // Apply distance attenuation
        const distanceAtten = 1.0 / Math.max(1, distance);
        leftGain *= distanceAtten;
        rightGain *= distanceAtten;

        // Apply front-back reduction
        if (Math.abs(clampedAzimuth) > piOver2) {
            const backFactor = 0.8;
            leftGain *= backFactor;
            rightGain *= backFactor;
        }

        // Clamp to prevent negative or overly large gains after all factors
        leftGain = Math.max(0, Math.min(1.5, leftGain));
        rightGain = Math.max(0, Math.min(1.5, rightGain));

        return [leftGain, rightGain];
    }

    // ITD Calculation
    private calculateITDsamples(azimuthRad: number, sampleRate: number): number {
        const headRadius = 0.0875; // Approximate head radius in meters
        const speedOfSound = 343; // m/s
        // Woodworth's formula approximation for ITD
        // Ensure azimuthRad is within +/- PI for Math.sin
        const clampedAzimuth = Math.max(-Math.PI, Math.min(Math.PI, azimuthRad));
        const itdSeconds = (headRadius / speedOfSound) * (clampedAzimuth + Math.sin(clampedAzimuth));
        // Clamp ITD to a reasonable maximum (e.g., ~0.7ms)
        const clampedITD = Math.max(-0.0007, Math.min(0.0007, itdSeconds));
        return Math.round(clampedITD * sampleRate);
    }
}

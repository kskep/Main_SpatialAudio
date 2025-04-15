// src/sound/audio-processor_modified.ts
// Integrates HRTFProcessor and dynamic ray hit processing for realism

import { Camera } from '../camera/camera';
import { Room } from '../room/room';
import { WaveformRenderer } from '../visualization/waveform-renderer';
import { RayHit } from '../raytracer/raytracer';
// Removed: import { DiffuseFieldModelModified } from './diffuse-field-model_modified'; // No longer using separate diffuse model
import { vec3 } from 'gl-matrix';

export class AudioProcessorModified {
    private audioCtx: AudioContext;
    private room: Room;
    private camera: Camera;
    // Removed: private diffuseFieldModel: DiffuseFieldModelModified;
    private impulseResponseBuffer: AudioBuffer | null = null;
    private lastImpulseData: Float32Array | null = null;
    private sampleRate: number;
    private lastRayHits: RayHit[] = [];
    private currentSourceNode: AudioBufferSourceNode | null = null;

    // Add allPass filter states for late reverb decorrelation
    private allPassStateL = { x: new Float32Array(4), y: new Float32Array(4), d: 3, g: 0.5 };
    private allPassStateR = { x: new Float32Array(6), y: new Float32Array(6), d: 5, g: 0.45 };

    // Constants for dynamic transition calculation
    private readonly MIN_SPACING_THRESHOLD_MS = 1.5; // Time spacing threshold in milliseconds
    private readonly ANALYSIS_WINDOW_MS = 15;       // Time window for density analysis in milliseconds
    private readonly MIN_HITS_IN_WINDOW = 5;        // Minimum hits required in the window for stable analysis
    private readonly MAX_TRANSITION_TIME_S = 0.5;   // Max time to search for transition before using default


    constructor(audioCtx: AudioContext, room: Room, camera: Camera, sampleRate: number) {
        this.audioCtx = audioCtx;
        this.room = room;
        this.camera = camera;
        this.sampleRate = sampleRate;

        // Removed DiffuseFieldModel instantiation
        // this.diffuseFieldModel = new DiffuseFieldModelModified(this.sampleRate, roomConfigForModel);
        console.log("AudioProcessorModified initialized (using dynamic ray processing)");
    }

    async processRayHits(
        rayHits: RayHit[],
    ): Promise<void> {
        try {
            if (!rayHits || !Array.isArray(rayHits) || rayHits.length === 0) {
                console.warn('No valid ray hits to process'); return;
            }
            // Removed check for diffuseFieldModel initialization

            const validHits = rayHits.filter(hit => hit && hit.position && hit.energies && isFinite(hit.time));
            if (validHits.length === 0) {
                console.warn('No valid ray hits after filtering');
                return;
            }

            this.lastRayHits = validHits; // Store the valid hits

            // Generate IR directly from ray hits using dynamic transition
            const [leftIR, rightIR] = this.processRayHitsInternal(validHits);

            // Create interleaved stereo data for visualization
            const stereoData = new Float32Array(leftIR.length * 2);
            for (let i = 0; i < leftIR.length; i++) {
                stereoData[i * 2] = leftIR[i];
                stereoData[i * 2 + 1] = rightIR[i];
            }
            this.lastImpulseData = stereoData;

            // Set up the final impulse response buffer
            await this.setupImpulseResponseBuffer(leftIR, rightIR);
        } catch (error) {
            console.error('Error processing ray hits:', error);
            throw error;
        }
    }

    private processRayHitsInternal(hits: RayHit[]): [Float32Array, Float32Array] {
        const irLength = Math.max(Math.ceil(this.sampleRate * 2), 1000); // At least 2 seconds or 1000 samples
        console.log(`[Debug] Creating IR buffers with length: ${irLength} samples (${irLength/this.sampleRate}s)`);

        const leftIR = new Float32Array(irLength);
        const rightIR = new Float32Array(irLength);

        if (hits.length < 5) { // Need some hits to process meaningfully
            console.warn("Not enough ray hits to process (< 5). Returning empty IR.");
            return [leftIR, rightIR];
        }

        try {
            const sortedHits = [...hits].sort((a, b) => a.time - b.time);

            // --- Determine Dynamic Transition Time ---
            let transitionTime = 0.1; // Default fallback if density threshold not met
            const analysisWindowSamples = Math.floor(this.ANALYSIS_WINDOW_MS / 1000 * this.sampleRate);
            let cumulativeSpacing = 0;
            let hitsInWindow = 0;
            let windowStartIndex = 0;
            let transitionFound = false;

            console.log(`[Debug Transition] Starting analysis. Threshold: ${this.MIN_SPACING_THRESHOLD_MS}ms, Window: ${this.ANALYSIS_WINDOW_MS}ms`);

            // Start analysis from the second hit
            for (let i = 1; i < sortedHits.length; i++) {
                const currentTime = sortedHits[i].time;
                const deltaTime = currentTime - sortedHits[i - 1].time;

                // Update sliding window: Remove hits falling out of the window from the start
                while (windowStartIndex < i && sortedHits[windowStartIndex].time < currentTime - (this.ANALYSIS_WINDOW_MS / 1000)) {
                    if (windowStartIndex + 1 < i) { // Ensure we have a delta to remove
                         const removedDeltaTime = sortedHits[windowStartIndex + 1].time - sortedHits[windowStartIndex].time;
                         if (hitsInWindow > 0) { // Avoid subtracting if window was empty
                            cumulativeSpacing -= removedDeltaTime;
                            hitsInWindow--;
                         }
                    }
                    windowStartIndex++;
                }

                // Add current delta time to window stats if valid
                if(deltaTime >= 0) { // Basic sanity check for delta time
                    cumulativeSpacing += deltaTime;
                    hitsInWindow++;
                } else {
                    console.warn(`[Debug Transition] Negative deltaTime detected at index ${i}: ${deltaTime}`);
                }


                // Check criterion if window is sufficiently populated
                if (hitsInWindow >= this.MIN_HITS_IN_WINDOW) {
                    const averageSpacing = cumulativeSpacing / hitsInWindow;
                    if (averageSpacing >= 0 && (averageSpacing * 1000) < this.MIN_SPACING_THRESHOLD_MS) {
                        transitionTime = currentTime; // Found the transition point
                        console.log(`[Debug Transition] Found transition at ${transitionTime.toFixed(4)}s. Avg spacing: ${(averageSpacing * 1000).toFixed(2)}ms with ${hitsInWindow} hits in window.`);
                        transitionFound = true;
                        break; // Stop searching once density is reached
                    }
                }

                 // Ensure loop terminates if no transition found within reasonable time
                 if (!transitionFound && currentTime > this.MAX_TRANSITION_TIME_S) {
                     console.warn(`[Debug Transition] Did not find transition point below ${this.MIN_SPACING_THRESHOLD_MS}ms spacing before ${this.MAX_TRANSITION_TIME_S}s. Using default: ${transitionTime}s`);
                     break; // Use default transition time
                 }
            }
            // Handle case where loop finishes without finding transition but before MAX_TRANSITION_TIME_S
             if (!transitionFound) {
                 console.warn(`[Debug Transition] Reached end of hits without meeting density criteria. Using default: ${transitionTime}s`);
             }

            // --- Process Early Hits (Before Transition Time) ---
            const earlyHits = sortedHits.filter(hit => hit.time < transitionTime);
            console.log(`[Debug Early Hits] Processing ${earlyHits.length} hits before transition time ${transitionTime.toFixed(4)}s using Gain+ITD model.`);
            const earlyLoudnessScale = 0.15; // Keep loudness scale for early hits

            for (const hit of earlyHits) {
                const sampleIndex = Math.floor(hit.time * this.sampleRate);
                if (sampleIndex < 0 || sampleIndex >= irLength || !isFinite(sampleIndex)) continue;

                // 1. Calculate Amplitude
                const totalEnergy = Object.values(hit.energies).reduce((sum: number, e) => sum + (typeof e === 'number' ? e : 0), 0);
                let amplitude = Math.sqrt(Math.max(0, totalEnergy)) * Math.exp(-hit.bounces * 0.2) * earlyLoudnessScale; // Apply bounce attenuation and scale
                amplitude = Math.max(0, Math.min(1.0, amplitude));
                if (!isFinite(amplitude) || amplitude < 1e-6) continue;

                // 2. Calculate Direction & Spatialization Params
                const listenerPos = this.camera.getPosition();
                const direction = vec3.create();
                vec3.subtract(direction, hit.position, listenerPos);
                const distance = vec3.length(direction);
                vec3.normalize(direction, direction);
                const listenerRight = this.camera.getRight();
                const listenerFront = this.camera.getFront();
                const listenerUp = this.camera.getUp();
                const dotRight = vec3.dot(direction, listenerRight);
                const dotFront = vec3.dot(direction, listenerFront);
                const dotUp = vec3.dot(direction, listenerUp);
                const azimuthRad = Math.atan2(dotRight, dotFront);
                const elevationRad = Math.asin(Math.max(-1, Math.min(1, dotUp)));

                // 3. Apply Gain + ITD Model
                let [leftGain, rightGain] = this.calculateBalancedSpatialGains(azimuthRad, elevationRad, distance);
                if (!isFinite(leftGain) || !isFinite(rightGain)) { leftGain = 0; rightGain = 0; }
                let itd_samples = this.calculateITDsamples(azimuthRad, this.sampleRate);
                if (!isFinite(itd_samples)) { itd_samples = 0; }

                let leftDelaySamples = 0;
                let rightDelaySamples = 0;
                // --- Re-enable ITD (adjust if needed) ---
                if (itd_samples > 0) { rightDelaySamples = itd_samples; }
                else if (itd_samples < 0) { leftDelaySamples = -itd_samples; }
                // ---

                // 4. Add to IR Buffers
                const baseLeftIndex = sampleIndex + leftDelaySamples;
                const baseRightIndex = sampleIndex + rightDelaySamples;
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
            }

            // --- Process Late Hits (At and After Transition Time) ---
            const lateHits = sortedHits.filter(hit => hit.time >= transitionTime);
            console.log(`[Debug Late Hits] Processing ${lateHits.length} hits from transition time ${transitionTime.toFixed(4)}s using simplified model.`);
            const lateGainFactor = 0.001; // Significantly reduced gain for late hits

            for (const hit of lateHits) {
                const sampleIndex = Math.floor(hit.time * this.sampleRate);
                if (sampleIndex < 0 || sampleIndex >= irLength || !isFinite(sampleIndex)) continue;

                // Apply time-dependent decay
                const timeSinceTransition = Math.max(0, hit.time - transitionTime);
                const timeDecayFactor = Math.exp(-timeSinceTransition * 5.0);
                
                // Apply bounce-dependent attenuation
                const bounceDecayFactor = Math.exp(-hit.bounces * 0.3);

                // Calculate frequency-weighted energy
                const energies = hit.energies;
                const weightedEnergy = (
                    (energies.energy125Hz || 0) * 0.5 +  // Reduce low freqs
                    (energies.energy250Hz || 0) * 0.7 +
                    (energies.energy500Hz || 0) * 0.9 +
                    (energies.energy1kHz || 0) * 1.0 +   // Reference
                    (energies.energy2kHz || 0) * 1.0 +
                    (energies.energy4kHz || 0) * 0.9 +
                    (energies.energy8kHz || 0) * 0.8 +
                    (energies.energy16kHz || 0) * 0.7    // Reduce high freqs
                ) / 8.0;

                let baseAmplitude = Math.sqrt(Math.max(0, weightedEnergy)) * 
                                  lateGainFactor * 
                                  timeDecayFactor * 
                                  bounceDecayFactor;
                
                baseAmplitude = Math.max(0, Math.min(1.0, baseAmplitude));
                if (!isFinite(baseAmplitude) || baseAmplitude < 1e-6) continue;

                // Basic direction-based gains
                const listenerPos = this.camera.getPosition();
                const direction = vec3.create();
                vec3.subtract(direction, hit.position, listenerPos);
                const listenerRight = this.camera.getRight();
                const listenerFront = this.camera.getFront();
                const dotRight = vec3.dot(direction, listenerRight);
                const dotFront = vec3.dot(direction, listenerFront);
                const azimuthRad = Math.atan2(dotRight, dotFront);

                // Calculate basic ITD for late reflections
                let itd_samples = this.calculateITDsamples(azimuthRad, this.sampleRate);
                if (!isFinite(itd_samples)) itd_samples = 0;

                // Simple balanced panning
                const angle = (azimuthRad + Math.PI / 2) * 0.5;
                const leftGain = Math.max(0, Math.cos(angle));
                const rightGain = Math.max(0, Math.sin(angle));

                // Apply all-pass filters for decorrelation
                const inputL = baseAmplitude * leftGain;
                const inputR = baseAmplitude * rightGain;

                // Apply All-Pass L
                for (let k = this.allPassStateL.d; k > 0; k--) {
                    this.allPassStateL.x[k] = this.allPassStateL.x[k-1];
                    this.allPassStateL.y[k] = this.allPassStateL.y[k-1];
                }
                this.allPassStateL.x[0] = inputL;
                const allPassOutputL = this.allPassStateL.g * inputL + 
                                     this.allPassStateL.x[this.allPassStateL.d] - 
                                     this.allPassStateL.g * this.allPassStateL.y[this.allPassStateL.d];
                this.allPassStateL.y[0] = allPassOutputL;

                // Apply All-Pass R
                for (let k = this.allPassStateR.d; k > 0; k--) {
                    this.allPassStateR.x[k] = this.allPassStateR.x[k-1];
                    this.allPassStateR.y[k] = this.allPassStateR.y[k-1];
                }
                this.allPassStateR.x[0] = inputR;
                const allPassOutputR = this.allPassStateR.g * inputR + 
                                     this.allPassStateR.x[this.allPassStateR.d] - 
                                     this.allPassStateR.g * this.allPassStateR.y[this.allPassStateR.d];
                this.allPassStateR.y[0] = allPassOutputR;

                // Add filtered outputs to IR with ITD
                const leftIndex = sampleIndex + (itd_samples > 0 ? 0 : -itd_samples);
                const rightIndex = sampleIndex + (itd_samples > 0 ? itd_samples : 0);

                if (leftIndex >= 0 && leftIndex < irLength) {
                    leftIR[leftIndex] += allPassOutputL;
                }
                if (rightIndex >= 0 && rightIndex < irLength) {
                    rightIR[rightIndex] += allPassOutputR;
                }
            }

            // --- No Crossfade Needed ---
            // Both early and late hits are added to the same buffers.

            // --- Sanitize and Return ---
            this.sanitizeIRBuffers(leftIR, rightIR);

            // --- Debug Log Start ---
            let finalEnergyL_dbg = 0;
            let finalEnergyR_dbg = 0;
            for(let i=0; i<leftIR.length; i++) finalEnergyL_dbg += leftIR[i]*leftIR[i];
            for(let i=0; i<rightIR.length; i++) finalEnergyR_dbg += rightIR[i]*rightIR[i];
            console.log(`[Debug] Total Energy After Dynamic Processing/Sanitize - L: ${finalEnergyL_dbg.toExponential(2)}, R: ${finalEnergyR_dbg.toExponential(2)}`);
            // --- Debug Log End ---

            return [leftIR, rightIR];

        } catch (error) {
            console.error('Error in processRayHitsInternal:', error);
            return [new Float32Array(irLength), new Float32Array(irLength)]; // Return empty on error
        }
    }

    // --- Helper methods (sanitizeIRBuffers, setupImpulseResponseBuffer, etc.) ---

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
        const targetPeak = 0.9; // Target peak level

        // Only normalize if peak is too high
        if (maxValue > targetPeak) {
            const gainFactor = targetPeak / maxValue;
            console.log(`[Debug Sanitize DOWN] MaxValue: ${maxValue.toExponential(3)} > TargetPeak: ${targetPeak}. Normalizing down with GainFactor: ${gainFactor.toFixed(3)}`);
            for (let i = 0; i < leftIR.length; i++) {
                leftIR[i] *= gainFactor;
                rightIR[i] *= gainFactor;
            }
        } else {
             console.log(`[Debug Sanitize DOWN] No down-normalization needed. MaxValue: ${maxValue.toExponential(3)} <= TargetPeak: ${targetPeak}`);
        }


        // Zero out the first sample to prevent potential DC offset clicks
        if (leftIR.length > 0) leftIR[0] = 0;
        if (rightIR.length > 0) rightIR[0] = 0;

        // Apply gentle fade-in/out (e.g., 5ms)
        const fadeSamples = Math.min(Math.floor(this.sampleRate * 0.005), 50);
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
                this.impulseResponseBuffer = null; return;
            }
            if (leftIR.length === 0 || rightIR.length === 0) {
                console.error('IR buffers have zero length');
                this.impulseResponseBuffer = null; return;
            }
            if (leftIR.length !== rightIR.length) {
                console.error(`IR buffer length mismatch: L=${leftIR.length}, R=${rightIR.length}`);
                const maxLength = Math.max(leftIR.length, rightIR.length);
                if (leftIR.length < maxLength) {
                    const newLeftIR = new Float32Array(maxLength); newLeftIR.set(leftIR); leftIR = newLeftIR;
                    console.log('Left IR buffer padded to match right buffer length');
                } else {
                    const newRightIR = new Float32Array(maxLength); newRightIR.set(rightIR); rightIR = newRightIR;
                    console.log('Right IR buffer padded to match left buffer length');
                }
            }

            // Check for NaN or Infinity values
            let hasInvalidValues = false;
            for (let i = 0; i < leftIR.length; i++) {
                if (!isFinite(leftIR[i])) { leftIR[i] = 0; hasInvalidValues = true; }
                if (!isFinite(rightIR[i])) { rightIR[i] = 0; hasInvalidValues = true; }
            }
            if (hasInvalidValues) console.warn('Fixed invalid values in IR buffers');

            // Check if buffers have any non-zero content
            let hasContent = false;
            for (let i = 0; i < leftIR.length; i++) {
                if (Math.abs(leftIR[i]) > 1e-10 || Math.abs(rightIR[i]) > 1e-10) { hasContent = true; break; }
            }
            if (!hasContent) {
                console.warn('IR buffers contain only zeros or very small values, adding minimal impulse');
                leftIR[0] = 0.01; rightIR[0] = 0.01; // Add tiny impulse
            }

            const length = leftIR.length;
            if (this.audioCtx.state === 'suspended') await this.audioCtx.resume();

            this.impulseResponseBuffer = this.audioCtx.createBuffer(2, length, this.audioCtx.sampleRate);
            this.impulseResponseBuffer.copyToChannel(leftIR, 0);
            this.impulseResponseBuffer.copyToChannel(rightIR, 1);
            console.log(`Impulse response buffer created/updated, length: ${length / this.sampleRate}s`);

        } catch (error) {
            console.error('Error setting up impulse response buffer:', error);
            this.impulseResponseBuffer = null;
             try { // Fallback buffer
                const minLength = Math.ceil(0.1 * this.sampleRate); // 100ms minimum
                const fallbackBuffer = this.audioCtx.createBuffer(2, minLength, this.audioCtx.sampleRate);
                fallbackBuffer.getChannelData(0)[0] = 0.1; fallbackBuffer.getChannelData(1)[0] = 0.1;
                this.impulseResponseBuffer = fallbackBuffer; console.log('Created fallback impulse response buffer');
            } catch (fallbackError) { console.error('Failed to create fallback buffer:', fallbackError); }
        }
    }

    // Getter for the current IR buffer
    public getImpulseResponseBuffer(): AudioBuffer | null {
        return this.impulseResponseBuffer;
    }

    // --- Debug and Playback Methods ---

    public async visualizeImpulseResponse(renderer: WaveformRenderer): Promise<void> {
        if (this.lastImpulseData) {
            await renderer.drawWaveformWithFFT(this.lastImpulseData);
        } else {
            console.warn('No impulse data available to visualize.');
        }
    }

    public createConvolvedSource(
        audioBufferToConvolve: AudioBuffer,
        impulseResponseBuffer: AudioBuffer
    ): { source: AudioBufferSourceNode, convolver: ConvolverNode, wetGain: GainNode } | null {
        if (this.audioCtx.state === 'suspended') {
             this.audioCtx.resume().catch(err => console.error("Error resuming audio context:", err));
        }
        try {
            const convolver = this.audioCtx.createConvolver();
            convolver.normalize = false; // Normalization handled in sanitizeIRBuffers
            convolver.buffer = impulseResponseBuffer;

            const source = this.audioCtx.createBufferSource();
            source.buffer = audioBufferToConvolve;

            const wetGain = this.audioCtx.createGain();
            wetGain.gain.value = 1.0; // Fully wet signal

            source.connect(convolver);
            convolver.connect(wetGain);
            return { source, convolver, wetGain };
        } catch (error) {
            console.error('Error creating convolved source nodes:', error);
            return null;
        }
    }

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

    public async playAudioWithIR(audioBuffer: AudioBuffer): Promise<void> {
        this.stopAllSounds();
        if (!this.impulseResponseBuffer) { console.warn('No impulse response buffer available'); return; }
        try {
            const nodes = this.createConvolvedSource(audioBuffer, this.impulseResponseBuffer);
            if (!nodes) { console.error('Failed to create audio nodes'); return; }
            const { source, wetGain } = nodes;
            wetGain.connect(this.audioCtx.destination);
            this.currentSourceNode = source;
            source.onended = () => {
                if (this.currentSourceNode === source) this.currentSourceNode = null;
                try { wetGain.disconnect(); nodes.convolver.disconnect(); source.disconnect(); }
                catch (e) { /* Ignore */ }
            };
            source.start(0);
        } catch (error) {
            console.error('Error playing audio with IR:', error);
            this.currentSourceNode = null;
        }
    }

    public async playAudioWithoutIR(audioBuffer: AudioBuffer): Promise<void> {
        this.stopAllSounds();
        try {
            const source = this.audioCtx.createBufferSource();
            source.buffer = audioBuffer;
            const gainNode = this.audioCtx.createGain();
            gainNode.gain.value = 0.7; // Slight reduction to match IR volume
            
            source.connect(gainNode);
            gainNode.connect(this.audioCtx.destination);
            
            this.currentSourceNode = source;
            source.onended = () => {
                if (this.currentSourceNode === source) {
                    this.currentSourceNode = null;
                }
                try {
                    gainNode.disconnect();
                    source.disconnect();
                } catch (e) { /* Ignore */ }
            };
            source.start(0);
        } catch (error) {
            console.error('Error playing unprocessed audio:', error);
            this.currentSourceNode = null;
        }
    }

    public stopAllSounds(): void {
        if (this.currentSourceNode) {
            try { this.currentSourceNode.stop(); }
            catch (error) { console.error('Error stopping sound source:', error); }
            // onended handler should clear/disconnect
            this.currentSourceNode = null; // Force clear just in case
        }
    }

    // --- Helper Functions for Gain + ITD Model ---

    private calculateBalancedSpatialGains(azimuthRad: number, elevationRad: number, distance: number): [number, number] {
        const pi = Math.PI;
        const piOver2 = pi / 2;
        const clampedAzimuth = Math.max(-pi, Math.min(pi, azimuthRad));
        const sinAz = Math.sin(clampedAzimuth);
        const baseGain = 0.707; // sqrt(0.5) for approx constant power
        let leftGain = baseGain * (1 - sinAz * 0.8); // Modulate gain based on azimuth
        let rightGain = baseGain * (1 + sinAz * 0.8);

        // Apply elevation attenuation (sounds quieter when above/below)
        const elevationFactor = 1.0 - Math.abs(elevationRad) / piOver2 * 0.3; // Reduce gain by up to 30% at poles
        leftGain *= elevationFactor;
        rightGain *= elevationFactor;

        // Apply distance attenuation (1/distance)
        const distanceAtten = 1.0 / Math.max(1, distance); // Avoid division by zero, start attenuation at 1m
        leftGain *= distanceAtten;
        rightGain *= distanceAtten;

        // Apply front-back attenuation (sounds quieter from behind)
        if (Math.abs(clampedAzimuth) > piOver2) {
            const backFactor = 0.8; // Reduce gain by 20% when behind
            leftGain *= backFactor;
            rightGain *= backFactor;
        }

        // Final clamp to prevent extreme values
        leftGain = Math.max(0, Math.min(1.5, leftGain));
        rightGain = Math.max(0, Math.min(1.5, rightGain));
        return [leftGain, rightGain];
    }

    private calculateITDsamples(azimuthRad: number, sampleRate: number): number {
        const headRadius = 0.0875; // meters
        const speedOfSound = 343; // m/s
        const clampedAzimuth = Math.max(-Math.PI, Math.min(Math.PI, azimuthRad));
        // Woodworth's formula approximation
        const itdSeconds = (headRadius / speedOfSound) * (clampedAzimuth + Math.sin(clampedAzimuth));
        const clampedITD = Math.max(-0.0007, Math.min(0.0007, itdSeconds)); // Clamp to +/- 0.7ms
        return Math.round(clampedITD * sampleRate);
    }

    // Removed calculateDecayCurve as it's less relevant now
    // Removed debugPlaySineWave etc. as playback is handled via GUI
}
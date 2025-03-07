import { vec3 } from 'gl-matrix';
import { Camera } from '../camera/camera';

export class SpatialAudioProcessor {
    private sampleRate: number;
    private hrtfEnabled = false;
    private hrtfFilters: Map<string, Float32Array[]> = new Map();

    constructor(sampleRate: number = 44100) {
        this.sampleRate = sampleRate;
        this.initializeHRTF();
    }

    private initializeHRTF(): void {
        try {
            // Generate HRTF filters for key directions
            const directions = [
                { azimuth: 0, elevation: 0 },    // Front
                { azimuth: 90, elevation: 0 },   // Right
                { azimuth: 180, elevation: 0 },  // Back
                { azimuth: 270, elevation: 0 },  // Left
                { azimuth: 0, elevation: 45 },   // Above
                { azimuth: 0, elevation: -45 }   // Below
            ];
            
            for (const dir of directions) {
                const [leftFilter, rightFilter] = this.generateHRTFFilters(dir.azimuth, dir.elevation);
                this.hrtfFilters.set(`${dir.azimuth}_${dir.elevation}`, [leftFilter, rightFilter]);
            }
            
            console.log("Initialized HRTF filters");
            this.hrtfEnabled = true;
        } catch (error) {
            console.error("Failed to initialize HRTF filters:", error);
            this.hrtfEnabled = false;
        }
    }
    
    private generateHRTFFilters(azimuth: number, elevation: number): [Float32Array, Float32Array] {
        const filterLength = 128;
        const leftFilter = new Float32Array(filterLength);
        const rightFilter = new Float32Array(filterLength);
        
        // Convert angles to radians
        const azimuthRad = azimuth * Math.PI / 180;
        const elevationRad = elevation * Math.PI / 180;
        
        // Generate HRTF filter coefficients
        for (let i = 0; i < filterLength; i++) {
            const t = i / filterLength;
            
            // Basic head shadow and pinna effects
            const headShadow = Math.exp(-t * 8) * (1 - Math.abs(azimuthRad) / Math.PI);
            const pinnaEffect = Math.exp(-t * 4) * (1 - Math.abs(elevationRad) / (Math.PI/2));
            
            // Left ear response
            const leftPhase = -azimuthRad + Math.PI/4;
            leftFilter[i] = headShadow * pinnaEffect * Math.cos(2 * Math.PI * t + leftPhase);
            
            // Right ear response
            const rightPhase = azimuthRad + Math.PI/4;
            rightFilter[i] = headShadow * pinnaEffect * Math.cos(2 * Math.PI * t + rightPhase);
        }
        
        return [leftFilter, rightFilter];
    }

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
        
        // Calculate azimuth (horizontal angle)
        const dotRight = vec3.dot(direction, listenerRight);
        const dotFront = vec3.dot(direction, listenerFront);
        const azimuth = Math.atan2(dotRight, dotFront);
        
        // Calculate elevation (vertical angle)
        const dotUp = vec3.dot(direction, listenerUp);
        const elevation = Math.asin(Math.max(-1, Math.min(1, dotUp)));
        
        // Base gains using spherical head model
        let leftGain = 0.5, rightGain = 0.5;
        
        if (azimuth < 0) { // Source is to the left
            leftGain = 0.9 - 0.4 * azimuth/Math.PI;
            rightGain = 0.4 + 0.5 * (1 + azimuth/Math.PI);
        } else { // Source is to the right
            leftGain = 0.4 + 0.5 * (1 - azimuth/Math.PI);
            rightGain = 0.9 + 0.4 * azimuth/Math.PI;
        }
        
        // Apply elevation effects
        const elevationFactor = 1.0 - Math.abs(elevation) / (Math.PI/2) * 0.3;
        leftGain *= elevationFactor;
        rightGain *= elevationFactor;
        
        // Apply distance attenuation
        const distanceAtten = 1.0 / Math.max(1, distance);
        leftGain *= distanceAtten;
        rightGain *= distanceAtten;
        
        // Apply front-back disambiguation
        if (Math.abs(azimuth) > Math.PI/2) {
            const backFactor = 0.8;
            leftGain *= backFactor;
            rightGain *= backFactor;
        }
        
        return [leftGain, rightGain];
    }
}
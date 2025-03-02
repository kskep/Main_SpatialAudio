export interface RoomMaterials {
    left:   WallMaterial;
    right:  WallMaterial;
    top:    WallMaterial;
    bottom: WallMaterial;
    front:  WallMaterial;
    back:   WallMaterial;
}

export interface WallMaterial {
    // Absorption coefficients for 8 frequency bands
    absorption125Hz: number;  // 125 Hz
    absorption250Hz: number;  // 250 Hz
    absorption500Hz: number;  // 500 Hz
    absorption1kHz: number;   // 1000 Hz
    absorption2kHz: number;   // 2000 Hz
    absorption4kHz: number;   // 4000 Hz
    absorption8kHz: number;   // 8000 Hz
    absorption16kHz: number;  // 16000 Hz

    // Scattering coefficients for 8 frequency bands
    scattering125Hz: number;
    scattering250Hz: number;
    scattering500Hz: number;
    scattering1kHz: number;
    scattering2kHz: number;
    scattering4kHz: number;
    scattering8kHz: number;
    scattering16kHz: number;

    roughness: number;       // 0-1, affects reflection pattern
    phaseShift: number;      // Fixed phase shift on reflection (radians)
    phaseRandomization: number; // Max random phase variation (radians)
}

// Add standard material presets with more detailed frequency response
export const MATERIAL_PRESETS = {
    CONCRETE: {
        absorption125Hz: 0.01,
        absorption250Hz: 0.02,
        absorption500Hz: 0.02,
        absorption1kHz: 0.03,
        absorption2kHz: 0.03,
        absorption4kHz: 0.04,
        absorption8kHz: 0.05,
        absorption16kHz: 0.06,

        scattering125Hz: 0.10,
        scattering250Hz: 0.12,
        scattering500Hz: 0.14,
        scattering1kHz: 0.15,
        scattering2kHz: 0.18,
        scattering4kHz: 0.20,
        scattering8kHz: 0.22,
        scattering16kHz: 0.25,

        roughness: 0.1,
        phaseShift: 0,
        phaseRandomization: 0.1
    },

    WOOD: {
        absorption125Hz: 0.15,
        absorption250Hz: 0.13,
        absorption500Hz: 0.10,
        absorption1kHz: 0.09,
        absorption2kHz: 0.08,
        absorption4kHz: 0.07,
        absorption8kHz: 0.07,
        absorption16kHz: 0.06,

        scattering125Hz: 0.15,
        scattering250Hz: 0.20,
        scattering500Hz: 0.25,
        scattering1kHz: 0.30,
        scattering2kHz: 0.35,
        scattering4kHz: 0.40,
        scattering8kHz: 0.45,
        scattering16kHz: 0.50,

        roughness: 0.3,
        phaseShift: 0,
        phaseRandomization: 0.2
    }
};
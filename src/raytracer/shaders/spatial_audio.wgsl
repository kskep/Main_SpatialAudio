struct ListenerData {
    position: vec3f,
    forward: vec3f,
    up: vec3f,
    right: vec3f,
}

struct RayHit {
    position: vec3f,
    time: f32,
    normal: vec3f,
    energy: f32,
    // Eight frequency bands
    energy125: f32,  // Changed from 63Hz
    energy250: f32,
    energy500: f32,
    energy1k: f32,
    energy2k: f32,
    energy4k: f32,
    energy8k: f32,
    energy16k: f32,  // Changed from 8kHz to 16kHz
    // Wave properties
    phase: f32,
    frequency: f32,
    dopplerShift: f32,
    _padding: f32  // Maintain alignment
}

struct FrequencyBands {
    band125: f32,   // Changed from band63
    band250: f32,
    band500: f32,
    band1k: f32,
    band2k: f32,
    band4k: f32,
    band8k: f32,
    band16k: f32    // Changed from band8k
}

struct RoomAcoustics {
    // RT60 for each frequency band
    rt60_125: f32,  // Changed from rt60_63
    rt60_250: f32,
    rt60_500: f32,
    rt60_1k: f32,
    rt60_2k: f32,
    rt60_4k: f32,
    rt60_8k: f32,
    rt60_16k: f32,  // Changed from rt60_8k

    // Air absorption coefficients (increases with frequency)
    absorption_125: f32,  // Changed from absorption_63
    absorption_250: f32,
    absorption_500: f32,
    absorption_1k: f32,
    absorption_2k: f32,
    absorption_4k: f32,
    absorption_8k: f32,
    absorption_16k: f32,  // Changed from absorption_8k

    // Scattering coefficients (frequency dependent)
    scattering_125: f32,  // Changed from scattering_63
    scattering_250: f32,
    scattering_500: f32,
    scattering_1k: f32,
    scattering_2k: f32,
    scattering_4k: f32,
    scattering_8k: f32,
    scattering_16k: f32,  // Changed from scattering_8k

    // Room characteristics
    earlyReflectionTime: f32,
    roomVolume: f32,
    totalSurfaceArea: f32,
}

struct SpatialAudioParams {
    speedOfSound: f32,
    maxDistance: f32,
    minDistance: f32,
    temperature: f32,    // For air absorption calculation
    humidity: f32,       // For air absorption calculation
    sourcePower: f32,    // Source power in dB
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
    _padding4: f32,
    _padding5: f32      // Total size: 48 bytes (12 floats)
}

struct WaveProperties {
    phase: f32,
    frequency: f32,
    dopplerShift: f32,
    _padding: f32
}

@group(0) @binding(0) var<uniform> listener: ListenerData;
@group(0) @binding(1) var<storage, read> rayHits: array<RayHit>;
@group(0) @binding(2) var<storage, read_write> spatialIR: array<vec4f>;
@group(0) @binding(3) var<uniform> params: SpatialAudioParams;
@group(0) @binding(4) var<uniform> acoustics: RoomAcoustics;
@group(0) @binding(5) var<storage, read> waveProperties: array<WaveProperties>;

const SPEED_OF_SOUND = 343.0;
const AIR_DENSITY = 1.225;  // kg/m³ at room temperature
const REFERENCE_PRESSURE = 2e-5;  // 20 micropascals (threshold of hearing)

fn calculateDirectionalAttenuation(direction: vec3f, listenerForward: vec3f) -> f32 {
    let dir = normalize(direction);
    let forward = normalize(listenerForward);
    let cosAngle = dot(dir, forward);
    return clamp((cosAngle + 1.0) * 0.5, 0.0, 1.0);
}

fn calculateHRTF(direction: vec3f, listenerRight: vec3f) -> vec2f {
    let dir = normalize(direction);
    let right = normalize(listenerRight);
    let rightDot = dot(dir, right);
    let leftGain = clamp(1.0 - (rightDot + 1.0) * 0.25, 0.1, 1.0);
    let rightGain = clamp(1.0 + (rightDot - 1.0) * 0.25, 0.1, 1.0);
    return vec2f(leftGain, rightGain);
}

fn calculateScattering(direction: vec3f, normal: vec3f) -> FrequencyBands {
    let incidentAngle = acos(dot(normalize(direction), normalize(normal)));

    // Scattering increases with frequency according to acoustic theory
    return FrequencyBands(
        mix(0.9, cos(incidentAngle), acoustics.scattering_125),
        mix(0.8, cos(incidentAngle), acoustics.scattering_250),
        mix(0.7, cos(incidentAngle), acoustics.scattering_500),
        mix(0.6, cos(incidentAngle), acoustics.scattering_1k),
        mix(0.5, cos(incidentAngle), acoustics.scattering_2k),
        mix(0.4, cos(incidentAngle), acoustics.scattering_4k),
        mix(0.3, cos(incidentAngle), acoustics.scattering_8k),
        mix(0.2, cos(incidentAngle), acoustics.scattering_16k)
    );
}

fn calculateAirAbsorption(distance: f32) -> FrequencyBands {
    let d = max(distance, 0.001);
    return FrequencyBands(
        exp(-acoustics.absorption_125 * d),
        exp(-acoustics.absorption_250 * d),
        exp(-acoustics.absorption_500 * d),
        exp(-acoustics.absorption_1k * d),
        exp(-acoustics.absorption_2k * d),
        exp(-acoustics.absorption_4k * d),
        exp(-acoustics.absorption_8k * d),
        exp(-acoustics.absorption_16k * d)
    );
}

fn calculateEnergyDecay(time: f32, distance: f32, direction: vec3f, normal: vec3f) -> FrequencyBands {
    let t = max(time, 0.0);

    let timeDecay = FrequencyBands(
        exp(-3.0 * t / acoustics.rt60_125),
        exp(-3.0 * t / acoustics.rt60_250),
        exp(-3.0 * t / acoustics.rt60_500),
        exp(-3.0 * t / acoustics.rt60_1k),
        exp(-3.0 * t / acoustics.rt60_2k),
        exp(-3.0 * t / acoustics.rt60_4k),
        exp(-3.0 * t / acoustics.rt60_8k),
        exp(-3.0 * t / acoustics.rt60_16k)
    );

    let scattering = calculateScattering(direction, normal);

    return FrequencyBands(
        timeDecay.band125 * scattering.band125,
        timeDecay.band250 * scattering.band250,
        timeDecay.band500 * scattering.band500,
        timeDecay.band1k * scattering.band1k,
        timeDecay.band2k * scattering.band2k,
        timeDecay.band4k * scattering.band4k,
        timeDecay.band8k * scattering.band8k,
        timeDecay.band16k * scattering.band16k
    );
}

fn isEarlyReflection(time: f32, distance: f32) -> bool {
    let directSound = distance / SPEED_OF_SOUND;
    let normalizedTime = (time - directSound) / acoustics.earlyReflectionTime;
    return normalizedTime < 1.0;
}

fn dbToLinear(db: f32) -> f32 {
    return pow(10.0, db / 20.0);
}

fn calculateDopplerShift(rayVelocity: vec3f, rayDirection: vec3f, speedOfSound: f32) -> f32 {
    let relativeVelocity = dot(rayVelocity, rayDirection);
    return speedOfSound / (speedOfSound - relativeVelocity);
}

fn calculateWaveContribution(
    time: f32,
    phase: f32,
    frequency: f32,
    dopplerShift: f32,
    amplitude: f32,
    distance: f32
) -> f32 {
    let validFreq = max(frequency, 20.0);
    let validAmplitude = max(amplitude, 0.0);
    let validDistance = max(distance, 0.001);

    let shiftedFreq = validFreq * max(dopplerShift, 0.1);
    let wavelength = SPEED_OF_SOUND / shiftedFreq;
    let distancePhase = 2.0 * 3.14159 * validDistance / wavelength;
    let totalPhase = phase + distancePhase;

    // Modified window function for stronger early reflections
    let windowPos = clamp(time / (validDistance / SPEED_OF_SOUND), 0.0, 1.0);
    let window = 0.8 * (1.0 - cos(2.0 * 3.14159 * windowPos));

    // Amplify early reflections with distance-based boost
    let earlyBoost = 3.0;
    let distanceAttenuation = 1.0 / max(validDistance * validDistance, 0.01);
    
    // Apply frequency-dependent boost based on room acoustics
    var freqBoost = 1.0;
    if (validFreq < 250.0) {
        freqBoost = mix(1.2, 1.0, (validFreq - 20.0) / 230.0);  // Bass boost
    } else if (validFreq < 4000.0) {
        freqBoost = mix(1.0, 1.1, (validFreq - 250.0) / 3750.0);  // Mid boost
    } else {
        freqBoost = mix(1.1, 0.9, (validFreq - 4000.0) / 12000.0);  // High attenuation
    }

    return validAmplitude * earlyBoost * window * sin(totalPhase) * distanceAttenuation * freqBoost;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let hitIndex = global_id.x;
    if (hitIndex >= arrayLength(&rayHits)) {
        return;
    }

    let hit = rayHits[hitIndex];
    let toListener = normalize(listener.position - hit.position);

    let distance = length(listener.position - hit.position);
    let dirAttenuation = calculateDirectionalAttenuation(toListener, listener.forward);
    let hrtf = calculateHRTF(toListener, listener.right);

    let waveProps = waveProperties[hitIndex];
    // Calculate total energy across all frequency bands with frequency-dependent weighting
    var totalWeightedEnergy =
        hit.energy125 * 0.7 +   // Bass frequencies (more weight)
        hit.energy250 * 0.8 +
        hit.energy500 * 0.9 +
        hit.energy1k * 1.0 +    // Mid frequencies (full weight)
        hit.energy2k * 0.95 +
        hit.energy4k * 0.9 +
        hit.energy8k * 0.85 +
        hit.energy16k * 0.8;    // High frequencies (less weight)

    let totalBands = 8.0;

    // Calculate energy decay factors
    let airAbsorption = calculateAirAbsorption(distance);
    let energyDecay = calculateEnergyDecay(hit.time, distance, toListener, hit.normal);

    // Combine energy decay and frequency-dependent factors
    // Calculate final amplitude with all factors combined
    let decayFactor = (
        airAbsorption.band125 * energyDecay.band125 * 0.7 +
        airAbsorption.band250 * energyDecay.band250 * 0.8 +
        airAbsorption.band500 * energyDecay.band500 * 0.9 +
        airAbsorption.band1k * energyDecay.band1k * 1.0 +
        airAbsorption.band2k * energyDecay.band2k * 0.95 +
        airAbsorption.band4k * energyDecay.band4k * 0.9 +
        airAbsorption.band8k * energyDecay.band8k * 0.85 +
        airAbsorption.band16k * energyDecay.band16k * 0.8
    ) / totalBands;

    let amplitude = sqrt(max(totalWeightedEnergy / totalBands, 0.0)) * decayFactor;

    let contribution = calculateWaveContribution(
        hit.time,
        waveProps.phase,
        waveProps.frequency,
        waveProps.dopplerShift,
        amplitude,
        distance
    );

    let leftContribution = max(contribution * hrtf.x * dirAttenuation * 0.5, 0.0001);
    let rightContribution = max(contribution * hrtf.y * dirAttenuation * 0.5, 0.0001);

    spatialIR[hitIndex] = vec4f(
        leftContribution,
        rightContribution,
        waveProps.frequency,
        hit.time
    );
}
struct Ray {
    origin: vec3f,
    direction: vec3f,
    energy125Hz: f32,
    energy250Hz: f32,
    energy500Hz: f32,
    energy1kHz: f32,
    energy2kHz: f32,
    energy4kHz: f32,
    energy8kHz: f32,
    energy16kHz: f32,
    pathLength: f32,
    bounces: u32,
    isActive: u32,
    frequency: f32,    
    phase: f32,        
    time: f32         
};

struct Surface {
    normal: vec3f,
    position: vec3f,
    absorption125Hz: f32,
    absorption250Hz: f32,
    absorption500Hz: f32,
    absorption1kHz: f32,
    absorption2kHz: f32,
    absorption4kHz: f32,
    absorption8kHz: f32,
    absorption16kHz: f32,
    scattering: f32,
    roughness: f32
};

struct RayHit {
    position: vec3f,
    energy125Hz: f32,
    energy250Hz: f32,
    energy500Hz: f32,
    energy1kHz: f32,
    energy2kHz: f32,
    energy4kHz: f32,
    energy8kHz: f32,
    energy16kHz: f32,
    time: f32,
    phase: f32,       
    frequency: f32    
};

struct RayIntersection {
    hit: bool,
    distance: f32,
    position: vec3f,
    normal: vec3f,
    surfaceIndex: u32
};

struct RayPoint {
    position: vec3f,
    energy125Hz: f32,
    energy250Hz: f32,
    energy500Hz: f32,
    energy1kHz: f32,
    energy2kHz: f32,
    energy4kHz: f32,
    energy8kHz: f32,
    energy16kHz: f32,
    time: f32,
    phase: f32,
    frequency: f32
};



// Helper function to find closest intersection
fn findClosestIntersection(ray: Ray) -> RayIntersection {
    var closest: RayIntersection;
    closest.hit = false;
    closest.distance = 999999.0;

    for (var i = 0u; i < arrayLength(&surfaces); i++) {
        let surface = surfaces[i];

        // Calculate intersection with plane
        let denom = dot(ray.direction, surface.normal);
        if (abs(denom) > 0.0001) { // Avoid parallel rays
            let t = dot(surface.position - ray.origin, surface.normal) / denom;
            if (t > 0.0001 && t < closest.distance) {
                closest.hit = true;
                closest.distance = t;
                closest.position = ray.origin + ray.direction * t;
                closest.normal = surface.normal;
                closest.surfaceIndex = i;
            }
        }
    }

    return closest;
}

// Calculate reflected direction
fn reflect(incident: vec3f, normal: vec3f) -> vec3f {
    return incident - 2.0 * dot(incident, normal) * normal;
}

// Pseudo-random number generator
fn random(seed: vec2f) -> f32 {
    return fract(sin(dot(seed, vec2f(12.9898, 78.233))) * 43758.5453);
}

// Generate random unit vector for diffuse reflection
fn randomUnitVector(seed: vec2f) -> vec3f {
    let phi = 2.0 * 3.14159 * random(seed);
    let cosTheta = 2.0 * random(seed + vec2f(1.0, 0.0)) - 1.0;
    let sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    
    return vec3f(
        sinTheta * cos(phi),
        sinTheta * sin(phi),
        cosTheta
    );
}

// Calculate a diffuse reflection direction in hemisphere around normal
fn diffuseReflection(normal: vec3f, seed: vec2f) -> vec3f {
    let randomDir = randomUnitVector(seed);
    // Ensure direction is in the same hemisphere as normal
    if (dot(randomDir, normal) < 0.0) {
        return -randomDir;
    }
    return randomDir;
}

// Mix between specular and diffuse reflection based on scattering coefficient
fn calculateReflectionDirection(incident: vec3f, normal: vec3f, scattering: f32, roughness: f32, ray_index: u32, bounce: u32) -> vec3f {
    let specular = reflect(incident, normal);
    
    // Use ray index and bounce count to create a unique seed for randomness
    let seed = vec2f(f32(ray_index), f32(bounce));
    let diffuse = diffuseReflection(normal, seed);
    
    // Interpolate between specular and diffuse based on scattering coefficient
    // Also apply roughness to perturb the specular direction
    let roughnessEffect = roughness * randomUnitVector(seed + vec2f(0.5, 0.5));
    let specularPerturbed = normalize(specular + roughnessEffect);
    
    return normalize(mix(specularPerturbed, diffuse, scattering));
}

@group(0) @binding(0) var<storage, read_write> rays: array<Ray>;
@group(0) @binding(1) var<storage, read> surfaces: array<Surface>;
@group(0) @binding(2) var<storage, read_write> hits: array<RayHit>;
@group(0) @binding(3) var<storage, read_write> rayPoints: array<RayPoint>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let ray_index = global_id.x;
    if (ray_index >= arrayLength(&rays)) {
        return;
    }

    var ray = rays[ray_index];
    if (ray.isActive == 0u || ray.bounces >= 50u || 
        (ray.energy125Hz + ray.energy250Hz + ray.energy500Hz + ray.energy1kHz + 
         ray.energy2kHz + ray.energy4kHz + ray.energy8kHz + ray.energy16kHz) / 8.0 < 0.05) {
        return;
    }

    let intersection = findClosestIntersection(ray);
    if (intersection.hit) {
        let distance = intersection.distance;
        let speed_of_sound = 343.0;  // Speed of sound in m/s
        let travel_time = distance / speed_of_sound;

        // Sample points along the ray path before hitting the surface
        let sample_distance = 0.1; // Sample every 0.1 meters
        let num_samples = u32(distance / sample_distance);
        
        // Calculate base point index using total path length to ensure continuous indexing
        let base_point_index = ray_index * 1000u + u32(ray.pathLength / sample_distance);

        for(var i = 0u; i < num_samples; i++) {
            let t = f32(i) * sample_distance;
            let point_position = ray.origin + ray.direction * t;
            let travel_time = t / speed_of_sound;
            let phase = ray.phase + 2.0 * 3.14159 * 1000.0 * travel_time; // Use middle frequency for phase

            // Apply air absorption based on distance
            // Air absorption is frequency-dependent (higher frequencies attenuate faster)
            let airAbsorption125 = exp(-0.0005 * t); // Low attenuation for low frequencies
            let airAbsorption250 = exp(-0.001 * t);
            let airAbsorption500 = exp(-0.002 * t);
            let airAbsorption1k = exp(-0.004 * t);
            let airAbsorption2k = exp(-0.007 * t);
            let airAbsorption4k = exp(-0.011 * t);
            let airAbsorption8k = exp(-0.018 * t);
            let airAbsorption16k = exp(-0.025 * t); // High attenuation for high frequencies

            let point_index = base_point_index + i;
            rayPoints[point_index] = RayPoint(
                point_position,
                ray.energy125Hz * airAbsorption125,
                ray.energy250Hz * airAbsorption250,
                ray.energy500Hz * airAbsorption500,
                ray.energy1kHz * airAbsorption1k,
                ray.energy2kHz * airAbsorption2k,
                ray.energy4kHz * airAbsorption4k,
                ray.energy8kHz * airAbsorption8k,
                ray.energy16kHz * airAbsorption16k,
                ray.time + travel_time,
                phase,
                1000.0 // Use middle frequency for reference
            );
        }

        // Calculate phase change over distance
        let phase_change = 2.0 * 3.14159 * 1000.0 * travel_time; // Use middle frequency for phase
        let new_phase = ray.phase + phase_change;

        // Record hit with all frequency bands
        hits[ray_index] = RayHit(
            intersection.position,
            ray.energy125Hz,
            ray.energy250Hz,
            ray.energy500Hz,
            ray.energy1kHz,
            ray.energy2kHz,
            ray.energy4kHz,
            ray.energy8kHz,
            ray.energy16kHz,
            ray.time + travel_time,
            new_phase,
            1000.0 // Use middle frequency for reference
        );

        // Update ray for next bounce
        let surface = surfaces[intersection.surfaceIndex];
        
        // Calculate new direction with scattering
        let reflected = calculateReflectionDirection(
            ray.direction, 
            intersection.normal,
            surface.scattering,
            surface.roughness,
            ray_index,
            ray.bounces
        );

        ray.origin = intersection.position;
        ray.direction = reflected;
        
        // Enhance early reflections by increasing their amplitude
        // First few bounces should have less decay to create distinct echoes
        let bounceBoost = 1.0;
        if (ray.bounces < 3u) {
            bounceBoost = 1.5 - f32(ray.bounces) * 0.15; // Boost early reflections (1.5x, 1.35x, 1.2x)
        }
        
        // Apply frequency-dependent absorption with bounce boost for early reflections
        ray.energy125Hz *= (1.0 - surface.absorption125Hz) * bounceBoost;
        ray.energy250Hz *= (1.0 - surface.absorption250Hz) * bounceBoost;
        ray.energy500Hz *= (1.0 - surface.absorption500Hz) * bounceBoost;
        ray.energy1kHz *= (1.0 - surface.absorption1kHz) * bounceBoost;
        ray.energy2kHz *= (1.0 - surface.absorption2kHz) * bounceBoost;
        ray.energy4kHz *= (1.0 - surface.absorption4kHz) * bounceBoost;
        ray.energy8kHz *= (1.0 - surface.absorption8kHz) * bounceBoost;
        ray.energy16kHz *= (1.0 - surface.absorption16kHz) * bounceBoost;
        
        ray.pathLength += distance;
        ray.bounces += 1u;
        ray.time += travel_time;
        ray.phase = new_phase;

        rays[ray_index] = ray;
    } else {
        ray.isActive = 0u;
        rays[ray_index] = ray;
    }
}
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

struct Edge {
    start: vec3f,
    end: vec3f,
    adjacentSurfaces: vec2u,  // Indices of adjacent surfaces as 2D vector
}

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
    surfaceIndex: u32,
    isDiffracted: bool,
    diffractionCoef: f32,
    edgeVector: vec3f
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
    closest.isDiffracted = false;
    closest.diffractionCoef = 0.0;

    // Check surface intersections first
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
                closest.isDiffracted = false;
                closest.diffractionCoef = 0.0;
            }
        }
    }

    // Check for diffraction around edges
    for (var i = 0u; i < arrayLength(&edges); i++) {
        let edge = edges[i];
        
        // Calculate closest point on edge to ray
        let v1 = edge.start - ray.origin;
        let v2 = normalize(edge.end - edge.start);
        let v3 = ray.direction;
        
        // Calculate parameters for closest approach
        let dot_v2_v3 = dot(v2, v3);
        let dot_v1_v2 = dot(v1, v2);
        let dot_v1_v3 = dot(v1, v3);
        
        // Skip if ray is parallel to edge
        if (abs(1.0 - dot_v2_v3 * dot_v2_v3) < 0.0001) {
            continue;
        }
        
        // Calculate t values for closest approach
        let t = (dot_v1_v2 - dot_v1_v3 * dot_v2_v3) / (1.0 - dot_v2_v3 * dot_v2_v3);
        
        // Check if point is on edge segment
        if (t < 0.0 || t > length(edge.end - edge.start)) {
            continue;
        }
        
        // Calculate closest point on edge
        let closestPoint = edge.start + v2 * t;
        
        // Calculate distance to edge
        let distanceToEdge = length(closestPoint - ray.origin);
        
        // If this is close enough and frequency is low enough for diffraction
        if (distanceToEdge < closest.distance && ray.frequency < 2000.0) {
            // Apply UTD (Uniform Theory of Diffraction) model
            // Calculate diffraction coefficient based on wavelength and edge angle
            let wavelength = 343.0 / ray.frequency; // speed of sound / frequency
            let lambda = wavelength;
            
            // Calculate diffraction angle
            let incidentVector = normalize(closestPoint - ray.origin);
            let edgeVector = normalize(edge.end - edge.start);
            
            // Get normals of adjacent surfaces
            let normal1 = surfaces[edge.adjacentSurfaces.x].normal;
            let normal2 = surfaces[edge.adjacentSurfaces.y].normal;
            
            // Calculate wedge angle
            let wedgeAngle = acos(dot(normal1, normal2));
            
            // Diffraction coefficient (simplified UTD model)
            let diffractionCoef = min(1.0, lambda / (2.0 * distanceToEdge * sin(wedgeAngle/2.0)));
            
            // Only consider if significant diffraction occurs
            if (diffractionCoef > 0.1) {
                closest.hit = true;
                closest.distance = distanceToEdge;
                closest.position = closestPoint;
                closest.normal = normalize(cross(normalize(ray.origin - closestPoint), edgeVector));
                closest.isDiffracted = true;
                closest.diffractionCoef = diffractionCoef;
                closest.edgeVector = edgeVector;
            }
        }
    }

    return closest;
}

// Calculate reflected direction
fn calculateReflectionDirection(incident: vec3f, normal: vec3f, scattering: f32, roughness: f32, ray_index: u32, bounce: u32, isDiffracted: bool, diffractionCoef: f32, edgeVector: vec3f) -> vec3f {
    if (isDiffracted) {
        // For diffraction, create a semi-circular pattern around the edge
        let perp = normalize(cross(edgeVector, incident));
        let diffAngle = (random(vec2f(f32(ray_index), f32(bounce))) - 0.5) * 3.14159; // -π/2 to π/2
        
        // Rotate around edge by diffraction angle
        let cosA = cos(diffAngle);
        let sinA = sin(diffAngle);
        let rotatedPerp = perp * cosA + cross(edgeVector, perp) * sinA;
        
        // Blend between original direction and diffracted direction based on coefficient
        return normalize(mix(reflect(incident, normal), rotatedPerp, diffractionCoef));
    } 
    else {
        // Original reflection code
        let specular = reflect(incident, normal);
        let seed = vec2f(f32(ray_index), f32(bounce));
        let diffuse = diffuseReflection(normal, seed);
        let roughnessEffect = roughness * randomUnitVector(seed + vec2f(0.5, 0.5));
        let specularPerturbed = normalize(specular + roughnessEffect);
        
        return normalize(mix(specularPerturbed, diffuse, scattering));
    }
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

@group(0) @binding(0) var<storage, read_write> rays: array<Ray>;
@group(0) @binding(1) var<storage, read> surfaces: array<Surface>;
@group(0) @binding(2) var<storage, read_write> hits: array<RayHit>;
@group(0) @binding(3) var<storage, read_write> rayPoints: array<RayPoint>;
@group(0) @binding(4) var<storage, read> edges: array<Edge>;

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
            let phase = ray.phase + 2.0 * 3.14159 * ray.frequency * travel_time;

            // Apply air absorption based on distance and frequency
            let airAbsorption = exp(-0.0005 * t * (ray.frequency / 1000.0));
            
            // Store ray point with updated phase and energy
            let point_index = base_point_index + i;
            if (point_index < arrayLength(&rayPoints)) {
                rayPoints[point_index] = RayPoint(
                    point_position,
                    ray.energy125Hz * airAbsorption,
                    ray.energy250Hz * airAbsorption,
                    ray.energy500Hz * airAbsorption,
                    ray.energy1kHz * airAbsorption,
                    ray.energy2kHz * airAbsorption,
                    ray.energy4kHz * airAbsorption,
                    ray.energy8kHz * airAbsorption,
                    ray.energy16kHz * airAbsorption,
                    ray.time + travel_time,
                    phase,
                    ray.frequency
                );
            }
        }

        // Update ray properties after intersection
        ray.origin = intersection.position;
        
        // Calculate new direction considering diffraction
        let surface = surfaces[intersection.surfaceIndex];
        ray.direction = calculateReflectionDirection(
            ray.direction, 
            intersection.normal, 
            surface.scattering, 
            surface.roughness, 
            ray_index, 
            ray.bounces,
            intersection.isDiffracted,
            intersection.diffractionCoef,
            intersection.edgeVector
        );
        
        // Update ray energies based on surface absorption and diffraction
        if (intersection.isDiffracted) {
            let diffractionCoef = intersection.diffractionCoef;
            ray.energy125Hz *= diffractionCoef;
            ray.energy250Hz *= diffractionCoef;
            ray.energy500Hz *= diffractionCoef;
            ray.energy1kHz *= diffractionCoef;
            ray.energy2kHz *= diffractionCoef;
            ray.energy4kHz *= diffractionCoef;
            ray.energy8kHz *= diffractionCoef;
            ray.energy16kHz *= diffractionCoef;
        } else {
            ray.energy125Hz *= (1.0 - surface.absorption125Hz);
            ray.energy250Hz *= (1.0 - surface.absorption250Hz);
            ray.energy500Hz *= (1.0 - surface.absorption500Hz);
            ray.energy1kHz *= (1.0 - surface.absorption1kHz);
            ray.energy2kHz *= (1.0 - surface.absorption2kHz);
            ray.energy4kHz *= (1.0 - surface.absorption4kHz);
            ray.energy8kHz *= (1.0 - surface.absorption8kHz);
            ray.energy16kHz *= (1.0 - surface.absorption16kHz);
        }
        
        ray.pathLength += distance;
        ray.bounces += 1u;
        ray.time += travel_time;
        ray.phase += 2.0 * 3.14159 * ray.frequency * travel_time;

        rays[ray_index] = ray;
    } else {
        ray.isActive = 0u;
        rays[ray_index] = ray;
    }
}
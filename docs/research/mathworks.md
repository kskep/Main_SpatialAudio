link: https://www.mathworks.com/help/audio/ug/room-impulse-response-simulation-with-stochastic-ray-tracing.html

Stochastic Ray Tracing for Room Acoustics Simulation
Briefing Document: Room Impulse Response Simulation with Stochastic Ray Tracing
Subject: Analysis of a MATLAB example showcasing room impulse response (RIR) simulation using stochastic ray tracing.
Source: Excerpts from a MathWorks Help Center documentation titled "Room Impulse Response Simulation with Stochastic Ray Tracing."
Date: October 26, 2023
Executive Summary:
This document analyzes a MATLAB example that demonstrates the simulation of a room impulse response using the stochastic ray tracing method. The example provides a practical implementation of this technique, outlining the steps involved in defining room parameters, generating random rays, defining reflection and scattering coefficients, performing ray tracing, and ultimately generating an impulse response that can be used for auralization (simulating the sound of a space). The stochastic ray tracing method is presented as an alternative to the image-source method, as it incorporates sound diffusion.
Key Themes and Ideas:
1.
Room Impulse Response (RIR) Simulation: The core objective is to model the reverberant properties of a room without physical acoustic measurements. "Room impulse response simulation aims to model the reverberant properties of a space without having to perform acoustic measurements."
2.
Stochastic Ray Tracing: This method simulates sound propagation by tracing rays emitted from a source, bouncing off surfaces, and contributing to the energy received at a specific location. It is based on the assumption that "sound energy travels around the room in rays." Each ray incidence point is treated as a secondary source.
3.
Geometric Acoustics: Ray tracing is a geometric method. The document mentions the image-source method as another geometric approach. "The image-source method is a popular geometric method... One drawback of the image-source method is that it only models specular reflections... Stochastic ray tracing... address[es] this limitation by also taking sound diffusion into account."
4.
Ray Emission and Tracing: Rays are emitted from the sound source in random directions (uniform distribution). The algorithm traces these rays as they bounce off the room's boundaries (walls, floor, ceiling).
5.
Reflection and Scattering Coefficients: These coefficients define how sound is reflected at a surface. The reflection includes both specular and diffused components. "The reflection is a combination of a specular component and a diffused component. The relative strength of each component is determined by the reflection and scattering coefficients of the surfaces." Absorption coefficients are also defined to model energy loss at each surface.
6.
Energy Histogram: A two-dimensional histogram (time vs. frequency) is used to accumulate the energy received at the receiver location. This histogram represents the envelope of the RIR.
7.
Frequency Dependence: The ray tracing algorithm is frequency-selective, operating on specific frequency bands (six in this example). The frequency dependent data, for example absorbtion, is defined at the frequencies defined in the variable FVect.
8.
Poisson Random Process: The RIR is synthesized by weighting a Poisson-distributed noise process with the energy histogram values. The document states: "You then compute the room impulse response by weighting a Poisson random process by the histogram values".
9.
Auralization: The generated RIR can be convolved with an audio signal to simulate the sound of the room (auralization). "Apply the impulse response to an audio signal... Simulate the received audio by filtering with the impulse response."
Key Steps in the Algorithm:
1.
Define Room Parameters: Specifies room dimensions, source and receiver coordinates.
◦
roomDimensions = [10 8 4];
◦
sourceCoord = [2 2 2]; receiverCoord = [5 5 1.8]; r = 0.0875;
2.
Generate Random Rays: Creates rays emanating from the source with random directions.
◦
N = 5000; (Number of rays)
◦
rays = RandSampleSphere(N);
3.
Define Reflection and Scattering Coefficients: Sets frequency-dependent coefficients for each surface of the room.
◦
FVect = [125 250 500 1000 2000 4000 8000]; (Frequencies)
◦
A = [...]; (Absorption coefficients)
◦
R = 1-A; (Reflection coefficients)
◦
D = [...]; (Scattering coefficients)
4.
Initialize Energy Histogram: Creates a matrix to store energy values across time and frequency.
◦
histTimeStep = 0.0040; (Histogram time resolution)
◦
impResTime = 0.5; (Impulse response length)
◦
nTBins = round(impResTime/histTimeStep); (Number of time bins)
◦
nFBins = length(FVect); (Number of frequency bins)
◦
TFHist = zeros(nTBins,nFBins);
5.
Perform Ray Tracing: Iterates through each ray and frequency band, simulating ray bounces and energy accumulation. The energy is weighted based on the reflection and scattering coefficients, and the distance traveled.
◦
The core of the algorthim is implemented with nested for loops: for iBand = 1:nFBins and for iRay = 1:size(rays,1).
6.
Generate Room Impulse Response: Creates the RIR by weighting a Poisson random process with the energy histogram.
◦
fs = 44100; (Audio sample rate)
◦
Poisson random process generation
◦
Bandpass filtering
◦
Combination of filtered sequences using the energy histogram as weights.
7.
Auralization: Convolves the generated RIR with an audio signal to simulate the room's acoustics.
Helper Functions:
The example utilizes several helper functions:
•
plotRoom: Plots the 3D room with source and receiver positions.
•
RandSampleSphere: Generates random ray directions.
•
getImpactWall: Determines which wall a ray impacts and the displacement vector.
•
getWallNormalVector: Returns the normal vector of a given wall.
•
getBandedgeFrequencies: Computes the band edge frequencies for filtering.


# Acoustic Simulation Research Notes - MathWorks Resources

## Room Acoustics Fundamentals

### Sound Wave Propagation
- Sound waves propagate spherically from point sources
- Energy decreases with distance (inverse square law)
- Frequency-dependent behavior in:
  - Absorption
  - Reflection
  - Diffraction
  - Scattering

### Key Physical Parameters
1. **Speed of Sound**
   - c = 343 m/s (at 20°C)
   - Temperature dependent: c = 331.3 + 0.606T (T in °C)

2. **Wavelength**
   - λ = c/f
   - Important for diffraction effects
   - Typical ranges:
     - Low freq (125Hz): ~2.7m
     - Mid freq (1kHz): ~0.34m
     - High freq (4kHz): ~0.09m

3. **Air Absorption**
   - Frequency dependent
   - Increases with:
     - Higher frequencies
     - Distance
     - Humidity
     - Temperature

## Ray Tracing Implementation

### Energy Calculations

1. **Distance Attenuation**
```matlab
attenuation = 1 / (4 * pi * distance^2)
```

2. **Air Absorption**
```matlab
alpha_air = frequency_dependent_coefficient * distance
energy_loss = 10^(-alpha_air/10)
```

3. **Surface Reflection**
```matlab
incident_angle = acos(dot(ray_direction, surface_normal))
reflection_coeff = surface_absorption + (1-surface_absorption) * sin(incident_angle)^2
```

### Diffraction Modeling

1. **Edge Detection**
- Check proximity to surface edges
- Threshold based on wavelength
```matlab
edge_threshold = wavelength / 4
is_edge = distance_to_edge < edge_threshold
```

2. **Frequency-Dependent Diffraction**
```matlab
diffraction_factor = min(1, wavelength / (2 * pi * obstacle_size))
```

### Scattering Implementation

1. **Surface Roughness**
- Roughness relative to wavelength
```matlab
effective_roughness = surface_roughness * frequency / 1000
```

2. **Direction Perturbation**
```matlab
% Generate random perturbation angle based on roughness
theta = 2 * pi * rand
phi = acos(rand^(1/(roughness + 1)))

% Calculate perturbed direction
x = sin(phi) * cos(theta)
y = sin(phi) * sin(theta)
z = cos(phi)
```

## Material Properties

### Standard Material Coefficients

| Material  | 125Hz | 1kHz  | 4kHz  | Scattering |
|-----------|-------|-------|-------|------------|
| Concrete  | 0.02  | 0.03  | 0.04  | 0.1        |
| Wood      | 0.15  | 0.10  | 0.07  | 0.2        |
| Carpet    | 0.10  | 0.25  | 0.65  | 0.4        |
| Glass     | 0.03  | 0.02  | 0.02  | 0.1        |
| Curtains  | 0.07  | 0.40  | 0.70  | 0.7        |

### Temperature Effects
- Sound speed variation
- Air absorption changes
- Material property variations

## Performance Optimization

### Ray Generation
1. **Uniform Spherical Distribution**
```matlab
theta = 2 * pi * rand
phi = acos(2 * rand - 1)
x = sin(phi) * cos(theta)
y = sin(phi) * sin(theta)
z = cos(phi)
```

### Early Ray Termination
- Energy threshold: 0.01 (-20dB)
- Maximum bounces: 50-100
- Path length limit: room_size * 10

## References
1. MathWorks Acoustics Toolbox Documentation
2. Room Acoustics Simulation Best Practices
3. Ray Tracing for Audio Rendering
4. Acoustic Material Properties Database
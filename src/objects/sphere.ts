import { vec3 } from 'gl-matrix';

export class Sphere {
    private position: vec3;
    private radius: number;
    private readonly DEFAULT_RADIUS = 0.2; // Smaller radius for sound source visualization

    constructor(position: vec3 = vec3.fromValues(0, 1.7, 0), radius: number = 0.2) {
        this.position = vec3.clone(position); // Clone to ensure we have our own copy
        this.radius = radius;
    }

    public update(deltaTime: number, roomDimensions: { width: number, height: number, depth: number }): void {
        // Sound source is static, no update needed
    }

    public setPosition(position: vec3): void {
        vec3.copy(this.position, position);
    }

    public getPosition(): vec3 {
        return vec3.clone(this.position);
    }

    public getRadius(): number {
        return this.radius;
    }
}
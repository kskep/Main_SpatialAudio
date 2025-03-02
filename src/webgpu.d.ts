/// <reference types="@webgpu/types" />

interface Navigator {
    gpu: {
        requestAdapter(): Promise<GPUAdapter>;
        getPreferredCanvasFormat(): GPUTextureFormat;
    };
}

declare const GPUTextureUsage: {
    RENDER_ATTACHMENT: number;
    TEXTURE_BINDING: number;
    COPY_DST: number;
    COPY_SRC: number;
    STORAGE_BINDING: number;
};

interface HTMLCanvasElement extends HTMLElement {
    getContext(contextId: 'webgpu'): GPUCanvasContext | null;
}

interface GPUCanvasContext {
    configure(configuration: GPUCanvasConfiguration): void;
    getCurrentTexture(): GPUTexture;
    canvas: HTMLCanvasElement;
}

interface GPUCanvasConfiguration {
    device: GPUDevice;
    format: GPUTextureFormat;
    alphaMode?: 'premultiplied' | 'opaque';
    usage?: number;
}

interface GPUTexture {
    createView(descriptor?: GPUTextureViewDescriptor): GPUTextureView;
}

interface GPUTextureViewDescriptor {
    format?: GPUTextureFormat;
    dimension?: GPUTextureViewDimension;
    aspect?: GPUTextureAspect;
    baseMipLevel?: number;
    mipLevelCount?: number;
    baseArrayLayer?: number;
    arrayLayerCount?: number;
}

declare class GPUAdapter {
    readonly name: string;
    readonly features: GPUSupportedFeatures;
    readonly limits: GPUSupportedLimits;
    requestDevice(descriptor?: GPUDeviceDescriptor): Promise<GPUDevice>;
}

declare class GPUDevice {
    readonly adapter: GPUAdapter;
    readonly features: GPUSupportedFeatures;
    readonly limits: GPUSupportedLimits;
    queue: GPUQueue;

    createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup;
    createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout;
    createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
    createCommandEncoder(descriptor?: GPUCommandEncoderDescriptor): GPUCommandEncoder;
    createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor): GPUPipelineLayout;
    createRenderPipeline(descriptor: GPURenderPipelineDescriptor): GPURenderPipeline;
    createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
    createTexture(descriptor: GPUTextureDescriptor): GPUTexture;
}

declare class GPUQueue {
    writeBuffer(buffer: GPUBuffer, offset: number, data: BufferSource): void;
    submit(commandBuffers: Iterable<GPUCommandBuffer>): void;
}

declare class GPUBuffer {
    readonly size: number;
    readonly usage: number;
    readonly mapState: "mapped" | "unmapped";
    mapAsync(mode: number, offset?: number, size?: number): Promise<void>;
    getMappedRange(offset?: number, size?: number): ArrayBuffer;
    unmap(): void;
    destroy(): void;
}

declare class GPURenderPassEncoder {
    setPipeline(pipeline: GPURenderPipeline): void;
    setBindGroup(index: number, bindGroup: GPUBindGroup, dynamicOffsets?: Iterable<number>): void;
    setVertexBuffer(slot: number, buffer: GPUBuffer, offset?: number, size?: number): void;
    setIndexBuffer(buffer: GPUBuffer, indexFormat: GPUIndexFormat, offset?: number, size?: number): void;
    drawIndexed(indexCount: number, instanceCount?: number, firstIndex?: number, baseVertex?: number, firstInstance?: number): void;
    end(): void;
}

declare class GPURenderPipeline {
    getBindGroupLayout(index: number): GPUBindGroupLayout;
}

declare class GPUBindGroup {
    readonly label: string | undefined;
}

declare const GPUBufferUsage: {
    MAP_READ: number;
    MAP_WRITE: number;
    COPY_SRC: number;
    COPY_DST: number;
    INDEX: number;
    VERTEX: number;
    UNIFORM: number;
    STORAGE: number;
    INDIRECT: number;
    QUERY_RESOLVE: number;
};

declare const GPUShaderStage: {
    VERTEX: number;
    FRAGMENT: number;
    COMPUTE: number;
};

declare type GPUIndexFormat = "uint16" | "uint32";
declare type GPUBindingResource = GPUSampler | GPUTextureView | GPUBufferBinding;

declare interface GPUBufferBinding {
    buffer: GPUBuffer;
    offset?: number;
    size?: number;
}

declare interface GPUPipelineLayoutDescriptor {
    bindGroupLayouts: Iterable<GPUBindGroupLayout>;
}

declare interface GPUBindGroupLayoutEntry {
    binding: number;
    visibility: number;
    buffer?: {
        type: 'uniform' | 'storage' | 'read-only-storage';
        hasDynamicOffset?: boolean;
        minBindingSize?: number;
    };
    sampler?: {
        type?: 'filtering' | 'non-filtering' | 'comparison';
    };
    texture?: {
        sampleType?: 'float' | 'unfilterable-float' | 'depth' | 'sint' | 'uint';
        viewDimension?: '1d' | '2d' | '2d-array' | 'cube' | 'cube-array' | '3d';
        multisampled?: boolean;
    };
    storageTexture?: {
        access: 'write-only';
        format: GPUTextureFormat;
        viewDimension?: '1d' | '2d' | '2d-array' | '3d';
    };
}

declare interface GPUBindGroupLayoutDescriptor {
    entries: Iterable<GPUBindGroupLayoutEntry>;
}

declare interface GPUBindGroupDescriptor {
    layout: GPUBindGroupLayout;
    entries: Iterable<GPUBindGroupEntry>;
}

declare interface GPUBindGroupEntry {
    binding: number;
    resource: GPUBindingResource;
}

declare interface GPURenderPipelineDescriptor {
    layout: GPUPipelineLayout;
    vertex: GPUVertexState;
    fragment?: GPUFragmentState;
    primitive?: GPUPrimitiveState;
    depthStencil?: GPUDepthStencilState;
    multisample?: GPUMultisampleState;
}

declare type GPUCompareFunction = 'never' | 'less' | 'equal' | 'less-equal' | 'greater' | 'not-equal' | 'greater-equal' | 'always';

declare interface GPUDepthStencilState {
    format: GPUTextureFormat;
    depthWriteEnabled: boolean;
    depthCompare: GPUCompareFunction;
}

declare type GPUTextureFormat = 'bgra8unorm' | 'depth24plus' | string;

declare interface GPUVertexState extends GPUProgrammableStage {
    buffers?: Iterable<GPUVertexBufferLayout | null>;
}

declare interface GPUProgrammableStage {
    module: GPUShaderModule;
    entryPoint: string;
}

declare interface GPUVertexBufferLayout {
    arrayStride: number;
    stepMode?: 'vertex' | 'instance';
    attributes: Iterable<GPUVertexAttribute>;
}

declare interface GPUVertexAttribute {
    format: GPUVertexFormat;
    offset: number;
    shaderLocation: number;
}

declare type GPUVertexFormat = 'float32' | 'float32x2' | 'float32x3' | 'float32x4' | string;

declare interface GPUFragmentState extends GPUProgrammableStage {
    targets: Iterable<GPUColorTargetState>;
}

declare interface GPUColorTargetState {
    format: GPUTextureFormat;
    blend?: GPUBlendState;
    writeMask?: GPUColorWriteFlags;
}

declare interface GPUPrimitiveState {
    topology?: 'point-list' | 'line-list' | 'line-strip' | 'triangle-list' | 'triangle-strip';
    stripIndexFormat?: 'uint16' | 'uint32';
    frontFace?: 'ccw' | 'cw';
    cullMode?: 'none' | 'front' | 'back';
}

declare interface GPUBlendState {
    color: GPUBlendComponent;
    alpha: GPUBlendComponent;
}

declare interface GPUBlendComponent {
    operation?: GPUBlendOperation;
    srcFactor?: GPUBlendFactor;
    dstFactor?: GPUBlendFactor;
}

declare type GPUBlendOperation = 'add' | 'subtract' | 'reverse-subtract' | 'min' | 'max';
declare type GPUBlendFactor = 'zero' | 'one' | 'src' | 'one-minus-src' | 'dst' | 'one-minus-dst' | 'src-alpha' | 'one-minus-src-alpha' | 'dst-alpha' | 'one-minus-dst-alpha' | 'constant' | 'one-minus-constant';
declare type GPUColorWriteFlags = number;

declare interface GPUMultisampleState {
    count?: number;
    mask?: number;
    alphaToCoverageEnabled?: boolean;
}

declare interface GPUSupportedFeatures {
    has(feature: string): boolean;
}

declare interface GPUSupportedLimits {
    get(limit: string): number;
}

declare interface GPUCommandEncoderDescriptor {
    label?: string;
}

declare class GPUCommandEncoder {
    beginRenderPass(descriptor: GPURenderPassDescriptor): GPURenderPassEncoder;
    finish(): GPUCommandBuffer;
}

declare class GPUCommandBuffer {
    label?: string;
}

declare interface GPUTextureDescriptor {
    size: GPUExtent3D;
    format: GPUTextureFormat;
    usage: number;
    dimension?: GPUTextureDimension;
    mipLevelCount?: number;
    sampleCount?: number;
    viewFormats?: GPUTextureFormat[];
}

declare type GPUTextureDimension = "1d" | "2d" | "3d";

declare interface GPUExtent3D {
    width: number;
    height: number;
    depthOrArrayLayers?: number;
}

declare interface GPURenderPassDescriptor {
    colorAttachments: GPURenderPassColorAttachment[];
    depthStencilAttachment?: GPURenderPassDepthStencilAttachment;
}

declare interface GPURenderPassColorAttachment {
    view: GPUTextureView;
    resolveTarget?: GPUTextureView;
    clearValue?: GPUColor;
    loadOp: GPULoadOp;
    storeOp: GPUStoreOp;
}

declare interface GPURenderPassDepthStencilAttachment {
    view: GPUTextureView;
    depthClearValue?: number;
    depthLoadOp?: GPULoadOp;
    depthStoreOp?: GPUStoreOp;
    depthReadOnly?: boolean;
    stencilClearValue?: number;
    stencilLoadOp?: GPULoadOp;
    stencilStoreOp?: GPUStoreOp;
    stencilReadOnly?: boolean;
}

declare type GPULoadOp = 'load' | 'clear';
declare type GPUStoreOp = 'store' | 'discard';

declare interface GPUColor {
    r: number;
    g: number;
    b: number;
    a: number;
}
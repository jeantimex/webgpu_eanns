import type { WebGPUState } from '../../webgpu/utils';
import { BIRD_FLOATS, type FlappyBuffers } from './flappy_buffers';

const flappyRenderShader = /* wgsl */ `
struct Camera {
  halfWidth: f32,
  halfHeight: f32,
  bestIndex: u32,
  pad0: f32,
};

struct BirdState {
  pos: vec2f,
  velY: f32,
  alive: u32,
  score: u32,
  fitness: f32,
  jumpOutput: f32,
  pad0: f32,
};

struct Pipe {
  x: f32,
  topY: f32,
  bottomY: f32,
  width: f32,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> birds: array<BirdState>;
@group(0) @binding(2) var<storage, read> pipes: array<Pipe>;

const QUAD = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
  vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
);

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
};

@vertex
fn vertexMain(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  let birdCount = arrayLength(&birds);
  let pipeCount = arrayLength(&pipes);
  let totalPipeQuads = pipeCount * 2u; // Top pipe & bottom pipe per obstacle

  var world = vec2f(0.0, 0.0);
  var color = vec4f(0.0);

  if (ii < totalPipeQuads) {
    // 1. Render Pipes (Green columns)
    let pIdx = ii / 2u;
    let isBottom = ii % 2u == 1u;
    let pipe = pipes[pIdx];
    let c = QUAD[vi];

    if (!isBottom) {
      // Top pipe (from 0 to pipe.topY)
      let height = pipe.topY;
      world = vec2f(pipe.x + (c.x * 0.5 + 0.5) * pipe.width, (c.y * 0.5 + 0.5) * height);
      color = vec4f(0.2, 0.75, 0.3, 1.0); // Pipe Green
    } else {
      // Bottom pipe (from pipe.bottomY to 340)
      let height = 340.0 - pipe.bottomY;
      world = vec2f(pipe.x + (c.x * 0.5 + 0.5) * pipe.width, pipe.bottomY + (c.y * 0.5 + 0.5) * height);
      color = vec4f(0.2, 0.75, 0.3, 1.0); // Pipe Green
    }
  } else if (ii < totalPipeQuads + birdCount) {
    // 2. Render Birds (Yellow/Red quads with animated wing angle based on velY)
    let bIdx = ii - totalPipeQuads;
    let bird = birds[bIdx];
    if (bird.alive == 1u || bIdx == cam.bestIndex) {
      let c = QUAD[vi];
      let radius = 12.0;
      // Rotation angle based on vertical velocity
      let rot = clamp(bird.velY * 0.05, -0.6, 0.8);
      let ca = cos(rot);
      let sa = sin(rot);
      let localPos = vec2f(c.x * radius, c.y * radius);
      let rotPos = vec2f(localPos.x * ca - localPos.y * sa, localPos.x * sa + localPos.y * ca);

      world = bird.pos + rotPos;

      if (bIdx == cam.bestIndex) {
        color = vec4f(0.1, 0.85, 0.35, 1.0); // Best bird glowing green
      } else {
        color = vec4f(0.95, 0.8, 0.15, 0.85); // Birds golden yellow
      }
    }
  }

  var output: VertexOutput;
  // Convert 0..600 x, 0..400 y to NDC [-1, 1]
  let ndcX = (world.x / 300.0) - 1.0;
  let ndcY = 1.0 - (world.y / 200.0);
  output.position = vec4f(ndcX, ndcY, 0.0, 1.0);
  output.color = color;
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  return input.color;
}
`;

export class FlappyRenderer {
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly format: GPUTextureFormat;
  private readonly pipeline: GPURenderPipeline;
  private readonly bindGroup: GPUBindGroup;
  private readonly cameraBuffer: GPUBuffer;
  private readonly birdCount: number;
  private configured = false;
  private bestIndex = 0;

  constructor(
    _canvas: HTMLCanvasElement,
    gpu: WebGPUState,
    buffers: FlappyBuffers,
  ) {
    this.device = gpu.device;
    this.context = gpu.context;
    this.format = gpu.format;
    this.birdCount = buffers.birds.size / (BIRD_FLOATS * 4);

    this.cameraBuffer = this.device.createBuffer({
      label: 'flappy camera',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const shaderModule = this.device.createShaderModule({ code: flappyRenderShader });
    this.pipeline = this.device.createRenderPipeline({
      label: 'flappy render pipeline',
      layout: 'auto',
      vertex: { module: shaderModule, entryPoint: 'vertexMain' },
      fragment: {
        module: shaderModule,
        entryPoint: 'fragmentMain',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });

    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.cameraBuffer } },
        { binding: 1, resource: { buffer: buffers.birds } },
        { binding: 2, resource: { buffer: buffers.pipes } },
      ],
    });
  }

  setBestIndex(idx: number): void {
    this.bestIndex = idx;
  }

  render(pipeCount: number): void {
    if (!this.configured) {
      this.context.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });
      this.configured = true;
    }

    const cameraData = new ArrayBuffer(16);
    new Float32Array(cameraData).set([300, 200]);
    new Uint32Array(cameraData)[2] = this.bestIndex;
    this.device.queue.writeBuffer(this.cameraBuffer, 0, cameraData);

    const encoder = this.device.createCommandEncoder({ label: 'flappy frame' });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0.45, g: 0.75, b: 0.95, a: 1.0 }, // Sky Blue Background
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);

    const totalPipeQuads = pipeCount * 2;
    pass.draw(6, totalPipeQuads, 0, 0); // Render Pipes
    pass.draw(6, this.birdCount, 0, totalPipeQuads); // Render Birds
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}

import { resizeCanvasToDisplaySize, type WebGPUState } from '../../webgpu/utils';
import {
  BIRD_FLOATS,
  BIRD_H,
  BIRD_W,
  GROUND_Y,
  WORLD_H,
  WORLD_W,
  type FlappyBuffers,
} from './flappy_buffers';

// Sprite sizes from the source repo's assets.
const PIPE_UP_H = 242; // pipeUp.png (top pipe, bottom edge anchored at topY)
const PIPE_DOWN_H = 378; // pipeDown.png (bottom pipe, top edge anchored at bottomY)
const GROUND_W = 306; // ground.png
const GROUND_H = 118;

const ASSETS = ['bg', 'pipeUp', 'pipeDown', 'ground', 'bird'] as const;
type AssetName = (typeof ASSETS)[number];

const shader = /* wgsl */ `
struct Camera {
  scaleX: f32,
  scaleY: f32,
  offsetX: f32,
  offsetY: f32,
  bestIndex: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
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
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var tex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> birds: array<BirdState>;
@group(0) @binding(4) var<storage, read> pipes: array<Pipe>;

// Unit quad, (0,0) top-left .. (1,1) bottom-right; v=0 samples the image's top row.
const QUAD = array<vec2f, 6>(
  vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
  vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0)
);

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  @location(1) highlight: f32,
};

fn toNdc(world: vec2f) -> vec4f {
  return vec4f(world.x * cam.scaleX + cam.offsetX, world.y * cam.scaleY + cam.offsetY, 0.0, 1.0);
}

fn quadOut(c: vec2f, topLeft: vec2f, size: vec2f) -> VertexOutput {
  var output: VertexOutput;
  output.position = toNdc(topLeft + c * size);
  output.uv = c;
  output.highlight = 0.0;
  return output;
}

@vertex
fn vsBg(@builtin(vertex_index) vi: u32) -> VertexOutput {
  return quadOut(QUAD[vi], vec2f(0.0, 0.0), vec2f(${WORLD_W}.0, ${WORLD_H}.0));
}

@vertex
fn vsGround(@builtin(vertex_index) vi: u32) -> VertexOutput {
  return quadOut(QUAD[vi], vec2f(0.0, ${GROUND_Y}.0), vec2f(${GROUND_W}.0, ${GROUND_H}.0));
}

@vertex
fn vsPipeTop(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  let pipe = pipes[ii];
  return quadOut(QUAD[vi], vec2f(pipe.x, pipe.topY - ${PIPE_UP_H}.0), vec2f(pipe.width, ${PIPE_UP_H}.0));
}

@vertex
fn vsPipeBottom(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  let pipe = pipes[ii];
  return quadOut(QUAD[vi], vec2f(pipe.x, pipe.bottomY), vec2f(pipe.width, ${PIPE_DOWN_H}.0));
}

@vertex
fn vsBird(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  let bird = birds[ii];
  if (bird.alive == 0u) {
    // Dead birds vanish (original splices them out): collapse the quad off-screen.
    var dead: VertexOutput;
    dead.position = vec4f(-2.0, -2.0, 0.0, 1.0);
    dead.uv = vec2f(0.0, 0.0);
    dead.highlight = 0.0;
    return dead;
  }
  let c = QUAD[vi];
  var output: VertexOutput;
  output.position = toNdc(bird.pos + (c - 0.5) * vec2f(${BIRD_W}.0, ${BIRD_H}.0));
  output.uv = c;
  output.highlight = select(0.0, 1.0, ii == cam.bestIndex);
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let color = textureSample(tex, smp, input.uv);
  // Tint the best bird so it stands out in the flock.
  return mix(color, vec4f(0.4, 1.0, 0.55, color.a), input.highlight * 0.35);
}

@fragment
fn fragmentBird(input: VertexOutput) -> @location(0) vec4f {
  var color = textureSample(tex, smp, input.uv);
  // bird.png ships with broad semi-transparent pixels (~37% of the sprite), which
  // makes lone birds look ghostly; firm the alpha up so only edge AA stays soft.
  color.a = smoothstep(0.2, 0.6, color.a);
  return mix(color, vec4f(0.4, 1.0, 0.55, color.a), input.highlight * 0.35);
}
`;

async function loadTexture(device: GPUDevice, name: AssetName): Promise<GPUTexture> {
  const blob = await (await fetch(`/assets/flappy/${name}.png`)).blob();
  const bitmap = await createImageBitmap(blob);
  const texture = device.createTexture({
    label: `flappy ${name}`,
    size: [bitmap.width, bitmap.height],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture({ source: bitmap }, { texture }, [bitmap.width, bitmap.height]);
  return texture;
}

/** Draws the world with the source repo's sprites, aspect-fit into the canvas. */
export class FlappyRenderer {
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly cameraBuffer: GPUBuffer;
  private readonly pipelines: Record<'bg' | 'ground' | 'pipeTop' | 'pipeBottom' | 'bird', GPURenderPipeline>;
  private readonly bindGroups: Record<AssetName, GPUBindGroup>;
  private readonly birdCount: number;
  private bestIndex = 0;

  private constructor(
    private readonly canvas: HTMLCanvasElement,
    gpu: WebGPUState,
    buffers: FlappyBuffers,
    textures: Record<AssetName, GPUTexture>,
  ) {
    this.device = gpu.device;
    this.context = gpu.context;
    this.birdCount = buffers.birds.size / (BIRD_FLOATS * 4);

    this.cameraBuffer = this.device.createBuffer({
      label: 'flappy camera',
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const sampler = this.device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

    const layout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [layout] });

    const module = this.device.createShaderModule({ code: shader });
    const makePipeline = (entryPoint: string, fragment = 'fragmentMain'): GPURenderPipeline =>
      this.device.createRenderPipeline({
        label: `flappy ${entryPoint}`,
        layout: pipelineLayout,
        vertex: { module, entryPoint },
        fragment: {
          module,
          entryPoint: fragment,
          targets: [{
            format: gpu.format,
            blend: {
              color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
            },
          }],
        },
        primitive: { topology: 'triangle-list' },
      });
    this.pipelines = {
      bg: makePipeline('vsBg'),
      ground: makePipeline('vsGround'),
      pipeTop: makePipeline('vsPipeTop'),
      pipeBottom: makePipeline('vsPipeBottom'),
      bird: makePipeline('vsBird', 'fragmentBird'),
    };

    const makeBindGroup = (texture: GPUTexture): GPUBindGroup =>
      this.device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: this.cameraBuffer } },
          { binding: 1, resource: sampler },
          { binding: 2, resource: texture.createView() },
          { binding: 3, resource: { buffer: buffers.birds } },
          { binding: 4, resource: { buffer: buffers.pipes } },
        ],
      });
    this.bindGroups = {
      bg: makeBindGroup(textures.bg),
      ground: makeBindGroup(textures.ground),
      pipeUp: makeBindGroup(textures.pipeUp),
      pipeDown: makeBindGroup(textures.pipeDown),
      bird: makeBindGroup(textures.bird),
    };

    this.context.configure({ device: this.device, format: gpu.format, alphaMode: 'opaque' });
  }

  static async create(canvas: HTMLCanvasElement, gpu: WebGPUState, buffers: FlappyBuffers): Promise<FlappyRenderer> {
    const entries = await Promise.all(
      ASSETS.map(async (name) => [name, await loadTexture(gpu.device, name)] as const),
    );
    return new FlappyRenderer(canvas, gpu, buffers, Object.fromEntries(entries) as Record<AssetName, GPUTexture>);
  }

  setBestIndex(idx: number): void {
    this.bestIndex = idx;
  }

  render(pipeCount: number): void {
    resizeCanvasToDisplaySize(this.canvas, this.device.limits.maxTextureDimension2D);

    // Aspect-fit the 288x512 world into the canvas, centered.
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const scale = Math.min(cw / WORLD_W, ch / WORLD_H);
    const marginX = (cw - WORLD_W * scale) / 2;
    const marginY = (ch - WORLD_H * scale) / 2;
    const cameraData = new ArrayBuffer(32);
    new Float32Array(cameraData).set([
      (2 * scale) / cw,
      (-2 * scale) / ch,
      (2 * marginX) / cw - 1,
      1 - (2 * marginY) / ch,
    ]);
    new Uint32Array(cameraData)[4] = this.bestIndex;
    this.device.queue.writeBuffer(this.cameraBuffer, 0, cameraData);

    const encoder = this.device.createCommandEncoder({ label: 'flappy frame' });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0.08, g: 0.12, b: 0.12, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    // Same draw order as the original sketch: bg, pipes, birds, ground on top.
    // Clip to the world rect so top pipes don't overflow into the letterbox margin.
    // Clamp into range. Aspect-fit makes one margin exactly zero in theory, but
    // WORLD_W * (cw / WORLD_W) can land a few ulps *above* cw, so the margin comes
    // out at -1.4e-14 and Math.floor turns that into -1 — which setScissorRect
    // rejects outright. The ceil'd extent can likewise overshoot the target by a
    // pixel. Both are hard errors rather than visual glitches.
    const sx = Math.max(0, Math.floor(marginX));
    const sy = Math.max(0, Math.floor(marginY));
    pass.setScissorRect(
      sx,
      sy,
      Math.max(0, Math.min(Math.ceil(WORLD_W * scale), cw - sx)),
      Math.max(0, Math.min(Math.ceil(WORLD_H * scale), ch - sy)),
    );
    pass.setPipeline(this.pipelines.bg);
    pass.setBindGroup(0, this.bindGroups.bg);
    pass.draw(6);
    if (pipeCount > 0) {
      pass.setPipeline(this.pipelines.pipeTop);
      pass.setBindGroup(0, this.bindGroups.pipeUp);
      pass.draw(6, pipeCount);
      pass.setPipeline(this.pipelines.pipeBottom);
      pass.setBindGroup(0, this.bindGroups.pipeDown);
      pass.draw(6, pipeCount);
    }
    pass.setPipeline(this.pipelines.bird);
    pass.setBindGroup(0, this.bindGroups.bird);
    pass.draw(6, this.birdCount);
    pass.setPipeline(this.pipelines.ground);
    pass.setBindGroup(0, this.bindGroups.ground);
    pass.draw(6);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}

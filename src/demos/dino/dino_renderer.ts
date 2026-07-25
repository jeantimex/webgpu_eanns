import { resizeCanvasToDisplaySize, type WebGPUState } from '../../webgpu/utils';
import {
  DINO_FLOATS,
  DINO_H,
  DINO_W,
  DINO_X,
  PLAT_Y,
  WORLD_H,
  WORLD_W,
  type DinoBuffers,
  type ObstacleState,
} from './dino_buffers';

// Sprite sheet layout (sprite.png is 2404x130).
const SHEET_W = 2404;
const SHEET_H = 130;
const DINO_SRC_W = 88; // dino frames are 88x94 at y=0
const DINO_SRC_H = 94;
const DINO_JUMP_SRC_X = 1338;
const GROUND_SRC_Y = 104; // ground strip 2404x18
const GROUND_H = 18;

const shader = /* wgsl */ `
struct Uniforms {
  scaleX: f32,
  scaleY: f32,
  offsetX: f32,
  offsetY: f32,
  bestIndex: u32,
  runFrame: u32,
  groundScroll: f32,
  pad0: f32,
  obsDst: vec4f, // x, y, w, h on screen
  obsSrc: vec4f, // x, y, w, h in the sprite sheet
};

struct DinoState {
  y: f32,
  velY: f32,
  alive: u32,
  score: u32,
  fitness: f32,
  jumpOutput: f32,
  onGround: u32,
  pad0: f32,
};

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var tex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> dinos: array<DinoState>;

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
  return vec4f(world.x * uni.scaleX + uni.offsetX, world.y * uni.scaleY + uni.offsetY, 0.0, 1.0);
}

fn deadOut() -> VertexOutput {
  var output: VertexOutput;
  output.position = vec4f(-2.0, -2.0, 0.0, 1.0);
  output.uv = vec2f(0.0, 0.0);
  output.highlight = 0.0;
  return output;
}

@vertex
fn vsGround(@builtin(vertex_index) vi: u32) -> VertexOutput {
  let c = QUAD[vi];
  var output: VertexOutput;
  output.position = toNdc(vec2f(c.x * ${WORLD_W}.0, ${PLAT_Y - 24}.0 + c.y * ${GROUND_H}.0));
  // Repeat-addressed scroll window over the 2404px ground strip.
  output.uv = vec2f((c.x * ${WORLD_W}.0 + uni.groundScroll) / ${SHEET_W}.0, (${GROUND_SRC_Y}.0 + c.y * ${GROUND_H}.0) / ${SHEET_H}.0);
  output.highlight = 0.0;
  return output;
}

@vertex
fn vsObstacle(@builtin(vertex_index) vi: u32) -> VertexOutput {
  let c = QUAD[vi];
  var output: VertexOutput;
  output.position = toNdc(uni.obsDst.xy + c * uni.obsDst.zw);
  output.uv = (uni.obsSrc.xy + c * uni.obsSrc.zw) / vec2f(${SHEET_W}.0, ${SHEET_H}.0);
  output.highlight = 0.0;
  return output;
}

@vertex
fn vsDino(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  let dino = dinos[ii];
  if (dino.alive == 0u) { return deadOut(); }
  let c = QUAD[vi];
  var output: VertexOutput;
  output.position = toNdc(vec2f(${DINO_X}.0, dino.y) + c * vec2f(${DINO_W}.0, ${DINO_H}.0));
  // Run frames (onGround) alternate by generation tick; airborne shows the jump frame.
  let srcX = select(${DINO_JUMP_SRC_X}.0, f32(uni.runFrame), dino.onGround == 1u);
  output.uv = (vec2f(srcX, 0.0) + c * vec2f(${DINO_SRC_W}.0, ${DINO_SRC_H}.0)) / vec2f(${SHEET_W}.0, ${SHEET_H}.0);
  output.highlight = select(0.0, 1.0, ii == uni.bestIndex);
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let color = textureSample(tex, smp, input.uv);
  // Tint the best dino so it stands out in the herd.
  return mix(color, vec4f(0.4, 1.0, 0.55, color.a), input.highlight * 0.35);
}
`;

/** Draws the world with the source repo's sprite sheet, aspect-fit into the canvas. */
export class DinoRenderer {
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly uniformBuffer: GPUBuffer;
  private readonly pipelines: Record<'ground' | 'obstacle' | 'dino', GPURenderPipeline>;
  private readonly bindGroup: GPUBindGroup;
  private readonly dinoCount: number;
  private bestIndex = 0;

  private constructor(
    private readonly canvas: HTMLCanvasElement,
    gpu: WebGPUState,
    buffers: DinoBuffers,
    texture: GPUTexture,
  ) {
    this.device = gpu.device;
    this.context = gpu.context;
    this.dinoCount = buffers.dinos.size / (DINO_FLOATS * 4);

    this.uniformBuffer = this.device.createBuffer({
      label: 'dino uniforms',
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Nearest for pixel art; repeat-x wraps the scrolling ground strip.
    const sampler = this.device.createSampler({
      magFilter: 'nearest',
      minFilter: 'nearest',
      addressModeU: 'repeat',
      addressModeV: 'clamp-to-edge',
    });

    const layout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [layout] });

    const module = this.device.createShaderModule({ code: shader });
    const makePipeline = (entryPoint: string): GPURenderPipeline =>
      this.device.createRenderPipeline({
        label: `dino ${entryPoint}`,
        layout: pipelineLayout,
        vertex: { module, entryPoint },
        fragment: {
          module,
          entryPoint: 'fragmentMain',
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
      ground: makePipeline('vsGround'),
      obstacle: makePipeline('vsObstacle'),
      dino: makePipeline('vsDino'),
    };

    this.bindGroup = this.device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: sampler },
        { binding: 2, resource: texture.createView() },
        { binding: 3, resource: { buffer: buffers.dinos } },
      ],
    });

    this.context.configure({ device: this.device, format: gpu.format, alphaMode: 'opaque' });
  }

  static async create(canvas: HTMLCanvasElement, gpu: WebGPUState, buffers: DinoBuffers): Promise<DinoRenderer> {
    const blob = await (await fetch('/assets/dino/sprite.png')).blob();
    const bitmap = await createImageBitmap(blob);
    const texture = gpu.device.createTexture({
      label: 'dino sprite sheet',
      size: [bitmap.width, bitmap.height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    gpu.device.queue.copyExternalImageToTexture({ source: bitmap }, { texture }, [bitmap.width, bitmap.height]);
    return new DinoRenderer(canvas, gpu, buffers, texture);
  }

  setBestIndex(idx: number): void {
    this.bestIndex = idx;
  }

  render(obstacle: ObstacleState, groundScroll: number, runFrame: number): void {
    resizeCanvasToDisplaySize(this.canvas, this.device.limits.maxTextureDimension2D);

    // Aspect-fit the 1000x400 world into the canvas, centered.
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const scale = Math.min(cw / WORLD_W, ch / WORLD_H);
    const marginX = (cw - WORLD_W * scale) / 2;
    const marginY = (ch - WORLD_H * scale) / 2;
    const data = new ArrayBuffer(64);
    const f32 = new Float32Array(data);
    const u32 = new Uint32Array(data);
    f32.set([(2 * scale) / cw, (-2 * scale) / ch, (2 * marginX) / cw - 1, 1 - (2 * marginY) / ch]);
    u32[4] = this.bestIndex;
    u32[5] = runFrame;
    f32[6] = groundScroll;
    f32.set([obstacle.x, obstacle.y, obstacle.w, obstacle.h], 8);
    f32.set([obstacle.picX, 2, obstacle.w, obstacle.h], 12);
    this.device.queue.writeBuffer(this.uniformBuffer, 0, data);

    const encoder = this.device.createCommandEncoder({ label: 'dino frame' });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 1, g: 1, b: 1, a: 1 }, // the original's white background
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    // Clip to the world rect so off-screen obstacles don't leak into the margin.
    pass.setScissorRect(Math.floor(marginX), Math.floor(marginY), Math.ceil(WORLD_W * scale), Math.ceil(WORLD_H * scale));
    pass.setBindGroup(0, this.bindGroup);
    pass.setPipeline(this.pipelines.ground);
    pass.draw(6);
    pass.setPipeline(this.pipelines.obstacle);
    pass.draw(6);
    pass.setPipeline(this.pipelines.dino);
    pass.draw(6, this.dinoCount);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}

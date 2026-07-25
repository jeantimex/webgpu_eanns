import { resizeCanvasToDisplaySize, type WebGPUState } from '../../webgpu/utils';
import { WORLD_H, WORLD_W } from './maze';
import type { PacmanBuffers } from './pacman_buffers';

// Atlas layout (single texture, sprites in row 0, maze at y=16):
//   pacman strips (64px, 4 frames) at x = dir*64
//   ghost strips (32px, 2 frames) at x = 256 + ghost*128 + dir*32
//   scared blue/white at x = 768 / 800, eyes at x = 832 + dir*32
const ATLAS_W = 1024;
const ATLAS_H = 264;
const MAZE_Y = 16;
const GHOST_BASE_X = 256;
const SCARED_X = 768;
const EYES_BASE_X = 832;

const SPRITES: Array<{ name: string; x: number }> = [
  { name: 'pacman_up', x: 0 },
  { name: 'pacman_down', x: 64 },
  { name: 'pacman_left', x: 128 },
  { name: 'pacman_right', x: 192 },
  ...['blinky', 'pinky', 'inky', 'clyde'].flatMap((g, gi) =>
    ['up', 'down', 'left', 'right'].map((d, di) => ({ name: `${g}_${d}`, x: GHOST_BASE_X + gi * 128 + di * 32 })),
  ),
  { name: 'scared_blue', x: SCARED_X },
  { name: 'scared_white', x: SCARED_X + 32 },
  { name: 'eyes_up', x: EYES_BASE_X },
  { name: 'eyes_down', x: EYES_BASE_X + 32 },
  { name: 'eyes_left', x: EYES_BASE_X + 64 },
  { name: 'eyes_right', x: EYES_BASE_X + 96 },
  { name: 'cherry', x: 960 },
];

const shader = /* wgsl */ `
struct Uniforms {
  scaleX: f32,
  scaleY: f32,
  offsetX: f32,
  offsetY: f32,
  bestIndex: u32,
  animTime: f32,
  pad0: f32,
  pad1: f32,
};

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var atlas: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> agents: array<f32>;
@group(0) @binding(4) var<storage, read> pelletList: array<u32>; // c, r, power, pad per pellet

const AGENT_FLOATS = 64u;
const A_GHOSTS = 17u;
const A_PELLETS = 36u;

const QUAD = array<vec2f, 6>(
  vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
  vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0)
);

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  @location(1) color: vec4f, // solid-color path when alpha >= 0; textured when -1
};

fn toNdc(world: vec2f) -> vec4f {
  return vec4f(world.x * uni.scaleX + uni.offsetX, world.y * uni.scaleY + uni.offsetY, 0.0, 1.0);
}

fn texOut(c: vec2f, topLeft: vec2f, srcTopLeft: vec2f) -> VertexOutput {
  var output: VertexOutput;
  output.position = toNdc(topLeft + c * 16.0);
  output.uv = (srcTopLeft + c * 16.0) / vec2f(${ATLAS_W}.0, ${ATLAS_H}.0);
  output.color = vec4f(-1.0, 0.0, 0.0, 0.0);
  return output;
}

fn deadOut() -> VertexOutput {
  var output: VertexOutput;
  output.position = vec4f(-2.0, -2.0, 0.0, 1.0);
  output.uv = vec2f(0.0, 0.0);
  output.color = vec4f(-1.0, 0.0, 0.0, 0.0);
  return output;
}

// Grid coords -> sprite top-left in world px (characters are 2x2 tiles, 16px).
fn charTopLeft(x: f32, y: f32) -> vec2f {
  return vec2f((x - 0.5) * 8.0, (y - 0.5) * 8.0);
}

@vertex
fn vsMaze(@builtin(vertex_index) vi: u32) -> VertexOutput {
  let c = QUAD[vi];
  var output: VertexOutput;
  output.position = toNdc(c * vec2f(${WORLD_W}.0, ${WORLD_H}.0));
  output.uv = (vec2f(0.0, ${MAZE_Y}.0) + c * vec2f(${WORLD_W}.0, ${WORLD_H}.0)) / vec2f(${ATLAS_W}.0, ${ATLAS_H}.0);
  output.color = vec4f(-1.0, 0.0, 0.0, 0.0);
  return output;
}

@vertex
fn vsPellet(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  let base = uni.bestIndex * AGENT_FLOATS;
  let c = pelletList[ii * 4u];
  let r = pelletList[ii * 4u + 1u];
  let power = pelletList[ii * 4u + 2u];
  // Eaten? Collapse the quad.
  let idx = r * 28u + c;
  let word = bitcast<u32>(agents[base + A_PELLETS + (idx >> 5u)]);
  if (((word >> (idx & 31u)) & 1u) == 0u) { return deadOut(); }
  // Power pellets blink.
  if (power == 1u && fract(uni.animTime * 2.5) < 0.4) { return deadOut(); }

  let size = select(2.0, 6.0, power == 1u);
  let topLeft = vec2f(f32(c) * 8.0 + 4.0 - size * 0.5, f32(r) * 8.0 + 4.0 - size * 0.5);
  var output: VertexOutput;
  output.position = toNdc(topLeft + QUAD[vi] * size);
  output.uv = vec2f(0.0, 0.0);
  output.color = vec4f(1.0, 0.72, 0.68, 1.0); // arcade pellet cream
  return output;
}

@vertex
fn vsGhost(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  let base = uni.bestIndex * AGENT_FLOATS;
  let gb = base + A_GHOSTS + ii * 4u;
  let gx = agents[gb];
  let gy = agents[gb + 1u];
  let gdir = u32(agents[gb + 2u]);
  let mode = u32(agents[gb + 3u]);

  let frame = floor(uni.animTime * 10.0) % 2.0;
  var srcX = ${GHOST_BASE_X}.0 + f32(ii) * 128.0 + f32(gdir) * 32.0;
  if (mode == 2u) {
    srcX = ${EYES_BASE_X}.0 + f32(gdir) * 32.0;
  } else if (mode == 1u) {
    // Scared: blue, flashing white when the timer runs low.
    let flash = agents[base + 11u] < 2.0 && (floor(uni.animTime * 4.0) % 2.0) == 0.0;
    srcX = select(${SCARED_X}.0, ${SCARED_X + 32}.0, flash);
  }
  return texOut(QUAD[vi], charTopLeft(gx, gy), vec2f(srcX + frame * 16.0, 0.0));
}

@vertex
fn vsPac(@builtin(vertex_index) vi: u32) -> VertexOutput {
  let base = uni.bestIndex * AGENT_FLOATS;
  let pdir = u32(agents[base + 2u]);
  let moving = agents[base + 4u];
  let frame = select(0.0, floor(uni.animTime * 12.0) % 4.0, moving > 0.5);
  let srcX = f32(pdir) * 64.0 + frame * 16.0;
  return texOut(QUAD[vi], charTopLeft(agents[base], agents[base + 1u]), vec2f(srcX, 0.0));
}

@vertex
fn vsFruit(@builtin(vertex_index) vi: u32) -> VertexOutput {
  let base = uni.bestIndex * AGENT_FLOATS;
  if (agents[base + 35u] <= 0.0) { return deadOut(); } // no cherry on the board
  return texOut(QUAD[vi], charTopLeft(13.5, 17.0), vec2f(960.0, 0.0));
}

@fragment
fn fragmentTex(input: VertexOutput) -> @location(0) vec4f {
  return textureSample(atlas, smp, input.uv);
}

@fragment
fn fragmentSolid(input: VertexOutput) -> @location(0) vec4f {
  return input.color;
}
`;

async function buildAtlas(device: GPUDevice): Promise<GPUTexture> {
  const canvas = new OffscreenCanvas(ATLAS_W, ATLAS_H);
  const ctx = canvas.getContext('2d')!;
  const loads = SPRITES.map(async ({ name, x }) => {
    const blob = await (await fetch(`/assets/pacman/${name}.png`)).blob();
    const bitmap = await createImageBitmap(blob);
    ctx.drawImage(bitmap, x, 0);
  });
  const mazeBlob = await (await fetch('/assets/pacman/maze_blue.png')).blob();
  const maze = await createImageBitmap(mazeBlob);
  await Promise.all(loads);
  ctx.drawImage(maze, 0, MAZE_Y);

  const atlasBitmap = canvas.transferToImageBitmap();
  const texture = device.createTexture({
    label: 'pacman atlas',
    size: [ATLAS_W, ATLAS_H],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture({ source: atlasBitmap }, { texture }, [ATLAS_W, ATLAS_H]);
  return texture;
}

/** Best-agent view: maze, its remaining pellets, its 4 ghosts, its pacman. */
export class PacmanRenderer {
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly uniformBuffer: GPUBuffer;
  private readonly pipelines: Record<'maze' | 'pellet' | 'ghost' | 'pac' | 'fruit', GPURenderPipeline>;
  private readonly bindGroup: GPUBindGroup;
  private readonly pelletCount: number;
  private bestIndex = 0;

  private constructor(
    private readonly canvas: HTMLCanvasElement,
    gpu: WebGPUState,
    buffers: PacmanBuffers,
    atlas: GPUTexture,
  ) {
    this.device = gpu.device;
    this.context = gpu.context;
    this.pelletCount = buffers.pelletCount;

    this.uniformBuffer = this.device.createBuffer({
      label: 'pacman uniforms',
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const sampler = this.device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });

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
    const makePipeline = (entryPoint: string, fragment = 'fragmentTex'): GPURenderPipeline =>
      this.device.createRenderPipeline({
        label: `pacman ${entryPoint}`,
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
      maze: makePipeline('vsMaze'),
      pellet: makePipeline('vsPellet', 'fragmentSolid'),
      ghost: makePipeline('vsGhost'),
      pac: makePipeline('vsPac'),
      fruit: makePipeline('vsFruit'),
    };

    this.bindGroup = this.device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: sampler },
        { binding: 2, resource: atlas.createView() },
        { binding: 3, resource: { buffer: buffers.agents } },
        { binding: 4, resource: { buffer: buffers.pelletList } },
      ],
    });

    this.context.configure({ device: this.device, format: gpu.format, alphaMode: 'opaque' });
  }

  static async create(canvas: HTMLCanvasElement, gpu: WebGPUState, buffers: PacmanBuffers): Promise<PacmanRenderer> {
    return new PacmanRenderer(canvas, gpu, buffers, await buildAtlas(gpu.device));
  }

  setBestIndex(idx: number): void {
    this.bestIndex = idx;
  }

  render(animTime: number): void {
    resizeCanvasToDisplaySize(this.canvas, this.device.limits.maxTextureDimension2D);

    // Aspect-fit the 224x248 maze into the canvas, centered.
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const scale = Math.min(cw / WORLD_W, ch / WORLD_H);
    const marginX = (cw - WORLD_W * scale) / 2;
    const marginY = (ch - WORLD_H * scale) / 2;
    const data = new ArrayBuffer(32);
    new Float32Array(data).set([
      (2 * scale) / cw,
      (-2 * scale) / ch,
      (2 * marginX) / cw - 1,
      1 - (2 * marginY) / ch,
    ]);
    new Uint32Array(data)[4] = this.bestIndex;
    new Float32Array(data)[5] = animTime;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, data);

    const encoder = this.device.createCommandEncoder({ label: 'pacman frame' });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0.08, g: 0.12, b: 0.12, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    pass.setScissorRect(Math.floor(marginX), Math.floor(marginY), Math.ceil(WORLD_W * scale), Math.ceil(WORLD_H * scale));
    pass.setBindGroup(0, this.bindGroup);
    pass.setPipeline(this.pipelines.maze);
    pass.draw(6);
    pass.setPipeline(this.pipelines.pellet);
    pass.draw(6, this.pelletCount);
    pass.setPipeline(this.pipelines.ghost);
    pass.draw(6, 4);
    pass.setPipeline(this.pipelines.pac);
    pass.draw(6);
    pass.setPipeline(this.pipelines.fruit);
    pass.draw(6);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}

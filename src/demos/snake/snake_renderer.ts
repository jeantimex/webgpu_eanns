import { resizeCanvasToDisplaySize, type WebGPUState } from '../../webgpu/utils';
import { GRID, type SnakeBuffers } from './snake_buffers';

// Retro LCD look (colors sampled from the reference): light gray blocks on a
// darker olive field, a dark frame around the board, black-bordered body
// blocks, light head with two eyes, red-ringed apple.
const COLOR_SEAM: [number, number, number] = [0.62, 0.64, 0.56]; // #9ea390 between cells
const COLOR_BLOCK: [number, number, number] = [0.67, 0.69, 0.61]; // #aab09b empty blocks
const COLOR_BORDER: [number, number, number] = [0.18, 0.18, 0.16]; // #2e2e2a block borders + frame
const COLOR_BODY: [number, number, number] = [0.05, 0.04, 0.04]; // #0c0a0a block fill
const COLOR_HEAD: [number, number, number] = [0.62, 0.64, 0.57]; // #9fa491 head
const COLOR_APPLE_BORDER: [number, number, number] = [0.83, 0.24, 0.16];
const COLOR_APPLE_FILL: [number, number, number] = [0.61, 0.19, 0.12]; // #9c301e

const shader = /* wgsl */ `
struct Uniforms {
  scaleX: f32,
  scaleY: f32,
  offsetX: f32,
  offsetY: f32,
  bestIndex: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var<storage, read> agents: array<f32>;

const AGENT_FLOATS = 96u;
const A_HEAD_X = 0u;
const A_HEAD_Y = 1u;
const A_DIR = 2u;
const A_APPLE_X = 8u;
const A_APPLE_Y = 9u;
const A_MASK = 14u;

const QUAD = array<vec2f, 6>(
  vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
  vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0)
);

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
};

// Quad centered on a cell; size is in cell units.
fn cellQuad(c: vec2f, center: vec2f, size: f32, color: vec4f) -> VertexOutput {
  var output: VertexOutput;
  let world = (center + (c - 0.5) * size) / ${GRID}.0;
  output.position = vec4f(world.x * uni.scaleX + uni.offsetX, world.y * uni.scaleY + uni.offsetY, 0.0, 1.0);
  output.color = color;
  return output;
}

fn deadOut() -> VertexOutput {
  var output: VertexOutput;
  output.position = vec4f(-2.0, -2.0, 0.0, 1.0);
  output.color = vec4f(0.0, 0.0, 0.0, 0.0);
  return output;
}

fn dirVec(d: u32) -> vec2f {
  switch d {
    case 0u: { return vec2f(0.0, -1.0); }
    case 1u: { return vec2f(0.0, 1.0); }
    case 2u: { return vec2f(-1.0, 0.0); }
    default: { return vec2f(1.0, 0.0); }
  }
}

fn bodyCell(base: u32, ii: u32) -> bool {
  let word = bitcast<u32>(agents[base + A_MASK + (ii >> 5u)]);
  return ((word >> (ii & 31u)) & 1u) == 1u;
}

@vertex
fn vsFrame(@builtin(vertex_index) vi: u32) -> VertexOutput {
  // Dark frame: one cell thick around the 16x16 field.
  return cellQuad(QUAD[vi], vec2f(8.0, 8.0), 18.0, vec4f(${COLOR_BORDER}, 1.0));
}

@vertex
fn vsBoard(@builtin(vertex_index) vi: u32) -> VertexOutput {
  return cellQuad(QUAD[vi], vec2f(8.0, 8.0), 16.0, vec4f(${COLOR_SEAM}, 1.0));
}

@vertex
fn vsBg(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  let center = vec2f(f32(ii % ${GRID}u) + 0.5, f32(ii / ${GRID}u) + 0.5);
  return cellQuad(QUAD[vi], center, 0.88, vec4f(${COLOR_BLOCK}, 1.0));
}

@vertex
fn vsBodyBorder(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  let base = uni.bestIndex * AGENT_FLOATS;
  if (!bodyCell(base, ii)) { return deadOut(); }
  let center = vec2f(f32(ii % ${GRID}u) + 0.5, f32(ii / ${GRID}u) + 0.5);
  return cellQuad(QUAD[vi], center, 1.0, vec4f(${COLOR_BORDER}, 1.0));
}

@vertex
fn vsBodyFill(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  let base = uni.bestIndex * AGENT_FLOATS;
  if (!bodyCell(base, ii)) { return deadOut(); }
  let center = vec2f(f32(ii % ${GRID}u) + 0.5, f32(ii / ${GRID}u) + 0.5);
  return cellQuad(QUAD[vi], center, 0.74, vec4f(${COLOR_BODY}, 1.0));
}

@vertex
fn vsHead(@builtin(vertex_index) vi: u32) -> VertexOutput {
  let base = uni.bestIndex * AGENT_FLOATS;
  let center = vec2f(agents[base + A_HEAD_X] + 0.5, agents[base + A_HEAD_Y] + 0.5);
  return cellQuad(QUAD[vi], center, 0.92, vec4f(${COLOR_HEAD}, 1.0));
}

// Two eyes, placed perpendicular to the heading.
@vertex
fn vsEyes(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  let base = uni.bestIndex * AGENT_FLOATS;
  let fwd = dirVec(u32(agents[base + A_DIR]));
  let perp = vec2f(-fwd.y, fwd.x);
  let side = select(-0.2, 0.2, ii == 1u);
  let center = vec2f(agents[base + A_HEAD_X] + 0.5, agents[base + A_HEAD_Y] + 0.5)
    + fwd * 0.16 + perp * side;
  return cellQuad(QUAD[vi], center, 0.17, vec4f(${COLOR_BODY}, 1.0));
}

@vertex
fn vsAppleBorder(@builtin(vertex_index) vi: u32) -> VertexOutput {
  let base = uni.bestIndex * AGENT_FLOATS;
  let center = vec2f(agents[base + A_APPLE_X] + 0.5, agents[base + A_APPLE_Y] + 0.5);
  return cellQuad(QUAD[vi], center, 0.94, vec4f(${COLOR_APPLE_BORDER}, 1.0));
}

@vertex
fn vsAppleFill(@builtin(vertex_index) vi: u32) -> VertexOutput {
  let base = uni.bestIndex * AGENT_FLOATS;
  let center = vec2f(agents[base + A_APPLE_X] + 0.5, agents[base + A_APPLE_Y] + 0.5);
  return cellQuad(QUAD[vi], center, 0.62, vec4f(${COLOR_APPLE_FILL}, 1.0));
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  return input.color;
}
`;

const ENTRIES = ['vsFrame', 'vsBoard', 'vsBg', 'vsBodyBorder', 'vsBodyFill', 'vsHead', 'vsEyes', 'vsAppleBorder', 'vsAppleFill'] as const;
type Entry = (typeof ENTRIES)[number];

/** Best-agent board in the retro LCD style. No textures. */
export class SnakeRenderer {
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly uniformBuffer: GPUBuffer;
  private readonly pipelines: Record<Entry, GPURenderPipeline>;
  private readonly bindGroup: GPUBindGroup;
  private bestIndex = 0;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    gpu: WebGPUState,
    buffers: SnakeBuffers,
  ) {
    this.device = gpu.device;
    this.context = gpu.context;

    this.uniformBuffer = this.device.createBuffer({
      label: 'snake uniforms',
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const layout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    const module = this.device.createShaderModule({ code: shader });
    const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const makePipeline = (entryPoint: string): GPURenderPipeline =>
      this.device.createRenderPipeline({
        label: `snake ${entryPoint}`,
        layout: pipelineLayout,
        vertex: { module, entryPoint },
        fragment: { module, entryPoint: 'fragmentMain', targets: [{ format: gpu.format }] },
        primitive: { topology: 'triangle-list' },
      });
    this.pipelines = Object.fromEntries(ENTRIES.map((e) => [e, makePipeline(e)])) as Record<Entry, GPURenderPipeline>;

    this.bindGroup = this.device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: buffers.agents } },
      ],
    });

    this.context.configure({ device: this.device, format: gpu.format, alphaMode: 'opaque' });
  }

  setBestIndex(idx: number): void {
    this.bestIndex = idx;
  }

  render(): void {
    resizeCanvasToDisplaySize(this.canvas, this.device.limits.maxTextureDimension2D);

    // Aspect-fit the square board into the canvas, centered.
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const scale = Math.min(cw, ch);
    const marginX = (cw - scale) / 2;
    const marginY = (ch - scale) / 2;
    const data = new ArrayBuffer(32);
    new Float32Array(data).set([
      (2 * scale) / cw,
      (-2 * scale) / ch,
      (2 * marginX) / cw - 1,
      1 - (2 * marginY) / ch,
    ]);
    new Uint32Array(data)[4] = this.bestIndex;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, data);

    const encoder = this.device.createCommandEncoder({ label: 'snake frame' });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0.06, g: 0.09, b: 0.14, a: 1 }, // page margin, outside the frame
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    // Clip to board + 1-cell frame (clamped to the canvas: scissor must not
    // exceed the attachment or the whole pass is invalidated).
    const cell = scale / GRID;
    const sx = Math.max(0, Math.floor(marginX - cell));
    const sy = Math.max(0, Math.floor(marginY - cell));
    pass.setScissorRect(sx, sy, Math.min(Math.ceil(scale + 2 * cell), cw - sx), Math.min(Math.ceil(scale + 2 * cell), ch - sy));
    pass.setBindGroup(0, this.bindGroup);
    pass.setPipeline(this.pipelines.vsFrame);
    pass.draw(6);
    pass.setPipeline(this.pipelines.vsBoard);
    pass.draw(6);
    pass.setPipeline(this.pipelines.vsBg);
    pass.draw(6, 256);
    pass.setPipeline(this.pipelines.vsBodyBorder);
    pass.draw(6, 256);
    pass.setPipeline(this.pipelines.vsBodyFill);
    pass.draw(6, 256);
    pass.setPipeline(this.pipelines.vsHead);
    pass.draw(6);
    pass.setPipeline(this.pipelines.vsEyes);
    pass.draw(6, 2);
    pass.setPipeline(this.pipelines.vsAppleBorder);
    pass.draw(6);
    pass.setPipeline(this.pipelines.vsAppleFill);
    pass.draw(6);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}

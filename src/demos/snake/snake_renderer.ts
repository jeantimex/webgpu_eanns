import { resizeCanvasToDisplaySize, type WebGPUState } from '../../webgpu/utils';
import { A, AGENT_FLOATS, CELLS, GRID } from './snake_buffers';
import type { CoreBuffers } from '../../core';

const COLOR_PAGE: [number, number, number] = [0.08, 0.12, 0.12];
const COLOR_BOARD: [number, number, number] = [0.63, 0.67, 0.58]; // #a1ab94
const COLOR_CELL_EDGE: [number, number, number] = [0.56, 0.61, 0.52]; // #909c84
const COLOR_CELL_FACE: [number, number, number] = [0.68, 0.72, 0.62]; // #adb89e
const COLOR_CELL_INSET: [number, number, number] = [0.62, 0.67, 0.56]; // #9faa8f
const COLOR_DARK_OUTLINE: [number, number, number] = [0.09, 0.09, 0.08];
const COLOR_BODY: [number, number, number] = [0.03, 0.025, 0.025];
const COLOR_HEAD: [number, number, number] = [0.68, 0.72, 0.62];
const COLOR_APPLE_BORDER: [number, number, number] = [0.86, 0.18, 0.11];
const COLOR_APPLE_FILL: [number, number, number] = [0.58, 0.11, 0.08];

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

const AGENT_FLOATS = ${AGENT_FLOATS}u;
const A_HEAD_X = ${A.headX}u;
const A_HEAD_Y = ${A.headY}u;
const A_DIR = ${A.dir}u;
const A_APPLE_X = ${A.appleX}u;
const A_APPLE_Y = ${A.appleY}u;
const A_MASK = ${A.bodyMask}u;

const QUAD = array<vec2f, 6>(
  vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
  vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0)
);

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
};

fn quad(c: vec2f, center: vec2f, size: vec2f, color: vec4f) -> VertexOutput {
  var output: VertexOutput;
  let world = (center + (c - 0.5) * size) / ${GRID}.0;
  output.position = vec4f(world.x * uni.scaleX + uni.offsetX, world.y * uni.scaleY + uni.offsetY, 0.0, 1.0);
  output.color = color;
  return output;
}

// Quad centered on a cell; size is in cell units.
fn cellQuad(c: vec2f, center: vec2f, size: f32, color: vec4f) -> VertexOutput {
  return quad(c, center, vec2f(size, size), color);
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
  return cellQuad(QUAD[vi], vec2f(${GRID / 2}.0, ${GRID / 2}.0), ${GRID}.0, vec4f(${COLOR_CELL_EDGE}, 1.0));
}

@vertex
fn vsBoard(@builtin(vertex_index) vi: u32) -> VertexOutput {
  return cellQuad(QUAD[vi], vec2f(${GRID / 2}.0, ${GRID / 2}.0), ${GRID - 0.12}, vec4f(${COLOR_BOARD}, 1.0));
}

@vertex
fn vsCellEdge(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  let center = vec2f(f32(ii % ${GRID}u) + 0.5, f32(ii / ${GRID}u) + 0.5);
  return cellQuad(QUAD[vi], center, 0.86, vec4f(${COLOR_CELL_EDGE}, 1.0));
}

@vertex
fn vsCellFace(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  let center = vec2f(f32(ii % ${GRID}u) + 0.5, f32(ii / ${GRID}u) + 0.5);
  return cellQuad(QUAD[vi], center, 0.68, vec4f(${COLOR_CELL_FACE}, 1.0));
}

@vertex
fn vsCellInset(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  let center = vec2f(f32(ii % ${GRID}u) + 0.5, f32(ii / ${GRID}u) + 0.5);
  return cellQuad(QUAD[vi], center, 0.48, vec4f(${COLOR_CELL_INSET}, 1.0));
}

@vertex
fn vsBodyBorder(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  let base = uni.bestIndex * AGENT_FLOATS;
  if (!bodyCell(base, ii)) { return deadOut(); }
  let center = vec2f(f32(ii % ${GRID}u) + 0.5, f32(ii / ${GRID}u) + 0.5);
  return cellQuad(QUAD[vi], center, 0.84, vec4f(${COLOR_DARK_OUTLINE}, 1.0));
}

@vertex
fn vsBodyPad(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  let base = uni.bestIndex * AGENT_FLOATS;
  if (!bodyCell(base, ii)) { return deadOut(); }
  let center = vec2f(f32(ii % ${GRID}u) + 0.5, f32(ii / ${GRID}u) + 0.5);
  return cellQuad(QUAD[vi], center, 0.70, vec4f(${COLOR_CELL_FACE}, 1.0));
}

@vertex
fn vsBodyFill(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  let base = uni.bestIndex * AGENT_FLOATS;
  if (!bodyCell(base, ii)) { return deadOut(); }
  let center = vec2f(f32(ii % ${GRID}u) + 0.5, f32(ii / ${GRID}u) + 0.5);
  return cellQuad(QUAD[vi], center, 0.50, vec4f(${COLOR_BODY}, 1.0));
}

@vertex
fn vsHeadBorder(@builtin(vertex_index) vi: u32) -> VertexOutput {
  let base = uni.bestIndex * AGENT_FLOATS;
  let center = vec2f(agents[base + A_HEAD_X] + 0.5, agents[base + A_HEAD_Y] + 0.5);
  return cellQuad(QUAD[vi], center, 0.88, vec4f(${COLOR_DARK_OUTLINE}, 1.0));
}

@vertex
fn vsHead(@builtin(vertex_index) vi: u32) -> VertexOutput {
  let base = uni.bestIndex * AGENT_FLOATS;
  let center = vec2f(agents[base + A_HEAD_X] + 0.5, agents[base + A_HEAD_Y] + 0.5);
  return cellQuad(QUAD[vi], center, 0.72, vec4f(${COLOR_HEAD}, 1.0));
}

// Two eyes, placed perpendicular to the heading.
@vertex
fn vsEyes(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  let base = uni.bestIndex * AGENT_FLOATS;
  let fwd = dirVec(u32(agents[base + A_DIR]));
  let perp = vec2f(-fwd.y, fwd.x);
  let side = select(-0.18, 0.18, ii == 1u);
  let center = vec2f(agents[base + A_HEAD_X] + 0.5, agents[base + A_HEAD_Y] + 0.5)
    + fwd * 0.16 + perp * side;
  return cellQuad(QUAD[vi], center, 0.14, vec4f(${COLOR_BODY}, 1.0));
}

@vertex
fn vsAntenna(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  let base = uni.bestIndex * AGENT_FLOATS;
  let fwd = dirVec(u32(agents[base + A_DIR]));
  let perp = vec2f(-fwd.y, fwd.x);
  let head = vec2f(agents[base + A_HEAD_X] + 0.5, agents[base + A_HEAD_Y] + 0.5);
  if (ii == 0u) {
    return quad(QUAD[vi], head + fwd * 0.55, abs(fwd) * 0.32 + abs(perp) * 0.08, vec4f(${COLOR_DARK_OUTLINE}, 1.0));
  }
  let side = select(-0.1, 0.1, ii == 2u);
  return quad(QUAD[vi], head + fwd * 0.73 + perp * side, abs(fwd) * 0.08 + abs(perp) * 0.12, vec4f(${COLOR_DARK_OUTLINE}, 1.0));
}

@vertex
fn vsAppleBorder(@builtin(vertex_index) vi: u32) -> VertexOutput {
  let base = uni.bestIndex * AGENT_FLOATS;
  let center = vec2f(agents[base + A_APPLE_X] + 0.5, agents[base + A_APPLE_Y] + 0.5);
  return cellQuad(QUAD[vi], center, 0.72, vec4f(${COLOR_APPLE_BORDER}, 1.0));
}

@vertex
fn vsAppleFill(@builtin(vertex_index) vi: u32) -> VertexOutput {
  let base = uni.bestIndex * AGENT_FLOATS;
  let center = vec2f(agents[base + A_APPLE_X] + 0.5, agents[base + A_APPLE_Y] + 0.5);
  return cellQuad(QUAD[vi], center, 0.42, vec4f(${COLOR_APPLE_FILL}, 1.0));
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  return input.color;
}
`;

const ENTRIES = [
  'vsFrame',
  'vsBoard',
  'vsCellEdge',
  'vsCellFace',
  'vsCellInset',
  'vsBodyBorder',
  'vsBodyPad',
  'vsBodyFill',
  'vsHeadBorder',
  'vsHead',
  'vsEyes',
  'vsAntenna',
  'vsAppleBorder',
  'vsAppleFill',
] as const;
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
    buffers: CoreBuffers,
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
        clearValue: { r: COLOR_PAGE[0], g: COLOR_PAGE[1], b: COLOR_PAGE[2], a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    // Clip to the board (clamped to the canvas: scissor must not
    // exceed the attachment or the whole pass is invalidated).
    const sx = Math.max(0, Math.floor(marginX));
    const sy = Math.max(0, Math.floor(marginY));
    pass.setScissorRect(sx, sy, Math.min(Math.ceil(scale), cw - sx), Math.min(Math.ceil(scale), ch - sy));
    pass.setBindGroup(0, this.bindGroup);
    pass.setPipeline(this.pipelines.vsFrame);
    pass.draw(6);
    pass.setPipeline(this.pipelines.vsBoard);
    pass.draw(6);
    pass.setPipeline(this.pipelines.vsCellEdge);
    pass.draw(6, CELLS);
    pass.setPipeline(this.pipelines.vsCellFace);
    pass.draw(6, CELLS);
    pass.setPipeline(this.pipelines.vsCellInset);
    pass.draw(6, CELLS);
    pass.setPipeline(this.pipelines.vsBodyBorder);
    pass.draw(6, CELLS);
    pass.setPipeline(this.pipelines.vsBodyPad);
    pass.draw(6, CELLS);
    pass.setPipeline(this.pipelines.vsBodyFill);
    pass.draw(6, CELLS);
    pass.setPipeline(this.pipelines.vsHeadBorder);
    pass.draw(6);
    pass.setPipeline(this.pipelines.vsHead);
    pass.draw(6);
    pass.setPipeline(this.pipelines.vsEyes);
    pass.draw(6, 2);
    pass.setPipeline(this.pipelines.vsAntenna);
    pass.draw(6, 3);
    pass.setPipeline(this.pipelines.vsAppleBorder);
    pass.draw(6);
    pass.setPipeline(this.pipelines.vsAppleFill);
    pass.draw(6);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}

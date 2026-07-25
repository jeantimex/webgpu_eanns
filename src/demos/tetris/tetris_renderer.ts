import { resizeCanvasToDisplaySize, type WebGPUState } from '../../webgpu/utils';
import { PIECE_COLORS, wgslPieceTable } from './tetris_pieces';
import type { TetrisBuffers } from './tetris_buffers';

const colors = PIECE_COLORS.map((c) => `vec3f(${c.join(',')})`).join(',\n  ');

// NES frame colors (see reference): gray outer, pale-teal trim, black playfield.
const COLOR_FRAME: [number, number, number] = [0.5, 0.54, 0.56];
const COLOR_TRIM: [number, number, number] = [0.48, 0.66, 0.63];

const shader = /* wgsl */ `
struct Uniforms {
  scaleX: f32,
  scaleY: f32,
  offsetX: f32,
  offsetY: f32,
  bestIndex: u32,
  tickCount: f32,
  pad0: f32,
  pad1: f32,
};

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var<storage, read> agents: array<f32>;

${wgslPieceTable()}
const PIECE_COLORS = array<vec3f, 7>(
  ${colors}
);

const AGENT_FLOATS = 80u;
const A_LEVEL = 2u;
const A_PIECES = 3u;
const A_PCOL = 8u;
const A_PROT = 9u;
const A_PLAND = 10u;
const A_PTYPE = 11u;
const A_BOARD = 16u;

const QUAD = array<vec2f, 6>(
  vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
  vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0)
);

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
};

fn deadOut() -> VertexOutput {
  var output: VertexOutput;
  output.position = vec4f(-2.0, -2.0, 0.0, 1.0);
  output.color = vec4f(0.0, 0.0, 0.0, 0.0);
  return output;
}

// Quad centered on (cx, cy) in cell units; size is in cell units; board at (0..10, 0..20).
fn quadAt(c: vec2f, cx: f32, cy: f32, size: f32, color: vec4f) -> VertexOutput {
  var output: VertexOutput;
  let world = vec2f(cx, cy) + (c - 0.5) * size;
  output.position = vec4f(world.x * uni.scaleX + uni.offsetX, world.y * uni.scaleY + uni.offsetY, 0.0, 1.0);
  output.color = color;
  return output;
}

fn boardType(base: u32, x: u32, y: u32) -> u32 {
  let idx = y * 10u + x;
  let word = bitcast<u32>(agents[base + A_BOARD + (idx >> 2u)]);
  return (word >> ((idx & 3u) * 8u)) & 0xffu;
}

fn pieceCellAt(ptype: i32, rot: i32, k: u32) -> vec2i {
  let v = PIECES[(ptype * 4 + rot) * 2 + i32(k >> 1u)];
  let c = (k & 1u) * 2u;
  return vec2i(v[c], v[c + 1u]);
}

// Rectangle centered on (cx, cy) in cell units, board at (0..10, 0..20).
fn rectOut(c: vec2f, cx: f32, cy: f32, w: f32, h: f32, color: vec4f) -> VertexOutput {
  var output: VertexOutput;
  let world = vec2f(cx + (c.x - 0.5) * w, cy + (c.y - 0.5) * h);
  output.position = vec4f(world.x * uni.scaleX + uni.offsetX, world.y * uni.scaleY + uni.offsetY, 0.0, 1.0);
  output.color = color;
  return output;
}

@vertex
fn vsFrame(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  // 3 nested rects: gray frame, teal trim, black playfield.
  if (ii == 0u) { return rectOut(QUAD[vi], 5.0, 10.0, 12.0, 22.0, vec4f(${COLOR_FRAME}, 1.0)); }
  if (ii == 1u) { return rectOut(QUAD[vi], 5.0, 10.0, 11.4, 21.4, vec4f(${COLOR_TRIM}, 1.0)); }
  return rectOut(QUAD[vi], 5.0, 10.0, 10.4, 20.4, vec4f(0.0, 0.0, 0.0, 1.0));
}

fn cellLayer(vi: u32, ii: u32, size: f32, off: vec2f, shade: f32) -> VertexOutput {
  let base = uni.bestIndex * AGENT_FLOATS;
  let t = boardType(base, ii % 10u, ii / 10u);
  if (t == 0u) { return deadOut(); }
  let col = PIECE_COLORS[t - 1u] * shade;
  return quadAt(QUAD[vi], f32(ii % 10u) + 0.5 + off.x, f32(ii / 10u) + 0.5 + off.y, size, vec4f(col, 1.0));
}

// Beveled tile: dark border, color fill, white top-left highlight.
@vertex
fn vsCellBorder(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  return cellLayer(vi, ii, 1.0, vec2f(0.0, 0.0), 0.55);
}

@vertex
fn vsCellFill(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  return cellLayer(vi, ii, 0.74, vec2f(0.0, 0.0), 1.0);
}

@vertex
fn vsCellHi(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  return cellLayer(vi, ii, 0.2, vec2f(-0.2, -0.2), 1.6);
}

// The last locked piece, falling from the spawn box to its landing row (cosmetic).
fn fallRow(base: u32) -> f32 {
  let land = agents[base + A_PLAND];
  let level = agents[base + A_LEVEL];
  let duration = max(2.0, (land + 5.0) * 3.0 / (1.0 + level));
  let progress = clamp((uni.tickCount - agents[base + A_PIECES]) / duration, 0.0, 1.0);
  if (progress >= 1.0) { return 99.0; } // already in the board
  return -4.0 + (land + 4.0) * progress;
}

fn fallLayer(vi: u32, ii: u32, size: f32, off: vec2f, shade: f32) -> VertexOutput {
  let base = uni.bestIndex * AGENT_FLOATS;
  let row = fallRow(base);
  if (row > 50.0) { return deadOut(); }
  let cell = pieceCellAt(i32(agents[base + A_PTYPE]), i32(agents[base + A_PROT]), ii);
  let col = PIECE_COLORS[u32(agents[base + A_PTYPE])] * shade;
  return quadAt(QUAD[vi], agents[base + A_PCOL] + f32(cell.x) + 0.5 + off.x, row + f32(cell.y) + 0.5 + off.y, size, vec4f(col, 1.0));
}

@vertex
fn vsFallBorder(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  return fallLayer(vi, ii, 1.0, vec2f(0.0, 0.0), 0.55);
}

@vertex
fn vsFallFill(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  return fallLayer(vi, ii, 0.74, vec2f(0.0, 0.0), 1.0);
}

@vertex
fn vsFallHi(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  return fallLayer(vi, ii, 0.2, vec2f(-0.2, -0.2), 1.6);
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  return input.color;
}
`;

const ENTRIES = ['vsFrame', 'vsCellBorder', 'vsCellFill', 'vsCellHi', 'vsFallBorder', 'vsFallFill', 'vsFallHi'] as const;
type Entry = (typeof ENTRIES)[number];

/** Best-agent playfield, NES style: framed black field, beveled tiles, falling piece. */
export class TetrisRenderer {
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly uniformBuffer: GPUBuffer;
  private readonly pipelines: Record<Entry, GPURenderPipeline>;
  private readonly bindGroup: GPUBindGroup;
  private bestIndex = 0;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    gpu: WebGPUState,
    buffers: TetrisBuffers,
  ) {
    this.device = gpu.device;
    this.context = gpu.context;

    this.uniformBuffer = this.device.createBuffer({
      label: 'tetris uniforms',
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
        label: `tetris ${entryPoint}`,
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

  /** Board world is 12x22 cells (10x20 playfield + frame). */
  boardRect(): { x: number; y: number; w: number; h: number } {
    const cell = Math.min(this.canvas.width / 12, this.canvas.height / 22);
    return {
      x: (this.canvas.width - 12 * cell) / 2,
      y: (this.canvas.height - 22 * cell) / 2,
      w: 12 * cell,
      h: 22 * cell,
    };
  }

  render(tickCount: number): void {
    resizeCanvasToDisplaySize(this.canvas, this.device.limits.maxTextureDimension2D);

    // Aspect-fit the 12x22 framed board into the canvas, centered.
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const cell = Math.min(cw / 12, ch / 22);
    const marginX = (cw - 12 * cell) / 2;
    const marginY = (ch - 22 * cell) / 2;
    const data = new ArrayBuffer(32);
    new Float32Array(data).set([
      (2 * cell) / cw,
      (-2 * cell) / ch,
      (2 * marginX) / cw - 1 + (2 * cell) / cw, // frame occupies the outer cell ring
      1 - (2 * marginY) / ch - (2 * cell) / ch,
    ]);
    new Uint32Array(data)[4] = this.bestIndex;
    new Float32Array(data)[5] = tickCount;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, data);

    const encoder = this.device.createCommandEncoder({ label: 'tetris frame' });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0.13, g: 0.15, b: 0.17, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    // Clamp both sides: float dust can push marginY to -1 and 22*cell past the
    // canvas height; an out-of-bounds scissor invalidates the whole pass.
    const sx = Math.max(0, Math.floor(marginX));
    const sy = Math.max(0, Math.floor(marginY));
    pass.setScissorRect(sx, sy, Math.min(Math.ceil(12 * cell), cw - sx), Math.min(Math.ceil(22 * cell), ch - sy));
    pass.setBindGroup(0, this.bindGroup);
    pass.setPipeline(this.pipelines.vsFrame);
    pass.draw(6, 3);
    pass.setPipeline(this.pipelines.vsCellBorder);
    pass.draw(6, 200);
    pass.setPipeline(this.pipelines.vsCellFill);
    pass.draw(6, 200);
    pass.setPipeline(this.pipelines.vsCellHi);
    pass.draw(6, 200);
    pass.setPipeline(this.pipelines.vsFallBorder);
    pass.draw(6, 4);
    pass.setPipeline(this.pipelines.vsFallFill);
    pass.draw(6, 4);
    pass.setPipeline(this.pipelines.vsFallHi);
    pass.draw(6, 4);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}

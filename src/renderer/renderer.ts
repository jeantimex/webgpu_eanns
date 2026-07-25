import { CAR_FLOATS, type SimBuffers } from '../gpu/buffers';
import { buildRenderWalls, type Track } from '../sim/track';
import { createBufferWithData, resizeCanvasToDisplaySize, type WebGPUState } from '../webgpu/utils';

const renderShader = /* wgsl */ `
struct Camera {
  center: vec2f,
  halfSize: vec2f,
  bestIndex: u32,
  pad: u32,
};

// Same layout as the compute shader's CarState (48-byte stride).
struct CarState {
  pos: vec2f,
  angle: f32,
  vel: f32,
  alive: u32,
  cpIndex: u32,
  timeSinceCp: f32,
  fitness: f32,
  outputs: vec2f,
  pad: vec2f,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> cars: array<CarState>;
@group(0) @binding(2) var<storage, read> walls: array<vec4f>;
@group(0) @binding(3) var<storage, read> sensors: array<f32>;

const DEG2RAD = 0.017453292519943295;
const SENSOR_ANGLES = array<f32, 5>(-0.7853981633974483, -0.3587706702705722, 0.0, 0.3587706702705722, 0.7853981633974483);
const SENSOR_ORIGINS = array<vec2f, 5>(vec2f(-0.3, 0.54), vec2f(-0.3, 0.84), vec2f(0.0, 0.84), vec2f(0.3, 0.84), vec2f(0.3, 0.54));
const QUAD = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
  vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
);

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
};

@vertex
fn vertexMain(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
  let wallCount = arrayLength(&walls);
  let carCount = arrayLength(&cars);
  let wallInstances = wallCount * 3u; // strip + 2 joint discs per segment
  // Hidden instances keep world at one point: degenerate triangles rasterize nothing.
  var world = vec2f(0.0, 0.0);
  var color = vec4f(0.0);

  if (ii < wallCount) {
    // Gray wall strip (one elongated quad per segment). Segments are strip
    // centerlines (see buildRenderWalls), drawn 0.5 thick like Unity's sprites.
    // Vertices past the quad collapse to a point (degenerate).
    let w = walls[ii];
    let tangent = w.zw - w.xy;
    let len = length(tangent);
    let dir = tangent / len;
    let nrm = vec2f(-dir.y, dir.x);
    let c = QUAD[min(vi, 5u)];
    world = w.xy + dir * (c.x * 0.5 + 0.5) * len + nrm * c.y * 0.25 * step(f32(vi), 5.0);
    color = vec4f(0.5, 0.5, 0.5, 1.0);
  } else if (ii < wallInstances) {
    // Round joint disc at a segment endpoint (8-triangle fan, radius = half the
    // strip thickness) so corners connect smoothly.
    let ji = ii - wallCount;
    let w = walls[ji / 2u];
    let center = select(w.xy, w.zw, ji % 2u == 1u);
    let tri = vi / 3u;
    let slot = vi % 3u;
    let ang = (f32(tri) + f32(max(slot, 1u) - 1u)) * 0.7853981633974483;
    world = center + vec2f(cos(ang), sin(ang)) * 0.25 * f32(min(slot, 1u));
    color = vec4f(0.5, 0.5, 0.5, 1.0);
  } else if (ii < wallInstances + carCount) {
    // Car quad 1x2 (Unity Car.prefab: 1x1 collider, transform scale (1,2)),
    // forward = +Y at angle 0. Red; the best car green.
    // Dead cars are hidden, except the best one (the camera follows it).
    let ci = ii - wallInstances;
    let car = cars[ci];
    if (car.alive == 1u || ci == cam.bestIndex) {
      let a = car.angle * DEG2RAD;
      let fwd = vec2f(-sin(a), cos(a));
      let right = vec2f(cos(a), sin(a));
      let c = QUAD[vi];
      world = car.pos + right * c.x * 0.5 + fwd * c.y * 1.0;
      color = select(vec4f(0.75, 0.15, 0.15, 1.0), vec4f(0.2, 0.7, 0.25, 1.0), ci == cam.bestIndex);
    }
  } else {
    // Sensor-hit crosses for alive cars: two thin quads per (car, sensor) at the hit point.
    let xi = ii - wallInstances - carCount;
    let ci = xi / 5u;
    let si = xi % 5u;
    let car = cars[ci];
    if (car.alive == 1u) {
      let a = car.angle * DEG2RAD;
      let fwd = vec2f(-sin(a), cos(a));
      let ca = cos(SENSOR_ANGLES[si]);
      let sa = sin(SENSOR_ANGLES[si]);
      let dir = vec2f(fwd.x * ca - fwd.y * sa, fwd.x * sa + fwd.y * ca);
      let right = vec2f(cos(a), sin(a));
      let origin = car.pos + right * SENSOR_ORIGINS[si].x + fwd * SENSOR_ORIGINS[si].y;
      let hit = origin + dir * sensors[ci * 5u + si];
      let rot = select(0.7853981633974483, -0.7853981633974483, vi < 6u);
      let sd = vec2f(cos(rot), sin(rot));
      let sn = vec2f(-sd.y, sd.x);
      let c = QUAD[vi % 6u];
      world = hit + sd * c.x * 0.6 + sn * c.y * 0.1;
      color = vec4f(0.35, 0.35, 0.95, 0.8);
    }
  }

  var output: VertexOutput;
  output.position = vec4f((world - cam.center) / cam.halfSize, 0.0, 1.0);
  output.color = color;
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  return input.color;
}
`;

/** Top-down ortho renderer: walls, instanced cars (best = green), sensor-hit crosses. */
export class Renderer {
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly format: GPUTextureFormat;
  private readonly pipeline: GPURenderPipeline;
  private readonly bindGroup: GPUBindGroup;
  private readonly cameraBuffer: GPUBuffer;
  private readonly carCount: number;
  private readonly wallCount: number;
  private readonly bounds: { minX: number; maxX: number; minY: number; maxY: number };
  private configured = false;

  // Camera state: lerp-follows the best car like CameraMovement.cs (CamSpeed = 5).
  private camX: number;
  private camY: number;
  private targetX: number;
  private targetY: number;
  private bestIndex = 0;
  private halfHeight = 30;
  private lastFrameTime?: number;
  private drag?: { x: number; y: number };
  /** Called when the user starts dragging; wire to disable follow-cam. */
  onPanStart?: () => void;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    gpu: WebGPUState,
    track: Track,
    buffers: SimBuffers,
  ) {
    this.device = gpu.device;
    this.context = gpu.context;
    this.format = gpu.format;
    this.camX = this.targetX = track.start.x;
    this.camY = this.targetY = track.start.y;
    this.carCount = buffers.cars.size / (CAR_FLOATS * 4);

    // Render-only wall strips (centerlines), separate from the sim's edge buffer.
    const renderWalls = buildRenderWalls(track).flat();
    this.wallCount = renderWalls.length / 4;
    const renderWallBuffer = createBufferWithData(
      this.device,
      'render walls',
      new Float32Array(renderWalls),
      GPUBufferUsage.STORAGE,
    );

    const xs = track.walls.flatMap((w) => [w[0], w[2]]);
    const ys = track.walls.flatMap((w) => [w[1], w[3]]);
    this.bounds = {
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      minY: Math.min(...ys),
      maxY: Math.max(...ys),
    };

    this.cameraBuffer = this.device.createBuffer({
      label: 'camera',
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const shaderModule = this.device.createShaderModule({ code: renderShader });
    this.pipeline = this.device.createRenderPipeline({
      label: 'track pipeline',
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
        { binding: 1, resource: { buffer: buffers.cars } },
        { binding: 2, resource: { buffer: renderWallBuffer } },
        { binding: 3, resource: { buffer: buffers.sensors } },
      ],
    });

    canvas.addEventListener(
      'wheel',
      (e) => {
        this.halfHeight = Math.min(120, Math.max(15, this.halfHeight * (e.deltaY > 0 ? 1.1 : 0.9)));
      },
      { passive: true },
    );

    // Drag to pan. Panning implies the user wants camera control, so onPanStart
    // is expected to disable follow-cam.
    canvas.addEventListener('pointerdown', (e) => {
      canvas.setPointerCapture(e.pointerId);
      this.drag = { x: e.clientX, y: e.clientY };
      this.onPanStart?.();
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!this.drag) return;
      const worldPerPixel = (2 * this.halfHeight) / canvas.clientHeight;
      const dx = (e.clientX - this.drag.x) * worldPerPixel;
      const dy = (e.clientY - this.drag.y) * worldPerPixel;
      this.drag = { x: e.clientX, y: e.clientY };
      // Content follows the pointer; keep target in sync so the follow lerp doesn't pull back.
      this.camX = this.targetX = this.camX - dx;
      this.camY = this.targetY = this.camY + dy;
    });
    const endDrag = (): void => {
      this.drag = undefined;
    };
    canvas.addEventListener('pointerup', endDrag);
    canvas.addEventListener('pointercancel', endDrag);
  }

  /** Update the best car (green highlight); optionally aim the camera at it. */
  follow(x: number, y: number, bestIndex: number, moveCamera = true): void {
    this.bestIndex = bestIndex;
    if (!moveCamera) return;
    this.targetX = x;
    this.targetY = y;
  }

  render(): void {
    const resized = resizeCanvasToDisplaySize(this.canvas, this.device.limits.maxTextureDimension2D);
    if (resized || !this.configured) {
      this.context.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });
      this.configured = true;
    }

    // Camera: lerp toward the target, then clamp so the view stays on the track.
    const now = performance.now();
    const dt = this.lastFrameTime === undefined ? 0 : Math.min(0.1, (now - this.lastFrameTime) / 1000);
    this.lastFrameTime = now;
    const t = Math.min(1, 5 * dt);
    this.camX += (this.targetX - this.camX) * t;
    this.camY += (this.targetY - this.camY) * t;

    const halfWidth = (this.halfHeight * this.canvas.width) / this.canvas.height;
    const { minX, maxX, minY, maxY } = this.bounds;
    this.camX = maxX - minX > 2 * halfWidth ? Math.min(maxX - halfWidth, Math.max(minX + halfWidth, this.camX)) : (minX + maxX) / 2;
    this.camY = maxY - minY > 2 * this.halfHeight ? Math.min(maxY - this.halfHeight, Math.max(minY + this.halfHeight, this.camY)) : (minY + maxY) / 2;

    const cameraData = new ArrayBuffer(32);
    new Float32Array(cameraData).set([this.camX, this.camY, halfWidth, this.halfHeight]);
    new Uint32Array(cameraData)[4] = this.bestIndex;
    this.device.queue.writeBuffer(this.cameraBuffer, 0, cameraData);

    const encoder = this.device.createCommandEncoder({ label: 'frame encoder' });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0.96, g: 0.96, b: 0.96, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(24, this.wallCount * 3, 0, 0);
    pass.draw(6, this.carCount, 0, this.wallCount * 3);
    pass.draw(12, this.carCount * 5, 0, this.wallCount * 3 + this.carCount);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}

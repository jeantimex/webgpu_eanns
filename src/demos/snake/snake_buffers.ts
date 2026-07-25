import { createBufferWithData } from '../../webgpu/utils';

/** Genome layout [14 -> 12 -> 3], layer-major row-major with bias rows last: 14x12 + 12 + 12x3 + 3. */
export const SNAKE_GENOME_SIZE = 219;

/** Board is 20x20 cells; world units are cells. */
export const GRID = 16;
export const CELLS = 256;

/**
 * Agent state, 128 f32 = 512 bytes per agent. Raw-indexed (no WGSL struct) so
 * CPU/GPU layouts can't drift. u32 fields are bit patterns (bitcast in WGSL).
 */
export const AGENT_FLOATS = 96;
export const A = {
  headX: 0,
  headY: 1,
  dir: 2, // 0=up, 1=down, 2=left, 3=right
  gameOver: 3,
  length: 4,
  apples: 5,
  moves: 6, // u32
  sinceEat: 7, // moves since last apple; stall timeout
  appleX: 8,
  appleY: 9,
  ringHead: 10, // next write slot in the segment ring
  ringTail: 11, // oldest segment slot
  rng: 12, // u32 xorshift state
  bodyMask: 14, // 8 u32 words: bit (y*16+x) = cell occupied by the snake
  ring: 22, // 64 u32 = 256 bytes, one packed segment (y*16+x) per byte (fits: max 255)
} as const;

export const STALL_MOVES = 200; // no apple for this many moves ends the game
export const MAX_MOVES = 10000; // hard cap per game

/** Static wall blocks (retro map: 3 corner Ls + center C), as [x, y] cells. */
export const OBSTACLES: ReadonlyArray<readonly [number, number]> = [
  // Top-left L
  [2, 2], [3, 2], [4, 2], [2, 3], [2, 4],
  // Top-right L
  [13, 2], [12, 2], [11, 2], [13, 3], [13, 4],
  // Bottom-right L
  [13, 13], [12, 13], [11, 13], [13, 12], [13, 11],
  // Center C
  [6, 5], [7, 5], [8, 5], [6, 6], [6, 7], [7, 7], [8, 7], [9, 7],
];

/** Obstacle bits in body-mask layout (bit y*16+x of 8 u32 words). */
function obstacleMask(): Uint32Array<ArrayBuffer> {
  const bits = new Uint32Array(8);
  for (const [x, y] of OBSTACLES) {
    const cell = y * GRID + x;
    bits[cell >>> 5] |= 1 << (cell & 31);
  }
  return bits;
}

export interface SnakeBuffers {
  params: GPUBuffer;
  genomes: GPUBuffer;
  agents: GPUBuffer;
  readback: GPUBuffer;
}

/**
 * All agents: length-3 snake at the center heading right; apple at a
 * deterministic per-agent spot away from the body; per-agent rng seed.
 */
export function initialAgentStates(count: number): Float32Array<ArrayBuffer> {
  const states = new Float32Array(count * AGENT_FLOATS);
  for (let i = 0; i < count; i++) {
    const o = i * AGENT_FLOATS;
    states[o + A.headX] = 8;
    states[o + A.headY] = 8;
    states[o + A.dir] = 3; // right
    states[o + A.length] = 3;
    states[o + A.ringHead] = 3;
    states[o + A.ringTail] = 0;
    new Uint32Array(states.buffer)[o + A.rng] = ((i + 1) * 2654435761) >>> 0;

    const u32 = new Uint32Array(states.buffer);
    // Static obstacles: OR'd into the body mask, so collision, danger inputs,
    // apple spawning, and rendering all respect them for free.
    const obstacles = obstacleMask();
    for (let w = 0; w < 8; w++) u32[o + A.bodyMask + w] |= obstacles[w];
    // Body (6,8),(7,8),(8,8): mask bits + ring bytes (tail first).
    for (let s = 0; s < 3; s++) {
      const cx = 6 + s;
      const cy = 8;
      const cell = cy * GRID + cx;
      u32[o + A.bodyMask + (cell >>> 5)] |= 1 << (cell & 31);
      const w = o + A.ring + (s >>> 2);
      u32[w] |= cell << ((s & 3) * 8);
    }
    // Apple: deterministic per agent, off the initial body and obstacles.
    let ax = (i * 37 + 13) % GRID;
    let ay = (i * 53 + 7) % GRID;
    for (let k = 0; k < CELLS; k++) {
      const cell = ay * GRID + ax;
      const occupied = (u32[o + A.bodyMask + (cell >>> 5)] & (1 << (cell & 31))) !== 0;
      if (!occupied) break;
      ax = (ax + 1) % GRID;
      if (ax === 0) ay = (ay + 1) % GRID;
    }
    states[o + A.appleX] = ax;
    states[o + A.appleY] = ay;
  }
  return states;
}

export function createSnakeBuffers(device: GPUDevice, populationSize: number): SnakeBuffers {
  // SimParams uniform: agentCount (u32) = 4 bytes, padded to 16.
  const paramsData = new ArrayBuffer(16);
  new Uint32Array(paramsData)[0] = populationSize;

  const genomes = device.createBuffer({
    label: 'snake genomes',
    size: populationSize * SNAKE_GENOME_SIZE * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const agentBytes = populationSize * AGENT_FLOATS * 4;
  const agents = device.createBuffer({
    label: 'snake agents',
    size: agentBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  const readback = device.createBuffer({
    label: 'snake readback',
    size: agentBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  return {
    params: createBufferWithData(device, 'snake params', new Uint32Array(paramsData), GPUBufferUsage.UNIFORM),
    genomes,
    agents,
    readback,
  };
}

import { createBufferWithData } from '../../webgpu/utils';

/** Genome layout [14 -> 12 -> 3], layer-major row-major with bias rows last: 14x12 + 12 + 12x3 + 3. */
export const SNAKE_GENOME_SIZE = 219;

/** Board is 20x20 cells; world units are cells. */
export const GRID = 20;
export const CELLS = GRID * GRID;
export const MASK_WORDS = Math.ceil(CELLS / 32);

/**
 * Agent state. Raw-indexed (no WGSL struct) so CPU/GPU layouts can't drift.
 * u32 fields are bit patterns (bitcast in WGSL).
 */
export const AGENT_FLOATS = 27 + CELLS;
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
  bodyMask: 14, // 13 u32 words: bit (y*20+x) = cell occupied by the snake
  ring: 27, // one u32 cell id per segment
} as const;

export const STALL_MOVES = 200; // no apple for this many moves ends the game
export const MAX_MOVES = 10000; // hard cap per game

/** Static wall blocks (retro map: 3 corner Ls + center C), as [x, y] cells. */
export const OBSTACLES: ReadonlyArray<readonly [number, number]> = [
  // Top-left L
  [2, 2], [3, 2], [4, 2], [2, 3], [2, 4],
  // Top-right L
  [16, 2], [17, 2], [18, 2], [18, 3], [18, 4],
  // Bottom-right L
  [16, 18], [17, 18], [18, 18], [18, 17], [18, 16],
  // Center C
  [8, 9], [9, 9], [10, 9], [8, 10], [9, 10], [10, 10],
  [8, 11], [8, 12], [8, 13], [9, 13], [10, 13], [11, 13], [12, 13], [13, 13],
];

/** Obstacle bits in body-mask layout. */
function obstacleMask(): Uint32Array<ArrayBuffer> {
  const bits = new Uint32Array(MASK_WORDS);
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
 * All agents: length-8 snake near the upper-right heading up; apple at a
 * deterministic per-agent spot away from the body; per-agent rng seed.
 */
export function initialAgentStates(count: number): Float32Array<ArrayBuffer> {
  const states = new Float32Array(count * AGENT_FLOATS);
  for (let i = 0; i < count; i++) {
    const o = i * AGENT_FLOATS;
    states[o + A.headX] = 13;
    states[o + A.headY] = 4;
    states[o + A.dir] = 0; // up
    states[o + A.length] = 8;
    states[o + A.ringHead] = 8;
    states[o + A.ringTail] = 0;
    new Uint32Array(states.buffer)[o + A.rng] = ((i + 1) * 2654435761) >>> 0;

    const u32 = new Uint32Array(states.buffer);
    // Static obstacles: OR'd into the body mask, so collision, danger inputs,
    // apple spawning, and rendering all respect them for free.
    const obstacles = obstacleMask();
    for (let w = 0; w < MASK_WORDS; w++) u32[o + A.bodyMask + w] |= obstacles[w];
    // Body (13,11)..(13,4): mask bits + ring cell ids (tail first).
    for (let s = 0; s < 8; s++) {
      const cx = 13;
      const cy = 11 - s;
      const cell = cy * GRID + cx;
      u32[o + A.bodyMask + (cell >>> 5)] |= 1 << (cell & 31);
      u32[o + A.ring + s] = cell;
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

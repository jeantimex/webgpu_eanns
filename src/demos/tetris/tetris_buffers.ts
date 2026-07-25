import { createBufferWithData } from '../../webgpu/utils';

/** Genome layout [15 -> 8 -> 1], layer-major row-major with bias rows last: 15x8 + 8 + 8x1 + 1. */
export const TETRIS_GENOME_SIZE = 137;

/** Board is 10 cols x 20 rows; world units are cells. */
export const COLS = 10;
export const ROWS = 20;

/**
 * Agent state, 80 f32 = 320 bytes per agent. Raw-indexed (no WGSL struct) so
 * CPU/GPU layouts can't drift. u32 fields are bit patterns (bitcast in WGSL).
 * The board is 200 cells, one byte each (0=empty, 1..7=piece type), packed
 * 4 per u32 at floats 16..65.
 */
export const AGENT_FLOATS = 80;
export const A = {
  score: 0,
  lines: 1,
  level: 2,
  pieces: 3, // pieces locked so far (also the animation tick)
  gameOver: 4,
  curType: 5, // 0..6
  nextType: 6,
  rng: 7, // u32 xorshift
  // Last placement, for the fall animation (cosmetic):
  placeCol: 8,
  placeRot: 9,
  placeLandRow: 10,
  placeType: 11,
  holes: 12, // latest board holes (fitness shaping)
  bump: 13, // latest bumpiness (fitness shaping)
  board: 16, // 50 u32 = 200 bytes, cell (y*10+x) at byte (y*10+x)
} as const;

export const MAX_PIECES = 2000; // hard cap per game

/**
 * One piece sequence shared by ALL agents in a generation (NES single reroll:
 * same as the previous dealt type → reroll once). Same conditions for every
 * genome = selection compares skill, not luck.
 */
export function makeSequence(rng: () => number, length = MAX_PIECES + 2): Uint32Array<ArrayBuffer> {
  const seq = new Uint32Array(length);
  seq[0] = Math.floor(rng() * 7);
  for (let i = 1; i < length; i++) {
    seq[i] = Math.floor(rng() * 7);
    if (seq[i] === seq[i - 1]) seq[i] = Math.floor(rng() * 7);
  }
  return seq;
}

export interface TetrisBuffers {
  params: GPUBuffer;
  genomes: GPUBuffer;
  agents: GPUBuffer;
  sequence: GPUBuffer;
  readback: GPUBuffer;
}

/** All agents: empty board; first two pieces come from the shared sequence. */
export function initialAgentStates(count: number, seq: Uint32Array): Float32Array<ArrayBuffer> {
  const states = new Float32Array(count * AGENT_FLOATS);
  for (let i = 0; i < count; i++) {
    const o = i * AGENT_FLOATS;
    states[o + A.curType] = seq[0];
    states[o + A.nextType] = seq[1];
  }
  return states;
}

export function createTetrisBuffers(device: GPUDevice, populationSize: number): TetrisBuffers {
  // SimParams uniform: agentCount (u32) = 4 bytes, padded to 16.
  const paramsData = new ArrayBuffer(16);
  new Uint32Array(paramsData)[0] = populationSize;

  const genomes = device.createBuffer({
    label: 'tetris genomes',
    size: populationSize * TETRIS_GENOME_SIZE * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const agentBytes = populationSize * AGENT_FLOATS * 4;
  const agents = device.createBuffer({
    label: 'tetris agents',
    size: agentBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  const readback = device.createBuffer({
    label: 'tetris readback',
    size: agentBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  return {
    params: createBufferWithData(device, 'tetris params', new Uint32Array(paramsData), GPUBufferUsage.UNIFORM),
    genomes,
    agents,
    sequence: device.createBuffer({
      label: 'tetris piece sequence',
      size: (MAX_PIECES + 2) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    readback,
  };
}

/** SnakeAI's playable board is 38x38 cells; world units are cells. */
export const GRID = 38;
export const CELLS = GRID * GRID;
export const MASK_WORDS = Math.ceil(CELLS / 32);

/**
 * Agent state. Raw-indexed (no WGSL struct) so CPU/GPU layouts can't drift.
 * u32 fields are bit patterns (bitcast in WGSL).
 */
export const A = {
  headX: 0,
  headY: 1,
  dir: 2, // 0=up, 1=down, 2=left, 3=right, 4=not moving yet
  gameOver: 3,
  length: 4,
  apples: 5,
  moves: 6, // u32
  sinceEat: 7, // SnakeAI move budget: moves left before starvation (200 start, +100/apple, cap 500)
  appleX: 8,
  appleY: 9,
  ringHead: 10, // next write slot in the segment ring
  ringTail: 11, // oldest segment slot
  rng: 12, // u32 xorshift state
  score: 13, // shaped training reward
  bodyMask: 14, // u32 words: bit (y*GRID+x) = cell occupied by the snake
  ring: 14 + MASK_WORDS, // one u32 cell id per segment, tail first
} as const;
export const AGENT_FLOATS = A.ring + CELLS;

export const LIFE_START = 200; // SnakeAI move budget: starting moves
export const LIFE_PER_APPLE = 100; // moves gained per apple
export const LIFE_MAX = 500; // budget cap
export const MAX_MOVES = 10000; // hard cap per game

/**
 * All agents: length-3 snake centered on the board (SnakeAI's start);
 * apple at a deterministic per-agent spot away from the body; per-agent rng seed.
 */
export function initialAgentStates(count: number, seedOffset = 0): Float32Array<ArrayBuffer> {
  const states = new Float32Array(count * AGENT_FLOATS);
  for (let i = 0; i < count; i++) {
    const o = i * AGENT_FLOATS;
    const startX = Math.floor(GRID / 2);
    const startY = Math.floor(GRID / 2);
    states[o + A.headX] = startX;
    states[o + A.headY] = startY;
    states[o + A.dir] = 4; // no velocity until the first decision, like SnakeAI
    states[o + A.length] = 3;
    states[o + A.score] = 3;
    states[o + A.sinceEat] = LIFE_START;
    states[o + A.ringHead] = 3;
    states[o + A.ringTail] = 0;
    new Uint32Array(states.buffer)[o + A.rng] = ((seedOffset + i + 1) * 2654435761) >>> 0;

    const u32 = new Uint32Array(states.buffer);
    // Body is tail first in the ring: (x,y+2),(x,y+1),(x,y/head).
    for (let s = 0; s < 3; s++) {
      const cx = startX;
      const cy = startY + 2 - s;
      const cell = cy * GRID + cx;
      u32[o + A.bodyMask + (cell >>> 5)] |= 1 << (cell & 31);
      u32[o + A.ring + s] = cell;
    }
    // Apple: deterministic per agent, off the initial body.
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

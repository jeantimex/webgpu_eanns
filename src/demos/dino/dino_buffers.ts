/** Dino state, 8 f32 = 32 bytes: y@0, velY@1, alive(u32)@2, score(u32)@3, fitness@4, jumpOutput@5, onGround(u32)@6, pad@7. */
export const DINO_FLOATS = 8;

// World constants from the source repo (dino.html canvas 1000x400, js/dinoscript.js).
// Mirrored as literals in dino.wgsl.ts — keep the two in sync.
export const WORLD_W = 1000;
export const WORLD_H = 400;
export const PLAT_Y = WORLD_H - 100; // platform line; ground sprite drawn 24px above it
export const DINO_X = 100;
export const DINO_W = 89;
export const DINO_H = 94;
export const BASE_SPEED = 7; // px/frame at score 0
export const MAX_SPEED = 17;

export interface ObstacleState {
  /** Left edge x on screen (WORLD_W - scroll). */
  x: number;
  /** Top edge y on screen. */
  y: number;
  /** Rendered width (baseW * multi). */
  w: number;
  h: number;
  /** Source x in the sprite sheet. */
  picX: number;
  /** Distance traveled so far (source repo's `scroll`). */
  scroll: number;
  /** Base width before the 1..3 multiplier (34 small / 49 big). */
  baseW: number;
  small: boolean;
  multi: number;
}

/** All dinos on the ground at the start pose, alive, score 0. */
export function initialDinoStates(count: number): Float32Array<ArrayBuffer> {
  const states = new Float32Array(count * DINO_FLOATS);
  for (let i = 0; i < count; i++) {
    const o = i * DINO_FLOATS;
    states[o] = PLAT_Y - DINO_H; // y
    // velY, fitness, jumpOutput = 0
    new Uint32Array(states.buffer)[o + 2] = 1; // alive (u32 bits, not float 1.0)
    new Uint32Array(states.buffer)[o + 6] = 1; // onGround
  }
  return states;
}

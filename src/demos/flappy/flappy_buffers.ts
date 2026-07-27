/** Bird state, 8 f32 = 32 bytes: pos(2), velY, alive(u32), score(u32), fitness, jumpOutput, pad. */
export const BIRD_FLOATS = 8;
/** Pipe slots in the GPU pipe buffer; oldest pipes are dropped past this. */
export const MAX_PIPES = 16;

// World constants, derived from the source repo's image assets (assets/*.png).
// Mirrored as literals in flappy.wgsl.ts — keep the two in sync.
export const WORLD_W = 288; // bg.png
export const WORLD_H = 512; // bg.png
export const GROUND_Y = WORLD_H - 118; // ground.png height; "actualHeight" in the original
export const BIRD_X = 50;
export const BIRD_W = 38; // bird.png
export const BIRD_H = 26; // bird.png
export const PIPE_W = 52; // pipeDown.png / pipeUp.png
export const PIPE_GAP = 75;
export const PIPE_SPEED = 6;
export const PIPE_SPAWN_FRAMES = 50;
/** First pipe top = 59..242 (original: centerOfPipe in [96.5, 279.5] minus gap/2). */
export const PIPE_TOP_MIN = 59;
export const PIPE_TOP_MAX = 242;

export interface PipeState {
  x: number;
  topY: number;
  bottomY: number;
  width: number;
}

/** All birds at the start pose, alive, score 0. */
export function initialBirdStates(count: number): Float32Array<ArrayBuffer> {
  const states = new Float32Array(count * BIRD_FLOATS);
  for (let i = 0; i < count; i++) {
    const o = i * BIRD_FLOATS;
    states[o] = BIRD_X;
    states[o + 1] = GROUND_Y / 2;
    // velY, fitness, jumpOutput = 0
    new Uint32Array(states.buffer)[o + 3] = 1; // alive (u32 bits, not float 1.0)
  }
  return states;
}

export function uploadFlappyPipes(device: GPUDevice, pipes: GPUBuffer, pipesList: PipeState[]): void {
  if (pipesList.length === 0) return;
  const flat = new Float32Array(pipesList.length * 4);
  for (let i = 0; i < pipesList.length; i++) {
    flat[i * 4] = pipesList[i].x;
    flat[i * 4 + 1] = pipesList[i].topY;
    flat[i * 4 + 2] = pipesList[i].bottomY;
    flat[i * 4 + 3] = pipesList[i].width;
  }
  device.queue.writeBuffer(pipes, 0, flat);
}

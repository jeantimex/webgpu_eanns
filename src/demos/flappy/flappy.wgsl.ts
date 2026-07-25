/**
 * GPU flappy sim: one thread per bird, per-frame physics (60 fps, no dt) mirroring
 * the source repo's Bird.update/jump and Pipe.checkCollision. World constants are
 * literals mirrored from flappy_buffers.ts — keep the two in sync.
 */
export const flappyShader = /* wgsl */ `
struct SimParams {
  birdCount: u32,
  pipeCount: u32,
  pad0: u32,
  pad1: u32,
};

// 32-byte stride (8 f32): see BIRD_FLOATS in flappy_buffers.ts.
struct BirdState {
  pos: vec2f,
  velY: f32,
  alive: u32,
  score: u32,
  fitness: f32,
  jumpOutput: f32,
  pad0: f32,
};

struct Pipe {
  x: f32,
  topY: f32,
  bottomY: f32,
  width: f32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> genomes: array<f32>; // 57 weights per bird [5, 8, 1]
@group(0) @binding(2) var<storage, read_write> birds: array<BirdState>;
@group(0) @binding(3) var<storage, read> pipes: array<Pipe>;

const WORLD_W = 288.0;
const GROUND_Y = 394.0; // 512 - ground.png height
const BIRD_X = 50.0;
const BIRD_HALF_W = 19.0; // bird.png 38x26
const BIRD_HALF_H = 13.0;
const GRAVITY = 0.8;
const JUMP_UPLIFT = -12.0;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.birdCount) { return; }
  var bird = birds[i];
  if (bird.alive == 0u) { return; }

  // 1. Closest upcoming pipe (first whose right edge is still ahead of the bird).
  var closestIdx = params.pipeCount;
  var minDiff = 999999.0;
  for (var p = 0u; p < params.pipeCount; p++) {
    let pipe = pipes[p];
    let diff = pipe.x + pipe.width - BIRD_X;
    if (diff > 0.0 && diff < minDiff) {
      minDiff = diff;
      closestIdx = p;
    }
  }

  // 2. The 5 NN inputs, normalized like the original's p5 map() calls.
  var inputs: array<f32, 5>;
  inputs[3] = clamp(bird.pos.y / GROUND_Y, 0.0, 1.0);
  inputs[4] = clamp((bird.velY + 12.0) / 24.0, 0.0, 1.0);
  if (closestIdx < params.pipeCount) {
    let pipe = pipes[closestIdx];
    inputs[0] = clamp((pipe.x - BIRD_X) / (WORLD_W - BIRD_X), 0.0, 1.0);
    inputs[1] = clamp(pipe.topY / GROUND_Y, 0.0, 1.0);
    inputs[2] = clamp(pipe.bottomY / GROUND_Y, 0.0, 1.0);
  } else {
    inputs[0] = 1.0;
    inputs[1] = 0.5;
    inputs[2] = 0.5;
  }

  // 3. Forward pass [5 -> 8 -> 1]: relu hidden, sigmoid output (TF.js original).
  var offset = i * 57u;
  var hidden: array<f32, 8>;
  for (var h = 0u; h < 8u; h++) {
    var sum = genomes[offset + 40u + h]; // bias row
    for (var k = 0u; k < 5u; k++) {
      sum += inputs[k] * genomes[offset + k * 8u + h];
    }
    hidden[h] = max(sum, 0.0);
  }
  offset += 48u; // 5x8 weights + 8 biases
  var outSum = genomes[offset + 8u]; // output bias
  for (var h = 0u; h < 8u; h++) {
    outSum += hidden[h] * genomes[offset + h];
  }
  let action = 1.0 / (1.0 + exp(-outSum)); // sigmoid
  bird.jumpOutput = action;

  // 4. Physics: jump (also damped, like Bird.jump), then gravity (Bird.update).
  if (action > 0.5) {
    bird.velY += JUMP_UPLIFT;
    bird.velY *= 0.9;
  }
  bird.velY += GRAVITY;
  bird.velY *= 0.9;
  bird.pos.y += bird.velY;
  bird.score += 1u;

  // 5. Death: ceiling/ground, then any pipe (Pipe.checkCollision per pipe).
  if (bird.pos.y - BIRD_HALF_H < 0.0 || bird.pos.y + BIRD_HALF_H > GROUND_Y) {
    bird.alive = 0u;
  }
  for (var p = 0u; p < params.pipeCount; p++) {
    let pipe = pipes[p];
    if (BIRD_X + BIRD_HALF_W >= pipe.x && BIRD_X - BIRD_HALF_W <= pipe.x + pipe.width) {
      if (bird.pos.y - BIRD_HALF_H <= pipe.topY || bird.pos.y + BIRD_HALF_H >= pipe.bottomY) {
        bird.alive = 0u;
      }
    }
  }

  // Fitness = score^2 (the original's normalizeFitness squares the score).
  let sc = f32(bird.score);
  bird.fitness = sc * sc;

  birds[i] = bird;
}
`;

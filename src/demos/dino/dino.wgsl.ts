/**
 * GPU dino sim: one thread per dino, per-frame physics (60 fps, no dt) mirroring
 * the source repo's update() — gravity only while airborne, ground clamp, one
 * shared obstacle. World constants are literals mirrored from dino_buffers.ts.
 * NN designed for this port: [6 -> 8 -> 1], relu hidden, sigmoid output.
 */
export const dinoShader = /* wgsl */ `
struct SimParams {
  dinoCount: u32,
  gamespeed: f32,
  pad0: u32,
  pad1: u32,
};

// 32-byte stride (8 f32): see DINO_FLOATS in dino_buffers.ts.
struct DinoState {
  y: f32,
  velY: f32,
  alive: u32,
  score: u32,
  fitness: f32,
  jumpOutput: f32,
  onGround: u32,
  pad0: f32,
};

struct Obstacle {
  x: f32,
  y: f32,
  w: f32,
  h: f32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> genomes: array<f32>; // 65 weights per dino [6, 8, 1]
@group(0) @binding(2) var<storage, read_write> dinos: array<DinoState>;
@group(0) @binding(3) var<storage, read> obstacle: Obstacle;

const WORLD_W = 1000.0;
const PLAT_Y = 300.0;
const DINO_X = 100.0;
const DINO_W = 89.0;
const DINO_H = 94.0;
const GRAVITY = 0.6;
const JUMP_VEL = -15.0;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.dinoCount) { return; }
  var dino = dinos[i];
  if (dino.alive == 0u) { return; }
  let obs = obstacle;

  // 1. The 6 NN inputs, normalized to [0, 1]:
  //    distance to obstacle, its width & height, dino height above ground,
  //    vertical velocity, and current game speed.
  var inputs: array<f32, 6>;
  inputs[0] = clamp((obs.x - (DINO_X + DINO_W)) / WORLD_W, 0.0, 1.0);
  inputs[1] = clamp(obs.w / 150.0, 0.0, 1.0);
  inputs[2] = clamp(obs.h / 100.0, 0.0, 1.0);
  inputs[3] = clamp((PLAT_Y - (dino.y + DINO_H)) / 200.0, 0.0, 1.0);
  inputs[4] = clamp((dino.velY - JUMP_VEL) / 30.0, 0.0, 1.0);
  inputs[5] = clamp(params.gamespeed / 17.0, 0.0, 1.0);

  // 2. Forward pass [6 -> 8 -> 1]: relu hidden, sigmoid output.
  var offset = i * 65u;
  var hidden: array<f32, 8>;
  for (var h = 0u; h < 8u; h++) {
    var sum = genomes[offset + 48u + h]; // bias row
    for (var k = 0u; k < 6u; k++) {
      sum += inputs[k] * genomes[offset + k * 8u + h];
    }
    hidden[h] = max(sum, 0.0);
  }
  offset += 56u; // 6x8 weights + 8 biases
  var outSum = genomes[offset + 8u]; // output bias
  for (var h = 0u; h < 8u; h++) {
    outSum += hidden[h] * genomes[offset + h];
  }
  let action = 1.0 / (1.0 + exp(-outSum)); // sigmoid
  dino.jumpOutput = action;

  // 3. Physics: jump only from the ground (keyDown in the original), gravity
  //    only while airborne, then ground clamp.
  if (action > 0.5 && dino.onGround == 1u) {
    dino.velY = JUMP_VEL;
  }
  if (dino.onGround == 0u) {
    dino.velY += GRAVITY;
  }
  dino.y += dino.velY;
  dino.onGround = 0u;
  if (dino.y + DINO_H > PLAT_Y) {
    dino.y = PLAT_Y - DINO_H;
    dino.onGround = 1u;
  }
  dino.score += 1u;

  // 4. Collision (pbox: x=100, w=89, h=75; same checks as the original).
  if (obs.x < DINO_X + DINO_W && obs.x + obs.w > DINO_X && dino.y > obs.y - 75.0) {
    dino.alive = 0u;
  }

  // Fitness = score^2 (same convention as the other demos).
  let sc = f32(dino.score);
  dino.fitness = sc * sc;

  dinos[i] = dino;
}
`;

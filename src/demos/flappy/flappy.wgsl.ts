export const flappyShader = /* wgsl */ `
struct SimParams {
  birdCount: u32,
  pipeCount: u32,
  dt: f32,
  frameCounter: u32,
};

struct BirdState {
  pos: vec2f,      // [x, y]
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

const PLAYABLE_HEIGHT = 340.0;
const WORLD_WIDTH = 600.0;
const BIRD_X = 50.0;
const BIRD_RADIUS = 12.0;
const GRAVITY = 0.8;
const JUMP_UPLIFT = -11.0;

fn softSign(x: f32) -> f32 {
  return x / (1.0 + abs(x));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.birdCount) { return; }
  var bird = birds[i];
  if (bird.alive == 0u) { return; }

  // 1. Find the closest upcoming pipe
  var closestIdx = 999999u;
  var minDiff = 999999.0;
  for (var p = 0u; p < params.pipeCount; p++) {
    let pipe = pipes[p];
    let diff = pipe.x + pipe.width - BIRD_X;
    if (diff > 0.0 && diff < minDiff) {
      minDiff = diff;
      closestIdx = p;
    }
  }

  // 2. Prepare 5 Neural Network Inputs (normalized [0, 1])
  var in0 = 1.0;
  var in1 = 0.5;
  var in2 = 0.5;
  var in3 = bird.pos.y / PLAYABLE_HEIGHT;
  var in4 = clamp((bird.velY + 12.0) / 24.0, 0.0, 1.0);

  if (closestIdx < params.pipeCount) {
    let pipe = pipes[closestIdx];
    in0 = clamp((pipe.x + pipe.width - BIRD_X) / WORLD_WIDTH, 0.0, 1.0);
    in1 = clamp(pipe.topY / PLAYABLE_HEIGHT, 0.0, 1.0);
    in2 = clamp(pipe.bottomY / PLAYABLE_HEIGHT, 0.0, 1.0);
  }

  // 3. FNN Forward Pass [5 -> 8 -> 1] (57 floats genome offset)
  var offset = i * 57u;
  var hidden: array<f32, 8>;
  // Layer 1: 5 inputs -> 8 hidden nodes
  for (var h = 0u; h < 8u; h++) {
    var sum = in0 * genomes[offset + 0u * 8u + h]
            + in1 * genomes[offset + 1u * 8u + h]
            + in2 * genomes[offset + 2u * 8u + h]
            + in3 * genomes[offset + 3u * 8u + h]
            + in4 * genomes[offset + 4u * 8u + h]
            + genomes[offset + 40u + h]; // bias row
    hidden[h] = softSign(sum);
  }
  offset += 48u; // 5x8 + 8 bias

  // Layer 2: 8 hidden -> 1 output
  var outSum = 0.0;
  for (var h = 0u; h < 8u; h++) {
    outSum += hidden[h] * genomes[offset + h];
  }
  outSum += genomes[offset + 8u]; // bias
  let action = softSign(outSum);
  bird.jumpOutput = action;

  // 4. Physics & Jump Action
  if (action > 0.0) {
    bird.velY += JUMP_UPLIFT;
    bird.velY *= 0.9;
  }

  bird.velY += GRAVITY;
  bird.velY *= 0.9;
  bird.pos.y += bird.velY;
  bird.score += 1u;

  // 5. Collisions (Ground / Ceiling / Pipes)
  if (bird.pos.y - BIRD_RADIUS < 0.0 || bird.pos.y + BIRD_RADIUS > PLAYABLE_HEIGHT) {
    bird.alive = 0u;
  }

  if (closestIdx < params.pipeCount) {
    let pipe = pipes[closestIdx];
    // Check if bird is horizontally inside the pipe
    if (BIRD_X + BIRD_RADIUS > pipe.x && BIRD_X - BIRD_RADIUS < pipe.x + pipe.width) {
      // Check vertical collision with top or bottom pipe
      if (bird.pos.y - BIRD_RADIUS < pipe.topY || bird.pos.y + BIRD_RADIUS > pipe.bottomY) {
        bird.alive = 0u;
      }
    }
  }

  let sc = f32(bird.score);
  bird.fitness = sc * sc; // Quadratic score fitness

  birds[i] = bird;
}
`;

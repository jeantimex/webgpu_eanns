/**
 * GPU snake sim: one thread per agent, one dispatch per move (turn-based).
 * Body = per-agent bitmask (O(1) collision) + segment ring buffer (tail removal);
 * apples spawn from a per-agent xorshift RNG. State is raw-indexed (see A in
 * snake_buffers.ts) — no WGSL struct, so CPU/GPU layouts cannot drift.
 */
import { AGENT_FLOATS, CELLS, GRID, MASK_WORDS, A } from './snake_buffers';
import { snakeNetwork } from './snake_net';

export const snakeShader = /* wgsl */ `
struct SimParams {
  agentCount: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> genomes: array<f32>; // 740 weights per agent [24, 16, 16, 4]
@group(0) @binding(2) var<storage, read_write> agents: array<f32>;

// Agent layout (must match A in snake_buffers.ts).
const A_HEAD_X = ${A.headX}u;
const A_HEAD_Y = ${A.headY}u;
const A_DIR = ${A.dir}u;
const A_OVER = ${A.gameOver}u;
const A_LENGTH = ${A.length}u;
const A_APPLES = ${A.apples}u;
const A_MOVES = ${A.moves}u;
const A_SINCE_EAT = ${A.sinceEat}u;
const A_APPLE_X = ${A.appleX}u;
const A_APPLE_Y = ${A.appleY}u;
const A_RING_HEAD = ${A.ringHead}u;
const A_RING_TAIL = ${A.ringTail}u;
const A_RNG = ${A.rng}u;
const A_SCORE = ${A.score}u;
const A_MASK = ${A.bodyMask}u;
const A_RING = ${A.ring}u;

const AGENT_FLOATS = ${AGENT_FLOATS}u;
const GENOME_FLOATS = ${snakeNetwork.genomeSize}u;
const GRID = ${GRID};
const GRID_U = ${GRID}u;
const CELLS = ${CELLS}u;
const MASK_WORDS = ${MASK_WORDS}u;
const LIFE_MAX = 500.0;
const MAX_MOVES = 10000.0;

// SnakeAI vision: 8 compass directions (W, NW, N, NE, E, SE, S, SW).
const DIRS = array<vec2i, 8>(
  vec2i(-1, 0), vec2i(-1, -1), vec2i(0, -1), vec2i(1, -1),
  vec2i(1, 0), vec2i(1, 1), vec2i(0, 1), vec2i(-1, 1)
);

fn activate(x: f32) -> f32 {
  return max(0.0, x);
};

fn dirVec(d: u32) -> vec2f {
  switch d {
    case 0u: { return vec2f(0.0, -1.0); }
    case 1u: { return vec2f(0.0, 1.0); }
    case 2u: { return vec2f(-1.0, 0.0); }
    default: { return vec2f(1.0, 0.0); }
  }
}

fn bodyBit(b: u32, cell: u32) -> u32 {
  if (cell >= CELLS) { return 1u; }
  let word = bitcast<u32>(agents[b + A_MASK + (cell >> 5u)]);
  return (word >> (cell & 31u)) & 1u;
}

fn setBodyBit(b: u32, cell: u32, on: bool) {
  let wb = b + A_MASK + (cell >> 5u);
  let mask = 1u << (cell & 31u);
  var word = bitcast<u32>(agents[wb]);
  if (on) { word |= mask; } else { word &= ~mask; }
  agents[wb] = bitcast<f32>(word);
}

fn ringRead(b: u32, idx: u32) -> u32 {
  return bitcast<u32>(agents[b + A_RING + idx]);
}

fn ringWrite(b: u32, idx: u32, val: u32) {
  agents[b + A_RING + idx] = bitcast<f32>(val);
}

fn nextRand(state: u32) -> u32 {
  var x = state;
  x ^= x << 13u;
  x ^= x >> 17u;
  x ^= x << 5u;
  return x;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.agentCount) { return; }
  let b = i * AGENT_FLOATS;
  if (agents[b + A_OVER] > 0.5) { return; }

  var dir = u32(agents[b + A_DIR]);
  let hx = agents[b + A_HEAD_X];
  let hy = agents[b + A_HEAD_Y];
  let ax = agents[b + A_APPLE_X];
  let ay = agents[b + A_APPLE_Y];

  // --- SnakeAI vision (24 inputs): per direction, [food flag, body flag, 1/dist] ---
  var inputs: array<f32, 24>;
  for (var d = 0u; d < 8u; d++) {
    var px = i32(hx);
    var py = i32(hy);
    var foodFlag = 0.0;
    var bodyFlag = 0.0;
    var dist = 0;
    loop {
      px += DIRS[d].x;
      py += DIRS[d].y;
      dist++;
      if (px < 0 || px >= GRID || py < 0 || py >= GRID) { break; }
      if (foodFlag == 0.0 && f32(px) == ax && f32(py) == ay) { foodFlag = 1.0; }
      if (bodyFlag == 0.0 && bodyBit(b, u32(py * GRID + px)) == 1u) { bodyFlag = 1.0; }
    }
    inputs[d * 3u] = foodFlag;
    inputs[d * 3u + 1u] = bodyFlag;
    inputs[d * 3u + 2u] = 1.0 / f32(dist);
  }

  // --- Forward pass [24 -> 16 -> 16 -> 4], ReLU everywhere (SnakeAI). ---
  var offset = i * GENOME_FLOATS;
  var h1: array<f32, 16>;
  for (var h = 0u; h < 16u; h++) {
    var sum = genomes[offset + 384u + h]; // bias row
    for (var k = 0u; k < 24u; k++) {
      sum += inputs[k] * genomes[offset + k * 16u + h];
    }
    h1[h] = activate(sum);
  }
  offset += 400u; // 25x16
  var h2: array<f32, 16>;
  for (var h = 0u; h < 16u; h++) {
    var sum = genomes[offset + 256u + h]; // bias row
    for (var k = 0u; k < 16u; k++) {
      sum += h1[k] * genomes[offset + k * 16u + h];
    }
    h2[h] = activate(sum);
  }
  offset += 272u; // 17x16
  var bestOut = -1e30;
  var newDir = dir;
  for (var j = 0u; j < 4u; j++) {
    var sum = genomes[offset + 64u + j]; // bias row
    for (var h = 0u; h < 16u; h++) {
      sum += h2[h] * genomes[offset + h * 4u + j];
    }
    sum = activate(sum);
    if (sum > bestOut) {
      bestOut = sum;
      newDir = j;
    }
  }
  // Absolute directions with the no-reverse guard (moveUp/Down/Left/Right).
  if (dir > 3u || newDir != (dir ^ 1u)) {
    dir = newDir;
  }
  agents[b + A_DIR] = f32(dir);

  // SnakeAI counts the attempted move before resolving food/collisions.
  agents[b + A_MOVES] = f32(u32(agents[b + A_MOVES]) + 1u);
  agents[b + A_SINCE_EAT] = agents[b + A_SINCE_EAT] - 1.0;

  var grow = false;
  if (hx == ax && hy == ay) {
    // Food is consumed at the start of a tick when the head is on it, matching SnakeAI.
    agents[b + A_APPLES] = agents[b + A_APPLES] + 1.0;
    agents[b + A_LENGTH] = agents[b + A_LENGTH] + 1.0;
    agents[b + A_SCORE] = agents[b + A_SCORE] + 1.0;
    let life = agents[b + A_SINCE_EAT];
    agents[b + A_SINCE_EAT] = select(life + 100.0, LIFE_MAX, life > 400.0);
    grow = true;

    var rng = nextRand(bitcast<u32>(agents[b + A_RNG]));
    var spawn = rng % CELLS;
    for (var k = 0u; k < CELLS; k++) {
      let cand = (spawn + k) % CELLS;
      if (bodyBit(b, cand) == 0u) {
        agents[b + A_APPLE_X] = f32(cand % GRID_U);
        agents[b + A_APPLE_Y] = f32(cand / GRID_U);
        break;
      }
    }
    agents[b + A_RNG] = bitcast<f32>(rng);
  }

  // --- Move ---
  let dv = dirVec(dir);
  let nx = i32(hx + dv.x);
  let ny = i32(hy + dv.y);
  if (nx < 0 || nx >= GRID || ny < 0 || ny >= GRID) {
    agents[b + A_OVER] = 1.0;
    return;
  }
  let cell = u32(ny * GRID + nx);
  let tailCell = ringRead(b, u32(agents[b + A_RING_TAIL]));
  if (bodyBit(b, cell) == 1u && (grow || cell != tailCell)) {
    agents[b + A_OVER] = 1.0;
    return;
  }
  setBodyBit(b, cell, true);
  ringWrite(b, u32(agents[b + A_RING_HEAD]), cell);
  agents[b + A_RING_HEAD] = f32((u32(agents[b + A_RING_HEAD]) + 1u) % CELLS);
  agents[b + A_HEAD_X] = f32(nx);
  agents[b + A_HEAD_Y] = f32(ny);

  if (!grow) {
    // No apple: pop the tail.
    setBodyBit(b, tailCell, false);
    agents[b + A_RING_TAIL] = f32((u32(agents[b + A_RING_TAIL]) + 1u) % CELLS);
  }

  // Move budget: starve at 0 (loops die; eaters keep going).
  if (agents[b + A_SINCE_EAT] <= 0.0 || agents[b + A_MOVES] >= MAX_MOVES || agents[b + A_LENGTH] >= f32(CELLS)) {
    agents[b + A_OVER] = 1.0;
  }
}
`;

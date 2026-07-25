/**
 * GPU snake sim: one thread per agent, one dispatch per move (turn-based).
 * Body = per-agent bitmask (O(1) collision) + segment ring buffer (tail removal);
 * apples spawn from a per-agent xorshift RNG. State is raw-indexed (see A in
 * snake_buffers.ts) — no WGSL struct, so CPU/GPU layouts cannot drift.
 */
import { AGENT_FLOATS, CELLS, GRID, MASK_WORDS, SNAKE_GENOME_SIZE, A } from './snake_buffers';

export const snakeShader = /* wgsl */ `
struct SimParams {
  agentCount: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> genomes: array<f32>; // 243 weights per agent [16, 12, 3]
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
const GENOME_FLOATS = ${SNAKE_GENOME_SIZE}u;
const INPUTS = 16u;
const HIDDEN = 12u;
const OUTPUTS = 3u;
const GRID = ${GRID};
const GRID_U = ${GRID}u;
const CELLS = ${CELLS}u;
const MASK_WORDS = ${MASK_WORDS}u;
const STALL = 200.0;
const MAX_MOVES = 10000.0;

fn dirVec(d: u32) -> vec2f {
  switch d {
    case 0u: { return vec2f(0.0, -1.0); }
    case 1u: { return vec2f(0.0, 1.0); }
    case 2u: { return vec2f(-1.0, 0.0); }
    default: { return vec2f(1.0, 0.0); }
  }
}

fn leftOf(d: u32) -> u32 {
  switch d {
    case 0u: { return 2u; }
    case 2u: { return 1u; }
    case 1u: { return 3u; }
    default: { return 0u; }
  }
}

fn rightOf(d: u32) -> u32 {
  switch d {
    case 0u: { return 3u; }
    case 3u: { return 1u; }
    case 1u: { return 2u; }
    default: { return 0u; }
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

  // --- NN inputs (16) ---
  let dl = leftOf(dir);
  let dr = rightOf(dir);
  let dvS = dirVec(dir);
  let dvL = dirVec(dl);
  let dvR = dirVec(dr);
  let appleVec = vec2f(ax - hx, ay - hy);

  var inputs: array<f32, 16>;
  // 0-2: danger straight/left/right one step out (wall or body)
  let cS = vec2i(i32(hx + dvS.x), i32(hy + dvS.y));
  let cL = vec2i(i32(hx + dvL.x), i32(hy + dvL.y));
  let cR = vec2i(i32(hx + dvR.x), i32(hy + dvR.y));
  inputs[0] = select(0.0, 1.0, cS.x < 0 || cS.x >= GRID || cS.y < 0 || cS.y >= GRID || bodyBit(b, u32(clamp(cS.y * GRID + cS.x, 0, GRID * GRID - 1))) == 1u);
  inputs[1] = select(0.0, 1.0, cL.x < 0 || cL.x >= GRID || cL.y < 0 || cL.y >= GRID || bodyBit(b, u32(clamp(cL.y * GRID + cL.x, 0, GRID * GRID - 1))) == 1u);
  inputs[2] = select(0.0, 1.0, cR.x < 0 || cR.x >= GRID || cR.y < 0 || cR.y >= GRID || bodyBit(b, u32(clamp(cR.y * GRID + cR.x, 0, GRID * GRID - 1))) == 1u);
  // 3-5: free distance straight/left/right (/grid)
  for (var d = 0u; d < 3u; d++) {
    let dv = select(dvS, select(dvL, dvR, d == 2u), d == 1u);
    var dist = 0;
    for (var s = 1; s <= GRID; s++) {
      let cc = i32(hx + dv.x * f32(s));
      let rr = i32(hy + dv.y * f32(s));
      if (cc < 0 || cc >= GRID || rr < 0 || rr >= GRID || bodyBit(b, u32(rr * GRID + cc)) == 1u) { break; }
      dist = s;
    }
  inputs[3u + d] = f32(dist) / f32(GRID);
  }
  // 6-7: apple delta (/grid)
  inputs[6] = clamp((ax - hx) / f32(GRID), -1.0, 1.0);
  inputs[7] = clamp((ay - hy) / f32(GRID), -1.0, 1.0);
  // 8-9: current direction vector
  inputs[8] = dvS.x;
  inputs[9] = dvS.y;
  // 10: length / cells
  inputs[10] = agents[b + A_LENGTH] / f32(CELLS);
  // 11: stall clock (/200)
  inputs[11] = clamp(agents[b + A_SINCE_EAT] / STALL, 0.0, 1.0);
  // 12-13: tail delta (/grid) — long snakes must not lose their own tail
  let tailCell = ringRead(b, u32(agents[b + A_RING_TAIL]));
  inputs[12] = clamp((f32(tailCell % GRID_U) - hx) / f32(GRID), -1.0, 1.0);
  inputs[13] = clamp((f32(tailCell / GRID_U) - hy) / f32(GRID), -1.0, 1.0);
  // 14-15: apple in the snake's local frame. Positive fwd = ahead;
  // positive right = apple is to the right. This makes greedy turns easy.
  inputs[14] = clamp(dot(appleVec, dvS) / f32(GRID), -1.0, 1.0);
  inputs[15] = clamp(dot(appleVec, dvR) / f32(GRID), -1.0, 1.0);

  // --- Forward pass [16 -> 12 -> 3]: relu hidden, linear outputs, argmax turn. ---
  var offset = i * GENOME_FLOATS;
  var hidden: array<f32, 12>;
  for (var h = 0u; h < 12u; h++) {
    var sum = genomes[offset + INPUTS * HIDDEN + h]; // bias row
    for (var k = 0u; k < INPUTS; k++) {
      sum += inputs[k] * genomes[offset + k * HIDDEN + h];
    }
    hidden[h] = max(sum, 0.0);
  }
  offset += (INPUTS + 1u) * HIDDEN;
  var bestOut = -1e30;
  var turn = 1u; // 0=left, 1=straight, 2=right
  for (var j = 0u; j < OUTPUTS; j++) {
    var sum = genomes[offset + HIDDEN * OUTPUTS + j]; // bias row
    for (var h = 0u; h < 12u; h++) {
      sum += hidden[h] * genomes[offset + h * OUTPUTS + j];
    }
    if (sum > bestOut) {
      bestOut = sum;
      turn = j;
    }
  }
  if (turn == 0u) { dir = dl; }
  else if (turn == 2u) { dir = dr; }
  agents[b + A_DIR] = f32(dir);

  // --- Move ---
  let dv = dirVec(dir);
  let oldDist = abs(ax - hx) + abs(ay - hy);
  let nx = i32(hx + dv.x);
  let ny = i32(hy + dv.y);
  if (nx < 0 || nx >= GRID || ny < 0 || ny >= GRID) {
    agents[b + A_OVER] = 1.0;
    return;
  }
  let cell = u32(ny * GRID + nx);
  if (bodyBit(b, cell) == 1u) {
    agents[b + A_OVER] = 1.0;
    return;
  }
  setBodyBit(b, cell, true);
  ringWrite(b, u32(agents[b + A_RING_HEAD]), cell);
  agents[b + A_RING_HEAD] = f32((u32(agents[b + A_RING_HEAD]) + 1u) % CELLS);
  agents[b + A_HEAD_X] = f32(nx);
  agents[b + A_HEAD_Y] = f32(ny);
  let newDist = abs(ax - f32(nx)) + abs(ay - f32(ny));
  agents[b + A_SCORE] = agents[b + A_SCORE] + (oldDist - newDist) * 0.2 - 0.01;

  if (f32(nx) == ax && f32(ny) == ay) {
    // Ate the apple: grow (no tail pop), respawn on a random empty cell.
    agents[b + A_APPLES] = agents[b + A_APPLES] + 1.0;
    agents[b + A_LENGTH] = agents[b + A_LENGTH] + 1.0;
    agents[b + A_SCORE] = agents[b + A_SCORE] + 25.0 + agents[b + A_LENGTH] * 0.25;
    agents[b + A_SINCE_EAT] = 0.0;
    var rng = nextRand(bitcast<u32>(agents[b + A_RNG]));
    agents[b + A_RNG] = bitcast<f32>(rng);
    var spawn = rng % CELLS;
    for (var k = 0u; k < CELLS; k++) {
      let cand = (spawn + k) % CELLS;
      if (bodyBit(b, cand) == 0u) {
        agents[b + A_APPLE_X] = f32(cand % GRID_U);
        agents[b + A_APPLE_Y] = f32(cand / GRID_U);
        break;
      }
    }
  } else {
    // No apple: pop the tail.
    let tc = ringRead(b, u32(agents[b + A_RING_TAIL]));
    setBodyBit(b, tc, false);
    agents[b + A_RING_TAIL] = f32((u32(agents[b + A_RING_TAIL]) + 1u) % CELLS);
  }

  agents[b + A_MOVES] = f32(u32(agents[b + A_MOVES]) + 1u);
  agents[b + A_SINCE_EAT] = agents[b + A_SINCE_EAT] + 1.0;
  if (agents[b + A_SINCE_EAT] > STALL || agents[b + A_MOVES] >= MAX_MOVES || agents[b + A_LENGTH] >= f32(CELLS)) {
    agents[b + A_OVER] = 1.0;
  }
}
`;

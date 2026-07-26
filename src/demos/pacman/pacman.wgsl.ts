/**
 * GPU Pac-Man sim: one thread per agent; each agent owns a full game — pacman,
 * 4 ghosts, pellet field — mirroring pacman-js (MIT): tile coords (integer =
 * tile center), px/ms speeds expressed in tiles/sec, scatter 7s / chase 20s,
 * greedy-flee frightened ghosts, timer-based house release, eyes return.
 * State is raw-indexed (see A in pacman_buffers.ts) — no WGSL struct, so the
 * CPU/GPU layouts cannot drift.
 *
 * The controller follows the EANN-Pacman writeup: an 8-dimensional perception
 * vector feeds a [8 -> 6 ReLU -> 4] net, softmax over the 4 outputs, and the
 * action is *sampled* from that distribution. The one deviation is the metric —
 * distances are maze steps off the precomputed all-pairs table (mazeGraph in
 * maze.ts), not straight-line, because a wall-blind metric has local minima
 * inside wall blocks and makes pacman oscillate between two tiles.
 *
 * Like the snake sim, this shader only *plays* — it carries no reward shaping at
 * all. Fitness is judged on the CPU from what actually happened (see
 * PacmanEvolution.checkAndEvolve).
 */
import {
  A,
  AGENT_FLOATS,
  FRIGHT_SECS,
  MAX_GAME_TICKS,
  PACMAN_GENOME_SIZE,
  PACMAN_HIDDEN,
  PACMAN_INPUTS,
  PAC_SPEED,
  STALL_SECS,
} from './pacman_buffers';
import { COLS, mazeGraph, ROWS, UNREACHABLE } from './maze';

const graph = mazeGraph();

export const pacmanShader = /* wgsl */ `
struct SimParams {
  agentCount: u32,
  episodes: u32, // agents per genome; agent i runs genome i / episodes
  playMode: u32,
  rngSeed: u32,
  // §5.1 environment jitter: re-rolled每generation so strategies must generalise
  // rather than overfit one fixed ghost timing.
  ghostSpeedScale: f32,
  houseReleaseScale: f32,
  actionTemperature: f32, // 0 = take the mode; >0 = sample, lower is sharper
  levelTickLimit: u32, // per-board time budget, adjustable at runtime
  ghostChaos: f32, // chance a ghost takes a random legal turn instead of the best
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> genomes: array<f32>; // [8 -> 6 -> 4] per agent
@group(0) @binding(2) var<storage, read_write> agents: array<f32>;
@group(0) @binding(3) var<storage, read> mazeBits: array<u32>; // 31 words, bit c = wall
@group(0) @binding(4) var<storage, read> initPellets: array<u32>; // 28 words
@group(0) @binding(5) var<storage, read> tileIndex: array<i32>; // tile -> walkable index, -1 = wall
@group(0) @binding(6) var<storage, read> pathDist: array<u32>; // all-pairs maze steps, u8 x4 per word

// Agent layout (must match A in pacman_buffers.ts).
const A_DIR = ${A.dir}u;
const A_DESIRED = ${A.desired}u;
const A_MOVING = ${A.moving}u;
const A_DOTS = ${A.dotsLeft}u;
const A_SCORE = ${A.score}u;
const A_LEVEL = ${A.level}u;
const A_MODET = ${A.modeTimer}u;
const A_PHASE = ${A.phase}u;
const A_FRIGHT = ${A.frightTimer}u;
const A_COMBO = ${A.combo}u;
const A_HOUSET = ${A.houseTimer}u;
const A_RELEASED = ${A.released}u;
const A_OVER = ${A.gameOver}u;
const A_LEVEL_TICKS = ${A.levelTicks}u;
const A_GHOSTS = ${A.ghosts}u; // 4 x [x, y, dir, mode]
const A_TICKS = ${A.ticks}u;
const A_SINCE_EAT = ${A.sinceEat}u;
const A_FRUIT = ${A.fruit}u;
const A_PELLETS = ${A.pellets}u; // 28 u32 words
const A_WASTED = ${A.wasted}u;
const A_RNG = ${A.rng}u;

const AGENT_FLOATS = ${AGENT_FLOATS}u;
const GENOME_SIZE = ${PACMAN_GENOME_SIZE}u;
const INPUTS = ${PACMAN_INPUTS}u;
const HIDDEN = ${PACMAN_HIDDEN}u;
const DT = 0.016666667; // 1/60
const PAC_SPEED = ${PAC_SPEED}.0;
const MAX_TICKS = ${MAX_GAME_TICKS}u;
const STALL_SECS = ${STALL_SECS}.0;

// Maze graph (built once on the CPU, see mazeGraph in maze.ts).
const TILE_COUNT = ${COLS * ROWS}u;
const WALK_STRIDE = ${graph.stride}u;
const UNREACHABLE = ${UNREACHABLE}.0;
const DIAMETER = ${graph.diameter}.0;

// Ghost modes: 0=normal, 1=scared, 2=eyes, 3=idle, 4=leaving.

fn dirVec(d: u32) -> vec2f {
  switch d {
    case 0u: { return vec2f(0.0, -1.0); }
    case 1u: { return vec2f(0.0, 1.0); }
    case 2u: { return vec2f(-1.0, 0.0); }
    default: { return vec2f(1.0, 0.0); }
  }
}

fn opp(d: u32) -> u32 {
  return d ^ 1u;
}

fn roundByDir(v: f32, d: u32) -> f32 {
  if (d == 0u || d == 2u) { return floor(v); }
  return ceil(v);
}

fn isWallCell(c: i32, r: i32) -> bool {
  if (r == 14 && (c < 0 || c > 27)) { return false; }
  if (c < 0 || c > 27 || r < 0 || r > 30) { return true; }
  return ((mazeBits[r] >> u32(c)) & 1u) == 1u;
}

fn isGhostPathCell(c: i32, r: i32) -> bool {
  if (c == 13 && r >= 10 && r <= 14) { return true; }
  if (c == 14 && r >= 10 && r <= 14) { return true; }
  return !isWallCell(c, r);
}

/** Tile -> compact walkable index, wrapping the row-14 tunnel. -1 off-graph. */
fn walkIndex(c: i32, r: i32) -> i32 {
  if (r < 0 || r > 30) { return -1; }
  var cc = c;
  if (r == 14) {
    if (cc < 0) { cc = 27; }
    else if (cc > 27) { cc = 0; }
  }
  if (cc < 0 || cc > 27) { return -1; }
  return tileIndex[u32(r) * 28u + u32(cc)];
}

/** Shortest maze path between two walkable tiles, in steps. */
fn pathSteps(a: i32, b2: i32) -> f32 {
  if (a < 0 || b2 < 0) { return UNREACHABLE; }
  let offset = u32(a) * WALK_STRIDE + u32(b2);
  return f32((pathDist[offset >> 2u] >> ((offset & 3u) * 8u)) & 0xffu);
}

/** xorshift32, one stream per agent, for softmax action sampling. */
fn nextRand(state: u32) -> u32 {
  var x = state;
  x ^= x << 13u;
  x ^= x >> 17u;
  x ^= x << 5u;
  return x;
}

/**
 * Which neighbour starts the shortest path to whatever this array measured.
 * Blocked or unreachable directions hold UNREACHABLE and so never win.
 */
fn argminDir(d: ptr<function, array<f32, 4>>) -> u32 {
  var best = 0u;
  for (var k = 1u; k < 4u; k++) {
    if ((*d)[k] < (*d)[best]) { best = k; }
  }
  return best;
}

fn wallAhead(x: f32, y: f32, d: u32, step: f32) -> bool {
  let dv = dirVec(d);
  let nx = x + dv.x * step;
  let ny = y + dv.y * step;
  return isWallCell(i32(roundByDir(nx, d)), i32(roundByDir(ny, d)));
}

fn warpX(x: f32, y: f32) -> f32 {
  if (abs(y - 14.0) < 0.75) {
    if (x < -0.75) { return 27.75; }
    if (x > 27.75) { return -0.75; }
    return x;
  }
  return clamp(x, 0.0, 27.0);
}

fn keepInMaze(pos: vec2f, previous: vec2f) -> vec2f {
  var p = pos;
  // Row 14 wraps end to end, but that only applies to a ghost which has actually
  // run off one of its ends. Snapping *every* ghost near y=14 onto the row pins
  // vertical traffic at columns 6, 9, 18 and 21 — corridors that cross the
  // tunnel — and a pinned ghost never moves, so it stays snapped, re-decides the
  // same direction every tick and freezes there permanently.
  if (abs(p.y - 14.0) < 0.75 && (p.x < 0.0 || p.x > 27.0)) {
    return vec2f(warpX(p.x, 14.0), 14.0);
  }
  p.x = clamp(p.x, 0.0, 27.0);
  p.y = clamp(p.y, 0.0, 30.0);
  let c = i32(floor(p.x + 0.5));
  let r = i32(floor(p.y + 0.5));
  if (!isGhostPathCell(c, r)) {
    return previous;
  }
  return p;
}

fn keepHouseExit(pos: vec2f, previous: vec2f) -> vec2f {
  var p = pos;
  if (p.y < 10.5) {
    return vec2f(13.5, 10.5);
  }
  if (abs(p.x - 13.5) < 0.35 && p.y <= 14.75) {
    p.x = 13.5;
    p.y = clamp(p.y, 10.5, 14.5);
    return p;
  }
  if (p.y > 13.25 && p.y <= 14.75) {
    p.x = clamp(p.x, 11.0, 16.0);
    p.y = 14.0;
    return p;
  }
  if (p.y <= 13.25) {
    p.x = 13.5;
    p.y = clamp(p.y, 10.5, 14.5);
    return p;
  }
  return keepInMaze(p, previous);
}

fn dist2(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
  let dx = x1 - x2;
  let dy = y1 - y2;
  return dx * dx + dy * dy;
}

fn frightenedDuration(level: f32) -> f32 {
  return max(1.5, ${FRIGHT_SECS}.0 - floor(level - 1.0) * 0.45);
}

fn normalGhostSpeed(level: f32) -> f32 {
  return min(0.95, 0.76 + floor(level - 1.0) * 0.02);
}

fn scaredGhostSpeed(level: f32) -> f32 {
  return max(0.35, 0.5 - floor(level - 1.0) * 0.015);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.agentCount) { return; }
  let b = i * AGENT_FLOATS;
  let isPlayer = params.playMode == 1u && i == params.agentCount - 1u;

  if (agents[b + A_OVER] > 0.5) { return; }

  let ticks = u32(agents[b + A_TICKS]) + 1u;
  agents[b + A_TICKS] = f32(ticks);
  if (ticks > MAX_TICKS) {
    agents[b + A_OVER] = 1.0;
    return;
  }
  let levelTicks = u32(agents[b + A_LEVEL_TICKS]) + 1u;
  agents[b + A_LEVEL_TICKS] = f32(levelTicks);
  if (!isPlayer && levelTicks > params.levelTickLimit) {
    agents[b + A_OVER] = 1.0;
    return;
  }

  // Stall timeout: a pac that stops eating (e.g. camps a corner) ends its game.
  agents[b + A_SINCE_EAT] = agents[b + A_SINCE_EAT] + DT;
  if (!isPlayer && agents[b + A_SINCE_EAT] > STALL_SECS) {
    agents[b + A_OVER] = 1.0;
    return;
  }

  // --- Timers: scatter 7s / chase 20s, fright expiry, house release chain. ---
  agents[b + A_MODET] = agents[b + A_MODET] + DT;
  let phaseLimit = select(20.0, 7.0, agents[b + A_PHASE] < 0.5);
  if (agents[b + A_MODET] >= phaseLimit) {
    agents[b + A_MODET] = 0.0;
    agents[b + A_PHASE] = 1.0 - agents[b + A_PHASE];
    for (var g = 0u; g < 4u; g++) {
      let gb = b + A_GHOSTS + g * 4u;
      if (agents[gb + 3u] == 0.0) { // normal ghosts reverse on mode flip
        agents[gb + 2u] = f32(opp(u32(agents[gb + 2u])));
      }
    }
  }
  if (agents[b + A_FRIGHT] > 0.0) {
    agents[b + A_FRIGHT] = agents[b + A_FRIGHT] - DT;
    if (agents[b + A_FRIGHT] <= 0.0) {
      agents[b + A_FRIGHT] = 0.0;
      for (var g = 0u; g < 4u; g++) {
        let gb = b + A_GHOSTS + g * 4u;
        if (agents[gb + 3u] == 1.0) { agents[gb + 3u] = 0.0; }
      }
    }
  }
  if (agents[b + A_RELEASED] < 3.0) {
    agents[b + A_HOUSET] = agents[b + A_HOUSET] + DT;
    if (agents[b + A_HOUSET] >= 8.0 * params.houseReleaseScale) {
      agents[b + A_HOUSET] = 0.0;
      let gi = u32(agents[b + A_RELEASED]) + 1u; // pinky, then inky, then clyde
      let gb = b + A_GHOSTS + gi * 4u;
      if (agents[gb + 3u] == 3.0) { agents[gb + 3u] = 4.0; }
      agents[b + A_RELEASED] = agents[b + A_RELEASED] + 1.0;
    }
  }

  var px = agents[b];
  var py = agents[b + 1u];
  var pdir = u32(agents[b + A_DIR]);
  let gbase = b + A_GHOSTS;

  // --- Perception, network, action (EANN-Pacman writeup §2.1-2.2). ---
  if (agents[b + A_MOVING] > 0.5) {
    let step = PAC_SPEED * DT;
    let snapped = select(abs(px - round(px)) < step * 0.5, abs(py - round(py)) < step * 0.5, pdir <= 1u);

    var desired = pdir;
    if (isPlayer) {
      let queued = u32(agents[b + A_DESIRED]);
      if ((snapped || queued == opp(pdir)) && !wallAhead(px, py, queued, step)) {
        desired = queued;
      }
    } else if (snapped) {
      // One decision per tile, matching the writeup's step-based simulator.
      let pc = i32(round(px));
      let pr = i32(round(py));
      let curTile = walkIndex(pc, pr);

      var candTile = array<i32, 4>(-1, -1, -1, -1);
      for (var d = 0u; d < 4u; d++) {
        let dv = dirVec(d);
        candTile[d] = select(-1, walkIndex(pc + i32(dv.x), pr + i32(dv.y)), !wallAhead(px, py, d, step));
      }

      // Nearest pellet: distance from here, and which neighbour starts the path.
      var pelletCur = UNREACHABLE;
      var pelletNear = array<f32, 4>(UNREACHABLE, UNREACHABLE, UNREACHABLE, UNREACHABLE);
      for (var w = 0u; w < 28u; w++) {
        var word = bitcast<u32>(agents[b + A_PELLETS + w]);
        while (word != 0u) {
          let bit = firstTrailingBit(word);
          word &= word - 1u;
          let idx = w * 32u + bit;
          if (idx >= TILE_COUNT) { continue; }
          let pt = tileIndex[idx];
          if (pt < 0) { continue; }
          pelletCur = min(pelletCur, pathSteps(curTile, pt));
          for (var d = 0u; d < 4u; d++) {
            if (candTile[d] < 0) { continue; }
            pelletNear[d] = min(pelletNear[d], pathSteps(candTile[d], pt));
          }
        }
      }

      // Nearest power pellet, from the 4 fixed corner tiles.
      var powerCur = UNREACHABLE;
      var powerNear = array<f32, 4>(UNREACHABLE, UNREACHABLE, UNREACHABLE, UNREACHABLE);
      let powerTiles = array<u32, 4>(3u * 28u + 1u, 3u * 28u + 26u, 23u * 28u + 1u, 23u * 28u + 26u);
      for (var q = 0u; q < 4u; q++) {
        let idx = powerTiles[q];
        if (((bitcast<u32>(agents[b + A_PELLETS + (idx >> 5u)]) >> (idx & 31u)) & 1u) == 0u) { continue; }
        let pt = tileIndex[idx];
        powerCur = min(powerCur, pathSteps(curTile, pt));
        for (var d = 0u; d < 4u; d++) {
          if (candTile[d] < 0) { continue; }
          powerNear[d] = min(powerNear[d], pathSteps(candTile[d], pt));
        }
      }

      // The two nearest on-board ghosts. Reporting only the closest one leaves
      // the agent blind to a pincer, so it flees one ghost into the other.
      var ghostCur = UNREACHABLE;
      var ghostTile = -1;
      var ghostState = 0.0;
      var ghost2Cur = UNREACHABLE;
      var ghost2Tile = -1;
      for (var g = 0u; g < 4u; g++) {
        let gb = gbase + g * 4u;
        let gmode = agents[gb + 3u];
        if (gmode != 0.0 && gmode != 1.0) { continue; } // eyes / housebound: not on the board
        let gt = walkIndex(i32(round(agents[gb])), i32(round(agents[gb + 1u])));
        let dHere = pathSteps(curTile, gt);
        if (dHere < ghostCur) {
          ghost2Cur = ghostCur;
          ghost2Tile = ghostTile;
          ghostCur = dHere;
          ghostTile = gt;
          ghostState = select(0.0, 1.0, gmode == 1.0);
        } else if (dHere < ghost2Cur) {
          ghost2Cur = dHere;
          ghost2Tile = gt;
        }
      }
      // Per-direction distance to each of those two specifically, so the
      // direction inputs point at a known ghost rather than a mixture.
      var ghostNear = array<f32, 4>(UNREACHABLE, UNREACHABLE, UNREACHABLE, UNREACHABLE);
      var ghost2Near = array<f32, 4>(UNREACHABLE, UNREACHABLE, UNREACHABLE, UNREACHABLE);
      for (var d = 0u; d < 4u; d++) {
        if (candTile[d] < 0) { continue; }
        ghostNear[d] = pathSteps(candTile[d], ghostTile);
        ghost2Near[d] = pathSteps(candTile[d], ghost2Tile);
      }

      // Directions as unit vectors: "head that way" is then a linear map onto
      // the four outputs, instead of four bumps on one ordinal axis.
      let pelletVec = select(vec2f(0.0), dirVec(argminDir(&pelletNear)), pelletCur < UNREACHABLE);
      let ghostVec = select(vec2f(0.0), dirVec(argminDir(&ghostNear)), ghostCur < UNREACHABLE);
      let ghost2Vec = select(vec2f(0.0), dirVec(argminDir(&ghost2Near)), ghost2Cur < UNREACHABLE);
      let powerVec = select(vec2f(0.0), dirVec(argminDir(&powerNear)), powerCur < UNREACHABLE);

      var inputs: array<f32, INPUTS>;
      inputs[0] = min(pelletCur, DIAMETER) / DIAMETER;
      inputs[1] = pelletVec.x;
      inputs[2] = pelletVec.y;
      inputs[3] = min(ghostCur, DIAMETER) / DIAMETER;
      inputs[4] = ghostVec.x;
      inputs[5] = ghostVec.y;
      inputs[6] = ghostState;
      inputs[7] = min(ghost2Cur, DIAMETER) / DIAMETER;
      inputs[8] = ghost2Vec.x;
      inputs[9] = ghost2Vec.y;
      inputs[10] = powerVec.x;
      inputs[11] = powerVec.y;
      inputs[12] = select(0.0, 1.0, !wallAhead(px, py, pdir, step));
      inputs[13] = (244.0 - agents[b + A_DOTS]) / 244.0;

      // Forward pass [8 -> 6 ReLU -> 4], then softmax, then sample.
      let gBase = (i / params.episodes) * GENOME_SIZE;
      var hidden: array<f32, HIDDEN>;
      for (var h = 0u; h < HIDDEN; h++) {
        var sum = genomes[gBase + INPUTS * HIDDEN + h]; // bias row
        for (var k = 0u; k < INPUTS; k++) {
          sum += inputs[k] * genomes[gBase + k * HIDDEN + h];
        }
        hidden[h] = max(0.0, sum); // ReLU
      }
      let outBase = gBase + (INPUTS + 1u) * HIDDEN;
      var logits: array<f32, 4>;
      var maxLogit = -1e30;
      for (var m = 0u; m < 4u; m++) {
        var sum = genomes[outBase + HIDDEN * 4u + m]; // bias row
        for (var h = 0u; h < HIDDEN; h++) {
          sum += hidden[h] * genomes[outBase + h * 4u + m];
        }
        logits[m] = sum;
        maxLogit = max(maxLogit, sum);
      }
      // Sampling, not argmax: randomness aids exploration (§2.2). Temperature
      // sets how much. Every agent uses the same setting, replay included: this
      // policy leans on noise to break out of loops, so an argmax version of the
      // same genome is a genuinely different and much weaker controller.
      var rngState = nextRand(bitcast<u32>(agents[b + A_RNG]));
      agents[b + A_RNG] = bitcast<f32>(rngState);
      if (params.actionTemperature <= 0.0) {
        var bestLogit = -1e30;
        for (var m = 0u; m < 4u; m++) {
          if (logits[m] > bestLogit) { bestLogit = logits[m]; desired = m; }
        }
      } else {
        var total = 0.0;
        var probs: array<f32, 4>;
        for (var m = 0u; m < 4u; m++) {
          probs[m] = exp(clamp((logits[m] - maxLogit) / params.actionTemperature, -30.0, 0.0));
          total += probs[m];
        }
        var roll = f32(rngState & 0x00ffffffu) / 16777216.0 * total;
        desired = 3u;
        for (var m = 0u; m < 4u; m++) {
          roll -= probs[m];
          if (roll <= 0.0) { desired = m; break; }
        }
      }

      // The net is free to pick a wall — §2.1 gives it an "is the way ahead
      // clear" input and §3.1 charges it a 无意义移动惩罚 instead of masking the
      // move away, so the behaviour has to be learned rather than enforced.
      if (candTile[desired] < 0) {
        agents[b + A_WASTED] = agents[b + A_WASTED] + 1.0;
        desired = pdir;
      }
      agents[b + A_DESIRED] = f32(desired);
    }

    if (snapped) {
      if (wallAhead(px, py, desired, step)) {
        if (pdir <= 1u) { py = round(py); } else { px = round(px); }
      } else {
        if (desired != pdir) {
          // Turning: land exactly on the center before heading off the new axis.
          if (pdir <= 1u) { py = round(py); } else { px = round(px); }
        }
        pdir = desired;
        let dv = dirVec(pdir);
        px += dv.x * step; py += dv.y * step;
      }
    } else if (desired == opp(pdir) && !wallAhead(px, py, desired, step)) {
      // Player U-turn mid-corridor; no centering needed, the axis is unchanged.
      pdir = desired;
      let dv = dirVec(pdir);
      px += dv.x * step; py += dv.y * step;
    } else if (wallAhead(px, py, pdir, step)) {
      if (pdir <= 1u) { py = round(py); } else { px = round(px); }
    } else {
      // Mid-tile: clamp to the next center so no pellet is stepped over.
      let dv = dirVec(pdir);
      let nx = px + dv.x * step;
      let ny = py + dv.y * step;
      if (floor(nx) != floor(px) || floor(ny) != floor(py)) {
        if (pdir <= 1u) { py = roundByDir(py, pdir); } else { px = roundByDir(px, pdir); }
      } else {
        px = nx; py = ny;
      }
    }
    agents[b + A_DIR] = f32(pdir);
    px = warpX(px, py);
    py = clamp(py, 0.0, 30.0);
  }
  agents[b] = px;
  agents[b + 1u] = py;

  // --- Pellet eating (tile containing pac's center). ---
  let ec = i32(floor(px + 0.5));
  let er = i32(floor(py + 0.5));
  if (ec >= 0 && ec <= 27 && er >= 0 && er <= 30) {
    let idx = u32(er) * 28u + u32(ec);
    let w = idx >> 5u;
    let bit = idx & 31u;
    let pb = b + A_PELLETS + w;
    let word = bitcast<u32>(agents[pb]);
    if (((word >> bit) & 1u) == 1u) {
      agents[pb] = bitcast<f32>(word & ~(1u << bit));
      agents[b + A_DOTS] = agents[b + A_DOTS] - 1.0;
      agents[b + A_SINCE_EAT] = 0.0;
      let isPower = (er == 3 || er == 23) && (ec == 1 || ec == 26);
      if (isPower) {
        agents[b + A_SCORE] = agents[b + A_SCORE] + 50.0;
        agents[b + A_FRIGHT] = frightenedDuration(agents[b + A_LEVEL]);
        agents[b + A_COMBO] = 0.0;
        for (var g = 0u; g < 4u; g++) {
          let gb = gbase + g * 4u;
          if (agents[gb + 3u] == 0.0) {
            agents[gb + 3u] = 1.0; // scared
            agents[gb + 2u] = f32(opp(u32(agents[gb + 2u]))); // reverse
          }
        }
      } else {
        agents[b + A_SCORE] = agents[b + A_SCORE] + 10.0;
      }
    }
  }

  // --- Cherry: spawns at 70 and 170 dots eaten, 10s on the board, 100 pts. ---
  let dotsEaten = 244.0 - agents[b + A_DOTS];
  let fruitT = agents[b + A_FRUIT];
  if (fruitT > 0.0) {
    var nt = fruitT - DT;
    if (dist2(px, py, 13.5, 17.0) < 1.0) {
      agents[b + A_SCORE] = agents[b + A_SCORE] + 100.0;
      nt = select(-1.0, -2.0, dotsEaten >= 170.0);
    } else if (nt <= 0.0) {
      nt = select(-1.0, -2.0, dotsEaten >= 170.0);
    }
    agents[b + A_FRUIT] = nt;
  } else if ((fruitT == 0.0 && dotsEaten >= 70.0) || (fruitT == -1.0 && dotsEaten >= 170.0)) {
    agents[b + A_FRUIT] = 10.0;
  }

  // --- Ghosts ---
  for (var g = 0u; g < 4u; g++) {
    let gb = gbase + g * 4u;
    var gx = agents[gb];
    var gy = agents[gb + 1u];
    let oldGhostPos = vec2f(gx, gy);
    var gdir = u32(agents[gb + 2u]);
    var mode = u32(agents[gb + 3u]);

    if (mode == 3u) {
      // Idle in the house: bounce between y 13.5 and 14.5.
      if (gy <= 13.5) { gdir = 1u; }
      else if (gy >= 14.5) { gdir = 0u; }
      let dv = dirVec(gdir);
      gy += dv.y * 0.4 * PAC_SPEED * DT;
    } else if (mode == 4u) {
      // Leaving the house: center on x=13.5, up through the door, then left.
      if (abs(gx - 13.5) < 0.2 && gy <= 11.0) {
        mode = 0u;
        gx = 13.5;
        gy = 10.5;
        gdir = 2u;
      } else if (gx > 13.4 && gx < 13.6) {
        gx = 13.5;
        gdir = 0u;
      } else if (gy > 13.9 && gy < 14.2) {
        gy = 14.0;
        gdir = select(2u, 3u, gx < 13.5);
      }
      let dv = dirVec(gdir);
      gx += dv.x * 0.4 * PAC_SPEED * DT;
      gy += dv.y * 0.4 * PAC_SPEED * DT;
    } else {
      // Speed: eyes 2x, tunnel/house 0.4x, scared 0.5x, else 0.76x of pac speed.
      var mult = normalGhostSpeed(agents[b + A_LEVEL]) * params.ghostSpeedScale;
      if (mode == 2u) { mult = 2.0; }
      else if ((abs(gy - 14.0) < 0.75 && (gx < 6.0 || gx > 21.0)) || (gx > 9.0 && gx < 18.0 && gy > 11.0 && gy < 17.0)) { mult = 0.4; }
      else if (mode == 1u) { mult = scaredGhostSpeed(agents[b + A_LEVEL]); }
      let step = mult * PAC_SPEED * DT;

      let snapped = select(abs(gx - round(gx)) < step * 0.5, abs(gy - round(gy)) < step * 0.5, gdir <= 1u);
      if (snapped) {
        // Decide a new direction at the tile center.
        let gc = i32(round(gx));
        let gr = i32(round(gy));
        // Target tile by ghost id and mode.
        var tx = px;
        var ty = py;
        if (mode == 2u) {
          tx = 13.5; ty = 10.0;
        } else if (agents[b + A_PHASE] < 0.5 && mode != 1u) {
          // Scatter corners: blinky, pinky, inky, clyde.
          tx = select(0.0, 27.0, g == 0u || g == 2u);
          ty = select(30.0, 0.0, g <= 1u);
        } else if (mode == 0u) {
          if (g == 1u) {
            let dv = dirVec(pdir);
            tx = px + 4.0 * dv.x; ty = py + 4.0 * dv.y;
          } else if (g == 2u) {
            let dv = dirVec(pdir);
            let pivotX = px + 2.0 * dv.x;
            let pivotY = py + 2.0 * dv.y;
            tx = 2.0 * pivotX - agents[gbase];
            ty = 2.0 * pivotY - agents[gbase + 1u];
          } else if (g == 3u && dist2(gx, gy, px, py) <= 64.0) {
            tx = 0.0; ty = 30.0;
          }
        }
        // Candidates: open neighbors minus reverse; pick min (scared: max) dist.
        let fromDir = gdir;
        var bestD = select(1e30, -1e30, mode == 1u);
        var bestDir = gdir;
        var count = 0u;
        var legal = array<u32, 4>(0u, 0u, 0u, 0u);
        for (var dd = 0u; dd < 4u; dd++) {
          if (dd == opp(fromDir)) { continue; }
          let dv = dirVec(dd);
          if (isWallCell(gc + i32(dv.x), gr + i32(dv.y))) { continue; }
          legal[count] = dd;
          count++;
          let candX = gx + dv.x;
          let candY = gy + dv.y;
          let dd2 = dist2(candX, candY, tx, ty);
          if (mode == 1u) {
            if (dd2 > bestD) { bestD = dd2; bestDir = dd; }
          } else if (dd2 < bestD) { bestD = dd2; bestDir = dd; }
        }
        if (count > 0u) {
          gdir = bestDir;
          // Arcade ghosts are wholly deterministic, so every game plays out the
          // same way. An occasional random legal turn makes each run differ.
          // The draw comes from the generation seed and the *episode* index, not
          // from this agent's evolving RNG, so every genome in an episode still
          // faces the same ghost behaviour (common random numbers).
          if (params.ghostChaos > 0.0) {
            let roll = nextRand(params.rngSeed ^ ((i % params.episodes) * 0x9e3779b9u) ^ (ticks * 2654435761u) ^ ((g + 1u) * 40503u));
            if (f32(roll & 0xffffu) / 65536.0 < params.ghostChaos) {
              gdir = legal[(roll >> 16u) % count];
            }
          }
        } else if (isWallCell(gc + i32(dirVec(fromDir).x), gr + i32(dirVec(fromDir).y))) {
          gdir = opp(fromDir);
        }
        let dv = dirVec(gdir);
        gx += dv.x * step; gy += dv.y * step;
      } else {
        // Ghost-house hooks for eyes / restored ghosts (handleGhostHouse).
        if (mode == 2u && gy == 11.0 && gx > 13.4 && gx < 13.6) {
          gx = 13.5; gy = 11.0; gdir = 1u;
        } else if (mode == 2u && gx == 13.5 && gy > 13.8 && gy < 14.2) {
          gx = 13.5; gy = 14.0; mode = 0u; gdir = 0u;
        } else if (mode != 2u && gx == 13.5 && gy > 10.8 && gy < 11.0) {
          gx = 13.5; gy = 11.0; gdir = 2u;
        }
        let dv = dirVec(gdir);
        let nx = gx + dv.x * step;
        let ny = gy + dv.y * step;
        if (floor(nx) != floor(gx) || floor(ny) != floor(gy)) {
          if (gdir <= 1u) { gy = roundByDir(gy, gdir); } else { gx = roundByDir(gx, gdir); }
        } else {
          gx = nx; gy = ny;
        }
      }
    }

    let boundedGhostPos = select(select(keepInMaze(vec2f(gx, gy), oldGhostPos), keepHouseExit(vec2f(gx, gy), oldGhostPos), mode == 4u), vec2f(gx, clamp(gy, 13.5, 14.5)), mode == 3u);
    gx = boundedGhostPos.x;
    gy = boundedGhostPos.y;

    agents[gb] = gx;
    agents[gb + 1u] = gy;
    agents[gb + 2u] = f32(gdir);
    agents[gb + 3u] = f32(mode);

    // Collision with pacman (< 1 tile, eyes excluded). One attempt: game over.
    if (mode != 2u && dist2(gx, gy, px, py) < 1.0) {
      if (mode == 1u) {
        let combo = min(agents[b + A_COMBO] + 1.0, 4.0);
        agents[b + A_COMBO] = combo;
        agents[b + A_SCORE] = agents[b + A_SCORE] + 100.0 * pow(2.0, combo);
        agents[gb + 3u] = 2.0; // eyes
      } else {
        agents[b + A_OVER] = 1.0;
      }
    }
  }

  // --- Level clear: advance to the next board, keeping score alive. ---
  if (agents[b + A_DOTS] <= 0.0) {
    agents[b + A_LEVEL] = agents[b + A_LEVEL] + 1.0;
    agents[b] = 13.5;
    agents[b + 1u] = 23.0;
    agents[b + A_DIR] = 2.0;
    agents[b + A_DESIRED] = 2.0;
    agents[b + A_MOVING] = 1.0;
    agents[b + A_DOTS] = 244.0;
    agents[b + A_MODET] = 0.0;
    agents[b + A_PHASE] = 0.0;
    agents[b + A_FRIGHT] = 0.0;
    agents[b + A_COMBO] = 0.0;
    agents[b + A_HOUSET] = 0.0;
    agents[b + A_RELEASED] = 0.0;
    agents[b + A_LEVEL_TICKS] = 0.0;
    agents[b + A_SINCE_EAT] = 0.0;
    agents[b + A_FRUIT] = 0.0;
    agents[b + A_WASTED] = 0.0;
    for (var w = 0u; w < 28u; w++) {
      agents[b + A_PELLETS + w] = bitcast<f32>(initPellets[w]);
    }
    let resetG = b + A_GHOSTS;
    agents[resetG] = 13.5; agents[resetG + 1u] = 11.0; agents[resetG + 2u] = 2.0; agents[resetG + 3u] = 0.0;
    agents[resetG + 4u] = 13.5; agents[resetG + 5u] = 14.0; agents[resetG + 6u] = 1.0; agents[resetG + 7u] = 3.0;
    agents[resetG + 8u] = 11.5; agents[resetG + 9u] = 14.0; agents[resetG + 10u] = 0.0; agents[resetG + 11u] = 3.0;
    agents[resetG + 12u] = 15.5; agents[resetG + 13u] = 14.0; agents[resetG + 14u] = 0.0; agents[resetG + 15u] = 3.0;
    return;
  }
}
`;

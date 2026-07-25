/**
 * GPU Pac-Man sim: one thread per agent; each agent owns a full game — pacman,
 * 4 ghosts, pellet field — mirroring pacman-js (MIT): tile coords (integer =
 * tile center), px/ms speeds expressed in tiles/sec, scatter 7s / chase 20s,
 * greedy-flee frightened ghosts, timer-based house release, eyes return.
 * State is raw-indexed (see A in pacman_buffers.ts) — no WGSL struct, so the
 * CPU/GPU layouts cannot drift.
 */
export const pacmanShader = /* wgsl */ `
struct SimParams {
  agentCount: u32,
  trainCount: u32,
  playMode: u32,
  rngSeed: u32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> genomes: array<f32>; // [16 feature inputs -> 4 direction outputs]
@group(0) @binding(2) var<storage, read_write> agents: array<f32>;
@group(0) @binding(3) var<storage, read> mazeBits: array<u32>; // 31 words, bit c = wall

// Agent layout (must match A in pacman_buffers.ts).
const A_DIR = 2u;
const A_DESIRED = 3u;
const A_MOVING = 4u;
const A_DOTS = 6u;
const A_SCORE = 7u;
const A_LEVEL = 8u;
const A_MODET = 9u;
const A_PHASE = 10u;
const A_FRIGHT = 11u;
const A_COMBO = 12u;
const A_HOUSET = 13u;
const A_RELEASED = 14u;
const A_OVER = 15u;
const A_MOVE_TICKS = 16u;
const A_GHOSTS = 17u; // 4 x [x, y, dir, mode]
const A_TICKS = 33u;
const A_SINCE_EAT = 34u;
const A_FRUIT = 35u;
const A_PELLETS = 36u; // 28 u32 words
const A_PELLET_PROGRESS = 64u;
const A_TOTAL_REWARD = 65u;

const AGENT_FLOATS = 68u;
const DT = 0.016666667; // 1/60
const PAC_SPEED = 11.0;
const MAX_TICKS = 21600u;

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

fn wallAhead(x: f32, y: f32, d: u32, step: f32) -> bool {
  let dv = dirVec(d);
  let nx = x + dv.x * step;
  let ny = y + dv.y * step;
  return isWallCell(i32(roundByDir(nx, d)), i32(roundByDir(ny, d)));
}

fn warpX(x: f32, y: f32) -> f32 {
  if (y > 13.0 && y < 15.0) {
    if (x < -0.75) { return 27.75; }
    if (x > 27.75) { return -0.75; }
    return x;
  }
  return clamp(x, 0.0, 27.0);
}

fn dist2(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
  let dx = x1 - x2;
  let dy = y1 - y2;
  return dx * dx + dy * dy;
}

fn hash(seed: u32) -> u32 {
  var x = seed;
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  x ^= x >> 16u;
  return x;
}

fn rand01(seed: u32) -> f32 {
  return f32(hash(seed) & 0x00ffffffu) / 16777216.0;
}

fn nearestPelletDist(x: f32, y: f32, b: u32) -> f32 {
  var best = 1e9;
  for (var w = 0u; w < 28u; w++) {
    let word = bitcast<u32>(agents[b + A_PELLETS + w]);
    if (word == 0u) { continue; }
    for (var bit = 0u; bit < 32u; bit++) {
      if (((word >> bit) & 1u) == 0u) { continue; }
      let idx = w * 32u + bit;
      let dx = f32(idx % 28u) - x;
      let dy = f32(idx / 28u) - y;
      best = min(best, sqrt(dx * dx + dy * dy));
    }
  }
  if (best > 1e8) { return 0.0; }
  return best;
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

  // Stall timeout: a pac that stops eating (e.g. camps a corner) ends its game.
  agents[b + A_SINCE_EAT] = agents[b + A_SINCE_EAT] + DT;
  if (!isPlayer && agents[b + A_SINCE_EAT] > 8.0) {
    agents[b + A_TOTAL_REWARD] = agents[b + A_TOTAL_REWARD] - 100.0;
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
    if (agents[b + A_HOUSET] >= 8.0) {
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

  let pc = i32(floor(px + 0.5));
  let pr = i32(floor(py + 0.5));

  // --- NN inputs ported from MatheusPaixaoG/Pacman-with-GA. ---
  // Four feature vectors: nearest pellet, nearest power pellet, ghost
  // repulsion, nearest ghost. The first 8 inputs are active in normal mode;
  // the last 8 are active in powered mode, so the network can evolve separate
  // behavior for chasing vs avoiding ghosts.
  var pelletVec = vec2f(0.0);
  var pelletBest = 1e9;
  var powerVec = vec2f(0.0);
  var powerBest = 1e9;
  for (var w = 0u; w < 28u; w++) {
    let word = bitcast<u32>(agents[b + A_PELLETS + w]);
    if (word == 0u) { continue; }
    for (var bit = 0u; bit < 32u; bit++) {
      if (((word >> bit) & 1u) == 0u) { continue; }
      let idx = w * 32u + bit;
      let tx = f32(idx % 28u);
      let ty = f32(idx / 28u);
      let dx = tx - px;
      let dy = ty - py;
      let d = abs(dx) + abs(dy);
      if (d < pelletBest) {
        pelletBest = d;
        let len = max(sqrt(dx * dx + dy * dy), 1e-6);
        pelletVec = vec2f(dx / len, dy / len);
      }
      let isPower = (idx / 28u == 3u || idx / 28u == 23u) && (idx % 28u == 1u || idx % 28u == 26u);
      if (isPower && d < powerBest) {
        powerBest = d;
        let len = max(sqrt(dx * dx + dy * dy), 1e-6);
        powerVec = vec2f(dx / len, dy / len);
      }
    }
  }

  var ghostRepelVec = vec2f(0.0);
  var nearestGhostVec = vec2f(0.0);
  var nearestGhostD = 1e9;
  for (var g = 0u; g < 4u; g++) {
    let gb = gbase + g * 4u;
    let gmode = agents[gb + 3u];
    let gdx = agents[gb] - px;
    let gdy = agents[gb + 1u] - py;
    let gd = gdx * gdx + gdy * gdy;
    if (gmode != 2.0) {
      if (gd < nearestGhostD) {
        nearestGhostD = gd;
        nearestGhostVec = vec2f(gdx, gdy);
      }
      if (gmode != 1.0) {
        let manhattan = abs(gdx) + abs(gdy);
        let len = max(sqrt(gd), 1e-6);
        // Source vector points from each ghost to Pacman, scaled by 100/(1+d).
        ghostRepelVec += vec2f(-gdx / len, -gdy / len) * (100.0 / (1.0 + manhattan));
      }
    }
  }
  if (!isPlayer && nearestGhostD < 36.0) {
    agents[b + A_TOTAL_REWARD] = agents[b + A_TOTAL_REWARD] - (6.0 - sqrt(nearestGhostD)) * 0.08;
  }

  var inputs: array<f32, 16>;
  let inputBase = select(0u, 8u, agents[b + A_FRIGHT] > 0.0);
  inputs[inputBase] = pelletVec.x;
  inputs[inputBase + 1u] = pelletVec.y;
  inputs[inputBase + 2u] = powerVec.x;
  inputs[inputBase + 3u] = powerVec.y;
  inputs[inputBase + 4u] = ghostRepelVec.x;
  inputs[inputBase + 5u] = ghostRepelVec.y;
  inputs[inputBase + 6u] = nearestGhostVec.x;
  inputs[inputBase + 7u] = nearestGhostVec.y;

  var outputs: array<f32, 4>;
  let actionToDir = array<u32, 4>(0u, 3u, 1u, 2u);
  var bestOut = -1e30;
  var bestAction = 0u;
  let genomeBase = i * 68u;
  for (var j = 0u; j < 4u; j++) {
    let candDir = actionToDir[j];
    var sum = genomes[genomeBase + 64u + j];
    for (var k = 0u; k < 16u; k++) {
      sum += inputs[k] * genomes[genomeBase + k * 4u + j];
    }
    // Inertia bias: slightly prefer continuing in current direction
    if (candDir == pdir) { sum += 0.2; }
    // Ghost hazard mask: heavily penalize moving into an immediately adjacent ghost
    let checkDv = dirVec(candDir);
    let checkC = pc + i32(checkDv.x);
    let checkR = pr + i32(checkDv.y);
    for (var g = 0u; g < 4u; g++) {
      let gb = gbase + g * 4u;
      let gc = i32(floor(agents[gb] + 0.5));
      let gr = i32(floor(agents[gb + 1u] + 0.5));
      if (gc == checkC && gr == checkR && agents[gb + 3u] != 2.0 && agents[gb + 3u] != 1.0) {
        sum -= 5.0; // heavy danger penalty
      }
    }
    outputs[j] = sum;
    if (sum > bestOut) {
      bestOut = sum;
      bestAction = j;
    }
  }
  var desired = actionToDir[bestAction];
  if (isPlayer) {
    desired = u32(agents[b + A_DESIRED]);
  }

  // --- Pacman movement (handleSnapped/UnsnappedMovement in the repo). ---
  if (agents[b + A_MOVING] > 0.5) {
    let oldPx = px;
    let oldPy = py;
    let oldPelletDist = nearestPelletDist(px, py, b);
    let step = PAC_SPEED * DT;
    let snapped = select(abs(px - round(px)) < step * 0.5, abs(py - round(py)) < step * 0.5, pdir <= 1u);
    let isCloseGhost = nearestGhostD < 16.0; // ghost within 4 tiles
    if (snapped || isCloseGhost) {
      let exploreSeed = params.rngSeed ^ (i * 747796405u) ^ (ticks * 2891336453u);
      if (i < params.trainCount && rand01(exploreSeed) < 0.08) {
        var legalCount = 0u;
        for (var a = 0u; a < 4u; a++) {
          if (!wallAhead(px, py, actionToDir[a], step)) { legalCount++; }
        }
        if (legalCount > 0u) {
          let pickIndex = u32(floor(rand01(exploreSeed ^ 0xa511e9b3u) * f32(legalCount)));
          var seen = 0u;
          for (var a = 0u; a < 4u; a++) {
            let cand = actionToDir[a];
            if (!wallAhead(px, py, cand, step)) {
              if (seen == pickIndex) { desired = cand; }
              seen++;
            }
          }
        }
      }
      if (wallAhead(px, py, desired, step)) {
        if (isPlayer) {
          // Queued turn: keep the current heading until the turn opens up.
          desired = pdir;
        } else {
          var legalBest = -1e30;
          for (var a = 0u; a < 4u; a++) {
            let cand = actionToDir[a];
            if (!wallAhead(px, py, cand, step) && outputs[a] > legalBest) {
              legalBest = outputs[a];
              desired = cand;
            }
          }
        }
      }
      if (!isPlayer) {
        agents[b + A_DESIRED] = f32(desired);
      }
      if (wallAhead(px, py, desired, step)) {
        if (pdir <= 1u) { py = round(py); } else { px = round(px); }
      } else {
        if (desired != pdir) {
          if (pdir <= 1u) { py = round(py); } else { px = round(px); }
        }
        pdir = desired;
        agents[b + A_DIR] = f32(pdir);
        let dv = dirVec(pdir);
        px += dv.x * step; py += dv.y * step;
      }
    } else {
      // Mid-tile reversal is a player-only privilege. AI agents reverse only at
      // snap points (or near ghosts via isCloseGhost above): instant reversal
      // lets a flip-flopping net jitter between two tiles and starve.
      if (isPlayer && desired == opp(pdir) && !wallAhead(px, py, desired, step)) {
        pdir = desired;
        agents[b + A_DIR] = f32(pdir);
        let dv = dirVec(pdir);
        px += dv.x * step; py += dv.y * step;
      } else if (wallAhead(px, py, pdir, step)) {
        if (pdir <= 1u) { py = round(py); } else { px = round(px); }
      } else {
        let dv = dirVec(pdir);
        let nx = px + dv.x * step;
        let ny = py + dv.y * step;
        if (floor(nx) != floor(px) || floor(ny) != floor(py)) {
          if (pdir <= 1u) { py = roundByDir(py, pdir); } else { px = roundByDir(px, pdir); }
        } else {
          px = nx; py = ny;
        }
      }
    }
    px = warpX(px, py);
    py = clamp(py, 0.0, 30.0);
    if (abs(px - oldPx) > 0.001 || abs(py - oldPy) > 0.001) {
      agents[b + A_MOVE_TICKS] = agents[b + A_MOVE_TICKS] + 1.0;
      let newPelletDist = nearestPelletDist(px, py, b);
      let progress = oldPelletDist - newPelletDist;
      agents[b + A_PELLET_PROGRESS] = agents[b + A_PELLET_PROGRESS] + max(progress, 0.0);
      agents[b + A_TOTAL_REWARD] = agents[b + A_TOTAL_REWARD] + clamp(progress, -0.05, 0.25);
    }
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
        agents[b + A_TOTAL_REWARD] = agents[b + A_TOTAL_REWARD] + 50.0;
        agents[b + A_FRIGHT] = 6.0;
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
        agents[b + A_TOTAL_REWARD] = agents[b + A_TOTAL_REWARD] + 10.0;
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
      agents[b + A_TOTAL_REWARD] = agents[b + A_TOTAL_REWARD] + 100.0;
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
      if (gx == 13.5 && gy > 10.8 && gy < 11.0) {
        mode = 0u;
        gy = 10.5;
        gdir = 2u;
      } else if (gx > 13.4 && gx < 13.6) {
        gx = 13.5;
        gdir = 0u;
      } else if (gy > 13.9 && gy < 14.2) {
        gy = 14.0;
        gdir = select(3u, 2u, gx < 13.5);
      }
      let dv = dirVec(gdir);
      gx += dv.x * 0.4 * PAC_SPEED * DT;
      gy += dv.y * 0.4 * PAC_SPEED * DT;
    } else {
      // Speed: eyes 2x, tunnel/house 0.4x, scared 0.5x, else 0.76x of pac speed.
      var mult = 0.76;
      if (mode == 2u) { mult = 2.0; }
      else if ((gy == 14.0 && (gx < 6.0 || gx > 21.0)) || (gx > 9.0 && gx < 18.0 && gy > 11.0 && gy < 17.0)) { mult = 0.4; }
      else if (mode == 1u) { mult = 0.5; }
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
        var bestD = select(1e30, -1e30, mode == 1u);
        var bestDir = gdir;
        var count = 0u;
        for (var dd = 0u; dd < 4u; dd++) {
          if (dd == opp(gdir)) { continue; }
          let dv = dirVec(dd);
          if (isWallCell(gc + i32(dv.x), gr + i32(dv.y))) { continue; }
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
        } else if (isWallCell(gc + i32(dirVec(gdir).x), gr + i32(dirVec(gdir).y))) {
          gdir = opp(gdir);
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
      gx = warpX(gx, gy);
      gy = clamp(gy, 0.0, 30.0);
    }

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
        agents[b + A_TOTAL_REWARD] = agents[b + A_TOTAL_REWARD] + 100.0 * pow(2.0, combo);
        agents[gb + 3u] = 2.0; // eyes
      } else {
        agents[b + A_TOTAL_REWARD] = agents[b + A_TOTAL_REWARD] - 1000.0;
        agents[b + A_OVER] = 1.0;
      }
    }
  }

  // --- Level clear: terminal success for training. ---
  if (agents[b + A_DOTS] <= 0.0) {
    agents[b + A_TOTAL_REWARD] = agents[b + A_TOTAL_REWARD] + 1000.0 + f32(MAX_TICKS - ticks) * 0.1;
    agents[b + A_OVER] = 1.0;
    return;
  }
}
`;

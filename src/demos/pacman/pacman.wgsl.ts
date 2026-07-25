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
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> genomes: array<f32>; // 256 weights per agent [16, 12, 4]
@group(0) @binding(2) var<storage, read_write> agents: array<f32>;
@group(0) @binding(3) var<storage, read> mazeBits: array<u32>; // 31 words, bit c = wall
@group(0) @binding(4) var<storage, read> initPellets: array<u32>; // 28 words

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
const A_GHOSTS = 17u; // 4 x [x, y, dir, mode]
const A_TICKS = 33u;
const A_SINCE_EAT = 34u;
const A_FRUIT = 35u;
const A_PELLETS = 36u; // 28 u32 words

const AGENT_FLOATS = 64u;
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
  if (c < 0 || c > 27 || r < 0 || r > 30) { return false; }
  return ((mazeBits[r] >> u32(c)) & 1u) == 1u;
}

fn wallAhead(x: f32, y: f32, d: u32, step: f32) -> bool {
  let dv = dirVec(d);
  let nx = x + dv.x * step;
  let ny = y + dv.y * step;
  return isWallCell(i32(roundByDir(nx, d)), i32(roundByDir(ny, d)));
}

fn warpX(x: f32) -> f32 {
  if (x < -0.75) { return 27.75; }
  if (x > 27.75) { return -0.75; }
  return x;
}

fn dist2(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
  let dx = x1 - x2;
  let dy = y1 - y2;
  return dx * dx + dy * dy;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.agentCount) { return; }
  let b = i * AGENT_FLOATS;

  if (agents[b + A_OVER] > 0.5) { return; }

  let ticks = u32(agents[b + A_TICKS]) + 1u;
  agents[b + A_TICKS] = f32(ticks);
  if (ticks > MAX_TICKS) {
    agents[b + A_OVER] = 1.0;
    return;
  }

  // Stall timeout: a pac that stops eating (e.g. camps a corner) ends its game.
  agents[b + A_SINCE_EAT] = agents[b + A_SINCE_EAT] + DT;
  if (agents[b + A_SINCE_EAT] > 20.0) {
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

  // --- NN inputs (16) ---
  var inputs: array<f32, 17>;
  // 0-3: wall distance up/down/left/right from current tile (tiles, /10)
  let pc = i32(floor(px + 0.5));
  let pr = i32(floor(py + 0.5));
  for (var d = 0u; d < 4u; d++) {
    let dv = dirVec(d);
    var dist = 0;
    for (var s = 1; s <= 10; s++) {
      if (isWallCell(pc + i32(dv.x) * s, pr + i32(dv.y) * s)) { break; }
      dist = s;
    }
    inputs[d] = f32(dist) / 10.0;
  }
  // 4-5: nearest pellet dx, dy (/10)
  var bestDot = 9999.0;
  var dotDx = 0.0;
  var dotDy = 0.0;
  for (var w = 0u; w < 28u; w++) {
    let word = bitcast<u32>(agents[b + A_PELLETS + w]);
    if (word == 0u) { continue; }
    for (var bit = 0u; bit < 32u; bit++) {
      if (((word >> bit) & 1u) == 0u) { continue; }
      let idx = w * 32u + bit;
      let cc = f32(idx % 28u) - px;
      let rr = f32(idx / 28u) - py;
      let dd = cc * cc + rr * rr;
      if (dd < bestDot) {
        bestDot = dd;
        dotDx = cc;
        dotDy = rr;
      }
    }
  }
  inputs[4] = clamp(dotDx / 10.0, -1.0, 1.0);
  inputs[5] = clamp(dotDy / 10.0, -1.0, 1.0);
  // 6-7: nearest power pellet dx, dy (/10)
  var bestPow = 9999.0;
  var powDx = 0.0;
  var powDy = 0.0;
  let powTiles = array<vec2f, 4>(vec2f(1.0, 3.0), vec2f(26.0, 3.0), vec2f(1.0, 23.0), vec2f(26.0, 23.0));
  for (var p = 0u; p < 4u; p++) {
    let idx = u32(powTiles[p].y) * 28u + u32(powTiles[p].x);
    let word = bitcast<u32>(agents[b + A_PELLETS + (idx >> 5u)]);
    if (((word >> (idx & 31u)) & 1u) == 0u) { continue; }
    let cc = powTiles[p].x - px;
    let rr = powTiles[p].y - py;
    let dd = cc * cc + rr * rr;
    if (dd < bestPow) {
      bestPow = dd;
      powDx = cc;
      powDy = rr;
    }
  }
  inputs[6] = clamp(powDx / 10.0, -1.0, 1.0);
  inputs[7] = clamp(powDy / 10.0, -1.0, 1.0);
  // 8-11: two nearest active ghosts dx, dy; 12-13: nearest scared ghost dx, dy
  var g1d = 9999.0; var g2d = 9999.0; var gsd = 9999.0;
  var g1 = vec2f(0.0, 0.0); var g2 = vec2f(0.0, 0.0); var gs = vec2f(0.0, 0.0);
  for (var g = 0u; g < 4u; g++) {
    let gb = gbase + g * 4u;
    let mode = agents[gb + 3u];
    let rel = vec2f(agents[gb] - px, agents[gb + 1u] - py);
    let dd = rel.x * rel.x + rel.y * rel.y;
    if (mode == 0.0) {
      if (dd < g1d) { g2d = g1d; g2 = g1; g1d = dd; g1 = rel; }
      else if (dd < g2d) { g2d = dd; g2 = rel; }
    } else if (mode == 1.0 && dd < gsd) {
      gsd = dd;
      gs = rel;
    }
  }
  inputs[8] = clamp(g1.x / 10.0, -1.0, 1.0);
  inputs[9] = clamp(g1.y / 10.0, -1.0, 1.0);
  inputs[10] = clamp(g2.x / 10.0, -1.0, 1.0);
  inputs[11] = clamp(g2.y / 10.0, -1.0, 1.0);
  inputs[12] = clamp(gs.x / 10.0, -1.0, 1.0);
  inputs[13] = clamp(gs.y / 10.0, -1.0, 1.0);
  // 14-15: fright remaining, scatter/chase phase
  inputs[14] = clamp(agents[b + A_FRIGHT] / 6.0, 0.0, 1.0);
  inputs[15] = agents[b + A_PHASE];
  // 16: stall clock (seconds since last pellet / 20) — eating urgency.
  inputs[16] = clamp(agents[b + A_SINCE_EAT] / 20.0, 0.0, 1.0);

  // --- Forward pass [17 -> 12 -> 4]: relu hidden, linear outputs, argmax. ---
  var offset = i * 268u;
  var hidden: array<f32, 12>;
  for (var h = 0u; h < 12u; h++) {
    var sum = genomes[offset + 204u + h]; // bias row
    for (var k = 0u; k < 17u; k++) {
      sum += inputs[k] * genomes[offset + k * 12u + h];
    }
    hidden[h] = max(sum, 0.0);
  }
  offset += 216u; // 17x12 weights + 12 biases
  var bestOut = -1e30;
  var desired = pdir;
  for (var j = 0u; j < 4u; j++) {
    var sum = genomes[offset + 48u + j]; // bias row
    for (var h = 0u; h < 12u; h++) {
      sum += hidden[h] * genomes[offset + h * 4u + j];
    }
    if (sum > bestOut) {
      bestOut = sum;
      desired = j;
    }
  }
  agents[b + A_DESIRED] = f32(desired);

  // --- Pacman movement (handleSnapped/UnsnappedMovement in the repo). ---
  if (agents[b + A_MOVING] > 0.5) {
    let step = PAC_SPEED * DT;
    let snapped = select(px == floor(px), py == floor(py), pdir <= 1u);
    if (snapped) {
      if (wallAhead(px, py, desired, step)) {
        // Desired is blocked: keep the current direction if open, else wait a
        // tick (the NN may pick a legal direction next tick — never freeze).
        if (!wallAhead(px, py, pdir, step)) {
          let dv = dirVec(pdir);
          px += dv.x * step; py += dv.y * step;
        }
      } else {
        pdir = desired;
        agents[b + A_DIR] = f32(pdir);
        let dv = dirVec(pdir);
        px += dv.x * step; py += dv.y * step;
      }
    } else {
      if (desired == opp(pdir)) {
        pdir = desired;
        agents[b + A_DIR] = f32(pdir);
        let dv = dirVec(pdir);
        px += dv.x * step; py += dv.y * step;
      } else {
        let dv = dirVec(pdir);
        let nx = px + dv.x * step;
        let ny = py + dv.y * step;
        if (floor(nx) != floor(px) || floor(ny) != floor(py)) {
          // Crossed a tile boundary: snap the movement axis (cornering assist).
          if (pdir <= 1u) { py = roundByDir(py, pdir); } else { px = roundByDir(px, pdir); }
        } else {
          px = nx; py = ny;
        }
      }
    }
    px = warpX(px);
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

      let snapped = select(gx == floor(gx), gy == floor(gy), gdir <= 1u);
      if (snapped) {
        // Decide a new direction at the tile center.
        let gc = i32(gx);
        let gr = i32(gy);
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
        if (count > 0u) { gdir = bestDir; }
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
      gx = warpX(gx);
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
        agents[gb + 3u] = 2.0; // eyes
      } else {
        agents[b + A_OVER] = 1.0;
      }
    }
  }

  // --- Level clear: restore pellets, bump level, respawn positions. ---
  if (agents[b + A_DOTS] <= 0.0) {
    agents[b + A_LEVEL] = agents[b + A_LEVEL] + 1.0;
    agents[b + A_DOTS] = 244.0;
    agents[b + A_FRUIT] = 0.0;
    agents[b + A_SINCE_EAT] = 0.0;
    for (var w = 0u; w < 28u; w++) {
      agents[b + A_PELLETS + w] = bitcast<f32>(initPellets[w]);
    }
    agents[b] = 13.5; agents[b + 1u] = 23.0;
    agents[b + A_DIR] = 2.0; agents[b + A_DESIRED] = 2.0; agents[b + A_MOVING] = 1.0;
    let g = b + A_GHOSTS;
    agents[g] = 13.5; agents[g + 1u] = 11.0; agents[g + 2u] = 2.0; agents[g + 3u] = 0.0;
    agents[g + 4u] = 13.5; agents[g + 5u] = 14.0; agents[g + 6u] = 1.0; agents[g + 7u] = 3.0;
    agents[g + 8u] = 11.5; agents[g + 9u] = 14.0; agents[g + 10u] = 0.0; agents[g + 11u] = 3.0;
    agents[g + 12u] = 15.5; agents[g + 13u] = 14.0; agents[g + 14u] = 0.0; agents[g + 15u] = 3.0;
    agents[b + A_MODET] = 0.0; agents[b + A_PHASE] = 0.0;
    agents[b + A_FRIGHT] = 0.0; agents[b + A_COMBO] = 0.0;
    agents[b + A_HOUSET] = 0.0; agents[b + A_RELEASED] = 0.0;
  }
}
`;

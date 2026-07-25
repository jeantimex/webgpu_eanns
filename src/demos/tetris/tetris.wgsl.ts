import { LINE_POINTS, wgslPieceTable } from './tetris_pieces';

/**
 * GPU Tetris sim: one thread per agent, one dispatch per piece placement.
 * The NN is a board-state EVALUATOR (the classic Dellacherie structure): the
 * sim enumerates every legal (column, rotation) for the current piece and the
 * network scores the resulting board — inputs include "lines cleared by this
 * move", so line clears are directly valued by the genome, not left to luck.
 * NES scoring (40/100/300/1200 x level+1, level-up every 10 lines).
 * Board = 200 packed type bytes per agent; state is raw-indexed (see A in
 * tetris_buffers.ts) — no WGSL struct, so CPU/GPU layouts cannot drift.
 */
export const tetrisShader = /* wgsl */ `
struct SimParams {
  agentCount: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> genomes: array<f32>; // 137 weights per agent [15, 8, 1]
@group(0) @binding(2) var<storage, read_write> agents: array<f32>;
@group(0) @binding(3) var<storage, read> sequence: array<u32>; // shared piece order for this generation

${wgslPieceTable()}
const LINE_POINTS = array<i32, 5>(${LINE_POINTS.join(',')});

// Agent layout (must match A in tetris_buffers.ts).
const A_SCORE = 0u;
const A_LINES = 1u;
const A_LEVEL = 2u;
const A_PIECES = 3u;
const A_OVER = 4u;
const A_CUR = 5u;
const A_NEXT = 6u;
const A_PCOL = 8u;
const A_PROT = 9u;
const A_PLAND = 10u;
const A_PTYPE = 11u;
const A_BOARD = 16u; // 50 u32 = 200 bytes

const AGENT_FLOATS = 80u;
const MAX_PIECES = 2000.0;

fn getType(b: u32, x: i32, y: i32) -> u32 {
  let idx = u32(y * 10 + x);
  let word = bitcast<u32>(agents[b + A_BOARD + (idx >> 2u)]);
  return (word >> ((idx & 3u) * 8u)) & 0xffu;
}

fn setType(b: u32, x: i32, y: i32, t: u32) {
  let idx = u32(y * 10 + x);
  let wb = b + A_BOARD + (idx >> 2u);
  let shift = (idx & 3u) * 8u;
  var word = bitcast<u32>(agents[wb]);
  word = (word & ~(0xffu << shift)) | (t << shift);
  agents[wb] = bitcast<f32>(word);
}

fn pieceCell(ptype: i32, rot: i32, k: u32) -> vec2i {
  let v = PIECES[(ptype * 4 + rot) * 2 + i32(k >> 1u)];
  let c = (k & 1u) * 2u;
  return vec2i(v[c], v[c + 1u]);
}

// Score one candidate's resulting board with the genome [15 -> 8 -> 1].
fn evaluate(b: u32, inputs: array<f32, 15>) -> f32 {
  var offset = b / AGENT_FLOATS * 137u;
  var hidden: array<f32, 8>;
  for (var h = 0u; h < 8u; h++) {
    var sum = genomes[offset + 120u + h]; // bias row
    for (var k = 0u; k < 15u; k++) {
      sum += inputs[k] * genomes[offset + k * 8u + h];
    }
    hidden[h] = max(sum, 0.0);
  }
  offset += 128u; // 15x8 weights + 8 biases
  var sum = genomes[offset + 8u]; // output bias
  for (var h = 0u; h < 8u; h++) {
    sum += hidden[h] * genomes[offset + h];
  }
  return sum;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.agentCount) { return; }
  let b = i * AGENT_FLOATS;
  if (agents[b + A_OVER] > 0.5) { return; }
  if (agents[b + A_PIECES] >= MAX_PIECES) {
    agents[b + A_OVER] = 1.0;
    return;
  }

  let curType = i32(sequence[u32(agents[b + A_PIECES])]);

  // --- Current board scan: column heights, row fill counts, holes. ---
  var heights: array<i32, 10>;
  var rowFill: array<i32, 20>;
  var holes = 0;
  for (var c = 0; c < 10; c++) {
    var top = 20;
    for (var r = 0; r < 20; r++) {
      if (getType(b, c, r) != 0u) {
        top = r;
        break;
      }
    }
    heights[c] = 20 - top;
    if (top < 20) {
      for (var r = top + 1; r < 20; r++) {
        if (getType(b, c, r) == 0u) { holes++; }
      }
    }
  }
  for (var r = 0; r < 20; r++) {
    var fill = 0;
    for (var c = 0; c < 10; c++) {
      if (getType(b, c, r) != 0u) { fill++; }
    }
    rowFill[r] = fill;
  }

  // --- Evaluate every legal (rotation, column) on its resulting board. ---
  var bestScore = -1e30;
  var bestCol = 0;
  var bestRot = 0;
  var bestLand = -4;
  for (var rot = 0; rot < 4; rot++) {
    var width = 1;
    for (var k = 0u; k < 4u; k++) {
      width = max(width, pieceCell(curType, rot, k).x + 1);
    }
    for (var col = 0; col <= 10 - width; col++) {
      // Landing row from stack tops: box top = min(20 - h[x] - 1 - dy).
      var land = 20;
      for (var k = 0u; k < 4u; k++) {
        let cell = pieceCell(curType, rot, k);
        land = min(land, 20 - heights[col + cell.x] - 1 - cell.y);
      }
      // Resulting column heights + cells per column/row of the piece.
      var h2 = heights;
      var colCells: array<i32, 10>;
      var rowCells: array<i32, 20>;
      for (var k = 0u; k < 4u; k++) {
        let cell = pieceCell(curType, rot, k);
        colCells[col + cell.x]++;
        if (land + cell.y >= 0) { rowCells[land + cell.y]++; }
        h2[col + cell.x] = max(h2[col + cell.x], 20 - land - cell.y);
      }
      // Lines this move completes.
      var lines = 0;
      for (var k = 0u; k < 4u; k++) {
        let r = land + pieceCell(curType, rot, k).y;
        if (r >= 0 && rowFill[r] + rowCells[r] == 10) { lines++; }
      }
      // Holes after: gaps between the old stack top and the new cells.
      var holes2 = holes;
      for (var c = 0; c < 10; c++) {
        holes2 += (h2[c] - heights[c]) - colCells[c];
      }
      var bump2 = 0;
      var agg2 = 0;
      var maxWell2 = 0;
      for (var c = 0; c < 10; c++) {
        if (c < 9) { bump2 += abs(h2[c + 1] - h2[c]); }
        agg2 += h2[c];
        let left = select(20, h2[c - 1], c > 0);
        let right = select(20, h2[c + 1], c < 9);
        maxWell2 = max(maxWell2, max(min(left, right) - h2[c], 0));
      }

      var inputs: array<f32, 15>;
      for (var c = 0; c < 10; c++) {
        inputs[c] = f32(h2[c]) / 20.0;
      }
      inputs[10] = f32(lines) / 4.0;
      inputs[11] = f32(holes2) / 200.0;
      inputs[12] = f32(bump2) / 50.0;
      inputs[13] = f32(agg2) / 200.0;
      inputs[14] = f32(maxWell2) / 20.0;
      let s = evaluate(b, inputs);
      if (s > bestScore) {
        bestScore = s;
        bestCol = col;
        bestRot = rot;
        bestLand = land;
      }
    }
  }

  // --- Lock the winning placement (cell above the field = top out). ---
  var topOut = bestLand < -3;
  for (var k = 0u; k < 4u; k++) {
    let cell = pieceCell(curType, bestRot, k);
    let x = bestCol + cell.x;
    let y = bestLand + cell.y;
    if (y < 0) {
      topOut = true;
    } else {
      setType(b, x, y, u32(curType) + 1u);
    }
  }
  agents[b + A_PCOL] = f32(bestCol);
  agents[b + A_PROT] = f32(bestRot);
  agents[b + A_PLAND] = f32(bestLand);
  agents[b + A_PTYPE] = f32(curType);

  // --- Line clear: compact full rows down. ---
  var cleared = 0;
  var write = 19;
  for (var r = 19; r >= 0; r--) {
    var full = true;
    for (var c = 0; c < 10; c++) {
      if (getType(b, c, r) == 0u) {
        full = false;
        break;
      }
    }
    if (full) {
      cleared++;
      continue;
    }
    if (write != r) {
      for (var c = 0; c < 10; c++) {
        setType(b, c, write, getType(b, c, r));
      }
    }
    write--;
  }
  for (var r = 0; r <= write; r++) {
    for (var c = 0; c < 10; c++) {
      setType(b, c, r, 0u);
    }
  }

  let level = agents[b + A_LEVEL];
  agents[b + A_LINES] = agents[b + A_LINES] + f32(cleared);
  agents[b + A_SCORE] = agents[b + A_SCORE] + f32(LINE_POINTS[cleared]) * (level + 1.0);
  agents[b + A_LEVEL] = floor(agents[b + A_LINES] / 10.0);
  agents[b + A_PIECES] = agents[b + A_PIECES] + 1.0;
  agents[b + 12u] = f32(holes);
  let newCur = i32(sequence[u32(agents[b + A_PIECES])]);
  agents[b + A_CUR] = f32(newCur);
  agents[b + A_NEXT] = f32(sequence[u32(agents[b + A_PIECES]) + 1u]);

  // Top out if the new piece's spawn cells are blocked.
  for (var k = 0u; k < 4u; k++) {
    let cell = pieceCell(newCur, 0, k);
    if (getType(b, 3 + cell.x, cell.y) != 0u) {
      topOut = true;
    }
  }
  if (topOut) {
    agents[b + A_OVER] = 1.0;
  }
}
`;

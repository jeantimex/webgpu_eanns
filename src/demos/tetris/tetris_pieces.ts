/**
 * Tetromino definitions. Types 0..6 = I, O, T, L, J, S, Z.
 * Each rotation is 4 cells as (dx, dy) in a local box; no wall kicks (NES style).
 * The WGSL table is generated from this single source (see wgslPieceTable).
 */
export type Cell = readonly [number, number];
export type Rotation = readonly [Cell, Cell, Cell, Cell];

export const PIECES: ReadonlyArray<readonly Rotation[]> = [
  // I
  [
    [[0, 1], [1, 1], [2, 1], [3, 1]],
    [[2, 0], [2, 1], [2, 2], [2, 3]],
    [[0, 1], [1, 1], [2, 1], [3, 1]],
    [[2, 0], [2, 1], [2, 2], [2, 3]],
  ],
  // O
  [
    [[0, 0], [1, 0], [0, 1], [1, 1]],
    [[0, 0], [1, 0], [0, 1], [1, 1]],
    [[0, 0], [1, 0], [0, 1], [1, 1]],
    [[0, 0], [1, 0], [0, 1], [1, 1]],
  ],
  // T
  [
    [[1, 0], [0, 1], [1, 1], [2, 1]],
    [[1, 0], [1, 1], [2, 1], [1, 2]],
    [[0, 1], [1, 1], [2, 1], [1, 2]],
    [[1, 0], [0, 1], [1, 1], [1, 2]],
  ],
  // L
  [
    [[2, 0], [0, 1], [1, 1], [2, 1]],
    [[1, 0], [1, 1], [1, 2], [2, 2]],
    [[0, 1], [1, 1], [2, 1], [0, 2]],
    [[0, 0], [1, 0], [1, 1], [1, 2]],
  ],
  // J
  [
    [[0, 0], [0, 1], [1, 1], [2, 1]],
    [[1, 0], [2, 0], [1, 1], [1, 2]],
    [[0, 1], [1, 1], [2, 1], [2, 2]],
    [[1, 0], [1, 1], [0, 2], [1, 2]],
  ],
  // S
  [
    [[1, 0], [2, 0], [0, 1], [1, 1]],
    [[1, 0], [1, 1], [2, 1], [2, 2]],
    [[1, 0], [2, 0], [0, 1], [1, 1]],
    [[1, 0], [1, 1], [2, 1], [2, 2]],
  ],
  // Z
  [
    [[0, 0], [1, 0], [1, 1], [2, 1]],
    [[2, 0], [1, 1], [2, 1], [1, 2]],
    [[0, 0], [1, 0], [1, 1], [2, 1]],
    [[2, 0], [1, 1], [2, 1], [1, 2]],
  ],
];

/** Occupied width of a piece rotation (max dx + 1). */
export function pieceWidth(type: number, rot: number): number {
  return Math.max(...PIECES[type][rot].map((c) => c[0])) + 1;
}

/** NES-flavored tile colors (border shading is derived in the renderer). */
export const PIECE_COLORS: ReadonlyArray<readonly [number, number, number]> = [
  [0.36, 0.76, 0.96], // I light blue
  [0.62, 0.86, 0.98], // O pale blue
  [0.66, 0.5, 0.9], // T purple
  [0.95, 0.62, 0.24], // L orange
  [0.3, 0.42, 0.9], // J blue
  [0.36, 0.82, 0.44], // S green
  [0.92, 0.32, 0.32], // Z red
];

/** NES line-clear points [0,1,2,3,4 lines], multiplied by (level+1). */
export const LINE_POINTS = [0, 40, 100, 300, 1200];

/**
 * WGSL literal for the rotation table: 28 vec4i, one per (type, rotation),
 * each holding 4 cells packed as (x0,y0,x1,y1,x2,y2,...)? No — 4 cells x 2
 * coords = 8 ints, so TWO vec4i per (type, rotation), 56 total.
 */
export function wgslPieceTable(): string {
  const rows: string[] = [];
  for (const piece of PIECES) {
    for (const rot of piece) {
      const flat = rot.flat();
      rows.push(`vec4i(${flat.slice(0, 4).join(',')}), vec4i(${flat.slice(4).join(',')})`);
    }
  }
  return `const PIECES = array<vec4i, 56>(\n  ${rows.join(',\n  ')}\n);`;
}

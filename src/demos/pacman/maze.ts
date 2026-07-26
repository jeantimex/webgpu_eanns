/**
 * The arcade maze, verbatim from pacman-js (gameCoordinator.js:21-53, MIT).
 * 'X' = wall, 'o' = pacdot (10 pts), 'O' = power pellet (50 pts), ' ' = open.
 */
export const MAZE_GRID = [
  'XXXXXXXXXXXXXXXXXXXXXXXXXXXX',
  'XooooooooooooXXooooooooooooX',
  'XoXXXXoXXXXXoXXoXXXXXoXXXXoX',
  'XOXXXXoXXXXXoXXoXXXXXoXXXXOX',
  'XoXXXXoXXXXXoXXoXXXXXoXXXXoX',
  'XooooooooooooooooooooooooooX',
  'XoXXXXoXXoXXXXXXXXoXXoXXXXoX',
  'XoXXXXoXXoXXXXXXXXoXXoXXXXoX',
  'XooooooXXooooXXooooXXooooooX',
  'XXXXXXoXXXXX XX XXXXXoXXXXXX',
  'XXXXXXoXXXXX XX XXXXXoXXXXXX',
  'XXXXXXoXX          XXoXXXXXX',
  'XXXXXXoXX XXXXXXXX XXoXXXXXX',
  'XXXXXXoXX X      X XXoXXXXXX',
  '      o   X      X   o      ',
  'XXXXXXoXX X      X XXoXXXXXX',
  'XXXXXXoXX XXXXXXXX XXoXXXXXX',
  'XXXXXXoXX          XXoXXXXXX',
  'XXXXXXoXX XXXXXXXX XXoXXXXXX',
  'XXXXXXoXX XXXXXXXX XXoXXXXXX',
  'XooooooooooooXXooooooooooooX',
  'XoXXXXoXXXXXoXXoXXXXXoXXXXoX',
  'XoXXXXoXXXXXoXXoXXXXXoXXXXoX',
  'XOooXXooooooo  oooooooXXooOX',
  'XXXoXXoXXoXXXXXXXXoXXoXXoXXX',
  'XXXoXXoXXoXXXXXXXXoXXoXXoXXX',
  'XooooooXXooooXXooooXXooooooX',
  'XoXXXXXXXXXXoXXoXXXXXXXXXXoX',
  'XoXXXXXXXXXXoXXoXXXXXXXXXXoX',
  'XooooooooooooooooooooooooooX',
  'XXXXXXXXXXXXXXXXXXXXXXXXXXXX',
];

export const COLS = 28;
export const ROWS = 31;
export const TILE = 8;
export const WORLD_W = COLS * TILE; // 224
export const WORLD_H = ROWS * TILE; // 248

/** Wall bitmask: one u32 per row, bit c = wall at (c, r). */
export function mazeWallBits(): Uint32Array<ArrayBuffer> {
  const bits = new Uint32Array(ROWS);
  for (let r = 0; r < ROWS; r++) {
    let word = 0;
    for (let c = 0; c < COLS; c++) {
      if (MAZE_GRID[r][c] === 'X') word |= 1 << c;
    }
    bits[r] = word >>> 0;
  }
  return bits;
}

export interface PelletTile {
  c: number;
  r: number;
  power: boolean;
}

/** All 244 pellet tiles (240 pacdots + 4 power pellets), row-major. */
export function pelletTiles(): PelletTile[] {
  const tiles: PelletTile[] = [];
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const ch = MAZE_GRID[r][c];
      if (ch === 'o' || ch === 'O') tiles.push({ c, r, power: ch === 'O' });
    }
  }
  return tiles;
}

/** Pellet-present bitmask over all tiles: bit (r*28+c) of 28 u32 words. */
export function pelletMaskInit(): Uint32Array<ArrayBuffer> {
  const bits = new Uint32Array(28);
  for (const { c, r } of pelletTiles()) {
    const idx = r * COLS + c;
    bits[idx >>> 5] |= 1 << (idx & 31);
  }
  return bits;
}

/** Distance stored in the path table for a pair with no route between them. */
export const UNREACHABLE = 255;

/** Pacman's start tile — the root of the reachable-tile set. */
const START_TILE: readonly [number, number] = [14, 23];

function isOpen(c: number, r: number): boolean {
  if (r < 0 || r >= ROWS) return false;
  if (c < 0 || c >= COLS) return false;
  return MAZE_GRID[r][c] !== 'X';
}

/**
 * The 4 walkable neighbours of a tile, with the row-14 tunnel wrapping column
 * 0 to column 27 exactly as `warpX` does in the shader.
 */
function walkableNeighbors(c: number, r: number): [number, number][] {
  const out: [number, number][] = [];
  for (const [dx, dy] of [
    [0, -1],
    [0, 1],
    [-1, 0],
    [1, 0],
  ] as const) {
    let nc = c + dx;
    const nr = r + dy;
    if (nr === 14) {
      if (nc < 0) nc = COLS - 1;
      else if (nc >= COLS) nc = 0;
    }
    if (isOpen(nc, nr)) out.push([nc, nr]);
  }
  return out;
}

export interface MazeGraph {
  /** Tile (r*28+c) -> compact walkable index, or -1 for walls and the ghost house. */
  tileIndex: Int32Array<ArrayBuffer>;
  /** Step counts between walkable tiles, one u8 per pair, 4 packed per u32. */
  pathDist: Uint32Array<ArrayBuffer>;
  /** Number of tiles pacman can actually stand on. */
  walkCount: number;
  /** Row stride of `pathDist` in bytes (walkCount padded to a multiple of 4). */
  stride: number;
  /** Longest shortest-path in the maze; the normalizer for distance inputs. */
  diameter: number;
}

let cachedGraph: MazeGraph | null = null;

/**
 * All-pairs shortest paths over the tiles pacman can reach, by BFS from each
 * one. This is the maze's true metric: unlike straight-line distance it knows
 * about walls, so a greedy descent toward the nearest pellet always makes
 * progress instead of stalling against a wall or oscillating between two tiles.
 *
 * The 18 ghost-house tiles are open in the grid but sealed off from the board,
 * so they are excluded — a ghost sitting in the house maps to index -1 and
 * registers as no threat, which is exactly right.
 *
 * ~300 tiles => a 90 KB table, built once at startup and uploaded read-only.
 */
export function mazeGraph(): MazeGraph {
  if (cachedGraph) return cachedGraph;

  const tileIndex = new Int32Array(COLS * ROWS).fill(-1);
  const walkTiles: [number, number][] = [];
  const queue: [number, number][] = [[...START_TILE] as [number, number]];
  tileIndex[START_TILE[1] * COLS + START_TILE[0]] = 0;
  walkTiles.push([...START_TILE] as [number, number]);
  for (let head = 0; head < queue.length; head++) {
    const [c, r] = queue[head];
    for (const [nc, nr] of walkableNeighbors(c, r)) {
      if (tileIndex[nr * COLS + nc] !== -1) continue;
      tileIndex[nr * COLS + nc] = walkTiles.length;
      walkTiles.push([nc, nr]);
      queue.push([nc, nr]);
    }
  }

  const walkCount = walkTiles.length;
  const stride = (walkCount + 3) & ~3;
  const bytes = new Uint8Array(walkCount * stride).fill(UNREACHABLE);
  const dist = new Int32Array(walkCount);
  let diameter = 0;
  for (let src = 0; src < walkCount; src++) {
    dist.fill(-1);
    dist[src] = 0;
    const frontier: number[] = [src];
    for (let head = 0; head < frontier.length; head++) {
      const at = frontier[head];
      const [c, r] = walkTiles[at];
      for (const [nc, nr] of walkableNeighbors(c, r)) {
        const next = tileIndex[nr * COLS + nc];
        if (next < 0 || dist[next] !== -1) continue;
        dist[next] = dist[at] + 1;
        frontier.push(next);
      }
    }
    for (let dst = 0; dst < walkCount; dst++) {
      if (dist[dst] < 0) continue;
      bytes[src * stride + dst] = Math.min(dist[dst], UNREACHABLE);
      if (dist[dst] > diameter) diameter = dist[dst];
    }
  }

  cachedGraph = {
    tileIndex,
    pathDist: new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4),
    walkCount,
    stride,
    diameter,
  };
  return cachedGraph;
}

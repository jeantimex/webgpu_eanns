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

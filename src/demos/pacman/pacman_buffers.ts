import { createBufferWithData } from '../../webgpu/utils';
import { mazeWallBits, pelletMaskInit, pelletTiles } from './maze';

export const PACMAN_INPUTS = 16;
export const PACMAN_OUTPUTS = 4;
export const PACMAN_TOPOLOGY = [PACMAN_INPUTS, PACMAN_OUTPUTS] as const;

/**
 * Genome layout [16 -> 4], input-major row-major with bias row last.
 * Inputs are 4 feature vectors x/y in normal mode, then the same 4 vectors in
 * powered mode: pellet, power pellet, ghost repulsion, nearest ghost.
 */
export const PACMAN_GENOME_SIZE = PACMAN_INPUTS * PACMAN_OUTPUTS + PACMAN_OUTPUTS;

/**
 * Agent state, 68 f32 = 272 bytes per agent. Raw-indexed (no WGSL struct) so the
 * CPU and GPU layouts can't drift apart. Pellets live at floats 36..63 as u32
 * bit patterns (bitcast in WGSL).
 */
export const AGENT_FLOATS = 68;
export const A = {
  posX: 0,
  posY: 1,
  dir: 2, // 0=up, 1=down, 2=left, 3=right
  desired: 3,
  moving: 4,
  lives: 5,
  dotsLeft: 6,
  score: 7,
  level: 8,
  modeTimer: 9, // seconds into the current scatter/chase phase
  phase: 10, // 0=scatter (7s), 1=chase (20s)
  frightTimer: 11,
  combo: 12, // ghost-eat combo within one power pellet
  houseTimer: 13, // chained ghost-release timer
  released: 14, // how many house ghosts released so far (0..3)
  gameOver: 15,
  moveTicks: 16, // hidden training signal: ticks where Pac-Man changed position
  ghosts: 17, // 4 ghosts x 4 floats: x, y, dir, mode (0=normal, 1=scared, 2=eyes, 3=idle, 4=leaving)
  ticks: 33, // u32, ticks alive this game (hard cap in the shader)
  sinceEat: 34, // seconds since the last pellet; stall timeout
  fruit: 35, // >0: cherry active (seconds left), 0: not yet, -1/-2: spawns consumed
  pellets: 36, // 28 u32 words: bit (r*28+c) = pellet present
  pelletProgress: 64, // hidden training signal: accumulated decrease in nearest-pellet distance
  totalReward: 65, // hidden training signal: accumulated environment reward
  previousChoice: 66, // hidden training signal: last AI direction choice
  reversalStreak: 67, // hidden training signal: consecutive opposite choices
} as const;

// Movement/scoring constants from the source repo (speeds in tiles/sec; engine dt = 1/60).
export const PAC_SPEED = 11;
export const GHOST_SLOW = 0.76; // x PAC_SPEED (level-1 slowSpeed)
export const GHOST_SCARED = 0.5;
export const GHOST_TRANSITION = 0.4; // tunnel & ghost house
export const GHOST_EYES = 2.0;
export const SCATTER_SECS = 7;
export const CHASE_SECS = 20;
export const FRIGHT_SECS = 6; // level-1 frightened duration
export const HOUSE_RELEASE_SECS = 8; // chained, level 1
export const START_LIVES = 0; // spare lives: 1 attempt per thread, first hit ends the game
export const MAX_GAME_TICKS = 21600; // 6 min at 60 Hz, then the game ends
export const STALL_SECS = 8; // no pellet for this long ends the game (anti standstill)
export const FRUIT_SECS = 10; // cherry lifetime, spawns at 70 and 170 dots eaten (100 pts)

export interface PacmanBuffers {
  params: GPUBuffer;
  genomes: GPUBuffer;
  agents: GPUBuffer;
  mazeBits: GPUBuffer;
  initPellets: GPUBuffer;
  pelletList: GPUBuffer;
  readback: GPUBuffer;
  pelletCount: number;
  totalAgents: number;
}

/** All agents at the arcade start state (positions in the repo's tile coords). */
export function initialAgentStates(count: number): Float32Array<ArrayBuffer> {
  const states = new Float32Array(count * AGENT_FLOATS);
  const pellets = pelletMaskInit();
  for (let i = 0; i < count; i++) {
    const o = i * AGENT_FLOATS;
    states[o + A.posX] = 13.5;
    states[o + A.posY] = 23;
    states[o + A.dir] = 2; // left
    states[o + A.desired] = 2;
    states[o + A.previousChoice] = 2;
    states[o + A.moving] = 1;
    states[o + A.lives] = START_LIVES;
    states[o + A.dotsLeft] = 244;
    states[o + A.level] = 1;
    // Ghosts: blinky outside heading left; pinky/inky/clyde idle in the house.
    const g = o + A.ghosts;
    states.set([13.5, 11, 2, 0], g); // blinky
    states.set([13.5, 14, 1, 3], g + 4); // pinky (down)
    states.set([11.5, 14, 0, 3], g + 8); // inky (up)
    states.set([15.5, 14, 0, 3], g + 12); // clyde (up)
    new Uint32Array(states.buffer, o * 4 + A.pellets * 4, 28).set(pellets);
  }
  return states;
}

export function createPacmanBuffers(device: GPUDevice, populationSize: number): PacmanBuffers {
  const totalAgents = populationSize + 1;
  // SimParams uniform: agentCount, trainCount (u32), padded to 16.
  const paramsData = new ArrayBuffer(16);
  new Uint32Array(paramsData).set([totalAgents, populationSize]);

  const genomes = device.createBuffer({
    label: 'pacman genomes',
    size: totalAgents * PACMAN_GENOME_SIZE * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const agentBytes = totalAgents * AGENT_FLOATS * 4;
  const agents = device.createBuffer({
    label: 'pacman agents',
    size: agentBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  const readback = device.createBuffer({
    label: 'pacman readback',
    size: agentBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // Renderer pellet list: c, r, power (u32) + pad = 16 bytes per pellet.
  const tiles = pelletTiles();
  const pelletData = new Uint32Array(tiles.length * 4);
  tiles.forEach((t, i) => {
    pelletData[i * 4] = t.c;
    pelletData[i * 4 + 1] = t.r;
    pelletData[i * 4 + 2] = t.power ? 1 : 0;
  });

  return {
    params: createBufferWithData(device, 'pacman params', new Uint32Array(paramsData), GPUBufferUsage.UNIFORM),
    genomes,
    agents,
    mazeBits: createBufferWithData(device, 'pacman maze bits', mazeWallBits(), GPUBufferUsage.STORAGE),
    initPellets: createBufferWithData(device, 'pacman init pellets', pelletMaskInit(), GPUBufferUsage.STORAGE),
    pelletList: createBufferWithData(device, 'pacman pellet list', pelletData, GPUBufferUsage.STORAGE),
    readback,
    pelletCount: tiles.length,
    totalAgents,
  };
}

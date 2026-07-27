import { episodeSeed } from '../../utils/evaluation';
import { pelletMaskInit } from './maze';

/**
 * Network shape from the EANN-Pacman writeup (§2.2): a shallow feedforward net,
 * one hidden layer of 4-8 ReLU units, and 4 outputs turned into a probability
 * distribution by softmax and then *sampled* rather than argmax'd — the
 * randomness is what keeps the population exploring.
 */
export const PACMAN_INPUTS = 14;
export const PACMAN_HIDDEN = 6;
export const PACMAN_OUTPUTS = 4;
export const PACMAN_TOPOLOGY = [PACMAN_INPUTS, PACMAN_HIDDEN, PACMAN_OUTPUTS] as const;

/**
 * Flat genome, layer-major and input-major with each layer's bias row last:
 * (8+1)x6 + (6+1)x4 = 82 weights. Matches the layout NetworkPanel draws.
 */
export const PACMAN_GENOME_SIZE = (PACMAN_INPUTS + 1) * PACMAN_HIDDEN + (PACMAN_HIDDEN + 1) * PACMAN_OUTPUTS;

/**
 * The 8-dimensional 感知向量 (perception vector) of §2.1. The writeup measures
 * both distances as straight-line; we measure them as maze steps off the
 * precomputed all-pairs table instead. Same inputs, same count — but a
 * wall-blind metric has local minima inside wall blocks, which is what makes a
 * greedy pac oscillate between two tiles, and it is the same limitation the
 * writeup itself runs into in §4.4 (全局策略缺失).
 *
 * Directions are unit vectors, not the writeup's direction *index*. An index is
 * a categorical variable flattened onto one ordinal axis (up=0, down=.33,
 * left=.67, right=1), so following it requires the hidden layer to build four
 * separate bump functions on that axis — roughly 8 ReLUs' worth of structure,
 * found by random search, with only 6 units available. As a vector the same
 * behaviour is a plain linear map (out_up = -dy, out_right = +dx), which
 * evolution finds almost immediately. This is §5.2's 改进感知向量 advice;
 * 11 dims still sits inside the writeup's 8-12 ideal band.
 *
 * The *two* nearest ghosts are reported, not one. With a single ghost visible an
 * agent cannot perceive a pincer — two ghosts closing from opposite ends of a
 * corridor — so it flees the one it can see straight into the one it cannot.
 * That is the classic way Pac-Man dies, and ghosts became the leading cause of
 * death (53%) once the population grew bold enough to eat near them.
 *
 *     0 pelletDist   maze steps to the nearest pellet / maze diameter
 *   1-2 pelletDir    unit vector of the neighbour starting that shortest path
 *     3 ghostDist    maze steps to the nearest ghost / maze diameter
 *   4-5 ghostDir     unit vector of the neighbour starting that shortest path
 *     6 ghostState   0 = the nearest ghost hunts, 1 = frightened and edible
 *     7 ghost2Dist   maze steps to the second nearest ghost / maze diameter
 *  8-9 ghost2Dir     unit vector of the neighbour starting that shortest path
 * 10-11 powerDir     unit vector toward the nearest power pellet, (0,0) if none
 *    12 aheadClear   1 when the tile in front of the current heading is open
 *    13 progress     pellets eaten on this board / 244
 */
export const PACMAN_INPUT_LABELS = [
  'PELLET d',
  'PELLET dx',
  'PELLET dy',
  'GHOST d',
  'GHOST dx',
  'GHOST dy',
  'GHOST state',
  'GHOST2 d',
  'GHOST2 dx',
  'GHOST2 dy',
  'POWER dx',
  'POWER dy',
  'AHEAD clear',
  'PROGRESS',
] as const;

/** Output index maps straight onto the direction encoding (0=up,1=down,2=left,3=right). */
export const PACMAN_OUTPUT_LABELS = ['UP', 'DOWN', 'LEFT', 'RIGHT'] as const;

/**
 * Composite fitness of §3.1. The writeup's first attempt was `fitness = score`,
 * and evolution promptly discovered that standing perfectly still scores 0 but
 * never gets caught — the classic local optimum. These four terms are the fix.
 * Survival deliberately carries the *smallest* weight: 单纯生存不能给太高权重，
 * 否则又会催生"躲藏策略" (weight it heavily and you breed hiding strategies).
 *
 * The writeup's raw weights assume its own scoring scale. Every term here is
 * first normalised to a 0-100 percentage so the ratios 1 : 0.1 : 2 actually mean
 * what §3.1 intends — otherwise raw arcade score (2440 for one board, plus up to
 * 1600 per power pellet for ghost combos) swamps the pellet term and breeds
 * ghost-hunters instead of board-clearers.
 */
export const FIT_SCORE_W = 1.0;
export const FIT_SURVIVAL_W = 0.1;
export const FIT_PELLET_PCT_W = 2.0;
/**
 * §3.1's 无意义移动惩罚, as the *fraction of its life* the agent spent shoving
 * into a wall. Counting raw bumps instead makes this term dominate everything
 * at generation 1, when random genomes bump on almost every tick.
 */
export const FIT_WASTED_W = 0.3;
/** Arcade points that count as "one board's worth" when normalising score. */
export const FIT_SCORE_NORM = 2440;

/**
 * Agent state, 66 f32 = 264 bytes per agent. Raw-indexed (no WGSL struct) so the
 * CPU and GPU layouts can't drift apart. Pellets live at floats 36..63 as u32
 * bit patterns (bitcast in WGSL).
 */
export const AGENT_FLOATS = 66;
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
  levelTicks: 16, // ticks spent on the current board
  ghosts: 17, // 4 ghosts x 4 floats: x, y, dir, mode (0=normal, 1=scared, 2=eyes, 3=idle, 4=leaving)
  ticks: 33, // u32, ticks alive this game (hard cap in the shader)
  sinceEat: 34, // seconds since the last pellet; stall timeout
  fruit: 35, // >0: cherry active (seconds left), 0: not yet, -1/-2: spawns consumed
  pellets: 36, // 28 u32 words: bit (r*28+c) = pellet present
  wasted: 64, // blocked turns / wall bumps, the fitness penalty term
  rng: 65, // u32 xorshift state, drives softmax action sampling
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
export const LEVEL_SECS = 90; // default per-board time budget; adjustable at runtime
export const LEVEL_SECS_MIN = 30;
export const LEVEL_SECS_MAX = 180;

/**
 * Chance that a ghost takes a random legal turn instead of its best one. The
 * arcade AI is fully deterministic, so without this every game is the same
 * replay. Small enough to keep the authentic feel, large enough that the same
 * genome meets different situations run to run.
 */
export const GHOST_CHAOS = 0.1;

/**
 * Episodes each genome is scored over; see src/utils/evaluation.ts. Measured at
 * generation 40, population 300: 1 episode gives 54.6 average pellets, 3 gives
 * 64.5, 6 gives 93.7 — and the gap between a saved model's recorded score and
 * what it actually replays narrows from 8.7x to 4.0x over the same range.
 * Episodes are extra threads in the same dispatch, so 6 costs 1800 agent slots
 * and no extra wall-clock.
 */
export const EPISODES_PER_GENOME = 6;
export const STALL_SECS = 8; // no pellet for this long ends the game (anti standstill)
export const FRUIT_SECS = 10; // cherry lifetime, spawns at 70 and 170 dots eaten (100 pts)

/**
 * All agents at the arcade start state (positions in the repo's tile coords).
 *
 * `episodes` follows the layout in src/utils/evaluation.ts: agent
 * `genome * episodes + episode`. The RNG seed depends only on the *episode*
 * index, never the genome, so every genome in a generation plays the same set of
 * episodes (common random numbers) and comparisons between them are not
 * contaminated by luck that only one of them happened to get.
 */
export function initialAgentStates(count: number, seed = 1, episodes = 1): Float32Array<ArrayBuffer> {
  const states = new Float32Array(count * AGENT_FLOATS);
  const asU32 = new Uint32Array(states.buffer);
  const pellets = pelletMaskInit();
  for (let i = 0; i < count; i++) {
    const o = i * AGENT_FLOATS;
    states[o + A.posX] = 13.5;
    states[o + A.posY] = 23;
    states[o + A.dir] = 2; // left
    states[o + A.desired] = 2;
    states[o + A.moving] = 1;
    states[o + A.lives] = START_LIVES;
    states[o + A.dotsLeft] = 244;
    states[o + A.level] = 1;
    asU32[o + A.rng] = episodeSeed(seed, i % episodes);
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

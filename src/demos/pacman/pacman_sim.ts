import type { Evolution, Simulation } from '../../core';
import { episodeSpread } from '../../utils/evaluation';
import { createBufferWithData } from '../../webgpu/utils';
import { mazeGraph, mazeWallBits, pelletMaskInit } from './maze';
import {
  A,
  AGENT_FLOATS,
  FIT_PELLET_PCT_W,
  FIT_SCORE_NORM,
  FIT_SCORE_W,
  FIT_SURVIVAL_W,
  FIT_WASTED_W,
  GHOST_CHAOS,
  initialAgentStates,
  LEVEL_SECS,
  LEVEL_SECS_MAX,
  LEVEL_SECS_MIN,
} from './pacman_buffers';
import { pacmanShader } from './pacman.wgsl';

// GA parameters from §3.2-3.4 of the writeup (the tournament config lives in main.ts).
/** §3.3's adaptive mechanism: nudge mutation up after this many flat generations. */
const BASE_MUTATE_RATE = 0.05;
const STAGNATION_LIMIT = 5;
const MUTATE_STEP = 0.02;
const MAX_MUTATE_RATE = 0.12;
/** Softmax temperature for training agents; lower = more decisive, less exploration. */
const ACTION_TEMPERATURE = 1;
/** Dots the on-screen agent may fall behind the leader before the view switches. */
const DISPLAY_STICKINESS = 12;

/**
 * Live environment/GA state. Everything the settings panel, the params uniform,
 * and the HUD touch between generations — the driver owns the rest.
 */
export const pacmanState = {
  playMode: false,
  episodeSeed: 1,
  replaySeed: 1,
  mutateRate: BASE_MUTATE_RATE,
  stagnantGenerations: 0,
  lastBestFitness: -Infinity,
  actionTemperature: ACTION_TEMPERATURE,
  ghostChaos: GHOST_CHAOS,
  levelSeconds: LEVEL_SECS,
  ghostSpeedScale: 1,
  houseReleaseScale: 1,
  bestScore: 0,
  bestLevel: 1,
  /** Highest level any agent has ever reached, for the HUD. */
  highestLevel: 1,
  lastSpread: 0,
  shownIndex: -1,
};

/** Runtime knobs from the settings panel; each pushes the params uniform live. */
export const pacmanControls = {
  /** Per-board time budget, live. Agents already past the new limit end on the next tick. */
  setLevelSeconds(evo: Evolution, seconds: number): void {
    pacmanState.levelSeconds = Math.min(LEVEL_SECS_MAX, Math.max(LEVEL_SECS_MIN, seconds));
    evo.writeParams();
  },
  /** Chance a ghost takes a random legal turn; 0 restores pure arcade behaviour. */
  setGhostChaos(evo: Evolution, p: number): void {
    pacmanState.ghostChaos = Math.min(1, Math.max(0, p));
    evo.writeParams();
  },
  /** Softmax temperature for training agents. 0 takes the mode (no exploration). */
  setActionTemperature(evo: Evolution, t: number): void {
    pacmanState.actionTemperature = Math.max(0, t);
    evo.writeParams();
  },
  /** Steer the human player's agent (the display slot) in Play mode. */
  setPlayerDesiredDir(evo: Evolution, dir: number): void {
    const o = evo.displayAgentIndex * AGENT_FLOATS;
    evo.device.queue.writeBuffer(evo.buffers.agents, (o + A.desired) * 4, new Float32Array([dir]));
  },
};

/**
 * Which agent the camera/panel follows. Train mode follows the current leader
 * with stickiness (re-picking every frame reads as camera jitter); Play/Test
 * mode follows the replay slot.
 */
export function pickShownAgent(states: Float32Array, evo: Evolution): number {
  const play = pacmanState.playMode;
  let shown = evo.displayAgentIndex;
  let bestDotsLeft = Infinity;
  let bestScore = -Infinity;
  let bestTicks = Infinity;
  for (let i = 0; i < evo.trainingAgents; i++) {
    const o = i * AGENT_FLOATS;
    if (states[o + A.gameOver] < 0.5) {
      if (!play) {
        const dotsLeft = states[o + A.dotsLeft];
        const score = states[o + A.score];
        const ticks = states[o + A.ticks];
        if (dotsLeft < bestDotsLeft || (dotsLeft === bestDotsLeft && (score > bestScore || (score === bestScore && ticks < bestTicks)))) {
          shown = i;
          bestDotsLeft = dotsLeft;
          bestScore = score;
          bestTicks = ticks;
        }
      }
    }
  }
  if (!play && evo.trainingAgents === 1) shown = 0;
  if (!play && pacmanState.shownIndex >= 0 && pacmanState.shownIndex < evo.trainingAgents) {
    const prev = pacmanState.shownIndex * AGENT_FLOATS;
    if (states[prev + A.gameOver] < 0.5 && states[prev + A.dotsLeft] <= bestDotsLeft + DISPLAY_STICKINESS) {
      shown = pacmanState.shownIndex;
    }
  }
  pacmanState.shownIndex = shown;
  return shown;
}

export const pacmanSim: Simulation = {
  agentFloats: AGENT_FLOATS,
  shader: pacmanShader,
  initialStates: (count, seed, episodes) => initialAgentStates(count, seed, episodes),

  paramsBytes: 48,
  writeParams(data, evo) {
    const s = pacmanState;
    const totalAgents = evo.trainingAgents + 1;
    // Training agents sample from the softmax; the replay/test slot takes the
    // mode so the pacman you watch behaves deterministically.
    new Uint32Array(data).set([totalAgents, evo.episodes, s.playMode ? 1 : 0, s.episodeSeed]);
    new Float32Array(data, 16).set([s.ghostSpeedScale, s.houseReleaseScale]);
    // Every network-driven agent samples at the same temperature it was evolved
    // under, including the replay/Test slot. Forcing the replay to argmax makes
    // the pacman you watch (and the model you export) a *different, far worse*
    // policy than the one selection actually scored — measured at 10 pellets
    // versus 187 for the very same genome. The human player's agent never runs
    // the network at all, so play mode needs no special case.
    new Float32Array(data, 24).set([s.actionTemperature]);
    new Uint32Array(data, 28).set([Math.round(s.levelSeconds * 60)]);
    new Float32Array(data, 32).set([s.ghostChaos]);
  },

  extraBuffers: (device) => {
    const graph = mazeGraph();
    return {
      mazeBits: createBufferWithData(device, 'pacman maze bits', mazeWallBits(), GPUBufferUsage.STORAGE),
      initPellets: createBufferWithData(device, 'pacman init pellets', pelletMaskInit(), GPUBufferUsage.STORAGE),
      tileIndex: createBufferWithData(device, 'pacman tile index', graph.tileIndex, GPUBufferUsage.STORAGE),
      pathDist: createBufferWithData(device, 'pacman path distances', graph.pathDist, GPUBufferUsage.STORAGE),
    };
  },

  isAgentDone: (states, i) => states[i * AGENT_FLOATS + A.gameOver] > 0.5,

  /** Generation ends when every game is over (lives exhausted or tick cap). */
  isGenerationOver(states, evo) {
    for (let i = 0; i < evo.trainingAgents; i++) {
      if (!this.isAgentDone(states, i)) return false;
    }
    return true;
  },

  /** The replay agent restarts as soon as it dies, mid-generation included.
   *  Re-seeded on every restart: actions are sampled, so a fixed seed would
   *  replay one canned episode forever. */
  maintain(evo, states) {
    if (this.isAgentDone(states, evo.displayAgentIndex)) {
      pacmanState.replaySeed = (pacmanState.replaySeed + 1) >>> 0;
      evo.resetAgent(evo.displayAgentIndex, pacmanState.replaySeed);
    }
  },

  /**
   * Composite fitness of §3.1, scored per *episode*; the driver aggregates
   * episodes (see src/utils/evaluation.ts). Survival is weighted *lowest* (0.1)
   * because weighting it heavily breeds hiding strategies; the percentage of
   * pellets eaten is the core driving force at 2.0. No clamp to a positive
   * floor: tournament selection only compares fitnesses, and clamping would
   * flatten the whole of generation 1 to a single value.
   */
  fitness(states, evo) {
    const s = pacmanState;
    const scores = new Float64Array(evo.trainingAgents);
    for (let i = 0; i < evo.trainingAgents; i++) {
      const o = i * AGENT_FLOATS;
      const score = states[o + A.score];
      const level = states[o + A.level];
      const levelsCleared = Math.max(0, level - 1);
      const ticks = states[o + A.ticks];
      // All four terms on a 0-100 scale, so the writeup's 1 : 0.1 : 2 ratios
      // carry their intended meaning rather than being swamped by raw score.
      const scorePct = (score / FIT_SCORE_NORM) * 100;
      const survivalPct = (ticks / (s.levelSeconds * 60)) * 100;
      const pelletPct = ((levelsCleared * 244 + (244 - states[o + A.dotsLeft])) / 244) * 100;
      const wastedPct = (states[o + A.wasted] / Math.max(1, ticks)) * 100;
      scores[i] =
        scorePct * FIT_SCORE_W +
        survivalPct * FIT_SURVIVAL_W +
        pelletPct * FIT_PELLET_PCT_W -
        wastedPct * FIT_WASTED_W;
      s.highestLevel = Math.max(s.highestLevel, level);
    }
    s.lastSpread = episodeSpread(scores, evo.populationSize, evo.episodes).meanRange;
    return scores;
  },

  /**
   * Which genome gets saved is ranked by the level it *typically* reaches
   * first, and only then by fitness — composite fitness alone nearly ties a
   * board-clearer (pelletPct 100 on a fresh board) against one that ate 243
   * pellets and died (99.6), and "save the best model" should mean the
   * board-clearer. Selection inside the GA is untouched and still uses the
   * composite: a sparse integer would be a poor gradient to evolve against.
   */
  rankBest(states, fitnesses, evo) {
    const s = pacmanState;
    let bestGenomeIndex = -1;
    // Mirrors the original in-loop ratchet: each successively better genome
    // raises the bar the next one must clear (level first, then fitness).
    let recordedFitness = evo.bestFitness;
    for (let g = 0; g < evo.populationSize; g++) {
      // Median episode, so the recorded figures are ones it can reproduce.
      const runs = Array.from({ length: evo.episodes }, (_, e) => g * evo.episodes + e)
        .sort((a, b) => (evo.lastScores?.[a] ?? 0) - (evo.lastScores?.[b] ?? 0));
      const typical = runs[runs.length >> 1] * AGENT_FLOATS;
      const typicalLevel = states[typical + A.level];
      const better =
        typicalLevel > s.bestLevel ||
        (typicalLevel === s.bestLevel && fitnesses[g] > recordedFitness);
      if (!better) continue;
      bestGenomeIndex = g;
      recordedFitness = fitnesses[g];
      s.bestScore = states[typical + A.score];
      s.bestLevel = typicalLevel;
    }
    return bestGenomeIndex;
  },

  bestMeta: () => ({ score: pacmanState.bestScore, level: pacmanState.bestLevel }),

  /**
   * §3.3's adaptive mutation: nudge the rate up after several generations with
   * no improvement, decay it back down as progress resumes — §3.3 wants a
   * *lower* rate late on; ratcheting up and staying there turns the GA into
   * random search.
   */
  beforeGaStep(evo, fitnesses) {
    const s = pacmanState;
    let generationBest = -Infinity;
    for (let g = 0; g < evo.populationSize; g++) generationBest = Math.max(generationBest, fitnesses[g]);
    if (generationBest > s.lastBestFitness) {
      s.lastBestFitness = generationBest;
      s.stagnantGenerations = 0;
      s.mutateRate = Math.max(BASE_MUTATE_RATE, s.mutateRate - MUTATE_STEP);
    } else if (++s.stagnantGenerations >= STAGNATION_LIMIT) {
      s.stagnantGenerations = 0;
      s.mutateRate = Math.min(MAX_MUTATE_RATE, s.mutateRate + MUTATE_STEP);
    }
    evo.setMutationRate(s.mutateRate);
  },

  /** §5.1: re-roll the environment every generation (kept mild, ±15%) so the
   *  population cannot overfit one fixed ghost timing; also re-seed episodes. */
  onNewGeneration(evo) {
    const s = pacmanState;
    s.shownIndex = -1;
    s.episodeSeed = Math.floor(evo.rngForHooks() * 0xffffffff) >>> 0;
    s.ghostSpeedScale = 0.85 + evo.rngForHooks() * 0.3;
    s.houseReleaseScale = 0.85 + evo.rngForHooks() * 0.3;
    evo.writeParams();
  },

  trainingSeed: () => pacmanState.episodeSeed,
  displaySeed: () => pacmanState.episodeSeed,
};

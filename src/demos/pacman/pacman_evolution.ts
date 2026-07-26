import { nextTournamentGeneration } from '../../utils/ga';
import { mulberry32, type Rng } from '../../utils/rng';
import {
  A,
  AGENT_FLOATS,
  createPacmanBuffers,
  FIT_PELLET_PCT_W,
  FIT_SCORE_NORM,
  FIT_SCORE_W,
  FIT_SURVIVAL_W,
  FIT_WASTED_W,
  initialAgentStates,
  LEVEL_SECS,
  PACMAN_GENOME_SIZE,
  PACMAN_TOPOLOGY,
  type PacmanBuffers,
} from './pacman_buffers';
import { pacmanShader } from './pacman.wgsl';
import { autosavePacmanBest } from './pacman_model';

const INIT_RANGE = 1;
/** Dots the on-screen agent may fall behind the leader before the view switches. */
const DISPLAY_STICKINESS = 12;

// GA parameters from §3.2-3.4 of the writeup.
const TOURNAMENT_K = 3;
const ELITE_FRACTION = 0.02;
const CROSSOVER_RATE = 0.8;
const BASE_MUTATE_RATE = 0.05;
/** §3.3's adaptive mechanism: nudge mutation up after this many flat generations. */
const STAGNATION_LIMIT = 5;
const MUTATE_STEP = 0.02;
const MAX_MUTATE_RATE = 0.12;

/** Generation 1 is pure noise — 完全随机, as the writeup's §3.2 starts. */
function randomNetwork(rng: Rng): Float64Array {
  const genome = new Float64Array(PACMAN_GENOME_SIZE);
  for (let k = 0; k < PACMAN_GENOME_SIZE; k++) genome[k] = (rng() * 2 - 1) * INIT_RANGE;
  return genome;
}

export interface BestAgentSnapshot {
  index: number;
  score: number;
  lives: number;
  dotsLeft: number;
  level: number;
  levelTimeLeft: number;
  gameOver: boolean;
  aliveCount: number;
  bestScore: number;
  bestGeneration: number;
  bestLevel: number;
}

/**
 * Generation driver. Each agent plays one full attempt (or until the 6-minute
 * cap) inside the compute shader — there is no shared world state, so a
 * generation is a fixed number of dispatches followed by a GA step.
 */
export class PacmanEvolution {
  generation = 1;

  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroup: GPUBindGroup;
  private readonly workgroups: number;
  private readonly rng: Rng;
  private genomes: Float64Array[];
  private readonly displayIndex: number;
  private bestGenome: Float64Array;
  private bestFitness = -Infinity;
  private bestScore = 0;
  private bestLevel = 1;
  private bestGeneration = 1;
  private readPending: Promise<Float32Array<ArrayBuffer>> | null = null;
  private isEvolving = false;
  private playMode = false;
  private episodeSeed = 1;
  private shownIndex = -1;
  private mutateRate = BASE_MUTATE_RATE;
  private stagnantGenerations = 0;
  private lastBestFitness = -Infinity;
  private ghostSpeedScale = 1;
  private houseReleaseScale = 1;

  private constructor(
    private readonly device: GPUDevice,
    readonly buffers: PacmanBuffers,
    readonly populationSize: number,
    seed: number,
  ) {
    this.rng = mulberry32(seed);
    this.displayIndex = populationSize;
    this.genomes = Array.from({ length: buffers.totalAgents }, () => randomNetwork(this.rng));
    this.bestGenome = new Float64Array(this.genomes[0]);
    this.genomes[this.displayIndex].set(this.bestGenome);
    this.uploadGenomes();
    this.device.queue.writeBuffer(this.buffers.agents, 0, initialAgentStates(this.buffers.totalAgents));

    this.pipeline = device.createComputePipeline({
      label: 'pacman pipeline',
      layout: 'auto',
      compute: { module: device.createShaderModule({ label: 'pacman shader', code: pacmanShader }), entryPoint: 'main' },
    });
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.params } },
        { binding: 1, resource: { buffer: buffers.genomes } },
        { binding: 2, resource: { buffer: buffers.agents } },
        { binding: 3, resource: { buffer: buffers.mazeBits } },
        { binding: 4, resource: { buffer: buffers.initPellets } },
        { binding: 5, resource: { buffer: buffers.tileIndex } },
        { binding: 6, resource: { buffer: buffers.pathDist } },
      ],
    });
    this.workgroups = Math.ceil(this.buffers.totalAgents / 64);
  }

  static init(device: GPUDevice, populationSize = 300, seed = 1): PacmanEvolution {
    return new PacmanEvolution(device, createPacmanBuffers(device, populationSize), populationSize, seed);
  }

  static readonly topology = PACMAN_TOPOLOGY;

  /** CPU-side genome of agent `index`, uploaded to the GPU each generation. */
  genomeAt(index: number): Float64Array {
    return this.genomes[index];
  }

  displayGenome(): Float64Array {
    return this.genomes[this.displayIndex];
  }

  /** Live GA state for the HUD: the adaptive mutation rate and its stall counter. */
  gaState(): { mutateRate: number; stagnant: number } {
    return { mutateRate: this.mutateRate, stagnant: this.stagnantGenerations };
  }

  bestMeta(): { generation: number; eval: number; score: number; level: number } {
    return {
      generation: this.bestGeneration,
      eval: Number.isFinite(this.bestFitness) ? this.bestFitness : 0,
      score: this.bestScore,
      level: this.bestLevel,
    };
  }

  injectBest(weights: Float64Array, eval_ = 0, score = 0, generation = this.generation, level = 1): void {
    this.genomes[0] = weights.slice();
    this.genomes[this.displayIndex] = weights.slice();
    this.bestGenome = weights.slice();
    this.bestFitness = eval_;
    this.bestScore = score;
    this.bestLevel = level;
    this.bestGeneration = generation;
    this.uploadGenomes();
    this.device.queue.writeBuffer(this.buffers.agents, 0, initialAgentStates(this.buffers.totalAgents));
  }

  setPlayMode(playMode: boolean): void {
    this.playMode = playMode;
    this.writeParams();
  }

  private writeParams(): void {
    const data = new ArrayBuffer(32);
    // Training agents sample from the softmax; the replay/test slot takes the
    // mode so the pacman you watch behaves deterministically.
    new Uint32Array(data).set([this.buffers.totalAgents, this.populationSize, this.playMode ? 1 : 0, this.episodeSeed]);
    new Float32Array(data, 16).set([this.ghostSpeedScale, this.houseReleaseScale]);
    new Uint32Array(data, 24).set([this.playMode || this.populationSize === 1 ? 0 : 1, 0]);
    this.device.queue.writeBuffer(this.buffers.params, 0, new Uint8Array(data));
  }

  /**
   * §5.1: re-roll the environment every generation so the population cannot
   * overfit one fixed ghost timing. Kept mild (±15%) — the point is variation,
   * not a different game.
   */
  private rerollEnvironment(): void {
    this.ghostSpeedScale = 0.85 + this.rng() * 0.3;
    this.houseReleaseScale = 0.85 + this.rng() * 0.3;
  }

  setPlayerDesiredDir(dir: number): void {
    const o = this.displayIndex * AGENT_FLOATS;
    const data = new Float32Array([dir]);
    this.device.queue.writeBuffer(this.buffers.agents, (o + A.desired) * 4, data);
  }

  resetDisplayAgent(): void {
    this.resetAgent(this.displayIndex);
  }

  async restartTestAgentIfDead(): Promise<void> {
    const states = await this.readStates();
    if (states[A.gameOver] > 0.5) {
      this.resetAgent(0);
    }
  }

  private uploadGenomes(): void {
    const flat = new Float32Array(this.genomes.length * PACMAN_GENOME_SIZE);
    for (let i = 0; i < this.genomes.length; i++) flat.set(this.genomes[i], i * PACMAN_GENOME_SIZE);
    this.device.queue.writeBuffer(this.buffers.genomes, 0, flat);
  }

  private resetAgent(index: number): void {
    this.device.queue.writeBuffer(this.buffers.agents, index * AGENT_FLOATS * 4, initialAgentStates(1));
  }

  /** Dispatch k per-frame game ticks (60 Hz each). */
  substeps(k: number): void {
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    for (let i = 0; i < k; i++) pass.dispatchWorkgroups(this.workgroups);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /** Copy of the agent-state buffer; concurrent calls share one readback. */
  readStates(): Promise<Float32Array<ArrayBuffer>> {
    if (!this.readPending) {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.buffers.agents, 0, this.buffers.readback, 0, this.buffers.agents.size);
      this.device.queue.submit([encoder.finish()]);
      this.readPending = this.buffers.readback
        .mapAsync(GPUMapMode.READ)
        .then(() => new Float32Array(this.buffers.readback.getMappedRange().slice(0)))
        .finally(() => {
          this.buffers.readback.unmap();
          this.readPending = null;
        });
    }
    return this.readPending;
  }

  /** Generation ends when every game is over (lives exhausted or tick cap). */
  async checkAndEvolve(): Promise<void> {
    if (this.isEvolving) return;
    this.isEvolving = true;
    try {
      const states = await this.readStates();
      if (states[this.displayIndex * AGENT_FLOATS + A.gameOver] > 0.5) {
        this.resetAgent(this.displayIndex);
      }
      let aliveCount = 0;
      for (let i = 0; i < this.populationSize; i++) {
        if (states[i * AGENT_FLOATS + A.gameOver] < 0.5) aliveCount++;
      }
      if (aliveCount > 0) return;

      // Composite fitness of §3.1. The writeup's first attempt was
      // `fitness = score`, and evolution immediately found the local optimum of
      // standing still: score 0, but never caught. Four terms fix it, and the
      // relative weights carry the lesson — survival is weighted *lowest* (0.1)
      // because 单纯生存不能给太高权重，否则又会催生"躲藏策略", while the
      // percentage of pellets eaten is the 核心驱动力 at 2.0.
      const fitnesses = new Float64Array(this.populationSize);
      for (let i = 0; i < this.populationSize; i++) {
        const o = i * AGENT_FLOATS;
        const score = states[o + A.score];
        const level = states[o + A.level];
        const levelsCleared = Math.max(0, level - 1);
        const ticks = states[o + A.ticks];
        // All four terms on a 0-100 scale, so the writeup's 1 : 0.1 : 2 ratios
        // carry their intended meaning rather than being swamped by raw score.
        const scorePct = (score / FIT_SCORE_NORM) * 100;
        const survivalPct = (ticks / (LEVEL_SECS * 60)) * 100;
        const pelletPct = ((levelsCleared * 244 + (244 - states[o + A.dotsLeft])) / 244) * 100;
        const wastedPct = (states[o + A.wasted] / Math.max(1, ticks)) * 100;
        // No clamp to a positive floor: tournament selection only ever compares
        // fitnesses, so negatives are fine — and clamping would flatten the
        // whole of generation 1 to a single value and blind the tournament.
        const composite =
          scorePct * FIT_SCORE_W +
          survivalPct * FIT_SURVIVAL_W +
          pelletPct * FIT_PELLET_PCT_W -
          wastedPct * FIT_WASTED_W;
        this.bestLevel = Math.max(this.bestLevel, level);
        fitnesses[i] = composite;
        if (fitnesses[i] > this.bestFitness) {
          this.bestFitness = fitnesses[i];
          this.bestScore = score;
          this.bestLevel = level;
          this.bestGeneration = this.generation;
          this.bestGenome = new Float64Array(this.genomes[i]);
          this.genomes[this.displayIndex].set(this.bestGenome);
          autosavePacmanBest(this.bestGenome, this.bestGeneration, this.bestFitness, this.bestScore, this.bestLevel);
        }
      }

      // §3.3's adaptive mutation: 当连续5代最佳适应度没有显著提升时，轻微提高变异率.
      // Raising it is how the population climbs out of the §4.4 plateau, where
      // novel individuals otherwise get culled before they can pay off.
      let generationBest = -Infinity;
      for (let i = 0; i < this.populationSize; i++) generationBest = Math.max(generationBest, fitnesses[i]);
      if (generationBest > this.lastBestFitness) {
        this.lastBestFitness = generationBest;
        this.stagnantGenerations = 0;
        // Decay back down as progress resumes — §3.3 wants a *lower* rate late on
        // (降低变异率，进行精细优化). Ratcheting up and staying there turns the GA
        // into random search and the population actively gets worse.
        this.mutateRate = Math.max(BASE_MUTATE_RATE, this.mutateRate - MUTATE_STEP);
      } else if (++this.stagnantGenerations >= STAGNATION_LIMIT) {
        this.stagnantGenerations = 0;
        this.mutateRate = Math.min(MAX_MUTATE_RATE, this.mutateRate + MUTATE_STEP);
      }

      // Tournament selection (§3.2): 对于《吃豆人》这个问题，锦标赛选择的效果更好,
      // because it purges the useless random strategies fast.
      const next = nextTournamentGeneration(this.genomes.slice(0, this.populationSize), fitnesses, this.rng, {
        tournamentSize: TOURNAMENT_K,
        eliteFraction: ELITE_FRACTION,
        crossoverRate: CROSSOVER_RATE,
        mutateRate: this.mutateRate,
        mutateRange: INIT_RANGE,
      });
      this.genomes = [...next, new Float64Array(this.bestGenome)];
      this.generation++;
      this.shownIndex = -1;
      this.episodeSeed = Math.floor(this.rng() * 0xffffffff) >>> 0;
      this.rerollEnvironment();
      this.writeParams();
      this.uploadGenomes();
      this.device.queue.writeBuffer(this.buffers.agents, 0, initialAgentStates(this.buffers.totalAgents, this.episodeSeed));
    } finally {
      this.isEvolving = false;
    }
  }

  /** Snapshot of the most successful visible agent. Train mode follows the current population; Play mode follows the player/best replay slot. */
  async readBestAgentState(): Promise<BestAgentSnapshot> {
    const states = await this.readStates();
    let aliveCount = 0;
    let shown = this.displayIndex;
    let bestDotsLeft = Infinity;
    let bestScore = -Infinity;
    let bestTicks = Infinity;
    for (let i = 0; i < this.populationSize; i++) {
      const o = i * AGENT_FLOATS;
      if (states[o + A.gameOver] < 0.5) {
        aliveCount++;
        if (!this.playMode) {
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
    if (!this.playMode && this.populationSize === 1) shown = 0;
    // Stick with whoever is on screen while they are still alive and still the
    // leader by dot count. Re-picking a strictly-better agent every frame makes
    // the camera jump between two pacmen mid-corridor, which reads as jitter
    // even though each individual agent is moving smoothly.
    if (!this.playMode && this.shownIndex >= 0 && this.shownIndex < this.populationSize) {
      const prev = this.shownIndex * AGENT_FLOATS;
      if (states[prev + A.gameOver] < 0.5 && states[prev + A.dotsLeft] <= bestDotsLeft + DISPLAY_STICKINESS) {
        shown = this.shownIndex;
      }
    }
    this.shownIndex = shown;
    const o = shown * AGENT_FLOATS;
    return {
      index: shown,
      score: states[o + A.score],
      lives: states[o + A.lives],
      dotsLeft: states[o + A.dotsLeft],
      level: states[o + A.level],
      levelTimeLeft: Math.max(0, LEVEL_SECS - states[o + A.levelTicks] / 60),
      gameOver: states[o + A.gameOver] > 0.5,
      aliveCount,
      bestScore: this.bestScore,
      bestGeneration: this.bestGeneration,
      bestLevel: this.bestLevel,
    };
  }
}

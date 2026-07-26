import { aggregateEpisodes, episodeSpread } from '../../utils/evaluation';
import { nextTournamentGeneration } from '../../utils/ga';
import { mulberry32, type Rng } from '../../utils/rng';
import {
  A,
  AGENT_FLOATS,
  createPacmanBuffers,
  EPISODES_PER_GENOME,
  GHOST_CHAOS,
  FIT_PELLET_PCT_W,
  FIT_SCORE_NORM,
  FIT_SCORE_W,
  FIT_SURVIVAL_W,
  FIT_WASTED_W,
  initialAgentStates,
  LEVEL_SECS,
  LEVEL_SECS_MAX,
  LEVEL_SECS_MIN,
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
/** Half the mutations reset (explore), half drift (let confident weights grow). */
const RESET_SHARE = 0.5;
const DRIFT_SIGMA = 0.5;
const WEIGHT_CLAMP = 8;
/** Softmax temperature for training agents; lower = more decisive, less exploration. */
const ACTION_TEMPERATURE = 1;

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
  /** Genome slot holding the best-so-far replay genome. */
  private readonly displayGenomeIndex: number;
  /** Agent slot that replay genome runs in (the trailing agent). */
  private readonly displayAgentIndex: number;
  private bestGenome: Float64Array;
  private bestFitness = -Infinity;
  private bestScore = 0;
  private bestLevel = 1;
  /** Highest level any agent has ever reached, for the HUD. */
  private highestLevel = 1;
  private bestGeneration = 1;
  private readPending: Promise<Float32Array<ArrayBuffer>> | null = null;
  private isEvolving = false;
  private playMode = false;
  private episodeSeed = 1;
  private shownIndex = -1;
  private replaySeed = 1;
  private lastSpread = 0;
  private mutateRate = BASE_MUTATE_RATE;
  private stagnantGenerations = 0;
  private lastBestFitness = -Infinity;
  private actionTemperature = ACTION_TEMPERATURE;
  private ghostChaos = GHOST_CHAOS;
  private levelSeconds = LEVEL_SECS;
  private ghostSpeedScale = 1;
  private houseReleaseScale = 1;

  private constructor(
    private readonly device: GPUDevice,
    readonly buffers: PacmanBuffers,
    readonly populationSize: number,
    seed: number,
    readonly episodes: number,
  ) {
    this.rng = mulberry32(seed);
    this.displayGenomeIndex = populationSize;
    this.displayAgentIndex = populationSize * episodes;
    this.genomes = Array.from({ length: populationSize + 1 }, () => randomNetwork(this.rng));
    this.bestGenome = new Float64Array(this.genomes[0]);
    this.genomes[this.displayGenomeIndex].set(this.bestGenome);
    this.uploadGenomes();
    this.device.queue.writeBuffer(this.buffers.agents, 0, initialAgentStates(this.buffers.totalAgents, this.episodeSeed, this.episodes));

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

  static init(device: GPUDevice, populationSize = 300, seed = 1, episodes = EPISODES_PER_GENOME): PacmanEvolution {
    const eps = Math.max(1, Math.floor(episodes));
    return new PacmanEvolution(device, createPacmanBuffers(device, populationSize, eps), populationSize, seed, eps);
  }

  static readonly topology = PACMAN_TOPOLOGY;

  /** The genome driving a given *agent* slot (episodes of one genome are adjacent). */
  genomeAt(agentIndex: number): Float64Array {
    return this.genomes[Math.min(this.populationSize, Math.floor(agentIndex / this.episodes))];
  }

  displayGenome(): Float64Array {
    return this.genomes[this.displayGenomeIndex];
  }

  /** Live GA state for the HUD: the adaptive mutation rate and its stall counter. */
  gaState(): { mutateRate: number; stagnant: number; episodes: number; spread: number } {
    return {
      mutateRate: this.mutateRate,
      stagnant: this.stagnantGenerations,
      episodes: this.episodes,
      spread: this.lastSpread,
    };
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
    this.genomes[this.displayGenomeIndex] = weights.slice();
    this.bestGenome = weights.slice();
    this.bestFitness = eval_;
    this.bestScore = score;
    this.bestLevel = level;
    this.bestGeneration = generation;
    this.uploadGenomes();
    this.device.queue.writeBuffer(this.buffers.agents, 0, initialAgentStates(this.buffers.totalAgents, this.episodeSeed, this.episodes));
  }

  setPlayMode(playMode: boolean): void {
    this.playMode = playMode;
    this.writeParams();
  }

  /**
   * Per-board time budget, live. Agents already past the new limit end on the
   * next tick, so lowering it mid-generation culls the dawdlers immediately.
   */
  setLevelSeconds(seconds: number): void {
    this.levelSeconds = Math.min(LEVEL_SECS_MAX, Math.max(LEVEL_SECS_MIN, seconds));
    this.writeParams();
  }

  levelSecondsValue(): number {
    return this.levelSeconds;
  }

  /** Chance a ghost takes a random legal turn; 0 restores pure arcade behaviour. */
  setGhostChaos(p: number): void {
    this.ghostChaos = Math.min(1, Math.max(0, p));
    this.writeParams();
  }

  /** Softmax temperature for training agents. 0 takes the mode (no exploration). */
  setActionTemperature(t: number): void {
    this.actionTemperature = Math.max(0, t);
    this.writeParams();
  }

  private writeParams(): void {
    const data = new ArrayBuffer(48);
    // Training agents sample from the softmax; the replay/test slot takes the
    // mode so the pacman you watch behaves deterministically.
    new Uint32Array(data).set([this.buffers.totalAgents, this.episodes, this.playMode ? 1 : 0, this.episodeSeed]);
    new Float32Array(data, 16).set([this.ghostSpeedScale, this.houseReleaseScale]);
    // Every network-driven agent samples at the same temperature it was evolved
    // under, including the replay/Test slot. Forcing the replay to argmax makes
    // the pacman you watch (and the model you export) a *different, far worse*
    // policy than the one selection actually scored — measured at 10 pellets
    // versus 187 for the very same genome. The human player's agent never runs
    // the network at all, so play mode needs no special case.
    new Float32Array(data, 24).set([this.actionTemperature]);
    new Uint32Array(data, 28).set([Math.round(this.levelSeconds * 60)]);
    new Float32Array(data, 32).set([this.ghostChaos]);
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
    const o = this.displayAgentIndex * AGENT_FLOATS;
    const data = new Float32Array([dir]);
    this.device.queue.writeBuffer(this.buffers.agents, (o + A.desired) * 4, data);
  }

  resetDisplayAgent(): void {
    this.resetAgent(this.displayAgentIndex);
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

  /**
   * Re-seeded on every restart. Actions are sampled, so a fixed seed would make
   * the agent replay one canned episode forever — you would never see the range
   * of behaviour the policy actually has.
   */
  private resetAgent(index: number): void {
    this.replaySeed = (this.replaySeed + 1) >>> 0;
    this.device.queue.writeBuffer(this.buffers.agents, index * AGENT_FLOATS * 4, initialAgentStates(1, this.replaySeed));
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
      if (states[this.displayAgentIndex * AGENT_FLOATS + A.gameOver] > 0.5) {
        this.resetAgent(this.displayAgentIndex);
      }
      const trainingAgents = this.populationSize * this.episodes;
      let aliveCount = 0;
      for (let i = 0; i < trainingAgents; i++) {
        if (states[i * AGENT_FLOATS + A.gameOver] < 0.5) aliveCount++;
      }
      if (aliveCount > 0) return;

      // Composite fitness of §3.1. The writeup's first attempt was
      // `fitness = score`, and evolution immediately found the local optimum of
      // standing still: score 0, but never caught. Four terms fix it, and the
      // relative weights carry the lesson — survival is weighted *lowest* (0.1)
      // because weighting it heavily breeds hiding strategies, while the
      // percentage of pellets eaten is the core driving force at 2.0.
      //
      // Scored per *episode*; genomes are then judged on the aggregate, because
      // one episode of a sampled policy is closer to a lottery ticket than a
      // measurement. See src/utils/evaluation.ts.
      const episodeScores = new Float64Array(trainingAgents);
      for (let i = 0; i < trainingAgents; i++) {
        const o = i * AGENT_FLOATS;
        const score = states[o + A.score];
        const level = states[o + A.level];
        const levelsCleared = Math.max(0, level - 1);
        const ticks = states[o + A.ticks];
        // All four terms on a 0-100 scale, so the writeup's 1 : 0.1 : 2 ratios
        // carry their intended meaning rather than being swamped by raw score.
        const scorePct = (score / FIT_SCORE_NORM) * 100;
        const survivalPct = (ticks / (this.levelSeconds * 60)) * 100;
        const pelletPct = ((levelsCleared * 244 + (244 - states[o + A.dotsLeft])) / 244) * 100;
        const wastedPct = (states[o + A.wasted] / Math.max(1, ticks)) * 100;
        // No clamp to a positive floor: tournament selection only ever compares
        // fitnesses, so negatives are fine — and clamping would flatten the
        // whole of generation 1 to a single value and blind the tournament.
        episodeScores[i] =
          scorePct * FIT_SCORE_W +
          survivalPct * FIT_SURVIVAL_W +
          pelletPct * FIT_PELLET_PCT_W -
          wastedPct * FIT_WASTED_W;
        this.highestLevel = Math.max(this.highestLevel, level);
      }

      const fitnesses = aggregateEpisodes(episodeScores, this.populationSize, this.episodes);
      this.lastSpread = episodeSpread(episodeScores, this.populationSize, this.episodes).meanRange;

      // Which genome gets saved is ranked by the level it *typically* reaches
      // first, and only then by fitness.
      //
      // Composite fitness alone nearly ties the two cases we care most about
      // separating: a genome that has just cleared a board sits at pelletPct
      // 100 with an empty new board, and one that ate 243 pellets and died sits
      // at 99.6. Sorting on level first makes the board-clearer win outright,
      // which is what "save the best model" should mean. Levels are coarse and
      // most genomes never clear one, so in practice this is a tie-break that
      // only fires where it matters.
      //
      // Selection inside the GA is untouched and still uses the composite: a
      // sparse integer would be a poor gradient to evolve against.
      for (let g = 0; g < this.populationSize; g++) {
        // Median episode, so the recorded figures are ones it can reproduce.
        const runs = Array.from({ length: this.episodes }, (_, e) => g * this.episodes + e)
          .sort((a, b) => episodeScores[a] - episodeScores[b]);
        const typical = runs[runs.length >> 1] * AGENT_FLOATS;
        const typicalLevel = states[typical + A.level];
        const better =
          typicalLevel > this.bestLevel ||
          (typicalLevel === this.bestLevel && fitnesses[g] > this.bestFitness);
        if (!better) continue;
        this.bestFitness = fitnesses[g];
        this.bestScore = states[typical + A.score];
        this.bestLevel = typicalLevel;
        this.bestGeneration = this.generation;
        this.bestGenome = new Float64Array(this.genomes[g]);
        this.genomes[this.displayGenomeIndex].set(this.bestGenome);
        autosavePacmanBest(this.bestGenome, this.bestGeneration, this.bestFitness, this.bestScore, this.bestLevel);
      }

      // §3.3's adaptive mutation: nudge the rate up after several generations
      // with no improvement in the best fitness. Raising it is how the
      // population climbs out of the §4.4 plateau, where novel individuals
      // otherwise get culled before they can pay off.
      let generationBest = -Infinity;
      for (let g = 0; g < this.populationSize; g++) generationBest = Math.max(generationBest, fitnesses[g]);
      if (generationBest > this.lastBestFitness) {
        this.lastBestFitness = generationBest;
        this.stagnantGenerations = 0;
        // Decay back down as progress resumes — §3.3 wants a *lower* rate late
        // on, for fine optimisation. Ratcheting up and staying there turns the
        // GA into random search and the population actively gets worse.
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
        resetShare: RESET_SHARE,
        driftSigma: DRIFT_SIGMA,
        clamp: WEIGHT_CLAMP,
      });
      this.genomes = [...next, new Float64Array(this.bestGenome)];
      this.generation++;
      this.shownIndex = -1;
      this.episodeSeed = Math.floor(this.rng() * 0xffffffff) >>> 0;
      this.rerollEnvironment();
      this.writeParams();
      this.uploadGenomes();
      this.device.queue.writeBuffer(this.buffers.agents, 0, initialAgentStates(this.buffers.totalAgents, this.episodeSeed, this.episodes));
    } finally {
      this.isEvolving = false;
    }
  }

  /** Snapshot of the most successful visible agent. Train mode follows the current population; Play mode follows the player/best replay slot. */
  async readBestAgentState(): Promise<BestAgentSnapshot> {
    const states = await this.readStates();
    let aliveCount = 0;
    let shown = this.displayAgentIndex;
    let bestDotsLeft = Infinity;
    let bestScore = -Infinity;
    let bestTicks = Infinity;
    const trainingAgents = this.populationSize * this.episodes;
    for (let i = 0; i < trainingAgents; i++) {
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
    if (!this.playMode && trainingAgents === 1) shown = 0;
    // Stick with whoever is on screen while they are still alive and still the
    // leader by dot count. Re-picking a strictly-better agent every frame makes
    // the camera jump between two pacmen mid-corridor, which reads as jitter
    // even though each individual agent is moving smoothly.
    if (!this.playMode && this.shownIndex >= 0 && this.shownIndex < trainingAgents) {
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
      levelTimeLeft: Math.max(0, this.levelSeconds - states[o + A.levelTicks] / 60),
      gameOver: states[o + A.gameOver] > 0.5,
      aliveCount,
      bestScore: this.bestScore,
      bestGeneration: this.bestGeneration,
      bestLevel: this.highestLevel,
    };
  }
}

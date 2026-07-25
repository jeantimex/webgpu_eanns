import { nextCrossoverGeneration } from '../../utils/ga';
import { mulberry32, type Rng } from '../../utils/rng';
import {
  A,
  AGENT_FLOATS,
  createPacmanBuffers,
  initialAgentStates,
  PACMAN_GENOME_SIZE,
  PACMAN_TOPOLOGY,
  type PacmanBuffers,
} from './pacman_buffers';
import { pacmanShader } from './pacman.wgsl';
import { autosavePacmanBest } from './pacman_model';

export interface BestAgentSnapshot {
  index: number;
  score: number;
  lives: number;
  dotsLeft: number;
  level: number;
  gameOver: boolean;
  aliveCount: number;
  bestScore: number;
  bestGeneration: number;
}

/**
 * Generation driver. Each agent plays a full game (3 attempts, level resets,
 * 6-minute cap) inside the compute shader — there is no shared world state, so
 * a generation is a fixed number of dispatches followed by a GA step.
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
  private bestGeneration = 1;
  private readPending: Promise<Float32Array<ArrayBuffer>> | null = null;
  private isEvolving = false;

  private constructor(
    private readonly device: GPUDevice,
    readonly buffers: PacmanBuffers,
    readonly populationSize: number,
    seed: number,
  ) {
    this.rng = mulberry32(seed);
    this.displayIndex = populationSize;
    this.genomes = Array.from({ length: buffers.totalAgents }, () => {
      const g = new Float64Array(PACMAN_GENOME_SIZE);
      for (let k = 0; k < PACMAN_GENOME_SIZE; k++) g[k] = this.rng() * 2 - 1;
      return g;
    });
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

  bestMeta(): { generation: number; eval: number; score: number } {
    return {
      generation: this.bestGeneration,
      eval: Number.isFinite(this.bestFitness) ? this.bestFitness : 0,
      score: this.bestScore,
    };
  }

  injectBest(weights: Float64Array): void {
    this.genomes[0] = weights.slice();
    this.genomes[this.displayIndex] = weights.slice();
    this.bestGenome = weights.slice();
    this.bestFitness = Infinity;
    this.bestScore = 0;
    this.bestGeneration = this.generation;
    this.uploadGenomes();
    this.device.queue.writeBuffer(this.buffers.agents, 0, initialAgentStates(this.buffers.totalAgents));
  }

  setPlayMode(playMode: boolean): void {
    const paramsData = new Uint32Array([this.buffers.totalAgents, this.populationSize, playMode ? 1 : 0, 0]);
    this.device.queue.writeBuffer(this.buffers.params, 0, paramsData);
  }

  setPlayerDesiredDir(dir: number): void {
    const o = this.displayIndex * AGENT_FLOATS;
    const data = new Float32Array([dir]);
    this.device.queue.writeBuffer(this.buffers.agents, (o + A.desired) * 4, data);
  }

  resetDisplayAgent(): void {
    this.resetAgent(this.displayIndex);
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

      // GA sees final episode fitness: completion is the top signal,
      // then faster completion, pellet progress, and movement as a tie-breaker.
      const fitnesses = new Float64Array(this.populationSize);
      for (let i = 0; i < this.populationSize; i++) {
        const o = i * AGENT_FLOATS;
        const score = states[o + A.score];
        const dotsLeft = states[o + A.dotsLeft];
        const pelletsEaten = 244 - dotsLeft;
        const moveTicks = states[o + A.moveTicks];
        const pelletProgress = states[o + A.pelletProgress];
        const totalReward = states[o + A.totalReward];
        const ticks = Math.max(1, states[o + A.ticks]);
        const scoreRate = score / ticks;
        const finished = dotsLeft <= 0 ? 1 : 0;
        const fitness =
          pelletsEaten * 500 +
          score * 100 +
          pelletProgress * 50 +
          scoreRate * 10000 +
          finished * 1_000_000 +
          finished * (21600 - ticks) * 100 +
          Math.max(-500, totalReward * 10) +
          moveTicks * 0.1;
        fitnesses[i] = Math.max(1, fitness);
        if (fitnesses[i] > this.bestFitness) {
          this.bestFitness = fitnesses[i];
          this.bestScore = score;
          this.bestGeneration = this.generation;
          this.bestGenome = new Float64Array(this.genomes[i]);
          this.genomes[this.displayIndex].set(this.bestGenome);
          autosavePacmanBest(this.bestGenome, this.bestGeneration, this.bestFitness, this.bestScore);
        }
      }
      const next = nextCrossoverGeneration(
        this.genomes.slice(0, this.populationSize),
        fitnesses,
        this.rng,
        PACMAN_TOPOLOGY,
        { eliteCount: 8, mutateRate: 0.05, sigma: 0.1, clamp: 1.0 },
      );
      this.genomes = [...next, new Float64Array(this.bestGenome)];
      this.generation++;
      this.uploadGenomes();
      this.device.queue.writeBuffer(this.buffers.agents, 0, initialAgentStates(this.buffers.totalAgents));
    } finally {
      this.isEvolving = false;
    }
  }

  /** Snapshot of the highest-score agent still playing (falls back to best overall), for the HUD/renderer. */
  async readBestAgentState(): Promise<BestAgentSnapshot> {
    const states = await this.readStates();
    let aliveCount = 0;
    for (let i = 0; i < this.populationSize; i++) {
      const o = i * AGENT_FLOATS;
      if (states[o + A.gameOver] < 0.5) {
        aliveCount++;
      }
    }
    const shown = this.displayIndex;
    const o = shown * AGENT_FLOATS;
    return {
      index: shown,
      score: states[o + A.score],
      lives: states[o + A.lives],
      dotsLeft: states[o + A.dotsLeft],
      level: states[o + A.level],
      gameOver: states[o + A.gameOver] > 0.5,
      aliveCount,
      bestScore: this.bestScore,
      bestGeneration: this.bestGeneration,
    };
  }
}

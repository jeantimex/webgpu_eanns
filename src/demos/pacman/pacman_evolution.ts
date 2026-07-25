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

const WEIGHT_MIN = 1e-5;
const WEIGHT_MAX = 20;
const ELITE_COUNT = 5;
const TOURNAMENT_SIZE = 5;
const CROSSOVER_PROB = 0.7;
const CROSSOVER_ALPHA = 0.7;
const MUTATION_PROB = 0.6;

function randomWeight(rng: Rng): number {
  return WEIGHT_MIN + rng() * (WEIGHT_MAX - WEIGHT_MIN);
}

function tournamentPick(population: Float64Array[], fitnesses: ArrayLike<number>, rng: Rng): Float64Array {
  let best = Math.floor(rng() * population.length);
  for (let k = 1; k < Math.min(TOURNAMENT_SIZE, population.length); k++) {
    const candidate = Math.floor(rng() * population.length);
    if (fitnesses[candidate] > fitnesses[best]) best = candidate;
  }
  return population[best];
}

function nextWeightedVectorGeneration(population: Float64Array[], fitnesses: ArrayLike<number>, rng: Rng): Float64Array[] {
  const order = population.map((_, i) => i).sort((a, b) => fitnesses[b] - fitnesses[a]);
  const next: Float64Array[] = [];
  for (let i = 0; i < Math.min(ELITE_COUNT, population.length); i++) {
    next.push(population[order[i]].slice());
  }

  while (next.length < population.length) {
    const p1 = tournamentPick(population, fitnesses, rng);
    const p2 = tournamentPick(population, fitnesses, rng);
    const child = p1.slice();
    if (rng() <= CROSSOVER_PROB) {
      const cut = Math.min(PACMAN_GENOME_SIZE, Math.floor(rng() * PACMAN_GENOME_SIZE) + 1);
      for (let k = cut; k < PACMAN_GENOME_SIZE; k++) {
        child[k] = CROSSOVER_ALPHA * p1[k] + (1 - CROSSOVER_ALPHA) * p2[k];
      }
    }
    if (rng() <= MUTATION_PROB) {
      child[Math.floor(rng() * 4)] = randomWeight(rng);
      child[4 + Math.floor(rng() * 4)] = randomWeight(rng);
    }
    next.push(child);
  }
  return next;
}

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
  private bestGeneration = 1;
  private readPending: Promise<Float32Array<ArrayBuffer>> | null = null;
  private isEvolving = false;
  private playMode = false;
  private episodeSeed = 1;

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
      for (let k = 0; k < PACMAN_GENOME_SIZE; k++) g[k] = randomWeight(this.rng);
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

  injectBest(weights: Float64Array, eval_ = 0, score = 0, generation = this.generation): void {
    this.genomes[0] = weights.slice();
    this.genomes[this.displayIndex] = weights.slice();
    this.bestGenome = weights.slice();
    this.bestFitness = eval_;
    this.bestScore = score;
    this.bestGeneration = generation;
    this.uploadGenomes();
    this.device.queue.writeBuffer(this.buffers.agents, 0, initialAgentStates(this.buffers.totalAgents));
  }

  setPlayMode(playMode: boolean): void {
    this.playMode = playMode;
    this.writeParams();
  }

  private writeParams(): void {
    const paramsData = new Uint32Array([this.buffers.totalAgents, this.populationSize, this.playMode ? 1 : 0, this.episodeSeed]);
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

      // MatheusPaixaoG/Pacman-with-GA uses final game score as fitness.
      const fitnesses = new Float64Array(this.populationSize);
      for (let i = 0; i < this.populationSize; i++) {
        const o = i * AGENT_FLOATS;
        const score = states[o + A.score];
        fitnesses[i] = Math.max(1, score);
        if (fitnesses[i] > this.bestFitness) {
          this.bestFitness = fitnesses[i];
          this.bestScore = score;
          this.bestGeneration = this.generation;
          this.bestGenome = new Float64Array(this.genomes[i]);
          this.genomes[this.displayIndex].set(this.bestGenome);
          autosavePacmanBest(this.bestGenome, this.bestGeneration, this.bestFitness, this.bestScore);
        }
      }
      const next = nextWeightedVectorGeneration(this.genomes.slice(0, this.populationSize), fitnesses, this.rng);
      this.genomes = [...next, new Float64Array(this.bestGenome)];
      this.generation++;
      this.episodeSeed = Math.floor(this.rng() * 0xffffffff) >>> 0;
      this.writeParams();
      this.uploadGenomes();
      this.device.queue.writeBuffer(this.buffers.agents, 0, initialAgentStates(this.buffers.totalAgents));
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

import { nextCrossoverGeneration } from '../../utils/ga';
import { mulberry32, type Rng } from '../../utils/rng';
import {
  A,
  AGENT_FLOATS,
  createSnakeBuffers,
  initialAgentStates,
  SNAKE_GENOME_SIZE,
  SNAKE_TOPOLOGY,
  type SnakeBuffers,
} from './snake_buffers';
import { snakeShader } from './snake.wgsl';
import { autosaveSnakeBest } from './model';

export interface BestSnakeSnapshot {
  index: number;
  apples: number;
  length: number;
  moves: number;
  score: number;
  life: number;
  gameOver: boolean;
  aliveCount: number;
}

/** Generation driver: each agent plays one snake game inside the compute shader. */
export class SnakeEvolution {
  generation = 1;
  readonly displayIndex: number;

  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroup: GPUBindGroup;
  private readonly workgroups: number;
  private readonly rng: Rng;
  private readonly agentCount: number;
  private genomes: Float64Array[];
  private readPending: Promise<Float32Array<ArrayBuffer>> | null = null;
  private isEvolving = false;
  private bestFitness = -Infinity;
  private bestScore = 3;
  private bestGeneration = 1;
  private bestReplaySeed = 0;

  private constructor(
    private readonly device: GPUDevice,
    readonly buffers: SnakeBuffers,
    readonly populationSize: number,
    seed: number,
  ) {
    this.rng = mulberry32(seed);
    this.displayIndex = populationSize;
    this.agentCount = populationSize + 1;
    this.genomes = Array.from({ length: this.agentCount }, () => {
      const g = new Float64Array(SNAKE_GENOME_SIZE);
      for (let k = 0; k < SNAKE_GENOME_SIZE; k++) g[k] = this.rng() * 2 - 1;
      return g;
    });
    this.genomes[this.displayIndex] = this.genomes[0].slice();
    this.uploadGenomes();
    this.device.queue.writeBuffer(this.buffers.agents, 0, initialAgentStates(this.agentCount));
    this.resetDisplayAgent();

    this.pipeline = device.createComputePipeline({
      label: 'snake pipeline',
      layout: 'auto',
      compute: { module: device.createShaderModule({ label: 'snake shader', code: snakeShader }), entryPoint: 'main' },
    });
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.params } },
        { binding: 1, resource: { buffer: buffers.genomes } },
        { binding: 2, resource: { buffer: buffers.agents } },
      ],
    });
    this.workgroups = Math.ceil(this.agentCount / 64);
  }

  static init(device: GPUDevice, populationSize = 300, seed = 1): SnakeEvolution {
    return new SnakeEvolution(device, createSnakeBuffers(device, populationSize + 1), populationSize, seed);
  }

  /** CPU-side genome of agent `index` (740 floats, uploaded to the GPU each generation). */
  genomeAt(index: number): Float64Array {
    return this.genomes[index];
  }

  bestGenome(): Float64Array {
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
    this.bestFitness = Infinity;
    this.bestScore = 0;
    this.bestGeneration = this.generation;
    this.bestReplaySeed = 0;
    this.uploadGenomes();
    this.resetTrainingAgents();
    this.resetDisplayAgent();
  }

  private uploadGenomes(): void {
    const flat = new Float32Array(this.genomes.length * SNAKE_GENOME_SIZE);
    for (let i = 0; i < this.genomes.length; i++) flat.set(this.genomes[i], i * SNAKE_GENOME_SIZE);
    this.device.queue.writeBuffer(this.buffers.genomes, 0, flat);
  }

  private resetAgents(count: number, seedOffset: number, byteOffset: number): void {
    this.device.queue.writeBuffer(this.buffers.agents, byteOffset, initialAgentStates(count, seedOffset));
  }

  private resetTrainingAgents(): void {
    this.resetAgents(this.populationSize, 0, 0);
  }

  private resetDisplayAgent(): void {
    this.resetAgents(1, this.bestReplaySeed, this.displayIndex * AGENT_FLOATS * 4);
  }

  private fitnessAt(states: Float32Array<ArrayBuffer>, index: number): number {
    const o = index * AGENT_FLOATS;
    const score = states[o + A.score];
    const lifetimeSq = states[o + A.moves] ** 2;
    return score < 10 ? lifetimeSq * 2 ** score : lifetimeSq * 1024 * (score - 9);
  }

  /** Dispatch k moves (turn-based: one dispatch = one move). */
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

  /** Generation ends when every snake is dead (wall, self, stall, or cap). */
  async checkAndEvolve(): Promise<void> {
    if (this.isEvolving) return;
    this.isEvolving = true;
    try {
      const states = await this.readStates();
      let aliveCount = 0;
      for (let i = 0; i < this.populationSize; i++) {
        if (states[i * AGENT_FLOATS + A.gameOver] < 0.5) aliveCount++;
      }
      if (aliveCount > 0) return;
      if (states[this.displayIndex * AGENT_FLOATS + A.gameOver] < 0.5) return;

      // SnakeAI fitness: lifetime^2 x 2^score, with a 2^10 bonus per score past 9
      // (score = length; lifetime = moves survived).
      const fitnesses = new Float64Array(this.populationSize);
      let bestIndex = 0;
      for (let i = 0; i < this.populationSize; i++) {
        fitnesses[i] = this.fitnessAt(states, i);
        if (fitnesses[i] > fitnesses[bestIndex]) bestIndex = i;
      }
      if (fitnesses[bestIndex] > this.bestFitness) {
        this.bestFitness = fitnesses[bestIndex];
        this.bestScore = states[bestIndex * AGENT_FLOATS + A.score];
        this.bestGeneration = this.generation;
        this.bestReplaySeed = bestIndex;
        this.genomes[this.displayIndex] = this.genomes[bestIndex].slice();
        autosaveSnakeBest(this.genomes[this.displayIndex], this.bestGeneration, this.bestFitness, this.bestScore);
      }
      const nextTraining = nextCrossoverGeneration(this.genomes.slice(0, this.populationSize), fitnesses, this.rng, SNAKE_TOPOLOGY);
      this.genomes.splice(0, this.populationSize, ...nextTraining);
      this.generation++;
      this.uploadGenomes();
      this.resetTrainingAgents();
      this.resetDisplayAgent();
    } finally {
      this.isEvolving = false;
    }
  }

  /** Test-mode behavior: replay the loaded/best model again after it dies. */
  async restartDisplayIfDead(): Promise<void> {
    const states = await this.readStates();
    if (states[this.displayIndex * AGENT_FLOATS + A.gameOver] > 0.5) {
      this.resetDisplayAgent();
    }
  }

  /** Snapshot of the replay agent shown during training; aliveCount is the active training population. */
  async readBestAgentState(): Promise<BestSnakeSnapshot> {
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
      apples: states[o + A.apples],
      length: states[o + A.length],
      moves: states[o + A.moves],
      score: states[o + A.score],
      life: states[o + A.sinceEat],
      gameOver: states[o + A.gameOver] > 0.5,
      aliveCount,
    };
  }
}

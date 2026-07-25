import { nextRouletteGeneration } from '../../utils/ga';
import { mulberry32, type Rng } from '../../utils/rng';
import {
  A,
  AGENT_FLOATS,
  createSnakeBuffers,
  initialAgentStates,
  SNAKE_GENOME_SIZE,
  type SnakeBuffers,
} from './snake_buffers';
import { snakeShader } from './snake.wgsl';

export interface BestSnakeSnapshot {
  index: number;
  apples: number;
  length: number;
  moves: number;
  gameOver: boolean;
  aliveCount: number;
}

/** Generation driver: each agent plays one snake game inside the compute shader. */
export class SnakeEvolution {
  generation = 1;

  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroup: GPUBindGroup;
  private readonly workgroups: number;
  private readonly rng: Rng;
  private genomes: Float64Array[];
  private readPending: Promise<Float32Array<ArrayBuffer>> | null = null;
  private isEvolving = false;

  private constructor(
    private readonly device: GPUDevice,
    readonly buffers: SnakeBuffers,
    readonly populationSize: number,
    seed: number,
  ) {
    this.rng = mulberry32(seed);
    this.genomes = Array.from({ length: populationSize }, () => {
      const g = new Float64Array(SNAKE_GENOME_SIZE);
      for (let k = 0; k < SNAKE_GENOME_SIZE; k++) g[k] = this.rng() * 2 - 1;
      return g;
    });
    this.uploadGenomes();
    this.device.queue.writeBuffer(this.buffers.agents, 0, initialAgentStates(populationSize));

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
    this.workgroups = Math.ceil(populationSize / 64);
  }

  static init(device: GPUDevice, populationSize = 300, seed = 1): SnakeEvolution {
    return new SnakeEvolution(device, createSnakeBuffers(device, populationSize), populationSize, seed);
  }

  /** CPU-side genome of agent `index` (219 floats, uploaded to the GPU each generation). */
  genomeAt(index: number): Float64Array {
    return this.genomes[index];
  }

  private uploadGenomes(): void {
    const flat = new Float32Array(this.genomes.length * SNAKE_GENOME_SIZE);
    for (let i = 0; i < this.genomes.length; i++) flat.set(this.genomes[i], i * SNAKE_GENOME_SIZE);
    this.device.queue.writeBuffer(this.buffers.genomes, 0, flat);
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

      // Apples dominate (quadratic); tiny survival term separates zero-apple agents.
      const fitnesses = new Float64Array(this.populationSize);
      for (let i = 0; i < this.populationSize; i++) {
        const o = i * AGENT_FLOATS;
        fitnesses[i] = states[o + A.apples] ** 2 + states[o + A.moves] * 0.001;
      }
      this.genomes = nextRouletteGeneration(this.genomes, fitnesses, this.rng);
      this.generation++;
      this.uploadGenomes();
      this.device.queue.writeBuffer(this.buffers.agents, 0, initialAgentStates(this.populationSize));
    } finally {
      this.isEvolving = false;
    }
  }

  /** Snapshot of the agent with the most apples still playing (fallback: best overall). */
  async readBestAgentState(): Promise<BestSnakeSnapshot> {
    const states = await this.readStates();
    let best = 0;
    let bestAlive = -1;
    let aliveCount = 0;
    for (let i = 0; i < this.populationSize; i++) {
      const o = i * AGENT_FLOATS;
      if (states[o + A.apples] > states[best * AGENT_FLOATS + A.apples]) best = i;
      if (states[o + A.gameOver] < 0.5) {
        aliveCount++;
        if (bestAlive < 0 || states[o + A.apples] > states[bestAlive * AGENT_FLOATS + A.apples]) bestAlive = i;
      }
    }
    const shown = bestAlive >= 0 ? bestAlive : best;
    const o = shown * AGENT_FLOATS;
    return {
      index: shown,
      apples: states[o + A.apples],
      length: states[o + A.length],
      moves: states[o + A.moves],
      gameOver: states[o + A.gameOver] > 0.5,
      aliveCount,
    };
  }
}

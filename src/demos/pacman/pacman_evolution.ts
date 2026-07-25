import { nextRouletteGeneration } from '../../utils/ga';
import { mulberry32, type Rng } from '../../utils/rng';
import {
  A,
  AGENT_FLOATS,
  createPacmanBuffers,
  initialAgentStates,
  PACMAN_GENOME_SIZE,
  type PacmanBuffers,
} from './pacman_buffers';
import { pacmanShader } from './pacman.wgsl';

export interface BestAgentSnapshot {
  index: number;
  score: number;
  lives: number;
  dotsLeft: number;
  level: number;
  gameOver: boolean;
  aliveCount: number;
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
  private readPending: Promise<Float32Array<ArrayBuffer>> | null = null;
  private isEvolving = false;

  private constructor(
    private readonly device: GPUDevice,
    readonly buffers: PacmanBuffers,
    readonly populationSize: number,
    seed: number,
  ) {
    this.rng = mulberry32(seed);
    this.genomes = Array.from({ length: populationSize }, () => {
      const g = new Float64Array(PACMAN_GENOME_SIZE);
      for (let k = 0; k < PACMAN_GENOME_SIZE; k++) g[k] = this.rng() * 2 - 1;
      return g;
    });
    this.uploadGenomes();
    this.device.queue.writeBuffer(this.buffers.agents, 0, initialAgentStates(populationSize));

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
      ],
    });
    this.workgroups = Math.ceil(populationSize / 64);
  }

  static init(device: GPUDevice, populationSize = 300, seed = 1): PacmanEvolution {
    return new PacmanEvolution(device, createPacmanBuffers(device, populationSize), populationSize, seed);
  }

  /** CPU-side genome of agent `index` (256 floats, uploaded to the GPU each generation). */
  genomeAt(index: number): Float64Array {
    return this.genomes[index];
  }

  private uploadGenomes(): void {
    const flat = new Float32Array(this.genomes.length * PACMAN_GENOME_SIZE);
    for (let i = 0; i < this.genomes.length; i++) flat.set(this.genomes[i], i * PACMAN_GENOME_SIZE);
    this.device.queue.writeBuffer(this.buffers.genomes, 0, flat);
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
      let aliveCount = 0;
      for (let i = 0; i < this.populationSize; i++) {
        if (states[i * AGENT_FLOATS + A.gameOver] < 0.5) aliveCount++;
      }
      if (aliveCount > 0) return;

      // Greed shaping: score x (1 + score rate) — fast eaters beat campers at equal score.
      const fitnesses = new Float64Array(this.populationSize);
      for (let i = 0; i < this.populationSize; i++) {
        const o = i * AGENT_FLOATS;
        const score = states[o + A.score];
        const ticks = Math.max(1, states[o + A.ticks]);
        fitnesses[i] = score * (1 + score / ticks);
      }
      this.genomes = nextRouletteGeneration(this.genomes, fitnesses, this.rng);
      this.generation++;
      this.uploadGenomes();
      this.device.queue.writeBuffer(this.buffers.agents, 0, initialAgentStates(this.populationSize));
    } finally {
      this.isEvolving = false;
    }
  }

  /** Snapshot of the highest-score agent still playing (falls back to best overall), for the HUD/renderer. */
  async readBestAgentState(): Promise<BestAgentSnapshot> {
    const states = await this.readStates();
    let best = 0;
    let bestAlive = -1;
    let aliveCount = 0;
    for (let i = 0; i < this.populationSize; i++) {
      const o = i * AGENT_FLOATS;
      if (states[o + A.score] > states[best * AGENT_FLOATS + A.score]) best = i;
      if (states[o + A.gameOver] < 0.5) {
        aliveCount++;
        if (bestAlive < 0 || states[o + A.score] > states[bestAlive * AGENT_FLOATS + A.score]) bestAlive = i;
      }
    }
    const shown = bestAlive >= 0 ? bestAlive : best;
    const o = shown * AGENT_FLOATS;
    return {
      index: shown,
      score: states[o + A.score],
      lives: states[o + A.lives],
      dotsLeft: states[o + A.dotsLeft],
      level: states[o + A.level],
      gameOver: states[o + A.gameOver] > 0.5,
      aliveCount,
    };
  }
}

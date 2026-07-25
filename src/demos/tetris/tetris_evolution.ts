import { nextRouletteGeneration } from '../../utils/ga';
import { mulberry32, type Rng } from '../../utils/rng';
import {
  A,
  AGENT_FLOATS,
  createTetrisBuffers,
  initialAgentStates,
  makeSequence,
  TETRIS_GENOME_SIZE,
  type TetrisBuffers,
} from './tetris_buffers';
import { tetrisShader } from './tetris.wgsl';

export interface BestTetrisSnapshot {
  index: number;
  score: number;
  lines: number;
  level: number;
  pieces: number;
  nextType: number;
  gameOver: boolean;
  aliveCount: number;
}

/** Generation driver: each agent plays one Tetris game inside the compute shader. */
export class TetrisEvolution {
  generation = 1;
  /** Total placement ticks dispatched (drives the fall animation). */
  tickCount = 0;

  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroup: GPUBindGroup;
  private readonly workgroups: number;
  private readonly rng: Rng;
  private genomes: Float64Array[];
  private readPending: Promise<Float32Array<ArrayBuffer>> | null = null;
  private isEvolving = false;

  private constructor(
    private readonly device: GPUDevice,
    readonly buffers: TetrisBuffers,
    readonly populationSize: number,
    seed: number,
  ) {
    this.rng = mulberry32(seed);
    this.genomes = Array.from({ length: populationSize }, () => {
      const g = new Float64Array(TETRIS_GENOME_SIZE);
      for (let k = 0; k < TETRIS_GENOME_SIZE; k++) g[k] = this.rng() * 2 - 1;
      return g;
    });
    this.uploadGenomes();
    this.resetGames();

    this.pipeline = device.createComputePipeline({
      label: 'tetris pipeline',
      layout: 'auto',
      compute: { module: device.createShaderModule({ label: 'tetris shader', code: tetrisShader }), entryPoint: 'main' },
    });
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.params } },
        { binding: 1, resource: { buffer: buffers.genomes } },
        { binding: 2, resource: { buffer: buffers.agents } },
        { binding: 3, resource: { buffer: buffers.sequence } },
      ],
    });
    this.workgroups = Math.ceil(populationSize / 64);
  }

  static init(device: GPUDevice, populationSize = 300, seed = 1): TetrisEvolution {
    return new TetrisEvolution(device, createTetrisBuffers(device, populationSize), populationSize, seed);
  }

  /** Fresh boards + a fresh shared piece sequence (same conditions for all agents). */
  private resetGames(): void {
    const seq = makeSequence(this.rng);
    this.device.queue.writeBuffer(this.buffers.sequence, 0, seq);
    this.device.queue.writeBuffer(this.buffers.agents, 0, initialAgentStates(this.populationSize, seq));
  }

  /** CPU-side genome of agent `index` (386 floats, uploaded to the GPU each generation). */
  genomeAt(index: number): Float64Array {
    return this.genomes[index];
  }

  private uploadGenomes(): void {
    const flat = new Float32Array(this.genomes.length * TETRIS_GENOME_SIZE);
    for (let i = 0; i < this.genomes.length; i++) flat.set(this.genomes[i], i * TETRIS_GENOME_SIZE);
    this.device.queue.writeBuffer(this.buffers.genomes, 0, flat);
  }

  /** Dispatch k placement ticks (one piece per tick per agent). */
  substeps(k: number): void {
    this.tickCount += k;
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

  /** Generation ends when every game has topped out (or hit the piece cap). */
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

      // Dense, strictly positive: survival (pieces^2) dominates early and implicitly
      // rewards tidy stacking; lines take over as play improves. No clamping —
      // clamped fitness collapses roulette onto a single arbitrary parent.
      const fitnesses = new Float64Array(this.populationSize);
      for (let i = 0; i < this.populationSize; i++) {
        const o = i * AGENT_FLOATS;
        const p = states[o + A.pieces];
        fitnesses[i] = states[o + A.lines] * 1000 + states[o + A.score] + p * p * 0.1;
      }
      this.genomes = nextRouletteGeneration(this.genomes, fitnesses, this.rng, { mutateRate: 0.03, eliteCount: 2 });
      this.generation++;
      this.uploadGenomes();
      this.resetGames();
    } finally {
      this.isEvolving = false;
    }
  }

  /** Snapshot of the highest-score agent still playing (fallback: best overall). */
  async readBestAgentState(): Promise<BestTetrisSnapshot> {
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
      lines: states[o + A.lines],
      level: states[o + A.level],
      pieces: states[o + A.pieces],
      nextType: states[o + A.nextType],
      gameOver: states[o + A.gameOver] > 0.5,
      aliveCount,
    };
  }
}

import { mulberry32, type Rng } from '../../utils/rng';
import {
  BIRD_FLOATS,
  BIRD_X,
  createFlappyBuffers,
  FLAPPY_GENOME_SIZE,
  initialBirdStates,
  PIPE_GAP,
  PIPE_SPAWN_FRAMES,
  PIPE_SPEED,
  PIPE_TOP_MAX,
  PIPE_TOP_MIN,
  PIPE_W,
  uploadFlappyPipes,
  WORLD_W,
  type FlappyBuffers,
  type PipeState,
} from './flappy_buffers';
import { flappyShader } from './flappy.wgsl';

export interface BestBirdSnapshot {
  index: number;
  x: number;
  y: number;
  velY: number;
  score: number;
  fitness: number;
  pipes: number;
  alive: boolean;
  aliveCount: number;
}

/** Box–Muller on the seeded rng, for the original's randomGaussian() mutation. */
function gaussian(rng: Rng): number {
  const u = Math.max(rng(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

/**
 * The source repo's geneticAlgorithm.js, on flat genomes: fitness is already
 * score^2 (computed in the shader); normalize, roulette-wheel pool selection,
 * child = copy + per-weight gaussian mutation with probability 0.1.
 */
export function nextFlappyGeneration(population: Float64Array[], fitnesses: ArrayLike<number>, rng: Rng): Float64Array[] {
  const n = population.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += fitnesses[i];
  const probs = new Float64Array(n);
  if (sum > 0) for (let i = 0; i < n; i++) probs[i] = fitnesses[i] / sum;

  return Array.from({ length: n }, () => {
    // Roulette-wheel selection (poolSelection).
    let r = rng();
    let idx = 0;
    while (r > 0 && idx < n) {
      r -= probs[idx];
      idx++;
    }
    const parent = population[Math.max(0, idx - 1)];
    // copy() + mutate(0.1).
    const child = parent.slice();
    for (let k = 0; k < FLAPPY_GENOME_SIZE; k++) {
      if (rng() < 0.1) child[k] += gaussian(rng);
    }
    return child;
  });
}

/**
 * Generation driver. Pipes are shared across birds, so they tick on the CPU
 * (one pipe state per frame, uploaded before each dispatch); bird physics and
 * the NN forward passes run in the compute shader.
 */
export class FlappyEvolution {
  generation = 1;
  pipesList: PipeState[] = [];
  /** Bars passed this generation (same for every bird: they never move horizontally). */
  pipesPassed = 0;

  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroup: GPUBindGroup;
  private readonly workgroups: number;
  private readonly rng: Rng;
  private genomes: Float64Array[];
  private frameCounter = 0;
  private readPending: Promise<Float32Array<ArrayBuffer>> | null = null;
  private isEvolving = false;

  private constructor(
    private readonly device: GPUDevice,
    readonly buffers: FlappyBuffers,
    readonly populationSize: number,
    seed: number,
  ) {
    this.rng = mulberry32(seed);
    this.genomes = Array.from({ length: populationSize }, () => {
      const g = new Float64Array(FLAPPY_GENOME_SIZE);
      for (let k = 0; k < FLAPPY_GENOME_SIZE; k++) g[k] = this.rng() * 2 - 1;
      return g;
    });
    this.uploadGenomes();
    this.resetBirds();

    this.pipeline = device.createComputePipeline({
      label: 'flappy pipeline',
      layout: 'auto',
      compute: { module: device.createShaderModule({ label: 'flappy shader', code: flappyShader }), entryPoint: 'main' },
    });
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.params } },
        { binding: 1, resource: { buffer: buffers.genomes } },
        { binding: 2, resource: { buffer: buffers.birds } },
        { binding: 3, resource: { buffer: buffers.pipes } },
      ],
    });
    this.workgroups = Math.ceil(populationSize / 64);
  }

  static init(device: GPUDevice, populationSize = 300, seed = 1): FlappyEvolution {
    return new FlappyEvolution(device, createFlappyBuffers(device, populationSize), populationSize, seed);
  }

  private uploadGenomes(): void {
    const flat = new Float32Array(this.genomes.length * FLAPPY_GENOME_SIZE);
    for (let i = 0; i < this.genomes.length; i++) flat.set(this.genomes[i], i * FLAPPY_GENOME_SIZE);
    this.device.queue.writeBuffer(this.buffers.genomes, 0, flat);
  }

  /** Respawn all birds and clear the pipes (resetGame in the original). */
  private resetBirds(): void {
    this.device.queue.writeBuffer(this.buffers.birds, 0, initialBirdStates(this.populationSize));
    this.frameCounter = 0;
    this.pipesList = [];
    this.pipesPassed = 0;
  }

  private spawnPipe(): void {
    const topY = PIPE_TOP_MIN + this.rng() * (PIPE_TOP_MAX - PIPE_TOP_MIN);
    this.pipesList.push({ x: WORLD_W, topY, bottomY: topY + PIPE_GAP, width: PIPE_W });
  }

  /** Dispatch k per-frame physics ticks; pipes advance one tick per frame like the original. */
  substeps(k: number): void {
    const paramsData = new ArrayBuffer(16);
    const paramsU32 = new Uint32Array(paramsData);
    paramsU32[0] = this.populationSize;

    for (let s = 0; s < k; s++) {
      for (let p = this.pipesList.length - 1; p >= 0; p--) {
        const wasAhead = this.pipesList[p].x + this.pipesList[p].width >= BIRD_X;
        this.pipesList[p].x -= PIPE_SPEED;
        if (wasAhead && this.pipesList[p].x + this.pipesList[p].width < BIRD_X) this.pipesPassed++;
        if (this.pipesList[p].x + this.pipesList[p].width < 0) this.pipesList.splice(p, 1);
      }
      if (this.frameCounter % PIPE_SPAWN_FRAMES === 0) this.spawnPipe();
      this.frameCounter++;
      if (this.pipesList.length > this.buffers.maxPipes) this.pipesList.shift();
      uploadFlappyPipes(this.device, this.buffers, this.pipesList);

      paramsU32[1] = this.pipesList.length;
      this.device.queue.writeBuffer(this.buffers.params, 0, paramsData);

      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.dispatchWorkgroups(this.workgroups);
      pass.end();
      this.device.queue.submit([encoder.finish()]);
    }
  }

  /** Copy of the bird-state buffer; concurrent calls share one readback. */
  private readStates(): Promise<Float32Array<ArrayBuffer>> {
    if (!this.readPending) {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.buffers.birds, 0, this.buffers.readback, 0, this.buffers.birds.size);
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

  /** CPU-side genome of bird `index` (57 floats, uploaded to the GPU each generation). */
  genomeAt(index: number): Float64Array {
    return this.genomes[index];
  }

  /** Generation ends when every bird is dead (pipe, ground, or ceiling). */
  async checkAndEvolve(): Promise<void> {
    if (this.isEvolving) return;
    this.isEvolving = true;
    try {
      const states = await this.readStates();
      const u32 = new Uint32Array(states.buffer);
      let aliveCount = 0;
      for (let i = 0; i < this.populationSize; i++) {
        if (u32[i * BIRD_FLOATS + 3] === 1) aliveCount++;
      }
      if (aliveCount > 0) return;

      this.genomes = nextFlappyGeneration(this.genomes, states.filter((_, idx) => idx % BIRD_FLOATS === 5), this.rng);
      this.generation++;
      this.uploadGenomes();
      this.resetBirds();
    } finally {
      this.isEvolving = false;
    }
  }

  /** Snapshot of the highest-score bird, for the HUD/renderer. */
  async readBestBirdState(): Promise<BestBirdSnapshot> {
    const states = await this.readStates();
    const u32 = new Uint32Array(states.buffer);
    let best = 0;
    let aliveCount = 0;
    for (let i = 0; i < this.populationSize; i++) {
      if (u32[i * BIRD_FLOATS + 4] > u32[best * BIRD_FLOATS + 4]) best = i;
      if (u32[i * BIRD_FLOATS + 3] === 1) aliveCount++;
    }
    const o = best * BIRD_FLOATS;
    return {
      index: best,
      x: states[o],
      y: states[o + 1],
      velY: states[o + 2],
      score: u32[o + 4],
      fitness: states[o + 5],
      pipes: this.pipesPassed,
      alive: u32[o + 3] === 1,
      aliveCount,
    };
  }
}

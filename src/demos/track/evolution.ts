import { initPopulation, mulberry32, nextGeneration, type Rng } from './ga';
import { autosaveBest } from './model';
import type { Track } from './track';
import { CAR_FLOATS, createSimBuffers, initialCarStates, uploadGenomes, type SimBuffers } from './buffers';
import { simShader } from './sim.wgsl';

function createSimPipeline(device: GPUDevice): GPUComputePipeline {
  return device.createComputePipeline({
    label: 'sim pipeline',
    layout: 'auto',
    compute: { module: device.createShaderModule({ label: 'sim shader', code: simShader }), entryPoint: 'main' },
  });
}

function createBindGroup(device: GPUDevice, pipeline: GPUComputePipeline, buffers: SimBuffers): GPUBindGroup {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.params } },
      { binding: 1, resource: { buffer: buffers.genomes } },
      { binding: 2, resource: { buffer: buffers.cars } },
      { binding: 3, resource: { buffer: buffers.walls } },
      { binding: 4, resource: { buffer: buffers.checkpoints } },
      { binding: 5, resource: { buffer: buffers.sensors } },
    ],
  });
}

export interface BestCarSnapshot {
  index: number;
  x: number;
  y: number;
  angleDeg: number;
  turn: number;
  engine: number;
  fitness: number;
  alive: boolean;
  aliveCount: number;
}

/**
 * Generation driver. Population size is fixed at init (buffers are sized for it);
 * changing it requires re-init.
 */
export class Evolution {
  generation = 1;

  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroup: GPUBindGroup;
  private readonly buffers: SimBuffers;
  private readonly workgroups: number;
  private readonly rng: Rng;
  private genomes: Float64Array[];
  private readPending: Promise<Float32Array<ArrayBuffer>> | null = null;

  private constructor(
    private readonly device: GPUDevice,
    private readonly track: Track,
    readonly populationSize: number,
    seed: number,
  ) {
    this.rng = mulberry32(seed);
    this.genomes = initPopulation(populationSize, this.rng);
    this.buffers = createSimBuffers(device, track, populationSize);
    uploadGenomes(device, this.buffers, this.genomes);
    device.queue.writeBuffer(this.buffers.cars, 0, initialCarStates(track, populationSize));
    this.pipeline = createSimPipeline(device);
    this.bindGroup = createBindGroup(device, this.pipeline, this.buffers);
    this.workgroups = Math.ceil(populationSize / 64);
  }

  static init(device: GPUDevice, track: Track, populationSize = 30, seed = 1): Evolution {
    return new Evolution(device, track, populationSize, seed);
  }

  /** GPU buffers, shared with the renderer (car states, walls, sensor distances). */
  get simBuffers(): SimBuffers {
    return this.buffers;
  }

  /** CPU-side genome of car `index` (47 floats, uploaded to the GPU each generation). */
  genomeAt(index: number): Float64Array {
    return this.genomes[index];
  }

  /** Dispatch k physics substeps. */
  substeps(k: number): void {
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    for (let i = 0; i < k; i++) pass.dispatchWorkgroups(this.workgroups);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /** Copy of the car-state buffer; concurrent calls share one readback. */
  private readStates(): Promise<Float32Array<ArrayBuffer>> {
    if (!this.readPending) {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.buffers.cars, 0, this.buffers.readback, 0, this.buffers.cars.size);
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

  private isEvolving = false;

  /** Generation ends when every car is dead (wall or checkpoint timeout). */
  async isGenerationOver(): Promise<boolean> {
    const states = await this.readStates();
    const u32 = new Uint32Array(states.buffer);
    for (let i = 0; i < this.populationSize; i++) {
      if (u32[i * CAR_FLOATS + 4] === 1) return false;
    }
    return true;
  }

  async readFitness(): Promise<Float32Array<ArrayBuffer>> {
    const states = await this.readStates();
    const fitness = new Float32Array(this.populationSize);
    for (let i = 0; i < this.populationSize; i++) fitness[i] = states[i * CAR_FLOATS + 7];
    return fitness;
  }

  /**
   * Safely check if generation is over and perform evolution step (guarded against
   * concurrent calls and duplicate frame dispatches).
   */
  async checkAndEvolve(isTest: boolean): Promise<void> {
    if (this.isEvolving) return;
    this.isEvolving = true;
    try {
      const over = await this.isGenerationOver();
      if (!over) {
        this.isEvolving = false;
        return;
      }
      if (isTest) {
        this.resetStates();
      } else {
        const fitnesses = await this.readFitness();
        let best = 0;
        for (let i = 1; i < this.populationSize; i++) if (fitnesses[i] > fitnesses[best]) best = i;
        autosaveBest(this.track.name, this.genomes[best], this.generation, fitnesses[best]);
        this.genomes = nextGeneration(this.genomes, fitnesses, this.rng);
        uploadGenomes(this.device, this.buffers, this.genomes);
        this.resetStates();
        this.generation++;
      }
    } finally {
      this.isEvolving = false;
    }
  }

  /** CPU GA step from the last fitnesses, then upload new genomes and reset states. */
  async evolve(): Promise<void> {
    const fitnesses = await this.readFitness();
    let best = 0;
    for (let i = 1; i < this.populationSize; i++) if (fitnesses[i] > fitnesses[best]) best = i;
    autosaveBest(this.track.name, this.genomes[best], this.generation, fitnesses[best]);
    this.genomes = nextGeneration(this.genomes, fitnesses, this.rng);
    uploadGenomes(this.device, this.buffers, this.genomes);
    this.resetStates();
    this.generation++;
  }

  /** Replace genome 0 (the elite slot) with `weights`, upload, and reset the sim. */
  injectBest(weights: Float64Array): void {
    this.genomes[0] = weights.slice();
    uploadGenomes(this.device, this.buffers, this.genomes);
    this.resetStates();
  }

  /** Respawn all cars at the track start with the current genomes. */
  resetStates(): void {
    this.device.queue.writeBuffer(this.buffers.cars, 0, initialCarStates(this.track, this.populationSize));
  }

  /** Snapshot of the highest-fitness car, for the HUD/renderer (Phase 3). */
  async readBestCarState(): Promise<BestCarSnapshot> {
    const states = await this.readStates();
    const aliveFlags = new Uint32Array(states.buffer);
    let best = 0;
    let aliveCount = 0;
    for (let i = 0; i < this.populationSize; i++) {
      if (states[i * CAR_FLOATS + 7] > states[best * CAR_FLOATS + 7]) best = i;
      if (aliveFlags[i * CAR_FLOATS + 4] === 1) aliveCount++;
    }
    const o = best * CAR_FLOATS;
    return {
      index: best,
      x: states[o],
      y: states[o + 1],
      angleDeg: states[o + 2],
      turn: states[o + 8],
      engine: states[o + 9],
      fitness: states[o + 7],
      alive: aliveFlags[o + 4] === 1,
      aliveCount,
    };
  }
}

/**
 * Standalone GPU population run, used by the selftest to compare against the CPU
 * reference sim. Returns per-car fitness after `steps` substeps.
 */
export async function simulatePopulationGpu(
  device: GPUDevice,
  track: Track,
  genomes: Float64Array[],
  steps: number,
): Promise<Float32Array<ArrayBuffer>> {
  const buffers = createSimBuffers(device, track, genomes.length);
  uploadGenomes(device, buffers, genomes);
  device.queue.writeBuffer(buffers.cars, 0, initialCarStates(track, genomes.length));
  const pipeline = createSimPipeline(device);
  const bindGroup = createBindGroup(device, pipeline, buffers);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  const workgroups = Math.ceil(genomes.length / 64);
  for (let i = 0; i < steps; i++) pass.dispatchWorkgroups(workgroups);
  pass.end();
  encoder.copyBufferToBuffer(buffers.cars, 0, buffers.readback, 0, buffers.cars.size);
  device.queue.submit([encoder.finish()]);

  await buffers.readback.mapAsync(GPUMapMode.READ);
  const states = new Float32Array(buffers.readback.getMappedRange().slice(0));
  buffers.readback.unmap();
  const fitness = new Float32Array(genomes.length);
  for (let i = 0; i < genomes.length; i++) fitness[i] = states[i * CAR_FLOATS + 7];
  return fitness;
}

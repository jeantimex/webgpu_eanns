import { aggregateEpisodes, type EpisodeAggregate } from '../utils/evaluation';
import { mulberry32, type Rng } from '../utils/rng';
import { nextGeneration, type GAConfig } from './ga';
import type { Network } from './network';
import { createPopulation } from './population';

/**
 * The shared generation driver, extracted from the per-demo `*_evolution.ts`
 * copies. It owns the genome store, the standard GPU buffers, dispatch, readback,
 * generation-end detection, the GA step, and best-genome tracking. The demo owns
 * the environment: one WGSL sim shader, the agent-state layout, and the fitness
 * reducer, all supplied through `Simulation`.
 *
 * GPU buffer contract (what the shader sees):
 *
 *     binding 0  params  uniform         (16 bytes by default; sim.paramsBytes + writeParams)
 *     binding 1  genomes storage read    (genomeCount * genomeSize f32)
 *     binding 2  agents  storage read_write (agentCount * agentFloats f32)
 *     binding 3+ extras                   (sim.extraBuffers, in declaration order)
 *
 * One thread per agent. With `episodes > 1` the layout follows
 * src/utils/evaluation.ts: agentIndex = genomeIndex * episodes + episodeIndex,
 * and the trailing display agent (if any) sits at populationSize * episodes.
 */

export interface ProbeResult {
  inputs?: Float32Array;
  stats?: [string, string | number][];
}

export interface Simulation {
  /** Floats per agent in the state buffer. */
  agentFloats: number;
  /** WGSL compute shader with entry point `main`. */
  shader: string;
  /** Initial contents for `count` agent slots. */
  initialStates(count: number, seed: number, episodes: number): Float32Array<ArrayBuffer>;

  /** Params uniform size in bytes (default 16, u32[0] = total agent count). */
  paramsBytes?: number;
  /** Fill the params buffer; called at init and on every evo.writeParams(). */
  writeParams?(data: ArrayBuffer, evo: Evolution): void;
  /** Demo-owned buffers bound at 3+ in declaration order; merged into evo.buffers. */
  extraBuffers?(device: GPUDevice, evo: Evolution): Record<string, GPUBuffer>;
  /**
   * Shared-CPU-world hook (dino obstacles, flappy pipes): tick the world and
   * queue uploads before this step's dispatch. When present, each substep gets
   * its own encoder/submit; otherwise k dispatches share one encoder.
   */
  beforeStep?(evo: Evolution, step: number): void;

  /** True when one agent's run has ended (its gameOver/dead flag). */
  isAgentDone(states: Float32Array, agentIndex: number): boolean;
  /** True when the generation should end (usually: every agent done). */
  isGenerationOver(states: Float32Array, evo: Evolution): boolean;
  /** Per-*agent* scores, length trainingAgents; the driver aggregates episodes. */
  fitness(states: Float32Array, evo: Evolution): Float64Array;
  /**
   * Genome index to record as the new best, or -1 for none this generation.
   * Default: argmax of the (episode-aggregated) fitnesses, recorded when it
   * beats evo.bestFitness. Override for custom rankings (pacman's level-first).
   */
  rankBest?(states: Float32Array, fitnesses: Float64Array, evo: Evolution): number;
  /** Extra meta recorded with a new best (goes to autosave and the HUD). */
  bestMeta?(states: Float32Array, bestGenomeIndex: number, fitness: number, evo: Evolution): Record<string, number>;
  /** Mid-generation upkeep on read-back states (pacman's display-agent restart). */
  maintain?(evo: Evolution, states: Float32Array): void;
  /** Runs after fitness, before the GA step (adaptive mutation via evo.setMutationRate). */
  beforeGaStep?(evo: Evolution, fitnesses: Float64Array): void;
  /** Runs after the GA step, before agents reset (env re-roll, seed rotation, writeParams). */
  onNewGeneration?(evo: Evolution): void;
  /** Seed for the training agents' initialStates each generation (default 0). */
  trainingSeed?(evo: Evolution): number;
  /** Seed for the display agent's initialStates (default: the best genome's index). */
  displaySeed?(evo: Evolution): number;
  /** Rebuild an agent's network inputs from state, for the NetworkPanel/HUD. */
  probe?(states: Float32Array, agentIndex: number): ProbeResult;
}

export interface EvolutionCallbacks {
  onGeneration?(evo: Evolution): void;
  onFitness?(evo: Evolution, fitnesses: Float64Array): void;
  onNewBest?(evo: Evolution, genome: Float64Array, fitness: number, meta: Record<string, number>): void;
}

export interface EvolutionConfig {
  populationSize?: number;
  seed?: number;
  episodes?: number;
  /** How per-episode scores collapse into genome fitness (default 'mean'). */
  episodeAggregate?: EpisodeAggregate;
  /** Trailing replay agent showing the best-so-far genome (default true). */
  displayAgent?: boolean;
  ga: GAConfig;
  callbacks?: EvolutionCallbacks;
}

export interface CoreBuffers {
  params: GPUBuffer;
  genomes: GPUBuffer;
  agents: GPUBuffer;
  readback: GPUBuffer;
  [extra: string]: GPUBuffer;
}

export class Evolution {
  generation = 1;
  readonly populationSize: number;
  readonly episodes: number;
  /** populationSize * episodes. */
  readonly trainingAgents: number;
  /** Trailing replay slot; equals trainingAgents when the display agent exists. */
  readonly displayAgentIndex: number;
  readonly buffers: CoreBuffers;

  bestFitness = -Infinity;
  bestGeneration = 1;
  bestGenome: Float64Array | null = null;
  bestMetaValue: Record<string, number> = {};
  /** Agent index the last best genome was found at (default display replay seed). */
  lastBestIndex = 0;

  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroup: GPUBindGroup;
  private readonly workgroups: number;
  private readonly rng: Rng;
  private readonly displayGenomeIndex: number;
  private readonly hasDisplay: boolean;
  private genomes: Float64Array[];
  private mutateRateOverride: number | null = null;
  private readPending: Promise<Float32Array> | null = null;
  private isEvolving = false;

  private constructor(
    readonly device: GPUDevice,
    readonly net: Network,
    readonly sim: Simulation,
    private readonly config: Required<Omit<EvolutionConfig, 'callbacks'>> & Pick<EvolutionConfig, 'callbacks'>,
  ) {
    this.populationSize = config.populationSize;
    this.episodes = config.episodes;
    this.trainingAgents = this.populationSize * this.episodes;
    this.hasDisplay = config.displayAgent;
    this.displayAgentIndex = this.trainingAgents;
    this.displayGenomeIndex = this.populationSize;
    const agentCount = this.trainingAgents + (this.hasDisplay ? 1 : 0);
    const genomeCount = this.populationSize + (this.hasDisplay ? 1 : 0);

    this.rng = mulberry32(config.seed);
    this.genomes = createPopulation(net, genomeCount, { seed: config.seed });
    if (this.hasDisplay) this.genomes[this.displayGenomeIndex] = this.genomes[0].slice();

    // Standard buffers.
    const paramsBytes = sim.paramsBytes ?? 16;
    const params = device.createBuffer({ label: 'params', size: paramsBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const genomes = device.createBuffer({
      label: 'genomes',
      size: genomeCount * net.genomeSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const agentBytes = agentCount * sim.agentFloats * 4;
    const agents = device.createBuffer({
      label: 'agents',
      size: agentBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const readback = device.createBuffer({ label: 'readback', size: agentBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const extras = sim.extraBuffers?.(device, this) ?? {};
    this.buffers = { params, genomes, agents, readback, ...extras };

    if (sim.writeParams) this.writeParams();
    else {
      const data = new ArrayBuffer(paramsBytes);
      new Uint32Array(data)[0] = agentCount;
      device.queue.writeBuffer(params, 0, data);
    }

    this.uploadGenomes();
    this.resetAgents();

    this.pipeline = device.createComputePipeline({
      label: 'evolution pipeline',
      layout: 'auto',
      compute: { module: device.createShaderModule({ label: 'sim shader', code: sim.shader }), entryPoint: 'main' },
    });
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: params } },
      { binding: 1, resource: { buffer: genomes } },
      { binding: 2, resource: { buffer: agents } },
      ...Object.values(extras).map((buffer, i) => ({ binding: 3 + i, resource: { buffer } })),
    ];
    this.bindGroup = device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries });
    this.workgroups = Math.ceil(agentCount / 64);
  }

  static init(device: GPUDevice, net: Network, sim: Simulation, config: EvolutionConfig): Evolution {
    return new Evolution(device, net, sim, {
      populationSize: config.populationSize ?? 300,
      seed: config.seed ?? 1,
      episodes: Math.max(1, Math.floor(config.episodes ?? 1)),
      episodeAggregate: config.episodeAggregate ?? 'mean',
      displayAgent: config.displayAgent ?? true,
      ga: config.ga,
      callbacks: config.callbacks,
    });
  }

  get rngForHooks(): Rng {
    return this.rng;
  }

  /** Genome driving a given *agent* slot (episodes of one genome are adjacent). */
  genomeAt(agentIndex: number): Float64Array {
    return this.genomes[Math.min(this.populationSize, Math.floor(agentIndex / this.episodes))];
  }

  displayGenome(): Float64Array | null {
    return this.hasDisplay ? this.genomes[this.displayGenomeIndex] : null;
  }

  /** Load a genome into slot 0 and the display slot, then reset the sim. */
  injectBest(genome: Float64Array, meta?: Record<string, number>): void {
    if (genome.length !== this.net.genomeSize) {
      throw new Error(`Injected genome has ${genome.length} weights, expected ${this.net.genomeSize}.`);
    }
    this.genomes[0] = genome.slice();
    if (this.hasDisplay) this.genomes[this.displayGenomeIndex] = genome.slice();
    this.bestGenome = genome.slice();
    // Without a recorded eval, a hand-loaded model must not be overwritten by
    // the next generation's autosave — hence Infinity, matching the old demos.
    this.bestFitness = meta?.eval ?? Infinity;
    this.bestGeneration = this.generation;
    this.bestMetaValue = meta ?? { generation: this.generation, eval: this.bestFitness };
    this.uploadGenomes();
    this.resetAgents();
  }

  /** Adaptive-mutation knob; the GA step uses this instead of ga.mutateRate when set. */
  setMutationRate(rate: number): void {
    this.mutateRateOverride = rate;
  }

  /** Re-run the sim's writeParams and upload (e.g. after a runtime knob changed). */
  writeParams(): void {
    const data = new ArrayBuffer(this.buffers.params.size);
    this.sim.writeParams?.(data, this);
    this.device.queue.writeBuffer(this.buffers.params, 0, data);
  }

  /** Dispatch k sim steps. */
  substeps(k: number): void {
    if (this.sim.beforeStep) {
      for (let s = 0; s < k; s++) {
        this.sim.beforeStep(this, s);
        this.dispatch(1);
      }
      return;
    }
    this.dispatch(k);
  }

  private dispatch(k: number): void {
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    for (let i = 0; i < k; i++) pass.dispatchWorkgroups(this.workgroups);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /** Copy of the agent-state buffer; concurrent calls share one readback. */
  readStates(): Promise<Float32Array> {
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

  /** Live agents among the training population. */
  countAlive(states: Float32Array): number {
    let alive = 0;
    for (let i = 0; i < this.trainingAgents; i++) {
      if (!this.sim.isAgentDone(states, i)) alive++;
    }
    return alive;
  }

  /** Reset training agents, and the display agent if there is one. */
  resetAgents(): void {
    const trainSeed = this.sim.trainingSeed?.(this) ?? 0;
    this.device.queue.writeBuffer(
      this.buffers.agents,
      0,
      this.sim.initialStates(this.trainingAgents, trainSeed, this.episodes),
    );
    if (this.hasDisplay) this.resetDisplayAgent();
  }

  resetDisplayAgent(): void {
    if (!this.hasDisplay) return;
    const seed = this.sim.displaySeed?.(this) ?? this.lastBestIndex;
    this.resetAgent(this.displayAgentIndex, seed);
  }

  /** Reset a single agent slot (e.g. replaying a test model in slot 0). */
  resetAgent(index: number, seed: number): void {
    this.device.queue.writeBuffer(
      this.buffers.agents,
      index * this.sim.agentFloats * 4,
      this.sim.initialStates(1, seed, this.episodes),
    );
  }

  /** Restart one agent when its run has ended. */
  async restartAgentIfDone(index: number, seed: number): Promise<void> {
    const states = await this.readStates();
    if (this.sim.isAgentDone(states, index)) this.resetAgent(index, seed);
  }

  /** Test-mode behavior: replay the loaded/best model again after it dies. */
  async restartDisplayIfDead(): Promise<void> {
    if (!this.hasDisplay) return;
    const seed = this.sim.displaySeed?.(this) ?? this.lastBestIndex;
    await this.restartAgentIfDone(this.displayAgentIndex, seed);
  }

  /** Test mode without a display agent: reset the whole sim once the run is over. */
  async resetIfOver(): Promise<void> {
    const states = await this.readStates();
    if (this.sim.isGenerationOver(states, this)) this.resetAgents();
  }

  /** Generation end → fitness → best tracking → GA step → reset. */
  async checkAndEvolve(): Promise<void> {
    if (this.isEvolving) return;
    this.isEvolving = true;
    try {
      const states = await this.readStates();
      this.sim.maintain?.(this, states);
      if (!this.sim.isGenerationOver(states, this)) return;

      const scores = this.sim.fitness(states, this);
      const fitnesses =
        this.episodes > 1
          ? aggregateEpisodes(scores, this.populationSize, this.episodes, this.config.episodeAggregate)
          : scores;
      this.lastScores = scores;
      this.lastFitnesses = fitnesses;
      this.config.callbacks?.onFitness?.(this, fitnesses);

      const bestIndex = this.sim.rankBest
        ? this.sim.rankBest(states, fitnesses, this)
        : this.argmax(fitnesses);
      const isNewBest = this.sim.rankBest
        ? bestIndex >= 0
        : bestIndex >= 0 && fitnesses[bestIndex] > this.bestFitness;
      if (isNewBest) {
        this.lastBestIndex = bestIndex;
        this.bestFitness = fitnesses[bestIndex];
        this.bestGeneration = this.generation;
        this.bestGenome = this.genomes[bestIndex].slice();
        if (this.hasDisplay) this.genomes[this.displayGenomeIndex] = this.bestGenome.slice();
        this.bestMetaValue = {
          generation: this.generation,
          eval: fitnesses[bestIndex],
          ...(this.sim.bestMeta?.(states, bestIndex, fitnesses[bestIndex], this) ?? {}),
        };
        this.config.callbacks?.onNewBest?.(this, this.bestGenome, fitnesses[bestIndex], this.bestMetaValue);
      }

      this.sim.beforeGaStep?.(this, fitnesses);
      const ga = this.mutateRateOverride === null ? this.config.ga : { ...this.config.ga, mutateRate: this.mutateRateOverride };
      const next = nextGeneration(this.genomes.slice(0, this.populationSize), fitnesses, this.rng, this.net, ga);
      this.genomes.splice(0, this.populationSize, ...next);
      this.generation++;
      this.sim.onNewGeneration?.(this);
      this.uploadGenomes();
      this.resetAgents();
      this.config.callbacks?.onGeneration?.(this);
    } finally {
      this.isEvolving = false;
    }
  }

  lastScores: Float64Array | null = null;
  lastFitnesses: Float64Array | null = null;

  private argmax(fitnesses: Float64Array): number {
    let best = 0;
    for (let i = 1; i < fitnesses.length; i++) if (fitnesses[i] > fitnesses[best]) best = i;
    return best;
  }

  private uploadGenomes(): void {
    const flat = new Float32Array(this.genomes.length * this.net.genomeSize);
    for (let i = 0; i < this.genomes.length; i++) flat.set(this.genomes[i], i * this.net.genomeSize);
    this.device.queue.writeBuffer(this.buffers.genomes, 0, flat);
  }
}

import { nextRouletteGeneration } from '../../utils/ga';
import { mulberry32, type Rng } from '../../utils/rng';
import {
  BASE_SPEED,
  createDinoBuffers,
  DINO_FLOATS,
  DINO_GENOME_SIZE,
  initialDinoStates,
  MAX_SPEED,
  WORLD_W,
  type DinoBuffers,
  type ObstacleState,
} from './dino_buffers';
import { dinoShader } from './dino.wgsl';

export interface BestDinoSnapshot {
  index: number;
  y: number;
  velY: number;
  score: number;
  fitness: number;
  alive: boolean;
  aliveCount: number;
}

/** Small cactus: 34x70 at y=230, variants at sheet x=446/548. Big: 49x100 at y=201, x=652/802. */
function makeObstacle(small: boolean, rng: Rng): ObstacleState {
  const multi = Math.floor(rng() * 3) + 1;
  const baseW = small ? 34 : 49;
  return {
    x: 0,
    y: small ? 230 : 201,
    w: baseW * multi,
    h: small ? 70 : 100,
    picX: small ? 446 + Math.floor(rng() * 2) * 102 : 652 + Math.floor(rng() * 2) * 150,
    scroll: small ? -100 : -200,
    baseW,
    small,
    multi,
  };
}

/**
 * Generation driver. The world (one alternating obstacle, speed ramp, ground
 * scroll, run animation) is shared across dinos, so it ticks on the CPU; dino
 * physics and the NN forward passes run in the compute shader.
 */
export class DinoEvolution {
  generation = 1;
  obstacle: ObstacleState;
  /** Obstacles cleared this generation (same for every dino: shared world). */
  cleared = 0;
  /** Original-style score ticks (one per ~7 frames); drives the speed ramp. */
  tickScore = 0;
  gamespeed = BASE_SPEED;
  groundScroll = 0;
  /** Sprite sheet x of the current run frame (1514/1602); jump frame is 1338. */
  runFrame = 1514;

  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroup: GPUBindGroup;
  private readonly workgroups: number;
  private readonly rng: Rng;
  private genomes: Float64Array[];
  private scoreInterval = 0;
  private frameInterval = 0;
  private readPending: Promise<Float32Array<ArrayBuffer>> | null = null;
  private isEvolving = false;

  private constructor(
    private readonly device: GPUDevice,
    readonly buffers: DinoBuffers,
    readonly populationSize: number,
    seed: number,
  ) {
    this.rng = mulberry32(seed);
    this.genomes = Array.from({ length: populationSize }, () => {
      const g = new Float64Array(DINO_GENOME_SIZE);
      for (let k = 0; k < DINO_GENOME_SIZE; k++) g[k] = this.rng() * 2 - 1;
      return g;
    });
    this.obstacle = makeObstacle(true, this.rng);
    this.uploadGenomes();
    this.resetDinos();

    this.pipeline = device.createComputePipeline({
      label: 'dino pipeline',
      layout: 'auto',
      compute: { module: device.createShaderModule({ label: 'dino shader', code: dinoShader }), entryPoint: 'main' },
    });
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.params } },
        { binding: 1, resource: { buffer: buffers.genomes } },
        { binding: 2, resource: { buffer: buffers.dinos } },
        { binding: 3, resource: { buffer: buffers.obstacle } },
      ],
    });
    this.workgroups = Math.ceil(populationSize / 64);
  }

  static init(device: GPUDevice, populationSize = 300, seed = 1): DinoEvolution {
    return new DinoEvolution(device, createDinoBuffers(device, populationSize), populationSize, seed);
  }

  /** CPU-side genome of dino `index` (65 floats, uploaded to the GPU each generation). */
  genomeAt(index: number): Float64Array {
    return this.genomes[index];
  }

  private uploadGenomes(): void {
    const flat = new Float32Array(this.genomes.length * DINO_GENOME_SIZE);
    for (let i = 0; i < this.genomes.length; i++) flat.set(this.genomes[i], i * DINO_GENOME_SIZE);
    this.device.queue.writeBuffer(this.buffers.genomes, 0, flat);
  }

  /** Respawn all dinos and restart the world (gameover() in the original). */
  private resetDinos(): void {
    this.device.queue.writeBuffer(this.buffers.dinos, 0, initialDinoStates(this.populationSize));
    this.obstacle = makeObstacle(true, this.rng);
    this.cleared = 0;
    this.tickScore = 0;
    this.gamespeed = BASE_SPEED;
    this.scoreInterval = 0;
  }

  /** Dispatch k per-frame physics ticks; the shared world advances one tick per frame. */
  substeps(k: number): void {
    const paramsData = new ArrayBuffer(16);
    const paramsU32 = new Uint32Array(paramsData);
    paramsU32[0] = this.populationSize;
    const paramsF32 = new Float32Array(paramsData);

    for (let s = 0; s < k; s++) {
      // Speed ramp: 7 + score/100, capped at 17 (score ticks every ~7 frames).
      this.scoreInterval++;
      if (this.scoreInterval > 6) {
        this.tickScore++;
        this.scoreInterval = 0;
      }
      this.gamespeed = Math.min(MAX_SPEED, BASE_SPEED + this.tickScore / 100);

      // Run animation toggles every 5 frames.
      this.frameInterval++;
      if (this.frameInterval > 5) {
        this.runFrame = this.runFrame === 1514 ? 1602 : 1514;
        this.frameInterval = 0;
      }

      // Obstacle advance + recycle, alternating small/big like the original.
      const obs = this.obstacle;
      obs.scroll += this.gamespeed;
      this.groundScroll += this.gamespeed;
      if (obs.scroll > WORLD_W + obs.w * 3) {
        this.cleared++;
        this.obstacle = makeObstacle(!obs.small, this.rng);
      }
      this.obstacle.x = WORLD_W - this.obstacle.scroll;

      const cur = this.obstacle;
      this.device.queue.writeBuffer(this.buffers.obstacle, 0, new Float32Array([cur.x, cur.y, cur.w, cur.h]));
      paramsF32[1] = this.gamespeed;
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

  /** Copy of the dino-state buffer; concurrent calls share one readback. */
  private readStates(): Promise<Float32Array<ArrayBuffer>> {
    if (!this.readPending) {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.buffers.dinos, 0, this.buffers.readback, 0, this.buffers.dinos.size);
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

  /** Generation ends when every dino is dead (hit a cactus). */
  async checkAndEvolve(): Promise<void> {
    if (this.isEvolving) return;
    this.isEvolving = true;
    try {
      const states = await this.readStates();
      const u32 = new Uint32Array(states.buffer);
      let aliveCount = 0;
      for (let i = 0; i < this.populationSize; i++) {
        if (u32[i * DINO_FLOATS + 2] === 1) aliveCount++;
      }
      if (aliveCount > 0) return;

      this.genomes = nextRouletteGeneration(this.genomes, states.filter((_, idx) => idx % DINO_FLOATS === 4), this.rng);
      this.generation++;
      this.uploadGenomes();
      this.resetDinos();
    } finally {
      this.isEvolving = false;
    }
  }

  /** Snapshot of the highest-score dino, for the HUD/renderer. */
  async readBestDinoState(): Promise<BestDinoSnapshot> {
    const states = await this.readStates();
    const u32 = new Uint32Array(states.buffer);
    let best = 0;
    let aliveCount = 0;
    for (let i = 0; i < this.populationSize; i++) {
      if (u32[i * DINO_FLOATS + 3] > u32[best * DINO_FLOATS + 3]) best = i;
      if (u32[i * DINO_FLOATS + 2] === 1) aliveCount++;
    }
    const o = best * DINO_FLOATS;
    return {
      index: best,
      y: states[o],
      velY: states[o + 1],
      score: u32[o + 3],
      fitness: states[o + 4],
      alive: u32[o + 2] === 1,
      aliveCount,
    };
  }
}

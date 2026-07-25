import { nextGeneration } from '../../ai/ga';
import {
  BIRD_FLOATS,
  createFlappyBuffers,
  FLAPPY_GENOME_SIZE,
  initialBirdStates,
  uploadFlappyPipes,
  type FlappyBuffers,
  type PipeState,
} from './flappy_buffers';
import { flappyShader } from './flappy.wgsl';

export interface BestBirdSnapshot {
  index: number;
  x: number;
  y: number;
  score: number;
  fitness: number;
  alive: boolean;
}

export class FlappyEvolution {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroup: GPUBindGroup;
  generation = 1;
  population: Float64Array[];
  pipesList: PipeState[] = [];
  frameCounter = 0;

  private readbackPending = false;
  private isEvolving = false;

  private constructor(
    private readonly device: GPUDevice,
    readonly buffers: FlappyBuffers,
    populationSize: number,
  ) {
    // Generate initial random genomes [57 floats per bird]
    this.population = new Array(populationSize);
    for (let i = 0; i < populationSize; i++) {
      const g = new Float64Array(FLAPPY_GENOME_SIZE);
      for (let k = 0; k < FLAPPY_GENOME_SIZE; k++) {
        g[k] = (Math.random() * 2 - 1) * 0.5;
      }
      this.population[i] = g;
    }
    this.uploadGenomes();
    this.resetBirds();

    const module = device.createShaderModule({ code: flappyShader });
    this.pipeline = device.createComputePipeline({
      label: 'flappy pipeline',
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
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
  }

  static init(device: GPUDevice, populationSize: number): FlappyEvolution {
    const buffers = createFlappyBuffers(device, populationSize, 16);
    return new FlappyEvolution(device, buffers, populationSize);
  }

  uploadGenomes(): void {
    const flat = new Float32Array(this.population.length * FLAPPY_GENOME_SIZE);
    for (let i = 0; i < this.population.length; i++) {
      flat.set(this.population[i], i * FLAPPY_GENOME_SIZE);
    }
    this.device.queue.writeBuffer(this.buffers.genomes, 0, flat);
  }

  resetBirds(): void {
    const initial = initialBirdStates(this.population.length);
    this.device.queue.writeBuffer(this.buffers.birds, 0, initial.buffer);
    this.frameCounter = 0;
    this.pipesList = [];
    this.spawnPipe();
  }

  spawnPipe(): void {
    const gapHeight = 105;
    const playableHeight = 340;
    const minTop = 40;
    const maxTop = playableHeight - gapHeight - 40;
    const topY = minTop + Math.random() * (maxTop - minTop);
    const bottomY = topY + gapHeight;
    this.pipesList.push({
      x: 600,
      topY,
      bottomY,
      width: 52,
    });
  }

  updatePipes(): void {
    // Move pipes left at 2.5px/frame
    for (let i = this.pipesList.length - 1; i >= 0; i--) {
      this.pipesList[i].x -= 2.5;
      if (this.pipesList[i].x + this.pipesList[i].width < 0) {
        this.pipesList.splice(i, 1);
      }
    }

    // Spawn new pipe every 65 frames
    if (this.frameCounter % 65 === 0) {
      this.spawnPipe();
    }

    uploadFlappyPipes(this.device, this.buffers, this.pipesList);
  }

  substeps(n: number): void {
    for (let s = 0; s < n; s++) {
      this.frameCounter++;
      this.updatePipes();

      // Update uniform buffer params
      const paramsData = new ArrayBuffer(16);
      const paramsU32 = new Uint32Array(paramsData);
      paramsU32[0] = this.population.length;
      paramsU32[1] = this.pipesList.length;
      new Float32Array(paramsData)[2] = 1 / 60;
      paramsU32[3] = this.frameCounter;
      this.device.queue.writeBuffer(this.buffers.params, 0, paramsData);

      const encoder = this.device.createCommandEncoder({ label: 'flappy step' });
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.bindGroup);
      const workgroups = Math.ceil(this.population.length / 64);
      pass.dispatchWorkgroups(workgroups);
      pass.end();
      this.device.queue.submit([encoder.finish()]);
    }
  }

  async checkAndEvolve(): Promise<void> {
    if (this.isEvolving) return;
    this.isEvolving = true;

    try {
      const readData = await this.readBirdBuffer();
      const u32 = new Uint32Array(readData.buffer);

      let aliveCount = 0;
      for (let i = 0; i < this.population.length; i++) {
        if (u32[i * BIRD_FLOATS + 3] === 1) aliveCount++;
      }

      if (aliveCount === 0) {
        // All birds died -> Evolve next generation!
        const fitnesses = new Array<number>(this.population.length);
        for (let i = 0; i < this.population.length; i++) {
          fitnesses[i] = readData[i * BIRD_FLOATS + 5];
        }

        this.population = nextGeneration(this.population, fitnesses, Math.random);
        this.generation++;
        this.uploadGenomes();
        this.resetBirds();
      }
    } finally {
      this.isEvolving = false;
    }
  }

  async readBestBirdState(): Promise<BestBirdSnapshot> {
    const readData = await this.readBirdBuffer();
    const u32 = new Uint32Array(readData.buffer);

    let bestIdx = 0;
    let bestScore = -1;
    for (let i = 0; i < this.population.length; i++) {
      const score = u32[i * BIRD_FLOATS + 4];
      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    }

    const o = bestIdx * BIRD_FLOATS;
    return {
      index: bestIdx,
      x: readData[o],
      y: readData[o + 1],
      score: u32[o + 4],
      fitness: readData[o + 5],
      alive: u32[o + 3] === 1,
    };
  }

  private async readBirdBuffer(): Promise<Float32Array> {
    if (this.readbackPending) {
      return new Float32Array(this.population.length * BIRD_FLOATS);
    }
    this.readbackPending = true;
    const encoder = this.device.createCommandEncoder({ label: 'flappy readback' });
    encoder.copyBufferToBuffer(this.buffers.birds, 0, this.buffers.readback, 0, this.buffers.birds.size);
    this.device.queue.submit([encoder.finish()]);

    await this.buffers.readback.mapAsync(GPUMapMode.READ);
    const mapped = new Float32Array(this.buffers.readback.getMappedRange().slice(0));
    this.buffers.readback.unmap();
    this.readbackPending = false;
    return mapped;
  }
}

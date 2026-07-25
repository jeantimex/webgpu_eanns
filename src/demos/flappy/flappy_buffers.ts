import { createBufferWithData } from '../../webgpu/utils';

export const FLAPPY_GENOME_SIZE = 57; // [5 -> 8 -> 1] (5x8 + 8 + 8x1 + 1 = 57)
export const BIRD_FLOATS = 8; // pos(2), velY(1), alive(1 u32), score(1 u32), fitness(1), jumpOutput(1), pad0(1) = 32 bytes

export interface PipeState {
  x: number;
  topY: number;
  bottomY: number;
  width: number;
}

export interface FlappyBuffers {
  params: GPUBuffer;
  genomes: GPUBuffer;
  birds: GPUBuffer;
  pipes: GPUBuffer;
  readback: GPUBuffer;
  maxPipes: number;
}

export function initialBirdStates(count: number): Float32Array {
  const states = new Float32Array(count * BIRD_FLOATS);
  for (let i = 0; i < count; i++) {
    const o = i * BIRD_FLOATS;
    states[o] = 50.0;     // x
    states[o + 1] = 170.0; // y (halfway down playable height 340)
    states[o + 2] = 0.0;   // velY
    new Uint32Array(states.buffer)[o + 3] = 1; // alive (u32)
    new Uint32Array(states.buffer)[o + 4] = 0; // score (u32)
    states[o + 5] = 0.0;   // fitness
    states[o + 6] = 0.0;   // jumpOutput
    states[o + 7] = 0.0;   // pad0
  }
  return states;
}

export function createFlappyBuffers(device: GPUDevice, populationSize: number, maxPipes = 16): FlappyBuffers {
  const paramsData = new ArrayBuffer(16);
  const paramsU32 = new Uint32Array(paramsData);
  paramsU32[0] = populationSize;
  paramsU32[1] = 0; // pipeCount
  new Float32Array(paramsData)[2] = 1 / 60; // dt
  paramsU32[3] = 0; // frameCounter

  const genomes = device.createBuffer({
    label: 'flappy genomes',
    size: populationSize * FLAPPY_GENOME_SIZE * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const birdBytes = populationSize * BIRD_FLOATS * 4;
  const birds = device.createBuffer({
    label: 'flappy birds',
    size: birdBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  const readback = device.createBuffer({
    label: 'flappy readback',
    size: birdBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // Pipe struct: x(f32), topY(f32), bottomY(f32), width(f32) = 16 bytes per pipe
  const pipes = device.createBuffer({
    label: 'flappy pipes',
    size: Math.max(1, maxPipes) * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  return {
    params: createBufferWithData(device, 'flappy params', paramsU32, GPUBufferUsage.UNIFORM),
    genomes,
    birds,
    pipes,
    readback,
    maxPipes,
  };
}

export function uploadFlappyPipes(device: GPUDevice, buffers: FlappyBuffers, pipesList: PipeState[]): void {
  const flat = new Float32Array(pipesList.length * 4);
  for (let i = 0; i < pipesList.length; i++) {
    flat[i * 4] = pipesList[i].x;
    flat[i * 4 + 1] = pipesList[i].topY;
    flat[i * 4 + 2] = pipesList[i].bottomY;
    flat[i * 4 + 3] = pipesList[i].width;
  }
  if (pipesList.length > 0) {
    device.queue.writeBuffer(buffers.pipes, 0, flat);
  }
}

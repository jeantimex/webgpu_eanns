import { GENOME_SIZE } from './network';
import { buildCheckpointTable, checkpointsFlat, wallsFlat, type Track } from './track';
import { createBufferWithData } from '../../webgpu/utils';

/**
 * CarState layout, 12 f32 = 48 bytes per car (matches the WGSL struct):
 * [0..1] pos, [2] angle (deg), [3] vel, [4] alive (u32), [5] cpIndex (u32),
 * [6] timeSinceCp, [7] fitness, [8..9] outputs, [10..11] pad.
 */
export const CAR_FLOATS = 12;

export interface SimBuffers {
  params: GPUBuffer;
  genomes: GPUBuffer;
  cars: GPUBuffer;
  walls: GPUBuffer;
  checkpoints: GPUBuffer;
  /** 5 raw sensor distances per car, written by the sim, read by the renderer. */
  sensors: GPUBuffer;
  /** MAP_READ target for CPU readback of the cars buffer. */
  readback: GPUBuffer;
  wallCount: number;
  cpCount: number;
}

/** All cars at the track start, alive, targeting checkpoint 0 (the start line — the spawn may lie before it). */
export function initialCarStates(track: Track, count: number): Float32Array<ArrayBuffer> {
  const states = new Float32Array(count * CAR_FLOATS);
  for (let i = 0; i < count; i++) {
    const o = i * CAR_FLOATS;
    states[o] = track.start.x;
    states[o + 1] = track.start.y;
    states[o + 2] = track.start.angleDeg;
    // alive and cpIndex are u32 in the WGSL struct: write the bits, not float 1.0.
    new Uint32Array(states.buffer)[o + 4] = 1;
    new Uint32Array(states.buffer)[o + 5] = 0;
  }
  return states;
}

export function createSimBuffers(device: GPUDevice, track: Track, populationSize: number): SimBuffers {
  // SimParams uniform: carCount, wallCount, cpCount (u32) + dt (f32) = 16 bytes.
  const paramsData = new ArrayBuffer(16);
  const paramsU32 = new Uint32Array(paramsData);
  paramsU32[0] = populationSize;
  paramsU32[1] = track.walls.length;
  paramsU32[2] = track.checkpoints.length;
  new Float32Array(paramsData)[3] = 1 / 50;

  const genomes = device.createBuffer({
    label: 'genomes',
    size: populationSize * GENOME_SIZE * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const carBytes = populationSize * CAR_FLOATS * 4;
  const cars = device.createBuffer({
    label: 'car states',
    size: carBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    label: 'car states readback',
    size: carBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  return {
    params: createBufferWithData(device, 'sim params', paramsU32, GPUBufferUsage.UNIFORM),
    genomes,
    cars,
    walls: createBufferWithData(device, 'walls', wallsFlat(track), GPUBufferUsage.STORAGE),
    checkpoints: createBufferWithData(
      device,
      'checkpoints',
      checkpointsFlat(track, buildCheckpointTable(track)),
      GPUBufferUsage.STORAGE,
    ),
    readback,
    sensors: device.createBuffer({
      label: 'sensor distances',
      size: populationSize * 5 * 4,
      usage: GPUBufferUsage.STORAGE,
    }),
    wallCount: track.walls.length,
    cpCount: track.checkpoints.length,
  };
}

/** Flatten f64 genomes into the f32 genome buffer. */
export function uploadGenomes(device: GPUDevice, buffers: SimBuffers, genomes: Float64Array[]): void {
  const flat = new Float32Array(genomes.length * GENOME_SIZE);
  for (let i = 0; i < genomes.length; i++) flat.set(genomes[i], i * GENOME_SIZE);
  device.queue.writeBuffer(buffers.genomes, 0, flat);
}

import { parseModelText } from '../../core/modelStore';
import { nextRemainderBlendGeneration } from '../../utils/ga';
import { mulberry32, type Rng } from '../../utils/rng';
import { parseUnityGenotype } from './model';
import { GENOME_SIZE, TOPOLOGY, forward, trackNetwork } from './network';
import { CAR_FLOATS, createSimBuffers, initialCarStates, uploadGenomes } from './buffers';
import { simShader } from './sim.wgsl';
import { initCarState, simulatePopulation, stepCar } from './car';
import { buildCheckpointTable, wallsFlat, type Track } from './track';

/** Uniform in [-1, 1), GeneticAlgorithm.DefInitParamMin/Max. */
function initPopulation(n: number, rng: Rng): Float64Array[] {
  return Array.from({ length: n }, () => {
    const genome = new Float64Array(GENOME_SIZE);
    for (let i = 0; i < GENOME_SIZE; i++) genome[i] = rng() * 2 - 1;
    return genome;
  });
}

/**
 * Standalone GPU population run, used by the selftest to compare against the CPU
 * reference sim. Returns per-car fitness after `steps` substeps.
 */
async function simulatePopulationGpu(
  device: GPUDevice,
  track: Track,
  genomes: Float64Array[],
  steps: number,
): Promise<Float32Array<ArrayBuffer>> {
  const buffers = createSimBuffers(device, track, genomes.length);
  uploadGenomes(device, buffers, genomes);
  device.queue.writeBuffer(buffers.cars, 0, initialCarStates(track, genomes.length));
  const pipeline = device.createComputePipeline({
    label: 'sim pipeline',
    layout: 'auto',
    compute: { module: device.createShaderModule({ label: 'sim shader', code: simShader }), entryPoint: 'main' },
  });
  const bindGroup = device.createBindGroup({
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

export interface SelftestResult {
  pass: boolean;
  failures: string[];
}

function approx(actual: number, expected: number, eps: number): boolean {
  return Math.abs(actual - expected) <= eps;
}

/** (a) Hand-computed forward passes on known weight sets. */
function testForward(): string[] {
  const failures: string[] = [];
  const inputs = [10, 10, 10, 10, 10];

  // All-zero genome -> everything stays 0.
  const zero = forward(new Float64Array(GENOME_SIZE), inputs);
  if (!approx(zero[0], 0, 1e-12) || !approx(zero[1], 0, 1e-12)) failures.push(`forward(zero) = [${zero}]`);

  // Only the last layer's bias row set (offset 24+15+3*2=45): outputs softsign(±1) = ±0.5.
  const biasOnly = new Float64Array(GENOME_SIZE);
  biasOnly[45] = 1;
  biasOnly[46] = -1;
  const biasOut = forward(biasOnly, [1, 2, 3, 4, 5]);
  if (!approx(biasOut[0], 0.5, 1e-12) || !approx(biasOut[1], -0.5, 1e-12))
    failures.push(`forward(biasOnly) = [${biasOut}]`);

  // Chain [0,0] weights of each layer: 10 -> 10/11 -> 10/21 -> 10/31 on output 0.
  const chain = new Float64Array(GENOME_SIZE);
  chain[0] = 1; // layer 0, weight[0,0]
  chain[24] = 1; // layer 1, weight[0,0]
  chain[39] = 1; // layer 2, weight[0,0]
  const chainOut = forward(chain, inputs);
  if (!approx(chainOut[0], 10 / 31, 1e-12) || !approx(chainOut[1], 0, 1e-12))
    failures.push(`forward(chain) = [${chainOut}], expected [${10 / 31}, 0]`);

  return failures;
}

/** (b) GA determinism with a fixed seed + elite preserved unmodified. */
function testGa(): string[] {
  const failures: string[] = [];
  const fitnesses = [0.1, 0.9, 0.5, 0.3, 0.7, 0.2];
  const run = () => nextRemainderBlendGeneration(initPopulation(6, mulberry32(42)), fitnesses, mulberry32(7));
  const a = run();
  const b = run();
  for (let i = 0; i < 6; i++) {
    for (let k = 0; k < GENOME_SIZE; k++) {
      if (a[i][k] !== b[i][k]) failures.push(`GA not deterministic at genome ${i} param ${k}`);
    }
  }
  // The champion (index 1 by fitness) leads the elite slots unmodified, and every
  // elite slot is an unmodified clone of some parent — the pool can hold several
  // copies of one genome (relFitness >= 1 floors to >1 copy), so slot 1 is not
  // guaranteed to be the second-best parent.
  const pop = initPopulation(6, mulberry32(42));
  const next = nextRemainderBlendGeneration(pop, fitnesses, mulberry32(7));
  for (let k = 0; k < GENOME_SIZE; k++) {
    if (next[0][k] !== pop[1][k]) failures.push(`champion modified at param ${k}`);
  }
  const isClone = (genome: Float64Array): boolean =>
    pop.some((parent) => genome.every((v, k) => v === parent[k]));
  for (const slot of [0, 1]) {
    if (!isClone(next[slot])) failures.push(`elite slot ${slot} is not an unmodified parent clone`);
  }
  return failures;
}

/** (c) Constant engine=1, turn=0 on an open straight track: drives +Y, gains fitness. */
function testStraightCar(): string[] {
  const failures: string[] = [];
  const track: Track = {
    name: 'selftest-straight',
    start: { x: 0, y: 0, angleDeg: 0 },
    checkpoints: [
      [0, 0],
      [0, 20],
      [0, 40],
      [0, 60],
      [0, 80],
      [0, 100],
    ],
    walls: [
      [-50, -10, -50, 110],
      [50, -10, 50, 110],
      [-50, -10, 50, -10],
      [-50, 110, 50, 110],
    ],
  };
  const walls = wallsFlat(track);
  const table = buildCheckpointTable(track);
  const state = initCarState(track);
  for (let i = 0; i < 250; i++) stepCar(state, [0, 1], walls, table, track.checkpoints);

  if (!state.alive) failures.push('straight car died');
  if (!approx(state.x, 0, 1e-4)) failures.push(`straight car drifted: x = ${state.x}`);
  // 2.5 s to reach MAX_VEL=20, then constant: distance = 25 + 20*2.5 = 75.
  if (!approx(state.y, 75, 0.5)) failures.push(`straight car y = ${state.y}, expected ~75`);
  if (!approx(state.vel, 20, 1e-3)) failures.push(`straight car vel = ${state.vel}`);
  if (state.fitness < 0.4) failures.push(`straight car fitness = ${state.fitness}`);
  return failures;
}

/** (d) GPU sim vs CPU reference sim: same genomes/track/steps -> same fitness. */
async function testGpuVsCpu(device: GPUDevice): Promise<string[]> {
  const failures: string[] = [];
  const track: Track = {
    name: 'selftest-mini',
    start: { x: 0, y: 0, angleDeg: 0 },
    checkpoints: [
      [0, 0],
      [0, 30],
      [0, 60],
      [30, 90],
      [60, 90],
      [60, 60],
    ],
    walls: [
      [-8, -5, -8, 70],
      [-8, 70, 20, 100],
      [20, 100, 70, 100],
      [70, 100, 70, 50],
      [8, -5, 8, 60],
      [8, 60, 50, 80],
      [50, 80, 60, 80],
      [-8, -5, 8, -5],
    ],
  };
  const genomes = initPopulation(8, mulberry32(123));
  const steps = 500;
  const cpu = simulatePopulation(genomes, track, steps);
  const gpu = await simulatePopulationGpu(device, track, genomes, steps);
  for (let i = 0; i < genomes.length; i++) {
    if (!approx(gpu[i], cpu[i], 1e-4))
      failures.push(`GPU/CPU fitness mismatch car ${i}: gpu=${gpu[i]} cpu=${cpu[i]}`);
  }
  return failures;
}

/** (e) Model parser: our JSON format and Unity's native ';' genotype format. */
function testModelParse(): string[] {
  const failures: string[] = [];
  const weights = Array.from({ length: GENOME_SIZE }, (_, i) => i / 100 - 0.23);

  const fromJson = parseModelText(JSON.stringify({ topology: [...TOPOLOGY], weights, meta: { track: 't' } }), trackNetwork, parseUnityGenotype);
  const fromUnity = parseModelText(weights.map((w) => w.toFixed(6)).join(';'), trackNetwork, parseUnityGenotype);
  for (const [name, parsed] of [
    ['json', fromJson],
    ['unity', fromUnity],
  ] as const) {
    if (parsed.length !== GENOME_SIZE) failures.push(`${name}: parsed length ${parsed.length}`);
    for (let i = 0; i < GENOME_SIZE; i++) {
      if (!approx(parsed[i], weights[i], 1e-6)) failures.push(`${name}: weight ${i} = ${parsed[i]}, expected ${weights[i]}`);
    }
  }

  for (const bad of [weights.slice(1).join(';'), JSON.stringify({ topology: [5, 4, 3, 2], weights: weights.slice(1) })]) {
    try {
      parseModelText(bad, trackNetwork, parseUnityGenotype);
      failures.push('parser accepted a 46-weight model');
    } catch {
      // expected
    }
  }
  return failures;
}

/** Runs all asserts; needs a GPU device for the GPU==CPU check. */
export async function runSelftest(device: GPUDevice): Promise<SelftestResult> {
  const failures: string[] = [];
  const groups: [string, () => string[] | Promise<string[]>][] = [
    ['forward', testForward],
    ['ga', testGa],
    ['straightCar', testStraightCar],
    ['gpuVsCpu', () => testGpuVsCpu(device)],
    ['modelParse', testModelParse],
  ];
  for (const [name, fn] of groups) {
    try {
      failures.push(...(await fn()).map((f) => `${name}: ${f}`));
    } catch (error) {
      failures.push(`${name}: threw ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  return { pass: failures.length === 0, failures };
}

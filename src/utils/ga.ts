import type { Rng } from './rng';

/** Box–Muller on the seeded rng, for gaussian mutation. */
export function gaussian(rng: Rng): number {
  const u = Math.max(rng(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

export interface RouletteOptions {
  /** Per-weight mutation probability (default 0.1). */
  mutateRate?: number;
  /** Mutation magnitude, gaussian sigma (default 1). */
  sigma?: number;
  /** Top genomes copied to the next generation unmutated (default 0). */
  eliteCount?: number;
}

function roulettePick(probs: Float64Array, rng: Rng): number {
  let r = rng();
  let idx = 0;
  while (r > 0 && idx < probs.length) {
    r -= probs[idx];
    idx++;
  }
  return Math.max(0, idx - 1);
}

function fitnessProbs(fitnesses: ArrayLike<number>): Float64Array {
  let sum = 0;
  for (let i = 0; i < fitnesses.length; i++) sum += fitnesses[i];
  const probs = new Float64Array(fitnesses.length);
  if (sum > 0) for (let i = 0; i < fitnesses.length; i++) probs[i] = fitnesses[i] / sum;
  return probs;
}

/**
 * Roulette-wheel GA (the Flappy Bird source repo's geneticAlgorithm.js), on flat
 * genomes of any size: normalize fitness, pool selection, child = copy + per-weight
 * gaussian mutation. Optional elitism: the top `eliteCount` genomes pass through
 * unmutated (roulette alone forgets the best parent every generation).
 */
export function nextRouletteGeneration(
  population: Float64Array[],
  fitnesses: ArrayLike<number>,
  rng: Rng,
  options: RouletteOptions = {},
): Float64Array[] {
  const { mutateRate = 0.1, sigma = 1, eliteCount = 0 } = options;
  const n = population.length;
  const genomeSize = population[0].length;
  const probs = fitnessProbs(fitnesses);

  const next = Array.from({ length: n }, () => {
    const parent = population[roulettePick(probs, rng)];
    const child = parent.slice();
    for (let k = 0; k < genomeSize; k++) {
      if (rng() < mutateRate) child[k] += gaussian(rng) * sigma;
    }
    return child;
  });

  if (eliteCount > 0) {
    const order = population.map((_, i) => i).sort((a, b) => fitnesses[b] - fitnesses[a]);
    for (let i = 0; i < Math.min(eliteCount, n); i++) next[i] = population[order[i]].slice();
  }
  return next;
}

export interface CrossoverOptions {
  /** Top genomes cloned unmutated (default 1). */
  eliteCount?: number;
  /** Per-weight mutation probability (default 0.05). */
  mutateRate?: number;
  /** Mutation magnitude, gaussian sigma (default 0.2 = their randomGaussian()/5). */
  sigma?: number;
  /** Weights clamped to ±clamp after mutation (default 1). */
  clamp?: number;
}

/**
 * The SnakeAI (CodeBullet) GA: roulette selection of two parents, per-layer
 * single-point crossover (a random row/col cut per weight matrix), then
 * gaussian mutation with weights clamped to ±1. `topology` is the layer sizes
 * (e.g. [24, 16, 16, 4]); the flat genome stores each matrix as
 * input-major columns with the bias column last.
 */
export function nextCrossoverGeneration(
  population: Float64Array[],
  fitnesses: ArrayLike<number>,
  rng: Rng,
  topology: readonly number[],
  options: CrossoverOptions = {},
): Float64Array[] {
  const { eliteCount = 1, mutateRate = 0.05, sigma = 0.2, clamp = 1 } = options;
  const n = population.length;
  const probs = fitnessProbs(fitnesses);

  const next = Array.from({ length: n }, () => {
    const p1 = population[roulettePick(probs, rng)];
    const p2 = population[roulettePick(probs, rng)];
    const child = new Float64Array(p1.length);
    // Per-layer single-point crossover.
    let offset = 0;
    for (let l = 0; l < topology.length - 1; l++) {
      const rows = topology[l + 1];
      const cols = topology[l] + 1;
      const randR = Math.floor(rng() * rows);
      const randC = Math.floor(rng() * cols);
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const k = offset + c * rows + r;
          child[k] = r < randR || (r === randR && c <= randC) ? p1[k] : p2[k];
        }
      }
      offset += rows * cols;
    }
    // Gaussian mutation, clamped.
    for (let k = 0; k < child.length; k++) {
      if (rng() < mutateRate) {
        child[k] += gaussian(rng) * sigma;
        if (child[k] > clamp) child[k] = clamp;
        else if (child[k] < -clamp) child[k] = -clamp;
      }
    }
    return child;
  });

  const order = population.map((_, i) => i).sort((a, b) => fitnesses[b] - fitnesses[a]);
  for (let i = 0; i < Math.min(eliteCount, n); i++) next[i] = population[order[i]].slice();
  return next;
}

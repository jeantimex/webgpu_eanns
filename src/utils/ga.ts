import type { Rng } from './rng';

/** Box–Muller on the seeded rng, for gaussian mutation. */
export function gaussian(rng: Rng): number {
  const u = Math.max(rng(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

/**
 * Roulette-wheel GA (the Flappy Bird source repo's geneticAlgorithm.js), on flat
 * genomes of any size: normalize fitness, pool selection, child = copy + per-weight
 * gaussian mutation with probability `mutateRate`.
 */
export function nextRouletteGeneration(
  population: Float64Array[],
  fitnesses: ArrayLike<number>,
  rng: Rng,
  mutateRate = 0.1,
): Float64Array[] {
  const n = population.length;
  const genomeSize = population[0].length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += fitnesses[i];
  const probs = new Float64Array(n);
  if (sum > 0) for (let i = 0; i < n; i++) probs[i] = fitnesses[i] / sum;

  return Array.from({ length: n }, () => {
    let r = rng();
    let idx = 0;
    while (r > 0 && idx < n) {
      r -= probs[idx];
      idx++;
    }
    const parent = population[Math.max(0, idx - 1)];
    const child = parent.slice();
    for (let k = 0; k < genomeSize; k++) {
      if (rng() < mutateRate) child[k] += gaussian(rng);
    }
    return child;
  });
}

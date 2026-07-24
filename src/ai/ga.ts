import { GENOME_SIZE } from './network';

export type Rng = () => number;

/** Seeded RNG so training runs are reproducible. */
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Uniform in [-1, 1), GeneticAlgorithm.DefInitParamMin/Max. */
export function initPopulation(n: number, rng: Rng): Float64Array[] {
  return Array.from({ length: n }, () => {
    const genome = new Float64Array(GENOME_SIZE);
    for (let i = 0; i < GENOME_SIZE; i++) genome[i] = rng() * 2 - 1;
    return genome;
  });
}

/**
 * One GA step, reproducing the Unity original's *effective* behavior:
 * sort by fitness desc; best 2 pass through unmodified; the rest of the population
 * is filled with clones of the best two, then mutated per-param with prob 0.3 by a
 * uniform amount in ±2.0.
 *
 * Unity's crossover is a no-op by accident (`randomizer.Next() < 0.6` compares a
 * non-negative int against 0.6), so offspring there are always clones of the top 2.
 * That is what makes the original converge so fast — it is effectively a mutation-only
 * (2, N-2) evolution strategy. We previously "fixed" the crossover and training got
 * much slower, so the no-op is reproduced deliberately. Do not "fix" this.
 */
export function nextGeneration(
  population: Float64Array[],
  fitnesses: ArrayLike<number>,
  rng: Rng,
): Float64Array[] {
  const n = population.length;
  const order = population.map((_, i) => i).sort((a, b) => fitnesses[b] - fitnesses[a]);

  const next: Float64Array[] = [population[order[0]].slice(), population[order[1]].slice()];
  while (next.length < n) {
    next.push(population[order[0]].slice());
    if (next.length < n) next.push(population[order[1]].slice());
  }

  // Mutate all but the best two.
  for (let i = 2; i < n; i++) {
    for (let k = 0; k < GENOME_SIZE; k++) {
      if (rng() < 0.3) next[i][k] += rng() * 4 - 2;
    }
  }
  return next;
}

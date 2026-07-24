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
 * One GA step, reproducing the Unity original's *effective* behavior
 * (EvolutionManager.cs:106-119 wiring: DefaultSelectionOperator + RandomRecombination
 * + MutateAllButBestTwo):
 * - slots 0,1: the best two genomes, unmodified (RandomRecombination adds
 *   intermediate[0]/[1] verbatim and mutation skips them),
 * - the rest: clones of parents drawn at random from the top THREE
 *   (RandomRecombination picks two distinct random parents from the intermediate
 *   population = top 3; Unity's crossover is a no-op by accident —
 *   `randomizer.Next() < 0.6` compares a non-negative int against 0.6 — so each
 *   offspring is simply a clone of one parent),
 * - then all clones are mutated per-param with prob 0.3 by a uniform amount in ±2.0.
 *
 * Drawing parents from the top 3 (not just the top 2) is what keeps enough diversity
 * to escape local optima; an earlier version that cloned only the top 2 stalled more.
 */
export function nextGeneration(
  population: Float64Array[],
  fitnesses: ArrayLike<number>,
  rng: Rng,
): Float64Array[] {
  const n = population.length;
  const order = population.map((_, i) => i).sort((a, b) => fitnesses[b] - fitnesses[a]);
  const intermediate = order.slice(0, 3);

  const next: Float64Array[] = [population[order[0]].slice(), population[order[1]].slice()];
  while (next.length < n) {
    const i1 = Math.floor(rng() * intermediate.length);
    let i2 = Math.floor(rng() * intermediate.length);
    while (i2 === i1) i2 = Math.floor(rng() * intermediate.length);
    next.push(population[intermediate[i1]].slice());
    if (next.length < n) next.push(population[intermediate[i2]].slice());
  }

  // Mutate all but the best two.
  for (let i = 2; i < n; i++) {
    for (let k = 0; k < GENOME_SIZE; k++) {
      if (rng() < 0.3) next[i][k] += rng() * 4 - 2;
    }
  }
  return next;
}

import { GENOME_SIZE } from './network';
import { mulberry32, type Rng } from '../../utils/rng';

export { mulberry32, type Rng };

/** Uniform in [-1, 1), GeneticAlgorithm.DefInitParamMin/Max. */
export function initPopulation(n: number, rng: Rng): Float64Array[] {
  return Array.from({ length: n }, () => {
    const genome = new Float64Array(GENOME_SIZE);
    for (let i = 0; i < GENOME_SIZE; i++) genome[i] = rng() * 2 - 1;
    return genome;
  });
}

/**
 * One GA step, matching Unity's default configuration
 * (EvolutionManager.cs:113-119: RemainderStochasticSampling + RandomRecombination
 * + MutateAllButBestTwo):
 *
 * 1. Remainder Stochastic Sampling calculates relative fitness = evaluation / avgEvaluation.
 *    Genotypes with above-average evaluation get copies in the intermediate pool proportional
 *    to their fitness, while lower/medium evaluation genotypes get fractional chances.
 *    This preserves genetic diversity and prevents premature convergence / getting stuck at corners.
 * 2. RandomRecombination preserves the top two elites from the intermediate pool unmodified
 *    and fills the rest of the population with clones drawn at random from the intermediate pool.
 * 3. MutateAllButBestTwo mutates all non-elite clones per-parameter (p=0.3, uniform ±2.0).
 */
/**
 * Recombines two parent genomes using Blend Crossover (BLX) and Uniform Crossover.
 * This allows an offspring to inherit successful features (steering angles, throttle control)
 * from BOTH top-performing parents instead of just cloning a single parent.
 */
function crossover(p1: Float64Array, p2: Float64Array, rng: Rng): [Float64Array, Float64Array] {
  const o1 = new Float64Array(GENOME_SIZE);
  const o2 = new Float64Array(GENOME_SIZE);
  for (let k = 0; k < GENOME_SIZE; k++) {
    if (rng() < 0.6) {
      // Blend crossover: smooth interpolation between parent weights (-0.1 to 1.1 range)
      const alpha = rng() * 1.2 - 0.1;
      o1[k] = alpha * p1[k] + (1 - alpha) * p2[k];
      o2[k] = (1 - alpha) * p1[k] + alpha * p2[k];
    } else {
      // Uniform crossover: swap weights between parents
      if (rng() < 0.5) {
        o1[k] = p1[k];
        o2[k] = p2[k];
      } else {
        o1[k] = p2[k];
        o2[k] = p1[k];
      }
    }
  }
  return [o1, o2];
}

export function nextGeneration(
  population: Float64Array[],
  fitnesses: ArrayLike<number>,
  rng: Rng,
): Float64Array[] {
  const n = population.length;
  // Sort indices descending by evaluation / fitness
  const order = population.map((_, i) => i).sort((a, b) => fitnesses[b] - fitnesses[a]);

  // Calculate average evaluation to get relative fitness (Fitness = evaluation / averageEvaluation)
  let sum = 0;
  for (let i = 0; i < n; i++) sum += fitnesses[i];
  const avg = sum / n;

  const intermediate: number[] = [];

  if (avg > 1e-6) {
    // Remainder Stochastic Sampling (Unity EvolutionManager.cs:245-270)
    // 1. Integer portion: add floor(fitness) copies of each genotype
    for (const idx of order) {
      const relFitness = fitnesses[idx] / avg;
      if (relFitness < 1) break; // Sorted, so subsequent cars also have relFitness < 1
      const count = Math.floor(relFitness);
      for (let c = 0; c < count; c++) {
        intermediate.push(idx);
      }
    }

    // 2. Remainder portion: add 1 copy with probability (fitness - floor(fitness))
    for (const idx of order) {
      const relFitness = fitnesses[idx] / avg;
      const remainder = relFitness - Math.floor(relFitness);
      if (rng() < remainder) {
        intermediate.push(idx);
      }
    }
  }

  // Fallback if intermediate population has fewer than 2 members (e.g. all 0 fitness or flat)
  if (intermediate.length < 2) {
    for (const idx of order) {
      if (!intermediate.includes(idx)) {
        intermediate.push(idx);
        if (intermediate.length >= 2) break;
      }
    }
    while (intermediate.length < 2) {
      intermediate.push(order[0] ?? 0);
    }
  }

  // Dynamic Elite Scaling:
  // When cars start completing the track (maxFitness >= 0.95), expand the unmutated elite pool
  // to 50% of the population (e.g. 15 cars out of 30). This allows a large pack of successful
  // cars to complete the track together on screen, while the remaining 50% undergo light
  // crossover/mutation to optimize lap times and speed lines.
  const maxFitness = Number(fitnesses[order[0]] ?? 0);
  const isTrackCompleted = maxFitness >= 0.95;
  const numElites = isTrackCompleted ? Math.floor(n * 0.5) : Math.min(2, n);

  const next: Float64Array[] = [];

  // Fill unmutated elite slots from top intermediate performers
  for (let i = 0; i < numElites; i++) {
    const parentIdx = intermediate[i % intermediate.length];
    next.push(population[parentIdx].slice());
  }

  // Fill remaining slots with crossover offspring
  while (next.length < n) {
    const i1 = Math.floor(rng() * intermediate.length);
    let i2 = Math.floor(rng() * intermediate.length);
    while (intermediate.length > 1 && i2 === i1) {
      i2 = Math.floor(rng() * intermediate.length);
    }
    const [offspring1, offspring2] = crossover(
      population[intermediate[i1]],
      population[intermediate[i2]],
      rng,
    );
    next.push(offspring1);
    if (next.length < n) next.push(offspring2);
  }

  // Apply mutation ONLY to non-elite slots (from numElites to n - 1)
  for (let i = numElites; i < n; i++) {
    const p = isTrackCompleted ? 0.05 : 0.20;
    const amount = isTrackCompleted ? 0.2 : 1.2;

    for (let k = 0; k < GENOME_SIZE; k++) {
      if (rng() < p) {
        next[i][k] += rng() * (amount * 2) - amount;
      }
    }
  }

  return next;
}


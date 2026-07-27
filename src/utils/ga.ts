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

export interface TournamentOptions {
  /** Individuals drawn per tournament; higher = stronger selection pressure (default 3). */
  tournamentSize?: number;
  /** Fraction of the population carried through untouched (default 0.02). */
  eliteFraction?: number;
  /** Probability a child is crossed from two parents rather than cloned (default 0.8). */
  crossoverRate?: number;
  /** Per-weight probability of being reset to a fresh random value (default 0.05). */
  mutateRate?: number;
  /** Half-width of the range a mutated weight is reset into (default 1). */
  mutateRange?: number;
  /**
   * Share of mutations that are resets; the rest are gaussian drifts (default
   * 0.5). Do not set this to 1 without knowing what you are giving up: with
   * resets only, no weight can ever leave +/-mutateRange, because crossover
   * merely copies values that already exist. The reachable weight space becomes
   * a sealed box, logits stay small, and a softmax policy built on it can never
   * become confident. Measured cost on the pacman demo: 40.3 vs 68.4 pellets.
   */
  resetShare?: number;
  /** Gaussian sigma for drift mutations (default 0.5). */
  driftSigma?: number;
  /** Magnitude ceiling applied after drift (default 8). */
  clamp?: number;
}

/** Best of `k` random draws. Stronger pressure than roulette, and indifferent to fitness scale. */
function tournamentPick(fitnesses: ArrayLike<number>, rng: Rng, k: number): number {
  let best = Math.floor(rng() * fitnesses.length);
  for (let i = 1; i < k; i++) {
    const challenger = Math.floor(rng() * fitnesses.length);
    if (fitnesses[challenger] > fitnesses[best]) best = challenger;
  }
  return best;
}

/**
 * Tournament GA as described in the EANN-Pacman writeup (§3.2–3.4): tournament
 * selection at k=3, single-point crossover on the flat genome at rate ~0.8, and
 * mutation that *resets* a weight to a fresh small random value at rate ~0.05
 * (rather than nudging it, as the roulette variant above does).
 *
 * Tournament beat roulette there because it purges useless random strategies
 * fast; the cost is diversity, which is why `eliteFraction` stays small (2%) and
 * callers are expected to raise `mutateRate` when fitness stagnates.
 */
export function nextTournamentGeneration(
  population: Float64Array[],
  fitnesses: ArrayLike<number>,
  rng: Rng,
  options: TournamentOptions = {},
): Float64Array[] {
  const {
    tournamentSize = 3,
    eliteFraction = 0.02,
    crossoverRate = 0.8,
    mutateRate = 0.05,
    mutateRange = 1,
    resetShare = 0.5,
    driftSigma = 0.5,
    clamp = 8,
  } = options;
  const n = population.length;
  const genomeSize = population[0].length;

  const next = Array.from({ length: n }, () => {
    const parent = population[tournamentPick(fitnesses, rng, tournamentSize)];
    const child = parent.slice();
    if (rng() < crossoverRate) {
      // Single-point: keep this parent up to the cut, take the other's tail.
      const other = population[tournamentPick(fitnesses, rng, tournamentSize)];
      const cut = Math.floor(rng() * genomeSize);
      for (let k = cut; k < genomeSize; k++) child[k] = other[k];
    }
    for (let k = 0; k < genomeSize; k++) {
      if (rng() >= mutateRate) continue;
      if (rng() < resetShare) {
        child[k] = (rng() * 2 - 1) * mutateRange;
      } else {
        child[k] = Math.max(-clamp, Math.min(clamp, child[k] + gaussian(rng) * driftSigma));
      }
    }
    return child;
  });

  const eliteCount = Math.max(1, Math.round(n * eliteFraction));
  const order = population.map((_, i) => i).sort((a, b) => fitnesses[b] - fitnesses[a]);
  for (let i = 0; i < Math.min(eliteCount, n); i++) next[i] = population[order[i]].slice();
  return next;
}

/**
 * The track demo's GA, matching Unity's default configuration
 * (EvolutionManager.cs:113-119: RemainderStochasticSampling + RandomRecombination
 * + MutateAllButBestTwo), with blend/uniform crossover:
 *
 * 1. Remainder Stochastic Sampling calculates relative fitness = evaluation / avgEvaluation.
 *    Genotypes with above-average evaluation get copies in the intermediate pool proportional
 *    to their fitness, while lower/medium evaluation genotypes get fractional chances.
 *    This preserves genetic diversity and prevents premature convergence / getting stuck at corners.
 * 2. RandomRecombination preserves the top elites from the intermediate pool unmodified
 *    and fills the rest of the population with crossover offspring of pool members.
 * 3. Mutation hits all non-elite clones per-parameter (uniform +-amount).
 *
 * Dynamic elite scaling: once a car completes the track (maxFitness >= 0.95), the
 * unmutated elite pool expands to 50% of the population, so a pack of successful
 * cars finishes together while the rest optimize lap times.
 */
export function nextRemainderBlendGeneration(
  population: Float64Array[],
  fitnesses: ArrayLike<number>,
  rng: Rng,
): Float64Array[] {
  const n = population.length;
  const genomeSize = population[0].length;
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

  const maxFitness = Number(fitnesses[order[0]] ?? 0);
  const isTrackCompleted = maxFitness >= 0.95;
  const numElites = isTrackCompleted ? Math.floor(n * 0.5) : Math.min(2, n);

  const next: Float64Array[] = [];

  // Fill unmutated elite slots from top intermediate performers
  for (let i = 0; i < numElites; i++) {
    const parentIdx = intermediate[i % intermediate.length];
    next.push(population[parentIdx].slice());
  }

  // Fill remaining slots with blend/uniform crossover offspring: inherit successful
  // features (steering angles, throttle control) from BOTH top-performing parents.
  while (next.length < n) {
    const i1 = Math.floor(rng() * intermediate.length);
    let i2 = Math.floor(rng() * intermediate.length);
    while (intermediate.length > 1 && i2 === i1) {
      i2 = Math.floor(rng() * intermediate.length);
    }
    const p1 = population[intermediate[i1]];
    const p2 = population[intermediate[i2]];
    const o1 = new Float64Array(genomeSize);
    const o2 = new Float64Array(genomeSize);
    for (let k = 0; k < genomeSize; k++) {
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
    next.push(o1);
    if (next.length < n) next.push(o2);
  }

  // Apply mutation ONLY to non-elite slots (from numElites to n - 1)
  for (let i = numElites; i < n; i++) {
    const p = isTrackCompleted ? 0.05 : 0.2;
    const amount = isTrackCompleted ? 0.2 : 1.2;

    for (let k = 0; k < genomeSize; k++) {
      if (rng() < p) {
        next[i][k] += rng() * (amount * 2) - amount;
      }
    }
  }

  return next;
}

/**
 * Multi-episode fitness evaluation.
 *
 * A genome scored from a single episode is scored partly on luck whenever the
 * environment or the policy contains randomness. Measured on the pacman demo,
 * replaying one identical genome ten times produced 44 to 179 pellets — a 4x
 * spread. Selection over that is selecting noise as much as skill, and the
 * "best" genome an evolution run reports is really the luckiest one.
 *
 * The fix is to run each genome for several episodes and select on an aggregate.
 * On the GPU this is close to free: episodes are extra threads in the same
 * dispatch, not extra dispatches.
 *
 * Agent layout convention, shared by every caller:
 *
 *     agentIndex = genomeIndex * episodes + episodeIndex
 *
 * so one genome's episodes are adjacent, and integer division recovers the
 * genome. A trailing display/replay agent, if the demo has one, sits at
 * `populationSize * episodes` and maps to genome `populationSize` for free.
 */

/** How per-episode scores collapse into the single number selection sees. */
export type EpisodeAggregate = 'mean' | 'min' | 'median';

/**
 * Seed for episode `episodeIndex` of a generation — deliberately independent of
 * which genome is being evaluated.
 *
 * This is the *common random numbers* trick: every genome in a generation faces
 * the same set of episode seeds, so when two genomes are compared the shared
 * luck cancels instead of adding noise to the comparison. It reduces selection
 * error considerably more than averaging independent episodes does, and costs
 * nothing.
 */
export function episodeSeed(generationSeed: number, episodeIndex: number): number {
  let x = (generationSeed ^ Math.imul(episodeIndex + 1, 0x9e3779b9)) >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x7feb352d) >>> 0;
  x = Math.imul(x ^ (x >>> 15), 0x846ca68b) >>> 0;
  return (x ^ (x >>> 16)) >>> 0 || 1;
}

/**
 * Collapse `populationSize * episodes` per-episode scores into one fitness per
 * genome, with `scores[genomeIndex * episodes + episodeIndex]`.
 *
 * `mean` is the usual choice. `min` selects for consistency and is much harsher —
 * useful when a policy that occasionally fails badly is unacceptable. `median`
 * ignores a single catastrophic or lucky run.
 */
export function aggregateEpisodes(
  scores: ArrayLike<number>,
  populationSize: number,
  episodes: number,
  aggregate: EpisodeAggregate = 'mean',
): Float64Array {
  const fitnesses = new Float64Array(populationSize);
  const run = new Float64Array(episodes);
  for (let g = 0; g < populationSize; g++) {
    for (let e = 0; e < episodes; e++) run[e] = scores[g * episodes + e];
    if (aggregate === 'mean') {
      let sum = 0;
      for (let e = 0; e < episodes; e++) sum += run[e];
      fitnesses[g] = sum / episodes;
    } else if (aggregate === 'min') {
      let lo = Infinity;
      for (let e = 0; e < episodes; e++) lo = Math.min(lo, run[e]);
      fitnesses[g] = lo;
    } else {
      const sorted = Array.from(run).sort((a, b) => a - b);
      const mid = episodes >> 1;
      fitnesses[g] = episodes % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    }
  }
  return fitnesses;
}

/**
 * Spread of the per-episode scores, for reporting how noisy evaluation still is.
 * A large mean spread means selection is still partly guessing and `episodes`
 * should go up.
 */
export function episodeSpread(
  scores: ArrayLike<number>,
  populationSize: number,
  episodes: number,
): { meanRange: number; worstRange: number } {
  if (episodes < 2) return { meanRange: 0, worstRange: 0 };
  let total = 0;
  let worst = 0;
  for (let g = 0; g < populationSize; g++) {
    let lo = Infinity;
    let hi = -Infinity;
    for (let e = 0; e < episodes; e++) {
      const v = scores[g * episodes + e];
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    total += hi - lo;
    worst = Math.max(worst, hi - lo);
  }
  return { meanRange: total / populationSize, worstRange: worst };
}

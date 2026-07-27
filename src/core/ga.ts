import {
  nextCrossoverGeneration,
  nextRemainderBlendGeneration,
  nextRouletteGeneration,
  nextTournamentGeneration,
  type CrossoverOptions,
  type RouletteOptions,
  type TournamentOptions,
} from '../utils/ga';
import type { Rng } from '../utils/rng';
import type { Network } from './network';

/** PyGAD-style config-by-kwargs: pick the operator by name, tune by option. */
export type GAConfig =
  | ({ selection: 'roulette' } & RouletteOptions)
  | ({ selection: 'tournament' } & TournamentOptions)
  | ({ selection: 'layered-crossover' } & CrossoverOptions)
  | { selection: 'remainder-blend' };

/** One GA step over the training population. Thin dispatch over utils/ga.ts. */
export function nextGeneration(
  population: Float64Array[],
  fitnesses: ArrayLike<number>,
  rng: Rng,
  net: Network,
  config: GAConfig,
): Float64Array[] {
  switch (config.selection) {
    case 'roulette':
      return nextRouletteGeneration(population, fitnesses, rng, config);
    case 'tournament':
      return nextTournamentGeneration(population, fitnesses, rng, config);
    case 'layered-crossover':
      return nextCrossoverGeneration(population, fitnesses, rng, net.topology, config);
    case 'remainder-blend':
      return nextRemainderBlendGeneration(population, fitnesses, rng);
  }
}

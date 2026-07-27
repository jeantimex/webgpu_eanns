import { mulberry32 } from '../utils/rng';
import type { Network } from './network';

/**
 * TorchGA's create_population: with a `seedGenome`, solution 0 is the seed
 * verbatim and the rest are seed + U(-range, range); without one, every solution
 * is U(-range, range). Range defaults to 1, matching the existing demos.
 */
export function createPopulation(
  net: Network,
  count: number,
  opts: { seed?: number; initRange?: number; seedGenome?: Float64Array } = {},
): Float64Array[] {
  const { seed = 1, initRange = 1, seedGenome } = opts;
  if (seedGenome && seedGenome.length !== net.genomeSize) {
    throw new Error(`Seed genome has ${seedGenome.length} weights, expected ${net.genomeSize}.`);
  }
  const rng = mulberry32(seed);
  const perturb = () => (rng() * 2 - 1) * initRange;
  return Array.from({ length: count }, (_, i) => {
    if (seedGenome && i === 0) return seedGenome.slice();
    const genome = new Float64Array(net.genomeSize);
    for (let k = 0; k < net.genomeSize; k++) genome[k] = (seedGenome ? seedGenome[k] : 0) + perturb();
    return genome;
  });
}
